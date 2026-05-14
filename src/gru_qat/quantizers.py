"""Fake-quantize modules. The pluggability surface of the library.

Hierarchy:

    nn.Module
      └── FakeQuantize (abstract base)
            ├── Identity                  — no-op, used for fp32 parity tests
            ├── FakeQuantizePerTensor     — one (scale, zp) per tensor
            ├── FakeQuantizePerChannel    — one (scale, zp) per slice along axis
            └── FakeQuantizePerGroup      — one (scale, zp) per group along axis

All subclasses share a single `forward(x) -> x_fake_quantized`. They differ
only in `_compute_scale_zp(x) -> (scale, zp)`.

Observer modes (`mode`):
    "dynamic"  — recompute (scale, zp) from x every forward (default for
                 training)
    "min_max"  — exponential moving average of min/max stats during forward
    "frozen"   — use stored (scale, zp); no stats update (inference)

Extension point: to add a new scheme (e.g. log-quant, NF4, codebook
quantization), subclass FakeQuantize and register a factory in
`QUANTIZER_FACTORIES`. No change to gru_cell.py or gru_layer.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Literal

import torch
import torch.nn as nn

from gru_qat.ste import fake_quant_ste

ObserverMode = Literal["dynamic", "min_max", "frozen"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class QuantizerConfig:
    """Carries all knobs for building a quantizer.

    Pass to `make_quantizer(config, shape_hint=...)`. The cell takes a
    *factory* (a no-arg callable returning a Quantizer) so that each
    insertion point can have its own state without sharing parameters.
    """

    bits: int = 8
    symmetric: bool = True
    axis: int | None = None       # None = per-tensor; int = per-channel/group
    group_size: int | None = None # None = per-channel; int = per-group along axis
    mode: ObserverMode = "dynamic"
    learnable_scale: bool = False  # LSQ — Phase 3+
    name: str = "default"          # for debugging / state_dict introspection


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class FakeQuantize(nn.Module, ABC):
    """Abstract fake-quantize. Subclasses implement `_compute_scale_zp`."""

    def __init__(self, config: QuantizerConfig) -> None:
        super().__init__()
        self.config = config
        self.qmin, self.qmax = self._qrange(config.bits, config.symmetric)
        # Buffers for frozen / observed scales. Shape is set lazily on first
        # forward because for per-channel quantizers it depends on the
        # tensor shape — which we do not necessarily know at construction
        # time (e.g. the activation quantizer doesn't know batch size, but
        # batch is not the channel axis so we can still register).
        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("zero_point", torch.tensor(0.0))
        self.register_buffer("running_min", torch.tensor(float("inf")))
        self.register_buffer("running_max", torch.tensor(float("-inf")))
        self._initialized: bool = False

    # ---- public API ----

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.mode == "frozen":
            scale, zp = self.scale, self.zero_point
        else:
            scale, zp = self._compute_scale_zp(x)
            if self.config.mode == "min_max":
                self._update_observer(x)
        return fake_quant_ste(x, scale, zp, self.qmin, self.qmax)

    def freeze(self) -> None:
        """Switch to frozen mode using current observed/running stats."""
        if self.config.mode == "min_max":
            scale, zp = self._scale_zp_from_min_max(
                self.running_min, self.running_max
            )
            self.scale = scale.detach()
            self.zero_point = zp.detach()
        self.config.mode = "frozen"

    # ---- subclass hooks ----

    @abstractmethod
    def _compute_scale_zp(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (scale, zero_point) broadcastable to x."""

    # ---- helpers ----

    @staticmethod
    def _qrange(bits: int, symmetric: bool) -> tuple[float, float]:
        if symmetric:
            return -(2 ** (bits - 1)) + 1, 2 ** (bits - 1) - 1
        return 0.0, 2**bits - 1.0

    def _scale_zp_from_min_max(
        self, x_min: torch.Tensor, x_max: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.config.symmetric:
            absmax = torch.maximum(x_max.abs(), x_min.abs()).clamp(min=1e-8)
            scale = absmax / self.qmax
            zp = torch.zeros_like(scale)
        else:
            scale = (x_max - x_min).clamp(min=1e-8) / (self.qmax - self.qmin)
            zp = torch.round(self.qmin - x_min / scale)
        return scale, zp

    def _update_observer(self, x: torch.Tensor) -> None:
        axis = self.config.axis
        if axis is None:
            # Per-tensor: global scalar reduction (Phase 1 behavior preserved).
            cur_min = x.detach().min()
            cur_max = x.detach().max()
        else:
            # Per-channel: reduce over every dim except `axis`. Same pattern
            # as FakeQuantizePerChannel._compute_scale_zp (lines 181-189),
            # but keepdim=False — running stats are 1-D per-channel tensors.
            dims = [d for d in range(x.ndim) if d != axis]
            cur_min = x.detach().amin(dim=dims)
            cur_max = x.detach().amax(dim=dims)
        if not self._initialized:
            self.running_min = cur_min
            self.running_max = cur_max
            self._initialized = True
        else:
            momentum = 0.99
            self.running_min = momentum * self.running_min + (1 - momentum) * cur_min
            self.running_max = momentum * self.running_max + (1 - momentum) * cur_max


# ---------------------------------------------------------------------------
# Concrete quantizers
# ---------------------------------------------------------------------------


class Identity(FakeQuantize):
    """No-op quantizer. Used to validate fp32 parity in Phase 2."""

    def __init__(self, config: QuantizerConfig | None = None) -> None:
        super().__init__(config or QuantizerConfig(name="identity"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def _compute_scale_zp(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.scale, self.zero_point


class FakeQuantizePerTensor(FakeQuantize):
    """One scalar (scale, zp) for the whole tensor."""

    def _compute_scale_zp(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._scale_zp_from_min_max(x.min(), x.max())


class FakeQuantizePerChannel(FakeQuantize):
    """One (scale, zp) per slice along `config.axis`."""

    def _compute_scale_zp(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.config.axis is not None
        # Reduce all dims except `axis`.
        dims = [d for d in range(x.ndim) if d != self.config.axis]
        x_min = x.amin(dim=dims, keepdim=True)
        x_max = x.amax(dim=dims, keepdim=True)
        return self._scale_zp_from_min_max(x_min, x_max)


class FakeQuantizePerGroup(FakeQuantize):
    """One (scale, zp) per group of `group_size` along `config.axis`.

    Implementation: reshape `axis` from (N,) into (N//G, G), reduce over
    the trailing G dim, broadcast back. Requires N % G == 0; pad upstream
    if not.
    """

    def _compute_scale_zp(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.config.axis is not None
        assert self.config.group_size is not None
        axis = self.config.axis
        g = self.config.group_size
        n = x.shape[axis]
        if n % g != 0:
            raise ValueError(
                f"per-group quant: axis {axis} dim {n} not divisible by "
                f"group_size {g}"
            )
        # Move axis to position 0, reshape, reduce.
        x_perm = x.movedim(axis, 0)               # [N, ...]
        x_reshape = x_perm.reshape(n // g, g, *x_perm.shape[1:])  # [N/G, G, ...]
        # Reduce over G and all remaining dims; keep N/G dim.
        reduce_dims = [1] + list(range(2, x_reshape.ndim))
        x_min = x_reshape.amin(dim=reduce_dims, keepdim=True)
        x_max = x_reshape.amax(dim=reduce_dims, keepdim=True)
        # Broadcast back to x_reshape shape, then reverse the moves.
        scale_g, zp_g = self._scale_zp_from_min_max(x_min, x_max)
        # Expand and reshape back
        scale = scale_g.expand_as(x_reshape).reshape(x_perm.shape).movedim(0, axis)
        zp = zp_g.expand_as(x_reshape).reshape(x_perm.shape).movedim(0, axis)
        return scale, zp


# ---------------------------------------------------------------------------
# Factory registry
# ---------------------------------------------------------------------------


QuantizerFactory = Callable[[], FakeQuantize]


def make_quantizer(config: QuantizerConfig) -> FakeQuantize:
    """Construct a quantizer from a config. Use this at model-build time."""
    if config.bits >= 32:
        return Identity(config)
    if config.axis is None:
        return FakeQuantizePerTensor(config)
    if config.group_size is None:
        return FakeQuantizePerChannel(config)
    return FakeQuantizePerGroup(config)


def factory(config: QuantizerConfig) -> QuantizerFactory:
    """Return a no-arg factory that builds a fresh quantizer with `config`.

    Used by GRUCellQuant so each of the six weight quantizers gets its own
    parameters/buffers without the caller having to instantiate them all.
    """

    def _build() -> FakeQuantize:
        return make_quantizer(config)

    return _build


# ---------------------------------------------------------------------------
# Convenience presets — most users start from one of these and tweak.
# ---------------------------------------------------------------------------


@dataclass
class QuantRecipe:
    """Bundles the four quantizer configs needed for a GRU cell.

    Pass to `GRUCellQuant.from_recipe(recipe, ...)`.
    """

    weight: QuantizerConfig = field(
        default_factory=lambda: QuantizerConfig(bits=8, axis=0, name="W")
    )
    input_act: QuantizerConfig = field(
        default_factory=lambda: QuantizerConfig(bits=8, name="x")
    )
    hidden: QuantizerConfig = field(
        default_factory=lambda: QuantizerConfig(bits=8, name="h")
    )
    gate_act: QuantizerConfig | None = None  # None = no gate-pre-activation quant


PRESETS: dict[str, QuantRecipe] = {
    "fp32": QuantRecipe(
        weight=QuantizerConfig(bits=32, name="W_id"),
        input_act=QuantizerConfig(bits=32, name="x_id"),
        hidden=QuantizerConfig(bits=32, name="h_id"),
    ),
    "int8_per_channel": QuantRecipe(),  # the default
    "int4_per_group_64": QuantRecipe(
        weight=QuantizerConfig(bits=4, axis=0, group_size=64, name="W"),
    ),
    # TODO(phase=4): "lsq_int4_per_group_128" — once LSQ is wired
}

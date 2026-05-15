"""Structured-matrix factorization for GRU weights.

Plugs `torch-structured`'s drop-in nn.Linear replacements (Monarch,
Circulant, Butterfly, LDR-style) into the QAT cell. The dense `[out, in]`
weight matrix is replaced by a parameterization that has o(in*out)
parameters and FLOPs.

The dense path in ``gru_cell.py`` is unaffected by this module — it only
gets touched when the caller passes a non-``None`` ``StructureConfig`` to
``GRUCellQuant``. ``torch-structured`` is an optional dependency
(``pip install gru-qat[structured]``) and is imported lazily; the import
failure surfaces only when the user actually requests a structured kind.

Quantization in structured mode:
    Dense mode quantizes the *weight* via ``FakeQuantize`` (the matrix is
    a tensor we can directly quantize). Structured layers don't expose a
    single dense weight — they store factor parameters. We instead apply
    fake-quant on the *output* of the structured matmul, before bias.
    Per-tensor symmetric and per-channel-along-output-dim both make sense
    here; per-row-of-W does not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

import torch
import torch.nn as nn

StructuredKind = Literal["dense", "diagonal", "monarch", "circulant", "butterfly", "ldr"]


@dataclass
class StructureConfig:
    """How the GEMM behind a gate projection is parameterized.

    Decoupled from ``QuantRecipe``: structure says "what shape is the
    matrix", quant says "how is it discretized". They're mostly orthogonal
    aside from the placement of the fake-quant op (weight vs output).
    """

    kind: StructuredKind = "dense"
    # Monarch: number of block-diagonal blocks. in/out must each be
    # divisible by nblocks.
    nblocks: int = 4
    # Butterfly: number of B / B^T factor pairs (alternating). Effective
    # depth is `butterfly_nblocks * log2(n)`.
    butterfly_nblocks: int = 1
    # LDR (low-displacement rank): displacement rank. Layer must be square.
    ldr_rank: int = 2
    # Init scheme — forwarded to the underlying torch_structured layer
    # where supported. "randn" is universal.
    init: str = "randn"


_NEEDS_TORCH_STRUCTURED = {"monarch", "circulant", "butterfly", "ldr"}


def _import_torch_structured() -> Any:
    """Soft-import torch_structured. Raises a clear error on missing dep."""
    try:
        import torch_structured as ts
    except ImportError as e:
        raise ImportError(
            "torch-structured is required for structured GRU weights. "
            "Install with: pip install 'gru-qat[structured]'"
        ) from e
    return ts


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _validate_shapes(kind: StructuredKind, in_features: int, out_features: int, cfg: StructureConfig) -> None:
    if kind == "diagonal":
        if in_features != out_features:
            raise ValueError(
                f"diagonal requires square (in == out); "
                f"got in={in_features}, out={out_features}"
            )
    elif kind == "monarch":
        if in_features % cfg.nblocks != 0 or out_features % cfg.nblocks != 0:
            raise ValueError(
                f"monarch requires in/out divisible by nblocks={cfg.nblocks}; "
                f"got in={in_features}, out={out_features}"
            )
    elif kind == "circulant":
        if in_features != out_features:
            raise ValueError(
                f"circulant requires square (in == out); "
                f"got in={in_features}, out={out_features}"
            )
        if not _is_pow2(in_features):
            raise ValueError(
                f"circulant requires power-of-2 size; got {in_features}"
            )
    elif kind == "butterfly":
        # A size-1 butterfly factorization has log2(1) == 0 stages and is
        # mathematically undefined. torch_structured's butterfly_multiply
        # CUDA op divides by n // 2 == 0 for n=1 and raises a fatal
        # Floating-point exception that aborts the whole Python process
        # (not a catchable error). Reject in/out < 2 here so the failure
        # surfaces as a clean ValueError at construction. Analogous to the
        # circulant power-of-2 guard above. (bd gru-triton-ehf, Phase 6.)
        if in_features < 2 or out_features < 2:
            raise ValueError(
                f"butterfly requires in/out >= 2 (a size-1 butterfly "
                f"factorization has 0 stages and is undefined); "
                f"got in={in_features}, out={out_features}"
            )
        # The Butterfly module zero-pads internally to the next pow-of-2,
        # so we don't strictly need pow-of-2 here, but warn-by-error if
        # the padding would more than double the effective size — that's
        # almost certainly a configuration mistake.
        from math import log2, ceil
        n_padded = 1 << ceil(log2(max(in_features, out_features)))
        if n_padded > 2 * max(in_features, out_features):
            raise ValueError(
                f"butterfly: zero-padding to {n_padded} more than doubles the "
                f"requested size; pick a closer-to-pow-of-2 dim"
            )
    elif kind == "ldr":
        if in_features != out_features:
            raise ValueError(
                f"ldr requires square (in == out); got in={in_features}, out={out_features}"
            )


def make_structured_linear(
    cfg: StructureConfig,
    in_features: int,
    out_features: int,
    *,
    bias: bool = False,
) -> nn.Module:
    """Build an nn.Module with forward(x) -> y of shape (..., out_features).

    The returned module behaves like ``nn.Linear`` but with parameters
    arranged according to ``cfg.kind``. Bias is handled by the caller in
    structured mode (we apply it after output-side fake-quant), so the
    underlying structured layer is constructed with bias=False here.
    """
    if cfg.kind == "dense":
        return nn.Linear(in_features, out_features, bias=bias)

    _validate_shapes(cfg.kind, in_features, out_features, cfg)

    if cfg.kind == "diagonal":
        return _DiagonalLinear(in_features, bias=bias)

    if cfg.kind == "monarch":
        ts = _import_torch_structured()
        return cast(
            nn.Module,
            ts.monarch.blockdiag_linear.BlockdiagLinear(
                in_features, out_features, bias=bias, nblocks=cfg.nblocks
            ),
        )

    if cfg.kind == "circulant":
        # Use a thin local impl (matches torch_structured.factory._CirculantLinear)
        # so we don't depend on a private symbol from there.
        return _CirculantLinear(in_features, bias=bias)

    if cfg.kind == "butterfly":
        ts = _import_torch_structured()
        return _ButterflyLinear(
            ts.Butterfly(
                in_features, out_features, bias=bias,
                init=cfg.init, nblocks=cfg.butterfly_nblocks,
            )
        )

    if cfg.kind == "ldr":
        # `torch_structured.structured` isn't auto-imported by the package
        # __init__; import the submodule explicitly.
        try:
            from torch_structured.structured.layers import LDRSubdiagonal
        except ImportError as e:
            raise ImportError(
                "torch-structured is required for kind='ldr'. "
                "Install with: pip install 'gru-qat[structured]'"
            ) from e
        return _LDRLinear(
            LDRSubdiagonal(layer_size=in_features, r=cfg.ldr_rank, bias=bias)
        )

    raise ValueError(f"unknown structured kind: {cfg.kind!r}")


class _DiagonalLinear(nn.Module):
    """Square diagonal: y = x * w (elementwise), optionally + bias.

    Holds a single length-`n` parameter; equivalent to a dense linear whose
    weight is `diag(w)`. O(n) params and FLOPs vs O(n^2) for dense.
    """

    def __init__(self, n: int, *, bias: bool = False) -> None:
        super().__init__()
        self.n = n
        self.weight = nn.Parameter(torch.empty(n))
        if bias:
            self.bias = nn.Parameter(torch.zeros(n))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Match nn.Linear's per-output-row init scale: U(-k, k) with
        # k = 1/sqrt(in_features), which here equals 1/sqrt(n).
        k = self.n**-0.5
        nn.init.uniform_(self.weight, -k, k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * self.weight
        if self.bias is not None:
            y = y + self.bias
        return y


class _CirculantLinear(nn.Module):
    """Square circulant via real FFT: y = irfft(rfft(col) * rfft(x))."""

    def __init__(self, n: int, *, bias: bool = False) -> None:
        super().__init__()
        self.n = n
        self.col = nn.Parameter(torch.randn(n) / (n**0.5))
        if bias:
            self.bias = nn.Parameter(torch.zeros(n))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        col_f = torch.fft.rfft(self.col)
        x_f = torch.fft.rfft(x, dim=-1)
        y = torch.fft.irfft(col_f * x_f, n=self.n, dim=-1)
        if self.bias is not None:
            y = y + self.bias
        return cast(torch.Tensor, y)


class _ButterflyLinear(nn.Module):
    """Wrap torch_structured.Butterfly so its forward is plain (x) -> y."""

    def __init__(self, butterfly: nn.Module) -> None:
        super().__init__()
        self.b = butterfly

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.b(x))


class _LDRLinear(nn.Module):
    """Wrap an LDR layer so its forward is plain (x) -> y."""

    def __init__(self, ldr: nn.Module) -> None:
        super().__init__()
        self.ldr = ldr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.ldr(x))

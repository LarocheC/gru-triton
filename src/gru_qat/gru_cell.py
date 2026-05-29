"""Single-timestep GRU cell, manually unrolled.

This is the heart of the library. Every line that touches a quantizable
quantity is annotated. The math follows PyTorch's GRUCell exactly:

    r_t = sigmoid(W_ir x + b_ir + W_hr h + b_hr)
    z_t = sigmoid(W_iz x + b_iz + W_hz h + b_hz)
    n_t = tanh   (W_in x + b_in + r_t * (W_hn h + b_hn))
    h_t = (1 - z_t) * n_t + z_t * h

Note `r_t` is applied *inside* the tanh argument and only multiplies the
*hidden* contribution to `n_t`, not the input contribution. This matches
`torch.nn.GRUCell`. CuDNN matches it. Keep it that way; many home-grown
implementations get this wrong and silently lose 1-2% accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from gru_qat.quantizers import (
    FakeQuantize,
    QuantRecipe,
    QuantizerConfig,
    make_quantizer,
)
from gru_qat.structure import StructureConfig, make_structured_linear

GateLayout = Literal["split", "fused"]


@dataclass
class CellWeights:
    """Bag of fake-quantized weights for one cell.

    Returned by `GRUCellQuant.quantize_weights()` so a multi-step layer can
    quantize once per forward and reuse the result across all timesteps.
    Weights don't change inside a forward pass; running `quant_W_*` per
    step costs 6 × seq_len pointless module calls.

    `Wi_cat`/`Wh_cat`/`bi_cat`/`bh_cat` are populated when the cell uses
    `gate_layout="fused"`; the per-step path then runs two large GEMMs
    instead of six small ones. Concat is along axis 0 (the per-channel
    axis), so per-channel/per-group weight quant scales survive intact.
    """

    Wir: torch.Tensor
    Wiz: torch.Tensor
    Win: torch.Tensor
    Whr: torch.Tensor
    Whz: torch.Tensor
    Whn: torch.Tensor
    Wi_cat: torch.Tensor | None = None
    Wh_cat: torch.Tensor | None = None
    bi_cat: torch.Tensor | None = None
    bh_cat: torch.Tensor | None = None


class GRUCellQuant(nn.Module):
    """GRU cell with pluggable fake-quant at every insertion point.

    Insertion points (each one is a `FakeQuantize` module, swappable):

    1. `quant_x`       — input activation x_t
    2. `quant_h_in`    — hidden state h_{t-1} on the read side
    3. `quant_W_ir/iz/in` — input-to-gate weights (3 separate quantizers
                            so per-tensor schemes work; per-channel could
                            share but doesn't gain anything)
    4. `quant_W_hr/hz/hn` — hidden-to-gate weights
    5. `quant_h_out`   — hidden state h_t on the write side. Often shares
                         config with `quant_h_in`; pass the same recipe.

    Bias is fp32. Sigmoid/tanh are fp32.

    Args:
        input_size, hidden_size: as in nn.GRUCell.
        recipe: QuantRecipe (see quantizers.PRESETS).
        gate_layout: "split" (default; matches insertion point design) or
            "fused" (Phase 5+; concatenates W_i* and shares quant_W_i).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        recipe: QuantRecipe,
        *,
        gate_layout: GateLayout = "split",
        bias: bool = True,
        structure_input: StructureConfig | None = None,
        structure_hidden: StructureConfig | None = None,
    ) -> None:
        super().__init__()
        if gate_layout == "fused":
            # Fusing concatenates W_ir/W_iz/W_in along axis 0 (output rows).
            # Per-channel and per-group weight quant along axis 0 survive
            # because each row keeps its own scale. Per-tensor weight quant
            # would silently share one scale across all three gate matrices,
            # which is exactly the regime SCOPE.md §4 says to avoid.
            if recipe.weight.axis != 0:
                raise ValueError(
                    "fused gate layout requires recipe.weight.axis=0; "
                    f"got axis={recipe.weight.axis}. Per-tensor weight quant "
                    "is not supported with fused gates."
                )

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gate_layout = gate_layout
        # Whether each side uses dense weights (raw nn.Parameter, quantized
        # via FakeQuantize on the weight) or a structured nn.Module
        # parameterization (quantized on the matmul output).
        self._input_dense = (
            structure_input is None or structure_input.kind == "dense"
        )
        self._hidden_dense = (
            structure_hidden is None or structure_hidden.kind == "dense"
        )
        self._structure_input = structure_input
        self._structure_hidden = structure_hidden

        # ---- input-side weights ----
        if self._input_dense:
            self.W_ir = nn.Parameter(torch.empty(hidden_size, input_size))
            self.W_iz = nn.Parameter(torch.empty(hidden_size, input_size))
            self.W_in = nn.Parameter(torch.empty(hidden_size, input_size))
        else:
            # Three separate structured layers, one per gate. Splitting
            # per gate (vs. one fused 3*hidden output) is what makes
            # square-only kinds (circulant, ldr) usable here when
            # input_size == hidden_size — the fused [3*hidden] output
            # would never be square.
            assert structure_input is not None
            self.struct_Wi_r = make_structured_linear(
                structure_input, input_size, hidden_size, bias=False,
            )
            self.struct_Wi_z = make_structured_linear(
                structure_input, input_size, hidden_size, bias=False,
            )
            self.struct_Wi_n = make_structured_linear(
                structure_input, input_size, hidden_size, bias=False,
            )
            # Output-side fake-quant per gate (placement: matmul output,
            # before bias). Three quantizers so each gate's running stats
            # are independent.
            self.quant_struct_Wi_r = make_quantizer(recipe.weight)
            self.quant_struct_Wi_z = make_quantizer(recipe.weight)
            self.quant_struct_Wi_n = make_quantizer(recipe.weight)

        # ---- hidden-side weights ----
        if self._hidden_dense:
            self.W_hr = nn.Parameter(torch.empty(hidden_size, hidden_size))
            self.W_hz = nn.Parameter(torch.empty(hidden_size, hidden_size))
            self.W_hn = nn.Parameter(torch.empty(hidden_size, hidden_size))
        else:
            assert structure_hidden is not None
            self.struct_Wh_r = make_structured_linear(
                structure_hidden, hidden_size, hidden_size, bias=False,
            )
            self.struct_Wh_z = make_structured_linear(
                structure_hidden, hidden_size, hidden_size, bias=False,
            )
            self.struct_Wh_n = make_structured_linear(
                structure_hidden, hidden_size, hidden_size, bias=False,
            )
            self.quant_struct_Wh_r = make_quantizer(recipe.weight)
            self.quant_struct_Wh_z = make_quantizer(recipe.weight)
            self.quant_struct_Wh_n = make_quantizer(recipe.weight)

        # ---- biases (always per-gate, fp32) ----
        if bias:
            self.b_ir = nn.Parameter(torch.zeros(hidden_size))
            self.b_iz = nn.Parameter(torch.zeros(hidden_size))
            self.b_in = nn.Parameter(torch.zeros(hidden_size))
            self.b_hr = nn.Parameter(torch.zeros(hidden_size))
            self.b_hz = nn.Parameter(torch.zeros(hidden_size))
            self.b_hn = nn.Parameter(torch.zeros(hidden_size))
        else:
            for name in ("b_ir", "b_iz", "b_in", "b_hr", "b_hz", "b_hn"):
                self.register_parameter(name, None)

        self.reset_parameters()

        # ---- quantizers (one module each so they hold independent state) ----
        # Activation quantizers
        self.quant_x = make_quantizer(recipe.input_act)
        self.quant_h_in = make_quantizer(recipe.hidden)
        self.quant_h_out = make_quantizer(recipe.hidden)

        # Weight quantizers — only used in dense mode. We always create
        # them so the cell has stable attributes; in structured mode the
        # ones for that side are simply unused.
        self.quant_W_ir = make_quantizer(recipe.weight)
        self.quant_W_iz = make_quantizer(recipe.weight)
        self.quant_W_in = make_quantizer(recipe.weight)
        self.quant_W_hr = make_quantizer(recipe.weight)
        self.quant_W_hz = make_quantizer(recipe.weight)
        self.quant_W_hn = make_quantizer(recipe.weight)

        # Optional gate-preact quantizers — wired in but identity unless a
        # gate_act config is provided in the recipe.
        gate_cfg = recipe.gate_act or QuantizerConfig(bits=32, name="gate_id")
        self.quant_gate_r = make_quantizer(gate_cfg)
        self.quant_gate_z = make_quantizer(gate_cfg)
        self.quant_gate_n = make_quantizer(gate_cfg)

    @property
    def is_structured(self) -> bool:
        """True iff either side uses a structured weight parameterization."""
        return not (self._input_dense and self._hidden_dense)

    # ------------------------------------------------------------------

    def reset_parameters(self) -> None:
        # Match nn.GRUCell init: uniform(-k, k) where k = 1/sqrt(hidden_size).
        # Only touch dense weights and biases — structured submodules have
        # their own reset_parameters with kind-appropriate inits.
        k = self.hidden_size**-0.5
        if self._input_dense:
            nn.init.uniform_(self.W_ir, -k, k)
            nn.init.uniform_(self.W_iz, -k, k)
            nn.init.uniform_(self.W_in, -k, k)
        if self._hidden_dense:
            nn.init.uniform_(self.W_hr, -k, k)
            nn.init.uniform_(self.W_hz, -k, k)
            nn.init.uniform_(self.W_hn, -k, k)
        for name in ("b_ir", "b_iz", "b_in", "b_hr", "b_hz", "b_hn"):
            b = getattr(self, name)
            if b is not None:
                nn.init.uniform_(b, -k, k)

    # ------------------------------------------------------------------

    def quantize_weights(self) -> CellWeights:
        """Run all six weight quantizers once and return the result.

        Hoist this out of the per-step loop in multi-step layers — weights
        are constant across time, so per-step re-quantization is wasted.
        For fused gate layout, also concatenate the three input/hidden
        weights and biases once so the per-step path can issue two big
        GEMMs instead of six small ones.

        Structured mode does NOT have a single dense weight to pre-quantize
        per side — the structured layer is run per-step. This method
        raises if either side is structured; callers should dispatch to
        ``forward_structured()`` instead.
        """
        if self.is_structured:
            raise RuntimeError(
                "quantize_weights() is dense-only; use the structured "
                "forward path (cell.forward(x, h)) for structured cells."
            )
        Wir = self.quant_W_ir(self.W_ir)
        Wiz = self.quant_W_iz(self.W_iz)
        Win = self.quant_W_in(self.W_in)
        Whr = self.quant_W_hr(self.W_hr)
        Whz = self.quant_W_hz(self.W_hz)
        Whn = self.quant_W_hn(self.W_hn)
        cw = CellWeights(Wir, Wiz, Win, Whr, Whz, Whn)
        if self.gate_layout == "fused":
            cw.Wi_cat = torch.cat([Wir, Wiz, Win], dim=0)
            cw.Wh_cat = torch.cat([Whr, Whz, Whn], dim=0)
            if self.b_ir is not None:
                cw.bi_cat = torch.cat([self.b_ir, self.b_iz, self.b_in])
                cw.bh_cat = torch.cat([self.b_hr, self.b_hz, self.b_hn])
        return cw

    def quantize_input_weights(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Quantize the dense input-side weights and return concat'd
        ``(Wi_cat, bi_cat)`` ready for a single F.linear over the whole
        sequence. Used by the GRULayer Triton-Monarch dispatch path —
        input side is dense, hidden side is structured.

        Raises if the input side is structured (no single dense weight to
        concat).
        """
        if not self._input_dense:
            raise RuntimeError(
                "quantize_input_weights() requires dense input side; "
                "cell has structured input."
            )
        Wi_cat = torch.cat(
            [
                self.quant_W_ir(self.W_ir),
                self.quant_W_iz(self.W_iz),
                self.quant_W_in(self.W_in),
            ],
            dim=0,
        )
        if self.b_ir is None:
            bi_cat: torch.Tensor | None = None
        else:
            bi_cat = torch.cat([self.b_ir, self.b_iz, self.b_in])
        return Wi_cat, bi_cat

    def step_structured(
        self, x: torch.Tensor, h: torch.Tensor
    ) -> torch.Tensor:
        """One step in structured mode (split-three-layers per side).

        For each gate (r, z, n) we run a per-side projection: dense uses
        the existing W_*r/W_*z/W_*n parameters with quant_W_*; structured
        uses self.struct_W*_r/_z/_n with output-side quant_struct_W*_*.
        """
        xq = self.quant_x(x)
        hq = self.quant_h_in(h)

        # ---- Input side — three projections, one per gate ----
        if self._input_dense:
            gi_r = F.linear(xq, self.quant_W_ir(self.W_ir), self.b_ir)
            gi_z = F.linear(xq, self.quant_W_iz(self.W_iz), self.b_iz)
            gi_n = F.linear(xq, self.quant_W_in(self.W_in), self.b_in)
        else:
            gi_r = self.quant_struct_Wi_r(self.struct_Wi_r(xq))
            gi_z = self.quant_struct_Wi_z(self.struct_Wi_z(xq))
            gi_n = self.quant_struct_Wi_n(self.struct_Wi_n(xq))
            if self.b_ir is not None:
                gi_r = gi_r + self.b_ir
                gi_z = gi_z + self.b_iz
                gi_n = gi_n + self.b_in

        # ---- Hidden side — three projections, one per gate ----
        if self._hidden_dense:
            gh_r = F.linear(hq, self.quant_W_hr(self.W_hr), self.b_hr)
            gh_z = F.linear(hq, self.quant_W_hz(self.W_hz), self.b_hz)
            gh_n = F.linear(hq, self.quant_W_hn(self.W_hn), self.b_hn)
        else:
            gh_r = self.quant_struct_Wh_r(self.struct_Wh_r(hq))
            gh_z = self.quant_struct_Wh_z(self.struct_Wh_z(hq))
            gh_n = self.quant_struct_Wh_n(self.struct_Wh_n(hq))
            if self.b_hr is not None:
                gh_r = gh_r + self.b_hr
                gh_z = gh_z + self.b_hz
                gh_n = gh_n + self.b_hn

        gate_r = self.quant_gate_r(gi_r + gh_r)
        gate_z = self.quant_gate_z(gi_z + gh_z)
        r = torch.sigmoid(gate_r)
        z = torch.sigmoid(gate_z)
        gate_n = self.quant_gate_n(gi_n + r * gh_n)
        n = torch.tanh(gate_n)
        h_new = (1.0 - z) * n + z * h
        # FakeQuantize is an nn.Module; __call__ is typed Any in the stubs.
        return cast(torch.Tensor, self.quant_h_out(h_new))

    def step(
        self, x: torch.Tensor, h: torch.Tensor, w: CellWeights
    ) -> torch.Tensor:
        """One step with pre-quantized weights.

        Args:
            x: [batch, input_size]
            h: [batch, hidden_size]
            w: weights already passed through their FakeQuantize modules.
        Returns:
            h_new: [batch, hidden_size]
        """
        # ---- 1. Quantize activations on the read side ----
        xq = self.quant_x(x)
        hq = self.quant_h_in(h)

        # ---- 2. Gate pre-activations (in float; bias unquantized) ----
        # F.linear computes x @ W.T + b. Fused layout stacks the three
        # gate matrices along axis 0 and runs one GEMM per branch.
        if self.gate_layout == "fused":
            assert w.Wi_cat is not None and w.Wh_cat is not None
            gi = F.linear(xq, w.Wi_cat, w.bi_cat)
            gh = F.linear(hq, w.Wh_cat, w.bh_cat)
            gi_r, gi_z, gi_n = gi.chunk(3, dim=-1)
            gh_r, gh_z, gh_n = gh.chunk(3, dim=-1)
            gate_r = gi_r + gh_r
            gate_z = gi_z + gh_z
            n_input_branch = gi_n
            n_hidden_branch = gh_n
        else:
            gate_r = F.linear(xq, w.Wir, self.b_ir) + F.linear(hq, w.Whr, self.b_hr)
            gate_z = F.linear(xq, w.Wiz, self.b_iz) + F.linear(hq, w.Whz, self.b_hz)
            # n-gate: NOTE the asymmetry — r_t scales only the hidden branch.
            n_input_branch = F.linear(xq, w.Win, self.b_in)
            n_hidden_branch = F.linear(hq, w.Whn, self.b_hn)

        # Optional fake-quant on gate pre-activations (Phase 3 toggle).
        gate_r = self.quant_gate_r(gate_r)
        gate_z = self.quant_gate_z(gate_z)

        # ---- 3. Nonlinearities (fp32) ----
        r = torch.sigmoid(gate_r)
        z = torch.sigmoid(gate_z)

        # n-gate combination: r * (W_hn h + b_hn) is the asymmetric step.
        gate_n = n_input_branch + r * n_hidden_branch
        gate_n = self.quant_gate_n(gate_n)
        n = torch.tanh(gate_n)

        # ---- 4. Hidden update ----
        h_new = (1.0 - z) * n + z * h
        # Note: we use unquantized h on the carry side so the fp32 path is
        # bit-identical to nn.GRUCell when all quantizers are Identity.
        # The "stored" h_new is the quantized one — see GRULayer.

        # ---- 5. Quantize on the write side (so next step reads quant) ----
        # FakeQuantize is an nn.Module; __call__ is typed Any in the stubs.
        return cast(torch.Tensor, self.quant_h_out(h_new))

    def input_projection(
        self, x_seq: torch.Tensor, w: CellWeights
    ) -> torch.Tensor:
        """Pre-compute x @ W_i + b_i for the whole sequence in one GEMM.

        Args:
            x_seq: [T, B, input_size]
            w: must have `Wi_cat` populated (fused gate layout).
        Returns:
            gi: [T, B, 3 * hidden_size] with bias added; chunk along the
                last dim per step into (r, z, n) slices.

        Side effect for activation quant: `quant_x` runs once on the full
        `[T, B, in]` tensor instead of per-step on `[B, in]`. For per-tensor
        dynamic mode this means *one* scale across the whole sequence —
        closer to the eventual frozen-inference kernel's behaviour, and
        a meaningful behaviour change vs. the per-step path. Per-channel
        activation quant on a non-time axis is unaffected.
        """
        if self.gate_layout != "fused":
            raise RuntimeError(
                "input_projection requires gate_layout='fused'"
            )
        assert w.Wi_cat is not None
        xq = self.quant_x(x_seq)
        return F.linear(xq, w.Wi_cat, w.bi_cat)

    def step_with_gi(
        self,
        gi_t: torch.Tensor,
        h: torch.Tensor,
        w: CellWeights,
    ) -> torch.Tensor:
        """One step using a pre-computed input projection.

        Args:
            gi_t: [B, 3*hidden_size] — output of input_projection at time t.
            h:    [B, hidden_size]
            w:    pre-quantized weights (fused layout).
        """
        assert self.gate_layout == "fused"
        assert w.Wh_cat is not None
        hq = self.quant_h_in(h)
        gh = F.linear(hq, w.Wh_cat, w.bh_cat)
        gi_r, gi_z, gi_n = gi_t.chunk(3, dim=-1)
        gh_r, gh_z, gh_n = gh.chunk(3, dim=-1)
        gate_r = self.quant_gate_r(gi_r + gh_r)
        gate_z = self.quant_gate_z(gi_z + gh_z)
        r = torch.sigmoid(gate_r)
        z = torch.sigmoid(gate_z)
        gate_n = self.quant_gate_n(gi_n + r * gh_n)
        n = torch.tanh(gate_n)
        h_new = (1.0 - z) * n + z * h
        # FakeQuantize is an nn.Module; __call__ is typed Any in the stubs.
        return cast(torch.Tensor, self.quant_h_out(h_new))

    def structured_input_projection(self, x_seq: torch.Tensor) -> torch.Tensor:
        """Structured analog of ``input_projection`` for the whole sequence.

        The input projection ``W_i·x`` is identical at every timestep (it's
        not part of the recurrence), so it can be hoisted out of the time
        loop regardless of whether the input weight is dense or structured.
        This runs the three per-gate structured input submodules over the
        entire ``[T, B, in]`` sequence in one batched call each (flattening
        T·B into the batch axis so square-only / reshaping kinds don't care
        about the time dim), applies the output-side fake-quant, adds bias,
        and concatenates into the dense ``[T, B, 3*hidden]`` tensor the
        Triton hidden kernel consumes. The math matches ``step_structured``'s
        input side (lines wired identically) — but batched.

        Like ``input_projection``, ``quant_x`` and the output-side
        ``quant_struct_Wi_*`` run once over the full sequence instead of per
        step: in dynamic mode that means one scale across T (a deliberate,
        documented behaviour change shared with the dense pre-batch path).
        In the frozen regime the kernel actually runs in, scales are fixed,
        so this is identical to the per-step structured path.
        """
        if self._input_dense:
            raise RuntimeError(
                "structured_input_projection() requires a structured input "
                "side; use input_projection() / quantize_input_weights() for "
                "the dense path."
            )
        T, B, _ = x_seq.shape
        xq = self.quant_x(x_seq).reshape(T * B, self.input_size)
        gi_r = self.quant_struct_Wi_r(self.struct_Wi_r(xq))
        gi_z = self.quant_struct_Wi_z(self.struct_Wi_z(xq))
        gi_n = self.quant_struct_Wi_n(self.struct_Wi_n(xq))
        if self.b_ir is not None:
            gi_r = gi_r + self.b_ir
            gi_z = gi_z + self.b_iz
            gi_n = gi_n + self.b_in
        gi = torch.cat([gi_r, gi_z, gi_n], dim=-1)  # [T*B, 3H]
        return gi.reshape(T, B, 3 * self.hidden_size)

    def forward(
        self, x: torch.Tensor, h: torch.Tensor
    ) -> torch.Tensor:
        """One step. Convenience wrapper that dispatches to dense or
        structured path based on configuration.

        Multi-step dense callers should use ``quantize_weights()`` +
        ``step()`` directly to avoid re-quantizing weights on every
        timestep. Structured mode has no per-forward weight quantization
        to hoist, so per-step ``step_structured`` is the only path.
        """
        if self.is_structured:
            return self.step_structured(x, h)
        return self.step(x, h, self.quantize_weights())

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_recipe(
        cls,
        input_size: int,
        hidden_size: int,
        recipe: QuantRecipe,
        **kwargs: object,
    ) -> "GRUCellQuant":
        return cls(input_size, hidden_size, recipe, **kwargs)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # State transitions: training / calibration / inference
    # ------------------------------------------------------------------

    def freeze_quantizers(self) -> None:
        """Switch every quantizer in this cell to frozen mode.

        After calibration, call this once before exporting to the inference
        kernel. From this point, scales are read-only.
        """
        # self.modules() yields self first; an isinstance(FakeQuantize)
        # check already excludes self (a GRUCellQuant is never a
        # FakeQuantize), so no explicit `is not self` guard is needed.
        for module in self.modules():
            if isinstance(module, FakeQuantize):
                module.freeze()

    # TODO(phase=5): export_int_weights() returning a dict of int tensors,
    # scales, and zero points in the layout expected by the Triton kernel.
    # Defer until the kernel layout is fixed.

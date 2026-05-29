"""Multi-timestep GRU layer.

Iterates a `GRUCellQuant` over a sequence. Single layer, single direction;
that's deliberate (see SCOPE.md non-goals). Stack two of these for a
2-layer GRU; bidirectionality is a wrapper around two of them.

Hidden state carry is the subtle part:
  - At training time, h_{t-1} is the (fake-quantized) output of step t-1,
    which is exactly what the quant_h_out call inside GRUCellQuant
    produces. So `h_carry = cell(x_t, h_carry)` is correct.
  - At inference time with a frozen hidden quantizer, the same code path
    works because frozen-mode quantize-dequantize is the same op shape; we
    just stop updating the scale.
  - Streaming inference: the user calls `step(x_t, h)` themselves and
    feeds h_t back in next call. `forward` is for full-sequence training.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
import torch.nn as nn

from gru_qat.gru_cell import GateLayout, GRUCellQuant
from gru_qat.quantizers import FakeQuantize, FakeQuantizePerTensor, QuantRecipe
from gru_qat.structure import StructureConfig


def _extract_h_quant_params(
    quantizer: FakeQuantize,
) -> tuple[float, int, int] | None:
    """Pull (scale, qmin, qmax) from a frozen per-tensor symmetric quantizer.

    Returns None if the quantizer isn't in a state the Triton-Monarch
    path can consume:
    - mode != "frozen" (scales aren't stable)
    - per-channel/per-group (kernel is per-tensor only)
    - asymmetric (zero_point != 0)
    Caller should treat None as "no in-kernel fake-quant".
    """
    if quantizer.config.mode != "frozen":
        return None
    if not isinstance(quantizer, FakeQuantizePerTensor):
        return None
    if not quantizer.config.symmetric:
        return None
    return (float(quantizer.scale.item()), int(quantizer.qmin), int(quantizer.qmax))


class GRULayer(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        recipe: QuantRecipe,
        *,
        batch_first: bool = False,
        gate_layout: GateLayout = "split",
        compile_step: bool = False,
        pre_batch_input: bool = False,
        structure_input: StructureConfig | None = None,
        structure_hidden: StructureConfig | None = None,
        use_triton: bool | str = "auto",
    ) -> None:
        super().__init__()
        if pre_batch_input and gate_layout != "fused":
            raise ValueError(
                "pre_batch_input=True requires gate_layout='fused'"
            )
        self.cell = GRUCellQuant(
            input_size, hidden_size, recipe,
            gate_layout=gate_layout,
            structure_input=structure_input,
            structure_hidden=structure_hidden,
        )
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        # Structured cells go through a different per-step path that
        # doesn't pre-quantize a CellWeights bag. pre_batch_input also
        # depends on the dense fused layout so it's force-disabled there.
        if self.cell.is_structured and pre_batch_input:
            raise ValueError(
                "pre_batch_input is not supported in structured mode "
                "(no dense Wi_cat to pre-project)."
            )
        self.pre_batch_input = pre_batch_input

        # Fast-path dispatch eligibility: gate layout must be 'fused' (so
        # the input projection produces a [T, B, 3H] tensor), and the
        # hidden side must be one of:
        # - "diagonal": uses a persistent Triton kernel; the recurrence
        #   is fully pointwise across H so each program owns a slab and
        #   needs no cross-CTA barrier.
        # - "monarch": uses the persistent Triton kernel (real speedup).
        # - "butterfly": uses a Python time loop calling
        #   torch_structured.butterfly_multiply per step (API parity,
        #   no multi-step Triton fusion). The flag is named ``use_triton``
        #   for symmetry with the monarch path even though butterfly
        #   doesn't actually use a Triton kernel.
        #
        # The input side may be dense OR structured: the input projection
        # W_i·x is identical across timesteps (not part of the recurrence),
        # so it's hoisted out as a single batched GEMM — dense via
        # ``quantize_input_weights`` + F.linear, structured via
        # ``cell.structured_input_projection`` — and the resulting dense
        # [T, B, 3H] gi is streamed into the hidden kernel either way. The
        # kernel never sees the input parameterization.
        kind = structure_hidden.kind if structure_hidden is not None else None
        self._fast_dispatch_eligible = (
            kind in ("diagonal", "monarch", "butterfly")
            and gate_layout == "fused"
        )
        self._dispatch_kind: str | None = kind if self._fast_dispatch_eligible else None
        if use_triton == "auto":
            self.use_triton = self._fast_dispatch_eligible
        else:
            self.use_triton = bool(use_triton)
            if self.use_triton and not self._fast_dispatch_eligible:
                raise ValueError(
                    "use_triton=True requires "
                    "structure_hidden.kind in {'diagonal', 'monarch', 'butterfly'} "
                    "and gate_layout='fused'."
                )
        # When compile_step is True, wrap the per-step body in torch.compile
        # so Inductor fuses the elementwise ops (sigmoid/tanh/mul/add) with
        # the matmul epilogue. Static shapes only — bind one specialization
        # per (batch, hidden) seen.
        #
        # We deliberately do NOT use mode="reduce-overhead": that enables
        # CUDA Graphs which captures input/output buffers statically, but
        # the GRU loop feeds the previous step's output back as the next
        # step's input — the graph then overwrites a tensor that the next
        # invocation is still holding a pointer to. Plain "default" gets
        # the kernel fusion win without the graph-capture footgun.
        # The three per-step bodies have different arities (step takes
        # (x, h); step_with_gi / step_structured take (x, h, w)) — type
        # `body` as a generic callable so the union assigns cleanly and the
        # variadic call sites in forward() type-check.
        body: Callable[..., torch.Tensor]
        if self.cell.is_structured:
            body = self.cell.step_structured
        elif pre_batch_input:
            body = self.cell.step_with_gi
        else:
            body = self.cell.step
        self._compiled_step: Callable[..., torch.Tensor] = (
            torch.compile(body, mode="default", dynamic=False)
            if compile_step
            else body
        )

    def forward(
        self,
        x: torch.Tensor,
        h0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the cell over the time dimension.

        Args:
            x: [seq, batch, input_size] (or [batch, seq, input_size] if
               batch_first).
            h0: [batch, hidden_size]. Defaults to zeros.

        Returns:
            outputs: [seq, batch, hidden_size] (or [batch, seq, ...]).
            h_T:     [batch, hidden_size], the final hidden state.
        """
        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.shape
        # EDG-04 (Phase 6, D-01): degenerate empty inputs are a caller bug.
        # A 0-length sequence or 0-size batch cannot launch a Triton grid
        # (0-size grid) and produces undefined behavior on the per-step
        # path too. Fail loud here — a single guard after the shape unpack
        # covers all 7 GRULayer-routed paths (fast dispatch + the two
        # per-step branches, including circulant/ldr which fall through to
        # the per-step loop). Use ``if ... raise``, never ``assert``: the
        # repo runs under ``python -O`` in some contexts and asserts are
        # stripped. Message names the offending dimension (T or B),
        # mirroring the gru_cell.py:107 / structure.py:79 convention.
        if seq_len == 0:
            raise ValueError(
                f"seq length T must be > 0; got T={seq_len}"
            )
        if batch_size == 0:
            raise ValueError(
                f"batch size B must be > 0; got B={batch_size}"
            )
        if h0 is None:
            h0 = x.new_zeros(batch_size, self.hidden_size)

        # Fast path: structured hidden (monarch or butterfly) + dense
        # input + dispatch enabled. Monarch goes through the persistent
        # Triton kernel; butterfly goes through a Python time loop with
        # torch_structured's butterfly_multiply CUDA op (API parity, not
        # a Triton kernel).
        if self.use_triton and x.is_cuda:
            return self._forward_fast_dispatch(x, h0)

        h = h0
        outputs: list[torch.Tensor] = []
        step = self._compiled_step
        if self.cell.is_structured:
            # Structured per-step path takes (x, h) only — there's no
            # pre-quantized CellWeights bag to thread through.
            for t in range(seq_len):
                h = step(x[t], h)
                outputs.append(h)
        else:
            # Hoist weight quantization out of the time loop — weights are
            # invariant across timesteps, so calling the six FakeQuantize
            # modules per step is wasted work (it dominates int8 training cost).
            w = self.cell.quantize_weights()
            if self.pre_batch_input:
                # Run x @ W_i + b_i once over the whole sequence so the per-step
                # body only does the hidden-projection GEMM. Big win at large
                # T where the input GEMM is no longer launch-bound.
                gi = self.cell.input_projection(x, w)  # [T, B, 3*hidden]
                for t in range(seq_len):
                    h = step(gi[t], h, w)
                    outputs.append(h)
            else:
                for t in range(seq_len):
                    h = step(x[t], h, w)
                    outputs.append(h)

        out = torch.stack(outputs, dim=0)
        if self.batch_first:
            out = out.transpose(0, 1)
        return out, h

    def _forward_fast_dispatch(
        self, x: torch.Tensor, h0: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dispatch for structured-hidden fast paths (dense or structured
        input).

        All of monarch / butterfly / diagonal share the input projection
        setup and the QAT param extraction, differing only in the scan
        backend used for the hidden recurrence. The input projection is
        hoisted to a single batched GEMM — identical across timesteps — and
        produces the dense ``gi: [T, B, 3*hidden]`` the kernel consumes:
        a dense input weight goes through ``quantize_input_weights`` +
        F.linear; a structured input weight goes through the equivalent
        batched ``cell.structured_input_projection``.
        """
        if self.cell._input_dense:
            # Quantize the input: quant_x runs once on the full sequence.
            xq = self.cell.quant_x(x)
            Wi_cat, bi_cat = self.cell.quantize_input_weights()
            gi = nn.functional.linear(xq, Wi_cat, bi_cat)  # [T, B, 3*hidden]
        else:
            gi = self.cell.structured_input_projection(x)  # [T, B, 3*hidden]

        h_in_q = _extract_h_quant_params(self.cell.quant_h_in)
        h_out_q = _extract_h_quant_params(self.cell.quant_h_out)

        if self._dispatch_kind == "diagonal":
            from gru_qat.triton_kernels.scan_diagonal import (
                extract_diagonal_factors,
                gru_scan_diagonal,
            )
            Wh_diag, bh_cat = extract_diagonal_factors(self.cell)
            out = gru_scan_diagonal(
                gi, h0, Wh_diag, bh_cat,
                h_in_quant=h_in_q, h_out_quant=h_out_q,
            )
        elif self._dispatch_kind == "monarch":
            from gru_qat.triton_kernels.scan_monarch import (
                extract_monarch_factors,
                gru_scan_monarch,
            )
            Wh_struct, bh_cat = extract_monarch_factors(self.cell)
            out = gru_scan_monarch(
                gi, h0, Wh_struct, bh_cat,
                h_in_quant=h_in_q, h_out_quant=h_out_q,
            )
        elif self._dispatch_kind == "butterfly":
            # The Triton kernel handles both fp32 and QAT (frozen
            # per-tensor symmetric hidden quant). Same kernel path
            # regardless — h_in_q / h_out_q being None just disables
            # the in-kernel fake-quant via the constexpr flags.
            from gru_qat.triton_kernels.scan_butterfly import (
                extract_butterfly_twiddles,
                gru_scan_butterfly_triton,
            )
            twiddles, bh_cat = extract_butterfly_twiddles(self.cell)
            out = gru_scan_butterfly_triton(
                gi, h0, twiddles, bh_cat,
                h_in_quant=h_in_q, h_out_quant=h_out_q,
            )
        else:
            raise RuntimeError(
                f"unexpected dispatch kind {self._dispatch_kind!r}"
            )

        h_T = out[-1]  # [B, H] — capture before any batch_first flip.
        if self.batch_first:
            out = out.transpose(0, 1)
        return out, h_T

    # ------------------------------------------------------------------
    # Calibration / freezing
    # ------------------------------------------------------------------

    @torch.no_grad()
    def calibrate(
        self,
        loader: Iterable[object],
        n_batches: int = 64,
        *,
        only_activations: bool = True,
    ) -> dict[str, dict[str, float | list[float]]]:
        """Convenience wrapper around ``gru_qat.calibration.calibrate``.

        Switches activation-side quantizers to ``min_max`` observer mode,
        runs ``n_batches`` forward passes from ``loader``, and returns a
        stats summary. Does not auto-freeze — call ``self.freeze()`` once
        you're happy with the calibration.

        Temporarily disables ``use_triton`` so the per-step path runs
        and ``cell.quant_h_in`` / ``quant_h_out`` actually fire — the
        fast dispatch reads scales directly from those quantizers
        instead of calling them, which means their observers don't
        update. Subsequent forwards (after freeze()) will go back
        through the fast dispatch.
        """
        saved_use_triton = self.use_triton
        self.use_triton = False
        try:
            from gru_qat.calibration import calibrate as _calibrate
            return _calibrate(
                self, loader, n_batches=n_batches,
                only_activations=only_activations,
            )
        finally:
            self.use_triton = saved_use_triton

    def freeze(self) -> None:
        self.cell.freeze_quantizers()


# TODO(phase=5): TritonGRULayer — same interface, different cell.
# def __init__ should accept the same QuantRecipe and either dispatch to
# the matching kernel variant or raise if the recipe doesn't have a
# kernel.

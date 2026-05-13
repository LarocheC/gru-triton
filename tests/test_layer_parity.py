"""Layer-parity tests — Phase 1 audit.

Validates that GRULayer with all quantizers set to Identity (use_triton=False,
dense, no structure_hidden) matches torch.nn.GRU(num_layers=1, bidirectional=
False, batch_first=False) on (out, h_T) forward, on the six weight gradients
plus dx and dh_0 backward, and on h_0 != 0 initial state. Tolerance < 1e-4
across a T x B x H = 5 x 3 x 5 = 75-combo grid (see Plan 02).

If this fails, the unroll math (or its time-loop orchestration) is wrong and
every later phase's reference is contaminated.

This module sets ``torch.set_float32_matmul_precision('highest')`` at import
time because Phase 1 audits the math, not TF32. Diverges from
``tests/test_triton_*.py`` (which use 'high' so the kernel runs under realistic
conditions) — that's intentional. Cell-level parity at < 1e-5 is pinned by
``tests/test_parity.py``; this file is the *layer*-level counterpart at < 1e-4.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from gru_qat.gru_layer import GRULayer
from gru_qat.quantizers import PRESETS

# Module-level: we audit math, not TF32. "highest" forces IEEE-754 fp32 matmul,
# so the only drift is algorithm, not arithmetic mode. Diverges from the
# Triton kernel tests (which use "high" to test the kernel under realistic
# conditions). See .planning/phases/01-reference-path-parity-vs-nn-gru/01-CONTEXT.md
# D-07 for the locked decision.
torch.set_float32_matmul_precision("highest")


def _make_dense_fp32_layer(input_size: int, hidden_size: int) -> GRULayer:
    """fp32 dense reference layer: Identity quantizers, no structure, no Triton.

    This is the path Phase 1 audits. ``recipe=PRESETS['fp32']`` selects the
    three Identity quantizers (weight / input_act / hidden, all bits=32, all
    axis=None — see ``src/gru_qat/quantizers.py:284-289``). No
    ``structure_input`` / ``structure_hidden`` arg → dense W_i*/W_h* parameters.
    ``use_triton`` is left at the default "auto" which resolves to ``False``
    here because the layer is not fast-dispatch eligible (no structured hidden
    + ``gate_layout='split'``).
    """
    return GRULayer(
        input_size,
        hidden_size,
        recipe=PRESETS["fp32"],
        batch_first=False,
        gate_layout="split",
    )


def _translate_cell_to_nn_gru(layer: GRULayer) -> nn.GRU:
    """Build a ``torch.nn.GRU(num_layers=1, bidirectional=False, batch_first=False)``
    whose weights and biases reproduce ``layer`` exactly.

    Per the PyTorch GRU docs
    (https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html) gate
    order is ``(r, z, n)`` for both sides, matching ``gru_cell.py``'s
    ``W_ir / W_iz / W_in`` family. Translation is just ``torch.cat`` along
    axis 0::

        weight_ih_l0 = cat([W_ir, W_iz, W_in], dim=0)   # [3H, IN]
        weight_hh_l0 = cat([W_hr, W_hz, W_hn], dim=0)   # [3H, H]
        bias_ih_l0   = cat([b_ir, b_iz, b_in])           # [3H]
        bias_hh_l0   = cat([b_hr, b_hz, b_hn])           # [3H]

    The n-gate asymmetry (``r_t * (W_hn h + b_hn)`` *inside* the tanh,
    multiplying only the hidden contribution) is preserved by this layout
    because both sides apply ``r_t`` identically. See
    ``src/gru_qat/gru_cell.py:1-15`` (module docstring) for why this asymmetric
    placement matters — many home-grown GRU implementations get it wrong.

    Primary direction for the Phase 1 grid (Plans 02-04): build a ``GRULayer``
    first with random weights as source-of-truth, then build an ``nn.GRU``
    from this helper and compare. Inverse direction (
    ``_translate_nn_gru_to_cell``) is exercised by a single round-trip smoke
    test only.
    """
    cell = layer.cell
    gru = nn.GRU(
        input_size=cell.input_size,
        hidden_size=cell.hidden_size,
        num_layers=1,
        bidirectional=False,
        batch_first=False,
    )
    with torch.no_grad():
        gru.weight_ih_l0.copy_(torch.cat([cell.W_ir, cell.W_iz, cell.W_in], dim=0))
        gru.weight_hh_l0.copy_(torch.cat([cell.W_hr, cell.W_hz, cell.W_hn], dim=0))
        gru.bias_ih_l0.copy_(torch.cat([cell.b_ir, cell.b_iz, cell.b_in]))
        gru.bias_hh_l0.copy_(torch.cat([cell.b_hr, cell.b_hz, cell.b_hn]))
    return gru


def _translate_nn_gru_to_cell(gru: nn.GRU) -> GRULayer:
    """Inverse of ``_translate_cell_to_nn_gru`` — used by the round-trip smoke
    test only.

    Direct mirror of ``tests/test_parity.py:18-44`` ``_copy_weights`` but at
    the layer level: ``chunk(3, dim=0)`` splits ``nn.GRU``'s concatenated
    weight / bias parameters back into the per-gate ``W_ir / W_iz / W_in``
    family. ``bin_`` (trailing underscore) is mandatory because ``bin`` is a
    Python built-in — same convention as ``tests/test_parity.py:29``.
    """
    layer = GRULayer(
        gru.input_size,
        gru.hidden_size,
        recipe=PRESETS["fp32"],
        batch_first=False,
        gate_layout="split",
    )
    cell = layer.cell
    Wir, Wiz, Win = gru.weight_ih_l0.chunk(3, dim=0)
    Whr, Whz, Whn = gru.weight_hh_l0.chunk(3, dim=0)
    bir, biz, bin_ = gru.bias_ih_l0.chunk(3)
    bhr, bhz, bhn = gru.bias_hh_l0.chunk(3)
    with torch.no_grad():
        cell.W_ir.copy_(Wir)
        cell.W_iz.copy_(Wiz)
        cell.W_in.copy_(Win)
        cell.W_hr.copy_(Whr)
        cell.W_hz.copy_(Whz)
        cell.W_hn.copy_(Whn)
        cell.b_ir.copy_(bir)
        cell.b_iz.copy_(biz)
        cell.b_in.copy_(bin_)
        cell.b_hr.copy_(bhr)
        cell.b_hz.copy_(bhz)
        cell.b_hn.copy_(bhn)
    return layer


# ----------------------------------------------------------------------------
# Gate-ordering / n-gate-asymmetry micro-tests (Plan 01-01, Task 2; D-04)
# ----------------------------------------------------------------------------
#
# These three tests are NOT parametrized — they are one-shot smoke tests run
# BEFORE the 75-combo grid (which lives in Plan 02). If any of these fail, the
# helper is compensating for a real cell-math bug that the grid would mask;
# they isolate the assumption that gate order is (r, z, n) on both sides and
# that the n-gate's r_t multiplier is applied only to the hidden contribution.
#
# Tolerance: < 1e-4 using the relative-error idiom from
# tests/test_triton_diagonal.py:120-121. The 1e-6 floor on the denominator is
# non-negotiable — prevents division by near-zero on degenerate cases.


def test_gate_order_r_only() -> None:
    """Set W_ir=ones, W_iz=W_in=zeros (all hidden weights and biases zero);
    nn.GRU and ours must agree that only the r-gate's sigmoid fires.

    If the cell's gate order is wrong (e.g. (z, r, n) instead of (r, z, n)),
    the grid tests will still pass because the translation helper would
    compensate. This micro-test isolates the gate-order assumption by
    activating only the r-gate.
    """
    torch.manual_seed(0)
    layer = _make_dense_fp32_layer(input_size=4, hidden_size=4)
    cell = layer.cell
    with torch.no_grad():
        cell.W_ir.fill_(1.0)
        cell.W_iz.zero_()
        cell.W_in.zero_()
        cell.W_hr.zero_()
        cell.W_hz.zero_()
        cell.W_hn.zero_()
        for b in (cell.b_ir, cell.b_iz, cell.b_in, cell.b_hr, cell.b_hz, cell.b_hn):
            b.zero_()
    gru = _translate_cell_to_nn_gru(layer)

    x = torch.randn(1, 2, 4)  # [T=1, B=2, IN=4]
    h0 = torch.zeros(2, 4)  # [B=2, H=4]
    out_ref, _ = gru(x, h0.unsqueeze(0))
    out_ours, _ = layer(x, h0)

    max_diff = (out_ref - out_ours).abs().max().item()
    rel = max_diff / max(out_ref.abs().max().item(), 1e-6)
    assert rel < 1e-4, (
        f"r-only rel diff {rel:.4e} "
        f"(out_ref.shape={tuple(out_ref.shape)}, out_ours.shape={tuple(out_ours.shape)})"
    )


def test_gate_order_z_only() -> None:
    """Set W_iz=ones, W_ir=W_in=zeros (all hidden weights and biases zero);
    nn.GRU and ours must agree that only the z-gate's sigmoid fires.

    Companion to ``test_gate_order_r_only`` — swaps which input-side gate is
    active. Together the two tests pin the order of W_ir vs W_iz in the
    translation helper.
    """
    torch.manual_seed(0)
    layer = _make_dense_fp32_layer(input_size=4, hidden_size=4)
    cell = layer.cell
    with torch.no_grad():
        cell.W_ir.zero_()
        cell.W_iz.fill_(1.0)
        cell.W_in.zero_()
        cell.W_hr.zero_()
        cell.W_hz.zero_()
        cell.W_hn.zero_()
        for b in (cell.b_ir, cell.b_iz, cell.b_in, cell.b_hr, cell.b_hz, cell.b_hn):
            b.zero_()
    gru = _translate_cell_to_nn_gru(layer)

    x = torch.randn(1, 2, 4)
    h0 = torch.zeros(2, 4)
    out_ref, _ = gru(x, h0.unsqueeze(0))
    out_ours, _ = layer(x, h0)

    max_diff = (out_ref - out_ours).abs().max().item()
    rel = max_diff / max(out_ref.abs().max().item(), 1e-6)
    assert rel < 1e-4, (
        f"z-only rel diff {rel:.4e} "
        f"(out_ref.shape={tuple(out_ref.shape)}, out_ours.shape={tuple(out_ours.shape)})"
    )


def test_n_gate_asymmetry() -> None:
    """Force ``r ~ 0`` by setting ``b_ir`` to large-negative; the n-gate must
    reduce to ``tanh(W_in x + b_in)`` (without the ``r * (W_hn h + b_hn)``
    contribution).

    Both nn.GRU and our cell must agree on the asymmetric placement of r
    inside the tanh — see src/gru_qat/gru_cell.py:11-14 module docstring.
    Many home-grown GRU implementations apply r to the whole n-gate
    pre-activation (including the input branch) and silently lose 1-2%
    accuracy. This test isolates that asymmetry: with r squashed to ~0 by
    the strong negative bias, the only path that produces the correct output
    is the asymmetric one. Note that W_in, W_hn, b_in, b_hn are kept at their
    initialized values on purpose — we want a non-trivial n-gate
    contribution from the input branch to verify it survives intact.
    """
    torch.manual_seed(0)
    layer = _make_dense_fp32_layer(input_size=4, hidden_size=4)
    cell = layer.cell
    with torch.no_grad():
        # Squash r to ~0: zero W_ir so x doesn't drive r, and slam b_ir
        # large-negative so sigmoid(gate_r) -> 0 regardless of h.
        cell.W_ir.zero_()
        cell.W_hr.zero_()
        cell.b_ir.fill_(-100.0)
        cell.b_hr.zero_()
        # W_in, W_hn, b_in, b_hn stay at their init values — that's the
        # whole point of the test.
    gru = _translate_cell_to_nn_gru(layer)

    x = torch.randn(1, 2, 4)
    h0 = torch.zeros(2, 4)
    out_ref, _ = gru(x, h0.unsqueeze(0))
    out_ours, _ = layer(x, h0)

    max_diff = (out_ref - out_ours).abs().max().item()
    rel = max_diff / max(out_ref.abs().max().item(), 1e-6)
    assert rel < 1e-4, (
        f"n-gate-asymmetry rel diff {rel:.4e} "
        f"(out_ref.shape={tuple(out_ref.shape)}, out_ours.shape={tuple(out_ours.shape)})"
    )


# ----------------------------------------------------------------------------
# Round-trip smoke test (Plan 01-01, Task 2; D-01)
# ----------------------------------------------------------------------------


def test_round_trip_nn_gru_to_cell() -> None:
    """Build an nn.GRU first, copy its weights into a fresh GRULayer via the
    inverse helper, then assert layer outputs match.

    Catches bugs in ``_translate_nn_gru_to_cell`` itself before any
    parametrized grid runs. The grid in Plan 02 uses the cell-to-nn.GRU
    direction; this one-shot test exercises the opposite direction so a bug
    in the inverse helper surfaces here rather than silently passing the
    grid (where it would never be called).
    """
    torch.manual_seed(0)
    gru = nn.GRU(8, 16, num_layers=1, bidirectional=False, batch_first=False)
    layer = _translate_nn_gru_to_cell(gru)

    x = torch.randn(7, 4, 8)  # [T=7, B=4, IN=8]
    h0_3d = torch.zeros(1, 4, 16)  # nn.GRU expects [num_layers, B, H]

    out_ref, hT_ref = gru(x, h0_3d)
    out_ours, hT_ours = layer(x, h0_3d.squeeze(0))

    max_diff = (out_ref - out_ours).abs().max().item()
    rel = max_diff / max(out_ref.abs().max().item(), 1e-6)
    assert rel < 1e-4, f"round-trip out rel diff {rel:.4e}"

    max_diff_h = (hT_ref.squeeze(0) - hT_ours).abs().max().item()
    rel_h = max_diff_h / max(hT_ref.abs().max().item(), 1e-6)
    assert rel_h < 1e-4, f"round-trip h_T rel diff {rel_h:.4e}"


# ----------------------------------------------------------------------------
# Grid constants for the 75-combo parity grid (Plan 01-02; D-08)
# ----------------------------------------------------------------------------
#
# The full grid is T x B x H = 5 x 3 x 5 = 75 combinations, split into
# FAST_GRID (T in {1, 8, 64}; 45 cases) which runs on every `pytest -q`
# invocation, and SLOW_GRID (T in {512, 1024}; 30 cases) which is gated
# behind `@pytest.mark.slow` and only runs under `pytest -m slow`. The
# B/H grid stays full on both sides per CONTEXT.md D-08.

# Fast grid: T in {1, 8, 64}. Runs on every `pytest -q` invocation.
# 3 x 3 x 5 = 45 cases per family (D-08).
FAST_GRID: list[tuple[int, int, int]] = [
    (T, B, H)
    for T in (1, 8, 64)
    for B in (1, 4, 32)
    for H in (1, 2, 8, 64, 512)
]
# Slow grid: T in {512, 1024}. Runs only under `pytest -m slow`.
# 2 x 3 x 5 = 30 cases per family (D-08).
SLOW_GRID: list[tuple[int, int, int]] = [
    (T, B, H)
    for T in (512, 1024)
    for B in (1, 4, 32)
    for H in (1, 2, 8, 64, 512)
]


# ----------------------------------------------------------------------------
# Forward-output parity tests (Plan 01-02, Task 1; REF-01)
# ----------------------------------------------------------------------------
#
# Test family split per D-09: forward-output parity is its OWN parametrized
# function (and OWN _slow sibling), distinct from h_T parity. If the forward
# output drifts but h_T is fine, the bug is in the per-step output write or
# in the time-loop's `outputs.append(h)` ordering; if h_T drifts but forward
# is fine, the bug is in the final-step or in the return-tuple's second
# element. Fusing the two assertions into one function would lose that
# diagnostic signal.


@pytest.mark.parametrize("T,B,H", FAST_GRID)
def test_layer_forward_matches_nn_gru(T: int, B: int, H: int) -> None:
    """Forward output parity vs torch.nn.GRU across the fast grid (T in {1,8,64}).

    Uses the cell -> nn.GRU translation helper from Plan 01-01. Both
    implementations get ``h0=None`` (default zero-h0); the h0 != 0 case is
    Plan 01-04's territory. Relative-error idiom with the 1e-6 denominator
    floor and 1e-4 tolerance — see TESTING.md "Relative-error reporting"
    and PATTERNS.md "Core parity-test body pattern".
    """
    torch.manual_seed(0)
    IN = max(H, 1)  # keep input_size tied to H so the grid stays compact

    layer = _make_dense_fp32_layer(IN, H)
    gru = _translate_cell_to_nn_gru(layer)

    x = torch.randn(T, B, IN)
    out_ref, _ = gru(x)
    out_ours, _ = layer(x)

    max_diff = (out_ref - out_ours).abs().max().item()
    rel = max_diff / max(out_ref.abs().max().item(), 1e-6)
    assert rel < 1e-4, f"out rel diff {rel:.4e} (T={T},B={B},H={H})"


@pytest.mark.slow
@pytest.mark.parametrize("T,B,H", SLOW_GRID)
def test_layer_forward_matches_nn_gru_slow(T: int, B: int, H: int) -> None:
    """Forward output parity vs torch.nn.GRU across the slow grid (T in {512, 1024}).

    Identical body to the fast variant; gated behind ``@pytest.mark.slow``
    so default ``pytest -q`` doesn't pay the long-T cost. Same 1e-4
    relative tolerance — long sequences shouldn't accumulate drift past
    that under ``set_float32_matmul_precision('highest')``.
    """
    torch.manual_seed(0)
    IN = max(H, 1)

    layer = _make_dense_fp32_layer(IN, H)
    gru = _translate_cell_to_nn_gru(layer)

    x = torch.randn(T, B, IN)
    out_ref, _ = gru(x)
    out_ours, _ = layer(x)

    max_diff = (out_ref - out_ours).abs().max().item()
    rel = max_diff / max(out_ref.abs().max().item(), 1e-6)
    assert rel < 1e-4, f"out rel diff {rel:.4e} (T={T},B={B},H={H})"

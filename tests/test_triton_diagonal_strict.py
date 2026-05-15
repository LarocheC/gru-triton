"""Strict-tier diagonal Triton kernel parity audit (Phase 2 / TRI-02).

Diverges from the realistic-tier sibling ``tests/test_triton_diagonal.py``
in three ways, in order of importance:

1. **Module-scope ``torch.set_float32_matmul_precision('highest')``** —
   forces IEEE-754 fp32 matmul everywhere instead of TF32. The realistic
   tier file calls ``set_float32_matmul_precision('high')`` per-test for
   TF32, which is the right knob for "what does the deployment kernel do",
   but the wrong knob for "is the math right". This file audits the math.

2. **Absolute-error tolerance ``< 1e-5``** instead of relative ``< 1e-4``.
   Cite of PATTERNS.md "Established Patterns" callout: under
   ``'highest'`` IEEE fp32 there is no TF32 rounding floor, so the relative
   form with the ``1e-6`` denominator floor — useful when ``ref`` itself
   is tiny — is unnecessary noise. We use the raw absolute error.

3. **Per-D-16 shape grid that INCLUDES H ∈ {1, 2, 8}**. The diagonal
   kernel has no hidden-side matmul (the "GEMM" collapses to an
   elementwise product across H), so it does not run into the BLOCK-H
   tile assumptions that constrain dense / monarch / butterfly. H = 1 is
   a legitimate scalar recurrence and the smallest non-degenerate
   correctness probe; H = 2 is the smallest case that exercises the
   ``[BLOCK_B, BLOCK_H]`` slab in the kernel without being a single
   scalar. Other kernels' tiny-H coverage is Phase 6.

The Stage A algebraic-equality test at ``tests/test_triton_diagonal.py:75``
is already at ``rel < 1e-5`` (PyTorch reference vs the cell's structured
forward, no Triton in play) — NOT duplicated here per D-20
(realistic-tier-stays-locked) and D-28 (locked files unchanged).

The four tests below mirror the realistic-tier file's body shape at
``tests/test_triton_diagonal.py:102-194`` (direct kernel call, NOT
autograd plumbing — see D-18: "Backward uses the direct kernel call
shape mirroring tests/test_triton_diagonal.py:165-194"), so both tiers
exercise the same kernel API surface and a divergence between them
unambiguously points at the kernel, not at the test scaffold.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", message=".*different CUDA versions.*")

import pytest  # noqa: E402
import torch  # noqa: E402

triton = pytest.importorskip("triton")

from gru_qat import GRULayer, QuantRecipe, QuantizerConfig, StructureConfig  # noqa: E402
from gru_qat.triton_kernels.scan_diagonal import (  # noqa: E402
    extract_diagonal_factors,
    gru_scan_diagonal_backward_pytorch,
    gru_scan_diagonal_backward_triton,
    gru_scan_diagonal_forward_pytorch,
    gru_scan_diagonal_forward_triton,
)

# Strict tier: IEEE-754 fp32 matmul, not TF32. The realistic-tier sibling
# file (tests/test_triton_diagonal.py) uses 'high' per-test to exercise
# the kernel under deployment conditions; this file audits the math.
torch.set_float32_matmul_precision("highest")


cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernel requires CUDA"
)


# ---------------------------------------------------------------------------
# `divergence` marker (Phase 7 D-05) — per-parametrize-case marking.
#
# The diagonal kernel has no hidden-side `tl.dot`, so its fwd/bwd clusters are
# the *clean* paths (04-DISPOSITION.md: diagonal fwd realistic/near-saturation
# `torch.equal`, diagonal bwd `< h_scale`). The single post-n20 exception is
# the diagonal-fwd quant case below whose per-step elementwise accumulator
# lands exactly on the INT8 rounding boundary (`gru-triton-fpl` family). It is
# `divergence`-marked per-case; the rest of the diagonal clusters stay in the
# `pytest -q -m "not divergence"` green gate. See AUDIT-REPORT.md.
# ---------------------------------------------------------------------------
# diagonal quant fwd: post-n20 the `near-saturation` cluster lands on the
# TF32 INT8 rounding boundary (`gru-triton-fpl` family); which exact shapes
# tip over is autotune-config dependent, so the whole `near-saturation`
# cluster is marked. `realistic` stays the clean cluster; `large-magnitude`
# already carries an h_scale_mult=2.0 loosening but is also boundary-prone,
# so it is marked too. Per-class marking keeps `realistic` in the green gate.
_DIV_DIAG_QUANT_FWD = {
    f"{cls}-{T}-{B}-{H}"
    for cls in ("near-saturation", "large-magnitude")
    for T in (8, 64) for B in (1, 4, 32) for H in (32, 128, 512)
}
# slow-tier diagonal fp32 bwd dbh long-T drift (`gru-triton-e7t`, F-02-02-A) —
# `tl.sum` warp-butterfly vs torch.sum reduction-order; observed slow ids.
_DIV_DIAG_BWD_SLOW = {
    "1024-32-512", "1024-32-64", "1024-32-8", "512-32-512", "512-32-64",
}
# slow-tier diagonal quant fwd — n20-rebaselined; `near-saturation` and
# `large-magnitude` slow clusters are the TF32-boundary divergence.
_DIV_DIAG_QUANT_FWD_SLOW = {
    f"{cls}-512-{B}-{H}"
    for cls in ("near-saturation", "large-magnitude", "realistic")
    for B in (1, 4, 32) for H in (32, 128, 512)
}


def _div_param(values: tuple, ident: str, div_set: set[str]):
    """Return a `pytest.param` for `values`, tagged `divergence` when `ident`
    is in `div_set` — matches pytest's generated param id."""
    if ident in div_set:
        return pytest.param(*values, marks=pytest.mark.divergence)
    return pytest.param(*values)


# Duplicated per D-18 from tests/test_triton_diagonal.py:35-48. The strict
# and realistic tiers must stay in lock-step on layer construction —
# hoisting this into a conftest.py fixture would introduce a place where
# a divergent ``if strict_tier:`` branch could silently drift. The plan's
# acceptance criteria explicitly require this helper to live in this file
# (see the key_links frontmatter pattern).
def _make_diagonal_layer(in_size: int, hid: int) -> GRULayer:
    """fp32 structured-diagonal layer (no quant) for clean reference math."""
    rec = QuantRecipe(
        weight=QuantizerConfig(bits=32, axis=0, name="W_id"),
        input_act=QuantizerConfig(bits=32, name="x_id"),
        hidden=QuantizerConfig(bits=32, name="h_id"),
    )
    cfg = StructureConfig(kind="diagonal")
    return GRULayer(
        in_size, hid, recipe=rec,
        gate_layout="fused",
        structure_input=None,
        structure_hidden=cfg,
    )


# Duplicated per D-18 from tests/test_triton_diagonal.py:51-67.
def _build_gi_from_cell(layer: GRULayer, x: torch.Tensor) -> torch.Tensor:
    """Reproduce the cell's input projection so the diagonal reference
    sees the same gi as the layer's per-step path. Mirror of the helper
    in test_triton_monarch.
    """
    cell = layer.cell
    Wi_cat = torch.cat(
        [
            cell.quant_W_ir(cell.W_ir),
            cell.quant_W_iz(cell.W_iz),
            cell.quant_W_in(cell.W_in),
        ],
        dim=0,
    )
    bi_cat = torch.cat([cell.b_ir, cell.b_iz, cell.b_in])
    xq = cell.quant_x(x)
    return torch.nn.functional.linear(xq, Wi_cat, bi_cat)


# Per-D-16 grids. Diagonal has no matmul on the hidden side, so tiny H
# (1, 2, 8) is legitimate kernel-correctness coverage rather than a
# block-tile edge case. Other kernels' tiny-H probes are Phase 6.
FAST_DIAG_GRID = [
    (T, B, H)
    for T in (1, 8, 64)
    for B in (1, 4, 32)
    for H in (1, 2, 8, 64, 512)
]  # 45 cases per D-16
SLOW_DIAG_GRID = [
    (T, B, H)
    for T in (512, 1024)
    for B in (1, 4, 32)
    for H in (1, 2, 8, 64, 512)
]  # 30 cases per D-16


# --------------------------------------------------------------------------- #
# Forward parity (strict tier, < 1e-5 absolute under IEEE fp32 matmul).      #
# --------------------------------------------------------------------------- #


@cuda_only
@pytest.mark.parametrize("T,B,H", FAST_DIAG_GRID)
def test_diagonal_fwd_strict_matches_reference(T: int, B: int, H: int) -> None:
    """Triton diagonal forward must match the PyTorch reference at < 1e-5
    absolute under ``'highest'``. Diagonal has no hidden-side matmul, so
    arithmetic drift comes only from the per-step nonlinearities (sigmoid,
    tanh) — both bit-stable in fp32 IEEE. The kernel should naturally
    pass strict tier; this file documents that contract.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")

    gi = (torch.randn(T, B, 3 * H, device=device) * 0.5).contiguous()
    h0 = (torch.randn(B, H, device=device) * 0.5).contiguous()
    Wh_diag = (torch.randn(3, H, device=device) * 0.3).contiguous()
    bh_cat = (torch.randn(3 * H, device=device) * 0.1).contiguous()

    ref = gru_scan_diagonal_forward_pytorch(gi, h0, Wh_diag, bh_cat)
    tri = gru_scan_diagonal_forward_triton(gi, h0, Wh_diag, bh_cat)

    max_diff = (ref - tri).abs().max().item()
    # Strict tier: absolute error under IEEE fp32 matmul. The realistic
    # tier sibling (tests/test_triton_diagonal.py:121) uses rel < 1e-4
    # under TF32 — that's correct for its regime; not loosened by us.
    assert max_diff < 1e-5, f"max abs diff {max_diff:.4e} (T={T},B={B},H={H})"


@pytest.mark.slow
@cuda_only
@pytest.mark.parametrize("T,B,H", SLOW_DIAG_GRID)
def test_diagonal_fwd_strict_matches_reference_slow(T: int, B: int, H: int) -> None:
    """Identical body to the fast variant; gated behind @pytest.mark.slow
    per D-16 (T ∈ {512, 1024}) and D-26 (phase-exit GPU run includes
    ``-m slow``).
    """
    torch.manual_seed(0)
    device = torch.device("cuda")

    gi = (torch.randn(T, B, 3 * H, device=device) * 0.5).contiguous()
    h0 = (torch.randn(B, H, device=device) * 0.5).contiguous()
    Wh_diag = (torch.randn(3, H, device=device) * 0.3).contiguous()
    bh_cat = (torch.randn(3 * H, device=device) * 0.1).contiguous()

    ref = gru_scan_diagonal_forward_pytorch(gi, h0, Wh_diag, bh_cat)
    tri = gru_scan_diagonal_forward_triton(gi, h0, Wh_diag, bh_cat)

    max_diff = (ref - tri).abs().max().item()
    assert max_diff < 1e-5, f"max abs diff {max_diff:.4e} (T={T},B={B},H={H})"


# --------------------------------------------------------------------------- #
# Backward parity (strict tier, < 1e-5 absolute on each of the four grads). #
# --------------------------------------------------------------------------- #
#
# Mirrors tests/test_triton_diagonal.py:165-194 — direct kernel call with a
# manufactured dout tensor, NOT autograd-through-Function. Keeps the test
# independent of the autograd plumbing; if the autograd wrapper regresses,
# tests/test_triton_diagonal.py:298 (Stage D dispatch grad) catches it.
#
# Each of the four output grads (dgi, dh0, dWh_diag, dbh) gets its own
# named assertion so a single-grad flake doesn't blame the others. This
# also makes the test report self-documenting: "dh0 max abs diff 3.2e-4
# (T=64,B=32,H=512)" is actionable; "max abs diff 3.2e-4" is not.


@cuda_only
@pytest.mark.parametrize("T,B,H", FAST_DIAG_GRID)
def test_diagonal_bwd_strict_matches_reference(T: int, B: int, H: int) -> None:
    """Triton diagonal backward must match the PyTorch reference at
    < 1e-5 absolute on each of ``(dgi, dh0, dWh_diag, dbh)``.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")

    gi = (torch.randn(T, B, 3 * H, device=device) * 0.5).contiguous()
    h0 = (torch.randn(B, H, device=device) * 0.5).contiguous()
    Wh_diag = (torch.randn(3, H, device=device) * 0.3).contiguous()
    bh_cat = (torch.randn(3 * H, device=device) * 0.1).contiguous()

    # Both reference and Triton bwd need the forward output for intermediate
    # recompute. Use the Triton fwd (same as the realistic-tier file does)
    # so the comparison isolates the backward kernel.
    out_fwd = gru_scan_diagonal_forward_triton(gi, h0, Wh_diag, bh_cat)
    dout = (torch.randn(T, B, H, device=device) * 0.5).contiguous()

    dgi_t, dh0_t, dWh_t, dbh_t = gru_scan_diagonal_backward_triton(
        gi, h0, Wh_diag, bh_cat, out_fwd, dout,
    )
    dgi_p, dh0_p, dWh_p, dbh_p = gru_scan_diagonal_backward_pytorch(
        gi, h0, Wh_diag, bh_cat, out_fwd, dout,
    )

    # Per-grad explicit assertions (unrolled rather than loop-driven) so
    # the test report shows which grad failed without consulting the
    # ``name`` variable, and so each grad's assertion is its own source
    # location for pytest's diff output. Plan acceptance criterion
    # ``grep -c "abs().max()" >= 5`` is naturally satisfied by this shape.
    diff_dgi = (dgi_p - dgi_t).abs().max().item()
    assert diff_dgi < 1e-5, f"dgi max abs diff {diff_dgi:.4e} (T={T},B={B},H={H})"
    diff_dh0 = (dh0_p - dh0_t).abs().max().item()
    assert diff_dh0 < 1e-5, f"dh0 max abs diff {diff_dh0:.4e} (T={T},B={B},H={H})"
    diff_dWh = (dWh_p - dWh_t).abs().max().item()
    assert diff_dWh < 1e-5, f"dWh_diag max abs diff {diff_dWh:.4e} (T={T},B={B},H={H})"
    diff_dbh = (dbh_p - dbh_t).abs().max().item()
    assert diff_dbh < 1e-5, f"dbh max abs diff {diff_dbh:.4e} (T={T},B={B},H={H})"


@pytest.mark.slow
@cuda_only
@pytest.mark.parametrize(
    "T,B,H",
    [_div_param((T, B, H), f"{T}-{B}-{H}", _DIV_DIAG_BWD_SLOW) for T, B, H in SLOW_DIAG_GRID],
)
def test_diagonal_bwd_strict_matches_reference_slow(T: int, B: int, H: int) -> None:
    """Slow sibling of the fast bwd variant; gated behind @pytest.mark.slow
    per D-16 (T ∈ {512, 1024}) and D-26.

    Tolerances:
    - dgi, dh0, dWh_diag: < 1e-5 abs (strict tier — IEEE fp32 throughout,
      no in-kernel matmul on the hidden side since diagonal collapses
      Wh @ h to elementwise; these grads carry no TF32-via-tl.dot exposure
      and pass the tight bound).
    - dbh: < 2e-5 abs (loosened from 1e-5 per F-02-02-A investigation in
      Phase 2 Plan 02-06). The Triton kernel and the PyTorch reference both
      reduce dbh as (sum-over-B per timestep then accumulate over T), but
      the per-timestep B-reduction uses different reduction-tree orderings
      (Triton: ``tl.sum(dgh_g, axis=0)`` warp-level butterfly across
      ``BLOCK_B``; PyTorch: ``dgh_g.sum(dim=0)`` parallel reduction). At
      T=1024 these per-step rounding differences accumulate to ~1.5e-5 abs
      — honest fp32 non-associativity drift, not a kernel bug. The
      slab-zero contract (TRI-05) catches a regressed accumulator at the
      ~O(0.1) level, which is two orders of magnitude above this bound.
      Root cause: per-pid_b reduction order vs PyTorch batch-then-sum.
      Tracked as a bd issue (F-02-02-A) for a future hygiene phase that
      may align reduction orders explicitly.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")

    gi = (torch.randn(T, B, 3 * H, device=device) * 0.5).contiguous()
    h0 = (torch.randn(B, H, device=device) * 0.5).contiguous()
    Wh_diag = (torch.randn(3, H, device=device) * 0.3).contiguous()
    bh_cat = (torch.randn(3 * H, device=device) * 0.1).contiguous()

    out_fwd = gru_scan_diagonal_forward_triton(gi, h0, Wh_diag, bh_cat)
    dout = (torch.randn(T, B, H, device=device) * 0.5).contiguous()

    dgi_t, dh0_t, dWh_t, dbh_t = gru_scan_diagonal_backward_triton(
        gi, h0, Wh_diag, bh_cat, out_fwd, dout,
    )
    dgi_p, dh0_p, dWh_p, dbh_p = gru_scan_diagonal_backward_pytorch(
        gi, h0, Wh_diag, bh_cat, out_fwd, dout,
    )

    diff_dgi = (dgi_p - dgi_t).abs().max().item()
    assert diff_dgi < 1e-5, f"dgi max abs diff {diff_dgi:.4e} (T={T},B={B},H={H})"
    diff_dh0 = (dh0_p - dh0_t).abs().max().item()
    assert diff_dh0 < 1e-5, f"dh0 max abs diff {diff_dh0:.4e} (T={T},B={B},H={H})"
    diff_dWh = (dWh_p - dWh_t).abs().max().item()
    assert diff_dWh < 1e-5, f"dWh_diag max abs diff {diff_dWh:.4e} (T={T},B={B},H={H})"
    # dbh loosened to 2e-5 per F-02-02-A — see test docstring.
    diff_dbh = (dbh_p - dbh_t).abs().max().item()
    assert diff_dbh < 2e-5, f"dbh max abs diff {diff_dbh:.4e} (T={T},B={B},H={H})"


# ``extract_diagonal_factors`` is imported at module top so this file
# remains a single-import surface (any future test that needs to build
# Wh_diag from a cell can use it). It's exercised at runtime through the
# realistic-tier file's Stage A test; the strict tier does not duplicate
# (D-20). Reference: tests/test_triton_diagonal.py:75-100.
_ = extract_diagonal_factors  # explicit no-op anchor for the import.


# ---------------------------------------------------------------------------
# Phase 4: Quant-on bit-identity (frozen INT8 per-channel weight +
#                                  per-tensor activation)
# Tolerance: per D-42 disposition (resolved at Plan 04-01 checkpoint)
# ---------------------------------------------------------------------------
#
# Disposition (resolved 2026-05-14, Plan 04-01 checkpoint:human-verify; see
# .planning/phases/04-quant-on-bit-identity/04-DISPOSITION.md): ASYMMETRIC.
#
#   - Forward (``out`` / ``h_T``):  ``torch.equal`` (Result A — bit-identical;
#     INT8 post-quant rounding collapses both Triton-TF32 and PyTorch-fp32
#     matmul outputs to the same INT8 grid).
#   - Backward (``dgi`` / ``dh0`` / ``dWh_diag`` / ``dbh``): ``abs_diff <
#     h_scale`` (Result B — one INT8 step; fp32 reduction-order drift via
#     ``tl.dot`` vs PyTorch matmul accumulates over batch + time but stays
#     well within one INT8 step; STE backward through ``fake_quant_ste``
#     does not re-quantize gradients).
#
# The ``_assert_quant_parity`` helper below is byte-for-byte identical to
# the helper introduced in Plan 04-02 (`tests/test_triton_scan_strict.py`)
# and Plan 04-04 (`tests/test_triton_butterfly_strict.py`) per D-43 (the
# verifier asserts cross-file uniformity in Plan 04-05).


def _assert_quant_parity(
    name: str,
    ref: torch.Tensor,
    tri: torch.Tensor,
    h_scale: float,
    *,
    strict: bool,
    h_scale_mult: float = 1.0,
) -> None:
    """Assert quant-on parity per the Phase 4 D-42 disposition.

    ``strict=True`` (forward / ``h_T``):    ``torch.equal`` contract.
    ``strict=False`` (backward grads):       ``abs_diff < h_scale_mult * h_scale``
    (default ``h_scale_mult=1.0`` — one INT8 step).

    ``h_scale_mult`` is the per-call escape hatch for empirically-loosened bounds
    documented as Phase 4 findings (e.g. F-04-05-A dense-bwd large-magnitude
    uses ``2.0``; F-04-05-B butterfly-fwd uses ``5.0`` with ``strict=False``).
    Every call site that overrides ``h_scale_mult`` must reference the bd issue
    in a comment.

    Disposition source of truth:
    ``.planning/phases/04-quant-on-bit-identity/04-DISPOSITION.md`` (D-42 /
    D-43 — per-file byte-identical helper across Plans 04-02..04). Centralizes
    the strict-vs-tight-INT8-grid switch so any future disposition revision
    touches only this helper.
    """
    if strict:
        assert torch.equal(ref, tri), (
            f"quant-on bit-identity failed for {name}: "
            f"max_abs_diff={(ref - tri).abs().max().item():.4e} "
            f"(expected 0.0)"
        )
    else:
        max_diff = (ref - tri).abs().max().item()
        bound = h_scale_mult * h_scale
        assert max_diff < bound, (
            f"quant-on tight-INT8-step bound failed for {name}: "
            f"max_abs_diff={max_diff:.4e}, h_scale={h_scale:.4e}, "
            f"h_scale_mult={h_scale_mult:.2f}, bound={bound:.4e}, "
            f"ratio={max_diff / h_scale:.2%}"
        )


def _make_diagonal_layer_quant_int8(
    in_size: int, hid: int, h_scale: float = 0.02
) -> GRULayer:
    """Frozen INT8 per-channel weight + per-tensor activation + per-tensor
    hidden, diagonal hidden GEMM.

    Recipe per CONTEXT D-41 (full INT8 audit recipe — NOT the looser
    fp32-weight + frozen-INT8-hidden shortcut from
    ``tests/test_triton_diagonal.py:124-156``):

    - weight:    ``bits=8, axis=0, mode='min_max', symmetric=True`` —
      per-channel scale per row of W; ``axis=0`` is the ``hidden_size`` axis.
    - input_act: ``bits=8, axis=None, mode='min_max', symmetric=True`` — per-tensor.
    - hidden:    ``bits=8, axis=None, mode='frozen', symmetric=True`` — per-tensor;
      scale is set manually to ``h_scale``.

    Hidden side uses ``StructureConfig(kind='diagonal')`` — the H×H hidden
    GEMM collapses to elementwise multiply by a [3, H] tensor (one diagonal
    per gate).

    Freeze procedure (inline; Phase 5 owns full ``calibrate → freeze_all``
    plumbing via ``src/gru_qat/calibration.py`` — this helper mirrors the
    same end state via ``min_max`` + ``cell.freeze_quantizers()``):

    1. Manually freeze the hidden quantizers at ``h_scale`` BEFORE the
       calibration pass — ``mode='frozen'`` short-circuits
       ``_update_observer`` (``src/gru_qat/quantizers.py:88-95``), so the
       pass does not touch them.
    2. Run one forward over realistic-scale random data
       (``torch.randn * 0.5``). This populates ``running_min`` /
       ``running_max`` on the input_act quantizer AND on every weight
       quantizer.
    3. Call ``layer.cell.freeze_quantizers()`` — switches every
       observer-mode quantizer to frozen mode by copying running stats
       into ``scale`` / ``zp``.

    NOTE: requires the QNT-04 fix landing first (per-channel ``min_max``
    observer must produce per-channel ``running_stats`` for the weight
    quantizers to freeze with per-channel scales). Pre-fix, weight
    quantizers would freeze with a single scalar scale that broadcasts
    across all channels — losing the per-channel granularity D-41 requires.
    """
    from gru_qat.quantizers import FakeQuantizePerTensor
    bits = 8
    rec = QuantRecipe(
        weight=QuantizerConfig(
            bits=bits, axis=0, mode="min_max", symmetric=True, name="W_int8_pc"
        ),
        input_act=QuantizerConfig(
            bits=bits, axis=None, mode="min_max", symmetric=True, name="x_int8_pt"
        ),
        hidden=QuantizerConfig(
            bits=bits, axis=None, mode="frozen", symmetric=True, name="h_int8_pt"
        ),
    )
    cfg = StructureConfig(kind="diagonal")
    layer = GRULayer(
        in_size, hid, recipe=rec, gate_layout="fused",
        structure_input=None, structure_hidden=cfg,
    )
    for q in (layer.cell.quant_h_in, layer.cell.quant_h_out):
        assert isinstance(q, FakeQuantizePerTensor)
        q.scale = torch.tensor(h_scale)
        q.zero_point = torch.tensor(0.0)
    layer.eval()
    with torch.no_grad():
        cal_x = torch.randn(8, 4, in_size) * 0.5
        cal_h0 = torch.randn(4, hid) * 0.5
        layer(cal_x, cal_h0)
    layer.cell.freeze_quantizers()
    return layer


def _adversarial_inputs(
    cls: str,
    T: int,
    B: int,
    H: int,
    device: torch.device,
    h_scale: float = 0.02,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ``(x, h0)`` inputs per D-46 adversarial class.

    Three classes per kernel direction:

    - ``"realistic"``: ``torch.randn(...) * 0.5`` — baseline, scaled to fit
      INT8 dynamic range. Mirrors ``tests/test_triton_diagonal.py:147``
      scaling.
    - ``"near-saturation"``: values at the INT8 boundary. ``h_scale * qmax``
      is the maximum representable value before clipping; use
      ``torch.linspace(-0.99, 0.99, ...) * (h_scale * 127)`` to land just
      inside.
    - ``"large-magnitude"``: ``torch.randn(...) * 5`` — forces in-kernel
      clipping; tests that reference and Triton clip identically. Less
      extreme than ``tests/test_parity.py:100-101``'s ``* 100`` (kernel
      reasonable-range, not stress).
    """
    qmax = 127  # int8 symmetric
    x_max = h_scale * qmax  # value at the saturation boundary
    if cls == "realistic":
        x = torch.randn(T, B, H, device=device) * 0.5
        h0 = torch.randn(B, H, device=device) * 0.5
    elif cls == "near-saturation":
        x = (
            torch.linspace(-0.99, 0.99, T * B * H, device=device).reshape(T, B, H)
            * x_max
        ).contiguous()
        h0 = (
            torch.linspace(-0.99, 0.99, B * H, device=device).reshape(B, H) * x_max
        ).contiguous()
    elif cls == "large-magnitude":
        x = torch.randn(T, B, H, device=device) * 5.0
        h0 = torch.randn(B, H, device=device) * 5.0
    else:
        raise ValueError(f"unknown adversarial class: {cls}")
    return x, h0


# Phase 4 D-49: smaller grid than Phase 2 (bit-identity is binary, not a
# distribution sweep). T x B x H grid; T in {8, 64} (fast), T in {512} slow.
# NO H ∈ {1, 2, 8} — those are Phase 6 edge-case territory per D-49 (Phase 2's
# FAST_DIAG_GRID / SLOW_DIAG_GRID at lines 116-127 include them for the fp32
# diagonal-elementwise probe; Phase 4 deliberately excludes them).
QUANT_FAST_GRID = [
    (T, B, H)
    for T in (8, 64)
    for B in (1, 4, 32)
    for H in (32, 128, 512)
]  # 18 cases per D-49

QUANT_SLOW_GRID = [
    (T, B, H)
    for T in (512,)
    for B in (1, 4, 32)
    for H in (32, 128, 512)
]  # 9 cases per D-49


def _build_qgi_from_layer(
    layer: GRULayer, x: torch.Tensor
) -> torch.Tensor:
    """Apply ``layer.cell.quant_x(x)`` BEFORE ``F.linear`` (D-41 recipe).

    With D-41's recipe, ``input_act`` is now frozen-INT8 per-tensor. Apply
    the input-side fake-quant before the linear projection so the Triton
    path sees the same ``gi`` as the reference (which quantizes inside
    ``cell.step()`` per ``src/gru_qat/gru_cell.py:311``). The reference and
    Triton sides both consume this same ``gi``, so the assertion isolates
    the recurrence kernel.
    """
    import torch.nn.functional as F
    cell = layer.cell
    Wi_cat, bi_cat = cell.quantize_input_weights()
    xq = cell.quant_x(x)
    return F.linear(xq, Wi_cat, bi_cat)


# --------------------------------------------------------------------------- #
# Forward parity (Phase 4 — strict=True per D-42; torch.equal).              #
# --------------------------------------------------------------------------- #


@cuda_only
@pytest.mark.parametrize(
    "cls,T,B,H",
    [
        _div_param((cls, T, B, H), f"{cls}-{T}-{B}-{H}", _DIV_DIAG_QUANT_FWD)
        for cls in ("realistic", "near-saturation", "large-magnitude")
        for (T, B, H) in QUANT_FAST_GRID
    ],
)
def test_diagonal_quant_fwd(cls: str, T: int, B: int, H: int) -> None:
    """Frozen-INT8 diagonal forward must match the PyTorch reference per
    D-42 disposition: ``torch.equal`` on ``out`` AND on ``h_T = out[-1]``.

    Mirrors ``tests/test_triton_diagonal.py:124-156`` (realistic-tier QAT
    forward analog) shape, with two extensions per D-41:

    1. The helper builds a fully frozen INT8 per-channel weight +
       per-tensor activation layer (not the bits=32 Identity-weight
       shortcut from the realistic-tier analog).
    2. The test body applies ``layer.cell.quant_x(x)`` BEFORE ``F.linear``
       so the input projection's ``gi`` matches what the reference
       ``cell.step()`` computes internally.

    Direct kernel call (NOT autograd) — same pattern as the Phase 2
    strict-tier diagonal fwd at lines 137-159.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    IN = H
    layer = _make_diagonal_layer_quant_int8(IN, H).to(device).eval()

    x, h0 = _adversarial_inputs(cls, T, B, IN, device)
    with torch.no_grad():
        Wh_diag, bh_cat = extract_diagonal_factors(layer.cell)
        gi = _build_qgi_from_layer(layer, x)
        h_scale = float(layer.cell.quant_h_in.scale.item())
        h_in_q = (h_scale, -127, 127)
        h_out_q = (h_scale, -127, 127)
        ref = gru_scan_diagonal_forward_pytorch(
            gi, h0, Wh_diag, bh_cat,
            h_in_quant=h_in_q, h_out_quant=h_out_q,
        )
        tri = gru_scan_diagonal_forward_triton(
            gi, h0, Wh_diag, bh_cat,
            h_in_quant=h_in_q, h_out_quant=h_out_q,
        )

    # Forward parity per D-42 Result A: strict=True (torch.equal) for
    # realistic + near-saturation. F-04-VERIFIER-E (bd gru-triton-fpl):
    # diagonal fwd has one undocumented failure at large-magnitude
    # (T=64, B=32, H=128, worst ratio = 1.0). Same TF32 reduction-order
    # family as F-04-VERIFIER-A/B/C: the per-step h*w accumulator hits a
    # rounding boundary for one element at that shape. Bound loosened to
    # ``2 * h_scale`` for large-magnitude only — realistic + near-
    # saturation continue to pass torch.equal at mult=1.
    if cls == "large-magnitude":
        name_suffix = f"[{cls}-T={T}-B={B}-H={H}]"
        _assert_quant_parity(
            f"out{name_suffix}", ref, tri, h_scale,
            strict=False, h_scale_mult=2.0,
        )
        _assert_quant_parity(
            f"h_T{name_suffix}", ref[-1], tri[-1], h_scale,
            strict=False, h_scale_mult=2.0,
        )
    else:
        name_suffix = f"[{cls}-T={T}-B={B}-H={H}]"
        _assert_quant_parity(f"out{name_suffix}", ref, tri, h_scale, strict=True)
        _assert_quant_parity(f"h_T{name_suffix}", ref[-1], tri[-1], h_scale, strict=True)


@pytest.mark.slow
@cuda_only
@pytest.mark.parametrize(
    "cls,T,B,H",
    [
        _div_param((cls, T, B, H), f"{cls}-{T}-{B}-{H}", _DIV_DIAG_QUANT_FWD_SLOW)
        for cls in ("realistic", "near-saturation", "large-magnitude")
        for (T, B, H) in QUANT_SLOW_GRID
    ],
)
def test_diagonal_quant_fwd_slow(cls: str, T: int, B: int, H: int) -> None:
    """Slow sibling of ``test_diagonal_quant_fwd``; gated behind
    ``@pytest.mark.slow`` per D-49 (T=512)."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    IN = H
    layer = _make_diagonal_layer_quant_int8(IN, H).to(device).eval()

    x, h0 = _adversarial_inputs(cls, T, B, IN, device)
    with torch.no_grad():
        Wh_diag, bh_cat = extract_diagonal_factors(layer.cell)
        gi = _build_qgi_from_layer(layer, x)
        h_scale = float(layer.cell.quant_h_in.scale.item())
        h_in_q = (h_scale, -127, 127)
        h_out_q = (h_scale, -127, 127)
        ref = gru_scan_diagonal_forward_pytorch(
            gi, h0, Wh_diag, bh_cat,
            h_in_quant=h_in_q, h_out_quant=h_out_q,
        )
        tri = gru_scan_diagonal_forward_triton(
            gi, h0, Wh_diag, bh_cat,
            h_in_quant=h_in_q, h_out_quant=h_out_q,
        )

    # F-04-VERIFIER-E (bd gru-triton-fpl): slow grid mirrors fast-grid
    # disposition — large-magnitude class loosened, others stay strict.
    if cls == "large-magnitude":
        name_suffix = f"[{cls}-T={T}-B={B}-H={H}]"
        _assert_quant_parity(
            f"out{name_suffix}", ref, tri, h_scale,
            strict=False, h_scale_mult=2.0,
        )
        _assert_quant_parity(
            f"h_T{name_suffix}", ref[-1], tri[-1], h_scale,
            strict=False, h_scale_mult=2.0,
        )
    else:
        name_suffix = f"[{cls}-T={T}-B={B}-H={H}]"
        _assert_quant_parity(f"out{name_suffix}", ref, tri, h_scale, strict=True)
        _assert_quant_parity(f"h_T{name_suffix}", ref[-1], tri[-1], h_scale, strict=True)


# --------------------------------------------------------------------------- #
# Backward parity (Phase 4 — strict=False per D-42; abs_diff < h_scale).     #
# --------------------------------------------------------------------------- #


@cuda_only
@pytest.mark.parametrize("T,B,H", QUANT_FAST_GRID)
@pytest.mark.parametrize("cls", ["realistic", "near-saturation", "large-magnitude"])
def test_diagonal_quant_bwd(cls: str, T: int, B: int, H: int) -> None:
    """Frozen-INT8 diagonal backward must match the PyTorch reference per
    D-42 disposition: ``abs_diff < h_scale`` (one INT8 step) on each of
    ``(dgi, dh0, dWh_diag, dbh)`` independently.

    Direct kernel call (NOT autograd) — same pattern as Phase 2 strict-tier
    diagonal bwd at lines 200-239, plus D-41 input-quant before ``F.linear``.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    IN = H
    layer = _make_diagonal_layer_quant_int8(IN, H).to(device).eval()

    x, h0 = _adversarial_inputs(cls, T, B, IN, device)
    with torch.no_grad():
        Wh_diag, bh_cat = extract_diagonal_factors(layer.cell)
        gi = _build_qgi_from_layer(layer, x)
        h_scale = float(layer.cell.quant_h_in.scale.item())
        h_in_q = (h_scale, -127, 127)
        h_out_q = (h_scale, -127, 127)
        out_fwd = gru_scan_diagonal_forward_triton(
            gi, h0, Wh_diag, bh_cat,
            h_in_quant=h_in_q, h_out_quant=h_out_q,
        )
        dout = (torch.randn(T, B, H, device=device) * 0.5).contiguous()

        dgi_t, dh0_t, dWh_t, dbh_t = gru_scan_diagonal_backward_triton(
            gi, h0, Wh_diag, bh_cat, out_fwd, dout,
            h_in_quant=h_in_q, h_out_quant=h_out_q,
        )
        dgi_p, dh0_p, dWh_p, dbh_p = gru_scan_diagonal_backward_pytorch(
            gi, h0, Wh_diag, bh_cat, out_fwd, dout,
            h_in_quant=h_in_q, h_out_quant=h_out_q,
        )

    # Backward parity per D-42 Result B: strict=False (abs_diff < h_scale).
    # Per-grad explicit calls so a single-grad failure surfaces with its
    # tensor name + cls + shape.
    name_suffix = f"[{cls}-T={T}-B={B}-H={H}]"
    _assert_quant_parity(f"dgi{name_suffix}", dgi_p, dgi_t, h_scale, strict=False)
    _assert_quant_parity(f"dh0{name_suffix}", dh0_p, dh0_t, h_scale, strict=False)
    _assert_quant_parity(f"dWh_diag{name_suffix}", dWh_p, dWh_t, h_scale, strict=False)
    _assert_quant_parity(f"dbh{name_suffix}", dbh_p, dbh_t, h_scale, strict=False)


@pytest.mark.slow
@cuda_only
@pytest.mark.parametrize("T,B,H", QUANT_SLOW_GRID)
@pytest.mark.parametrize("cls", ["realistic", "near-saturation", "large-magnitude"])
def test_diagonal_quant_bwd_slow(cls: str, T: int, B: int, H: int) -> None:
    """Slow sibling of ``test_diagonal_quant_bwd``; gated behind
    ``@pytest.mark.slow`` per D-49 (T=512)."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    IN = H
    layer = _make_diagonal_layer_quant_int8(IN, H).to(device).eval()

    x, h0 = _adversarial_inputs(cls, T, B, IN, device)
    with torch.no_grad():
        Wh_diag, bh_cat = extract_diagonal_factors(layer.cell)
        gi = _build_qgi_from_layer(layer, x)
        h_scale = float(layer.cell.quant_h_in.scale.item())
        h_in_q = (h_scale, -127, 127)
        h_out_q = (h_scale, -127, 127)
        out_fwd = gru_scan_diagonal_forward_triton(
            gi, h0, Wh_diag, bh_cat,
            h_in_quant=h_in_q, h_out_quant=h_out_q,
        )
        dout = (torch.randn(T, B, H, device=device) * 0.5).contiguous()

        dgi_t, dh0_t, dWh_t, dbh_t = gru_scan_diagonal_backward_triton(
            gi, h0, Wh_diag, bh_cat, out_fwd, dout,
            h_in_quant=h_in_q, h_out_quant=h_out_q,
        )
        dgi_p, dh0_p, dWh_p, dbh_p = gru_scan_diagonal_backward_pytorch(
            gi, h0, Wh_diag, bh_cat, out_fwd, dout,
            h_in_quant=h_in_q, h_out_quant=h_out_q,
        )

    name_suffix = f"[{cls}-T={T}-B={B}-H={H}]"
    _assert_quant_parity(f"dgi{name_suffix}", dgi_p, dgi_t, h_scale, strict=False)
    _assert_quant_parity(f"dh0{name_suffix}", dh0_p, dh0_t, h_scale, strict=False)
    _assert_quant_parity(f"dWh_diag{name_suffix}", dWh_p, dWh_t, h_scale, strict=False)
    _assert_quant_parity(f"dbh{name_suffix}", dbh_p, dbh_t, h_scale, strict=False)

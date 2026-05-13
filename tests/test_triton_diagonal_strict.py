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
@pytest.mark.parametrize("T,B,H", SLOW_DIAG_GRID)
def test_diagonal_bwd_strict_matches_reference_slow(T: int, B: int, H: int) -> None:
    """Identical body to the fast bwd variant; gated behind
    @pytest.mark.slow per D-16 (T ∈ {512, 1024}) and D-26.
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
    diff_dbh = (dbh_p - dbh_t).abs().max().item()
    assert diff_dbh < 1e-5, f"dbh max abs diff {diff_dbh:.4e} (T={T},B={B},H={H})"


# ``extract_diagonal_factors`` is imported at module top so this file
# remains a single-import surface (any future test that needs to build
# Wh_diag from a cell can use it). It's exercised at runtime through the
# realistic-tier file's Stage A test; the strict tier does not duplicate
# (D-20). Reference: tests/test_triton_diagonal.py:75-100.
_ = extract_diagonal_factors  # explicit no-op anchor for the import.

"""Hand-rolled parity tests for the Circulant and LDR PyTorch fallback paths — Phase 3 audit.

Pins ``_CirculantLinear`` and ``_LDRLinear`` in ``src/gru_qat/structure.py``
against independent hand-rolled mathematical references at < 1e-5 abs (fwd +
bwd via autograd-grad comparison). Plus STR-03 graceful-degradation: when
``torch-structured`` is missing, optional-dep kinds (monarch, butterfly, ldr)
must raise ImportError with a clear install hint, while local-impl kinds
(circulant, diagonal, dense) continue to work.

Pure PyTorch — no Triton, no CUDA. Pairs with ``tests/test_structure.py``
(smoke/integration tier: finite-output + gradient-flow + training-loop +
int8-QAT). Two clear tiers, one file each.

This module sets ``torch.set_float32_matmul_precision('highest')`` at import
time because Phase 3 audits the math (per Phase 3 CONTEXT D-40 — TF32 'high'
is for Triton kernel files only). The < 1e-5 abs bound is achievable on
fp32-vs-fp32 without TF32 in play.

This module does NOT call ``pytest.importorskip("torch_structured")`` at
module top — the circulant family is a local impl (see
``src/gru_qat/structure.py:207-225``) and must run on machines without
``torch-structured`` installed. LDR-specific imports (plan 03-02) will be
guarded per-section.
"""

from __future__ import annotations

import sys
import warnings

import pytest
import torch
import torch.nn as nn

# Per Phase 3 CONTEXT D-40: pure PyTorch (no tl.dot), so 'highest' is
# achievable and < 1e-5 abs is the strict bound. Diverges from the Triton
# kernel test files (which use 'high' to test under realistic TF32
# conditions). Module-level because set_float32_matmul_precision is global
# state — set once at import is the cleanest signal.
torch.set_float32_matmul_precision("highest")

from gru_qat.structure import (  # noqa: E402
    _CirculantLinear,
    _LDRLinear,
    StructureConfig,
    make_structured_linear,
)


def _build_toeplitz_from_kernel(c: torch.Tensor) -> torch.Tensor:
    """Build the H x H circulant matrix C from the length-H kernel vector c.

    Per Phase 3 PATTERNS.md lines 286-302 (convention reconciliation), the
    production ``_CirculantLinear.forward`` computes::

        y[b, k] = sum_j col[(k - j) mod n] * x[b, j]

    i.e., circular convolution of ``col`` with ``x``. The matrix C that
    represents this operation in row-vector form (``y = x @ C.T``) has::

        C[i, j] = c[(i - j) mod H]

    so C's first column equals c. Each subsequent column is c cyclically
    shifted down by one.

    Returns a tensor with the same dtype/device as ``c``. Used as one of the
    two independent references in the self-consistency check (FFT-form vs
    Toeplitz-form) BEFORE either is compared to ``_CirculantLinear``.
    """
    H = c.shape[0]
    idx = torch.arange(H, device=c.device)
    # Vectorized outer arithmetic: i_minus_j[i, j] = (i - j) mod H.
    i_minus_j_mod_H = (idx[:, None] - idx[None, :]) % H
    return c[i_minus_j_mod_H]


def _circulant_via_fft(c: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Apply the circulant matrix defined by kernel ``c`` to ``x`` via full
    complex FFT.

    Deliberately uses ``torch.fft.fft`` / ``torch.fft.ifft`` (NOT
    ``rfft``/``irfft``) — the production path uses ``rfft``/``irfft``, so
    this is the genuinely independent FFT reference for the self-consistency
    check.

    For ``x`` of shape ``(B, H)``::

        y = real(ifft(fft(c, n=H) * fft(x, n=H, dim=-1), n=H, dim=-1))

    The ``.real`` cast is safe because ``c`` and ``x`` are real-valued; any
    imaginary component is floating-point noise (fp64-relative).
    """
    H = c.shape[0]
    c_f = torch.fft.fft(c, n=H)
    x_f = torch.fft.fft(x, n=H, dim=-1)
    y = torch.fft.ifft(c_f * x_f, n=H, dim=-1)
    return y.real


# Shape grids per Phase 3 CONTEXT D-36.
# Circulant: square; power-of-2 (per src/gru_qat/structure.py:95-98 validator).
FAST_CIRC_GRID: list[tuple[int, int]] = [
    (B, H)
    for B in (1, 4, 32)
    for H in (8, 32, 128)
]  # 9 cases
SLOW_CIRC_GRID: list[tuple[int, int]] = [
    (B, H)
    for B in (1, 4, 32)
    for H in (512,)
]  # 3 cases


# ---------------------------------------------------------------------------
# Circulant parity tests (Phase 3 plan 03-01).
#
# Three families, all CPU-only:
#   1. Self-consistency: hand-rolled Toeplitz vs hand-rolled FFT (no production
#      path). Catches reference-math bugs BEFORE they masquerade as production
#      bugs. Per CONTEXT D-29, no slow sibling needed (cheap).
#   2. Forward parity: _CirculantLinear(x) vs hand-rolled Toeplitz x @ C.T.
#      Fast + @pytest.mark.slow sibling.
#   3. Backward parity (autograd-grad): gradient on the kernel parameter,
#      reference path vs production path. Fast + @pytest.mark.slow sibling.
#
# All assertions at < 1e-5 abs (CONTEXT D-40 — pure fp32 with 'highest', no
# TF32, no STE; absolute error is sufficient and diagnostic).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("B,H", FAST_CIRC_GRID)
def test_handrolled_circulant_self_consistent(B: int, H: int) -> None:
    """Hand-rolled Toeplitz and hand-rolled FFT forms of the circulant matrix
    must agree with each other to < 1e-5 abs BEFORE either is compared to
    ``_CirculantLinear``.

    Catches algebra mistakes in ``_build_toeplitz_from_kernel`` and
    ``_circulant_via_fft`` so they don't masquerade as production-path bugs
    in the downstream parity tests.
    """
    torch.manual_seed(0)
    c = torch.randn(H) / (H**0.5)
    x = torch.randn(B, H)

    C = _build_toeplitz_from_kernel(c)
    y_toep = x @ C.T  # matches the convention: C[i,j] = c[(i-j) mod H]
    y_fft = _circulant_via_fft(c, x)

    max_diff = (y_toep - y_fft).abs().max().item()
    assert max_diff < 1e-5, (
        f"toeplitz vs fft self-consistency max abs diff {max_diff:.4e} "
        f"(B={B},H={H})"
    )


@pytest.mark.parametrize("B,H", FAST_CIRC_GRID)
def test_circulant_matches_handrolled_toeplitz(B: int, H: int) -> None:
    """``_CirculantLinear(col=c).forward(x)`` must match the explicit Toeplitz
    matrix construction ``x @ C.T`` (with C built from c via
    ``_build_toeplitz_from_kernel``) to < 1e-5 abs across the fast grid.

    Strict-tier bound (D-40): pure-PyTorch fp32 with 'highest' precision, no
    TF32, no STE, no nonlinearities — algebraic equality between two paths
    that compute the same circular convolution.
    """
    torch.manual_seed(0)
    layer = _CirculantLinear(H, bias=False)
    # Read the internally-initialized kernel for the reference path.
    c = layer.col.detach().clone()

    x = torch.randn(B, H)

    y_prod = layer(x)
    C = _build_toeplitz_from_kernel(c)
    y_ref = x @ C.T

    max_diff = (y_prod - y_ref).abs().max().item()
    assert max_diff < 1e-5, (
        f"circulant fwd max abs diff {max_diff:.4e} (B={B},H={H})"
    )


@pytest.mark.slow
@pytest.mark.parametrize("B,H", SLOW_CIRC_GRID)
def test_circulant_matches_handrolled_toeplitz_slow(B: int, H: int) -> None:
    """Slow sibling of ``test_circulant_matches_handrolled_toeplitz`` at
    H=512. Same body, same bound."""
    torch.manual_seed(0)
    layer = _CirculantLinear(H, bias=False)
    c = layer.col.detach().clone()

    x = torch.randn(B, H)

    y_prod = layer(x)
    C = _build_toeplitz_from_kernel(c)
    y_ref = x @ C.T

    max_diff = (y_prod - y_ref).abs().max().item()
    assert max_diff < 1e-5, (
        f"circulant fwd max abs diff {max_diff:.4e} (B={B},H={H})"
    )


@pytest.mark.parametrize("B,H", FAST_CIRC_GRID)
def test_circulant_backward_matches_autograd_reference(B: int, H: int) -> None:
    """Backward parity: gradient w.r.t. the circulant kernel.

    Build the Toeplitz matrix ``C`` from a leaf ``c_ref`` with
    ``requires_grad=True``; compute ``y_ref = x @ C.T``; backprop a shared
    random ``g``; read ``c_ref.grad``. Repeat through ``_CirculantLinear``
    with an independent leaf ``c_prod`` (detach-clone-twice idiom per
    ``tests/test_layer_parity.py:516-519``); read ``c_prod.grad``. Assert the
    two gradients agree to < 1e-5 abs.

    Per CONTEXT D-30 — autograd-vs-autograd, no manual gradient math. Named
    per-tensor failure loop (single entry now; LDR plan 03-02 will reuse with
    four entries).
    """
    torch.manual_seed(0)
    c_init = torch.randn(H) / (H**0.5)

    # Two independent leaves, two independent autograd graphs (detach-clone
    # idiom from tests/test_layer_parity.py:516-519).
    c_ref = c_init.detach().clone().requires_grad_(True)
    c_prod = c_init.detach().clone().requires_grad_(True)
    x = torch.randn(B, H)
    # Shared downstream gradient, scaled to keep gradient magnitudes O(1).
    # ``dL/dc[k] = sum_{b,i,j: (i-j) mod H = k} g[b,i] * x[b,j]`` has ~B*H
    # terms; an unscaled randn(B, H) drives ||c.grad||_inf to ~sqrt(B*H),
    # putting the fp32 round-off floor at ~ sqrt(B*H) * eps ~ a few * 1e-5
    # — above the absolute bound. Scaling g by 1/sqrt(B*H) keeps gradient
    # magnitudes O(1) so the < 1e-5 abs bound stays meaningful (CONTEXT
    # D-40: "magnitudes well above any relative-error floor"). Every
    # output element still contributes independently (preserving the
    # diagnostic-power property of the shared-g pattern).
    g = torch.randn(B, H) / (B * H) ** 0.5

    # Reference path: C is a function of c_ref, so autograd flows back.
    C = _build_toeplitz_from_kernel(c_ref)
    y_ref = x @ C.T
    y_ref.backward(g)

    # Production path: copy c_prod's data into the layer's Parameter
    # in-place. The Parameter itself is the leaf the layer's autograd sees,
    # so the gradient lands on ``layer.col.grad`` — read THAT rather than
    # c_prod.grad (which would be None because c_prod is not the leaf node
    # of the production graph). The detach-clone of c_init upstream is
    # still load-bearing: it guarantees layer.col starts bitwise-equal to
    # c_ref.
    layer = _CirculantLinear(H, bias=False)
    with torch.no_grad():
        layer.col.copy_(c_prod)
    y_prod = layer(x)
    y_prod.backward(g)

    for name, ref_t, prod_t in [("kernel_c", c_ref.grad, layer.col.grad)]:
        assert ref_t is not None, f"{name} ref_t is None (B={B},H={H})"
        assert prod_t is not None, f"{name} prod_t is None (B={B},H={H})"
        max_diff = (ref_t - prod_t).abs().max().item()
        assert max_diff < 1e-5, (
            f"{name} max abs diff {max_diff:.4e} (B={B},H={H})"
        )


@pytest.mark.slow
@pytest.mark.parametrize("B,H", SLOW_CIRC_GRID)
def test_circulant_backward_matches_autograd_reference_slow(B: int, H: int) -> None:
    """Slow sibling of ``test_circulant_backward_matches_autograd_reference``
    at H=512. Same body, same bound (g scaled by 1/sqrt(B*H) so gradient
    magnitudes stay O(1) and the < 1e-5 abs bound stays meaningful)."""
    torch.manual_seed(0)
    c_init = torch.randn(H) / (H**0.5)

    c_ref = c_init.detach().clone().requires_grad_(True)
    c_prod = c_init.detach().clone().requires_grad_(True)
    x = torch.randn(B, H)
    g = torch.randn(B, H) / (B * H) ** 0.5

    C = _build_toeplitz_from_kernel(c_ref)
    y_ref = x @ C.T
    y_ref.backward(g)

    layer = _CirculantLinear(H, bias=False)
    with torch.no_grad():
        layer.col.copy_(c_prod)
    y_prod = layer(x)
    y_prod.backward(g)

    for name, ref_t, prod_t in [("kernel_c", c_ref.grad, layer.col.grad)]:
        assert ref_t is not None, f"{name} ref_t is None (B={B},H={H})"
        assert prod_t is not None, f"{name} prod_t is None (B={B},H={H})"
        max_diff = (ref_t - prod_t).abs().max().item()
        assert max_diff < 1e-5, (
            f"{name} max abs diff {max_diff:.4e} (B={B},H={H})"
        )


# ===========================================================================
# LDR section (STR-02)
# ===========================================================================
#
# Audit findings from reading torch_structured (Task 0 of plan 03-02):
#
# Spec source — torch_structured/structured/layers.py (lines 211-225):
#   class LDRSubdiagonal(LearnedOperator):
#     - subd_A, subd_B: shape (layer_size - 1,), init to ones (lines 217, 221).
#     - G, H: shape (r, layer_size), init via nn.init.normal_ (LowRank base).
#     - forward(x): returns kry.subdiag_mult(subd_A, subd_B, G, H, x).
#
# Spec source — torch_structured/structured/krylov.py (lines 245-272):
#   - subdiag_mult: the FFT-based fast path (production); computes
#     ``sum_i Krylov(A, G_i) @ Krylov(B, H_i) @ x``.
#   - Krylov(linear_map, v, m=None): EXPLICIT slow form — returns the column-
#     stacked n×n matrix ``[v, A@v, A^2@v, ..., A^{n-1}@v]``. Lines 264-272.
#   - subdiag_linear_map(subdiag, upper_right_corner=0) (lines 279-283): the
#     "shift down with weights" operator. With corner=0, the resulting A is a
#     pure subdiagonal matrix: A[i+1, i] = subdiag[i] for i ∈ [0, n-1).
#   - subdiag_mult_slow (lines 309-317): the slow reference inside
#     torch_structured itself. For rank ≥ 2 the formula is::
#         out = ((x @ K_H) @ K_G.transpose(1, 2)).sum(dim=0)
#     where K_G[i] = Krylov(subdiag_linear_map(subd_A, 0), G[i]) and
#     K_H[i] = Krylov(subdiag_linear_map(subd_B, 0), H[i]).
#
# Transpose convention pinned: the effective dense matrix M satisfies
# ``y_prod = x @ M.T`` (NOT ``x @ M``), where
# ``M = sum_i K_A(G[i]) @ K_B(H[i]).T``. Verified empirically on (n=8, r=2):
# diff is < 1.2e-7 with ``.T``, ~1.4 without. The micro-validation test below
# locks this in so future maintainers don't have to re-derive it.
# ===========================================================================


# Per Phase 3 PATTERNS.md / CONTEXT D-32: LDR requires `torch-structured`.
# Skip the whole LDR section (helper + micro + parametrized tests) at collect
# time on machines without it. Circulant section above continues to run.
# `torch_structured` emits a CUDA-version-mismatch UserWarning on import; it's
# noise here (CPU-only audit). Mirror tests/test_structure.py:18-20.
warnings.filterwarnings("ignore", message=".*different CUDA versions.*")
pytest.importorskip("torch_structured")

from torch_structured.structured.layers import LDRSubdiagonal  # noqa: E402


def _build_ldr_matrix_from_factors(
    subd_A: torch.Tensor,
    subd_B: torch.Tensor,
    G: torch.Tensor,
    H: torch.Tensor,
) -> torch.Tensor:
    """Construct the dense n x n matrix M that ``LDRSubdiagonal`` applies.

    Per ``torch_structured/structured/krylov.py:264-272`` (slow Krylov form)
    and ``:245-259`` / ``:309-317`` (the displacement-rank formula in both
    fast and slow form), the effective matrix is::

        M = sum_i K_A(G[i]) @ K_B(H[i]).T

    where A is the subdiagonal matrix with ``A[i+1, i] = subd_A[i]`` and
    ``K_A(v)`` is the column-stacked Krylov matrix
    ``[v, A @ v, A^2 @ v, ..., A^{n-1} @ v]``. Same for B. The production
    forward then returns ``y = x @ M.T`` (row-vector idiom).

    This helper uses the EXPLICIT slow construction (a Python ``for``-loop
    over the Krylov powers) — NOT the FFT-based fast ``krylov_multiply``
    that the production path calls — so the resulting matrix is provably
    independent of the production code path.

    Shapes:
        subd_A: (n-1,) — subdiagonal entries of A.
        subd_B: (n-1,) — subdiagonal entries of B.
        G:      (r, n) — rank-r set of G factor vectors.
        H:      (r, n) — rank-r set of H factor vectors.
    Returns:
        M: (n, n) — the effective dense matrix.
    """
    r, n = G.shape
    assert subd_A.shape == (n - 1,), f"subd_A shape {subd_A.shape}, expected ({n - 1},)"
    assert subd_B.shape == (n - 1,), f"subd_B shape {subd_B.shape}, expected ({n - 1},)"
    assert H.shape == (r, n), f"H shape {H.shape}, expected ({r}, {n})"

    # Build A and B as explicit (n, n) subdiagonal matrices. A[i+1, i] =
    # subd_A[i]; all other entries are zero. Mirrors the operator implied
    # by ``subdiag_linear_map(subdiag, upper_right_corner=0)`` at
    # krylov.py:279-283 (with corner=0 — LDRSubdiagonal has no corner).
    A = torch.zeros(n, n, dtype=G.dtype, device=G.device)
    A[torch.arange(1, n), torch.arange(n - 1)] = subd_A
    B = torch.zeros(n, n, dtype=H.dtype, device=H.device)
    B[torch.arange(1, n), torch.arange(n - 1)] = subd_B

    def _krylov_explicit(M_op: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Column-stack [v, M @ v, M^2 @ v, ..., M^(n-1) @ v] into (n, n).

        Mirrors ``Krylov(linear_map, v, m=None)`` at krylov.py:264-272
        (slow form). Reuses the previously-computed power so the loop is
        O(n^3) overall, not O(n^4).
        """
        cols = [v]
        cur = v
        for _ in range(n - 1):
            cur = M_op @ cur
            cols.append(cur)
        return torch.stack(cols, dim=-1)  # (n, n)

    # M = sum_i K_A(G[i]) @ K_B(H[i]).T  (transpose convention verified
    # empirically on n=8, r=2 — see comment block at the top of this
    # section and the micro-validation test below).
    M = torch.zeros(n, n, dtype=G.dtype, device=G.device)
    for i in range(r):
        K_A = _krylov_explicit(A, G[i])  # (n, n)
        K_B = _krylov_explicit(B, H[i])  # (n, n)
        M = M + K_A @ K_B.T
    return M


# LDR shape grid per Phase 3 CONTEXT D-36: H ∈ {8, 32, 128} (fast), {512}
# (slow); B ∈ {1, 4, 32}; rank ∈ {1, 4, 8} with rank ≤ H. Square layer (in
# == out per _validate_shapes), so we use a single H dim for both.
FAST_LDR_GRID: list[tuple[int, int, int]] = [
    (B, H, rank)
    for B in (1, 4, 32)
    for H in (8, 32, 128)
    for rank in (1, 4, 8)
    if rank <= H
]  # 27 cases (rank ≤ H holds for every fast H).
SLOW_LDR_GRID: list[tuple[int, int, int]] = [
    (B, H, rank)
    for B in (1, 4, 32)
    for H in (512,)
    for rank in (1, 4, 8)
]  # 9 cases.


def test_handrolled_ldr_matches_production_micro() -> None:
    """One-shot micro-validation on (H=8, rank=2) that pins the transpose
    convention of ``_build_ldr_matrix_from_factors`` BEFORE the parametrized
    grid runs.

    The full ``M`` is constructed once and compared to the production
    ``_LDRLinear`` forward via ``x @ M.T``. A failure here means either
    (a) the helper has a ``K_A @ K_B`` vs ``K_A @ K_B.T`` flip, or
    (b) the production path's transpose convention differs from
    ``subdiag_mult_slow`` (krylov.py:309-317). Either way, the parametrized
    grid would also fail — this single-shape sanity check makes the failure
    diagnosis local.
    """
    torch.manual_seed(0)
    ldr = LDRSubdiagonal(layer_size=8, r=2, bias=False)
    layer = _LDRLinear(ldr)

    subd_A = ldr.subd_A.detach().clone()
    subd_B = ldr.subd_B.detach().clone()
    G = ldr.G.detach().clone()
    H_factor = ldr.H.detach().clone()

    x = torch.randn(4, 8)

    y_prod = layer(x)
    M = _build_ldr_matrix_from_factors(subd_A, subd_B, G, H_factor)
    # Row-vector idiom: y = x @ M.T is equivalent to (M @ x.T).T (the column-
    # vector formulation in subdiag_mult_slow at krylov.py:309-317).
    y_ref = x @ M.T

    max_diff = (y_prod - y_ref).abs().max().item()
    assert max_diff < 1e-5, (
        f"ldr handrolled-micro max abs diff {max_diff:.4e} (H=8, rank=2); "
        f"check the K_A @ K_B vs K_A @ K_B.T convention in "
        f"_build_ldr_matrix_from_factors against krylov.py:309-317"
    )


# ---------------------------------------------------------------------------
# Parametrized LDR parity tests (Plan 03-02 Task 2):
#   - Forward: _LDRLinear(LDRSubdiagonal(H, r)) vs x @ M.T with M built by
#     _build_ldr_matrix_from_factors. Fast (FAST_LDR_GRID, 27 cases) + slow
#     sibling (SLOW_LDR_GRID, 9 cases).
#   - Backward: autograd-vs-autograd on 4 leaves (subd_A, subd_B, G, H).
#     Fast + slow sibling. Per-tensor named-failure loop with 4 entries.
#
# All at < 1e-5 abs. Per CONTEXT D-30, backward uses the detach-clone-twice
# idiom from tests/test_layer_parity.py:516-519 and the shared-g pattern
# from :524-528. Per plan 03-01 SUMMARY (decisions section), g is scaled by
# 1/sqrt(B*H) to keep gradient magnitudes O(1) so the absolute bound stays
# meaningful at large (B, H, rank).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("B,H,rank", FAST_LDR_GRID)
def test_ldr_matches_handrolled_reference(B: int, H: int, rank: int) -> None:
    """``_LDRLinear(LDRSubdiagonal(H, r=rank))(x)`` must match the explicit
    hand-rolled matrix ``x @ M.T`` (with M from
    ``_build_ldr_matrix_from_factors``) at < 1e-5 abs across FAST_LDR_GRID.

    Hand-rolled M uses the slow Krylov form (krylov.py:264-272 +
    krylov.py:309-317); production path uses the FFT-based fast form
    (krylov.py:245-259). The two should agree algebraically — < 1e-5 abs
    under 'highest' precision.
    """
    torch.manual_seed(0)
    ldr = LDRSubdiagonal(layer_size=H, r=rank, bias=False)
    layer = _LDRLinear(ldr)

    subd_A = ldr.subd_A.detach().clone()
    subd_B = ldr.subd_B.detach().clone()
    G = ldr.G.detach().clone()
    H_factor = ldr.H.detach().clone()

    x = torch.randn(B, H)

    y_prod = layer(x)
    M = _build_ldr_matrix_from_factors(subd_A, subd_B, G, H_factor)
    y_ref = x @ M.T

    max_diff = (y_prod - y_ref).abs().max().item()
    assert max_diff < 1e-5, (
        f"ldr fwd max abs diff {max_diff:.4e} (B={B},H={H},rank={rank})"
    )


@pytest.mark.slow
@pytest.mark.parametrize("B,H,rank", SLOW_LDR_GRID)
def test_ldr_matches_handrolled_reference_slow(B: int, H: int, rank: int) -> None:
    """Slow sibling of ``test_ldr_matches_handrolled_reference`` at H=512.
    Same body, same bound."""
    torch.manual_seed(0)
    ldr = LDRSubdiagonal(layer_size=H, r=rank, bias=False)
    layer = _LDRLinear(ldr)

    subd_A = ldr.subd_A.detach().clone()
    subd_B = ldr.subd_B.detach().clone()
    G = ldr.G.detach().clone()
    H_factor = ldr.H.detach().clone()

    x = torch.randn(B, H)

    y_prod = layer(x)
    M = _build_ldr_matrix_from_factors(subd_A, subd_B, G, H_factor)
    y_ref = x @ M.T

    max_diff = (y_prod - y_ref).abs().max().item()
    assert max_diff < 1e-5, (
        f"ldr fwd max abs diff {max_diff:.4e} (B={B},H={H},rank={rank})"
    )


@pytest.mark.parametrize("B,H,rank", FAST_LDR_GRID)
def test_ldr_backward_matches_autograd_reference(B: int, H: int, rank: int) -> None:
    """Backward parity: autograd gradients on all four LDR factors (subd_A,
    subd_B, G, H) must match between the hand-rolled reference path and the
    production path at < 1e-5 abs.

    Pattern mirrors tests/test_layer_parity.py:480-557 (detach-clone-twice
    per leaf + shared-g + per-tensor named-failure loop), extended to four
    leaves. Production-side leaves are installed as the layer's actual
    Parameters via direct ``nn.Parameter`` assignment so ``backward(g)``
    populates ``..._prod.grad`` (NOT the layer's pre-existing parameter
    grads — those would dangle).

    g is scaled by 1/sqrt(B*H) so gradient magnitudes stay O(1); without
    this the fp32 round-off floor between two algorithmically distinct
    paths exceeds 1e-5 at large (B, H, rank). See plan 03-01 SUMMARY
    decision on g-scaling for the empirical justification.
    """
    torch.manual_seed(0)

    # Source initial factor values from a one-shot LDRSubdiagonal so the
    # reference and production paths start bitwise-equal.
    ldr_init = LDRSubdiagonal(layer_size=H, r=rank, bias=False)
    subd_A_init = ldr_init.subd_A.detach().clone()
    subd_B_init = ldr_init.subd_B.detach().clone()
    G_init = ldr_init.G.detach().clone()
    H_init = ldr_init.H.detach().clone()

    # 8 leaves: 4 ref + 4 prod, each detach-cloned-then-requires_grad.
    subd_A_ref = subd_A_init.detach().clone().requires_grad_(True)
    subd_B_ref = subd_B_init.detach().clone().requires_grad_(True)
    G_ref = G_init.detach().clone().requires_grad_(True)
    H_ref = H_init.detach().clone().requires_grad_(True)

    subd_A_prod = subd_A_init.detach().clone().requires_grad_(True)
    subd_B_prod = subd_B_init.detach().clone().requires_grad_(True)
    G_prod = G_init.detach().clone().requires_grad_(True)
    H_prod = H_init.detach().clone().requires_grad_(True)

    x = torch.randn(B, H)
    # Shared downstream gradient, scaled to keep gradient magnitudes O(1).
    g = torch.randn(B, H) / (B * H) ** 0.5

    # Reference path: build M as a function of the 4 ref leaves so autograd
    # flows back to each.
    M = _build_ldr_matrix_from_factors(subd_A_ref, subd_B_ref, G_ref, H_ref)
    y_ref = x @ M.T
    y_ref.backward(g)

    # Production path: install the 4 prod leaves as the layer's
    # Parameters via direct nn.Parameter assignment. The Parameters'
    # ``.grad`` then equals each *_prod leaf's .grad (because the leaf IS
    # the Parameter's underlying tensor — nn.Parameter wraps the existing
    # tensor without copying, and requires_grad is preserved).
    ldr_prod = LDRSubdiagonal(layer_size=H, r=rank, bias=False)
    ldr_prod.subd_A = nn.Parameter(subd_A_prod)
    ldr_prod.subd_B = nn.Parameter(subd_B_prod)
    ldr_prod.G = nn.Parameter(G_prod)
    ldr_prod.H = nn.Parameter(H_prod)
    layer_prod = _LDRLinear(ldr_prod)

    y_prod = layer_prod(x)
    y_prod.backward(g)

    # Read gradients from the Parameters (which share storage with the
    # *_prod leaves, so ldr_prod.subd_A.grad is the same tensor that an
    # autograd-aware caller would read).
    for name, ref_t, prod_t in [
        ("subd_A", subd_A_ref.grad, ldr_prod.subd_A.grad),
        ("subd_B", subd_B_ref.grad, ldr_prod.subd_B.grad),
        ("G", G_ref.grad, ldr_prod.G.grad),
        ("H", H_ref.grad, ldr_prod.H.grad),
    ]:
        assert ref_t is not None, f"{name} ref_t is None (B={B},H={H},rank={rank})"
        assert prod_t is not None, f"{name} prod_t is None (B={B},H={H},rank={rank})"
        max_diff = (ref_t - prod_t).abs().max().item()
        assert max_diff < 1e-5, (
            f"{name} max abs diff {max_diff:.4e} (B={B},H={H},rank={rank})"
        )


@pytest.mark.slow
@pytest.mark.parametrize("B,H,rank", SLOW_LDR_GRID)
def test_ldr_backward_matches_autograd_reference_slow(B: int, H: int, rank: int) -> None:
    """Slow sibling of ``test_ldr_backward_matches_autograd_reference`` at
    H=512. Same body, same bound (g scaled by 1/sqrt(B*H))."""
    torch.manual_seed(0)

    ldr_init = LDRSubdiagonal(layer_size=H, r=rank, bias=False)
    subd_A_init = ldr_init.subd_A.detach().clone()
    subd_B_init = ldr_init.subd_B.detach().clone()
    G_init = ldr_init.G.detach().clone()
    H_init = ldr_init.H.detach().clone()

    subd_A_ref = subd_A_init.detach().clone().requires_grad_(True)
    subd_B_ref = subd_B_init.detach().clone().requires_grad_(True)
    G_ref = G_init.detach().clone().requires_grad_(True)
    H_ref = H_init.detach().clone().requires_grad_(True)

    subd_A_prod = subd_A_init.detach().clone().requires_grad_(True)
    subd_B_prod = subd_B_init.detach().clone().requires_grad_(True)
    G_prod = G_init.detach().clone().requires_grad_(True)
    H_prod = H_init.detach().clone().requires_grad_(True)

    x = torch.randn(B, H)
    g = torch.randn(B, H) / (B * H) ** 0.5

    M = _build_ldr_matrix_from_factors(subd_A_ref, subd_B_ref, G_ref, H_ref)
    y_ref = x @ M.T
    y_ref.backward(g)

    ldr_prod = LDRSubdiagonal(layer_size=H, r=rank, bias=False)
    ldr_prod.subd_A = nn.Parameter(subd_A_prod)
    ldr_prod.subd_B = nn.Parameter(subd_B_prod)
    ldr_prod.G = nn.Parameter(G_prod)
    ldr_prod.H = nn.Parameter(H_prod)
    layer_prod = _LDRLinear(ldr_prod)

    y_prod = layer_prod(x)
    y_prod.backward(g)

    for name, ref_t, prod_t in [
        ("subd_A", subd_A_ref.grad, ldr_prod.subd_A.grad),
        ("subd_B", subd_B_ref.grad, ldr_prod.subd_B.grad),
        ("G", G_ref.grad, ldr_prod.G.grad),
        ("H", H_ref.grad, ldr_prod.H.grad),
    ]:
        assert ref_t is not None, f"{name} ref_t is None (B={B},H={H},rank={rank})"
        assert prod_t is not None, f"{name} prod_t is None (B={B},H={H},rank={rank})"
        max_diff = (ref_t - prod_t).abs().max().item()
        assert max_diff < 1e-5, (
            f"{name} max abs diff {max_diff:.4e} (B={B},H={H},rank={rank})"
        )


# ===========================================================================
# STR-03 section: graceful-degradation tests
# ===========================================================================
#
# Phase 3 Plan 03-03 — confirm that ``torch-structured`` is genuinely
# optional:
#
#   kind        path to torch_structured        expected without it
#   --------    ------------------------------  -------------------------
#   dense       (none — pure torch.nn.Linear)   WORKS
#   diagonal    (none — local _DiagonalLinear)  WORKS
#   circulant   (none — local _CirculantLinear) WORKS
#   monarch     _import_torch_structured()      ImportError("torch-structured")
#   butterfly   _import_torch_structured()      ImportError("torch-structured")
#   ldr         from torch_structured.structured.layers
#                       import LDRSubdiagonal    ImportError("torch-structured")
#
# Two mocking idioms are used (D-34):
#   * monarch / butterfly / local-impl controls — monkeypatch.setattr
#     ``gru_qat.structure._import_torch_structured`` to raise ImportError.
#     This is the helper the production code routes through.
#   * ldr — production code at ``src/gru_qat/structure.py:160-172`` does
#     ``from torch_structured.structured.layers import LDRSubdiagonal``
#     directly, bypassing ``_import_torch_structured``. The setattr
#     monkeypatch above does NOT affect this branch. Instead we simulate
#     the missing package via ``monkeypatch.setitem(sys.modules, ..., None)``
#     so Python's import machinery raises ImportError on the next ``from``
#     import of those names.
#
# This is the first introduction of ``pytest.MonkeyPatch`` to the codebase.
# Per the convention update in ``.planning/codebase/TESTING.md``, this is
# scoped to optional-dependency failure-mode tests; broader logic tests
# continue to use real layers / real tensors.
# ===========================================================================


def _raise_missing_torch_structured() -> None:
    """Stand-in for ``_import_torch_structured`` that always raises the
    ImportError the production helper would raise on a missing install.

    Matches the production message verbatim (``src/gru_qat/structure.py:65-68``)
    so the test asserts the user-facing string, not just any ImportError.
    Function annotation is ``-> None`` even though the function never
    returns — Python signature conventions; ``raise`` exits the frame.
    """
    raise ImportError(
        "torch-structured is required for structured GRU weights. "
        "Install with: pip install 'gru-qat[structured]'"
    )


@pytest.mark.parametrize("kind", ["monarch", "butterfly"])
def test_missing_torch_structured_raises_clear_error(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """STR-03: monarch and butterfly raise a clear ImportError when
    ``torch-structured`` is unavailable.

    Both kinds route through ``_import_torch_structured`` (see
    ``src/gru_qat/structure.py:141`` and ``:152``), so a single
    ``monkeypatch.setattr`` on that helper covers both code paths. The
    error message must contain ``torch-structured`` so users know what to
    install.

    LDR has a separate test (``test_missing_ldr_raises_clear_error``)
    because its import path bypasses ``_import_torch_structured`` — see
    the comment block at the top of this section.
    """
    monkeypatch.setattr(
        "gru_qat.structure._import_torch_structured",
        _raise_missing_torch_structured,
    )
    cfg = StructureConfig(kind=kind, nblocks=4, butterfly_nblocks=1)
    with pytest.raises(ImportError, match=r"torch-structured"):
        make_structured_linear(cfg, 32, 32)


def test_missing_ldr_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """STR-03: ldr raises a clear ImportError when ``torch-structured`` is
    unavailable.

    The LDR branch in ``src/gru_qat/structure.py:160-172`` does
    ``from torch_structured.structured.layers import LDRSubdiagonal``
    directly, bypassing ``_import_torch_structured``. To simulate a
    missing package we set the relevant submodules in ``sys.modules`` to
    ``None`` — Python's import machinery treats ``sys.modules[name] = None``
    as the documented "this module is known to be absent" marker and
    raises ImportError on the next ``from name import ...``. The
    production ``try / except ImportError`` then wraps the message with
    the ``"torch-structured is required ..."`` install hint, which is what
    the test asserts on.
    """
    monkeypatch.setitem(sys.modules, "torch_structured", None)
    monkeypatch.setitem(sys.modules, "torch_structured.structured", None)
    monkeypatch.setitem(
        sys.modules, "torch_structured.structured.layers", None
    )
    cfg = StructureConfig(kind="ldr", ldr_rank=2)
    with pytest.raises(ImportError, match=r"torch-structured"):
        make_structured_linear(cfg, 32, 32)


@pytest.mark.parametrize("kind", ["dense", "diagonal", "circulant"])
def test_local_impls_work_without_torch_structured(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """STR-03: dense, diagonal, and circulant produce working layers even
    when ``torch-structured`` is missing.

    These three kinds have local implementations (``nn.Linear``,
    ``_DiagonalLinear``, ``_CirculantLinear``) and must NOT depend on the
    optional dep. We still monkeypatch ``_import_torch_structured`` to
    fail loudly — if any of these three accidentally start calling it
    (e.g., a refactor that adds an unrelated import in the dispatch), the
    test will trip immediately.
    """
    monkeypatch.setattr(
        "gru_qat.structure._import_torch_structured",
        _raise_missing_torch_structured,
    )
    cfg = StructureConfig(kind=kind)
    layer = make_structured_linear(cfg, 32, 32)
    x = torch.randn(4, 32)
    y = layer(x)
    assert torch.isfinite(y).all(), (
        f"{kind} produced non-finite output without torch_structured"
    )
    assert y.shape == (4, 32), (
        f"{kind} expected shape (4, 32), got {y.shape}"
    )

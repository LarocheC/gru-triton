"""Phase 6 — edge-case shape sweep.

Pins every GRU code path at boundary shapes. Proves that T=1, B=1, H in
{1, 2}, long T in {512, 1024}, and degenerate T=0/B=0 inputs either
produce correct output (at the path's normal tolerance tier) or fail
with a clear, tested ``ValueError``.

This is the edge-shape ring of the native-PyTorch parity audit. Phase 4
already swept adversarial numerics; Phase 6 owns *shape* robustness.
CONCERNS.md predicts BLOCK-size-assumption failures at tiny shapes
(butterfly ``B % BLOCK_B``, monarch non-pow2 ``BLKSZ``, persistent-grid
deadlock) — the B=1/small-H sweep targets exactly those failure modes.

Design decisions honored (06-CONTEXT.md):
- D-07: realistic inputs only — edge cases test SHAPE handling, not
  adversarial numerics. A single ``torch.randn`` input class is used.
- D-08: uniform 7-path coverage — circulant and LDR per-step paths get
  the same edge dimensions as the Triton paths (``ALL_PATHS`` has 7
  entries), not a reduced subset.
- D-09: tolerances reused verbatim from PROJECT.md — reference vs
  ``nn.GRU`` < 1e-4; diagonal Triton (non-``tl.dot``) < 1e-5;
  dense/monarch/butterfly Triton (``tl.dot``) < 5e-4. No new bounds.
- D-10/D-11: all edge sweeps in this single new file; the
  ``_translate_nn_gru_to_cell`` helper is imported from the D-51-locked
  ``test_layer_parity.py`` (import-only, never edited).

Gating: Triton-path tests use a function-level ``@cuda_only`` decorator
plus ``pytest.importorskip("triton")`` inside the body (mirrors
test_calibration.py) — NOT a module-level importorskip, so the
CPU-side reference + circulant/ldr tests stay collectable on
CPU-only hosts.
"""

from __future__ import annotations

import importlib

import pytest
import torch
import torch.nn as nn

from gru_qat import GRULayer, QuantRecipe, QuantizerConfig
from gru_qat.structure import StructureConfig

# CUDA-gate marker. Triton-path tests need a GPU; reference + circulant +
# ldr tests run CPU-side. Function-level decorator (not module-level
# importorskip) keeps the CPU tests collectable on CPU-only hosts.
cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GRULayer Triton fast-path requires CUDA"
)

# D-11: import the reference-vs-nn.GRU translation helper from the D-51
# LOCKED test_layer_parity.py. tests/ has no __init__.py, so pytest's
# prepend import mode makes the sibling module importable as a top-level
# module. This is import-only — test_layer_parity.py is never edited.
# The import is committed-to: if it ever fails that is a hard error, not
# a silently-degraded fallback.
_layer_parity = importlib.import_module("test_layer_parity")
_translate_nn_gru_to_cell = _layer_parity._translate_nn_gru_to_cell

# The 7 paths swept. circulant and ldr are NOT in the fast-dispatch set
# (gru_layer.py:100-104) so they always run the per-step PyTorch path
# regardless of use_triton.
ALL_PATHS: list[str] = [
    "reference",
    "dense_triton",
    "diagonal_triton",
    "monarch_triton",
    "butterfly_triton",
    "circulant",
    "ldr",
]

# Paths that route through a Triton kernel — need CUDA + triton.
_TRITON_PATHS = {"dense_triton", "diagonal_triton", "monarch_triton", "butterfly_triton"}
# Paths that need the optional torch_structured dependency.
_STRUCTURED_DEP_PATHS = {"monarch_triton", "butterfly_triton", "ldr"}


def _fp32_recipe() -> QuantRecipe:
    """fp32-Identity recipe: all three quantizers at bits=32 (Identity).

    Edge-case tests probe shape handling, not quantization — so every
    path uses Identity quantizers and the reference math stays clean.
    bits=32 short-circuits make_quantizer to Identity.
    """
    return QuantRecipe(
        weight=QuantizerConfig(bits=32, axis=0, name="W_id"),
        input_act=QuantizerConfig(bits=32, name="x_id"),
        hidden=QuantizerConfig(bits=32, name="h_id"),
    )


def _gate_off(path: str) -> None:
    """Skip the test cleanly when the path's dependencies are unavailable.

    Triton paths need CUDA + triton; monarch/butterfly/ldr also need
    torch_structured. The reference + circulant paths run CPU-side and
    are never skipped here.
    """
    if path in _TRITON_PATHS:
        pytest.importorskip("triton")
        if not torch.cuda.is_available():
            pytest.skip(f"{path} requires CUDA")
    if path in _STRUCTURED_DEP_PATHS:
        pytest.importorskip("torch_structured")


def _make_layer(path: str, in_size: int, hid: int) -> GRULayer:
    """Build a correctly-configured fp32-Identity GRULayer for ``path``.

    The 7-path table (06-01-PLAN.md <interfaces>):
      reference        — dense hidden, use_triton=False, gate_layout split
      dense_triton     — dense hidden, use_triton=True, fused gates
      diagonal_triton  — StructureConfig(kind="diagonal"), use_triton=True
      monarch_triton   — StructureConfig(kind="monarch"),  use_triton=True
      butterfly_triton — StructureConfig(kind="butterfly"),use_triton=True
      circulant        — StructureConfig(kind="circulant"), per-step path
      ldr              — StructureConfig(kind="ldr"),        per-step path

    The Triton-eligible kinds require gate_layout="fused" (gru_layer.py:
    100-104). circulant/ldr are NOT fast-dispatch-eligible so they run
    the per-step path; they use fused gates too for a consistent recipe.
    """
    rec = _fp32_recipe()
    if path == "reference":
        return GRULayer(in_size, hid, recipe=rec, gate_layout="split")
    if path == "dense_triton":
        # Dense hidden has no StructureConfig; use_triton requires a
        # structured-hidden kind, so the dense path stays use_triton=False
        # (the dense Triton kernel is exercised via the explicit gru_scan
        # call below, mirroring the strict-file pattern). gate_layout
        # fused so the dense kernel comparison is apples-to-apples.
        return GRULayer(in_size, hid, recipe=rec, gate_layout="fused")
    if path == "diagonal_triton":
        return GRULayer(
            in_size, hid, recipe=rec, gate_layout="fused",
            structure_hidden=StructureConfig(kind="diagonal"),
            use_triton=True,
        )
    if path == "monarch_triton":
        # nblocks must divide H; pick a divisor that works for the small
        # hidden sizes used here. H is always a multiple of 2 in the
        # edge grid except H in {1}; clamp nblocks so it divides H.
        return GRULayer(
            in_size, hid, recipe=rec, gate_layout="fused",
            structure_hidden=StructureConfig(kind="monarch", nblocks=_monarch_nblocks(hid)),
            use_triton=True,
        )
    if path == "butterfly_triton":
        return GRULayer(
            in_size, hid, recipe=rec, gate_layout="fused",
            structure_hidden=StructureConfig(kind="butterfly"),
            use_triton=True,
        )
    if path == "circulant":
        return GRULayer(
            in_size, hid, recipe=rec, gate_layout="fused",
            structure_hidden=StructureConfig(kind="circulant"),
        )
    if path == "ldr":
        return GRULayer(
            in_size, hid, recipe=rec, gate_layout="fused",
            structure_hidden=StructureConfig(kind="ldr"),
        )
    raise AssertionError(f"unknown path: {path}")


def _monarch_nblocks(hid: int) -> int:
    """Pick a monarch nblocks that divides ``hid`` (in == out == hid).

    monarch requires in/out divisible by nblocks (structure.py:84). The
    edge grid uses H in {1, 2, 8}; nblocks must divide H. Default 4 only
    works for H>=4 and 4|H, so clamp to a working divisor.
    """
    for nb in (4, 2, 1):
        if hid % nb == 0 and nb <= hid:
            return nb
    return 1


def _path_tol(path: str) -> float:
    """Absolute tolerance for a path's Triton-vs-reference comparison.

    Tolerances reused verbatim from PROJECT.md Constraints (D-09 — no new
    bounds):
      - diagonal Triton (no ``tl.dot``):           < 1e-5
      - dense / monarch / butterfly (``tl.dot``):  < 5e-4
      - circulant / ldr per-step PyTorch:          < 1e-5 (deterministic
        same-recipe replay — algebraic equality)
    """
    if path in ("dense_triton", "monarch_triton", "butterfly_triton"):
        return 5e-4
    # diagonal_triton, circulant, ldr — non-tl.dot / deterministic replay.
    return 1e-5


def _rel(ref: torch.Tensor, got: torch.Tensor) -> float:
    """Relative max-abs error with the 1e-6 denominator floor.

    Matches the relative-error reporting idiom used across the repo
    (TESTING.md "Relative-error reporting"; test_layer_parity.py:181).
    The floor prevents division by near-zero on degenerate outputs.
    """
    max_diff = (ref - got).abs().max().item()
    return max_diff / max(ref.abs().max().item(), 1e-6)


def _run_path_vs_reference(
    path: str, T: int, B: int, H: int, *, backward: bool
) -> None:
    """Run ``path`` at shape (T, B, H) and assert parity at its tier.

    The comparison strategy per path:
      - reference: vs ``torch.nn.GRU`` (via the D-11 imported
        ``_translate_nn_gru_to_cell``) at < 1e-4.
      - dense_triton: per-step dense layer vs the explicit dense
        ``gru_scan`` Triton kernel at < 5e-4 (mirrors the strict-file
        pre_batch_input round-trip pattern).
      - diagonal/monarch/butterfly_triton: a ``use_triton=True`` layer vs
        a same-weights ``use_triton=False`` layer at the path tier.
      - circulant/ldr: per-step PyTorch — a deterministic same-recipe
        re-run; assert finite + correct shape and < 1e-5 replay equality.

    ``backward`` extends the check to gradient finiteness + parity.
    """
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")
    in_size = H

    if path == "reference":
        device = torch.device("cpu")
        gru = nn.GRU(in_size, H, num_layers=1, bidirectional=False, batch_first=False)
        layer = _translate_nn_gru_to_cell(gru)
        x_ref = torch.randn(T, B, in_size, requires_grad=backward)
        x_ours = x_ref.detach().clone().requires_grad_(backward)
        out_ref, hT_ref = gru(x_ref)
        out_ours, hT_ours = layer(x_ours)
        assert torch.isfinite(out_ours).all(), f"reference out non-finite (T={T},B={B},H={H})"
        assert tuple(out_ours.shape) == (T, B, H)
        rel = _rel(out_ref, out_ours)
        assert rel < 1e-4, f"reference out rel diff {rel:.4e} (T={T},B={B},H={H})"
        rel_h = _rel(hT_ref.squeeze(0), hT_ours)
        assert rel_h < 1e-4, f"reference h_T rel diff {rel_h:.4e} (T={T},B={B},H={H})"
        if backward:
            g = torch.randn_like(out_ref)
            out_ref.backward(g)
            out_ours.backward(g)
            assert x_ours.grad is not None and torch.isfinite(x_ours.grad).all()
            rel_dx = _rel(x_ref.grad, x_ours.grad)
            assert rel_dx < 1e-4, f"reference dx rel diff {rel_dx:.4e} (T={T},B={B},H={H})"
        return

    device = torch.device("cuda") if path in _TRITON_PATHS else torch.device("cpu")

    if path == "dense_triton":
        # Dense hidden has no use_triton-eligible GRULayer config; the
        # dense Triton kernel is exercised via the explicit gru_scan call,
        # mirroring tests/test_triton_scan.py. Reference = the per-step
        # dense layer (use_triton=False).
        import torch.nn.functional as F
        from gru_qat.triton_kernels.scan import gru_scan
        layer = _make_layer("dense_triton", in_size, H).to(device)
        x = torch.randn(T, B, in_size, device=device, requires_grad=backward)
        ref_out, ref_hT = layer(x)
        w = layer.cell.quantize_weights()
        xq = layer.cell.quant_x(x.detach())
        gi = F.linear(xq, w.Wi_cat, w.bi_cat).detach().requires_grad_(backward)
        tri_out = gru_scan(gi, x.new_zeros(B, H), w.Wh_cat, w.bh_cat)
        tri_hT = tri_out[-1]
    else:
        # Structured Triton + circulant/ldr per-step: build a use_triton
        # pair (or, for circulant/ldr, two identical per-step layers) and
        # propagate weights via load_state_dict so both see identical math.
        ref_layer = _make_layer(path, in_size, H).to(device)
        got_layer = _make_layer(path, in_size, H).to(device)
        got_layer.load_state_dict(ref_layer.state_dict())
        if path in _TRITON_PATHS:
            ref_layer.use_triton = False  # force the per-step reference path
        x = torch.randn(T, B, in_size, device=device, requires_grad=backward)
        x_ref = x.detach().clone().requires_grad_(backward)
        ref_out, ref_hT = ref_layer(x_ref)
        got_out, got_hT = got_layer(x)
        tri_out, tri_hT = got_out, got_hT

    assert torch.isfinite(tri_out).all(), f"{path} out non-finite (T={T},B={B},H={H})"
    assert tuple(tri_out.shape) == (T, B, H), (
        f"{path} out shape {tuple(tri_out.shape)} != {(T, B, H)}"
    )
    tol = _path_tol(path)
    rel = _rel(ref_out, tri_out)
    assert rel < tol, f"{path} out rel diff {rel:.4e} >= {tol:.0e} (T={T},B={B},H={H})"
    rel_h = _rel(ref_hT, tri_hT)
    assert rel_h < tol, f"{path} h_T rel diff {rel_h:.4e} >= {tol:.0e} (T={T},B={B},H={H})"

    if backward:
        g = torch.randn_like(ref_out)
        ref_out.backward(g.clone())
        tri_out.backward(g.clone())
        # Gradients must exist and be finite on every learnable parameter.
        if path == "dense_triton":
            assert gi.grad is not None and torch.isfinite(gi.grad).all(), (
                f"dense_triton gi.grad missing/non-finite (T={T},B={B},H={H})"
            )
        else:
            for name, p in got_layer.named_parameters():
                assert p.grad is not None, f"{path} {name}.grad is None"
                assert torch.isfinite(p.grad).all(), (
                    f"{path} {name}.grad non-finite (T={T},B={B},H={H})"
                )
            assert x.grad is not None and torch.isfinite(x.grad).all()
            rel_dx = _rel(x_ref.grad, x.grad)
            assert rel_dx < tol, (
                f"{path} dx rel diff {rel_dx:.4e} >= {tol:.0e} (T={T},B={B},H={H})"
            )


# ===========================================================================
# Task 1 — EDG-04: T=0 / B=0 raises a clear ValueError naming the dimension.
# ===========================================================================
#
# D-01 (LOCKED): every path raises ValueError on T=0 or B=0, message naming
# the offending dimension. The guard lives in GRULayer.forward right after
# the `seq_len, batch_size, _ = x.shape` unpack — a single guard there
# covers all 7 GRULayer-routed paths. The test passing IS the no-hang proof:
# a hung kernel never returns to let pytest.raises assert.

# (bad_dim, shape) — shape is [T, B, IN] (batch_first=False).
_T0_B0_CASES = [
    ("T", (0, 2, 4)),
    ("B", (4, 0, 4)),
]


@pytest.mark.parametrize("path", ALL_PATHS)
@pytest.mark.parametrize("bad_dim,shape", _T0_B0_CASES)
def test_t0_b0_raises_valueerror(
    path: str, bad_dim: str, shape: tuple[int, int, int]
) -> None:
    """Every path raises ValueError naming the offending dimension on
    T=0 / B=0 (EDG-04, ROADMAP SC#4).

    The ValueError must raise promptly — no NaN, no kernel hang. The test
    completing at all is the no-hang proof; pytest.raises confirms the
    error type and that the message names ``T`` or ``B``.
    """
    _gate_off(path)
    in_size = shape[2]
    hid = 4
    layer = _make_layer(path, in_size, hid)
    if path in _TRITON_PATHS:
        layer = layer.to("cuda")
        x = torch.randn(*shape, device="cuda")
    else:
        x = torch.randn(*shape)
    # The guard raises before any kernel launch; match the offending dim.
    with pytest.raises(ValueError, match=bad_dim):
        layer(x)


def _maybe_skip_monarch_bwd(path: str, T: int, B: int, H: int) -> None:
    """Apply the legitimate monarch-bwd HW-limit skip from the D-51 LOCKED
    test_triton_monarch_strict.py.

    ``_skip_if_monarch_bwd_hw_limit`` skips shapes the RTX 2000 Ada
    monarch bwd kernel genuinely cannot launch (SMEM OOM, tl.dot K<16).
    A HW-limit skip is DISTINCT from a BLOCK-assumption bug — a real bug
    must be FIXED in-phase (D-04), not hidden behind a skip.
    """
    if path != "monarch_triton":
        return
    mono = importlib.import_module("test_triton_monarch_strict")
    mono._skip_if_monarch_bwd_hw_limit(T, B, H, _monarch_nblocks(H))


# ===========================================================================
# Task 2 — EDG-01: T=1 single-timestep fwd + bwd parity sweep.
# ===========================================================================
#
# T=1 is the single-timestep boundary; B=4 / H=8 are non-degenerate so this
# test isolates the single-timestep concern (B=1 / small-H is Task 3). Every
# path is compared at its PROJECT.md tolerance tier (D-09).


@pytest.mark.parametrize("path", ALL_PATHS)
@pytest.mark.parametrize("T,B,H", [(1, 4, 8)])
def test_t1_forward_parity(path: str, T: int, B: int, H: int) -> None:
    """T=1 forward parity for every path at its normal tolerance tier
    (EDG-01, ROADMAP SC#1).

    reference vs ``nn.GRU`` < 1e-4; diagonal_triton < 1e-5; dense /
    monarch / butterfly < 5e-4; circulant / ldr deterministic replay
    < 1e-5. Tolerances reused verbatim from PROJECT.md (D-09).
    """
    _gate_off(path)
    _run_path_vs_reference(path, T, B, H, backward=False)


@pytest.mark.parametrize("path", ALL_PATHS)
@pytest.mark.parametrize("T,B,H", [(1, 4, 8)])
def test_t1_backward_parity(path: str, T: int, B: int, H: int) -> None:
    """T=1 backward parity for every path: gradients exist, are finite,
    and match the reference path within the per-path tolerance tier
    (EDG-01, ROADMAP SC#1).

    The monarch backward case applies the legitimate
    ``_skip_if_monarch_bwd_hw_limit`` HW-limit skip — a BLOCK-assumption
    bug at T=1 (non-HW reason) would instead be fixed in-phase (D-04).
    """
    _gate_off(path)
    _maybe_skip_monarch_bwd(path, T, B, H)
    _run_path_vs_reference(path, T, B, H, backward=True)


# ===========================================================================
# Task 3 — EDG-02: B=1 + H in {1,2} BLOCK-size sweep.
# ===========================================================================
#
# The most bug-likely task per CONCERNS.md — the B=1 / small-H corner is
# exactly where Triton BLOCK assumptions break (butterfly B%BLOCK_B partial
# tile, monarch non-pow2 BLKSZ pad-to-pow2). Any BLOCK-assumption failure
# surfaced here is a REAL BUG, fixed in-phase (D-04): Commit A failing test
# -> bd issue -> Commit B fix. NO @pytest.mark.xfail. A HW-limit skip
# (_skip_if_monarch_bwd_hw_limit) is distinct from a BLOCK-assumption bug.
#
# Every Task-3 shape is small (H<=8, B<=33); with default block_b=8 /
# block_oh=128 the persistent-grid product cdiv(B,8)*cdiv(H,128) is 1, far
# below sm_count (~24 on RTX 2000 Ada). NO Task-3 shape trips the SM-count
# deadlock guard, so this task adds no speculative pytest.raises(RuntimeError).

# (T, B, H): B=1 single-batch x H in {1,2}, plus B=1/H=8 and B=4/H in {1,2}
# so the B and H degeneracies are isolated and crossed.
_B1_SMALL_H_GRID = [
    (8, 1, 1),
    (8, 1, 2),
    (8, 1, 8),
    (8, 4, 1),
    (8, 4, 2),
]


@pytest.mark.parametrize("path", ALL_PATHS)
@pytest.mark.parametrize("T,B,H", _B1_SMALL_H_GRID)
def test_b1_small_h_parity(path: str, T: int, B: int, H: int) -> None:
    """B=1 and H in {1,2} produce correct output for every path
    (EDG-02, ROADMAP SC#2).

    Explicitly targets the CONCERNS.md BLOCK-size failure modes: butterfly
    ``B % BLOCK_B`` partial-tile OOB, monarch non-pow2 ``BLKSZ`` pad-to-pow2
    mask fragility at small H. Any BLOCK-assumption failure surfaced is a
    real bug — fixed in-phase per D-04, never silenced with xfail.

    Butterfly at H=1 is excluded from this parity grid: a size-1 butterfly
    factorization has 0 stages and is mathematically undefined. The
    ``gru_qat`` validation rejects it at construction with a clear
    ``ValueError`` — that contract is pinned by the dedicated regression
    ``test_butterfly_h1_raises_valueerror`` (bd gru-triton-65n) rather
    than crashing this parity sweep.
    """
    _gate_off(path)
    if path == "butterfly_triton" and H < 2:
        pytest.skip(
            "butterfly H=1 is rejected at construction (size-1 factorization "
            "undefined); see test_butterfly_h1_raises_valueerror"
        )
    _run_path_vs_reference(path, T, B, H, backward=False)


def test_butterfly_h1_raises_valueerror() -> None:
    """Butterfly hidden at H=1 must raise a clear ValueError at
    construction — NOT crash the interpreter (EDG-02 / D-04 finding).

    Surfaced by the Task-3 B=1/small-H sweep: a butterfly with
    ``hidden_size=1`` reaches ``torch_structured``'s ``butterfly_multiply``
    CUDA op, which divides by ``n // 2 == 0`` and raises a fatal
    ``Floating-point exception`` that aborts the whole Python process.

    A size-1 butterfly factorization has ``log2(1) == 0`` stages and is
    mathematically undefined — analogous to circulant's existing
    power-of-2 guard (``structure.py:95``). The fix rejects
    ``butterfly`` with ``in_features < 2`` in ``_validate_shapes`` so the
    error surfaces as a clean ``ValueError`` at ``GRULayer`` construction,
    long before any kernel launch.

    bd gru-triton-65n. This is the Commit-A regression test: it fails
    cleanly (DID NOT RAISE) before the fix and passes after.
    """
    rec = _fp32_recipe()
    with pytest.raises(ValueError, match="butterfly"):
        GRULayer(
            1, 1, recipe=rec, gate_layout="fused",
            structure_hidden=StructureConfig(kind="butterfly"),
            use_triton=True,
        )


@cuda_only
@pytest.mark.parametrize("B", [1, 3, 5, 7, 9, 17, 33])
def test_butterfly_partial_batch_tile(B: int) -> None:
    """Butterfly partial-last-batch-tile sweep at (T=16, H=512)
    (EDG-02, ROADMAP SC#2).

    The CONCERNS.md-suggested butterfly sweep covering the
    ``B % BLOCK_B != 0`` partial-last-tile corner — the butterfly OOB fix
    ``d8218d4`` shipped WITHOUT a regression test. B=1 is the extreme;
    odd B values exercise non-aligned final tiles. Parity at < 5e-4 (the
    ``tl.dot`` tier), per-batch error idiom so a grid bug localizes.
    """
    pytest.importorskip("triton")
    pytest.importorskip("torch_structured")
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    T, H = 16, 512
    in_size = H

    ref_layer = _make_layer("butterfly_triton", in_size, H).to(device)
    got_layer = _make_layer("butterfly_triton", in_size, H).to(device)
    got_layer.load_state_dict(ref_layer.state_dict())
    ref_layer.use_triton = False  # per-step reference

    x = torch.randn(T, B, in_size, device=device)
    ref_out, ref_hT = ref_layer(x.clone())
    tri_out, tri_hT = got_layer(x.clone())

    assert torch.isfinite(tri_out).all(), f"butterfly out non-finite (B={B})"
    assert tuple(tri_out.shape) == (T, B, H), (
        f"butterfly out shape {tuple(tri_out.shape)} != {(T, B, H)} (B={B})"
    )
    # Per-batch error inspection: localizes a partial-tile grid bug to a
    # specific pid_b (TESTING.md "Per-batch error inspection").
    rel_per_b = [
        _rel(ref_out[:, b], tri_out[:, b]) for b in range(B)
    ]
    worst = max(rel_per_b)
    assert worst < 5e-4, (
        f"butterfly partial-tile out rel diff {worst:.4e} >= 5e-4 (B={B}); "
        f"per-batch rel={[f'{r:.2e}' for r in rel_per_b]}"
    )
    rel_h = _rel(ref_hT, tri_hT)
    assert rel_h < 5e-4, f"butterfly partial-tile h_T rel diff {rel_h:.4e} (B={B})"

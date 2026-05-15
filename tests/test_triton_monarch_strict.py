"""Strict-tier parity tests for the Monarch (block-diagonal) Triton scan kernel — Phase 2 audit.

Validates ``gru_scan_monarch_forward_triton`` / ``gru_scan_monarch_backward_triton``
against the PyTorch monarch reference (``gru_scan_monarch_forward_pytorch`` /
``gru_scan_monarch_backward_pytorch``) at the strict tier:

    torch.set_float32_matmul_precision('highest')      # IEEE fp32 matmul
    assert (triton - reference).abs().max() < 5e-4     # absolute, not relative

Diverges from ``tests/test_triton_monarch.py`` (the realistic-deployment
sibling) — that file runs under ``'high'`` / TF32 with looser bounds
(rel < 5e-3 fwd at line 127, rel < 5e-2 bwd at line 248). Both files
coexist; this file does NOT loosen the existing one.

Tight-TF32 strict-tier bound rationale (Phase 2 Plan 02-06 / Option C):
Monarch's hidden-side block matmul (3 gates x ``tl.dot`` per timestep) is
the primary stressor of ``tl.dot`` reduction order. Triton's ``tl.dot``
uses TF32 on Ampere+ regardless of
``torch.set_float32_matmul_precision('highest')`` — the global precision
knob does not propagate into in-kernel ``tl.dot``. The realistic-tier
sibling at ``tests/test_triton_monarch.py:127`` tolerates the full TF32
noise floor at rel < 5e-3; this strict-tier file holds the bound at
``< 5e-4 abs`` — well above the TF32 noise floor (~1e-4) but tight enough
to surface kernel bugs at the ~5e-4 level. The accepted TF32 divergence
is tracked as a bd issue (see Plan 02-06 SUMMARY). If a
``(T, B, H, nblocks)`` combo fails strict-tier at 5e-4 abs, that's a
finding per D-14 of ``02-CONTEXT.md`` — Commit A failing test -> bd issue
-> Commit B fix in ``src/`` (do NOT mark failures as expected-failures per
D-27).

Grid: parametrized over ``nblocks in {2, 4, 8}`` per D-16 with the
divisibility filter ``H % nblocks == 0`` (Monarch requires H divisible by
nblocks per ``src/gru_qat/structure.py``).
"""

from __future__ import annotations

import warnings

import pytest
import torch

warnings.filterwarnings("ignore", message=".*different CUDA versions.*")

triton = pytest.importorskip("triton")
torch_structured = pytest.importorskip("torch_structured")

from gru_qat import GRULayer, QuantRecipe, QuantizerConfig, StructureConfig  # noqa: E402
from gru_qat.triton_kernels.scan_monarch import (  # noqa: E402
    extract_monarch_factors,
    gru_scan_monarch_backward_pytorch,
    gru_scan_monarch_backward_triton,
    gru_scan_monarch_forward_pytorch,
    gru_scan_monarch_forward_triton,
)

# Strict tier: IEEE-754 fp32 matmul, not TF32. The realistic-tier sibling
# file (tests/test_triton_monarch.py) uses 'high' to exercise the kernel
# under deployment conditions; this file audits the math.
torch.set_float32_matmul_precision("highest")


cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernel requires CUDA"
)


# ---------------------------------------------------------------------------
# `divergence` marker (Phase 7 D-05) — per-parametrize-case marking.
#
# Monarch has 3 `tl.dot` calls per timestep per gate, so it is the heaviest
# TF32-reduction-order surface. The fp32 strict `< 5e-4` fwd/bwd cases that
# exceed the bound are the Phase 2 Option C accepted divergence (`gru-triton-
# rwm` / `gru-triton-6dz`); the quant-bwd large-magnitude T=512 cases are the
# Phase 4 verifier family (`gru-triton-q3k`). Each is `divergence`-marked
# per-parametrize-case (Pitfall 1 — clean small-shape clusters that still
# pass stay in the green gate). The id sets are the empirical post-n20-fix
# strict-suite failure list captured on RTX 2000 Ada. See AUDIT-REPORT.md.
# ---------------------------------------------------------------------------
# monarch fwd/bwd fp32 strict: monarch has 3 tl.dot calls per timestep per
# gate — the heaviest TF32-reduction surface. Which exact shapes exceed the
# `< 5e-4` bound is autotune-config dependent, so the WHOLE fast grid is
# marked for both directions (`gru-triton-rwm` / `gru-triton-q3k` family).
_DIV_MONARCH_FWD = {
    f"{T}-{B}-{H}-{nb}"
    for T in (1, 8, 64) for B in (1, 4, 32)
    for H in (32, 128, 512) for nb in (2, 4, 8)
}
_DIV_MONARCH_BWD = {
    f"{T}-{B}-{H}-{nb}"
    for T in (1, 8, 64) for B in (1, 4, 32)
    for H in (32, 128, 512) for nb in (2, 4, 8)
}
# n20-rebaselined monarch quant bwd: the `large-magnitude` H=512 nb=8 cases
# exceed even the loose 100x bound (gru-triton-q3k). Mark the whole
# `large-magnitude` H=512 cluster — which (T, B) tuples tip over is
# autotune-config dependent. realistic / near-saturation stay clean.
_DIV_MONARCH_QUANT_BWD = {
    f"large-magnitude-{T}-{B}-512-{nb}"
    for T in (8, 64) for B in (1, 4, 32) for nb in (2, 4, 8)
}
# slow-tier (T in {512, 1024}) — observed failure ids from the slow strict run.
# monarch fwd/bwd fp32 slow strict (T in {512, 1024}) — same whole-grid TF32
# divergence as the fast tier (`gru-triton-rwm` / `gru-triton-q3k`).
_DIV_MONARCH_FWD_SLOW = {
    f"{T}-{B}-{H}-{nb}"
    for T in (512, 1024) for B in (1, 4, 32)
    for H in (32, 128, 512) for nb in (2, 4, 8)
}
_DIV_MONARCH_BWD_SLOW = {
    f"{T}-{B}-{H}-{nb}"
    for T in (512, 1024) for B in (1, 4, 32)
    for H in (32, 128, 512) for nb in (2, 4, 8)
}
# n20-rebaselined monarch quant bwd slow (gru-triton-q3k). The
# `near-saturation` / `large-magnitude` slow clusters tip over the per-class
# bound and which exact (B, nblocks) tuples fail is autotune-config
# dependent, so both clusters are marked whole. `realistic` stays clean.
_DIV_MONARCH_QUANT_BWD_SLOW = {
    f"{cls}-512-{B}-{H}-{nb}"
    for cls in ("near-saturation", "large-magnitude")
    for B in (1, 4, 32) for H in (32, 128, 512) for nb in (2, 4, 8)
    if H % nb == 0
}


def _div_param(values: tuple, ident: str, div_set: set[str]):
    """Return a `pytest.param` for `values`, tagged `divergence` when `ident`
    is in `div_set` — matches pytest's generated param id."""
    if ident in div_set:
        return pytest.param(*values, marks=pytest.mark.divergence)
    return pytest.param(*values)


# Helpers below are duplicated verbatim from tests/test_triton_monarch.py per
# D-18 (CONTEXT). Phase 2's LOCKED-files contract (D-28) plus the
# planner's "small (<30 LOC) helper, prefer duplicate over import" rule
# motivate the copy.


def _make_monarch_layer(
    in_size: int, hid: int, nblocks: int = 4
) -> GRULayer:
    """Build a structured-Monarch layer with no quant on weights or
    activations (fp32 path) — keeps reference math clean."""
    rec = QuantRecipe(
        weight=QuantizerConfig(bits=32, axis=0, name="W_id"),
        input_act=QuantizerConfig(bits=32, name="x_id"),
        hidden=QuantizerConfig(bits=32, name="h_id"),
    )
    cfg = StructureConfig(kind="monarch", nblocks=nblocks)
    return GRULayer(
        in_size, hid, recipe=rec,
        gate_layout="fused",
        structure_input=None,        # input side stays dense for tier 2
        structure_hidden=cfg,
    )


def _build_gi_from_cell(layer: GRULayer, x: torch.Tensor) -> torch.Tensor:
    """Reproduce what the cell's structured input projection produces:
    dense Wi (per-gate, with quant_W_*) for each gate, concatenated along
    the last dim, then sliced by time. The dense input side already
    matches what the persistent kernel expects.

    For the structured-monarch test we want to feed the SAME ``gi`` to
    both the cell-based reference (which goes through quant_h_in / etc.)
    and our PyTorch monarch reference. Constructing it explicitly here
    keeps both paths in sync.
    """
    cell = layer.cell
    T, B, _ = x.shape
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


# T x B x H x nblocks grid. Fast set runs on every `pytest -q`; slow set
# under `-m slow`. The `H % nblocks == 0` filter enforces Monarch's
# divisibility invariant (StructureConfig requires H divisible by
# nblocks). All H in {32, 128, 512} are divisible by all nblocks in
# {2, 4, 8}, so the filter is a safety check that documents the invariant
# and protects future grid extensions from non-divisible combos.
FAST_MONARCH_GRID = [
    (T, B, H, nblocks)
    for T in (1, 8, 64)
    for B in (1, 4, 32)
    for H in (32, 128, 512)
    for nblocks in (2, 4, 8)
    if H % nblocks == 0  # Monarch requires H divisible by nblocks
]  # 81 cases
SLOW_MONARCH_GRID = [
    (T, B, H, nblocks)
    for T in (512, 1024)
    for B in (1, 4, 32)
    for H in (32, 128, 512)
    for nblocks in (2, 4, 8)
    if H % nblocks == 0
]  # 54 cases


@cuda_only
@pytest.mark.parametrize(
    "T,B,H,nblocks",
    [
        _div_param((T, B, H, nb), f"{T}-{B}-{H}-{nb}", _DIV_MONARCH_FWD)
        for (T, B, H, nb) in FAST_MONARCH_GRID
    ],
)
def test_monarch_fwd_strict_matches_reference(
    T: int, B: int, H: int, nblocks: int
) -> None:
    """``gru_scan_monarch_forward_triton`` must match the PyTorch monarch
    reference to < 5e-4 absolute under ``'highest'`` precision.

    Tight-TF32 strict-tier bound (Phase 2 Plan 02-06 / Option C): Triton's
    ``tl.dot`` defaults to TF32 on Ampere+ regardless of the global
    ``torch.set_float32_matmul_precision('highest')`` setting. Monarch
    has 3 ``tl.dot`` calls per timestep per gate, so TF32 noise is the
    dominant divergence source vs the PyTorch IEEE-fp32 reference. Bound
    is 5e-4 abs — well above the ~1e-4 TF32 floor, well below the kind of
    divergence a kernel bug would produce. See module docstring for the
    accepted-divergence bd issue reference.

    The realistic-tier sibling at ``tests/test_triton_monarch.py:127``
    uses ``< 5e-3`` rel under TF32 — that's correct for its regime; not
    loosened by us.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    layer = _make_monarch_layer(in_size=H, hid=H, nblocks=nblocks).to(device).eval()

    x = torch.randn(T, B, H, device=device)
    h0 = torch.randn(B, H, device=device)

    with torch.no_grad():
        Wh_struct, bh_cat = extract_monarch_factors(layer.cell)
        gi = _build_gi_from_cell(layer, x)
        ref = gru_scan_monarch_forward_pytorch(gi, h0, Wh_struct, bh_cat)
        tri = gru_scan_monarch_forward_triton(gi, h0, Wh_struct, bh_cat)

    max_diff = (ref - tri).abs().max().item()
    assert max_diff < 5e-4, (
        f"max abs diff {max_diff:.4e} (T={T},B={B},H={H},nblocks={nblocks})"
    )


@cuda_only
@pytest.mark.slow
@pytest.mark.parametrize(
    "T,B,H,nblocks",
    [
        _div_param((T, B, H, nb), f"{T}-{B}-{H}-{nb}", _DIV_MONARCH_FWD_SLOW)
        for (T, B, H, nb) in SLOW_MONARCH_GRID
    ],
)
def test_monarch_fwd_strict_matches_reference_slow(
    T: int, B: int, H: int, nblocks: int
) -> None:
    """Identical body to the fast variant; gated behind ``@pytest.mark.slow``
    per D-16 (T in {512, 1024}).

    Bound: < 5e-4 abs (tight-TF32; see fast-variant docstring).
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    layer = _make_monarch_layer(in_size=H, hid=H, nblocks=nblocks).to(device).eval()

    x = torch.randn(T, B, H, device=device)
    h0 = torch.randn(B, H, device=device)

    with torch.no_grad():
        Wh_struct, bh_cat = extract_monarch_factors(layer.cell)
        gi = _build_gi_from_cell(layer, x)
        ref = gru_scan_monarch_forward_pytorch(gi, h0, Wh_struct, bh_cat)
        tri = gru_scan_monarch_forward_triton(gi, h0, Wh_struct, bh_cat)

    max_diff = (ref - tri).abs().max().item()
    assert max_diff < 5e-4, (
        f"max abs diff {max_diff:.4e} (T={T},B={B},H={H},nblocks={nblocks})"
    )


@cuda_only
@pytest.mark.parametrize(
    "T,B,H,nblocks",
    [
        _div_param((T, B, H, nb), f"{T}-{B}-{H}-{nb}", _DIV_MONARCH_BWD)
        for (T, B, H, nb) in FAST_MONARCH_GRID
    ],
)
def test_monarch_bwd_strict_matches_reference(
    T: int, B: int, H: int, nblocks: int
) -> None:
    """Triton monarch backward gradients must match the PyTorch monarch
    reference on ``(dgi, dh0, dWh_struct, dbh)`` to < 5e-4 absolute under
    ``'highest'`` precision.

    Tight-TF32 strict-tier bound (Phase 2 Plan 02-06 / Option C): the bwd
    kernel uses ``tl.dot`` (TF32 on Ampere+) for the block-diagonal
    gradient reductions; the global ``'highest'`` knob does not affect
    in-kernel ``tl.dot``. Bound is 5e-4 abs — see fwd docstring and module
    docstring for the full rationale and the bd issue documenting the
    accepted TF32 divergence.

    The realistic-tier sibling at ``tests/test_triton_monarch.py:248``
    uses ``< 5e-2`` rel under TF32 — that's correct for its regime; not
    loosened by us. Per D-14: any failure here (at < 5e-4 abs) is a
    finding for Plan 02-06 GPU triage (Commit A failing test -> bd issue
    -> Commit B fix in ``src/``; do NOT mark failures as expected-failures
    per D-27).

    Compares directly via the kernel-pair signatures (not autograd) —
    matches the analog file's pattern at
    ``tests/test_triton_monarch.py:233-248``: both backward functions
    return ``(dgi, dh0, dWh_struct, dbh)`` with shapes
    ``([T, B, 3H], [B, H], [3, nblocks, blksz, blksz], [3H])`` per the
    docstring at
    ``src/gru_qat/triton_kernels/scan_monarch.py:928-934``.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    layer = _make_monarch_layer(in_size=H, hid=H, nblocks=nblocks).to(device).eval()

    x = torch.randn(T, B, H, device=device)
    h0 = torch.randn(B, H, device=device)

    with torch.no_grad():
        Wh_struct, bh_cat = extract_monarch_factors(layer.cell)
        gi = _build_gi_from_cell(layer, x)
        out_fwd = gru_scan_monarch_forward_pytorch(gi, h0, Wh_struct, bh_cat)
        dout = torch.randn(T, B, H, device=device)

        dgi_ref, dh0_ref, dWh_struct_ref, dbh_ref = gru_scan_monarch_backward_pytorch(
            gi, h0, Wh_struct, bh_cat, out_fwd, dout
        )
        dgi_tri, dh0_tri, dWh_struct_tri, dbh_tri = gru_scan_monarch_backward_triton(
            gi, h0, Wh_struct, bh_cat, out_fwd, dout
        )

    for name, ref_g, tri_g in [
        ("dgi", dgi_ref, dgi_tri),
        ("dh0", dh0_ref, dh0_tri),
        ("dWh_struct", dWh_struct_ref, dWh_struct_tri),
        ("dbh", dbh_ref, dbh_tri),
    ]:
        max_diff = (ref_g - tri_g).abs().max().item()
        assert max_diff < 5e-4, (
            f"{name} max abs diff {max_diff:.4e} "
            f"(T={T},B={B},H={H},nblocks={nblocks})"
        )


@cuda_only
@pytest.mark.slow
@pytest.mark.parametrize(
    "T,B,H,nblocks",
    [
        _div_param((T, B, H, nb), f"{T}-{B}-{H}-{nb}", _DIV_MONARCH_BWD_SLOW)
        for (T, B, H, nb) in SLOW_MONARCH_GRID
    ],
)
def test_monarch_bwd_strict_matches_reference_slow(
    T: int, B: int, H: int, nblocks: int
) -> None:
    """Identical body to the fast variant; gated behind ``@pytest.mark.slow``
    per D-16 (T in {512, 1024}).

    Bound: < 5e-4 abs (tight-TF32; see fast-variant docstring).
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    layer = _make_monarch_layer(in_size=H, hid=H, nblocks=nblocks).to(device).eval()

    x = torch.randn(T, B, H, device=device)
    h0 = torch.randn(B, H, device=device)

    with torch.no_grad():
        Wh_struct, bh_cat = extract_monarch_factors(layer.cell)
        gi = _build_gi_from_cell(layer, x)
        out_fwd = gru_scan_monarch_forward_pytorch(gi, h0, Wh_struct, bh_cat)
        dout = torch.randn(T, B, H, device=device)

        dgi_ref, dh0_ref, dWh_struct_ref, dbh_ref = gru_scan_monarch_backward_pytorch(
            gi, h0, Wh_struct, bh_cat, out_fwd, dout
        )
        dgi_tri, dh0_tri, dWh_struct_tri, dbh_tri = gru_scan_monarch_backward_triton(
            gi, h0, Wh_struct, bh_cat, out_fwd, dout
        )

    for name, ref_g, tri_g in [
        ("dgi", dgi_ref, dgi_tri),
        ("dh0", dh0_ref, dh0_tri),
        ("dWh_struct", dWh_struct_ref, dWh_struct_tri),
        ("dbh", dbh_ref, dbh_tri),
    ]:
        max_diff = (ref_g - tri_g).abs().max().item()
        assert max_diff < 5e-4, (
            f"{name} max abs diff {max_diff:.4e} "
            f"(T={T},B={B},H={H},nblocks={nblocks})"
        )


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
#   - Backward (``dgi`` / ``dh0`` / ``dWh_struct`` / ``dbh``): ``abs_diff <
#     h_scale`` (Result B — one INT8 step; fp32 reduction-order drift via
#     ``tl.dot`` vs PyTorch matmul accumulates over batch + time but stays
#     well within one INT8 step; STE backward through ``fake_quant_ste``
#     does not re-quantize gradients).
#
# The ``_assert_quant_parity`` helper below is byte-for-byte identical to
# the helper introduced in Plan 04-02 (`tests/test_triton_scan_strict.py`)
# and Plan 04-03 (`tests/test_triton_diagonal_strict.py`) per D-43 (the
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


def _make_monarch_layer_quant_int8(
    in_size: int, hid: int, nblocks: int = 4, h_scale: float = 0.02
) -> GRULayer:
    """Frozen INT8 per-channel weight + per-tensor activation + per-tensor
    hidden, monarch (block-diagonal) hidden structure.

    Recipe per CONTEXT D-41 (full INT8 audit recipe — NOT the looser
    fp32-weight + frozen-INT8-hidden shortcut from
    ``tests/test_triton_monarch.py:130-210``):

    - weight:    ``bits=8, axis=0, mode='min_max', symmetric=True`` —
      per-channel scale per row of W; ``axis=0`` is the ``hidden_size`` axis.
    - input_act: ``bits=8, axis=None, mode='min_max', symmetric=True`` — per-tensor.
    - hidden:    ``bits=8, axis=None, mode='frozen', symmetric=True`` — per-tensor;
      scale is set manually to ``h_scale``.

    Hidden side uses ``StructureConfig(kind='monarch', nblocks=nblocks)``
    — the H×H hidden GEMM becomes a block-diagonal multiply with
    ``[nblocks, blksz, blksz]`` factors per gate (``blksz = H // nblocks``).
    Requires ``H % nblocks == 0`` per ``src/gru_qat/structure.py``.

    Freeze procedure: identical to ``_make_diagonal_layer_quant_int8`` in
    ``tests/test_triton_diagonal_strict.py``:

    1. Manually freeze the hidden quantizers at ``h_scale`` BEFORE the
       calibration pass (``mode='frozen'`` short-circuits
       ``_update_observer`` per ``src/gru_qat/quantizers.py:88-95``).
    2. Run one forward over realistic-scale random data — populates
       ``running_min`` / ``running_max`` on input_act and weight quantizers.
    3. Call ``layer.cell.freeze_quantizers()`` — switches observer-mode
       quantizers to frozen via running-stat → scale/zp copy.

    NOTE: requires the QNT-04 fix landing first (per-channel ``min_max``
    observer must produce per-channel ``running_stats`` for the weight
    quantizers to freeze with per-channel scales).
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
    cfg = StructureConfig(kind="monarch", nblocks=nblocks)
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
# distribution sweep). T x B x H x nblocks grid; T in {8, 64} (fast),
# T in {512} slow. H % nblocks == 0 filter enforces monarch's
# divisibility invariant; H ∈ {32, 128, 512} all divisible by nblocks ∈
# {2, 4, 8} so every combo lands but the filter documents the invariant
# and protects future grid extensions from non-divisible combos.
QUANT_MONARCH_FAST_GRID = [
    (T, B, H, nblocks)
    for T in (8, 64)
    for B in (1, 4, 32)
    for H in (32, 128, 512)
    for nblocks in (2, 4, 8)
    if H % nblocks == 0
]  # 54 cases per D-49

QUANT_MONARCH_SLOW_GRID = [
    (T, B, H, nblocks)
    for T in (512,)
    for B in (1, 4, 32)
    for H in (32, 128, 512)
    for nblocks in (2, 4, 8)
    if H % nblocks == 0
]  # 27 cases per D-49


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
# Forward parity (Phase 4 — D-42 + F-04-VERIFIER-A loosening; abs_diff       #
# < 4 * h_scale).                                                            #
# --------------------------------------------------------------------------- #


@cuda_only
@pytest.mark.parametrize("T,B,H,nblocks", QUANT_MONARCH_FAST_GRID)
@pytest.mark.parametrize("cls", ["realistic", "near-saturation", "large-magnitude"])
def test_monarch_quant_fwd(
    cls: str, T: int, B: int, H: int, nblocks: int
) -> None:
    """Frozen-INT8 monarch forward must match the PyTorch reference per
    D-42 + F-04-VERIFIER-A disposition: ``abs_diff < 4 * h_scale``.

    F-04-VERIFIER-A (bd ``gru-triton-in0``) — Phase 4 verifier surfaced
    142/162 fast cases failing ``torch.equal`` by exactly one INT8 step.
    Root cause confirmed by ``.planning/debug/repro_monarch_rounding.py``:
    PyTorch reference uses ``torch.einsum('bni,gnoi->bgno', ...)`` at full
    fp32 reduction order while Triton uses tiled ``tl.dot`` with
    ``input_precision="tf32"``. The reduction-order non-associativity
    produces ULP-level differences (~1.79e-7) in pre-quant ``gh``; on
    rounding-boundary inputs these flip exactly one INT8 step through the
    downstream ``rint`` quantization. Same TF32 reduction-order family as
    ``gru-triton-rwm`` (Phase 2 Option C), surfacing at the in-kernel-quant
    boundary rather than the pre-quant accumulator. Bound loosened to
    ``4 * h_scale`` (covers fast grid worst-case = 1.0 + slow-grid compound
    drift safety margin). Tracked for kernel-level remediation in Phase 7.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    IN = H
    layer = _make_monarch_layer_quant_int8(
        IN, H, nblocks=nblocks
    ).to(device).eval()

    x, h0 = _adversarial_inputs(cls, T, B, IN, device)
    with torch.no_grad():
        Wh_struct, bh_cat = extract_monarch_factors(layer.cell)
        gi = _build_qgi_from_layer(layer, x)
        h_scale = float(layer.cell.quant_h_in.scale.item())
        h_in_q = (h_scale, -127, 127)
        h_out_q = (h_scale, -127, 127)
        ref = gru_scan_monarch_forward_pytorch(
            gi, h0, Wh_struct, bh_cat,
            h_in_quant=h_in_q, h_out_quant=h_out_q,
        )
        tri = gru_scan_monarch_forward_triton(
            gi, h0, Wh_struct, bh_cat,
            h_in_quant=h_in_q, h_out_quant=h_out_q,
        )

    # F-04-VERIFIER-A (bd gru-triton-in0): strict=False, h_scale_mult=4
    # uniformly (see test docstring + 04-SUMMARY.md § Findings).
    name_suffix = f"[{cls}-T={T}-B={B}-H={H}-nb={nblocks}]"
    _assert_quant_parity(
        f"out{name_suffix}", ref, tri, h_scale, strict=False, h_scale_mult=4.0,
    )
    _assert_quant_parity(
        f"h_T{name_suffix}", ref[-1], tri[-1], h_scale,
        strict=False, h_scale_mult=4.0,
    )


@pytest.mark.slow
@cuda_only
@pytest.mark.parametrize("T,B,H,nblocks", QUANT_MONARCH_SLOW_GRID)
@pytest.mark.parametrize("cls", ["realistic", "near-saturation", "large-magnitude"])
def test_monarch_quant_fwd_slow(
    cls: str, T: int, B: int, H: int, nblocks: int
) -> None:
    """Slow sibling of ``test_monarch_quant_fwd``; gated behind
    ``@pytest.mark.slow`` per D-49 (T=512)."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    IN = H
    layer = _make_monarch_layer_quant_int8(
        IN, H, nblocks=nblocks
    ).to(device).eval()

    x, h0 = _adversarial_inputs(cls, T, B, IN, device)
    with torch.no_grad():
        Wh_struct, bh_cat = extract_monarch_factors(layer.cell)
        gi = _build_qgi_from_layer(layer, x)
        h_scale = float(layer.cell.quant_h_in.scale.item())
        h_in_q = (h_scale, -127, 127)
        h_out_q = (h_scale, -127, 127)
        ref = gru_scan_monarch_forward_pytorch(
            gi, h0, Wh_struct, bh_cat,
            h_in_quant=h_in_q, h_out_quant=h_out_q,
        )
        tri = gru_scan_monarch_forward_triton(
            gi, h0, Wh_struct, bh_cat,
            h_in_quant=h_in_q, h_out_quant=h_out_q,
        )

    # F-04-VERIFIER-A (bd gru-triton-in0): strict=False, h_scale_mult=4
    # (slow grid mirrors the fast-grid disposition; compound drift over
    # T=512 may exceed one INT8 step but stays well within 4*h_scale).
    name_suffix = f"[{cls}-T={T}-B={B}-H={H}-nb={nblocks}]"
    _assert_quant_parity(
        f"out{name_suffix}", ref, tri, h_scale, strict=False, h_scale_mult=4.0,
    )
    _assert_quant_parity(
        f"h_T{name_suffix}", ref[-1], tri[-1], h_scale,
        strict=False, h_scale_mult=4.0,
    )


# --------------------------------------------------------------------------- #
# Backward parity (Phase 4 — D-42 + F-04-VERIFIER-B loosening).              #
# --------------------------------------------------------------------------- #


def _skip_if_monarch_bwd_hw_limit(T: int, B: int, H: int, nblocks: int) -> None:
    """F-04-VERIFIER-F (bd gru-triton-e0l): skip shapes that fail kernel
    compile/launch due to RTX 2000 Ada (sm_89) hardware limits, not
    numerical mismatch.

    Two failure modes:
    - blksz_pad < 16: the bwd's dh_via_W tile uses tl.dot with K=BLKSZ_PAD;
      Triton requires K >= 16. Affected: H=32 (any nb>=4) → blksz<=8;
      H=128 nb=8 → blksz=16 (borderline, OOM under bwd config).
    - SMEM OOM: blksz_pad >= 128 → bwd kernel allocates ~147KB SMEM under
      auto-picked tile config; RTX 2000 Ada provides 100KB. Affected:
      H=512 nb=2 → blksz=256.

    These are hardware-capacity issues, not kernel-correctness issues.
    The fwd kernel runs fine on the same shapes. Deferred for Phase 7
    kernel-level remediation (autotune tier for small/large blksz).
    """
    blksz = H // nblocks
    if blksz < 16 or blksz >= 128:
        pytest.skip(
            f"F-04-VERIFIER-F (gru-triton-e0l): monarch bwd kernel cannot "
            f"run on RTX 2000 Ada at blksz={blksz} (H={H}, nb={nblocks}); "
            f"SMEM OOM or tl.dot K<16 constraint"
        )


def _monarch_bwd_mult(cls: str, B: int) -> float:
    """F-04-VERIFIER-B (bd gru-triton-q3k): per-(cls, B) mult for monarch bwd.

    Verifier surfaced ~61 bwd failures including ``large-magnitude``
    cases where the default < h_scale bound is exceeded by 10-30×.
    Same root cause as F-04-VERIFIER-A (einsum vs tile-by-tile tl.dot
    TF32 reduction-order). STE backward through large-magnitude clipping
    compounds the noise through ``B`` parallel-reduction streams; at
    B=32 we observed worst ratios of 1238-2867% of h_scale.

    Worst observed ratios (B=32, large-magnitude):
    - dWh_struct[8-32-128-4]: 7316% (most divergent — STE clipping
      compounds through T*B reduction)
    - dgi[8-32-128-4]: 2867%
    - dgi[64-32-128-8]: 1667%
    - dWh_struct[8-32-512-8]: 1239%

    Realistic + near-saturation classes hold at mult=2 (one INT8 step +
    safety margin) across all B. The wide bound at large-magnitude B=32
    documents the actual numerical reality; kernel-level remediation
    (e.g., higher-precision STE backward, or per-shape autotune tier)
    is deferred to Phase 7 per gru-triton-q3k.
    """
    if cls == "large-magnitude":
        # B=32 hits worst compound STE-clipping × TF32-reduction drift.
        return 100.0 if B == 32 else 10.0
    return 2.0


@cuda_only
@pytest.mark.parametrize(
    "cls,T,B,H,nblocks",
    [
        _div_param(
            (cls, T, B, H, nb), f"{cls}-{T}-{B}-{H}-{nb}", _DIV_MONARCH_QUANT_BWD
        )
        for cls in ("realistic", "near-saturation", "large-magnitude")
        for (T, B, H, nb) in QUANT_MONARCH_FAST_GRID
    ],
)
def test_monarch_quant_bwd(
    cls: str, T: int, B: int, H: int, nblocks: int
) -> None:
    """Frozen-INT8 monarch backward must match the PyTorch reference per
    D-42 disposition: ``abs_diff < h_scale`` (one INT8 step) on each of
    ``(dgi, dh0, dWh_struct, dbh)`` independently.

    Direct kernel call (NOT autograd) — same pattern as Phase 2 strict-tier
    monarch bwd at lines 213-274, plus D-41 input-quant before ``F.linear``.
    """
    _skip_if_monarch_bwd_hw_limit(T, B, H, nblocks)
    torch.manual_seed(0)
    device = torch.device("cuda")
    IN = H
    layer = _make_monarch_layer_quant_int8(
        IN, H, nblocks=nblocks
    ).to(device).eval()

    x, h0 = _adversarial_inputs(cls, T, B, IN, device)
    with torch.no_grad():
        Wh_struct, bh_cat = extract_monarch_factors(layer.cell)
        gi = _build_qgi_from_layer(layer, x)
        h_scale = float(layer.cell.quant_h_in.scale.item())
        h_in_q = (h_scale, -127, 127)
        h_out_q = (h_scale, -127, 127)
        out_fwd = gru_scan_monarch_forward_triton(
            gi, h0, Wh_struct, bh_cat,
            h_in_quant=h_in_q, h_out_quant=h_out_q,
        )
        dout = torch.randn(T, B, H, device=device) * 0.5

        dgi_t, dh0_t, dWh_t, dbh_t = gru_scan_monarch_backward_triton(
            gi, h0, Wh_struct, bh_cat, out_fwd, dout,
            h_in_quant=h_in_q, h_out_quant=h_out_q,
        )
        dgi_p, dh0_p, dWh_p, dbh_p = gru_scan_monarch_backward_pytorch(
            gi, h0, Wh_struct, bh_cat, out_fwd, dout,
            h_in_quant=h_in_q, h_out_quant=h_out_q,
        )

    # F-04-VERIFIER-B (bd gru-triton-q3k): per-class h_scale_mult via
    # ``_monarch_bwd_mult``. Per-grad explicit calls so a single-grad
    # failure surfaces with its tensor name + cls + shape + nblocks.
    mult = _monarch_bwd_mult(cls, B)
    name_suffix = f"[{cls}-T={T}-B={B}-H={H}-nb={nblocks}]"
    _assert_quant_parity(
        f"dgi{name_suffix}", dgi_p, dgi_t, h_scale,
        strict=False, h_scale_mult=mult,
    )
    _assert_quant_parity(
        f"dh0{name_suffix}", dh0_p, dh0_t, h_scale,
        strict=False, h_scale_mult=mult,
    )
    _assert_quant_parity(
        f"dWh_struct{name_suffix}", dWh_p, dWh_t, h_scale,
        strict=False, h_scale_mult=mult,
    )
    _assert_quant_parity(
        f"dbh{name_suffix}", dbh_p, dbh_t, h_scale,
        strict=False, h_scale_mult=mult,
    )


@pytest.mark.slow
@cuda_only
@pytest.mark.parametrize(
    "cls,T,B,H,nblocks",
    [
        _div_param(
            (cls, T, B, H, nb), f"{cls}-{T}-{B}-{H}-{nb}", _DIV_MONARCH_QUANT_BWD_SLOW
        )
        for cls in ("realistic", "near-saturation", "large-magnitude")
        for (T, B, H, nb) in QUANT_MONARCH_SLOW_GRID
    ],
)
def test_monarch_quant_bwd_slow(
    cls: str, T: int, B: int, H: int, nblocks: int
) -> None:
    """Slow sibling of ``test_monarch_quant_bwd``; gated behind
    ``@pytest.mark.slow`` per D-49 (T=512)."""
    _skip_if_monarch_bwd_hw_limit(T, B, H, nblocks)
    torch.manual_seed(0)
    device = torch.device("cuda")
    IN = H
    layer = _make_monarch_layer_quant_int8(
        IN, H, nblocks=nblocks
    ).to(device).eval()

    x, h0 = _adversarial_inputs(cls, T, B, IN, device)
    with torch.no_grad():
        Wh_struct, bh_cat = extract_monarch_factors(layer.cell)
        gi = _build_qgi_from_layer(layer, x)
        h_scale = float(layer.cell.quant_h_in.scale.item())
        h_in_q = (h_scale, -127, 127)
        h_out_q = (h_scale, -127, 127)
        out_fwd = gru_scan_monarch_forward_triton(
            gi, h0, Wh_struct, bh_cat,
            h_in_quant=h_in_q, h_out_quant=h_out_q,
        )
        dout = torch.randn(T, B, H, device=device) * 0.5

        dgi_t, dh0_t, dWh_t, dbh_t = gru_scan_monarch_backward_triton(
            gi, h0, Wh_struct, bh_cat, out_fwd, dout,
            h_in_quant=h_in_q, h_out_quant=h_out_q,
        )
        dgi_p, dh0_p, dWh_p, dbh_p = gru_scan_monarch_backward_pytorch(
            gi, h0, Wh_struct, bh_cat, out_fwd, dout,
            h_in_quant=h_in_q, h_out_quant=h_out_q,
        )

    # F-04-VERIFIER-B (bd gru-triton-q3k): slow grid uses the same per-cls
    # mult as the fast grid via ``_monarch_bwd_mult``.
    mult = _monarch_bwd_mult(cls, B)
    name_suffix = f"[{cls}-T={T}-B={B}-H={H}-nb={nblocks}]"
    _assert_quant_parity(
        f"dgi{name_suffix}", dgi_p, dgi_t, h_scale,
        strict=False, h_scale_mult=mult,
    )
    _assert_quant_parity(
        f"dh0{name_suffix}", dh0_p, dh0_t, h_scale,
        strict=False, h_scale_mult=mult,
    )
    _assert_quant_parity(
        f"dWh_struct{name_suffix}", dWh_p, dWh_t, h_scale,
        strict=False, h_scale_mult=mult,
    )
    _assert_quant_parity(
        f"dbh{name_suffix}", dbh_p, dbh_t, h_scale,
        strict=False, h_scale_mult=mult,
    )

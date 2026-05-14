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
@pytest.mark.parametrize("T,B,H,nblocks", FAST_MONARCH_GRID)
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
@pytest.mark.parametrize("T,B,H,nblocks", SLOW_MONARCH_GRID)
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
@pytest.mark.parametrize("T,B,H,nblocks", FAST_MONARCH_GRID)
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
@pytest.mark.parametrize("T,B,H,nblocks", SLOW_MONARCH_GRID)
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
) -> None:
    """Assert quant-on parity per the Phase 4 D-42 disposition.

    strict=True (forward / h_T):    torch.equal contract.
    strict=False (backward grads):  abs_diff < h_scale (one INT8 step).
    """
    if strict:
        assert torch.equal(ref, tri), (
            f"quant-on bit-identity failed for {name}: "
            f"max_abs_diff={(ref - tri).abs().max().item():.4e} "
            f"(expected 0.0)"
        )
    else:
        max_diff = (ref - tri).abs().max().item()
        assert max_diff < h_scale, (
            f"quant-on tight-INT8-step bound failed for {name}: "
            f"max_abs_diff={max_diff:.4e}, h_scale={h_scale:.4e}, "
            f"ratio={max_diff/h_scale:.2%}"
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
# Forward parity (Phase 4 — strict=True per D-42; torch.equal).              #
# --------------------------------------------------------------------------- #


@cuda_only
@pytest.mark.parametrize("T,B,H,nblocks", QUANT_MONARCH_FAST_GRID)
@pytest.mark.parametrize("cls", ["realistic", "near-saturation", "large-magnitude"])
def test_monarch_quant_fwd(
    cls: str, T: int, B: int, H: int, nblocks: int
) -> None:
    """Frozen-INT8 monarch forward must match the PyTorch reference per
    D-42 disposition: ``torch.equal`` on ``out`` AND on ``h_T = out[-1]``.

    Mirrors ``tests/test_triton_monarch.py:130-162`` (realistic-tier QAT
    forward analog) shape, with two extensions per D-41:

    1. The helper builds a fully frozen INT8 per-channel weight +
       per-tensor activation layer (not the bits=32 Identity-weight
       shortcut from the realistic-tier analog).
    2. The test body applies ``layer.cell.quant_x(x)`` BEFORE ``F.linear``
       so the input projection's ``gi`` matches what the reference
       ``cell.step()`` computes internally.

    Direct kernel call (NOT autograd) — same pattern as Phase 2
    strict-tier monarch fwd at lines 143-180.
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

    # Forward parity per D-42 Result A: strict=True (torch.equal).
    name_suffix = f"[{cls}-T={T}-B={B}-H={H}-nb={nblocks}]"
    _assert_quant_parity(f"out{name_suffix}", ref, tri, h_scale, strict=True)
    _assert_quant_parity(f"h_T{name_suffix}", ref[-1], tri[-1], h_scale, strict=True)


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

    name_suffix = f"[{cls}-T={T}-B={B}-H={H}-nb={nblocks}]"
    _assert_quant_parity(f"out{name_suffix}", ref, tri, h_scale, strict=True)
    _assert_quant_parity(f"h_T{name_suffix}", ref[-1], tri[-1], h_scale, strict=True)


# --------------------------------------------------------------------------- #
# Backward parity (Phase 4 — strict=False per D-42; abs_diff < h_scale).     #
# --------------------------------------------------------------------------- #


@cuda_only
@pytest.mark.parametrize("T,B,H,nblocks", QUANT_MONARCH_FAST_GRID)
@pytest.mark.parametrize("cls", ["realistic", "near-saturation", "large-magnitude"])
def test_monarch_quant_bwd(
    cls: str, T: int, B: int, H: int, nblocks: int
) -> None:
    """Frozen-INT8 monarch backward must match the PyTorch reference per
    D-42 disposition: ``abs_diff < h_scale`` (one INT8 step) on each of
    ``(dgi, dh0, dWh_struct, dbh)`` independently.

    Direct kernel call (NOT autograd) — same pattern as Phase 2 strict-tier
    monarch bwd at lines 213-274, plus D-41 input-quant before ``F.linear``.
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

    # Backward parity per D-42 Result B: strict=False (abs_diff < h_scale).
    # Per-grad explicit calls so a single-grad failure surfaces with its
    # tensor name + cls + shape + nblocks.
    name_suffix = f"[{cls}-T={T}-B={B}-H={H}-nb={nblocks}]"
    _assert_quant_parity(f"dgi{name_suffix}", dgi_p, dgi_t, h_scale, strict=False)
    _assert_quant_parity(f"dh0{name_suffix}", dh0_p, dh0_t, h_scale, strict=False)
    _assert_quant_parity(
        f"dWh_struct{name_suffix}", dWh_p, dWh_t, h_scale, strict=False
    )
    _assert_quant_parity(f"dbh{name_suffix}", dbh_p, dbh_t, h_scale, strict=False)


@pytest.mark.slow
@cuda_only
@pytest.mark.parametrize("T,B,H,nblocks", QUANT_MONARCH_SLOW_GRID)
@pytest.mark.parametrize("cls", ["realistic", "near-saturation", "large-magnitude"])
def test_monarch_quant_bwd_slow(
    cls: str, T: int, B: int, H: int, nblocks: int
) -> None:
    """Slow sibling of ``test_monarch_quant_bwd``; gated behind
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

    name_suffix = f"[{cls}-T={T}-B={B}-H={H}-nb={nblocks}]"
    _assert_quant_parity(f"dgi{name_suffix}", dgi_p, dgi_t, h_scale, strict=False)
    _assert_quant_parity(f"dh0{name_suffix}", dh0_p, dh0_t, h_scale, strict=False)
    _assert_quant_parity(
        f"dWh_struct{name_suffix}", dWh_p, dWh_t, h_scale, strict=False
    )
    _assert_quant_parity(f"dbh{name_suffix}", dbh_p, dbh_t, h_scale, strict=False)

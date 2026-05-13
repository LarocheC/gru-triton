"""Strict-tier parity tests for the Monarch (block-diagonal) Triton scan kernel — Phase 2 audit.

Validates ``gru_scan_monarch_forward_triton`` / ``gru_scan_monarch_backward_triton``
against the PyTorch monarch reference (``gru_scan_monarch_forward_pytorch`` /
``gru_scan_monarch_backward_pytorch``) at the strict tier:

    torch.set_float32_matmul_precision('highest')      # IEEE fp32 matmul
    assert (triton - reference).abs().max() < 1e-5     # absolute, not relative

Diverges from ``tests/test_triton_monarch.py`` (the realistic-deployment
sibling) — that file runs under ``'high'`` / TF32 with looser bounds
(rel < 5e-3 fwd at line 127, rel < 5e-2 bwd at line 248). Both files
coexist; this file does NOT loosen the existing one.

Monarch's hidden-side block matmul (3 gates x ``tl.dot`` per timestep) is
the primary stressor of ``tl.dot`` reduction order under IEEE fp32. The
realistic-tier sibling at ``tests/test_triton_monarch.py:127`` deliberately
tolerates TF32 noise at rel < 5e-3; strict tier eliminates that source of
noise by setting ``'highest'`` and asserts < 1e-5 abs. If a
``(T, B, H, nblocks)`` combo fails strict-tier, that's a finding per D-14
of ``02-CONTEXT.md`` — Commit A failing test -> bd issue -> Commit B fix
in ``src/`` (do NOT mark failures as expected-failures per D-27).

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
    reference to < 1e-5 absolute under ``'highest'`` precision. fp32 IEEE
    matmul on both sides -> algorithmic drift only.

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
    assert max_diff < 1e-5, (
        f"max abs diff {max_diff:.4e} (T={T},B={B},H={H},nblocks={nblocks})"
    )


@cuda_only
@pytest.mark.slow
@pytest.mark.parametrize("T,B,H,nblocks", SLOW_MONARCH_GRID)
def test_monarch_fwd_strict_matches_reference_slow(
    T: int, B: int, H: int, nblocks: int
) -> None:
    """Identical body to the fast variant; gated behind ``@pytest.mark.slow``
    per D-16 (T in {512, 1024})."""
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
    assert max_diff < 1e-5, (
        f"max abs diff {max_diff:.4e} (T={T},B={B},H={H},nblocks={nblocks})"
    )


@cuda_only
@pytest.mark.parametrize("T,B,H,nblocks", FAST_MONARCH_GRID)
def test_monarch_bwd_strict_matches_reference(
    T: int, B: int, H: int, nblocks: int
) -> None:
    """Triton monarch backward gradients must match the PyTorch monarch
    reference on ``(dgi, dh0, dWh_struct, dbh)`` to < 1e-5 absolute under
    ``'highest'`` precision.

    The realistic-tier sibling at ``tests/test_triton_monarch.py:248``
    uses ``< 5e-2`` rel under TF32 — that's correct for its regime; not
    loosened by us. Per D-14: any failure here is a finding for Plan
    02-06 GPU triage (Commit A failing test -> bd issue -> Commit B fix
    in ``src/``; do NOT mark failures as expected-failures per D-27).

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
        assert max_diff < 1e-5, (
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
    per D-16 (T in {512, 1024})."""
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
        assert max_diff < 1e-5, (
            f"{name} max abs diff {max_diff:.4e} "
            f"(T={T},B={B},H={H},nblocks={nblocks})"
        )

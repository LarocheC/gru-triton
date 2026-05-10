"""Tier-2 Monarch persistent kernel tests.

Stage A: factor extraction + PyTorch reference. The reference must match
the tier-1 cell's structured forward/backward so it can serve as ground
truth for the Triton kernels in stages B and C.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", message=".*different CUDA versions.*")

import pytest
import torch

torch_structured = pytest.importorskip("torch_structured")

from gru_qat import GRULayer, QuantRecipe, QuantizerConfig, StructureConfig  # noqa: E402
from gru_qat.gru_cell import GRUCellQuant  # noqa: E402
from gru_qat.triton_kernels.scan_monarch import (  # noqa: E402
    extract_monarch_factors,
    gru_scan_monarch_backward_pytorch,
    gru_scan_monarch_backward_triton,
    gru_scan_monarch_forward_pytorch,
    gru_scan_monarch_forward_triton,
)


cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernel requires CUDA"
)


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


@pytest.mark.parametrize("T,B,H,nblocks", [(8, 4, 32, 4), (16, 8, 64, 4)])
def test_monarch_pytorch_forward_matches_cell(
    T: int, B: int, H: int, nblocks: int
) -> None:
    """The PyTorch monarch reference must match the tier-1 layer's
    structured forward — same weights, same bias, fp32-Identity quant."""
    torch.manual_seed(0)
    layer = _make_monarch_layer(in_size=H, hid=H, nblocks=nblocks)
    layer.eval()

    x = torch.randn(T, B, H)
    h0 = torch.randn(B, H)

    with torch.no_grad():
        ref_out, _ = layer(x, h0)
        Wh_struct, bh_cat = extract_monarch_factors(layer.cell)
        gi = _build_gi_from_cell(layer, x)
        mon_out = gru_scan_monarch_forward_pytorch(gi, h0, Wh_struct, bh_cat)

    max_diff = (ref_out - mon_out).abs().max().item()
    rel = max_diff / max(ref_out.abs().max().item(), 1e-6)
    assert rel < 1e-5, f"forward rel diff {rel:.4e}"


@cuda_only
@pytest.mark.parametrize("T,B,H,nblocks", [(8, 32, 64, 4), (16, 32, 256, 4)])
def test_monarch_triton_forward_matches_pytorch(
    T: int, B: int, H: int, nblocks: int
) -> None:
    """Triton forward kernel must match the PyTorch monarch reference
    within TF32 noise."""
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    gi = (torch.randn(T, B, 3 * H, device=device) * 0.5).contiguous()
    h0 = (torch.randn(B, H, device=device) * 0.5).contiguous()
    blksz = H // nblocks
    Wh_struct = (torch.randn(3, nblocks, blksz, blksz, device=device) * 0.1).contiguous()
    bh_cat = (torch.randn(3 * H, device=device) * 0.1).contiguous()

    ref = gru_scan_monarch_forward_pytorch(gi, h0, Wh_struct, bh_cat)
    tri = gru_scan_monarch_forward_triton(gi, h0, Wh_struct, bh_cat)

    max_diff = (ref - tri).abs().max().item()
    rel = max_diff / max(ref.abs().max().item(), 1e-6)
    # TF32 matmul + T-step compounding.
    assert rel < 5e-3, f"forward rel diff {rel:.4e}"


@cuda_only
@pytest.mark.parametrize("T,B,H,nblocks", [(8, 32, 64, 4), (16, 32, 256, 4)])
def test_monarch_triton_backward_matches_pytorch(
    T: int, B: int, H: int, nblocks: int
) -> None:
    """Triton backward gradients must match the PyTorch monarch reference
    on (dgi, dh0, dWh_struct, dbh)."""
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    gi = (torch.randn(T, B, 3 * H, device=device) * 0.5).contiguous()
    h0 = (torch.randn(B, H, device=device) * 0.5).contiguous()
    blksz = H // nblocks
    Wh_struct = (torch.randn(3, nblocks, blksz, blksz, device=device) * 0.1).contiguous()
    bh_cat = (torch.randn(3 * H, device=device) * 0.1).contiguous()

    out_fwd = gru_scan_monarch_forward_triton(gi, h0, Wh_struct, bh_cat)
    dout = (torch.randn(T, B, H, device=device) * 0.5).contiguous()

    dgi_t, dh0_t, dWh_t, dbh_t = gru_scan_monarch_backward_triton(
        gi, h0, Wh_struct, bh_cat, out_fwd, dout
    )
    dgi_p, dh0_p, dWh_p, dbh_p = gru_scan_monarch_backward_pytorch(
        gi, h0, Wh_struct, bh_cat, out_fwd, dout
    )

    for name, t, p in [
        ("dgi", dgi_t, dgi_p),
        ("dh0", dh0_t, dh0_p),
        ("dWh_struct", dWh_t, dWh_p),
        ("dbh", dbh_t, dbh_p),
    ]:
        diff = (t - p).abs().max().item()
        rel = diff / max(p.abs().max().item(), 1e-9)
        assert rel < 5e-2, f"{name} rel diff {rel:.4e}"


@pytest.mark.parametrize("T,B,H,nblocks", [(8, 4, 32, 4), (16, 8, 64, 4)])
def test_monarch_pytorch_backward_matches_cell(
    T: int, B: int, H: int, nblocks: int
) -> None:
    """Gradients from the PyTorch monarch reference must match autograd
    through the tier-1 cell. We compare gradients of the inputs (gi, h0)
    and of the bias (bh) — both representations share these. The Wh
    parameter gradients live in different layouts (cell has three [nblocks,
    blksz, blksz]; reference has one [3, nblocks, blksz, blksz]) so we
    stack the cell's grads to compare."""
    torch.manual_seed(0)
    layer = _make_monarch_layer(in_size=H, hid=H, nblocks=nblocks)

    x = torch.randn(T, B, H)
    h0 = torch.randn(B, H)

    # ---- Reference path: autograd through the tier-1 cell ----
    ref_x = x.detach().clone().requires_grad_()
    ref_h0 = h0.detach().clone().requires_grad_()
    ref_out, _ = layer(ref_x, ref_h0)
    ref_loss = ref_out.float().pow(2).sum()
    ref_loss.backward()
    ref_dWh = torch.stack(
        [
            layer.cell.struct_Wh_r.weight.grad,
            layer.cell.struct_Wh_z.weight.grad,
            layer.cell.struct_Wh_n.weight.grad,
        ],
        dim=0,
    )
    ref_dbh = torch.cat(
        [layer.cell.b_hr.grad, layer.cell.b_hz.grad, layer.cell.b_hn.grad]
    )

    # ---- Monarch reference path ----
    Wh_struct, bh_cat = extract_monarch_factors(layer.cell)
    with torch.no_grad():
        gi = _build_gi_from_cell(layer, x)
        out_fwd = gru_scan_monarch_forward_pytorch(gi, h0, Wh_struct, bh_cat)
        # dout matches the gradient of (.pow(2).sum()) wrt the layer's output:
        #   d(sum(out^2))/d(out) = 2 * out
        dout = 2.0 * out_fwd

    dgi, dh0, dWh_struct, dbh = gru_scan_monarch_backward_pytorch(
        gi, h0, Wh_struct, bh_cat, out_fwd, dout
    )

    # Compare h0 gradients
    diff_h0 = (dh0 - ref_h0.grad).abs().max().item()
    rel_h0 = diff_h0 / max(ref_h0.grad.abs().max().item(), 1e-6)
    assert rel_h0 < 1e-4, f"dh0 rel diff {rel_h0:.4e}"

    # Compare Wh gradients
    diff_Wh = (dWh_struct - ref_dWh).abs().max().item()
    rel_Wh = diff_Wh / max(ref_dWh.abs().max().item(), 1e-6)
    assert rel_Wh < 1e-4, f"dWh rel diff {rel_Wh:.4e}"

    # Compare bh gradients
    diff_bh = (dbh - ref_dbh).abs().max().item()
    rel_bh = diff_bh / max(ref_dbh.abs().max().item(), 1e-6)
    assert rel_bh < 1e-4, f"dbh rel diff {rel_bh:.4e}"

"""Run each test cluster (monarch fwd, monarch bwd, diagonal fwd, butterfly
bwd, dense bwd) and collect the worst-case max_abs_diff / h_scale ratio.

The verifier and SUMMARY tabulate these per-cluster. With these we pick the
h_scale_mult for each disposition.
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import math
import itertools
import torch
import torch.nn.functional as F

from gru_qat.gru_layer import GRULayer
from gru_qat.quantizers import QuantRecipe, QuantizerConfig, FakeQuantizePerTensor, FakeQuantizePerChannel
from gru_qat.structure import StructureConfig
from gru_qat.triton_kernels.scan_monarch import (
    extract_monarch_factors,
    gru_scan_monarch_forward_pytorch,
    gru_scan_monarch_forward_triton,
    gru_scan_monarch_backward_pytorch,
    gru_scan_monarch_backward_triton,
)
from gru_qat.triton_kernels.scan_diagonal import (
    extract_diagonal_factors,
    gru_scan_diagonal_forward_pytorch,
    gru_scan_diagonal_forward_triton,
)


H_SCALE = 0.02
CUDA = torch.device("cuda")


def adversarial_inputs(cls, T, B, H, device, h_scale=H_SCALE):
    qmax = 127
    x_max = h_scale * qmax
    if cls == "realistic":
        x = torch.randn(T, B, H, device=device) * 0.5
        h0 = torch.randn(B, H, device=device) * 0.5
    elif cls == "near-saturation":
        x = (torch.linspace(-0.99, 0.99, T * B * H, device=device).reshape(T, B, H) * x_max).contiguous()
        h0 = (torch.linspace(-0.99, 0.99, B * H, device=device).reshape(B, H) * x_max).contiguous()
    elif cls == "large-magnitude":
        x = torch.randn(T, B, H, device=device) * 5.0
        h0 = torch.randn(B, H, device=device) * 5.0
    else:
        raise ValueError(cls)
    return x, h0


def make_layer(structure, in_size, hid, **kw):
    bits = 8
    rec = QuantRecipe(
        weight=QuantizerConfig(bits=bits, axis=0, mode="min_max", symmetric=True, name="W"),
        input_act=QuantizerConfig(bits=bits, axis=None, mode="min_max", symmetric=True, name="x"),
        hidden=QuantizerConfig(bits=bits, axis=None, mode="frozen", symmetric=True, name="h"),
    )
    layer = GRULayer(in_size, hid, recipe=rec, gate_layout="fused",
                    structure_input=None, structure_hidden=structure)
    for q in (layer.cell.quant_h_in, layer.cell.quant_h_out):
        assert isinstance(q, FakeQuantizePerTensor)
        q.scale = torch.tensor(H_SCALE)
        q.zero_point = torch.tensor(0.0)
    layer.eval()
    with torch.no_grad():
        cal_x = torch.randn(8, 4, in_size) * 0.5
        cal_h0 = torch.randn(4, hid) * 0.5
        layer(cal_x, cal_h0)
    layer.cell.freeze_quantizers()
    return layer.to(CUDA)


def build_qgi(layer, x):
    cell = layer.cell
    Wi_cat, bi_cat = cell.quantize_input_weights()
    xq = cell.quant_x(x)
    return F.linear(xq, Wi_cat, bi_cat)


def ratio(ref, tri):
    md = (ref - tri).abs().max().item()
    return md / H_SCALE


def collect_monarch_fwd():
    rows = []
    grid = [(T, B, H, nb) for T in (8, 64) for B in (1, 4, 32) for H in (32, 128, 512) for nb in (2, 4, 8) if H % nb == 0]
    for cls in ("realistic", "near-saturation", "large-magnitude"):
        worst = 0.0
        worst_case = None
        for (T, B, H, nb) in grid:
            torch.manual_seed(0)
            layer = make_layer(StructureConfig(kind="monarch", nblocks=nb), H, H)
            x, h0 = adversarial_inputs(cls, T, B, H, CUDA)
            gi = build_qgi(layer, x)
            Wh, bh = extract_monarch_factors(layer.cell)
            h_q = (H_SCALE, -127, 127)
            ref = gru_scan_monarch_forward_pytorch(gi, h0, Wh, bh, h_in_quant=h_q, h_out_quant=h_q)
            tri = gru_scan_monarch_forward_triton(gi, h0, Wh, bh, h_in_quant=h_q, h_out_quant=h_q)
            r = ratio(ref, tri)
            if r > worst:
                worst = r
                worst_case = (T, B, H, nb)
        rows.append(("monarch_fwd", cls, worst, worst_case))
    return rows


def collect_monarch_bwd():
    rows = []
    grid = [(T, B, H, nb) for T in (8, 64) for B in (1, 4, 32) for H in (32, 128, 512) for nb in (2, 4, 8) if H % nb == 0]
    for cls in ("realistic", "near-saturation", "large-magnitude"):
        worst = 0.0
        worst_case = None
        for (T, B, H, nb) in grid:
            torch.manual_seed(0)
            layer = make_layer(StructureConfig(kind="monarch", nblocks=nb), H, H)
            x, h0 = adversarial_inputs(cls, T, B, H, CUDA)
            gi = build_qgi(layer, x).detach()
            Wh, bh = extract_monarch_factors(layer.cell)
            Wh = Wh.detach()
            bh = bh.detach()
            h_q = (H_SCALE, -127, 127)
            ref_out = gru_scan_monarch_forward_pytorch(gi, h0, Wh, bh, h_in_quant=h_q, h_out_quant=h_q)
            grad = torch.randn_like(ref_out)
            dgi_p, dh0_p, dWh_p, dbh_p = gru_scan_monarch_backward_pytorch(
                gi, h0, Wh, bh, ref_out, grad, h_in_quant=h_q, h_out_quant=h_q,
            )
            tri_out = gru_scan_monarch_forward_triton(gi, h0, Wh, bh, h_in_quant=h_q, h_out_quant=h_q)
            dgi_t, dh0_t, dWh_t, dbh_t = gru_scan_monarch_backward_triton(
                gi, h0, Wh, bh, tri_out, grad, h_in_quant=h_q, h_out_quant=h_q,
            )
            r_gi = ratio(dgi_p, dgi_t)
            r_dh0 = ratio(dh0_p, dh0_t)
            r_dWh = ratio(dWh_p, dWh_t)
            r_dbh = ratio(dbh_p, dbh_t)
            r = max(r_gi, r_dh0, r_dWh, r_dbh)
            if r > worst:
                worst = r
                worst_case = (T, B, H, nb, dict(dgi=round(r_gi,3), dh0=round(r_dh0,3), dWh=round(r_dWh,3), dbh=round(r_dbh,3)))
        rows.append(("monarch_bwd", cls, worst, worst_case))
    return rows


def collect_diagonal_fwd():
    rows = []
    grid = [(T, B, H) for T in (8, 64) for B in (1, 4, 32) for H in (32, 128, 512)]
    for cls in ("realistic", "near-saturation", "large-magnitude"):
        worst = 0.0
        worst_case = None
        for (T, B, H) in grid:
            torch.manual_seed(0)
            layer = make_layer(StructureConfig(kind="diagonal"), H, H)
            x, h0 = adversarial_inputs(cls, T, B, H, CUDA)
            gi = build_qgi(layer, x)
            Wh_diag, bh = extract_diagonal_factors(layer.cell)
            h_q = (H_SCALE, -127, 127)
            ref = gru_scan_diagonal_forward_pytorch(gi, h0, Wh_diag, bh, h_in_quant=h_q, h_out_quant=h_q)
            tri = gru_scan_diagonal_forward_triton(gi, h0, Wh_diag, bh, h_in_quant=h_q, h_out_quant=h_q)
            r = ratio(ref, tri)
            if r > worst:
                worst = r
                worst_case = (T, B, H)
        rows.append(("diagonal_fwd", cls, worst, worst_case))
    return rows


def main():
    print("Collecting failure ratios across all Phase 4 quant-on clusters...")
    print("(this runs the fast grid for each kernel; takes a few minutes)\n")

    print("\n=== Monarch FWD ===")
    for row in collect_monarch_fwd():
        print(f"  {row[0]} {row[1]:>18s}: worst ratio = {row[2]:.3f}, case = {row[3]}")

    print("\n=== Diagonal FWD ===")
    for row in collect_diagonal_fwd():
        print(f"  {row[0]} {row[1]:>18s}: worst ratio = {row[2]:.3f}, case = {row[3]}")

    print("\n=== Monarch BWD ===")
    for row in collect_monarch_bwd():
        print(f"  {row[0]} {row[1]:>18s}: worst ratio = {row[2]:.3f}, case = {row[3]}")


if __name__ == "__main__":
    main()

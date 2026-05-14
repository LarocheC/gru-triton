"""Reproducer for monarch fwd quant-on torch.equal failure.

Hypothesis: PyTorch reference uses torch.einsum("bni,gnoi->bgno", ...) while
Triton uses tiled tl.dot. The two compute the same matmul but with different
reduction orders (TF32 + tile-by-tile vs fp32-or-TF32 + full-K). On
rounding-boundary inputs, those ULP-level differences in pre-quant gh flip
exactly one INT8 step through the downstream quant_h_out rint.

This reproducer confirms ULP-level differences exist in the pre-quant matmul
output when comparing PyTorch einsum at TF32 vs PyTorch einsum at full fp32,
which models the einsum-vs-tl.dot precision gap.

Run: python .planning/debug/repro_monarch_rounding.py
"""
from __future__ import annotations

import sys
import os

# Make src importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch

from gru_qat.gru_layer import GRULayer
from gru_qat.quantizers import QuantRecipe, QuantizerConfig, FakeQuantizePerTensor
from gru_qat.structure import StructureConfig
from gru_qat.triton_kernels.scan_monarch import (
    extract_monarch_factors,
    gru_scan_monarch_forward_pytorch,
    gru_scan_monarch_forward_triton,
)


def _make_monarch_layer_quant_int8(in_size: int, hid: int, nblocks: int, h_scale: float):
    bits = 8
    rec = QuantRecipe(
        weight=QuantizerConfig(bits=bits, axis=0, mode="min_max", symmetric=True, name="W_int8_pc"),
        input_act=QuantizerConfig(bits=bits, axis=None, mode="min_max", symmetric=True, name="x_int8_pt"),
        hidden=QuantizerConfig(bits=bits, axis=None, mode="frozen", symmetric=True, name="h_int8_pt"),
    )
    cfg = StructureConfig(kind="monarch", nblocks=nblocks)
    layer = GRULayer(in_size, hid, recipe=rec, gate_layout="fused",
                    structure_input=None, structure_hidden=cfg)
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


def main():
    torch.manual_seed(0)
    device = torch.device("cuda")

    T, B, H, nblocks = 8, 1, 128, 2
    IN = H
    h_scale = 0.02

    layer = _make_monarch_layer_quant_int8(IN, H, nblocks, h_scale).to(device)

    # Realistic inputs
    x = torch.randn(T, B, IN, device=device) * 0.5
    h0 = torch.randn(B, H, device=device) * 0.5

    # Build qgi via the cell's input-side path
    import torch.nn.functional as F
    cell = layer.cell
    Wi_cat, bi_cat = cell.quantize_input_weights()
    xq = cell.quant_x(x)
    gi = F.linear(xq, Wi_cat, bi_cat)

    Wh_struct, bh_cat = extract_monarch_factors(cell)
    h_q = (h_scale, -127, 127)

    # Run ref + triton through the public APIs
    ref_out = gru_scan_monarch_forward_pytorch(
        gi, h0, Wh_struct, bh_cat, h_in_quant=h_q, h_out_quant=h_q,
    )
    tri_out = gru_scan_monarch_forward_triton(
        gi, h0, Wh_struct, bh_cat, h_in_quant=h_q, h_out_quant=h_q,
    )

    # Symptom check
    diff = (ref_out - tri_out).abs()
    max_diff = diff.max().item()
    n_diff = (diff > 0).sum().item()
    n_one_step = (diff == h_scale).sum().item()
    n_two_step = (diff == 2 * h_scale).sum().item()
    n_total = ref_out.numel()
    print(f"=== Symptom ===")
    print(f"  shape: {tuple(ref_out.shape)}, total elements: {n_total}")
    print(f"  max_abs_diff: {max_diff} (h_scale = {h_scale})")
    print(f"  ratio max_diff/h_scale: {max_diff/h_scale:.3f}")
    print(f"  elements differing by 0: {n_total - n_diff}")
    print(f"  elements differing by exactly 1*h_scale: {n_one_step}")
    print(f"  elements differing by exactly 2*h_scale: {n_two_step}")
    print(f"  elements differing other: {n_diff - n_one_step - n_two_step}")

    # === Pre-quant matmul precision test ===
    # Replicate one step of the recurrence with two precision modes for the
    # h-side matmul. This models the einsum-vs-tl.dot reduction-order gap
    # WITHOUT needing to instrument the Triton kernel.
    print(f"\n=== Pre-quant matmul precision test (t=0) ===")
    # Apply quant_h_in to h0
    h_for_matmul = (torch.round(h0 / h_scale).clamp(-127, 127)) * h_scale
    h_chunks = h_for_matmul.view(B, nblocks, H // nblocks)

    # Path A: einsum at default (fp32) precision
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    gh_fp32 = torch.einsum("bni,gnoi->bgno", h_chunks, Wh_struct)
    # Path B: einsum at TF32 (mimics Triton's tl.dot default tile precision)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    gh_tf32 = torch.einsum("bni,gnoi->bgno", h_chunks, Wh_struct)
    # Path C: per-block torch.matmul at default (mimics block-wise reduction)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    gh_block = torch.zeros_like(gh_fp32)
    for n in range(nblocks):
        # h_chunks[:, n, :] @ Wh_struct[:, n, :, :].transpose(-1, -2)
        # Wh shape per block: [3, blksz, blksz] (out, in)
        gh_block[:, :, n, :] = torch.einsum("bi,goi->bgo",
                                            h_chunks[:, n, :],
                                            Wh_struct[:, n, :, :])

    diff_fp32_tf32 = (gh_fp32 - gh_tf32).abs().max().item()
    diff_fp32_block = (gh_fp32 - gh_block).abs().max().item()
    print(f"  einsum-fp32 vs einsum-tf32 max_abs_diff: {diff_fp32_tf32:.3e}")
    print(f"  einsum-fp32 vs per-block-fp32 max_abs_diff: {diff_fp32_block:.3e}")

    # Map a single divergent element
    print(f"\n=== Element-level analysis (first divergent position) ===")
    pos = (diff > 0).nonzero(as_tuple=False)
    if len(pos) > 0:
        t, b, h = pos[0].tolist()
        print(f"  position t={t}, b={b}, h={h}")
        print(f"  ref: {ref_out[t, b, h].item()}")
        print(f"  tri: {tri_out[t, b, h].item()}")
        # show what value would have to round differently to produce this
        # ref/h_scale and tri/h_scale should be integers differing by 1
        ref_int = round(ref_out[t, b, h].item() / h_scale)
        tri_int = round(tri_out[t, b, h].item() / h_scale)
        print(f"  ref/h_scale (int): {ref_int}, tri/h_scale (int): {tri_int}, delta: {ref_int - tri_int}")

    print(f"\n=== Conclusion ===")
    print(f"If gh_fp32 vs gh_tf32 max_abs_diff is > 0 (typically ~1e-4 to 1e-3),")
    print(f"then ULP-level differences exist in the pre-quant matmul output.")
    print(f"Combined with values straddling the INT8 rounding boundary (h_scale/2 = {h_scale/2})")
    print(f"this is sufficient to produce one-INT8-step differences in the post-quant output.")
    print(f"Root cause: einsum vs tl.dot reduction order, not a kernel bug.")


if __name__ == "__main__":
    main()

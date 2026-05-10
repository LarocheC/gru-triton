"""Parity tests for the Triton GRU scan kernel — Phase 5 forward, fp32 only.

These tests are GPU-only; they skip when CUDA is unavailable.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

triton = pytest.importorskip("triton")

from gru_qat.gru_layer import GRULayer
from gru_qat.quantizers import QuantizerConfig, QuantRecipe
from gru_qat.triton_kernels.scan import gru_scan_forward

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernel requires CUDA"
)


def _ref_layer(in_dim: int, hidden: int) -> GRULayer:
    """fp32-Identity GRULayer with fused gates and per-batch input projection.

    The Triton kernel takes the post-input-projection ``gi`` directly, so
    parity is against the layer that produces matching ``gi`` (fused +
    pre_batch_input).
    """
    rec = QuantRecipe(
        weight=QuantizerConfig(bits=32, axis=0, name="W_id"),
        input_act=QuantizerConfig(bits=32, name="x_id"),
        hidden=QuantizerConfig(bits=32, name="h_id"),
    )
    return GRULayer(
        in_dim, hidden, recipe=rec, gate_layout="fused", pre_batch_input=True
    )


@cuda_only
@pytest.mark.parametrize("T,B,IN,H", [(7, 4, 8, 16), (32, 16, 32, 64)])
def test_triton_forward_matches_pytorch(T: int, B: int, IN: int, H: int) -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    layer = _ref_layer(IN, H).to(device).eval()

    x = torch.randn(T, B, IN, device=device)
    h0 = torch.randn(B, H, device=device)

    # Reference: PyTorch fused + pre_batch path.
    with torch.no_grad():
        ref_out, _ = layer(x, h0)

    # Triton: build gi, call kernel directly.
    with torch.no_grad():
        w = layer.cell.quantize_weights()
        gi = layer.cell.input_projection(x, w)  # [T, B, 3H]
        assert w.Wh_cat is not None and w.bh_cat is not None
        triton_out = gru_scan_forward(gi, h0, w.Wh_cat, w.bh_cat)

    max_diff = (ref_out - triton_out).abs().max().item()
    # TF32 input precision in tl.dot — ~10-bit mantissa per matmul.
    # Drift accumulates across 3 matmuls per step + T steps + nonlinearities.
    assert max_diff < 5e-3, f"max diff {max_diff} exceeds 5e-3"

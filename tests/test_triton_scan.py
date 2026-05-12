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
from gru_qat.triton_kernels.scan import (
    gru_scan,
    gru_scan_forward,
    gru_scan_forward_persistent,
    gru_scan_backward_persistent,
    gru_scan_persistent,
    _gru_scan_backward_pytorch,
)

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
@pytest.mark.parametrize("T,B,IN,H", [(8, 16, 32, 128), (16, 32, 64, 256)])
def test_triton_forward_persistent_matches_default(
    T: int, B: int, IN: int, H: int
) -> None:
    """Persistent forward kernel must match the autotune (non-persistent)
    forward kernel within TF32 noise. Uses shapes whose persistent-grid
    size fits inside the SM count (block_b=8, block_oh=H caps grid)."""
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    layer = _ref_layer(IN, H).to(device).eval()

    x = torch.randn(T, B, IN, device=device)
    h0 = torch.randn(B, H, device=device)

    with torch.no_grad():
        w = layer.cell.quantize_weights()
        gi = layer.cell.input_projection(x, w)
        out_default = gru_scan_forward(gi, h0, w.Wh_cat, w.bh_cat)
        out_persist = gru_scan_forward_persistent(
            gi, h0, w.Wh_cat, w.bh_cat,
            block_b=8, block_oh=min(H, 128), block_k=32,
        )

    max_diff = (out_default - out_persist).abs().max().item()
    rel = max_diff / max(out_default.abs().max().item(), 1e-6)
    # TF32 in both paths but different accumulation orders → looser bound.
    assert rel < 5e-2, f"persistent vs default rel diff {rel:.4f}"


@cuda_only
@pytest.mark.parametrize("T,B,IN,H", [(8, 16, 32, 128), (16, 32, 64, 256)])
def test_triton_backward_persistent_matches_pytorch(
    T: int, B: int, IN: int, H: int
) -> None:
    """Persistent backward kernel must match the PyTorch reference
    backward to within TF32 noise on the four gradient outputs."""
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    gi = torch.randn(T, B, 3 * H, device=device) * 0.5
    h0 = torch.randn(B, H, device=device) * 0.5
    Wh = torch.randn(3 * H, H, device=device) * 0.1
    bh = torch.randn(3 * H, device=device) * 0.1
    out_fwd = gru_scan_forward(gi, h0, Wh, bh)
    dout = torch.randn(T, B, H, device=device) * 0.5

    dgi_t, dh0_t, dWh_t, dbh_t = gru_scan_backward_persistent(
        gi, h0, Wh, bh, out_fwd, dout,
        block_b=16, block_oh=min(H, 64), block_k=32,
    )
    dgi_p, dh0_p, dWh_p, dbh_p = _gru_scan_backward_pytorch(
        gi, h0, Wh, bh, out_fwd, dout,
    )

    for name, t, p in [
        ("dgi", dgi_t, dgi_p),
        ("dh0", dh0_t, dh0_p),
        ("dWh", dWh_t, dWh_p),
        ("dbh", dbh_t, dbh_p),
    ]:
        diff = (t - p).abs().max().item()
        rel = diff / max(p.abs().max().item(), 1e-9)
        assert rel < 1e-1, f"{name} rel diff {rel:.4f}"


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


@cuda_only
@pytest.mark.parametrize("T,B,IN,H", [(7, 4, 8, 16), (16, 8, 16, 32)])
def test_triton_backward_matches_pytorch(T: int, B: int, IN: int, H: int) -> None:
    """Gradients from the Triton autograd Function must match PyTorch
    autograd through the reference layer.

    Both paths are forced to TF32 for matmuls so the precision regimes
    match; remaining drift is from kernel logic, not arithmetic. Tolerance
    is set conservatively because TF32 has ~10-bit mantissa and gradient
    magnitudes compound across T timesteps and three matmuls per step.
    """
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    # Build a reference layer to materialize Wh_cat/bh_cat with matching init.
    ref_layer = _ref_layer(IN, H).to(device)

    x = torch.randn(T, B, IN, device=device, requires_grad=True)
    h0 = torch.randn(B, H, device=device, requires_grad=True)

    # Reference path: PyTorch autograd through the layer
    ref_x = x.detach().clone().requires_grad_()
    ref_h0 = h0.detach().clone().requires_grad_()
    ref_out, _ = ref_layer(ref_x, ref_h0)
    ref_loss = ref_out.float().pow(2).sum()
    ref_loss.backward()

    # Triton path: pre-batch input projection, run gru_scan, backward
    # We need to backprop through the input projection too, so we don't use
    # cell.input_projection (no_grad in eval). Build it explicitly.
    w = ref_layer.cell.quantize_weights()
    Wi_cat = w.Wi_cat.detach().clone()
    bi_cat = w.bi_cat.detach().clone()
    Wh_cat = w.Wh_cat.detach().clone().requires_grad_()
    bh_cat = w.bh_cat.detach().clone().requires_grad_()

    tri_x = x.detach().clone().requires_grad_()
    tri_h0 = h0.detach().clone().requires_grad_()
    gi = torch.nn.functional.linear(tri_x, Wi_cat, bi_cat)  # [T, B, 3H]
    out = gru_scan(gi, tri_h0, Wh_cat, bh_cat)
    loss = out.float().pow(2).sum()
    loss.backward()

    # Compare scalar loss (forward parity already tested elsewhere).
    assert abs(loss.item() - ref_loss.item()) / max(abs(ref_loss.item()), 1.0) < 1e-2

    # Reconstruct the reference dWh_cat / dbh_cat by concatenating the
    # per-gate cell weight grads in the same order that quantize_weights()
    # builds Wh_cat (r, z, n along axis 0).
    ref_dWh_cat = torch.cat(
        [ref_layer.cell.W_hr.grad, ref_layer.cell.W_hz.grad, ref_layer.cell.W_hn.grad],
        dim=0,
    )
    ref_dbh_cat = torch.cat(
        [ref_layer.cell.b_hr.grad, ref_layer.cell.b_hz.grad, ref_layer.cell.b_hn.grad],
        dim=0,
    )

    # Compare gradients on x, h0, the hidden weights, and the hidden bias.
    # dWh_cat / dbh_cat parity catches a class of autotune-related bugs in the
    # backward kernel where the partial-accumulator buffer is reused across
    # trial configs without being zeroed (regression test for the fix in
    # gru_scan_bwd_kernel that zeros the per-program slab on entry).
    for name, ref_g, tri_g in [
        ("x", ref_x.grad, tri_x.grad),
        ("h0", ref_h0.grad, tri_h0.grad),
        ("Wh_cat", ref_dWh_cat, Wh_cat.grad),
        ("bh_cat", ref_dbh_cat, bh_cat.grad),
    ]:
        assert ref_g is not None and tri_g is not None
        max_diff = (ref_g - tri_g).abs().max().item()
        rel = max_diff / max(ref_g.abs().max().item(), 1e-6)
        assert rel < 1e-1, f"{name} grad rel diff {rel:.4f} exceeds 1e-1"


@cuda_only
@pytest.mark.parametrize("T,B,IN,H", [(8, 16, 32, 128), (16, 32, 64, 256)])
def test_triton_qat_persistent_matches_pytorch(
    T: int, B: int, IN: int, H: int
) -> None:
    """Persistent kernels with in-kernel fake-quant must match the
    GRULayer reference (frozen hidden quantizer). Forward + gradients."""
    from gru_qat.gru_layer import GRULayer
    from gru_qat.quantizers import (
        FakeQuantizePerTensor,
        QuantizerConfig,
        QuantRecipe,
    )

    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    bits = 8
    qmin, qmax = -(2 ** (bits - 1)) + 1, 2 ** (bits - 1) - 1
    h_scale = 0.02

    rec = QuantRecipe(
        weight=QuantizerConfig(bits=32, axis=0, name="W_id"),
        input_act=QuantizerConfig(bits=32, name="x_id"),
        hidden=QuantizerConfig(bits=bits, mode="frozen", name="h_q"),
    )
    ref = GRULayer(
        IN, H, recipe=rec, gate_layout="fused", pre_batch_input=True,
    ).to(device)
    for q in (ref.cell.quant_h_in, ref.cell.quant_h_out):
        assert isinstance(q, FakeQuantizePerTensor)
        q.scale = torch.tensor(h_scale, device=device)
        q.zero_point = torch.tensor(0.0, device=device)

    x = torch.randn(T, B, IN, device=device) * 0.1
    h0 = torch.randn(B, H, device=device) * 0.1

    ref_x = x.detach().clone().requires_grad_()
    ref_h0 = h0.detach().clone().requires_grad_()
    ref_out, _ = ref(ref_x, ref_h0)
    ref_loss = ref_out.float().pow(2).sum()
    ref_loss.backward()

    w = ref.cell.quantize_weights()
    Wi_cat = w.Wi_cat.detach().clone()
    bi_cat = w.bi_cat.detach().clone()
    Wh_cat = w.Wh_cat.detach().clone().requires_grad_()
    bh_cat = w.bh_cat.detach().clone().requires_grad_()
    tri_x = x.detach().clone().requires_grad_()
    tri_h0 = h0.detach().clone().requires_grad_()
    gi = torch.nn.functional.linear(tri_x, Wi_cat, bi_cat)
    out = gru_scan_persistent(
        gi, tri_h0, Wh_cat, bh_cat,
        h_in_quant=(h_scale, qmin, qmax),
        h_out_quant=(h_scale, qmin, qmax),
    )
    loss = out.float().pow(2).sum()
    loss.backward()

    fwd_diff = (ref_out - out).abs().max().item()
    fwd_rel = fwd_diff / max(ref_out.abs().max().item(), 1e-6)
    assert fwd_rel < 1e-1, f"fwd rel diff {fwd_rel:.4f}"

    for name, ref_g, tri_g in [
        ("x", ref_x.grad, tri_x.grad),
        ("h0", ref_h0.grad, tri_h0.grad),
    ]:
        max_diff = (ref_g - tri_g).abs().max().item()
        rel = max_diff / max(ref_g.abs().max().item(), 1e-6)
        assert rel < 2e-1, f"{name} grad rel diff {rel:.4f}"


@cuda_only
@pytest.mark.parametrize("T,B,IN,H", [(8, 4, 16, 32), (16, 8, 16, 32)])
def test_triton_qat_matches_pytorch(T: int, B: int, IN: int, H: int) -> None:
    """In-kernel fake-quant on the hidden state must match the cell's
    quant_h_in / quant_h_out semantics, both forward and backward.

    Reference path: a GRULayer with hidden quantizer manually frozen at a
    chosen scale. Triton path: gru_scan() with the same scale / qrange.
    """
    from gru_qat.gru_layer import GRULayer
    from gru_qat.quantizers import (
        FakeQuantizePerTensor,
        QuantizerConfig,
        QuantRecipe,
    )
    from gru_qat.triton_kernels.scan import gru_scan

    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    # int8 symmetric, frozen at a sensible scale for the synthetic data.
    bits = 8
    qmin, qmax = -(2 ** (bits - 1)) + 1, 2 ** (bits - 1) - 1
    h_scale = 0.02  # keeps activations in-range for the random init below

    rec = QuantRecipe(
        weight=QuantizerConfig(bits=32, axis=0, name="W_id"),  # weights fp32
        input_act=QuantizerConfig(bits=32, name="x_id"),       # input fp32
        hidden=QuantizerConfig(bits=bits, mode="frozen", name="h_q"),
    )
    ref = GRULayer(
        IN, H, recipe=rec, gate_layout="fused", pre_batch_input=True
    ).to(device)
    # Manually set the frozen scales on quant_h_in and quant_h_out.
    for q in (ref.cell.quant_h_in, ref.cell.quant_h_out):
        assert isinstance(q, FakeQuantizePerTensor)
        q.scale = torch.tensor(h_scale, device=device)
        q.zero_point = torch.tensor(0.0, device=device)

    x = torch.randn(T, B, IN, device=device) * 0.1
    h0 = torch.randn(B, H, device=device) * 0.1

    # ----- Reference forward+backward -----
    ref_x = x.detach().clone().requires_grad_()
    ref_h0 = h0.detach().clone().requires_grad_()
    ref_out, _ = ref(ref_x, ref_h0)
    ref_loss = ref_out.float().pow(2).sum()
    ref_loss.backward()

    # ----- Triton path -----
    w = ref.cell.quantize_weights()
    Wi_cat = w.Wi_cat.detach().clone()
    bi_cat = w.bi_cat.detach().clone()
    Wh_cat = w.Wh_cat.detach().clone().requires_grad_()
    bh_cat = w.bh_cat.detach().clone().requires_grad_()
    tri_x = x.detach().clone().requires_grad_()
    tri_h0 = h0.detach().clone().requires_grad_()
    gi = torch.nn.functional.linear(tri_x, Wi_cat, bi_cat)
    out = gru_scan(
        gi, tri_h0, Wh_cat, bh_cat,
        h_in_quant=(h_scale, qmin, qmax),
        h_out_quant=(h_scale, qmin, qmax),
    )
    loss = out.float().pow(2).sum()
    loss.backward()

    # Forward output parity. Tolerance is loose: per-step fake-quant noise
    # plus TF32 matmul noise compounds across T timesteps.
    fwd_diff = (ref_out - out).abs().max().item()
    fwd_rel = fwd_diff / max(ref_out.abs().max().item(), 1e-6)
    assert fwd_rel < 1e-1, f"fwd rel diff {fwd_rel:.4f}"

    # Gradient parity on x and h0
    for name, ref_g, tri_g in [
        ("x", ref_x.grad, tri_x.grad),
        ("h0", ref_h0.grad, tri_h0.grad),
    ]:
        max_diff = (ref_g - tri_g).abs().max().item()
        rel = max_diff / max(ref_g.abs().max().item(), 1e-6)
        assert rel < 2e-1, f"{name} grad rel diff {rel:.4f}"

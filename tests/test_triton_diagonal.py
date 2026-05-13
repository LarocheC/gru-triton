"""Diagonal-hidden-side persistent kernel tests.

Mirrors test_triton_monarch.py:
- Stage A: factor extraction + PyTorch reference parity vs the cell.
- Stage B: Triton forward (fp32 + QAT) vs PyTorch reference.
- Stage C: Triton backward (fp32 + QAT) vs PyTorch reference.
- Stage D: end-to-end GRULayer dispatch — output and gradients match the
  per-step structured path.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", message=".*different CUDA versions.*")

import pytest  # noqa: E402
import torch  # noqa: E402

from gru_qat import GRULayer, QuantRecipe, QuantizerConfig, StructureConfig  # noqa: E402
from gru_qat.triton_kernels.scan_diagonal import (  # noqa: E402
    extract_diagonal_factors,
    gru_scan_diagonal_backward_pytorch,
    gru_scan_diagonal_backward_triton,
    gru_scan_diagonal_forward_pytorch,
    gru_scan_diagonal_forward_triton,
)


cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernel requires CUDA"
)


def _make_diagonal_layer(in_size: int, hid: int) -> GRULayer:
    """fp32 structured-diagonal layer (no quant) for clean reference math."""
    rec = QuantRecipe(
        weight=QuantizerConfig(bits=32, axis=0, name="W_id"),
        input_act=QuantizerConfig(bits=32, name="x_id"),
        hidden=QuantizerConfig(bits=32, name="h_id"),
    )
    cfg = StructureConfig(kind="diagonal")
    return GRULayer(
        in_size, hid, recipe=rec,
        gate_layout="fused",
        structure_input=None,
        structure_hidden=cfg,
    )


def _build_gi_from_cell(layer: GRULayer, x: torch.Tensor) -> torch.Tensor:
    """Reproduce the cell's input projection so the diagonal reference
    sees the same gi as the layer's per-step path. Mirror of the helper
    in test_triton_monarch.
    """
    cell = layer.cell
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


# ---------------------------------------------------------------------------
# Stage A: PyTorch reference vs the cell's structured forward path.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("T,B,H", [(8, 4, 32), (16, 8, 64)])
def test_diagonal_pytorch_forward_matches_cell(T: int, B: int, H: int) -> None:
    """gru_scan_diagonal_forward_pytorch must match the layer's per-step
    structured path bit-for-bit (fp32, Identity quant)."""
    torch.manual_seed(0)
    layer = _make_diagonal_layer(in_size=H, hid=H)
    layer.eval()

    x = torch.randn(T, B, H)
    h0 = torch.randn(B, H)

    with torch.no_grad():
        ref_out, _ = layer(x, h0)
        Wh_diag, bh_cat = extract_diagonal_factors(layer.cell)
        gi = _build_gi_from_cell(layer, x)
        diag_out = gru_scan_diagonal_forward_pytorch(gi, h0, Wh_diag, bh_cat)

    max_diff = (ref_out - diag_out).abs().max().item()
    rel = max_diff / max(ref_out.abs().max().item(), 1e-6)
    assert rel < 1e-5, f"forward rel diff {rel:.4e}"


# ---------------------------------------------------------------------------
# Stage B: Triton forward vs PyTorch reference.
# ---------------------------------------------------------------------------

@cuda_only
@pytest.mark.parametrize("T,B,H", [(8, 32, 64), (16, 32, 256), (64, 32, 512)])
def test_diagonal_triton_forward_matches_pytorch(T: int, B: int, H: int) -> None:
    """Triton forward must match the PyTorch reference. The diagonal
    recurrence has no matmul on the hidden side, so the only floating-point
    noise comes from the input-side gi (already provided as a tensor here)
    and the per-step nonlinearities — both bit-stable. Tight tolerance."""
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    gi = (torch.randn(T, B, 3 * H, device=device) * 0.5).contiguous()
    h0 = (torch.randn(B, H, device=device) * 0.5).contiguous()
    Wh_diag = (torch.randn(3, H, device=device) * 0.3).contiguous()
    bh_cat = (torch.randn(3 * H, device=device) * 0.1).contiguous()

    ref = gru_scan_diagonal_forward_pytorch(gi, h0, Wh_diag, bh_cat)
    tri = gru_scan_diagonal_forward_triton(gi, h0, Wh_diag, bh_cat)

    max_diff = (ref - tri).abs().max().item()
    rel = max_diff / max(ref.abs().max().item(), 1e-6)
    assert rel < 1e-4, f"forward rel diff {rel:.4e}"


@cuda_only
@pytest.mark.parametrize("T,B,H", [(8, 32, 64), (16, 32, 256)])
def test_diagonal_triton_qat_forward_matches_pytorch(T: int, B: int, H: int) -> None:
    """In-kernel fake-quant forward: Triton must match PyTorch reference."""
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    gi = (torch.randn(T, B, 3 * H, device=device) * 0.1).contiguous()
    h0 = (torch.randn(B, H, device=device) * 0.1).contiguous()
    Wh_diag = (torch.randn(3, H, device=device) * 0.3).contiguous()
    bh_cat = (torch.randn(3 * H, device=device) * 0.05).contiguous()

    bits = 8
    qmin, qmax = -(2 ** (bits - 1)) + 1, 2 ** (bits - 1) - 1
    h_in_q = (0.02, qmin, qmax)
    h_out_q = (0.02, qmin, qmax)

    ref = gru_scan_diagonal_forward_pytorch(
        gi, h0, Wh_diag, bh_cat,
        h_in_quant=h_in_q, h_out_quant=h_out_q,
    )
    tri = gru_scan_diagonal_forward_triton(
        gi, h0, Wh_diag, bh_cat,
        h_in_quant=h_in_q, h_out_quant=h_out_q,
    )

    max_diff = (ref - tri).abs().max().item()
    rel = max_diff / max(ref.abs().max().item(), 1e-6)
    # Diagonal has no matmul, so torch.round and tl.rint should agree
    # bit-for-bit on the same inputs — quant_h_in/out drift across the
    # T-step recurrence is the only noise source.
    assert rel < 1e-3, f"qat forward rel diff {rel:.4e}"


# ---------------------------------------------------------------------------
# Stage C: Triton backward vs PyTorch reference.
# ---------------------------------------------------------------------------

@cuda_only
@pytest.mark.parametrize("T,B,H", [(8, 32, 64), (16, 32, 256)])
def test_diagonal_triton_backward_matches_pytorch(T: int, B: int, H: int) -> None:
    """Triton backward must match PyTorch on (dgi, dh0, dWh_diag, dbh)."""
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    gi = (torch.randn(T, B, 3 * H, device=device) * 0.5).contiguous()
    h0 = (torch.randn(B, H, device=device) * 0.5).contiguous()
    Wh_diag = (torch.randn(3, H, device=device) * 0.3).contiguous()
    bh_cat = (torch.randn(3 * H, device=device) * 0.1).contiguous()

    out_fwd = gru_scan_diagonal_forward_triton(gi, h0, Wh_diag, bh_cat)
    dout = (torch.randn(T, B, H, device=device) * 0.5).contiguous()

    dgi_t, dh0_t, dWh_t, dbh_t = gru_scan_diagonal_backward_triton(
        gi, h0, Wh_diag, bh_cat, out_fwd, dout,
    )
    dgi_p, dh0_p, dWh_p, dbh_p = gru_scan_diagonal_backward_pytorch(
        gi, h0, Wh_diag, bh_cat, out_fwd, dout,
    )

    for name, t, p in [
        ("dgi", dgi_t, dgi_p),
        ("dh0", dh0_t, dh0_p),
        ("dWh_diag", dWh_t, dWh_p),
        ("dbh", dbh_t, dbh_p),
    ]:
        diff = (t - p).abs().max().item()
        rel = diff / max(p.abs().max().item(), 1e-9)
        assert rel < 1e-3, f"{name} rel diff {rel:.4e}"


@cuda_only
@pytest.mark.parametrize("T,B,H", [(8, 32, 64), (16, 32, 256)])
def test_diagonal_triton_qat_backward_matches_pytorch(T: int, B: int, H: int) -> None:
    """In-kernel fake-quant backward: Triton must match PyTorch."""
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    gi = (torch.randn(T, B, 3 * H, device=device) * 0.1).contiguous()
    h0 = (torch.randn(B, H, device=device) * 0.1).contiguous()
    Wh_diag = (torch.randn(3, H, device=device) * 0.3).contiguous()
    bh_cat = (torch.randn(3 * H, device=device) * 0.05).contiguous()

    bits = 8
    qmin, qmax = -(2 ** (bits - 1)) + 1, 2 ** (bits - 1) - 1
    h_in_q = (0.02, qmin, qmax)
    h_out_q = (0.02, qmin, qmax)

    out_fwd = gru_scan_diagonal_forward_triton(
        gi, h0, Wh_diag, bh_cat,
        h_in_quant=h_in_q, h_out_quant=h_out_q,
    )
    dout = (torch.randn(T, B, H, device=device) * 0.1).contiguous()

    dgi_t, dh0_t, dWh_t, dbh_t = gru_scan_diagonal_backward_triton(
        gi, h0, Wh_diag, bh_cat, out_fwd, dout,
        h_in_quant=h_in_q, h_out_quant=h_out_q,
    )
    dgi_p, dh0_p, dWh_p, dbh_p = gru_scan_diagonal_backward_pytorch(
        gi, h0, Wh_diag, bh_cat, out_fwd, dout,
        h_in_quant=h_in_q, h_out_quant=h_out_q,
    )

    for name, t, p in [
        ("dgi", dgi_t, dgi_p),
        ("dh0", dh0_t, dh0_p),
        ("dWh_diag", dWh_t, dWh_p),
        ("dbh", dbh_t, dbh_p),
    ]:
        diff = (t - p).abs().max().item()
        rel = diff / max(p.abs().max().item(), 1e-9)
        # STE rounding can flip mask bits at boundaries → looser tol than fp32.
        assert rel < 1e-2, f"qat {name} rel diff {rel:.4e}"


# ---------------------------------------------------------------------------
# Stage D: end-to-end GRULayer dispatch.
# ---------------------------------------------------------------------------

@cuda_only
@pytest.mark.parametrize("T,B,H", [(8, 32, 64), (16, 32, 128)])
def test_diagonal_dispatch_matches_per_step(T: int, B: int, H: int) -> None:
    """use_triton=True for diagonal must produce the same forward output
    as the per-step structured path within fp32 noise."""
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    layer = _make_diagonal_layer(H, H).to(device)
    x = torch.randn(T, B, H, device=device)
    h0 = torch.randn(B, H, device=device)

    layer.use_triton = False
    with torch.no_grad():
        ref_out, _ = layer(x, h0)
    layer.use_triton = True
    with torch.no_grad():
        tri_out, _ = layer(x, h0)

    max_diff = (ref_out - tri_out).abs().max().item()
    rel = max_diff / max(ref_out.abs().max().item(), 1e-6)
    assert rel < 1e-4, f"dispatch fwd rel diff {rel:.4e}"


@cuda_only
def test_diagonal_dispatch_grad_flows() -> None:
    """Gradients flow through the Triton dispatch path on all params and
    h0 / x — covers the autograd.Function plumbing."""
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    T, B, H = 8, 16, 64

    layer = _make_diagonal_layer(H, H).to(device)
    x = torch.randn(T, B, H, device=device, requires_grad=True)
    h0 = torch.randn(B, H, device=device, requires_grad=True)

    layer.use_triton = True
    out, _ = layer(x, h0)
    loss = (out**2).mean()
    loss.backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert h0.grad is not None and torch.isfinite(h0.grad).all()
    # Hidden-side params (diagonal weights + biases).
    for name, p in layer.cell.named_parameters(recurse=True):
        if p.requires_grad:
            assert p.grad is not None, f"{name} has no grad"
            assert torch.isfinite(p.grad).all(), f"{name} grad not finite"


@cuda_only
def test_diagonal_dispatch_grad_matches_per_step() -> None:
    """Backward through the Triton dispatch path must match the per-step
    structured-autograd path on input and parameter gradients."""
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    T, B, H = 16, 8, 64

    # Reference: per-step structured path (use_triton=False), autograd.
    layer_ref = _make_diagonal_layer(H, H).to(device)
    layer_ref.use_triton = False
    x_ref = torch.randn(T, B, H, device=device, requires_grad=True)
    h0_ref = torch.randn(B, H, device=device, requires_grad=True)
    out_ref, _ = layer_ref(x_ref, h0_ref)
    g = torch.randn_like(out_ref) * 0.1
    out_ref.backward(g)

    # Triton dispatch: same weights, same inputs.
    layer_tri = _make_diagonal_layer(H, H).to(device)
    layer_tri.load_state_dict(layer_ref.state_dict())
    layer_tri.use_triton = True
    x_tri = x_ref.detach().clone().requires_grad_(True)
    h0_tri = h0_ref.detach().clone().requires_grad_(True)
    out_tri, _ = layer_tri(x_tri, h0_tri)
    out_tri.backward(g)

    for name, ref_p, tri_p in [
        ("x", x_ref.grad, x_tri.grad),
        ("h0", h0_ref.grad, h0_tri.grad),
    ]:
        diff = (ref_p - tri_p).abs().max().item()
        rel = diff / max(ref_p.abs().max().item(), 1e-9)
        assert rel < 1e-3, f"{name} grad rel diff {rel:.4e}"

    for name_ref, p_ref in layer_ref.cell.named_parameters():
        p_tri = dict(layer_tri.cell.named_parameters())[name_ref]
        if p_ref.grad is None:
            continue
        diff = (p_ref.grad - p_tri.grad).abs().max().item()
        rel = diff / max(p_ref.grad.abs().max().item(), 1e-9)
        assert rel < 1e-3, f"param {name_ref} grad rel diff {rel:.4e}"


@cuda_only
def test_diagonal_grulayer_qat_after_calibration() -> None:
    """End-to-end QAT flow: build, calibrate, freeze, forward on the
    diagonal fast path. Output must be finite."""
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    T, B, H = 16, 8, 64

    rec = QuantRecipe(
        weight=QuantizerConfig(bits=32, axis=0, name="W_id"),
        input_act=QuantizerConfig(bits=32, name="x_id"),
        hidden=QuantizerConfig(bits=8, name="h_q"),
    )
    layer = GRULayer(
        H, H, recipe=rec,
        gate_layout="fused",
        structure_hidden=StructureConfig(kind="diagonal"),
        use_triton="auto",
    ).to(device)

    cal = [
        (torch.randn(T, B, H, device=device), torch.randn(B, H, device=device))
        for _ in range(4)
    ]
    layer.calibrate(cal, n_batches=4)
    layer.freeze()

    with torch.no_grad():
        out, _ = layer(cal[0][0], cal[0][1])
    assert torch.isfinite(out).all()
    assert out.shape == (T, B, H)

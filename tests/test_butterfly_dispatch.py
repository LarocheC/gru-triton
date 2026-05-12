"""Butterfly fast-path dispatch tests.

Validates ``GRULayer(use_triton=True, structure_hidden=ButterflyCfg)``:
- Forward parity with the per-step PyTorch path.
- End-to-end QAT (calibrate -> freeze -> forward) runs and produces
  finite output.
- gru_scan_butterfly directly: backward gradients exist on all params.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", message=".*different CUDA versions.*")

import pytest
import torch

torch_structured = pytest.importorskip("torch_structured")

from gru_qat import GRULayer, QuantizerConfig, QuantRecipe, StructureConfig  # noqa: E402
from gru_qat.triton_kernels.scan_butterfly import (  # noqa: E402
    extract_butterfly_factors,
    extract_butterfly_twiddles,
    gru_scan_butterfly,
    gru_scan_butterfly_backward_triton,
    gru_scan_butterfly_forward_triton,
)


cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="butterfly dispatch path is CUDA-only"
)


def _make_layer(
    H: int, *, use_triton: bool, hidden_bits: int = 32
) -> GRULayer:
    rec = QuantRecipe(
        weight=QuantizerConfig(bits=32, axis=0, name="W_id"),
        input_act=QuantizerConfig(bits=32, name="x_id"),
        hidden=QuantizerConfig(bits=hidden_bits, name="h" if hidden_bits < 32 else "h_id"),
    )
    return GRULayer(
        H, H, recipe=rec, gate_layout="fused",
        structure_hidden=StructureConfig(kind="butterfly"),
        use_triton=use_triton,
    )


@cuda_only
@pytest.mark.parametrize("T,B,H", [(8, 4, 32), (16, 8, 64)])
def test_butterfly_dispatch_matches_per_step(T: int, B: int, H: int) -> None:
    """use_triton=True for butterfly must produce the same forward output
    as the PyTorch per-step path (use_triton=False)."""
    torch.manual_seed(0)
    device = torch.device("cuda")

    pt_layer = _make_layer(H, use_triton=False).to(device)
    fast_layer = _make_layer(H, use_triton=True).to(device)
    fast_layer.load_state_dict(pt_layer.state_dict())

    x = torch.randn(T, B, H, device=device) * 0.1
    h0 = torch.randn(B, H, device=device) * 0.1

    pt_out, pt_hT = pt_layer(x, h0)
    fast_out, fast_hT = fast_layer(x, h0)

    rel_out = (pt_out - fast_out).abs().max().item() / max(pt_out.abs().max().item(), 1e-6)
    rel_hT = (pt_hT - fast_hT).abs().max().item() / max(pt_hT.abs().max().item(), 1e-6)
    # Triton-kernel path uses different rounding order from the CUDA-op
    # per-step path; tolerance loose enough to absorb log_H stages × T
    # timesteps of accumulated noise.
    assert rel_out < 1e-1, f"out rel diff {rel_out:.4e}"
    assert rel_hT < 1e-1, f"hT rel diff {rel_hT:.4e}"


@cuda_only
def test_butterfly_grulayer_qat_after_calibration() -> None:
    """End-to-end: train (synthetic), calibrate, freeze, forward via
    butterfly fast path. Output finite, no errors."""
    torch.manual_seed(0)
    device = torch.device("cuda")

    H = 32
    T, B = 8, 16
    layer = _make_layer(H, use_triton=True, hidden_bits=8).to(device)

    def loader(n):
        for _ in range(n):
            yield torch.randn(T, B, H, device=device) * 0.1

    layer.calibrate(loader(8), n_batches=8)
    layer.freeze()

    x = torch.randn(T, B, H, device=device) * 0.1
    out, hT = layer(x)
    assert torch.isfinite(out).all()
    assert out.shape == (T, B, H)
    assert hT.shape == (B, H)


@cuda_only
def test_butterfly_grulayer_dispatch_grad_flows() -> None:
    """Backward must populate gradients on all learnable params when the
    fast dispatch is used (parameters live inside the Butterfly modules
    + dense input weights + biases)."""
    torch.manual_seed(0)
    device = torch.device("cuda")

    H = 32
    T, B = 6, 8
    layer = _make_layer(H, use_triton=True).to(device)

    x = (torch.randn(T, B, H, device=device) * 0.1).requires_grad_()
    h0 = torch.randn(B, H, device=device) * 0.1
    out, _ = layer(x, h0)
    loss = out.float().pow(2).sum()
    loss.backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    # Every learnable parameter that participated in the forward should
    # have a grad tensor populated.
    for name, p in layer.named_parameters():
        if not p.requires_grad:
            continue
        assert p.grad is not None, f"no grad on {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad on {name}"


@cuda_only
@pytest.mark.parametrize("T,B,H", [(8, 16, 32), (16, 32, 64), (8, 32, 128)])
def test_butterfly_triton_forward_matches_per_step(T: int, B: int, H: int) -> None:
    """Multi-step persistent Triton butterfly forward must match the
    per-step CUDA-op path (gru_scan_butterfly with same modules)."""
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    layer = _make_layer(H, use_triton=False).to(device).eval()

    x = torch.randn(T, B, H, device=device) * 0.1
    h0 = torch.randn(B, H, device=device) * 0.1

    with torch.no_grad():
        # Reference: gru_scan_butterfly via the existing CUDA per-step path.
        xq = layer.cell.quant_x(x)
        Wi_cat, bi_cat = layer.cell.quantize_input_weights()
        gi = torch.nn.functional.linear(xq, Wi_cat, bi_cat)
        modules, bh_cat = extract_butterfly_factors(layer.cell)
        ref_out = gru_scan_butterfly(gi, h0, modules, bh_cat)

        # Triton path: same gi/h0/bh, but twiddles flattened for the kernel.
        twiddles, _ = extract_butterfly_twiddles(layer.cell)
        tri_out = gru_scan_butterfly_forward_triton(gi, h0, twiddles, bh_cat)

    max_diff = (ref_out - tri_out).abs().max().item()
    rel = max_diff / max(ref_out.abs().max().item(), 1e-6)
    # CUDA op vs Triton kernel — different rounding order on log_H * T
    # stages plus TF32 in the recurrence; ~0.5-1% drift is normal.
    assert rel < 2e-2, f"butterfly Triton forward rel diff {rel:.4e}"


@cuda_only
def test_butterfly_triton_forward_scratch_oob_regression() -> None:
    """Regression for the scratch-OOB bug at large batch / hidden size.

    The forward kernel previously indexed its per-program scratch slab
    using absolute ``offs_b = pid_b * BLOCK_B + arange(BLOCK_B)`` instead
    of local ``arange(BLOCK_B)``. For ``pid_b > 0`` that shifted scratch
    writes by ``pid_b * BLOCK_B * H_PAD`` floats within the slab; for the
    last program (e.g. pid_b=3 at B=32/BLOCK_B=8) two of its three gate
    stores went past the slab end into whatever neighboring allocation
    lived there. This manifested as silent corruption of nearby tensor
    storages — typically ``rel_max > 1e3`` on bias gradients of any
    layer that happened to live in the affected memory region.

    The fix indexes scratch with ``local_b = arange(BLOCK_B)``. This
    test exercises the (B=32, H=512) shape where the OOB was 8KB —
    if it regresses, the per-batch output at batches 24..31 will
    diverge by an order of magnitude from the per-step reference.
    """
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    T, B, H = 16, 32, 512
    layer = _make_layer(H, use_triton=False).to(device).eval()
    x = torch.randn(T, B, H, device=device) * 0.1
    h0 = torch.randn(B, H, device=device) * 0.1

    with torch.no_grad():
        xq = layer.cell.quant_x(x)
        Wi_cat, bi_cat = layer.cell.quantize_input_weights()
        gi = torch.nn.functional.linear(xq, Wi_cat, bi_cat)
        modules, bh_cat = extract_butterfly_factors(layer.cell)
        ref_out = gru_scan_butterfly(gi, h0, modules, bh_cat)

        twiddles, _ = extract_butterfly_twiddles(layer.cell)
        tri_out = gru_scan_butterfly_forward_triton(gi, h0, twiddles, bh_cat)

    # At H=512 the cuBLAS/torch_structured-butterfly vs Triton tl.dot TF32
    # reduction-order drift accumulates to a few percent over T steps; the
    # OOB-corruption bug was order ~1.0+ relative, so a 5% threshold cleanly
    # distinguishes "fixed" from "regressed". Inspect per-batch maxes so a
    # regression localized to the last program (pid_b=3, batches 24..31)
    # surfaces clearly.
    rel_per_b = (
        (tri_out - ref_out).abs().amax(dim=(0, 2))
        / ref_out.abs().amax().clamp(min=1e-6)
    )
    assert rel_per_b.max().item() < 5e-2, (
        f"butterfly forward max per-batch rel diff {rel_per_b.max().item():.4e} "
        f"exceeds 5e-2; worst batch={rel_per_b.argmax().item()}, "
        f"rel by batch={rel_per_b.tolist()}"
    )


@cuda_only
@pytest.mark.parametrize("T,B,H", [(4, 8, 16), (8, 16, 32), (8, 32, 64)])
def test_butterfly_triton_backward_matches_autograd(T: int, B: int, H: int) -> None:
    """Triton butterfly backward must match autograd gradients through
    the existing gru_scan_butterfly (PyTorch loop + CUDA butterfly_multiply).

    Compares (dgi, dh0, dtwiddles, dbh).
    """
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    layer = _make_layer(H, use_triton=False).to(device)
    x = torch.randn(T, B, H, device=device) * 0.1
    h0 = torch.randn(B, H, device=device) * 0.1

    # Build gi, twiddles, bh_cat. Need to call the CUDA path with
    # parameters set requires_grad to track gradients through it.
    xq = layer.cell.quant_x(x)
    Wi_cat, bi_cat = layer.cell.quantize_input_weights()
    gi_const = torch.nn.functional.linear(xq, Wi_cat, bi_cat).detach()

    # Reference: autograd through gru_scan_butterfly.
    # twiddle shape: [nstacks=1, nblocks, log_n, n//2, 2, 2]. squeeze nstacks
    # (kept in the kernel layout); nblocks is preserved.
    twiddles_ref = (
        torch.stack([
            layer.cell.struct_Wh_r.b.twiddle.squeeze(0).clone(),
            layer.cell.struct_Wh_z.b.twiddle.squeeze(0).clone(),
            layer.cell.struct_Wh_n.b.twiddle.squeeze(0).clone(),
        ], dim=0)
        .detach().requires_grad_()
    )
    bh_ref = (
        torch.cat([layer.cell.b_hr, layer.cell.b_hz, layer.cell.b_hn])
        .detach().clone().requires_grad_()
    )
    gi_ref = gi_const.detach().clone().requires_grad_()
    h0_ref = h0.detach().clone().requires_grad_()

    # Reconstruct three Butterfly modules with the same twiddles for the
    # CUDA-op reference path.
    import torch_structured as ts
    log_H = int.bit_length(H - 1)
    modules = []
    for g in range(3):
        b = ts.Butterfly(H, H, bias=False, init="randn")
        with torch.no_grad():
            b.twiddle.copy_(twiddles_ref[g].view(b.twiddle.shape))
        modules.append(b.to(device))
    # Tie the modules' twiddles to twiddles_ref so backward populates it.
    # Easier: skip the modules' parameter and use the ref directly inside
    # the loop. Build a custom reference that uses twiddles_ref directly.
    from torch_structured.butterfly.multiply import butterfly_multiply
    bh3 = bh_ref.view(3, H)

    def ref_scan(gi, h0, twiddles, bh3):
        out = []
        h = h0
        for t in range(T):
            ghs = []
            for g in range(3):
                # Reshape h to [B, 1, H] for butterfly_multiply (nstacks=1).
                # twiddles[g] is [nblocks, log_H, H/2, 2, 2]; add nstacks=1 axis.
                tw = twiddles[g].unsqueeze(0)  # [1, nblocks, log_H, H/2, 2, 2]
                gh_g = butterfly_multiply(tw, h.unsqueeze(1), True).squeeze(1) + bh3[g]
                ghs.append(gh_g)
            gh_r, gh_z, gh_n = ghs
            gi_r = gi[t, :, 0:H]
            gi_z = gi[t, :, H:2*H]
            gi_n = gi[t, :, 2*H:3*H]
            r = torch.sigmoid(gi_r + gh_r)
            z = torch.sigmoid(gi_z + gh_z)
            n = torch.tanh(gi_n + r * gh_n)
            h = (1 - z) * n + z * h
            out.append(h)
        return torch.stack(out, dim=0)

    ref_out = ref_scan(gi_ref, h0_ref, twiddles_ref, bh_ref.view(3, H))
    dout = torch.randn_like(ref_out) * 0.1
    ref_out.backward(dout)

    # Triton backward path.
    twiddles_tri = twiddles_ref.detach().clone()
    bh_tri = bh_ref.detach().clone()
    out_fwd = gru_scan_butterfly_forward_triton(gi_const, h0, twiddles_tri, bh_tri)
    dgi_t, dh0_t, dtw_t, dbh_t = gru_scan_butterfly_backward_triton(
        gi_const, h0, twiddles_tri, bh_tri, out_fwd, dout,
    )

    for name, t, p in [
        ("dgi", dgi_t, gi_ref.grad),
        ("dh0", dh0_t, h0_ref.grad),
        ("dtwiddles", dtw_t, twiddles_ref.grad),
        ("dbh", dbh_t, bh_ref.grad),
    ]:
        diff = (t - p).abs().max().item()
        rel = diff / max(p.abs().max().item(), 1e-9)
        assert rel < 5e-2, f"{name} rel diff {rel:.4e}"


@cuda_only
@pytest.mark.parametrize("T,B,H", [(8, 16, 32), (16, 32, 64)])
def test_butterfly_grulayer_triton_path_matches_per_step(
    T: int, B: int, H: int
) -> None:
    """GRULayer with use_triton=True (no QAT) routes to the Triton
    butterfly kernels and must produce the same output as use_triton=False."""
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    pt_layer = _make_layer(H, use_triton=False).to(device)
    tri_layer = _make_layer(H, use_triton=True).to(device)
    tri_layer.load_state_dict(pt_layer.state_dict())

    x = torch.randn(T, B, H, device=device) * 0.1
    h0 = torch.randn(B, H, device=device) * 0.1

    pt_out, _ = pt_layer(x, h0)
    tri_out, _ = tri_layer(x, h0)

    rel = (pt_out - tri_out).abs().max().item() / max(pt_out.abs().max().item(), 1e-6)
    assert rel < 5e-3, f"out rel diff {rel:.4e}"


@cuda_only
def test_butterfly_grulayer_triton_path_full_train_step() -> None:
    """Full train step through GRULayer with Triton butterfly path:
    forward + backward + grad on every parameter."""
    torch.manual_seed(0)
    device = torch.device("cuda")

    H, T, B = 32, 8, 16
    layer = _make_layer(H, use_triton=True).to(device)
    x = torch.randn(T, B, H, device=device) * 0.1
    h0 = torch.randn(B, H, device=device) * 0.1
    out, _ = layer(x, h0)
    loss = out.float().pow(2).sum()
    loss.backward()
    for name, p in layer.named_parameters():
        if not p.requires_grad:
            continue
        assert p.grad is not None, f"no grad on {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad on {name}"


@cuda_only
@pytest.mark.parametrize("T,B,H", [(8, 16, 32), (8, 32, 64)])
def test_butterfly_triton_qat_forward_matches_per_step(
    T: int, B: int, H: int
) -> None:
    """In-kernel fake-quant forward: Triton butterfly with quant params
    must match the per-step CUDA-op path with matching h_in/h_out scales."""
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    layer = _make_layer(H, use_triton=False).to(device).eval()
    x = torch.randn(T, B, H, device=device) * 0.1
    h0 = torch.randn(B, H, device=device) * 0.1

    bits = 8
    qmin, qmax = -(2 ** (bits - 1)) + 1, 2 ** (bits - 1) - 1
    h_q = (0.02, qmin, qmax)

    with torch.no_grad():
        xq = layer.cell.quant_x(x)
        Wi_cat, bi_cat = layer.cell.quantize_input_weights()
        gi = torch.nn.functional.linear(xq, Wi_cat, bi_cat)

        # Per-step reference (CUDA op): use gru_scan_butterfly with
        # h_in_quant / h_out_quant kwargs (already supports it).
        modules, bh_cat = extract_butterfly_factors(layer.cell)
        ref_out = gru_scan_butterfly(
            gi, h0, modules, bh_cat,
            h_in_quant=h_q, h_out_quant=h_q,
        )

        # Triton path with the same quant params.
        from gru_qat.triton_kernels.scan_butterfly import gru_scan_butterfly_triton
        twiddles, _ = extract_butterfly_twiddles(layer.cell)
        tri_out = gru_scan_butterfly_triton(
            gi, h0, twiddles, bh_cat,
            h_in_quant=h_q, h_out_quant=h_q,
        )

    rel = (ref_out - tri_out).abs().max().item() / max(ref_out.abs().max().item(), 1e-6)
    assert rel < 1e-1, f"qat forward rel diff {rel:.4e}"


@cuda_only
def test_butterfly_grulayer_qat_calibrate_freeze_triton_path() -> None:
    """End-to-end: build, calibrate, freeze, forward through Triton
    (QAT params get extracted from the frozen quantizers and the kernel
    path runs)."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    H, T, B = 32, 8, 16
    layer = _make_layer(H, use_triton=True, hidden_bits=8).to(device)

    def loader(n):
        for _ in range(n):
            yield torch.randn(T, B, H, device=device) * 0.1

    layer.calibrate(loader(8), n_batches=8)
    layer.freeze()

    x = torch.randn(T, B, H, device=device) * 0.1
    out, hT = layer(x)
    assert torch.isfinite(out).all() and torch.isfinite(hT).all()
    assert out.shape == (T, B, H)


@cuda_only
def test_butterfly_extract_and_gru_scan_directly() -> None:
    """Calling gru_scan_butterfly with factors extracted from a layer
    must produce the same result as routing through GRULayer."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    H = 32
    T, B = 8, 4
    layer = _make_layer(H, use_triton=True).to(device).eval()

    x = torch.randn(T, B, H, device=device) * 0.1
    h0 = torch.randn(B, H, device=device) * 0.1

    with torch.no_grad():
        layer_out, _ = layer(x, h0)

    # Same flow but stitched together by hand.
    with torch.no_grad():
        xq = layer.cell.quant_x(x)
        Wi_cat, bi_cat = layer.cell.quantize_input_weights()
        gi = torch.nn.functional.linear(xq, Wi_cat, bi_cat)
        modules, bh_cat = extract_butterfly_factors(layer.cell)
        manual_out = gru_scan_butterfly(gi, h0, modules, bh_cat)

    rel = (layer_out - manual_out).abs().max().item() / max(layer_out.abs().max().item(), 1e-6)
    # use_triton=True now routes through the Triton kernel; manual_out
    # uses the CUDA-op per-step path. Different rounding order, loose
    # tolerance.
    assert rel < 1e-1, f"manual vs layer rel diff {rel:.4e}"

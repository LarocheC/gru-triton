"""Strict-tier parity tests for the dense Triton scan kernel — Phase 2 audit.

Validates ``gru_scan`` / ``gru_scan_forward`` / ``gru_scan_persistent`` (and
their fwd/bwd helpers) against the Phase 1 reference path
(``GRULayer(use_triton=False, dense, Identity quantizers)``) at the strict
tier::

    torch.set_float32_matmul_precision('highest')      # IEEE fp32 matmul
    assert (triton - reference).abs().max() < 1e-5     # absolute, not relative

Diverges intentionally from ``tests/test_triton_scan.py`` (which runs under
``'high'`` / TF32 with 5e-3..1e-1 relative bounds — that's the
realistic-deployment tier). Both files coexist; this file does NOT loosen the
existing one (D-20). The realistic-tier sibling is the deployment regime; this
file audits the math.

Also hosts in the same module:

- TRI-05 regression (``test_autotune_dWh_dbh_zero_init_across_configs``) — the
  autotune-config rotation of the slab-zero bug fixed in commit ``c001a8a``.
  Existing single-config regression lives at ``tests/test_triton_scan.py:202-215``.
- TRI-06 regression (``test_persistent_kernel_deterministic``) — 50-run
  bit-identical guard for the release/acquire cross-CTA fence
  (see ``src/gru_qat/triton_kernels/scan.py:184-208``).
- D-25 static canary (``test_no_cv_cache_modifier_live_uses_in_scan_source``)
  — asserts ``cache_modifier=".cv"`` does not appear in any *live*
  (non-comment) line of ``src/gru_qat/triton_kernels/scan*.py``.

The cell-parity contract in ``tests/test_parity.py`` and the layer-parity
contract in ``tests/test_layer_parity.py`` are LOCKED by D-28 and are NOT
duplicated here.
"""

from __future__ import annotations

import pathlib  # noqa: F401  (used by Task 3 D-25 static canary, appended below)

import pytest
import torch
import torch.nn as nn  # noqa: F401  (imported for parity with TF32 sibling)

triton = pytest.importorskip("triton")

from gru_qat.gru_layer import GRULayer  # noqa: E402
from gru_qat.quantizers import QuantizerConfig, QuantRecipe  # noqa: E402
from gru_qat.triton_kernels.scan import (  # noqa: E402
    gru_scan,
    gru_scan_forward,
    gru_scan_forward_persistent,  # noqa: F401  (imported for symmetry with sibling)
    gru_scan_backward_persistent,  # noqa: F401  (imported for symmetry with sibling)
    gru_scan_persistent,  # noqa: F401  (used by Task 2 TRI-06 regression, appended below)
    _gru_scan_backward_pytorch,  # noqa: F401  (imported for symmetry with sibling)
)

# Strict tier: IEEE-754 fp32 matmul, not TF32. The realistic-tier sibling
# file (tests/test_triton_scan.py) uses 'high' to exercise the kernel under
# deployment conditions; this file audits the math.
torch.set_float32_matmul_precision("highest")

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernel requires CUDA"
)


# duplicated per D-18 (< 30 LOC, inline beats shared module)
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


# Per CONTEXT D-16 dense grid. FAST set runs on every ``pytest -q``; SLOW set
# (T ∈ {512, 1024}) is gated behind ``@pytest.mark.slow``.
FAST_DENSE_GRID = [
    (T, B, H)
    for T in (1, 8, 64)
    for B in (1, 4, 32)
    for H in (32, 128, 512)
]  # 27 cases

SLOW_DENSE_GRID = [
    (T, B, H)
    for T in (512, 1024)
    for B in (1, 4, 32)
    for H in (32, 128, 512)
]  # 18 cases


@cuda_only
@pytest.mark.parametrize("T,B,H", FAST_DENSE_GRID)
def test_scan_fwd_strict_matches_reference(T: int, B: int, H: int) -> None:
    """``gru_scan_forward`` must match the reference GRULayer to < 1e-5
    absolute under ``'highest'`` precision. fp32 IEEE matmul on both sides —
    algorithmic drift only.

    Realistic-tier sibling (tests/test_triton_scan.py:139) uses < 5e-3 under
    TF32; that's correct for its regime and not loosened by this file.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    IN = H
    layer = _ref_layer(IN, H).to(device).eval()

    x = torch.randn(T, B, IN, device=device)
    h0 = torch.randn(B, H, device=device)

    with torch.no_grad():
        ref_out, _ = layer(x, h0)
        w = layer.cell.quantize_weights()
        gi = layer.cell.input_projection(x, w)
        assert w.Wh_cat is not None and w.bh_cat is not None
        triton_out = gru_scan_forward(gi, h0, w.Wh_cat, w.bh_cat)

    max_diff = (ref_out - triton_out).abs().max().item()
    assert max_diff < 1e-5, (
        f"max abs diff {max_diff:.4e} (T={T},B={B},H={H})"
    )


@pytest.mark.slow
@cuda_only
@pytest.mark.parametrize("T,B,H", SLOW_DENSE_GRID)
def test_scan_fwd_strict_matches_reference_slow(T: int, B: int, H: int) -> None:
    """Slow sibling of ``test_scan_fwd_strict_matches_reference`` over
    SLOW_DENSE_GRID (T ∈ {512, 1024}). Gated behind ``@pytest.mark.slow``."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    IN = H
    layer = _ref_layer(IN, H).to(device).eval()

    x = torch.randn(T, B, IN, device=device)
    h0 = torch.randn(B, H, device=device)

    with torch.no_grad():
        ref_out, _ = layer(x, h0)
        w = layer.cell.quantize_weights()
        gi = layer.cell.input_projection(x, w)
        assert w.Wh_cat is not None and w.bh_cat is not None
        triton_out = gru_scan_forward(gi, h0, w.Wh_cat, w.bh_cat)

    max_diff = (ref_out - triton_out).abs().max().item()
    assert max_diff < 1e-5, (
        f"max abs diff {max_diff:.4e} (T={T},B={B},H={H})"
    )


@cuda_only
@pytest.mark.parametrize("T,B,H", FAST_DENSE_GRID)
def test_scan_bwd_strict_matches_reference(T: int, B: int, H: int) -> None:
    """Triton autograd gradients must match PyTorch autograd through the
    reference layer to < 1e-5 absolute on x, h0, Wh_cat, bh_cat under
    ``'highest'`` precision.

    Realistic-tier sibling (tests/test_triton_scan.py:215) uses rel < 1e-1
    under TF32; this file's absolute < 1e-5 is the audit bound.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    IN = H

    ref_layer = _ref_layer(IN, H).to(device)
    x = torch.randn(T, B, IN, device=device, requires_grad=True)
    h0 = torch.randn(B, H, device=device, requires_grad=True)

    # Reference path: PyTorch autograd through the layer.
    ref_x = x.detach().clone().requires_grad_()
    ref_h0 = h0.detach().clone().requires_grad_()
    ref_out, _ = ref_layer(ref_x, ref_h0)
    ref_loss = ref_out.float().pow(2).sum()
    ref_loss.backward()

    # Triton path: pre-batch input projection (autograd-aware), then gru_scan.
    w = ref_layer.cell.quantize_weights()
    Wi_cat = w.Wi_cat.detach().clone()
    bi_cat = w.bi_cat.detach().clone()
    Wh_cat = w.Wh_cat.detach().clone().requires_grad_()
    bh_cat = w.bh_cat.detach().clone().requires_grad_()
    tri_x = x.detach().clone().requires_grad_()
    tri_h0 = h0.detach().clone().requires_grad_()
    gi = torch.nn.functional.linear(tri_x, Wi_cat, bi_cat)
    out = gru_scan(gi, tri_h0, Wh_cat, bh_cat)
    out.float().pow(2).sum().backward()

    # Reconstruct the reference dWh_cat / dbh_cat by concatenating per-gate
    # grads in the same order quantize_weights() builds Wh_cat (r, z, n).
    ref_dWh_cat = torch.cat(
        [ref_layer.cell.W_hr.grad, ref_layer.cell.W_hz.grad, ref_layer.cell.W_hn.grad],
        dim=0,
    )
    ref_dbh_cat = torch.cat(
        [ref_layer.cell.b_hr.grad, ref_layer.cell.b_hz.grad, ref_layer.cell.b_hn.grad],
        dim=0,
    )

    for name, ref_g, tri_g in [
        ("x", ref_x.grad, tri_x.grad),
        ("h0", ref_h0.grad, tri_h0.grad),
        ("Wh_cat", ref_dWh_cat, Wh_cat.grad),
        ("bh_cat", ref_dbh_cat, bh_cat.grad),
    ]:
        assert ref_g is not None and tri_g is not None
        max_diff = (ref_g - tri_g).abs().max().item()
        assert max_diff < 1e-5, (
            f"{name} max abs diff {max_diff:.4e} (T={T},B={B},H={H})"
        )


@pytest.mark.slow
@cuda_only
@pytest.mark.parametrize("T,B,H", SLOW_DENSE_GRID)
def test_scan_bwd_strict_matches_reference_slow(T: int, B: int, H: int) -> None:
    """Slow sibling of ``test_scan_bwd_strict_matches_reference`` over
    SLOW_DENSE_GRID (T ∈ {512, 1024})."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    IN = H

    ref_layer = _ref_layer(IN, H).to(device)
    x = torch.randn(T, B, IN, device=device, requires_grad=True)
    h0 = torch.randn(B, H, device=device, requires_grad=True)

    ref_x = x.detach().clone().requires_grad_()
    ref_h0 = h0.detach().clone().requires_grad_()
    ref_out, _ = ref_layer(ref_x, ref_h0)
    ref_loss = ref_out.float().pow(2).sum()
    ref_loss.backward()

    w = ref_layer.cell.quantize_weights()
    Wi_cat = w.Wi_cat.detach().clone()
    bi_cat = w.bi_cat.detach().clone()
    Wh_cat = w.Wh_cat.detach().clone().requires_grad_()
    bh_cat = w.bh_cat.detach().clone().requires_grad_()
    tri_x = x.detach().clone().requires_grad_()
    tri_h0 = h0.detach().clone().requires_grad_()
    gi = torch.nn.functional.linear(tri_x, Wi_cat, bi_cat)
    out = gru_scan(gi, tri_h0, Wh_cat, bh_cat)
    out.float().pow(2).sum().backward()

    ref_dWh_cat = torch.cat(
        [ref_layer.cell.W_hr.grad, ref_layer.cell.W_hz.grad, ref_layer.cell.W_hn.grad],
        dim=0,
    )
    ref_dbh_cat = torch.cat(
        [ref_layer.cell.b_hr.grad, ref_layer.cell.b_hz.grad, ref_layer.cell.b_hn.grad],
        dim=0,
    )

    for name, ref_g, tri_g in [
        ("x", ref_x.grad, tri_x.grad),
        ("h0", ref_h0.grad, tri_h0.grad),
        ("Wh_cat", ref_dWh_cat, Wh_cat.grad),
        ("bh_cat", ref_dbh_cat, bh_cat.grad),
    ]:
        assert ref_g is not None and tri_g is not None
        max_diff = (ref_g - tri_g).abs().max().item()
        assert max_diff < 1e-5, (
            f"{name} max abs diff {max_diff:.4e} (T={T},B={B},H={H})"
        )

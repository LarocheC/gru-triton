"""Structured-input + Triton-hidden fast-path fusion.

The input projection ``W_i·x`` is identical at every timestep, so it can be
hoisted out of the recurrence regardless of how the input weight is
parameterized. This file verifies that a *structured* input weight no longer
disables the Triton hidden kernel: the layer hoists the structured input
projection to a batched GEMM (``cell.structured_input_projection``), feeds the
resulting dense ``gi: [T, B, 3H]`` into the existing monarch/diagonal kernel,
and matches the per-step ``step_structured`` reference.

Gates the feature added in bd gru-triton (structured-input/Triton-hidden fusion).
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", message=".*different CUDA versions.*")

import pytest
import torch

pytest.importorskip("triton")
pytest.importorskip("torch_structured")

from gru_qat import GRULayer, QuantRecipe, QuantizerConfig, StructureConfig  # noqa: E402

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernel requires CUDA"
)


def _fp32_recipe() -> QuantRecipe:
    """Identity quantizers everywhere — keeps the parity math clean."""
    return QuantRecipe(
        weight=QuantizerConfig(bits=32, axis=0, name="W_id"),
        input_act=QuantizerConfig(bits=32, name="x_id"),
        hidden=QuantizerConfig(bits=32, name="h_id"),
    )


def test_structured_input_is_fast_dispatch_eligible() -> None:
    """A structured input weight no longer disables the Triton fast path;
    it stays eligible as long as the hidden kind is kernel-eligible and the
    gate layout is fused. (Construction-only — runs without CUDA.)"""
    layer = GRULayer(
        32, 32, recipe=_fp32_recipe(), gate_layout="fused",
        structure_input=StructureConfig(kind="monarch", nblocks=4),
        structure_hidden=StructureConfig(kind="monarch", nblocks=4),
        use_triton="auto",
    )
    assert layer._fast_dispatch_eligible is True
    assert layer.use_triton is True
    # And explicit use_triton=True must NOT raise just because input is structured.
    GRULayer(
        32, 32, recipe=_fp32_recipe(), gate_layout="fused",
        structure_input=StructureConfig(kind="butterfly"),
        structure_hidden=StructureConfig(kind="diagonal"),
        use_triton=True,
    )


@cuda_only
@pytest.mark.parametrize("input_kind", ["monarch", "butterfly", "diagonal"])
@pytest.mark.parametrize("hidden_kind", ["diagonal", "monarch"])
@pytest.mark.parametrize("T,B,H", [(8, 4, 32), (16, 8, 64)])
def test_structured_input_triton_matches_per_step_reference(
    input_kind: str, hidden_kind: str, T: int, B: int, H: int
) -> None:
    """use_triton=True (hoisted structured input + Triton hidden kernel)
    must match use_triton=False (per-step step_structured) forward and
    backward, under a matched fp32 recipe with shared weights.

    The input projection is identical fp32 in both paths (batched vs
    per-step row-wise matmul), so the only difference is the hidden
    recurrence: kernel vs PyTorch. Tolerances follow the documented
    per-kernel parity (diagonal ~machine eps, monarch ~TF32 tl.dot)."""
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    # The monarch backward kernel needs blksz = H/nblocks >= 16 (tl.dot's
    # K >= 16 constraint) — on RTX 2000 Ada blksz=8 (H=32, nb=4) OOMs / fails
    # to compile. The strict monarch suite skips the same configs (gru-triton-e0l).
    if hidden_kind == "monarch" and (H // 4) < 16:
        pytest.skip("monarch bwd kernel requires blksz>=16 on this GPU (gru-triton-e0l)")

    in_cfg = (
        StructureConfig(kind="monarch", nblocks=4)
        if input_kind == "monarch"
        else StructureConfig(kind=input_kind)
    )
    hid_cfg = (
        StructureConfig(kind="monarch", nblocks=4)
        if hidden_kind == "monarch"
        else StructureConfig(kind="diagonal")
    )

    def build(use_triton: bool) -> GRULayer:
        return GRULayer(
            H, H, recipe=_fp32_recipe(), gate_layout="fused",
            structure_input=in_cfg, structure_hidden=hid_cfg,
            use_triton=use_triton,
        ).to(device)

    ref = build(False)
    tri = build(True)
    tri.load_state_dict(ref.state_dict())
    assert tri.use_triton is True and ref.use_triton is False

    x_ref = (torch.randn(T, B, H, device=device) * 0.1).requires_grad_(True)
    x_tri = x_ref.detach().clone().requires_grad_(True)
    h0 = torch.randn(B, H, device=device) * 0.1

    ref_out, ref_hT = ref(x_ref, h0)
    tri_out, tri_hT = tri(x_tri, h0)

    denom = max(ref_out.abs().max().item(), 1e-6)
    rel_out = (ref_out - tri_out).abs().max().item() / denom
    rel_hT = (ref_hT - tri_hT).abs().max().item() / max(ref_hT.abs().max().item(), 1e-6)
    assert rel_out < 5e-3, f"fwd rel diff {rel_out:.4e} ({input_kind}->{hidden_kind})"
    assert rel_hT < 5e-3, f"hT rel diff {rel_hT:.4e} ({input_kind}->{hidden_kind})"

    # Backward must flow through the hoisted structured input projection.
    ref_out.float().pow(2).sum().backward()
    tri_out.float().pow(2).sum().backward()
    assert x_ref.grad is not None and x_tri.grad is not None
    g_denom = max(x_ref.grad.abs().max().item(), 1e-6)
    rel_dx = (x_ref.grad - x_tri.grad).abs().max().item() / g_denom
    assert rel_dx < 2e-2, f"dx rel diff {rel_dx:.4e} ({input_kind}->{hidden_kind})"


@cuda_only
def test_structured_input_qat_after_calibration_runs() -> None:
    """End-to-end QAT smoke: structured input + monarch hidden, int8 hidden
    quant, calibrate -> freeze -> forward through the Triton path."""
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    H, T, B = 32, 8, 16

    rec = QuantRecipe(
        weight=QuantizerConfig(bits=32, axis=0, name="W_id"),
        input_act=QuantizerConfig(bits=32, name="x_id"),
        hidden=QuantizerConfig(bits=8, name="h_q"),  # int8 hidden
    )
    layer = GRULayer(
        H, H, recipe=rec, gate_layout="fused",
        structure_input=StructureConfig(kind="monarch", nblocks=4),
        structure_hidden=StructureConfig(kind="monarch", nblocks=4),
        use_triton=True,
    ).to(device)

    def loader(n: int):
        for _ in range(n):
            yield torch.randn(T, B, H, device=device) * 0.1

    layer.calibrate(loader(8), n_batches=8)
    layer.freeze()

    out, hT = layer(torch.randn(T, B, H, device=device) * 0.1)
    assert torch.isfinite(out).all()
    assert out.shape == (T, B, H)
    assert hT.shape == (B, H)

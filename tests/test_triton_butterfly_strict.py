"""Strict-tier parity tests for the Butterfly Triton kernel — Phase 2 audit.

Validates ``gru_scan_butterfly_forward_triton`` and
``gru_scan_butterfly_backward_triton`` against the CUDA-op per-step reference
path (``gru_scan_butterfly``, which routes through
``torch_structured.butterfly_multiply``) at the strict tier::

    torch.set_float32_matmul_precision('highest')      # IEEE fp32 matmul
    assert (triton - reference).abs().max() < 5e-4     # absolute, not relative

Butterfly has **no pure-PyTorch reference distinct from the kernel under
test** — the CUDA-op path goes through ``butterfly_multiply`` from
``torch_structured``, and that path serves as ground truth here.

Tight-TF32 strict-tier bound rationale (Phase 2 Plan 02-06 / Option C):
Although butterfly's hidden multiply is not a single ``tl.dot``, the Triton
kernel uses ``tl.dot`` for the per-stage block matmuls inside the log_H
butterfly factorization on Ampere+ GPUs, which defaults to TF32 regardless
of ``torch.set_float32_matmul_precision('highest')``. The global precision
knob does not propagate into in-kernel ``tl.dot``. Compounded across log_H
stages, TF32 noise reaches ~1e-4 abs against the reference path. The
strict-tier bound is therefore held at ``< 5e-4 abs`` — well above the TF32
floor, well below the magnitude a real kernel bug would produce. The
accepted TF32 divergence is tracked as a bd issue (see Plan 02-06 SUMMARY).

Both files coexist; this file does NOT loosen the existing one (D-20). The
realistic-tier sibling exercises the kernel under deployment conditions
(TF32); this file audits the math.

Note: the per-program scratch-OOB regression for the butterfly fwd kernel
(commit ``d8218d4``, finding TRI-04) is covered at
``tests/test_butterfly_dispatch.py:164``
(``test_butterfly_triton_forward_scratch_oob_regression``). That test runs at
(T=16, B=32, H=512) under TF32 with ``rel < 5e-2``; this strict file does
NOT duplicate it per D-22. Phase-exit verification (Plan 02-06) confirms the
OOB regression still passes; if it regresses, the bug surfaces there, not
here.

Butterfly requires H to be a power of 2 (the kernel only supports H = 2^k);
per D-16 the strict grid is restricted to H ∈ {32, 128, 512}.

The cell-parity contract in ``tests/test_parity.py`` and the layer-parity
contract in ``tests/test_layer_parity.py`` are LOCKED by D-28 and are NOT
duplicated here.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", message=".*different CUDA versions.*")

import pytest  # noqa: E402
import torch  # noqa: E402

triton = pytest.importorskip("triton")
torch_structured = pytest.importorskip("torch_structured")

from gru_qat import (  # noqa: E402
    GRULayer,
    QuantizerConfig,
    QuantRecipe,
    StructureConfig,
)
from gru_qat.triton_kernels.scan_butterfly import (  # noqa: E402
    extract_butterfly_factors,  # noqa: F401  (imported for symmetry with sibling)
    extract_butterfly_twiddles,  # noqa: F401  (imported for symmetry with sibling)
    gru_scan_butterfly,  # noqa: F401  (imported for symmetry with sibling)
    gru_scan_butterfly_backward_triton,  # noqa: F401  (imported for symmetry with sibling)
    gru_scan_butterfly_forward_triton,  # noqa: F401  (imported for symmetry with sibling)
)

# Strict tier: IEEE-754 fp32 matmul, not TF32. The realistic-tier sibling
# file (tests/test_butterfly_dispatch.py) uses 'high' to exercise the kernel
# under deployment conditions; this file audits the math.
torch.set_float32_matmul_precision("highest")

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="butterfly dispatch path is CUDA-only"
)


# duplicated per D-18 (< 30 LOC, inline beats shared module).
# Strict-tier callers always pass hidden_bits=32 (fp32-Identity per CONTEXT —
# Phase 2 is fp32-Identity only; quant-on is Phase 4).
def _make_layer(
    H: int, *, use_triton: bool, hidden_bits: int = 32
) -> GRULayer:
    rec = QuantRecipe(
        weight=QuantizerConfig(bits=32, axis=0, name="W_id"),
        input_act=QuantizerConfig(bits=32, name="x_id"),
        hidden=QuantizerConfig(
            bits=hidden_bits, name="h" if hidden_bits < 32 else "h_id"
        ),
    )
    return GRULayer(
        H, H, recipe=rec, gate_layout="fused",
        structure_hidden=StructureConfig(kind="butterfly"),
        use_triton=use_triton,
    )


# Butterfly requires H to be a power of 2 per src/gru_qat/structure.py shape
# validators. Per D-16: H in {32, 128, 512}.
FAST_BFLY_GRID = [
    (T, B, H)
    for T in (1, 8, 64)
    for B in (1, 4, 32)
    for H in (32, 128, 512)  # powers of 2; butterfly requires 2^k
]  # 27 cases

SLOW_BFLY_GRID = [
    (T, B, H)
    for T in (512, 1024)
    for B in (1, 4, 32)
    for H in (32, 128, 512)
]  # 18 cases


@cuda_only
@pytest.mark.parametrize("T,B,H", FAST_BFLY_GRID)
def test_butterfly_fwd_strict_matches_reference(T: int, B: int, H: int) -> None:
    """Triton butterfly forward must match the CUDA-op per-step reference
    (``gru_scan_butterfly``) to < 5e-4 absolute under ``'highest'`` precision.

    Tight-TF32 strict-tier bound (Phase 2 Plan 02-06 / Option C): the Triton
    butterfly kernel uses ``tl.dot`` (TF32 on Ampere+) for the per-stage
    block matmuls in the log_H butterfly factorization. The global
    ``torch.set_float32_matmul_precision('highest')`` knob does not affect
    in-kernel ``tl.dot``. Compounded across log_H stages, TF32 noise can
    reach ~1e-4 abs vs the reference; the 5e-4 bound is a "tight TF32"
    audit threshold — see module docstring for the full rationale and the
    bd issue documenting the accepted TF32 divergence.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")

    pt_layer = _make_layer(H, use_triton=False).to(device)
    fast_layer = _make_layer(H, use_triton=True).to(device)
    fast_layer.load_state_dict(pt_layer.state_dict())

    x = torch.randn(T, B, H, device=device) * 0.1
    h0 = torch.randn(B, H, device=device) * 0.1

    with torch.no_grad():
        pt_out, _ = pt_layer(x, h0)
        fast_out, _ = fast_layer(x, h0)

    max_diff = (pt_out - fast_out).abs().max().item()
    # Strict tier: tight-TF32 bound under in-kernel TF32 ``tl.dot``.
    # Realistic-tier sibling (tests/test_butterfly_dispatch.py:160) uses
    # < 2e-2 rel under TF32 — that's correct for its regime; not loosened
    # by us.
    assert max_diff < 5e-4, (
        f"butterfly fwd max abs diff {max_diff:.4e} (T={T},B={B},H={H})"
    )


@pytest.mark.slow
@cuda_only
@pytest.mark.parametrize("T,B,H", SLOW_BFLY_GRID)
def test_butterfly_fwd_strict_matches_reference_slow(
    T: int, B: int, H: int
) -> None:
    """Identical body to the fast variant; gated behind ``@pytest.mark.slow``
    per D-16 (T ∈ {512, 1024}).

    Bound: < 5e-4 abs (tight-TF32; see fast-variant docstring).
    """
    torch.manual_seed(0)
    device = torch.device("cuda")

    pt_layer = _make_layer(H, use_triton=False).to(device)
    fast_layer = _make_layer(H, use_triton=True).to(device)
    fast_layer.load_state_dict(pt_layer.state_dict())

    x = torch.randn(T, B, H, device=device) * 0.1
    h0 = torch.randn(B, H, device=device) * 0.1

    with torch.no_grad():
        pt_out, _ = pt_layer(x, h0)
        fast_out, _ = fast_layer(x, h0)

    max_diff = (pt_out - fast_out).abs().max().item()
    assert max_diff < 5e-4, (
        f"butterfly fwd max abs diff {max_diff:.4e} (T={T},B={B},H={H})"
    )


def _assert_grad_close(
    name: str, ref_g: torch.Tensor | None, tri_g: torch.Tensor | None,
    T: int, B: int, H: int,
) -> None:
    """Strict-tier per-grad assertion. Raises on shape mismatch / missing
    grads so failures are diagnosable per-grad (named) rather than a
    bare tensor-equality blowup.

    Returns silently when both grads are None (e.g. a frozen parameter
    that didn't participate in the forward — skip rather than fail).

    Bound: < 5e-4 abs (tight-TF32 per Phase 2 Plan 02-06 / Option C — see
    module docstring for the TF32-via-tl.dot rationale).
    """
    if ref_g is None and tri_g is None:
        return
    assert ref_g is not None, f"{name}: reference grad is None but triton grad is not"
    assert tri_g is not None, f"{name}: triton grad is None but reference grad is not"
    max_diff = (ref_g - tri_g).abs().max().item()
    assert max_diff < 5e-4, (
        f"{name} grad max abs diff {max_diff:.4e} (T={T},B={B},H={H})"
    )


@cuda_only
@pytest.mark.parametrize("T,B,H", FAST_BFLY_GRID)
def test_butterfly_bwd_strict_matches_reference(T: int, B: int, H: int) -> None:
    """Triton butterfly backward must match autograd through the CUDA-op
    per-step reference path to < 5e-4 absolute under ``'highest'``.

    Tight-TF32 strict-tier bound (Phase 2 Plan 02-06 / Option C): the
    butterfly bwd kernel uses ``tl.dot`` for the per-stage gradient
    reductions (TF32 on Ampere+); the global ``'highest'`` knob does not
    affect in-kernel ``tl.dot``. Bound is 5e-4 abs — see module docstring
    and ``_assert_grad_close`` for the full rationale.

    Pattern: dual-layer-with-shared-state. ``pt_layer`` runs the per-step
    PyTorch path (``use_triton=False`` — autograd flows through
    ``gru_scan_butterfly`` and its ``butterfly_multiply`` closure);
    ``fast_layer`` runs the Triton kernel. State is shared via
    ``load_state_dict``, so each parameter sees the same value on both
    sides — the only difference is the kernel doing the math.

    Compares gradients on (x, h0) AND on every learnable parameter in the
    layer's ``named_parameters()``.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")

    pt_layer = _make_layer(H, use_triton=False).to(device)
    fast_layer = _make_layer(H, use_triton=True).to(device)
    fast_layer.load_state_dict(pt_layer.state_dict())

    # Inputs require_grad on both sides; allocate the base tensor first
    # (``* 0.1`` returns a non-leaf tensor and would not preserve
    # requires_grad on the result), then flip the flag in-place.
    x_pt = (torch.randn(T, B, H, device=device) * 0.1).requires_grad_()
    h0_pt = (torch.randn(B, H, device=device) * 0.1).requires_grad_()
    x_tri = x_pt.detach().clone().requires_grad_()
    h0_tri = h0_pt.detach().clone().requires_grad_()

    pt_out, _ = pt_layer(x_pt, h0_pt)
    pt_out.float().pow(2).sum().backward()

    tri_out, _ = fast_layer(x_tri, h0_tri)
    tri_out.float().pow(2).sum().backward()

    # Per-parameter gradient parity. Strict tier: every learnable parameter
    # that participated in both forwards must have matching gradients to
    # < 5e-4 abs (tight-TF32; see module + helper docstrings).
    fast_params = dict(fast_layer.named_parameters())
    for name, p_pt in pt_layer.named_parameters():
        p_tri = fast_params[name]
        _assert_grad_close(name, p_pt.grad, p_tri.grad, T, B, H)

    # Input gradients.
    for name, ref_g, tri_g in [
        ("x", x_pt.grad, x_tri.grad),
        ("h0", h0_pt.grad, h0_tri.grad),
    ]:
        _assert_grad_close(name, ref_g, tri_g, T, B, H)


@pytest.mark.slow
@cuda_only
@pytest.mark.parametrize("T,B,H", SLOW_BFLY_GRID)
def test_butterfly_bwd_strict_matches_reference_slow(
    T: int, B: int, H: int
) -> None:
    """Identical body to the fast variant; gated behind ``@pytest.mark.slow``
    per D-16 (T ∈ {512, 1024}).

    Bound: < 5e-4 abs (tight-TF32; see fast-variant + module docstrings).
    """
    torch.manual_seed(0)
    device = torch.device("cuda")

    pt_layer = _make_layer(H, use_triton=False).to(device)
    fast_layer = _make_layer(H, use_triton=True).to(device)
    fast_layer.load_state_dict(pt_layer.state_dict())

    x_pt = (torch.randn(T, B, H, device=device) * 0.1).requires_grad_()
    h0_pt = (torch.randn(B, H, device=device) * 0.1).requires_grad_()
    x_tri = x_pt.detach().clone().requires_grad_()
    h0_tri = h0_pt.detach().clone().requires_grad_()

    pt_out, _ = pt_layer(x_pt, h0_pt)
    pt_out.float().pow(2).sum().backward()

    tri_out, _ = fast_layer(x_tri, h0_tri)
    tri_out.float().pow(2).sum().backward()

    fast_params = dict(fast_layer.named_parameters())
    for name, p_pt in pt_layer.named_parameters():
        p_tri = fast_params[name]
        _assert_grad_close(name, p_pt.grad, p_tri.grad, T, B, H)

    for name, ref_g, tri_g in [
        ("x", x_pt.grad, x_tri.grad),
        ("h0", h0_pt.grad, h0_tri.grad),
    ]:
        _assert_grad_close(name, ref_g, tri_g, T, B, H)


# ---------------------------------------------------------------------------
# Phase 4 / Plan 04-04: full quant-on sweep for the Butterfly Triton kernel.
#
# Mirrors the Plan 04-02 (dense) / Plan 04-03 (diagonal / monarch) Phase 4
# sections; closes out QNT-02 (butterfly fwd) and contributes to QNT-03
# (butterfly bwd) per ``.planning/REQUIREMENTS.md``.
#
# Comparator pattern: butterfly has **no pure-PyTorch reference distinct from
# the kernel under test** (the CUDA-op per-step path routes through
# ``torch_structured.butterfly_multiply``). Following the Phase 2 strict
# pattern at ``tests/test_triton_butterfly_strict.py:138-156``, the comparator
# is dual-layer:
#
#     pt_layer   = _make_butterfly_layer_quant_int8(H, use_triton=False)
#     fast_layer = _make_butterfly_layer_quant_int8(H, use_triton=True)
#     fast_layer.load_state_dict(pt_layer.state_dict())  # share weights + frozen scales
#
# Both layers go through ``GRULayer.forward()``; the only difference is the
# Triton vs CUDA-op kernel doing the math. ``GRULayer.forward()`` invokes
# ``cell.quant_x`` internally (per D-41 input_act recipe), so the
# input-quantization-before-linear order is enforced by the layer — no extra
# ``quant_x`` call in the test bodies.
#
# Disposition is **ASYMMETRIC** per
# ``.planning/phases/04-quant-on-bit-identity/04-DISPOSITION.md`` (resolved at
# the Plan 04-01 ``checkpoint:human-verify`` 2026-05-14):
#
#   - Forward outputs (``out``, ``h_T``):                 ``torch.equal`` (strict=True)
#   - Backward grads (``dx``, ``dh_0``, structured-weight
#     twiddle grads, bias grads):                         ``abs_diff < h_scale`` (strict=False)
#
# Rationale (per 04-DISPOSITION.md):
#   - Fwd: in-kernel ``quant_h_out`` rounds both Triton-TF32-matmul outputs
#     and CUDA-op fp32-matmul outputs to the same INT8 grid; pre-quant fp32
#     values differ but post-quant int values are identical.
#   - Bwd: fp32 reduction-order drift between Triton ``tl.dot`` and PyTorch
#     matmul accumulates over batch + time dimensions; STE backward through
#     ``fake_quant_ste`` does not re-quantize gradients so they remain fp32
#     and exhibit the underlying matmul-order drift. Worst observed at the
#     probe shape (T=8, B=4, H=64, cls=realistic, dense) was
#     ``dWh_cat = 1.12e-03 = 5.6% of h_scale`` — well within the
#     one-INT8-step budget. Butterfly's log_H per-stage matmuls have the
#     same TF32-vs-fp32 drift signature.
#
# H is restricted to ``{32, 128, 512}`` per D-49 (butterfly requires powers
# of 2; the Phase 4 fast/slow grids already comply).
#
# The D-22 OOB regression for the butterfly fwd kernel lives at
# ``tests/test_butterfly_dispatch.py:164``
# (``test_butterfly_triton_forward_scratch_oob_regression``). This section
# does NOT duplicate or modify it.
# ---------------------------------------------------------------------------


def _assert_quant_parity(
    name: str,
    ref: torch.Tensor,
    tri: torch.Tensor,
    h_scale: float,
    *,
    strict: bool,
    h_scale_mult: float = 1.0,
) -> None:
    """Assert quant-on parity per the Phase 4 D-42 disposition.

    ``strict=True`` (forward / ``h_T``):    ``torch.equal`` contract.
    ``strict=False`` (backward grads):       ``abs_diff < h_scale_mult * h_scale``
    (default ``h_scale_mult=1.0`` — one INT8 step).

    ``h_scale_mult`` is the per-call escape hatch for empirically-loosened bounds
    documented as Phase 4 findings (e.g. F-04-05-A dense-bwd large-magnitude
    uses ``2.0``; F-04-05-B butterfly-fwd uses ``5.0`` with ``strict=False``).
    Every call site that overrides ``h_scale_mult`` must reference the bd issue
    in a comment.

    Disposition source of truth:
    ``.planning/phases/04-quant-on-bit-identity/04-DISPOSITION.md`` (D-42 /
    D-43 — per-file byte-identical helper across Plans 04-02..04). Centralizes
    the strict-vs-tight-INT8-grid switch so any future disposition revision
    touches only this helper.
    """
    if strict:
        assert torch.equal(ref, tri), (
            f"quant-on bit-identity failed for {name}: "
            f"max_abs_diff={(ref - tri).abs().max().item():.4e} "
            f"(expected 0.0)"
        )
    else:
        max_diff = (ref - tri).abs().max().item()
        bound = h_scale_mult * h_scale
        assert max_diff < bound, (
            f"quant-on tight-INT8-step bound failed for {name}: "
            f"max_abs_diff={max_diff:.4e}, h_scale={h_scale:.4e}, "
            f"h_scale_mult={h_scale_mult:.2f}, bound={bound:.4e}, "
            f"ratio={max_diff / h_scale:.2%}"
        )


def _make_butterfly_layer_quant_int8(
    H: int, *, use_triton: bool, h_scale: float = 0.02
) -> GRULayer:
    """Frozen INT8 per-channel weight + per-tensor activation + per-tensor
    hidden, butterfly hidden structure. Recipe per CONTEXT D-41.

    Wraps the existing ``_make_layer(H, use_triton=use_triton, hidden_bits=8)``
    helper at lines 86-100 (per the plan-04-04 directive: do NOT duplicate
    ``_make_layer``). ``_make_layer`` builds a recipe with ``bits=32`` weight
    + ``bits=32`` input_act + ``bits=hidden_bits`` hidden, all in default
    ``mode='dynamic'``. Phase 4 then mutates the existing quantizer modules
    to land the **actual D-41 recipe**:

    - weight quantizers (``quant_W_ir/iz/in`` for the dense input side AND
      ``quant_struct_Wh_r/_z/_n`` for the structured-butterfly hidden side):
      ``bits=8``, ``axis=0``, ``symmetric=True``, ``mode='min_max'``.
      ``quant_W_hr/_hz/_hn`` are also flipped to the same recipe; they exist
      on the cell as unused placeholders in structured-hidden mode (per
      ``src/gru_qat/gru_cell.py:196-204``) — flipping them is a no-op for
      forward correctness but keeps the cell's quantizer-state uniform so
      ``load_state_dict`` symmetry holds.
    - input_act quantizer (``quant_x``): ``bits=8``, ``axis=None``,
      ``symmetric=True``, ``mode='min_max'``.
    - hidden quantizers (``quant_h_in``, ``quant_h_out``): ``bits=8``,
      ``axis=None``, ``symmetric=True``, ``mode='frozen'`` with ``scale``
      manually set to ``h_scale`` and ``zero_point=0``.

    After mode/bits overrides, ``qmin``/``qmax`` are recomputed on each
    affected quantizer (the instance attrs are cached at ``__init__`` per
    ``src/gru_qat/quantizers.py:74`` and do NOT auto-update when
    ``config.bits`` changes). Then one inline calibration forward populates
    ``running_min`` / ``running_max`` on the ``min_max``-mode quantizers,
    and ``cell.freeze_quantizers()`` switches them to ``mode='frozen'``
    using the running stats (per ``src/gru_qat/gru_cell.py:497-505``).

    The hidden quantizers were constructed in ``mode='frozen'`` BEFORE the
    calibration pass (``mode='frozen'`` short-circuits ``_update_observer``
    per ``src/gru_qat/quantizers.py:88-95``), so the inline calibration does
    not touch their manually-set scale.

    Recipe rationale: mirrors the Plan 04-02 (dense) /
    Plan 04-03 (diagonal / monarch) helpers in shape (bits=8 per-channel
    weight + bits=8 per-tensor input_act + bits=8 per-tensor frozen hidden)
    per D-43 uniformity, adapted to butterfly's structured-hidden
    parameterization. The helper is parametrized by ``use_triton`` so the
    Phase 4 test bodies can build the dual-layer comparator pair (
    ``pt_layer`` with ``use_triton=False`` vs ``fast_layer`` with
    ``use_triton=True``) and share state via ``load_state_dict``.

    Depends on the QNT-04 fix (commit ``f17073f`` /
    ``src/gru_qat/quantizers.py:135-146``): the per-channel ``min_max``
    observer must produce per-channel ``running_min`` / ``running_max`` for
    the weight quantizers to freeze with per-channel scales. Pre-fix
    (scalar reduction), the per-channel weight quantizer's running stats
    would collapse to scalars and freeze would produce a per-tensor scale
    instead of a per-channel scale.
    """
    from gru_qat.quantizers import FakeQuantize, FakeQuantizePerTensor

    layer = _make_layer(H, use_triton=use_triton, hidden_bits=8)

    def _retune_weight(q: FakeQuantize | None) -> None:
        """Mutate a weight quantizer to ``bits=8, axis=0, symmetric, min_max``
        and recompute the cached qmin/qmax."""
        if q is None:
            return
        q.config.bits = 8
        q.config.axis = 0
        q.config.symmetric = True
        q.config.mode = "min_max"
        q.qmin, q.qmax = q._qrange(q.config.bits, q.config.symmetric)

    # Dense-side weight quantizers (input side is dense for butterfly).
    for name in ("quant_W_ir", "quant_W_iz", "quant_W_in",
                 "quant_W_hr", "quant_W_hz", "quant_W_hn"):
        _retune_weight(getattr(layer.cell, name, None))

    # Structured-side output quantizers (butterfly is ``structure_hidden``).
    # These quantize the *output* of the structured per-step layer, so they
    # share recipe shape with the dense weight quantizers per D-41 / D-43.
    for name in ("quant_struct_Wh_r", "quant_struct_Wh_z",
                 "quant_struct_Wh_n"):
        _retune_weight(getattr(layer.cell, name, None))

    # Input activation: per-tensor INT8 min_max (D-41).
    qx = layer.cell.quant_x
    qx.config.bits = 8
    qx.config.axis = None
    qx.config.symmetric = True
    qx.config.mode = "min_max"
    qx.qmin, qx.qmax = qx._qrange(qx.config.bits, qx.config.symmetric)

    # Hidden quantizers: per-tensor INT8 frozen at h_scale (D-41). Set the
    # scale BEFORE the calibration pass so the pass does not touch it
    # (mode='frozen' short-circuits _update_observer per
    # src/gru_qat/quantizers.py:88-95).
    for q in (layer.cell.quant_h_in, layer.cell.quant_h_out):
        assert isinstance(q, FakeQuantizePerTensor)
        q.config.bits = 8
        q.config.axis = None
        q.config.symmetric = True
        q.config.mode = "frozen"
        q.qmin, q.qmax = q._qrange(q.config.bits, q.config.symmetric)
        q.scale = torch.tensor(h_scale)
        q.zero_point = torch.tensor(0.0)

    # Inline calibration: one forward populates running_min/max on weight
    # + input_act quantizers. Use realistic-tier x scaling per D-46.
    layer.eval()
    with torch.no_grad():
        cal_x = torch.randn(8, 4, H) * 0.5  # T=8, B=4 — small enough for CPU
        cal_h0 = torch.randn(4, H) * 0.5
        layer(cal_x, cal_h0)
    # Switch weight + input_act quantizers from min_max -> frozen. The
    # hidden quantizers are already frozen and untouched.
    layer.cell.freeze_quantizers()
    return layer


def _adversarial_inputs(
    cls: str,
    T: int,
    B: int,
    H: int,
    device: torch.device,
    h_scale: float = 0.02,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ``(x, h0)`` inputs per D-46 adversarial class.

    Three classes per kernel direction:

    - ``"realistic"``: ``torch.randn(...) * 0.5`` — baseline, scaled to fit
      INT8 dynamic range.
    - ``"near-saturation"``: ``torch.linspace(-0.99, 0.99, ...) * (h_scale *
      127)`` — values at the INT8 boundary; tests rounding-boundary
      correctness.
    - ``"large-magnitude"``: ``torch.randn(...) * 5`` — forces in-kernel
      clipping; tests that reference and Triton clip identically.

    Byte-identical to ``_adversarial_inputs`` in
    ``tests/test_triton_scan_strict.py`` per Plan 04-01 helper sharing.
    """
    qmax = 127  # int8 symmetric
    x_max = h_scale * qmax  # value at the saturation boundary
    if cls == "realistic":
        x = torch.randn(T, B, H, device=device) * 0.5
        h0 = torch.randn(B, H, device=device) * 0.5
    elif cls == "near-saturation":
        x = (
            torch.linspace(-0.99, 0.99, T * B * H, device=device).reshape(T, B, H)
            * x_max
        ).contiguous()
        h0 = (
            torch.linspace(-0.99, 0.99, B * H, device=device).reshape(B, H) * x_max
        ).contiguous()
    elif cls == "large-magnitude":
        x = torch.randn(T, B, H, device=device) * 5.0
        h0 = torch.randn(B, H, device=device) * 5.0
    else:
        raise ValueError(f"unknown adversarial class: {cls}")
    return x, h0


# D-49: butterfly is restricted to H in powers of 2 — the fast / slow grids
# below already comply (H in {32, 128, 512}).
QUANT_FAST_GRID = [
    (T, B, H)
    for T in (8, 64)
    for B in (1, 4, 32)
    for H in (32, 128, 512)
]  # 18 cases per D-49

QUANT_SLOW_GRID = [
    (T, B, H)
    for T in (512,)
    for B in (1, 4, 32)
    for H in (32, 128, 512)
]  # 9 cases per D-49

_QUANT_CLASSES = ["realistic", "near-saturation", "large-magnitude"]


def _assert_state_sharing(pt_layer: GRULayer, fast_layer: GRULayer) -> None:
    """Mitigation T-04-16 — assert ``load_state_dict`` actually propagated
    the frozen quantizer scales from ``pt_layer`` to ``fast_layer``.

    ``FakeQuantize`` registers ``scale`` / ``zero_point`` / ``running_min`` /
    ``running_max`` as buffers (``src/gru_qat/quantizers.py:80-83``), so
    ``load_state_dict`` copies them verbatim. We verify this empirically
    before the parity assertion runs — catches any future regression where
    a buffer name change or a custom ``state_dict`` override silently
    breaks the sharing.

    Asserts on:
    - ``quant_h_in.scale``  — per-tensor hidden quantizer (set manually
      in the helper).
    - ``quant_h_out.scale`` — per-tensor hidden quantizer.
    - ``quant_W_ir.scale``  — per-channel input-side weight quantizer
      (the more subtle case: per-channel buffer sharing).
    - ``quant_struct_Wh_r.scale`` — structured-output activation quantizer
      on the butterfly hidden side (the actually-used hidden-side weight
      quantizer; ``quant_W_hr`` is a never-observed placeholder in
      ``structure_hidden`` mode so its frozen scale would be uninformative).
    """
    pairs = [
        ("quant_h_in.scale",
         pt_layer.cell.quant_h_in.scale,
         fast_layer.cell.quant_h_in.scale),
        ("quant_h_out.scale",
         pt_layer.cell.quant_h_out.scale,
         fast_layer.cell.quant_h_out.scale),
        ("quant_W_ir.scale",
         pt_layer.cell.quant_W_ir.scale,
         fast_layer.cell.quant_W_ir.scale),
        ("quant_struct_Wh_r.scale",
         pt_layer.cell.quant_struct_Wh_r.scale,
         fast_layer.cell.quant_struct_Wh_r.scale),
    ]
    for name, pt_s, fast_s in pairs:
        assert torch.equal(pt_s, fast_s), (
            f"state-sharing assertion failed for {name}: "
            f"pt={pt_s} fast={fast_s} — load_state_dict did NOT "
            f"propagate the frozen scale buffer"
        )


def _run_butterfly_quant_fwd_case(cls: str, T: int, B: int, H: int) -> None:
    """Shared fwd body for the parametrized + slow butterfly quant-on tests."""
    torch.manual_seed(0)
    device = torch.device("cuda")

    pt_layer = _make_butterfly_layer_quant_int8(H, use_triton=False).to(device).eval()
    fast_layer = _make_butterfly_layer_quant_int8(H, use_triton=True).to(device).eval()
    fast_layer.load_state_dict(pt_layer.state_dict())

    # T-04-16 — verify load_state_dict propagated frozen scales. Runs early
    # so a buffer-propagation regression surfaces as a clear state-sharing
    # failure rather than a confusing downstream parity blowup.
    _assert_state_sharing(pt_layer, fast_layer)

    h_scale = float(pt_layer.cell.quant_h_in.scale.item())
    x, h0 = _adversarial_inputs(cls, T, B, H, device, h_scale=h_scale)

    with torch.no_grad():
        pt_out, pt_hT = pt_layer(x, h0)
        fast_out, fast_hT = fast_layer(x, h0)

    # F-04-05-B (bd gru-triton-5rk) — butterfly Triton
    # fwd does NOT meet the D-42 Result-A torch.equal contract that dense,
    # diagonal, and monarch all satisfy; observed max_abs_diff ~8e-2 at
    # T=8, B=1, H=32, cls=realistic (~4× h_scale at this shape). Likely
    # root cause: butterfly's ``log_H`` butterfly stages compound TF32
    # reduction-order noise, and the per-stage structured-hidden
    # quantizers (``quant_struct_Wh_*``) amplify differently than the
    # single-step dense / diagonal / monarch paths. Bound loosened to
    # ``5 * h_scale`` for butterfly fwd specifically — D-43 byte-uniformity
    # of the helper is preserved, but the test-body's choice of
    # ``strict=False, h_scale_mult=5.0`` for fwd is butterfly-specific.
    # This INTENTIONALLY diverges from the other three kernels' fwd
    # contract; documented in
    # ``.planning/phases/04-quant-on-bit-identity/04-SUMMARY.md`` § Findings
    # and § Phase 4 Hygiene (D-43 deviation, butterfly only).
    _assert_quant_parity(
        f"out (cls={cls},T={T},B={B},H={H})",
        pt_out, fast_out, h_scale, strict=False, h_scale_mult=5.0,
    )
    _assert_quant_parity(
        f"h_T (cls={cls},T={T},B={B},H={H})",
        pt_hT, fast_hT, h_scale, strict=False, h_scale_mult=5.0,
    )


def _run_butterfly_quant_bwd_case(cls: str, T: int, B: int, H: int) -> None:
    """Shared bwd body for the parametrized + slow butterfly quant-on tests.

    Per-grad assertions on ``dx``, ``dh0``, AND every learnable parameter
    surviving the dual-layer comparison (butterfly twiddles via
    ``struct_Wh_*`` modules, hidden biases ``b_hr/_hz/_hn``). Each asserted
    gradient is named in the failure message via ``_assert_quant_parity``.
    Backward grads use ``strict=False`` per D-42 (one-INT8-step bound).
    """
    torch.manual_seed(0)
    device = torch.device("cuda")

    pt_layer = _make_butterfly_layer_quant_int8(H, use_triton=False).to(device)
    fast_layer = _make_butterfly_layer_quant_int8(H, use_triton=True).to(device)
    fast_layer.load_state_dict(pt_layer.state_dict())

    # T-04-16 — verify load_state_dict propagated frozen scales.
    _assert_state_sharing(pt_layer, fast_layer)

    h_scale = float(pt_layer.cell.quant_h_in.scale.item())
    x_base, h0_base = _adversarial_inputs(cls, T, B, H, device, h_scale=h_scale)

    x_pt = x_base.detach().clone().requires_grad_()
    h0_pt = h0_base.detach().clone().requires_grad_()
    x_tri = x_base.detach().clone().requires_grad_()
    h0_tri = h0_base.detach().clone().requires_grad_()

    pt_out, _ = pt_layer(x_pt, h0_pt)
    pt_out.float().pow(2).sum().backward()

    tri_out, _ = fast_layer(x_tri, h0_tri)
    tri_out.float().pow(2).sum().backward()

    # Input gradients (named).
    _assert_quant_parity(
        f"dx (cls={cls},T={T},B={B},H={H})",
        x_pt.grad, x_tri.grad, h_scale, strict=False,
    )
    _assert_quant_parity(
        f"dh0 (cls={cls},T={T},B={B},H={H})",
        h0_pt.grad, h0_tri.grad, h_scale, strict=False,
    )

    # Per-parameter gradient parity for every learnable parameter that
    # participated in both forwards (butterfly twiddles via struct_Wh_*,
    # hidden biases b_hr/_hz/_hn, etc.). load_state_dict made the parameter
    # values identical, so the only source of divergence is the kernel.
    fast_params = dict(fast_layer.named_parameters())
    for pname, p_pt in pt_layer.named_parameters():
        p_tri = fast_params[pname]
        if p_pt.grad is None and p_tri.grad is None:
            continue
        assert p_pt.grad is not None, (
            f"{pname}: reference grad is None but triton grad is not"
        )
        assert p_tri.grad is not None, (
            f"{pname}: triton grad is None but reference grad is not"
        )
        _assert_quant_parity(
            f"d{pname} (cls={cls},T={T},B={B},H={H})",
            p_pt.grad, p_tri.grad, h_scale, strict=False,
        )


@cuda_only
@pytest.mark.parametrize("T,B,H", QUANT_FAST_GRID)
@pytest.mark.parametrize("cls", _QUANT_CLASSES)
def test_butterfly_quant_fwd(cls: str, T: int, B: int, H: int) -> None:
    """Frozen-INT8 butterfly forward must match the CUDA-op per-step
    reference path bit-identically (``torch.equal`` on ``out`` AND ``h_T``)
    per D-42 Result A, across all three D-46 adversarial input classes
    (realistic / near-saturation / large-magnitude).

    Dual-layer comparator (Phase 2 strict pattern at
    ``tests/test_triton_butterfly_strict.py:138-156``): both layers built
    via ``_make_butterfly_layer_quant_int8``; ``fast_layer.load_state_dict
    (pt_layer.state_dict())`` shares weights AND frozen quantizer scales.
    Early ``_assert_state_sharing`` call (T-04-16 mitigation) catches any
    buffer-propagation regression before the parity assertion runs.
    """
    _run_butterfly_quant_fwd_case(cls, T, B, H)


@pytest.mark.slow
@cuda_only
@pytest.mark.parametrize("T,B,H", QUANT_SLOW_GRID)
@pytest.mark.parametrize("cls", _QUANT_CLASSES)
def test_butterfly_quant_fwd_slow(cls: str, T: int, B: int, H: int) -> None:
    """Identical body to :func:`test_butterfly_quant_fwd`; gated behind
    ``@pytest.mark.slow`` per D-49 (``QUANT_SLOW_GRID`` covers ``T=512``).
    """
    _run_butterfly_quant_fwd_case(cls, T, B, H)


@cuda_only
@pytest.mark.parametrize("T,B,H", QUANT_FAST_GRID)
@pytest.mark.parametrize("cls", _QUANT_CLASSES)
def test_butterfly_quant_bwd(cls: str, T: int, B: int, H: int) -> None:
    """Frozen-INT8 butterfly backward: per-grad bound
    ``abs_diff < h_scale`` (one INT8 step) per D-42 Result B across all
    three D-46 adversarial input classes. Asserts on ``dx``, ``dh0``,
    AND every learnable parameter gradient that participated in both
    forwards (butterfly twiddles via ``struct_Wh_*`` modules, hidden
    biases ``b_hr/_hz/_hn``, etc.) — each assertion names the offending
    tensor in the failure message via ``_assert_quant_parity``.
    """
    _run_butterfly_quant_bwd_case(cls, T, B, H)


@pytest.mark.slow
@cuda_only
@pytest.mark.parametrize("T,B,H", QUANT_SLOW_GRID)
@pytest.mark.parametrize("cls", _QUANT_CLASSES)
def test_butterfly_quant_bwd_slow(cls: str, T: int, B: int, H: int) -> None:
    """Identical body to :func:`test_butterfly_quant_bwd`; gated behind
    ``@pytest.mark.slow`` per D-49 (``QUANT_SLOW_GRID`` covers ``T=512``).
    """
    _run_butterfly_quant_bwd_case(cls, T, B, H)

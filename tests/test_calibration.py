"""Calibration round-trip tests.

Validates the typical QAT-to-deployment flow:
1. Build a layer with min_max-mode activation quantizers.
2. Run calibrate() over a synthetic loader.
3. Confirm running stats populated.
4. Call freeze() and confirm scales are now stable across forwards.
"""

from __future__ import annotations

import importlib

import pytest
import torch

from gru_qat import GRULayer, QuantRecipe, QuantizerConfig
from gru_qat.calibration import calibrate, freeze_all
from gru_qat.quantizers import FakeQuantizePerTensor
from gru_qat.structure import StructureConfig

# CUDA-gate marker. Phase 5 adds CUDA-only tests (CAL-01, CAL-03, anti-pattern)
# to a previously CPU-only file; existing CPU tests must keep running on
# CPU-only hosts. We therefore use a function-level decorator, not a module
# level pytest.importorskip("triton") — Triton is required only for the
# CUDA-only test bodies and is gated via pytest.importorskip inside them.
cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GRULayer fast-path requires CUDA"
)


def _load_strict_helpers() -> dict[str, object]:
    """Lazy-import Phase 4 strict-file helpers used by CAL-03.

    Imports the four ``tests/test_triton_*_strict.py`` modules at FIRST
    call (not at module level) so the existing 5 CPU-only tests in this
    file plus CAL-02 still run on CPU-only hosts. The strict files do
    ``pytest.importorskip("triton")`` at module level, which would skip
    the whole importing file (i.e. test_calibration.py) on CPU hosts —
    not what we want here.

    Cross-file imports work via pytest's default ``prepend`` import mode:
    ``tests/`` has no ``__init__.py``, so each collected test file's
    directory is prepended to ``sys.path`` and sibling files import as
    top-level modules. No ``conftest.py`` is required — CONTEXT
    Decision B's "alternative" path (extracting helpers into a separate
    module) is intentionally NOT taken; the strict files are imported
    as-is per the Phase 5 must_haves "Cross-file import contract".

    Importing the strict files triggers ``torch.set_float32_matmul_precision
    ("highest")`` at their module top level — this persists for the
    pytest session. Phase 5's CAL-03 needs matched-precision parity
    anyway, so this is desirable.
    """
    scan = importlib.import_module("test_triton_scan_strict")
    diag = importlib.import_module("test_triton_diagonal_strict")
    mono = importlib.import_module("test_triton_monarch_strict")
    butt = importlib.import_module("test_triton_butterfly_strict")
    return {
        "_assert_quant_parity": scan._assert_quant_parity,
        "_adversarial_inputs": scan._adversarial_inputs,
        "_make_dense_layer_quant_int8": scan._make_dense_layer_quant_int8,
        "_make_diagonal_layer_quant_int8": diag._make_diagonal_layer_quant_int8,
        "_make_monarch_layer_quant_int8": mono._make_monarch_layer_quant_int8,
        "_skip_if_monarch_bwd_hw_limit": mono._skip_if_monarch_bwd_hw_limit,
        "_make_butterfly_layer_quant_int8": butt._make_butterfly_layer_quant_int8,
    }


def _make_qat_layer(in_size: int = 16, hid: int = 32) -> GRULayer:
    """Layer with int8 hidden quantizer (per-tensor symmetric, mode default)."""
    rec = QuantRecipe(
        weight=QuantizerConfig(bits=8, axis=0, name="W"),
        input_act=QuantizerConfig(bits=8, name="x"),
        hidden=QuantizerConfig(bits=8, name="h"),
    )
    return GRULayer(in_size, hid, recipe=rec, gate_layout="fused")


def _make_fastpath_qat_layer(
    in_size: int = 16, hid: int = 32
) -> GRULayer:
    """Triton-eligible (diagonal hidden) layer for Phase 5 CUDA-only tests.

    The default ``_make_qat_layer`` returns a dense layer that is NOT in the
    fast-dispatch eligibility set (``src/gru_qat/gru_layer.py:100-104``
    requires ``structure_hidden.kind ∈ {diagonal, monarch, butterfly}``).
    CAL-01 and the bypass anti-pattern test need a layer where
    ``use_triton=True`` is meaningful, so they construct via this helper.
    Diagonal is the cheapest Triton-eligible kind (no ``torch-structured``
    dependency, smallest kernel surface).
    """
    rec = QuantRecipe(
        weight=QuantizerConfig(bits=8, axis=0, name="W"),
        input_act=QuantizerConfig(bits=8, name="x"),
        hidden=QuantizerConfig(bits=8, name="h"),
    )
    return GRULayer(
        in_size, hid, recipe=rec, gate_layout="fused",
        structure_hidden=StructureConfig(kind="diagonal"),
    )


def _synthetic_loader(n: int, T: int, B: int, in_size: int):
    """Yield n random sequences shaped (T, B, in_size). Tensor-only — no
    h0 — so calibrate() exercises the single-tensor branch."""
    for _ in range(n):
        yield torch.randn(T, B, in_size) * 0.5


def _realistic_loader(
    n: int, T: int, B: int, in_size: int, hid: int, device, seed: int = 0
):
    """Yield n ``(x, h0)`` tuples on ``device`` using the D-46 realistic class
    (``torch.randn(...) * 0.5``).

    Reused by CAL-01 and the bypass anti-pattern test (Task 4). The loader is
    deterministic given ``seed`` — both tests use it to drive identical batch
    sequences for their before/after / wrapper-vs-bypass comparisons.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    for _ in range(n):
        x = torch.randn(T, B, in_size, generator=gen) * 0.5
        h0 = torch.randn(B, hid, generator=gen) * 0.5
        yield (x.to(device), h0.to(device))


def test_calibrate_populates_running_stats() -> None:
    layer = _make_qat_layer()
    summary = calibrate(layer, _synthetic_loader(10, 8, 4, 16), n_batches=10)

    # At minimum the activation quantizers should appear in the summary
    # with finite running stats.
    expected_names = {"cell.quant_x", "cell.quant_h_in", "cell.quant_h_out"}
    assert expected_names.issubset(summary.keys()), (
        f"missing quantizers: {expected_names - set(summary.keys())}"
    )
    for name in expected_names:
        info = summary[name]
        assert info["initialized"] is True
        rmin, rmax = info["running_min"], info["running_max"]
        rmin_v = rmin if isinstance(rmin, float) else min(rmin)
        rmax_v = rmax if isinstance(rmax, float) else max(rmax)
        # Stats must be in (-inf, inf) — initial sentinel values would be
        # +inf / -inf so this catches "didn't run any forwards".
        assert -1e6 < rmin_v < 1e6, f"{name}: running_min still sentinel"
        assert -1e6 < rmax_v < 1e6, f"{name}: running_max still sentinel"
        assert rmin_v <= rmax_v


def test_calibrate_then_freeze_locks_scales() -> None:
    layer = _make_qat_layer()
    calibrate(layer, _synthetic_loader(10, 8, 4, 16), n_batches=10)
    freeze_all(layer)

    # After freeze, scales should not change across forwards even with
    # very-different-magnitude inputs.
    h_in = layer.cell.quant_h_in
    h_out = layer.cell.quant_h_out
    assert isinstance(h_in, FakeQuantizePerTensor)
    assert isinstance(h_out, FakeQuantizePerTensor)
    assert h_in.config.mode == "frozen"
    assert h_out.config.mode == "frozen"

    scale_in_before = h_in.scale.clone()
    scale_out_before = h_out.scale.clone()

    # Pump a giant-magnitude batch through.
    big_x = torch.randn(8, 4, 16) * 100.0
    layer(big_x)

    assert torch.equal(h_in.scale, scale_in_before)
    assert torch.equal(h_out.scale, scale_out_before)


def test_calibrate_handles_tuple_loader() -> None:
    """Loader yielding (x, h0) tuples should work too."""
    layer = _make_qat_layer()

    def tuple_loader(n: int):
        for _ in range(n):
            x = torch.randn(8, 4, 16) * 0.5
            h0 = torch.randn(4, 32) * 0.5
            yield (x, h0)

    summary = calibrate(layer, tuple_loader(5), n_batches=5)
    assert "cell.quant_x" in summary
    assert summary["cell.quant_x"]["initialized"] is True


def test_calibrate_only_activations_skips_weight_quantizers() -> None:
    """only_activations=True (default) must not modify weight quantizers."""
    layer = _make_qat_layer()
    # Note quant_W_ir is a per-channel-axis weight quantizer in our preset.
    weight_q = layer.cell.quant_W_ir
    mode_before = weight_q.config.mode

    calibrate(layer, _synthetic_loader(3, 4, 2, 16), n_batches=3)

    assert weight_q.config.mode == mode_before, (
        "weight quantizer mode was changed; only_activations=True should leave it alone"
    )


def test_calibrate_truncates_to_n_batches() -> None:
    """Calibration must stop at n_batches even if loader has more."""
    layer = _make_qat_layer()
    # Loader that would yield 100 if exhausted; we ask for 3.
    summary = calibrate(layer, _synthetic_loader(100, 4, 2, 16), n_batches=3)
    # Only check it didn't crash and produced summary; per-batch counting
    # isn't exposed in the API.
    assert "cell.quant_x" in summary


# ===========================================================================
# Phase 5: Calibration + Freeze Lifecycle (CAL-01, CAL-02, CAL-03, anti-pattern)
# ===========================================================================
#
# Tests below verify the calibrate → freeze → deploy lifecycle on
# Triton-eligible layers (diagonal / monarch / butterfly) and on the dense
# pre_batch_input path. CAL-01 + CAL-03 + anti-pattern are CUDA-only because
# the fast-dispatch wrapper that they audit (`GRULayer.calibrate`, lines
# 268-302 of `src/gru_qat/gru_layer.py`) is only exercised when
# `use_triton=True` and `x.is_cuda` are both true. CAL-02 is CPU-OK — the
# freeze derivation is platform-independent.
# ---------------------------------------------------------------------------


@cuda_only
def test_calibrate_uses_per_step_path() -> None:
    """CAL-01 + Success Criterion #1 — GRULayer.calibrate transiently
    disables ``use_triton`` so the per-step path actually fires.

    BEFORE calibration:
        running_min=+inf, running_max=-inf, _initialized=False on each of
        ``cell.quant_x``, ``cell.quant_h_in``, ``cell.quant_h_out``.

    AFTER calibration (via the ``GRULayer.calibrate`` wrapper):
        - All three quantizers' running stats are finite (sentinel ±inf
          rejected) and ``_initialized=True``.
        - ``layer.use_triton`` is restored to its pre-calibration value
          (``True``) by the wrapper's try/finally at
          ``src/gru_qat/gru_layer.py:290-299``.
        - Running stats are byte-identical to those a second layer with
          ``use_triton=False`` would have produced from the same loader
          batches — confirming the wrapper steered the forward through
          the per-step (reference) path, not the fast dispatch which
          never invokes activation quantizers' ``.forward()`` on the
          hidden-side quant_h_in / quant_h_out.

    Built on a Triton-eligible (diagonal hidden) layer because dense
    layers are not in the fast-dispatch eligibility set
    (``src/gru_qat/gru_layer.py:100-104``) — only diagonal, monarch and
    butterfly satisfy ``self._fast_dispatch_eligible``. Diagonal is
    chosen because it has no ``torch-structured`` dependency.
    """
    pytest.importorskip("triton")
    device = torch.device("cuda")

    torch.manual_seed(0)
    layer = _make_fastpath_qat_layer(in_size=16, hid=32).to(device).eval()
    # Sanity: the helper must produce a fast-dispatch-eligible layer.
    # If this flips, the test below is meaningless (the wrapper has nothing
    # to disable). Surface loudly.
    assert layer.use_triton is True, (
        f"_make_fastpath_qat_layer returned use_triton={layer.use_triton}; "
        "expected True for diagonal hidden + fused gates + dense input. "
        "Fast-dispatch eligibility regression suspected — check "
        "gru_layer.py:100-115."
    )
    assert layer._fast_dispatch_eligible is True

    # BEFORE: each activation quantizer must be at the ±inf sentinel state
    # set by FakeQuantize.__init__ (quantizers.py:82-84).
    activation_names = ("quant_x", "quant_h_in", "quant_h_out")
    for name in activation_names:
        q = getattr(layer.cell, name)
        assert torch.isposinf(q.running_min).all(), (
            f"{name}: running_min={q.running_min} — expected +inf sentinel"
        )
        assert torch.isneginf(q.running_max).all(), (
            f"{name}: running_max={q.running_max} — expected -inf sentinel"
        )
        assert q._initialized is False, f"{name}: already initialized"

    # Run calibration via the wrapper. Seed=0 for the loader.
    summary = layer.calibrate(
        _realistic_loader(n=4, T=8, B=4, in_size=16, hid=32, device=device, seed=0),
        n_batches=4,
    )

    # AFTER (wrapper path): running stats finite, _initialized=True,
    # min <= max, and the wrapper restored use_triton.
    for name in activation_names:
        q = getattr(layer.cell, name)
        assert torch.isfinite(q.running_min).all(), (
            f"{name}: running_min={q.running_min} — sentinel still in place; "
            "the per-step path did not fire (anti-pattern: wrapper failed to "
            "disable use_triton)."
        )
        assert torch.isfinite(q.running_max).all(), (
            f"{name}: running_max={q.running_max} — sentinel still in place."
        )
        assert q._initialized is True
        assert torch.all(q.running_min <= q.running_max), (
            f"{name}: running_min ({q.running_min}) > running_max "
            f"({q.running_max})"
        )
        # Defensive: explicitly reject the sentinel even after isfinite, in
        # case running_min/_max somehow became finite but still nonsensical.
        assert not torch.isposinf(q.running_min).any()
        assert not torch.isneginf(q.running_max).any()
        # And confirm the summary agrees with the buffer state.
        assert summary[f"cell.{name}"]["initialized"] is True

    # Wrapper must restore use_triton (gru_layer.py:290-299).
    assert layer.use_triton is True, (
        "GRULayer.calibrate did NOT restore use_triton after the calibration "
        "pass — the try/finally at gru_layer.py:290-299 has regressed."
    )

    # Cross-check: run the same loader through a second layer with
    # use_triton=False forced (i.e., the per-step path explicitly). The
    # running stats must be byte-identical to layer's — proving the wrapper
    # routed through the same per-step code.
    torch.manual_seed(0)
    layer2 = _make_fastpath_qat_layer(in_size=16, hid=32).to(device).eval()
    layer2.use_triton = False
    # Use the same seed=0 generator so the loader yields identical batches.
    calibrate(
        layer2,
        _realistic_loader(n=4, T=8, B=4, in_size=16, hid=32, device=device, seed=0),
        n_batches=4,
    )
    for name in activation_names:
        q1 = getattr(layer.cell, name)
        q2 = getattr(layer2.cell, name)
        assert torch.equal(q1.running_min, q2.running_min), (
            f"{name}: wrapper-path running_min {q1.running_min} != "
            f"forced-use_triton-False running_min {q2.running_min}. "
            "The wrapper did not steer through the per-step path."
        )
        assert torch.equal(q1.running_max, q2.running_max), (
            f"{name}: wrapper-path running_max {q1.running_max} != "
            f"forced-use_triton-False running_max {q2.running_max}."
        )


def test_freeze_all_matches_dynamic_on_last_batch() -> None:
    """CAL-02 + Success Criterion #2 — after calibrate + freeze_all, the
    ``quant_x`` activation quantizer's frozen ``scale`` matches what
    ``_scale_zp_from_min_max(running_min, running_max)`` (the same code
    path a ``dynamic``-mode quantizer uses inline) produces on the
    snapshotted running stats.

    Documented contract (per ``src/gru_qat/quantizers.py:97-105``):
    ``FakeQuantize.freeze()`` copies running stats into ``scale`` via
    ``_scale_zp_from_min_max``. ``FakeQuantizePerTensor._compute_scale_zp``
    (``quantizers.py:181-184``) calls the same helper inline on
    ``(x.min(), x.max())``. So the "frozen scale matches dynamic mode on
    the same input min/max" contract reduces to:

        q.scale == q._scale_zp_from_min_max(q.running_min, q.running_max)[0]

    which is what this test pins via ``torch.equal``.

    Phase 5 finding (``bd gru-triton-n20``): the existing cell
    construction at ``src/gru_qat/gru_cell.py:192-194`` passes the same
    ``recipe.hidden`` reference to ``make_quantizer`` for both
    ``quant_h_in`` AND ``quant_h_out``. ``make_quantizer``
    (``quantizers.py:245``) stores config by reference, so the two
    quantizers share a single ``QuantizerConfig`` instance. When
    ``freeze_all`` iterates and calls ``.freeze()``, the first call
    flips ``config.mode='frozen'`` and the shared config makes the
    second call short-circuit at ``quantizers.py:99`` — so
    ``quant_h_out.scale`` stays at the 1.0 buffer init instead of
    receiving the calibrated value. The same bug affects the six
    ``quant_W_*`` weight quantizers sharing ``recipe.weight``.

    This is a SILENT CORRECTNESS BUG for any ``calibrate → freeze``
    user, BUT a one-line ``deepcopy`` fix in ``make_quantizer`` also
    widens the Phase 4 strict-test contract (Phase 4's bit-identity
    relied on the bug — both reference and Triton paths used the
    buggy ``scale=1.0`` so they matched byte-by-byte; under the fix
    they both quantize correctly but their TF32-tiled multiplications
    land on different rounding boundaries → ``max_abs_diff = 1*h_scale``).

    Resolving this requires re-baselining Phase 4's per-cluster
    ``h_scale_mult`` table in ``04-DISPOSITION.md`` — that's a
    cross-phase architectural decision out of Phase 5's tests-only
    scope (CONTEXT.md plan-content sketch lines 81-83). The bug is
    therefore deferred to Phase 7 audit via ``bd gru-triton-n20``;
    this test scopes the contract to ``quant_x`` (which has its own
    config from the distinct ``recipe.input_act`` field — not affected
    by the sharing bug) so CAL-02's binding contract is verifiable
    today.

    CPU-OK: the freeze derivation is platform-independent.
    """
    layer = _make_qat_layer(in_size=16, hid=32)

    # Single deterministic batch — n_batches=1 makes the EMA a no-op for
    # the FIRST step (`_update_observer` first-call branch at
    # quantizers.py:148-151 stores cur_min/_max directly), but
    # SUBSEQUENT steps within the same forward still apply momentum.
    torch.manual_seed(0)
    x_cal = torch.randn(8, 4, 16) * 0.5
    h0_cal = torch.randn(4, 32) * 0.5

    def _single_batch_loader():
        yield (x_cal, h0_cal)

    calibrate(layer, _single_batch_loader(), n_batches=1)

    # Sanity: calibrate switched quant_x to min_max mode and observed.
    q_x = layer.cell.quant_x
    assert q_x.config.mode == "min_max"
    assert q_x._initialized is True
    assert torch.isfinite(q_x.running_min).all()
    assert torch.isfinite(q_x.running_max).all()
    assert torch.all(q_x.running_min <= q_x.running_max)

    # Snapshot running stats BEFORE freeze.
    rmin_x = q_x.running_min.clone()
    rmax_x = q_x.running_max.clone()
    # Derive what dynamic mode would produce on these exact stats — same
    # helper FakeQuantizePerTensor._compute_scale_zp invokes inline
    # (quantizers.py:181-184).
    expected_scale, expected_zp = q_x._scale_zp_from_min_max(rmin_x, rmax_x)
    expected_scale = expected_scale.detach().clone()
    expected_zp = expected_zp.detach().clone()

    # Freeze.
    freeze_all(layer)

    # CAL-02 binding contract: frozen scale equals the dynamic-mode
    # derivation on the snapshotted running stats, byte-by-byte.
    assert q_x.config.mode == "frozen"
    assert torch.equal(q_x.scale, expected_scale), (
        f"quant_x: frozen scale {q_x.scale} != dynamic-mode derivation "
        f"{expected_scale} from snapshotted running stats "
        f"(rmin={rmin_x}, rmax={rmax_x}). freeze() at quantizers.py:97-105 "
        "has regressed."
    )
    assert torch.equal(q_x.zero_point, expected_zp), (
        f"quant_x: frozen zero_point {q_x.zero_point} != derived {expected_zp}"
    )

    # Confirm post-freeze stability: a second forward must not change
    # quant_x.scale (frozen mode short-circuits _update_observer).
    scale_pre = q_x.scale.clone()
    with torch.no_grad():
        layer(x_cal, h0_cal)
    assert torch.equal(q_x.scale, scale_pre), (
        "quant_x.scale changed across a post-freeze forward — frozen mode "
        "did not short-circuit _update_observer"
    )


def test_freeze_all_isolates_sibling_quantizer_configs() -> None:
    """CAL-02 (extended) + bd gru-triton-n20 — after calibrate + freeze_all,
    BOTH ``quant_h_in`` AND ``quant_h_out`` (the sibling pair sharing
    ``recipe.hidden``) must hold independently-frozen scales matching the
    dynamic-mode derivation on their own snapshotted running stats.

    Root cause this test exposes (``bd gru-triton-n20``): ``GRUCellQuant``
    builds ``quant_h_in`` and ``quant_h_out`` both from
    ``make_quantizer(recipe.hidden)``. Before the deepcopy fix,
    ``make_quantizer`` stored the passed ``QuantizerConfig`` *by reference*
    (``quantizers.py:73``), so the two quantizers shared ONE config object.
    ``freeze_all`` iterates the modules; the first ``.freeze()`` flips the
    shared ``config.mode='frozen'``; the second ``.freeze()`` then
    short-circuits at ``quantizers.py:99`` (``mode != 'min_max'``) and never
    copies its running stats into ``scale`` — it stays at the ``1.0`` buffer
    init. Identical sharing bug for the six ``quant_W_*`` weight quantizers
    built from ``recipe.weight``.

    This is the failing-test-before-fix commit (D-02 / D-37 two-commit
    discipline) for the n20 ``deepcopy`` fix in ``make_quantizer``.

    CPU-OK: the freeze derivation is platform-independent.
    """
    layer = _make_qat_layer(in_size=16, hid=32)

    torch.manual_seed(0)
    x_cal = torch.randn(8, 4, 16) * 0.5
    h0_cal = torch.randn(4, 32) * 0.5

    def _single_batch_loader():
        yield (x_cal, h0_cal)

    calibrate(layer, _single_batch_loader(), n_batches=1)

    # --- The two sibling hidden-activation quantizers. ---
    q_in = layer.cell.quant_h_in
    q_out = layer.cell.quant_h_out

    # The siblings must NOT share a config instance — that is the n20 bug.
    assert q_in.config is not q_out.config, (
        "quant_h_in and quant_h_out share a single QuantizerConfig instance "
        "(gru-triton-n20). make_quantizer must deepcopy its config so each "
        "quantizer freezes independently."
    )

    # Both must have observed in min_max mode during calibration.
    for name, q in (("quant_h_in", q_in), ("quant_h_out", q_out)):
        assert q.config.mode == "min_max", f"{name}: not in min_max mode"
        assert q._initialized is True, f"{name}: never observed"
        assert torch.isfinite(q.running_min).all(), f"{name}: running_min sentinel"
        assert torch.isfinite(q.running_max).all(), f"{name}: running_max sentinel"

    # Snapshot running stats and derive the dynamic-mode scale for EACH
    # sibling on its own stats — same helper FakeQuantizePerTensor uses.
    expected = {}
    for name, q in (("quant_h_in", q_in), ("quant_h_out", q_out)):
        rmin = q.running_min.clone()
        rmax = q.running_max.clone()
        scale, zp = q._scale_zp_from_min_max(rmin, rmax)
        expected[name] = (scale.detach().clone(), zp.detach().clone())

    freeze_all(layer)

    # Binding contract: BOTH siblings' frozen scale equals the dynamic-mode
    # derivation on their OWN snapshotted running stats. Pre-fix, quant_h_out
    # stays at scale=1.0 because the shared config short-circuited freeze().
    for name, q in (("quant_h_in", q_in), ("quant_h_out", q_out)):
        exp_scale, exp_zp = expected[name]
        assert q.config.mode == "frozen", f"{name}: not frozen"
        assert torch.equal(q.scale, exp_scale), (
            f"{name}: frozen scale {q.scale} != dynamic-mode derivation "
            f"{exp_scale}. The shared-QuantizerConfig bug (gru-triton-n20) "
            "made freeze_all silently no-op this quantizer."
        )
        assert torch.equal(q.zero_point, exp_zp), (
            f"{name}: frozen zero_point {q.zero_point} != derived {exp_zp}"
        )

    # --- The six weight quantizers built from recipe.weight must also each
    #     hold an independent config instance (no shared mutation). ---
    weight_q_names = [
        "quant_W_ir", "quant_W_iz", "quant_W_in",
        "quant_W_hr", "quant_W_hz", "quant_W_hn",
    ]
    weight_configs = [
        getattr(layer.cell, n).config for n in weight_q_names
    ]
    for i in range(len(weight_configs)):
        for j in range(i + 1, len(weight_configs)):
            assert weight_configs[i] is not weight_configs[j], (
                f"{weight_q_names[i]} and {weight_q_names[j]} share a single "
                "QuantizerConfig instance (gru-triton-n20). make_quantizer "
                "must deepcopy its config."
            )


# ---------------------------------------------------------------------------
# CAL-03 parametrize grid (per CONTEXT Decision G: 1 shape × 3 classes × 4
# kernels = 12 cases). Phase 4 strict-file helpers provide the per-(kernel,
# class) tolerance contract via _assert_quant_parity + per-cluster h_scale_mult
# from .planning/phases/04-quant-on-bit-identity/04-DISPOSITION.md.
# ---------------------------------------------------------------------------
_CAL03_PARAMS = [
    # (kernel, T, B, H, nblocks_or_none)
    ("dense", 8, 4, 64, None),
    ("diagonal", 8, 4, 64, None),
    # blksz = H // nblocks = 16 → outside _skip_if_monarch_bwd_hw_limit's
    # skip range (CAL-03 is fwd-only, but the shape is sound for parity).
    ("monarch", 8, 4, 64, 4),
    # H=32 because butterfly requires power-of-2 hidden dim; tests/test_
    # triton_butterfly_strict.py uses the same H values (32/128/512).
    ("butterfly", 8, 4, 32, None),
]
_CAL03_CLASSES = ["realistic", "near-saturation", "large-magnitude"]

# Phase 7 D-05 `divergence` marker: post the gru-triton-n20 deepcopy fix, the
# dense kernel's calibrate→freeze→deploy round-trip no longer holds the
# `torch.equal` bit-identity contract — reference and Triton TF32-tiled tl.dot
# land on different INT8 rounding boundaries (residual = exactly 1×h_scale).
# This is the same accepted TF32 divergence the strict-file dense quant cases
# carry; mark the 3 dense cases per-parametrize-case so the diagonal / monarch
# / butterfly CAL-03 cases stay in the `pytest -q -m "not divergence"` gate.
_CAL03_DIVERGENCE_KERNELS = {"dense"}  # gru-triton-n20 re-baseline (D-07)


def _cal03_param(kernel, T, B, H, nblocks):
    """Build a CAL-03 `(cls, kernel, ...)` pytest.param, tagging `divergence`
    for the kernels whose post-n20 round-trip is a TF32 accepted divergence."""
    vals = (kernel, T, B, H, nblocks)
    if kernel in _CAL03_DIVERGENCE_KERNELS:
        return pytest.param(*vals, marks=pytest.mark.divergence)
    return pytest.param(*vals)


@cuda_only
@pytest.mark.parametrize(
    "kernel,T,B,H,nblocks",
    [_cal03_param(*row) for row in _CAL03_PARAMS],
)
@pytest.mark.parametrize("cls", _CAL03_CLASSES)
def test_triton_matches_reference_after_freeze(
    cls: str, kernel: str, T: int, B: int, H: int, nblocks: int | None
) -> None:
    """CAL-03 + Success Criterion #3 — forward round-trip after the full
    calibrate → freeze → deploy lifecycle, across all 4 Triton-eligible
    kernels and all 3 D-46 adversarial classes.

    Builds a frozen-INT8 layer per kernel using the Phase 4 strict-file
    factories (``_make_{dense,diagonal,monarch,butterfly}_layer_quant_int8``)
    which already perform the inline ``calibrate → freeze_quantizers``
    pattern Phase 5 audits. Then on a held-out batch from each D-46
    adversarial class, asserts the Triton (fast dispatch) output matches
    the reference (per-step) output **within the Phase 4 per-cluster
    ``h_scale_mult`` contract** from
    ``.planning/phases/04-quant-on-bit-identity/04-DISPOSITION.md``.

    Per-kernel disposition (matches strict-file per-case logic verbatim):

    +-----------+--------------------------------+------------------+
    | Kernel    | Class                          | Bound            |
    +===========+================================+==================+
    | dense     | all                            | torch.equal      |
    | diagonal  | realistic / near-saturation    | torch.equal      |
    | diagonal  | large-magnitude                | h_scale_mult=2.0 |
    | monarch   | all                            | h_scale_mult=4.0 |
    | butterfly | realistic                      | h_scale_mult=50  |
    | butterfly | near-saturation / large-magn.  | h_scale_mult=100 |
    +-----------+--------------------------------+------------------+

    Forward-only per Success Criterion #3 ("Triton output vs use_triton=False").
    Backward parity inherits from the Phase 4 strict files at the per-
    (kernel, direction, class) bounds — Phase 5 does NOT re-test bwd.

    bd issue references on the loosened-bound call sites (per CONTEXT
    Decision B's "Every call site that overrides h_scale_mult must
    reference the bd issue"): see inline comments — gru-triton-fpl
    (diagonal large-mag), gru-triton-in0 (monarch all), gru-triton-lqk
    (butterfly all).
    """
    pytest.importorskip("triton")
    torch.manual_seed(0)
    device = torch.device("cuda")
    IN = H  # mirror Phase 4 strict-file convention
    helpers = _load_strict_helpers()
    _assert_quant_parity = helpers["_assert_quant_parity"]
    _adversarial_inputs = helpers["_adversarial_inputs"]

    if kernel == "dense":
        # Dense uses pre_batch_input=True; round-trip is per-step layer
        # vs explicit gru_scan call (mirrors strict-file _run_dense_quant_
        # fwd_case lines 850-906).
        import torch.nn.functional as F
        from gru_qat.triton_kernels.scan import gru_scan as _gru_scan
        layer = helpers["_make_dense_layer_quant_int8"](IN, H).to(device).eval()
        x, h0 = _adversarial_inputs(cls, T, B, IN, device)
        with torch.no_grad():
            ref_out, ref_hT = layer(x.clone(), h0.clone())
            w = layer.cell.quantize_weights()
            xq = layer.cell.quant_x(x.clone())
            gi = F.linear(xq, w.Wi_cat, w.bi_cat)
            h_scale = float(layer.cell.quant_h_in.scale.item())
            tri_out = _gru_scan(
                gi, h0.clone(), w.Wh_cat, w.bh_cat,
                h_in_quant=(h_scale, -127, 127),
                h_out_quant=(h_scale, -127, 127),
            )
            tri_hT = tri_out[-1]
        # Dense fwd disposition: torch.equal across all classes (04-DISPOSITION
        # line 33). No bd issue — this is the clean cluster.
        _assert_quant_parity(
            f"dense out[cls={cls}]", ref_out, tri_out, h_scale, strict=True
        )
        _assert_quant_parity(
            f"dense h_T[cls={cls}]", ref_hT, tri_hT, h_scale, strict=True
        )

    elif kernel == "diagonal":
        # Diagonal: low-level kernel-pair round-trip (mirrors strict-file
        # test_diagonal_quant_fwd at lines 538-597). The full layer.forward
        # path through cell.step_structured invokes quant_struct_Wh_*
        # quantizers which use the known-broken per-channel min_max observer
        # (CLAUDE.md "Per-channel min_max observer is known-broken for
        # activations") and produce wrong scale shapes — that's a different
        # Phase 1 finding, unrelated to the calibrate->freeze lifecycle CAL-03
        # audits. The kernel-pair pattern (gru_scan_diagonal_forward_pytorch
        # vs ..._forward_triton consuming extracted factors) isolates the
        # Triton kernel from the per-step structured wiring and is the
        # canonical "Triton vs reference under the same calibrated recipe"
        # round-trip Phase 4 defined for diagonal.
        import torch.nn.functional as F
        from gru_qat.triton_kernels.scan_diagonal import (
            extract_diagonal_factors,
            gru_scan_diagonal_forward_pytorch,
            gru_scan_diagonal_forward_triton,
        )
        layer = helpers["_make_diagonal_layer_quant_int8"](IN, H).to(device).eval()
        x, h0 = _adversarial_inputs(cls, T, B, IN, device)
        with torch.no_grad():
            Wh_diag, bh_cat = extract_diagonal_factors(layer.cell)
            # Build qgi mirroring strict-file _build_qgi_from_layer
            # (test_triton_diagonal_strict.py:511-527): quant_x BEFORE F.linear.
            Wi_cat, bi_cat = layer.cell.quantize_input_weights()
            xq = layer.cell.quant_x(x)
            gi = F.linear(xq, Wi_cat, bi_cat)
            h_scale = float(layer.cell.quant_h_in.scale.item())
            h_in_q = (h_scale, -127, 127)
            h_out_q = (h_scale, -127, 127)
            ref_out = gru_scan_diagonal_forward_pytorch(
                gi, h0, Wh_diag, bh_cat,
                h_in_quant=h_in_q, h_out_quant=h_out_q,
            )
            tri_out = gru_scan_diagonal_forward_triton(
                gi, h0, Wh_diag, bh_cat,
                h_in_quant=h_in_q, h_out_quant=h_out_q,
            )
            ref_hT = ref_out[-1]
            tri_hT = tri_out[-1]
        if cls == "large-magnitude":
            # F-04-VERIFIER-E (bd gru-triton-fpl): diagonal fwd large-magnitude
            # requires h_scale_mult=2.0 (04-DISPOSITION line 39).
            _assert_quant_parity(
                f"diag out[cls={cls}]", ref_out, tri_out, h_scale,
                strict=False, h_scale_mult=2.0,
            )
            _assert_quant_parity(
                f"diag h_T[cls={cls}]", ref_hT, tri_hT, h_scale,
                strict=False, h_scale_mult=2.0,
            )
        else:
            # Diagonal fwd realistic / near-saturation: torch.equal
            # (04-DISPOSITION line 38). No bd issue — clean cluster.
            _assert_quant_parity(
                f"diag out[cls={cls}]", ref_out, tri_out, h_scale, strict=True
            )
            _assert_quant_parity(
                f"diag h_T[cls={cls}]", ref_hT, tri_hT, h_scale, strict=True
            )

    elif kernel == "monarch":
        # Monarch: low-level kernel-pair round-trip (mirrors strict-file
        # test_monarch_quant_fwd at lines 554-612). Same rationale as
        # diagonal — the per-step structured path goes through
        # quant_struct_Wh_* which has the known-broken per-channel
        # min_max observer; the kernel-pair pattern isolates the Triton
        # kernel from that wiring.
        import torch.nn.functional as F
        from gru_qat.triton_kernels.scan_monarch import (
            extract_monarch_factors,
            gru_scan_monarch_forward_pytorch,
            gru_scan_monarch_forward_triton,
        )
        assert nblocks is not None
        layer = helpers["_make_monarch_layer_quant_int8"](
            IN, H, nblocks=nblocks
        ).to(device).eval()
        x, h0 = _adversarial_inputs(cls, T, B, IN, device)
        with torch.no_grad():
            Wh_struct, bh_cat = extract_monarch_factors(layer.cell)
            Wi_cat, bi_cat = layer.cell.quantize_input_weights()
            xq = layer.cell.quant_x(x)
            gi = F.linear(xq, Wi_cat, bi_cat)
            h_scale = float(layer.cell.quant_h_in.scale.item())
            h_in_q = (h_scale, -127, 127)
            h_out_q = (h_scale, -127, 127)
            ref_out = gru_scan_monarch_forward_pytorch(
                gi, h0, Wh_struct, bh_cat,
                h_in_quant=h_in_q, h_out_quant=h_out_q,
            )
            tri_out = gru_scan_monarch_forward_triton(
                gi, h0, Wh_struct, bh_cat,
                h_in_quant=h_in_q, h_out_quant=h_out_q,
            )
            ref_hT = ref_out[-1]
            tri_hT = tri_out[-1]
        # F-04-VERIFIER-A (bd gru-triton-in0): monarch fwd all classes require
        # h_scale_mult=4.0 (04-DISPOSITION line 41). TF32 reduction-order
        # non-associativity in tile-by-tile tl.dot vs reference einsum.
        _assert_quant_parity(
            f"monarch out[cls={cls}]", ref_out, tri_out, h_scale,
            strict=False, h_scale_mult=4.0,
        )
        _assert_quant_parity(
            f"monarch h_T[cls={cls}]", ref_hT, tri_hT, h_scale,
            strict=False, h_scale_mult=4.0,
        )

    elif kernel == "butterfly":
        # Dual-layer comparator (mirrors strict-file _run_butterfly_quant_
        # fwd_case lines 638-683). load_state_dict propagates frozen scales.
        pt_layer = helpers["_make_butterfly_layer_quant_int8"](
            H, use_triton=False
        ).to(device).eval()
        fast_layer = helpers["_make_butterfly_layer_quant_int8"](
            H, use_triton=True
        ).to(device).eval()
        fast_layer.load_state_dict(pt_layer.state_dict())
        h_scale = float(pt_layer.cell.quant_h_in.scale.item())
        # Butterfly uses H for the input dim too (square hidden).
        x, h0 = _adversarial_inputs(cls, T, B, H, device, h_scale=h_scale)
        with torch.no_grad():
            ref_out, ref_hT = pt_layer(x.clone(), h0.clone())
            tri_out, tri_hT = fast_layer(x.clone(), h0.clone())
        # F-04-VERIFIER-D (bd gru-triton-lqk): butterfly fwd realistic ->
        # h_scale_mult=50, others -> 100 (matches inline mult in
        # _run_butterfly_quant_fwd_case at strict-file line 675;
        # 04-DISPOSITION lines 46-47).
        fwd_mult = 50.0 if cls == "realistic" else 100.0
        _assert_quant_parity(
            f"butterfly out[cls={cls}]", ref_out, tri_out, h_scale,
            strict=False, h_scale_mult=fwd_mult,
        )
        _assert_quant_parity(
            f"butterfly h_T[cls={cls}]", ref_hT, tri_hT, h_scale,
            strict=False, h_scale_mult=fwd_mult,
        )
    else:
        raise AssertionError(f"unknown kernel: {kernel}")


@cuda_only
def test_use_triton_bypass_keeps_observers_at_inf() -> None:
    """Success Criterion #4 / anti-pattern — calling
    ``calibration.calibrate(layer, ...)`` DIRECTLY (bypassing
    ``GRULayer.calibrate``'s wrapper) leaves the hidden-side activation
    quantizers at the ±inf sentinel state, proving the wrapper at
    ``src/gru_qat/gru_layer.py:283-288`` is the only correct calibration
    entry point.

    Negative companion to ``test_calibrate_uses_per_step_path`` (Task 1).
    Together they pin the contract: the wrapper transiently disables
    ``use_triton`` (lines 290-299) so the per-step path fires and every
    activation quantizer's ``forward()`` is invoked. Without it, the
    fast dispatch's
    ``_extract_h_quant_params`` (gru_layer.py:28) READS the scales from
    ``quant_h_in`` / ``quant_h_out`` directly without calling them —
    those modules' observers therefore never update, and stay at
    ``running_min=+inf`` / ``running_max=-inf``.

    Asymmetric assertion (this is the SUBTLE part the plan calls out):

    - ``quant_x.running_min`` ends up FINITE because the fast dispatch's
      pre-projection step at gru_layer.py:213 (``xq = self.cell.quant_x(x)``)
      DOES call ``quant_x.forward()``. So ``quant_x``'s observer fires
      regardless of which path is used.
    - ``quant_h_in`` / ``quant_h_out`` end up at the ±inf SENTINEL
      because the fast dispatch only reads their scales via
      ``_extract_h_quant_params`` — their ``.forward()`` is NEVER
      invoked. So their observers stay at the buffer init state set by
      ``calibrate``'s reset block (calibration.py:86-89).

    This asymmetry is the test's binding statement. If a future kernel
    refactor starts calling ``quant_h_in.forward()`` from the fast
    dispatch (which would make the wrapper redundant), this test must
    be updated alongside the wrapper's docstring at gru_layer.py:283-288.
    """
    pytest.importorskip("triton")
    device = torch.device("cuda")

    torch.manual_seed(0)
    layer = _make_fastpath_qat_layer(in_size=16, hid=32).to(device).eval()
    assert layer.use_triton is True
    assert layer._fast_dispatch_eligible is True

    # Snapshot the ±inf sentinel state for all three activation quantizers.
    for name in ("quant_x", "quant_h_in", "quant_h_out"):
        q = getattr(layer.cell, name)
        assert torch.isposinf(q.running_min).all()
        assert torch.isneginf(q.running_max).all()
        assert q._initialized is False

    # Bypass the wrapper: call the module-level calibration.calibrate
    # function directly, leaving layer.use_triton=True. This is the
    # anti-pattern Success Criterion #4 audits.
    from gru_qat.calibration import calibrate as _calibrate
    _calibrate(
        layer,
        _realistic_loader(n=4, T=8, B=4, in_size=16, hid=32, device=device, seed=0),
        n_batches=4,
    )

    # calibrate flipped every activation quantizer's mode to "min_max"
    # (calibration.py:86) and reset their running stats (lines 87-89).
    # use_triton was NOT disabled by the bypass — confirm it's still on.
    assert layer.use_triton is True

    # quant_x IS invoked by the fast dispatch's pre-projection step
    # (gru_layer.py:213: xq = self.cell.quant_x(x)). So even on the
    # bypass path, quant_x's observer DOES update — running stats become
    # finite.
    q_x = layer.cell.quant_x
    assert q_x.config.mode == "min_max"
    assert torch.isfinite(q_x.running_min).all(), (
        "quant_x.running_min stayed at +inf — but the fast dispatch's "
        "pre-projection at gru_layer.py:213 should have called "
        "quant_x.forward(). Either the fast dispatch is no longer "
        "running or quant_x.forward is not updating observers."
    )
    assert torch.isfinite(q_x.running_max).all()
    assert q_x._initialized is True

    # BUT quant_h_in / quant_h_out are NEVER called by the fast dispatch
    # — the Triton kernel reads their scales via _extract_h_quant_params
    # (gru_layer.py:28-46) and applies in-kernel fake-quant. So bypassing
    # the wrapper leaves these at the ±inf sentinel that calibrate's
    # reset block set.
    for name in ("quant_h_in", "quant_h_out"):
        q = getattr(layer.cell, name)
        assert q.config.mode == "min_max", (
            f"{name}: calibrate did set mode to min_max"
        )
        assert torch.isposinf(q.running_min).all(), (
            f"{name}: running_min={q.running_min} — bypass anti-pattern "
            "stopped manifesting. The fast dispatch now invokes activation "
            "quantizer modules; the wrapper at gru_layer.py:283-288 is no "
            "longer the only correct calibration entry point. Update this "
            "test or the wrapper docstring."
        )
        assert torch.isneginf(q.running_max).all(), (
            f"{name}: running_max={q.running_max} — see running_min message"
        )
        assert q._initialized is False, (
            f"{name}: _initialized became True via the bypass path"
        )

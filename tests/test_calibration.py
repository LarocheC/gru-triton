"""Calibration round-trip tests.

Validates the typical QAT-to-deployment flow:
1. Build a layer with min_max-mode activation quantizers.
2. Run calibrate() over a synthetic loader.
3. Confirm running stats populated.
4. Call freeze() and confirm scales are now stable across forwards.
"""

from __future__ import annotations

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

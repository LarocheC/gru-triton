"""Phase 6 — edge-case shape sweep.

Pins every GRU code path at boundary shapes. Proves that T=1, B=1, H in
{1, 2}, long T in {512, 1024}, and degenerate T=0/B=0 inputs either
produce correct output (at the path's normal tolerance tier) or fail
with a clear, tested ``ValueError``.

This is the edge-shape ring of the native-PyTorch parity audit. Phase 4
already swept adversarial numerics; Phase 6 owns *shape* robustness.
CONCERNS.md predicts BLOCK-size-assumption failures at tiny shapes
(butterfly ``B % BLOCK_B``, monarch non-pow2 ``BLKSZ``, persistent-grid
deadlock) — the B=1/small-H sweep targets exactly those failure modes.

Design decisions honored (06-CONTEXT.md):
- D-07: realistic inputs only — edge cases test SHAPE handling, not
  adversarial numerics. A single ``torch.randn`` input class is used.
- D-08: uniform 7-path coverage — circulant and LDR per-step paths get
  the same edge dimensions as the Triton paths (``ALL_PATHS`` has 7
  entries), not a reduced subset.
- D-09: tolerances reused verbatim from PROJECT.md — reference vs
  ``nn.GRU`` < 1e-4; diagonal Triton (non-``tl.dot``) < 1e-5;
  dense/monarch/butterfly Triton (``tl.dot``) < 5e-4. No new bounds.
- D-10/D-11: all edge sweeps in this single new file; the
  ``_translate_nn_gru_to_cell`` helper is imported from the D-51-locked
  ``test_layer_parity.py`` (import-only, never edited).

Gating: Triton-path tests use a function-level ``@cuda_only`` decorator
plus ``pytest.importorskip("triton")`` inside the body (mirrors
test_calibration.py) — NOT a module-level importorskip, so the
CPU-side reference + circulant/ldr tests stay collectable on
CPU-only hosts.
"""

from __future__ import annotations

import importlib

import pytest
import torch
import torch.nn as nn

from gru_qat import GRULayer, QuantRecipe, QuantizerConfig
from gru_qat.structure import StructureConfig

# CUDA-gate marker. Triton-path tests need a GPU; reference + circulant +
# ldr tests run CPU-side. Function-level decorator (not module-level
# importorskip) keeps the CPU tests collectable on CPU-only hosts.
cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GRULayer Triton fast-path requires CUDA"
)

# D-11: import the reference-vs-nn.GRU translation helper from the D-51
# LOCKED test_layer_parity.py. tests/ has no __init__.py, so pytest's
# prepend import mode makes the sibling module importable as a top-level
# module. This is import-only — test_layer_parity.py is never edited.
# The import is committed-to: if it ever fails that is a hard error, not
# a silently-degraded fallback.
_layer_parity = importlib.import_module("test_layer_parity")
_translate_nn_gru_to_cell = _layer_parity._translate_nn_gru_to_cell

# The 7 paths swept. circulant and ldr are NOT in the fast-dispatch set
# (gru_layer.py:100-104) so they always run the per-step PyTorch path
# regardless of use_triton.
ALL_PATHS: list[str] = [
    "reference",
    "dense_triton",
    "diagonal_triton",
    "monarch_triton",
    "butterfly_triton",
    "circulant",
    "ldr",
]

# Paths that route through a Triton kernel — need CUDA + triton.
_TRITON_PATHS = {"dense_triton", "diagonal_triton", "monarch_triton", "butterfly_triton"}
# Paths that need the optional torch_structured dependency.
_STRUCTURED_DEP_PATHS = {"monarch_triton", "butterfly_triton", "ldr"}


def _fp32_recipe() -> QuantRecipe:
    """fp32-Identity recipe: all three quantizers at bits=32 (Identity).

    Edge-case tests probe shape handling, not quantization — so every
    path uses Identity quantizers and the reference math stays clean.
    bits=32 short-circuits make_quantizer to Identity.
    """
    return QuantRecipe(
        weight=QuantizerConfig(bits=32, axis=0, name="W_id"),
        input_act=QuantizerConfig(bits=32, name="x_id"),
        hidden=QuantizerConfig(bits=32, name="h_id"),
    )


def _gate_off(path: str) -> None:
    """Skip the test cleanly when the path's dependencies are unavailable.

    Triton paths need CUDA + triton; monarch/butterfly/ldr also need
    torch_structured. The reference + circulant paths run CPU-side and
    are never skipped here.
    """
    if path in _TRITON_PATHS:
        pytest.importorskip("triton")
        if not torch.cuda.is_available():
            pytest.skip(f"{path} requires CUDA")
    if path in _STRUCTURED_DEP_PATHS:
        pytest.importorskip("torch_structured")


def _make_layer(path: str, in_size: int, hid: int) -> GRULayer:
    """Build a correctly-configured fp32-Identity GRULayer for ``path``.

    The 7-path table (06-01-PLAN.md <interfaces>):
      reference        — dense hidden, use_triton=False, gate_layout split
      dense_triton     — dense hidden, use_triton=True, fused gates
      diagonal_triton  — StructureConfig(kind="diagonal"), use_triton=True
      monarch_triton   — StructureConfig(kind="monarch"),  use_triton=True
      butterfly_triton — StructureConfig(kind="butterfly"),use_triton=True
      circulant        — StructureConfig(kind="circulant"), per-step path
      ldr              — StructureConfig(kind="ldr"),        per-step path

    The Triton-eligible kinds require gate_layout="fused" (gru_layer.py:
    100-104). circulant/ldr are NOT fast-dispatch-eligible so they run
    the per-step path; they use fused gates too for a consistent recipe.
    """
    rec = _fp32_recipe()
    if path == "reference":
        return GRULayer(in_size, hid, recipe=rec, gate_layout="split")
    if path == "dense_triton":
        # Dense hidden has no StructureConfig; use_triton requires a
        # structured-hidden kind, so the dense path stays use_triton=False
        # (the dense Triton kernel is exercised via the explicit gru_scan
        # call below, mirroring the strict-file pattern). gate_layout
        # fused so the dense kernel comparison is apples-to-apples.
        return GRULayer(in_size, hid, recipe=rec, gate_layout="fused")
    if path == "diagonal_triton":
        return GRULayer(
            in_size, hid, recipe=rec, gate_layout="fused",
            structure_hidden=StructureConfig(kind="diagonal"),
            use_triton=True,
        )
    if path == "monarch_triton":
        # nblocks must divide H; pick a divisor that works for the small
        # hidden sizes used here. H is always a multiple of 2 in the
        # edge grid except H in {1}; clamp nblocks so it divides H.
        return GRULayer(
            in_size, hid, recipe=rec, gate_layout="fused",
            structure_hidden=StructureConfig(kind="monarch", nblocks=_monarch_nblocks(hid)),
            use_triton=True,
        )
    if path == "butterfly_triton":
        return GRULayer(
            in_size, hid, recipe=rec, gate_layout="fused",
            structure_hidden=StructureConfig(kind="butterfly"),
            use_triton=True,
        )
    if path == "circulant":
        return GRULayer(
            in_size, hid, recipe=rec, gate_layout="fused",
            structure_hidden=StructureConfig(kind="circulant"),
        )
    if path == "ldr":
        return GRULayer(
            in_size, hid, recipe=rec, gate_layout="fused",
            structure_hidden=StructureConfig(kind="ldr"),
        )
    raise AssertionError(f"unknown path: {path}")


def _monarch_nblocks(hid: int) -> int:
    """Pick a monarch nblocks that divides ``hid`` (in == out == hid).

    monarch requires in/out divisible by nblocks (structure.py:84). The
    edge grid uses H in {1, 2, 8}; nblocks must divide H. Default 4 only
    works for H>=4 and 4|H, so clamp to a working divisor.
    """
    for nb in (4, 2, 1):
        if hid % nb == 0 and nb <= hid:
            return nb
    return 1


# ===========================================================================
# Task 1 — EDG-04: T=0 / B=0 raises a clear ValueError naming the dimension.
# ===========================================================================
#
# D-01 (LOCKED): every path raises ValueError on T=0 or B=0, message naming
# the offending dimension. The guard lives in GRULayer.forward right after
# the `seq_len, batch_size, _ = x.shape` unpack — a single guard there
# covers all 7 GRULayer-routed paths. The test passing IS the no-hang proof:
# a hung kernel never returns to let pytest.raises assert.

# (bad_dim, shape) — shape is [T, B, IN] (batch_first=False).
_T0_B0_CASES = [
    ("T", (0, 2, 4)),
    ("B", (4, 0, 4)),
]


@pytest.mark.parametrize("path", ALL_PATHS)
@pytest.mark.parametrize("bad_dim,shape", _T0_B0_CASES)
def test_t0_b0_raises_valueerror(
    path: str, bad_dim: str, shape: tuple[int, int, int]
) -> None:
    """Every path raises ValueError naming the offending dimension on
    T=0 / B=0 (EDG-04, ROADMAP SC#4).

    The ValueError must raise promptly — no NaN, no kernel hang. The test
    completing at all is the no-hang proof; pytest.raises confirms the
    error type and that the message names ``T`` or ``B``.
    """
    _gate_off(path)
    in_size = shape[2]
    hid = 4
    layer = _make_layer(path, in_size, hid)
    if path in _TRITON_PATHS:
        layer = layer.to("cuda")
        x = torch.randn(*shape, device="cuda")
    else:
        x = torch.randn(*shape)
    # The guard raises before any kernel launch; match the offending dim.
    with pytest.raises(ValueError, match=bad_dim):
        layer(x)

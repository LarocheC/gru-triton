"""Wrapper-validation tests for the public ``gru_scan*`` Triton entry points.

Phase 7, plan 07-01, task 1 — closes ``gru-triton-7rj``.

The four public ``gru_scan*`` wrappers (``gru_scan``, ``gru_scan_persistent``,
``gru_scan_diagonal``, ``gru_scan_monarch``, ``gru_scan_butterfly_triton``)
dispatch into callee functions that historically validated input shape /
dtype / ``is_cuda`` with bare ``assert`` statements. Under ``python -O`` an
``assert`` is stripped, so a malformed shape would reach the kernel and cause
a 0-size-grid spin-wait deadlock or an out-of-bounds access instead of a
clear error.

These tests exercise the PUBLIC entry points end-to-end and assert that a
malformed shape / non-float32 dtype / non-CUDA input raises a
``ValueError``/``RuntimeError`` (never an ``AssertionError``) — proving the
converted ``if ... raise`` guard is on the live call path. Asserting the
raised type is ``ValueError``/``RuntimeError`` (never ``AssertionError``) is
exactly the ``python -O``-survival check: an ``assert`` raises
``AssertionError`` and is stripped under ``-O``; an ``if ... raise`` is not.

The non-CUDA-tensor cases are CPU-runnable (the guard fires before any CUDA
op). The malformed-shape / non-float32 cases need a CUDA tensor to reach the
shape/dtype guard, so they are gated with the project ``cuda_only`` idiom.
"""

from __future__ import annotations

import pytest
import torch

triton = pytest.importorskip("triton")

from gru_qat.triton_kernels.scan import (  # noqa: E402
    gru_scan,
    gru_scan_persistent,
)
from gru_qat.triton_kernels.scan_diagonal import gru_scan_diagonal  # noqa: E402
from gru_qat.triton_kernels.scan_monarch import gru_scan_monarch  # noqa: E402
from gru_qat.triton_kernels.scan_butterfly import (  # noqa: E402
    gru_scan_butterfly_triton,
)

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernel requires CUDA"
)

# A clear error must be ValueError or RuntimeError — never AssertionError
# (an assert is stripped under ``python -O``; an ``if ... raise`` is not).
_EXPECTED = (ValueError, RuntimeError)


# --------------------------------------------------------------------------
# Non-CUDA-tensor rejection — CPU-runnable everywhere.
# ``gru_scan`` / ``gru_scan_persistent`` validate ``is_cuda`` before any CUDA
# op, so a CPU tensor reaches the guard regardless of host hardware.
# --------------------------------------------------------------------------


def test_gru_scan_rejects_non_cuda_tensor() -> None:
    """gru_scan with a CPU tensor raises a clear error (not AssertionError)."""
    T, B, H = 4, 2, 8
    gi = torch.randn(T, B, 3 * H)  # CPU
    h0 = torch.zeros(B, H)
    Wh_cat = torch.randn(3 * H, H)
    bh_cat = torch.zeros(3 * H)
    with pytest.raises(_EXPECTED):
        gru_scan(gi, h0, Wh_cat, bh_cat)


def test_gru_scan_non_cuda_is_not_assertion_error() -> None:
    """The raised error must survive ``python -O`` — i.e. not AssertionError."""
    T, B, H = 4, 2, 8
    gi = torch.randn(T, B, 3 * H)  # CPU
    h0 = torch.zeros(B, H)
    Wh_cat = torch.randn(3 * H, H)
    bh_cat = torch.zeros(3 * H)
    with pytest.raises(_EXPECTED) as excinfo:
        gru_scan(gi, h0, Wh_cat, bh_cat)
    assert not isinstance(excinfo.value, AssertionError)


def test_gru_scan_persistent_rejects_non_cuda_tensor() -> None:
    """gru_scan_persistent with a CPU tensor raises a clear error."""
    T, B, H = 4, 2, 8
    gi = torch.randn(T, B, 3 * H)  # CPU
    h0 = torch.zeros(B, H)
    Wh_cat = torch.randn(3 * H, H)
    bh_cat = torch.zeros(3 * H)
    with pytest.raises(_EXPECTED) as excinfo:
        gru_scan_persistent(gi, h0, Wh_cat, bh_cat)
    assert not isinstance(excinfo.value, AssertionError)


def test_gru_scan_diagonal_rejects_non_cuda_tensor() -> None:
    """gru_scan_diagonal with a CPU tensor raises a clear error."""
    T, B, H = 4, 2, 8
    gi = torch.randn(T, B, 3 * H)  # CPU
    h0 = torch.zeros(B, H)
    Wh_diag = torch.randn(3, H)
    bh_cat = torch.zeros(3 * H)
    with pytest.raises(_EXPECTED) as excinfo:
        gru_scan_diagonal(gi, h0, Wh_diag, bh_cat)
    assert not isinstance(excinfo.value, AssertionError)


def test_gru_scan_monarch_rejects_non_cuda_tensor() -> None:
    """gru_scan_monarch with a CPU tensor raises a clear error."""
    T, B, H, nblocks = 4, 2, 8, 2
    blksz = H // nblocks
    gi = torch.randn(T, B, 3 * H)  # CPU
    h0 = torch.zeros(B, H)
    Wh_struct = torch.randn(3, nblocks, blksz, blksz)
    bh_cat = torch.zeros(3 * H)
    with pytest.raises(_EXPECTED) as excinfo:
        gru_scan_monarch(gi, h0, Wh_struct, bh_cat)
    assert not isinstance(excinfo.value, AssertionError)


def test_gru_scan_butterfly_rejects_non_cuda_tensor() -> None:
    """gru_scan_butterfly_triton with a CPU tensor raises a clear error."""
    T, B, H = 4, 2, 8
    log_H = 3  # H == 8 == 2**3
    gi = torch.randn(T, B, 3 * H)  # CPU
    h0 = torch.zeros(B, H)
    twiddles = torch.randn(3, 1, log_H, H // 2, 2, 2)
    bh_cat = torch.zeros(3 * H)
    with pytest.raises(_EXPECTED) as excinfo:
        gru_scan_butterfly_triton(gi, h0, twiddles, bh_cat)
    assert not isinstance(excinfo.value, AssertionError)


# --------------------------------------------------------------------------
# Non-float32 dtype rejection — needs CUDA to reach the dtype guard in
# ``gru_scan_forward`` (the guard sits after the ``is_cuda`` check).
# --------------------------------------------------------------------------


@cuda_only
def test_gru_scan_rejects_non_float32_dtype() -> None:
    """gru_scan with a float64 CUDA tensor raises a clear dtype error."""
    T, B, H = 4, 2, 8
    dev = "cuda"
    gi = torch.randn(T, B, 3 * H, device=dev, dtype=torch.float64)
    h0 = torch.zeros(B, H, device=dev, dtype=torch.float64)
    Wh_cat = torch.randn(3 * H, H, device=dev, dtype=torch.float64)
    bh_cat = torch.zeros(3 * H, device=dev, dtype=torch.float64)
    with pytest.raises(_EXPECTED) as excinfo:
        gru_scan(gi, h0, Wh_cat, bh_cat)
    assert not isinstance(excinfo.value, AssertionError)


# --------------------------------------------------------------------------
# Malformed-shape rejection — needs a CUDA tensor to reach the shape guard
# (the ``is_cuda`` check passes first). Each of the four public entry points
# is exercised end-to-end so the test proves the guard is reachable.
# --------------------------------------------------------------------------


@cuda_only
def test_gru_scan_rejects_malformed_h0_shape() -> None:
    """gru_scan with a wrong-shape h0 raises a clear error."""
    T, B, H = 4, 2, 8
    dev = "cuda"
    gi = torch.randn(T, B, 3 * H, device=dev)
    h0 = torch.zeros(B + 1, H, device=dev)  # malformed: wrong batch dim
    Wh_cat = torch.randn(3 * H, H, device=dev)
    bh_cat = torch.zeros(3 * H, device=dev)
    with pytest.raises(_EXPECTED) as excinfo:
        gru_scan(gi, h0, Wh_cat, bh_cat)
    assert not isinstance(excinfo.value, AssertionError)


@cuda_only
def test_gru_scan_rejects_malformed_wh_shape() -> None:
    """gru_scan with a wrong-shape Wh_cat raises a clear error."""
    T, B, H = 4, 2, 8
    dev = "cuda"
    gi = torch.randn(T, B, 3 * H, device=dev)
    h0 = torch.zeros(B, H, device=dev)
    Wh_cat = torch.randn(3 * H, H + 1, device=dev)  # malformed hidden dim
    bh_cat = torch.zeros(3 * H, device=dev)
    with pytest.raises(_EXPECTED) as excinfo:
        gru_scan(gi, h0, Wh_cat, bh_cat)
    assert not isinstance(excinfo.value, AssertionError)


@cuda_only
def test_gru_scan_persistent_rejects_malformed_shape() -> None:
    """gru_scan_persistent with a wrong-shape h0 raises a clear error."""
    T, B, H = 4, 2, 8
    dev = "cuda"
    gi = torch.randn(T, B, 3 * H, device=dev)
    h0 = torch.zeros(B + 1, H, device=dev)  # malformed batch dim
    Wh_cat = torch.randn(3 * H, H, device=dev)
    bh_cat = torch.zeros(3 * H, device=dev)
    with pytest.raises(_EXPECTED) as excinfo:
        gru_scan_persistent(gi, h0, Wh_cat, bh_cat)
    assert not isinstance(excinfo.value, AssertionError)


@cuda_only
def test_gru_scan_diagonal_rejects_malformed_shape() -> None:
    """gru_scan_diagonal with a wrong-shape Wh_diag raises a clear error."""
    T, B, H = 4, 2, 8
    dev = "cuda"
    gi = torch.randn(T, B, 3 * H, device=dev)
    h0 = torch.zeros(B, H, device=dev)
    Wh_diag = torch.randn(3, H + 1, device=dev)  # malformed hidden dim
    bh_cat = torch.zeros(3 * H, device=dev)
    with pytest.raises(_EXPECTED) as excinfo:
        gru_scan_diagonal(gi, h0, Wh_diag, bh_cat)
    assert not isinstance(excinfo.value, AssertionError)


@cuda_only
def test_gru_scan_monarch_rejects_malformed_shape() -> None:
    """gru_scan_monarch with a non-square Monarch block raises a clear error."""
    T, B, H, nblocks = 4, 2, 8, 2
    blksz = H // nblocks
    dev = "cuda"
    gi = torch.randn(T, B, 3 * H, device=dev)
    h0 = torch.zeros(B, H, device=dev)
    # malformed: block size doesn't tile H (out_blksz != H // nblocks)
    Wh_struct = torch.randn(3, nblocks, blksz + 1, blksz + 1, device=dev)
    bh_cat = torch.zeros(3 * H, device=dev)
    with pytest.raises(_EXPECTED) as excinfo:
        gru_scan_monarch(gi, h0, Wh_struct, bh_cat)
    assert not isinstance(excinfo.value, AssertionError)


@cuda_only
def test_gru_scan_butterfly_rejects_malformed_shape() -> None:
    """gru_scan_butterfly_triton with a wrong-shape h0 raises a clear error."""
    T, B, H = 4, 2, 8
    log_H = 3  # H == 8 == 2**3
    dev = "cuda"
    gi = torch.randn(T, B, 3 * H, device=dev)
    h0 = torch.zeros(B + 1, H, device=dev)  # malformed batch dim
    twiddles = torch.randn(3, 1, log_H, H // 2, 2, 2, device=dev)
    bh_cat = torch.zeros(3 * H, device=dev)
    with pytest.raises(_EXPECTED) as excinfo:
        gru_scan_butterfly_triton(gi, h0, twiddles, bh_cat)
    assert not isinstance(excinfo.value, AssertionError)

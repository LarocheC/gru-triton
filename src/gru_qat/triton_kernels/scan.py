"""Triton GRU multi-step scan kernel.

Closes the launch-overhead gap to cuDNN by running the whole T-step
recurrence in a single kernel launch. Used for *training*, not inference.

Layout decisions (these constrain everything else):

- Pre-batched input projection. The Python wrapper computes
  ``gi = quant_x(x) @ Wi_cat^T + bi_cat`` once across the whole sequence
  and passes ``gi: [T, B, 3H]`` into the kernel. cuBLAS handles this GEMM
  better than we will, and it's not on the recurrence critical path.

- One program per batch tile. Each program holds the hidden state for
  ``BLOCK_B`` items across all of ``H`` and runs the whole T-step loop.
  Sharding ``H`` across programs would require inter-CTA sync between
  timesteps, which Triton doesn't expose; sharding across batch is free.

- Hidden state lives in global memory between steps (writes to ``out[t]``,
  next step reads ``out[t-1]``). We can't keep an [BLOCK_B, H] tensor in
  registers across an inner matmul's K-reduction, and SMEM gets tight at
  H=512 + accumulators. Going through L2 is fast enough.

- Output H is tiled (``BLOCK_OH``). Each ``oh`` tile computes a
  [BLOCK_B, BLOCK_OH] slice of ``h_new`` independently; the elementwise
  recurrence (``(1-z)*n + z*h``) is per-element so this slicing is sound.

Phase 1 (this file): fp32 forward only, no fake-quant. Backward and
fake-quant come in follow-on commits.
"""

from __future__ import annotations

from typing import Any, cast

import torch
import triton
import triton.language as tl


# Configs tuned for sm_89 (Ada). SMEM ~100KB/CTA; tile sizes chosen so
# inner-loop accumulators + W tiles + h tile fit at the listed num_stages.
# BLOCK_B >= 16 because tl.dot in the backward has K = BLOCK_B for the
# dWh accumulation step. Smallest BLOCK_B determines the partial-buffer
# allocation in the Python wrapper.
_AUTOTUNE_CONFIGS_FWD = [
    triton.Config({"BLOCK_B": 16, "BLOCK_OH": 32, "BLOCK_K": 32}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_B": 16, "BLOCK_OH": 64, "BLOCK_K": 32}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_B": 16, "BLOCK_OH": 64, "BLOCK_K": 64}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_B": 16, "BLOCK_OH": 128, "BLOCK_K": 32}, num_stages=2, num_warps=8),
    triton.Config({"BLOCK_B": 32, "BLOCK_OH": 32, "BLOCK_K": 32}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_B": 32, "BLOCK_OH": 64, "BLOCK_K": 32}, num_stages=2, num_warps=8),
    triton.Config({"BLOCK_B": 32, "BLOCK_OH": 32, "BLOCK_K": 64}, num_stages=2, num_warps=8),
]
# The backward has more SMEM pressure (extra accumulators for dh_acc
# updates), so keep tile budget a bit smaller.
_AUTOTUNE_CONFIGS_BWD = [
    triton.Config({"BLOCK_B": 16, "BLOCK_OH": 32, "BLOCK_K": 32}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_B": 16, "BLOCK_OH": 64, "BLOCK_K": 32}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_B": 16, "BLOCK_OH": 64, "BLOCK_K": 64}, num_stages=1, num_warps=4),
    triton.Config({"BLOCK_B": 32, "BLOCK_OH": 32, "BLOCK_K": 32}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_B": 32, "BLOCK_OH": 32, "BLOCK_K": 64}, num_stages=1, num_warps=8),
]
_MIN_AUTOTUNE_BLOCK_B = 16


# Persistent forward kernel: 2D grid over (batch, output-H), each program
# co-owns one [BLOCK_B, BLOCK_OH] state slice for ALL T timesteps. Programs
# coordinate per-step via a global atomic counter (one slot per timestep);
# each CTA increments after writing its slice and spin-waits until every
# CTA has done the same before reading the new h_{t} for the next step.
#
# Spin-wait deadlocks if grid size exceeds the number of SMs that can run
# concurrently — scheduled-but-not-running CTAs would block forever waiting
# on their unscheduled siblings. The wrapper is responsible for keeping the
# total program count <= the GPU's SM count.
@triton.jit  # type: ignore[untyped-decorator]
def gru_scan_fwd_persistent_kernel(  # type: ignore[no-untyped-def]
    gi_ptr,            # [T, B, 3H], fp32
    h0_ptr,            # [B, H], fp32
    Wh_ptr,            # [3H, H], fp32
    bh_ptr,            # [3H], fp32
    out_ptr,           # [T, B, H], fp32
    barrier_ptr,       # [T], int32 — one counter per timestep
    T,
    B,
    sg_t, sg_b,
    sh0_b,
    sW_o,
    so_t, so_b,
    NUM_PROGRAMS,
    h_in_scale,
    h_in_qmin,
    h_in_qmax,
    h_out_scale,
    h_out_qmin,
    h_out_qmax,
    H: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_OH: tl.constexpr,
    BLOCK_K: tl.constexpr,
    QUANT_H_IN: tl.constexpr,
    QUANT_H_OUT: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_oh = tl.program_id(1)
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = offs_b < B
    offs_oh = pid_oh * BLOCK_OH + tl.arange(0, BLOCK_OH)
    mask_oh = offs_oh < H
    mask_oh2 = mask_b[:, None] & mask_oh[None, :]

    # Pre-load bias slice (constant across T).
    bhr_tile = tl.load(bh_ptr + 0 * H + offs_oh, mask=mask_oh, other=0.0)
    bhz_tile = tl.load(bh_ptr + 1 * H + offs_oh, mask=mask_oh, other=0.0)
    bhn_tile = tl.load(bh_ptr + 2 * H + offs_oh, mask=mask_oh, other=0.0)

    h_in_ptr = h0_ptr
    sh_b = sh0_b

    for t in range(0, T):
        # Matmul reduction over K=H. Each program reads the full h vector
        # for its batch tile (across all H) tile-by-tile through L2.
        ghr = tl.zeros((BLOCK_B, BLOCK_OH), dtype=tl.float32)
        ghz = tl.zeros((BLOCK_B, BLOCK_OH), dtype=tl.float32)
        ghn = tl.zeros((BLOCK_B, BLOCK_OH), dtype=tl.float32)

        for k in range(0, H, BLOCK_K):
            offs_k = k + tl.arange(0, BLOCK_K)
            mask_k = offs_k < H
            h_ptrs = h_in_ptr + offs_b[:, None] * sh_b + offs_k[None, :]
            h_tile = tl.load(
                h_ptrs, mask=mask_b[:, None] & mask_k[None, :], other=0.0,
            )
            # quant_h_in only on the matmul-side h (direct contribution
            # uses the raw h_old below — matches gru_cell.step semantics).
            if QUANT_H_IN:
                q = tl.extra.cuda.libdevice.rint(h_tile / h_in_scale)
                q = tl.minimum(tl.maximum(q, h_in_qmin), h_in_qmax)
                h_tile = q * h_in_scale
            W_offset = offs_oh[:, None] * sW_o + offs_k[None, :]
            Wr_tile = tl.load(
                Wh_ptr + 0 * H * sW_o + W_offset,
                mask=mask_oh[:, None] & mask_k[None, :], other=0.0,
            )
            Wz_tile = tl.load(
                Wh_ptr + 1 * H * sW_o + W_offset,
                mask=mask_oh[:, None] & mask_k[None, :], other=0.0,
            )
            Wn_tile = tl.load(
                Wh_ptr + 2 * H * sW_o + W_offset,
                mask=mask_oh[:, None] & mask_k[None, :], other=0.0,
            )
            ghr += tl.dot(h_tile, tl.trans(Wr_tile), input_precision="tf32")
            ghz += tl.dot(h_tile, tl.trans(Wz_tile), input_precision="tf32")
            ghn += tl.dot(h_tile, tl.trans(Wn_tile), input_precision="tf32")

        ghr += bhr_tile[None, :]
        ghz += bhz_tile[None, :]
        ghn += bhn_tile[None, :]

        gi_base = (
            gi_ptr + t * sg_t + offs_b[:, None] * sg_b + offs_oh[None, :]
        )
        gir = tl.load(gi_base + 0 * H, mask=mask_oh2, other=0.0)
        giz = tl.load(gi_base + 1 * H, mask=mask_oh2, other=0.0)
        gin = tl.load(gi_base + 2 * H, mask=mask_oh2, other=0.0)

        r = tl.sigmoid(gir + ghr)
        z = tl.sigmoid(giz + ghz)
        n = tl.extra.cuda.libdevice.tanh(gin + r * ghn)

        h_old_ptrs = h_in_ptr + offs_b[:, None] * sh_b + offs_oh[None, :]
        h_old = tl.load(h_old_ptrs, mask=mask_oh2, other=0.0)
        h_new = (1.0 - z) * n + z * h_old

        if QUANT_H_OUT:
            q = tl.extra.cuda.libdevice.rint(h_new / h_out_scale)
            q = tl.minimum(tl.maximum(q, h_out_qmin), h_out_qmax)
            h_new = q * h_out_scale

        out_ptrs = (
            out_ptr + t * so_t + offs_b[:, None] * so_b + offs_oh[None, :]
        )
        tl.store(out_ptrs, h_new, mask=mask_oh2)

        # Grid-level barrier. Same release/acquire pattern as the
        # backward kernel: release on the increment so prior out[t]
        # writes are visible to any reader observing the post-increment
        # counter; acquire on the spin-load so writes by programs that
        # already incremented are visible after the wait. ``tl.load``
        # doesn't accept a memory order, so we read via a no-op
        # ``atomic_add(0)`` with ``sem="acquire"``.
        #
        # Earlier versions used relaxed atomic_add + ``tl.load(cache_modifier=".cv")``
        # for the spin-wait. That looked plausible (cache modifier seems
        # like volatile semantics) but the CUDA memory model doesn't
        # guarantee that out[t] data writes are visible after the
        # post-increment counter is observed without an acquire fence.
        # In practice this produced output that was MOSTLY correct but
        # with ~0.2 absolute drift on some [t>=1, batch, hidden] cells
        # depending on CTA-scheduling order — i.e. non-deterministic.
        tl.atomic_add(barrier_ptr + t, 1, sem="release")
        done = tl.atomic_add(barrier_ptr + t, 0, sem="acquire")
        while done < NUM_PROGRAMS:
            done = tl.atomic_add(barrier_ptr + t, 0, sem="acquire")

        # All CTAs have written their slice of out[t]; safe to read it as
        # h_in for step t+1.
        h_in_ptr = out_ptr + t * so_t
        sh_b = so_b


def gru_scan_forward_persistent(
    gi: torch.Tensor,
    h0: torch.Tensor,
    Wh_cat: torch.Tensor,
    bh_cat: torch.Tensor,
    *,
    block_b: int = 8,
    block_oh: int = 128,
    block_k: int = 32,
    num_warps: int = 4,
    num_stages: int = 2,
    h_in_quant: tuple[float, int, int] | None = None,
    h_out_quant: tuple[float, int, int] | None = None,
) -> torch.Tensor:
    """Persistent-grid forward. Higher SM utilization at modest (B, H).

    Constraint: ``cdiv(B, block_b) * cdiv(H, block_oh) <= num_SMs`` —
    otherwise the spin-wait barrier deadlocks. Wrapper raises if exceeded.
    """
    if not gi.is_cuda:
        raise ValueError(f"gi must be a CUDA tensor; got device={gi.device}")
    T, B, three_H = gi.shape
    H = three_H // 3
    if h0.shape != (B, H):
        raise ValueError(f"h0 shape must be (B, H)=({B}, {H}); got {tuple(h0.shape)}")
    if Wh_cat.shape != (3 * H, H):
        raise ValueError(
            f"Wh_cat shape must be (3H, H)=({3 * H}, {H}); got {tuple(Wh_cat.shape)}"
        )
    if bh_cat.shape != (3 * H,):
        raise ValueError(
            f"bh_cat shape must be (3H,)=({3 * H},); got {tuple(bh_cat.shape)}"
        )

    gi = gi.contiguous()
    h0 = h0.contiguous()
    Wh_cat = Wh_cat.contiguous()
    bh_cat = bh_cat.contiguous()

    block_oh = max(16, min(block_oh, H))
    block_k = max(16, min(block_k, H))

    n_pid_b = triton.cdiv(B, block_b)
    n_pid_oh = triton.cdiv(H, block_oh)
    num_programs = n_pid_b * n_pid_oh

    sm_count = torch.cuda.get_device_properties(gi.device).multi_processor_count
    if num_programs > sm_count:
        raise RuntimeError(
            f"persistent grid {num_programs} > SM count {sm_count}; "
            f"would deadlock on the spin-wait barrier. Increase block sizes."
        )

    out = torch.empty((T, B, H), device=gi.device, dtype=gi.dtype)
    barrier = torch.zeros((T,), device=gi.device, dtype=torch.int32)

    in_s, in_qmin, in_qmax = h_in_quant or (1.0, -2**31, 2**31 - 1)
    out_s, out_qmin, out_qmax = h_out_quant or (1.0, -2**31, 2**31 - 1)

    grid = (n_pid_b, n_pid_oh)
    gru_scan_fwd_persistent_kernel[grid](
        gi, h0, Wh_cat, bh_cat, out,
        barrier,
        T, B,
        gi.stride(0), gi.stride(1),
        h0.stride(0),
        Wh_cat.stride(0),
        out.stride(0), out.stride(1),
        num_programs,
        in_s, in_qmin, in_qmax,
        out_s, out_qmin, out_qmax,
        H=H,
        BLOCK_B=block_b,
        BLOCK_OH=block_oh,
        BLOCK_K=block_k,
        QUANT_H_IN=h_in_quant is not None,
        QUANT_H_OUT=h_out_quant is not None,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


@triton.jit  # type: ignore[untyped-decorator]
def gru_scan_bwd_persistent_kernel(  # type: ignore[no-untyped-def]
    # forward inputs (read-only)
    gi_ptr,           # [T, B, 3H]
    h0_ptr,           # [B, H]
    Wh_ptr,           # [3H, H]
    bh_ptr,           # [3H]
    out_ptr,          # [T, B, H]
    # upstream gradient
    dout_ptr,         # [T, B, H]
    # outputs
    dgi_ptr,          # [T, B, 3H]
    dh0_ptr,          # [B, H]
    # per-batch-tile output buffers (reduced across pid_oh inside the kernel
    # via atomic_add, then summed across pid_b in Python)
    dWh_partial_ptr,  # [num_pid_b, 3H, H]
    dbh_partial_ptr,  # [num_pid_b, 3H]
    # ping-pong dh_acc scratch — atomic-add target for cross-program
    # coordination of the dh_prev_via_W contribution. Both [B, H] fp32.
    dh_a_ptr,
    dh_b_ptr,
    # per-timestep barrier counter for cross-CTA sync, [T] int32
    barrier_ptr,
    T, B,
    sg_t, sg_b,
    sh0_b,
    sW_o,
    so_t, so_b,
    sdo_t, sdo_b,
    sdgi_t, sdgi_b,
    sdh0_b,
    sdWp_pid, sdWp_o,
    sdbp_pid,
    sdh_b,
    NUM_PROGRAMS,
    h_in_scale,
    h_in_qmin,
    h_in_qmax,
    h_out_scale,
    h_out_qmin,
    h_out_qmax,
    H: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_OH: tl.constexpr,
    BLOCK_K: tl.constexpr,
    QUANT_H_IN: tl.constexpr,
    QUANT_H_OUT: tl.constexpr,
):
    """Persistent backward kernel.

    2D grid (pid_b, pid_oh) mirroring the persistent forward. Per-step,
    each program:
    - Reads its [BLOCK_B, BLOCK_OH] slice of incoming dh_acc and zeros
      that slice (preparing it to be the next step's write buffer after
      the ping-pong).
    - Recomputes forward gates for its slice.
    - Stores its slice of dgi[t].
    - Atomic-adds the cross-H dh_prev_via_W contribution into the write
      buffer (matmul output spans all of H).
    - Atomic-adds dh_prev_direct into the write buffer at its OH slice
      (per-element, one writer).
    - Atomic-adds dWh_partial / dbh_partial slabs (each program owns a
      unique OH range, no contention there).
    - Spin-waits on the per-timestep barrier.

    Constraint as in the forward: total grid programs <= GPU SM count.
    """
    pid_b = tl.program_id(0)
    pid_oh = tl.program_id(1)
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = offs_b < B
    offs_oh = pid_oh * BLOCK_OH + tl.arange(0, BLOCK_OH)
    mask_oh = offs_oh < H
    mask_oh2 = mask_b[:, None] & mask_oh[None, :]

    # Pre-load bh slice for this oh range — constant across T.
    bhr_tile = tl.load(bh_ptr + 0 * H + offs_oh, mask=mask_oh, other=0.0)
    bhz_tile = tl.load(bh_ptr + 1 * H + offs_oh, mask=mask_oh, other=0.0)
    bhn_tile = tl.load(bh_ptr + 2 * H + offs_oh, mask=mask_oh, other=0.0)

    read_ptr = dh_a_ptr
    write_ptr = dh_b_ptr

    for t_rev in range(0, T):
        t = T - 1 - t_rev

        if t == 0:
            h_prev_ptr = h0_ptr
            sh_prev_b = sh0_b
        else:
            h_prev_ptr = out_ptr + (t - 1) * so_t
            sh_prev_b = so_b

        # ---- Recompute forward gh tiles for this oh range ----
        ghr = tl.zeros((BLOCK_B, BLOCK_OH), dtype=tl.float32)
        ghz = tl.zeros((BLOCK_B, BLOCK_OH), dtype=tl.float32)
        ghn = tl.zeros((BLOCK_B, BLOCK_OH), dtype=tl.float32)
        for k in range(0, H, BLOCK_K):
            offs_k = k + tl.arange(0, BLOCK_K)
            mask_k = offs_k < H
            h_prev_ptrs = h_prev_ptr + offs_b[:, None] * sh_prev_b + offs_k[None, :]
            h_prev_tile = tl.load(
                h_prev_ptrs, mask=mask_b[:, None] & mask_k[None, :], other=0.0,
            )
            if QUANT_H_IN:
                q = tl.extra.cuda.libdevice.rint(h_prev_tile / h_in_scale)
                q = tl.minimum(tl.maximum(q, h_in_qmin), h_in_qmax)
                h_prev_tile = q * h_in_scale
            W_offset = offs_oh[:, None] * sW_o + offs_k[None, :]
            Wr_tile = tl.load(
                Wh_ptr + 0 * H * sW_o + W_offset,
                mask=mask_oh[:, None] & mask_k[None, :], other=0.0,
            )
            Wz_tile = tl.load(
                Wh_ptr + 1 * H * sW_o + W_offset,
                mask=mask_oh[:, None] & mask_k[None, :], other=0.0,
            )
            Wn_tile = tl.load(
                Wh_ptr + 2 * H * sW_o + W_offset,
                mask=mask_oh[:, None] & mask_k[None, :], other=0.0,
            )
            ghr += tl.dot(h_prev_tile, tl.trans(Wr_tile), input_precision="tf32")
            ghz += tl.dot(h_prev_tile, tl.trans(Wz_tile), input_precision="tf32")
            ghn += tl.dot(h_prev_tile, tl.trans(Wn_tile), input_precision="tf32")
        ghr += bhr_tile[None, :]
        ghz += bhz_tile[None, :]
        ghn += bhn_tile[None, :]

        gi_base = (
            gi_ptr + t * sg_t + offs_b[:, None] * sg_b + offs_oh[None, :]
        )
        gir = tl.load(gi_base + 0 * H, mask=mask_oh2, other=0.0)
        giz = tl.load(gi_base + 1 * H, mask=mask_oh2, other=0.0)
        gin = tl.load(gi_base + 2 * H, mask=mask_oh2, other=0.0)
        r = tl.sigmoid(gir + ghr)
        z = tl.sigmoid(giz + ghz)
        n = tl.extra.cuda.libdevice.tanh(gin + r * ghn)

        h_prev_oh_ptrs = (
            h_prev_ptr + offs_b[:, None] * sh_prev_b + offs_oh[None, :]
        )
        h_prev_oh = tl.load(h_prev_oh_ptrs, mask=mask_oh2, other=0.0)

        # ---- Read incoming dh_acc and zero it (it'll be next step's write
        # buffer after the swap; each cell has exactly one reader). ----
        # cache_modifier=".cv" forces a fresh global load — needed because
        # atomic_add writes from sibling CTAs in the previous timestep may
        # not be visible through L1 cache without it.
        dh_acc_ptrs = read_ptr + offs_b[:, None] * sdh_b + offs_oh[None, :]
        dh_acc_oh = tl.load(
            dh_acc_ptrs, mask=mask_oh2, other=0.0,
        )
        tl.store(dh_acc_ptrs, tl.zeros_like(dh_acc_oh), mask=mask_oh2)

        dout_base = (
            dout_ptr + t * sdo_t + offs_b[:, None] * sdo_b + offs_oh[None, :]
        )
        dout_oh = tl.load(dout_base, mask=mask_oh2, other=0.0)
        dh_t = dout_oh + dh_acc_oh

        # STE backward of quant_h_out: incoming dh_t is grad on the
        # quantized h_t; multiply by clip mask of h_t_raw to get grad on
        # h_t_raw before propagating through the recurrence.
        if QUANT_H_OUT:
            h_t_raw = (1.0 - z) * n + z * h_prev_oh
            q_unclamped = tl.extra.cuda.libdevice.rint(h_t_raw / h_out_scale)
            mask_out = (q_unclamped >= h_out_qmin) & (q_unclamped <= h_out_qmax)
            dh_t = tl.where(mask_out, dh_t, 0.0)

        # h_t = (1 - z) * n + z * h_prev
        dn = dh_t * (1.0 - z)
        dz = dh_t * (h_prev_oh - n)
        dh_prev_direct = dh_t * z

        dgn_pre = dn * (1.0 - n * n)
        dgi_n = dgn_pre
        dr = dgn_pre * ghn
        dgh_n = dgn_pre * r

        dgz_pre = dz * z * (1.0 - z)
        dgi_z = dgz_pre
        dgh_z = dgz_pre

        dgr_pre = dr * r * (1.0 - r)
        dgi_r = dgr_pre
        dgh_r = dgr_pre

        # Store dgi[t][oh_slice] (no contention — each program owns its OH).
        tl.store(
            dgi_ptr + t * sdgi_t + offs_b[:, None] * sdgi_b + offs_oh[None, :] + 0 * H,
            dgi_r, mask=mask_oh2,
        )
        tl.store(
            dgi_ptr + t * sdgi_t + offs_b[:, None] * sdgi_b + offs_oh[None, :] + 1 * H,
            dgi_z, mask=mask_oh2,
        )
        tl.store(
            dgi_ptr + t * sdgi_t + offs_b[:, None] * sdgi_b + offs_oh[None, :] + 2 * H,
            dgi_n, mask=mask_oh2,
        )

        # dbh_partial accumulation — each program owns its OH range and is
        # the only writer to its rows; plain load+add+store (no atomic).
        dbh_base = dbh_partial_ptr + pid_b * sdbp_pid + offs_oh
        tl.store(
            dbh_base + 0 * H,
            tl.load(dbh_base + 0 * H, mask=mask_oh, other=0.0)
            + tl.sum(dgh_r, axis=0),
            mask=mask_oh,
        )
        tl.store(
            dbh_base + 1 * H,
            tl.load(dbh_base + 1 * H, mask=mask_oh, other=0.0)
            + tl.sum(dgh_z, axis=0),
            mask=mask_oh,
        )
        tl.store(
            dbh_base + 2 * H,
            tl.load(dbh_base + 2 * H, mask=mask_oh, other=0.0)
            + tl.sum(dgh_n, axis=0),
            mask=mask_oh,
        )

        # dh_prev_via_W: dgh @ Wh — accumulates into write_ptr ALL of H,
        # which is shared across pid_oh programs. Use atomic_add.
        for k in range(0, H, BLOCK_K):
            offs_k = k + tl.arange(0, BLOCK_K)
            mask_k = offs_k < H
            W_offset = offs_oh[:, None] * sW_o + offs_k[None, :]
            Wr_t = tl.load(
                Wh_ptr + 0 * H * sW_o + W_offset,
                mask=mask_oh[:, None] & mask_k[None, :], other=0.0,
            )
            Wz_t = tl.load(
                Wh_ptr + 1 * H * sW_o + W_offset,
                mask=mask_oh[:, None] & mask_k[None, :], other=0.0,
            )
            Wn_t = tl.load(
                Wh_ptr + 2 * H * sW_o + W_offset,
                mask=mask_oh[:, None] & mask_k[None, :], other=0.0,
            )
            contrib = (
                tl.dot(dgh_r, Wr_t, input_precision="tf32")
                + tl.dot(dgh_z, Wz_t, input_precision="tf32")
                + tl.dot(dgh_n, Wn_t, input_precision="tf32")
            )
            # STE backward of quant_h_in: contrib is grad on the quantized
            # h_prev (matmul side); zero where the value was clipped.
            if QUANT_H_IN:
                h_prev_k_ptrs = (
                    h_prev_ptr + offs_b[:, None] * sh_prev_b + offs_k[None, :]
                )
                h_prev_k = tl.load(
                    h_prev_k_ptrs,
                    mask=mask_b[:, None] & mask_k[None, :],
                    other=0.0,
                )
                q_in_unclamped = tl.extra.cuda.libdevice.rint(h_prev_k / h_in_scale)
                mask_in = (q_in_unclamped >= h_in_qmin) & (
                    q_in_unclamped <= h_in_qmax
                )
                contrib = tl.where(mask_in, contrib, 0.0)
            w_ptrs = write_ptr + offs_b[:, None] * sdh_b + offs_k[None, :]
            tl.atomic_add(
                w_ptrs, contrib,
                mask=mask_b[:, None] & mask_k[None, :],
            )

        # dh_prev_direct goes to write_ptr at this program's oh slice; each
        # cell has exactly one writer (only this program owns this OH), so
        # plain store is sufficient — but we use atomic_add to play nicely
        # with the via_W writes that may also hit this OH range.
        wd_ptrs = write_ptr + offs_b[:, None] * sdh_b + offs_oh[None, :]
        tl.atomic_add(wd_ptrs, dh_prev_direct, mask=mask_oh2)

        # dWh_partial accumulation: each program owns unique rows in
        # dWh_partial[pid_b, this_oh, :], so plain load+add+store works.
        for k in range(0, H, BLOCK_K):
            offs_k = k + tl.arange(0, BLOCK_K)
            mask_k = offs_k < H
            h_prev_ptrs = (
                h_prev_ptr + offs_b[:, None] * sh_prev_b + offs_k[None, :]
            )
            h_prev_tile = tl.load(
                h_prev_ptrs, mask=mask_b[:, None] & mask_k[None, :], other=0.0,
            )
            # Forward used hq = quant_h_in(h_prev) in the matmul, so dWh
            # accumulates against hq, not raw h_prev.
            if QUANT_H_IN:
                q = tl.extra.cuda.libdevice.rint(h_prev_tile / h_in_scale)
                q = tl.minimum(tl.maximum(q, h_in_qmin), h_in_qmax)
                h_prev_tile = q * h_in_scale
            dWr = tl.dot(tl.trans(dgh_r), h_prev_tile, input_precision="tf32")
            dWz = tl.dot(tl.trans(dgh_z), h_prev_tile, input_precision="tf32")
            dWn = tl.dot(tl.trans(dgh_n), h_prev_tile, input_precision="tf32")
            dWh_base = dWh_partial_ptr + pid_b * sdWp_pid
            Wr_ptrs = (
                dWh_base + (0 * H + offs_oh)[:, None] * sdWp_o + offs_k[None, :]
            )
            Wz_ptrs = (
                dWh_base + (1 * H + offs_oh)[:, None] * sdWp_o + offs_k[None, :]
            )
            Wn_ptrs = (
                dWh_base + (2 * H + offs_oh)[:, None] * sdWp_o + offs_k[None, :]
            )
            mask_okok = mask_oh[:, None] & mask_k[None, :]
            tl.store(
                Wr_ptrs,
                tl.load(Wr_ptrs, mask=mask_okok, other=0.0) + dWr,
                mask=mask_okok,
            )
            tl.store(
                Wz_ptrs,
                tl.load(Wz_ptrs, mask=mask_okok, other=0.0) + dWz,
                mask=mask_okok,
            )
            tl.store(
                Wn_ptrs,
                tl.load(Wn_ptrs, mask=mask_okok, other=0.0) + dWn,
                mask=mask_okok,
            )

        # Cross-CTA barrier. Release on the increment so prior data
        # atomic_adds are visible to any reader observing the post-
        # increment counter; acquire on the spin-load so writes by
        # programs that already incremented are visible after the wait.
        # tl.load doesn't accept a memory order, so we read via a no-op
        # atomic_add (add 0) with sem="acquire".
        tl.atomic_add(barrier_ptr + t_rev, 1, sem="release")
        done = tl.atomic_add(barrier_ptr + t_rev, 0, sem="acquire")
        while done < NUM_PROGRAMS:
            done = tl.atomic_add(barrier_ptr + t_rev, 0, sem="acquire")

        # Swap read/write for next step.
        tmp = read_ptr
        read_ptr = write_ptr
        write_ptr = tmp

    # After the loop, read_ptr holds the final dh_acc — write to dh0.
    # cache_modifier=".cv" same reason as the in-loop dh_acc read: this
    # buffer was just atomic-added to by sibling CTAs and the L1 view may
    # be stale.
    dh_final_ptrs = read_ptr + offs_b[:, None] * sdh_b + offs_oh[None, :]
    dh_final = tl.load(
        dh_final_ptrs, mask=mask_oh2, other=0.0,
    )
    dh0_ptrs = dh0_ptr + offs_b[:, None] * sdh0_b + offs_oh[None, :]
    tl.store(dh0_ptrs, dh_final, mask=mask_oh2)


def gru_scan_backward_persistent(
    gi: torch.Tensor,
    h0: torch.Tensor,
    Wh_cat: torch.Tensor,
    bh_cat: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    *,
    block_b: int = 16,
    block_oh: int = 64,
    block_k: int = 32,
    num_warps: int = 4,
    num_stages: int = 2,
    h_in_quant: tuple[float, int, int] | None = None,
    h_out_quant: tuple[float, int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Persistent backward. Same launch constraint as the forward kernel.

    BLOCK_B is fixed at 16 (vs 8 in the forward) because the dWh
    accumulation does ``tl.dot(trans(dgh), h_prev)`` with reduction
    K = BLOCK_B; tl.dot requires K >= 16.
    """
    T, B, three_H = gi.shape
    H = three_H // 3
    if h0.shape != (B, H):
        raise ValueError(f"h0 shape must be (B, H)=({B}, {H}); got {tuple(h0.shape)}")
    if Wh_cat.shape != (3 * H, H):
        raise ValueError(
            f"Wh_cat shape must be (3H, H)=({3 * H}, {H}); got {tuple(Wh_cat.shape)}"
        )
    if bh_cat.shape != (3 * H,):
        raise ValueError(
            f"bh_cat shape must be (3H,)=({3 * H},); got {tuple(bh_cat.shape)}"
        )
    if out.shape != (T, B, H):
        raise ValueError(
            f"out shape must be (T, B, H)=({T}, {B}, {H}); got {tuple(out.shape)}"
        )
    if dout.shape != (T, B, H):
        raise ValueError(
            f"dout shape must be (T, B, H)=({T}, {B}, {H}); got {tuple(dout.shape)}"
        )

    gi = gi.contiguous()
    h0 = h0.contiguous()
    Wh_cat = Wh_cat.contiguous()
    bh_cat = bh_cat.contiguous()
    out = out.contiguous()
    dout = dout.contiguous()

    block_oh = max(16, min(block_oh, H))
    block_k = max(16, min(block_k, H))

    n_pid_b = triton.cdiv(B, block_b)
    n_pid_oh = triton.cdiv(H, block_oh)
    num_programs = n_pid_b * n_pid_oh

    sm_count = torch.cuda.get_device_properties(gi.device).multi_processor_count
    if num_programs > sm_count:
        raise RuntimeError(
            f"persistent grid {num_programs} > SM count {sm_count}; "
            f"would deadlock on the spin-wait barrier. Increase block sizes."
        )

    dgi = torch.zeros_like(gi)
    dh0 = torch.zeros_like(h0)
    dWh_partial = torch.zeros((n_pid_b, 3 * H, H), device=gi.device, dtype=gi.dtype)
    dbh_partial = torch.zeros((n_pid_b, 3 * H), device=gi.device, dtype=gi.dtype)
    # Both ping-pong buffers start zeroed.
    dh_a = torch.zeros((B, H), device=gi.device, dtype=gi.dtype)
    dh_b = torch.zeros((B, H), device=gi.device, dtype=gi.dtype)
    barrier = torch.zeros((T,), device=gi.device, dtype=torch.int32)

    in_s, in_qmin, in_qmax = h_in_quant or (1.0, -2**31, 2**31 - 1)
    out_s, out_qmin, out_qmax = h_out_quant or (1.0, -2**31, 2**31 - 1)

    grid = (n_pid_b, n_pid_oh)
    gru_scan_bwd_persistent_kernel[grid](
        gi, h0, Wh_cat, bh_cat, out,
        dout,
        dgi, dh0,
        dWh_partial, dbh_partial,
        dh_a, dh_b,
        barrier,
        T, B,
        gi.stride(0), gi.stride(1),
        h0.stride(0),
        Wh_cat.stride(0),
        out.stride(0), out.stride(1),
        dout.stride(0), dout.stride(1),
        dgi.stride(0), dgi.stride(1),
        dh0.stride(0),
        dWh_partial.stride(0), dWh_partial.stride(1),
        dbh_partial.stride(0),
        dh_a.stride(0),
        num_programs,
        in_s, in_qmin, in_qmax,
        out_s, out_qmin, out_qmax,
        H=H, BLOCK_B=block_b, BLOCK_OH=block_oh, BLOCK_K=block_k,
        QUANT_H_IN=h_in_quant is not None,
        QUANT_H_OUT=h_out_quant is not None,
        num_warps=num_warps, num_stages=num_stages,
    )

    dWh = dWh_partial.sum(dim=0)
    dbh = dbh_partial.sum(dim=0)
    return dgi, dh0, dWh, dbh


@triton.autotune(configs=_AUTOTUNE_CONFIGS_FWD, key=["T", "B"])  # type: ignore[untyped-decorator]
@triton.jit  # type: ignore[untyped-decorator]
def gru_scan_fwd_kernel(  # type: ignore[no-untyped-def]
    gi_ptr,            # [T, B, 3H], fp32
    h0_ptr,            # [B, H], fp32
    Wh_ptr,            # [3H, H], fp32 (rows: r, z, n stacked)
    bh_ptr,            # [3H], fp32
    out_ptr,           # [T, B, H], fp32 (hidden state at each timestep)
    # shape
    T,
    B,
    # strides (in elements)
    sg_t, sg_b,        # gi: time, batch (last dim contiguous)
    sh0_b,             # h0: batch (last dim contiguous)
    sW_o,              # Wh: output dim (last dim contiguous)
    so_t, so_b,        # out: time, batch (last dim contiguous)
    # fake-quant params (per-tensor symmetric, frozen scale).
    # When QUANT_H_IN/OUT is False the corresponding scale is unused.
    h_in_scale,
    h_in_qmin,
    h_in_qmax,
    h_out_scale,
    h_out_qmin,
    h_out_qmax,
    # constexprs
    H: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_OH: tl.constexpr,
    BLOCK_K: tl.constexpr,
    QUANT_H_IN: tl.constexpr,
    QUANT_H_OUT: tl.constexpr,
):
    pid_b = tl.program_id(0)
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = offs_b < B

    # Per-step pointer to current hidden state. Starts at h0; after step t
    # we read from out[t]. Two pointer variables avoid a branch per step.
    h_in_ptr = h0_ptr
    sh_b = sh0_b

    for t in range(0, T):
        # Process output H in BLOCK_OH-wide tiles. Each tile is independent
        # in the elementwise recurrence; only the matmuls have to span all
        # of K=H and they reduce inside this loop.
        for oh in range(0, H, BLOCK_OH):
            offs_oh = oh + tl.arange(0, BLOCK_OH)
            mask_oh = offs_oh < H

            # Three accumulators for the three gates (r, z, n hidden side).
            ghr = tl.zeros((BLOCK_B, BLOCK_OH), dtype=tl.float32)
            ghz = tl.zeros((BLOCK_B, BLOCK_OH), dtype=tl.float32)
            ghn = tl.zeros((BLOCK_B, BLOCK_OH), dtype=tl.float32)

            for k in range(0, H, BLOCK_K):
                offs_k = k + tl.arange(0, BLOCK_K)
                mask_k = offs_k < H

                # Load h tile: [BLOCK_B, BLOCK_K]
                h_ptrs = (
                    h_in_ptr
                    + offs_b[:, None] * sh_b
                    + offs_k[None, :]
                )
                h_tile = tl.load(
                    h_ptrs,
                    mask=mask_b[:, None] & mask_k[None, :],
                    other=0.0,
                )

                # Apply quant_h_in to the matmul-side h. Per-tensor symmetric:
                # q = clamp(round(x / s), qmin, qmax); out = q * s.
                # Direct contribution to h_new uses the *unquantized* h (see
                # gru_cell.step), so we only mutate this matmul-local copy.
                if QUANT_H_IN:
                    q = tl.extra.cuda.libdevice.rint(h_tile / h_in_scale)
                    q = tl.minimum(tl.maximum(q, h_in_qmin), h_in_qmax)
                    h_tile = q * h_in_scale

                # Load Wh tiles: each [BLOCK_OH, BLOCK_K].
                # Wh layout is row-major [3H, H]; sW_o is the row stride (= H).
                # gate r is rows [0, H), z is [H, 2H), n is [2H, 3H).
                W_offset = offs_oh[:, None] * sW_o + offs_k[None, :]
                Wr_tile = tl.load(
                    Wh_ptr + 0 * H * sW_o + W_offset,
                    mask=mask_oh[:, None] & mask_k[None, :],
                    other=0.0,
                )
                Wz_tile = tl.load(
                    Wh_ptr + 1 * H * sW_o + W_offset,
                    mask=mask_oh[:, None] & mask_k[None, :],
                    other=0.0,
                )
                Wn_tile = tl.load(
                    Wh_ptr + 2 * H * sW_o + W_offset,
                    mask=mask_oh[:, None] & mask_k[None, :],
                    other=0.0,
                )

                # Wh[o, i] is the row-major weight; tl.dot(h, W^T) gives
                # the standard F.linear(h, Wh) output for this output tile.
                # TF32 input precision — uses tensor cores, ~3x faster than
                # ieee fp32 with ~10-bit-mantissa noise that doesn't matter
                # for QAT (the fake-quant noise dominates).
                ghr += tl.dot(h_tile, tl.trans(Wr_tile), input_precision="tf32")
                ghz += tl.dot(h_tile, tl.trans(Wz_tile), input_precision="tf32")
                ghn += tl.dot(h_tile, tl.trans(Wn_tile), input_precision="tf32")

            # Add hidden-side biases.
            bhr_tile = tl.load(bh_ptr + 0 * H + offs_oh, mask=mask_oh, other=0.0)
            bhz_tile = tl.load(bh_ptr + 1 * H + offs_oh, mask=mask_oh, other=0.0)
            bhn_tile = tl.load(bh_ptr + 2 * H + offs_oh, mask=mask_oh, other=0.0)
            ghr += bhr_tile[None, :]
            ghz += bhz_tile[None, :]
            ghn += bhn_tile[None, :]

            # Load corresponding gi[t] slices for this oh tile.
            gi_base = (
                gi_ptr
                + t * sg_t
                + offs_b[:, None] * sg_b
                + offs_oh[None, :]
            )
            mask_oh2 = mask_b[:, None] & mask_oh[None, :]
            gir = tl.load(gi_base + 0 * H, mask=mask_oh2, other=0.0)
            giz = tl.load(gi_base + 1 * H, mask=mask_oh2, other=0.0)
            gin = tl.load(gi_base + 2 * H, mask=mask_oh2, other=0.0)

            # Gate math (matches gru_cell.step_with_gi, no fake-quant).
            r = tl.sigmoid(gir + ghr)
            z = tl.sigmoid(giz + ghz)
            n = tl.extra.cuda.libdevice.tanh(gin + r * ghn)

            # h_new = (1 - z) * n + z * h_old. Need h_old for THIS oh tile.
            h_old_ptrs = (
                h_in_ptr + offs_b[:, None] * sh_b + offs_oh[None, :]
            )
            h_old = tl.load(h_old_ptrs, mask=mask_oh2, other=0.0)
            h_new = (1.0 - z) * n + z * h_old

            # Apply quant_h_out before storing — the next step will read
            # this back as h_prev, so it sees the post-quant value.
            if QUANT_H_OUT:
                q = tl.extra.cuda.libdevice.rint(h_new / h_out_scale)
                q = tl.minimum(tl.maximum(q, h_out_qmin), h_out_qmax)
                h_new = q * h_out_scale

            # Store h_new to out[t, :, oh:oh+BLOCK_OH].
            out_ptrs = (
                out_ptr
                + t * so_t
                + offs_b[:, None] * so_b
                + offs_oh[None, :]
            )
            tl.store(out_ptrs, h_new, mask=mask_oh2)

        # Next step reads from out[t]; switch the source pointer.
        h_in_ptr = out_ptr + t * so_t
        sh_b = so_b


@triton.autotune(configs=_AUTOTUNE_CONFIGS_BWD, key=["T", "B"])  # type: ignore[untyped-decorator]
@triton.jit  # type: ignore[untyped-decorator]
def gru_scan_bwd_kernel(  # type: ignore[no-untyped-def]
    # forward inputs (saved tensors, read-only)
    gi_ptr,           # [T, B, 3H]
    h0_ptr,           # [B, H]
    Wh_ptr,           # [3H, H]
    bh_ptr,           # [3H]
    out_ptr,          # [T, B, H]
    # backward upstream
    dout_ptr,         # [T, B, H]
    # backward outputs
    dgi_ptr,          # [T, B, 3H]
    dh0_ptr,          # [B, H]
    # per-program partial buffers (reduced across pid in Python wrapper)
    dWh_partial_ptr,  # [num_pid, 3H, H]
    dbh_partial_ptr,  # [num_pid, 3H]
    # scratch: two ping-pong buffers for dh_acc state, [B, H] each
    dh_a_ptr,
    dh_b_ptr,
    # shape
    T,
    B,
    # strides (in elements)
    sg_t, sg_b,
    sh0_b,
    sW_o,
    so_t, so_b,
    sdo_t, sdo_b,
    sdgi_t, sdgi_b,
    sdh0_b,
    sdWp_pid, sdWp_o,
    sdbp_pid,
    sdh_b,
    # fake-quant params (per-tensor symmetric, frozen scale)
    h_in_scale,
    h_in_qmin,
    h_in_qmax,
    h_out_scale,
    h_out_qmin,
    h_out_qmax,
    # constexprs
    H: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_OH: tl.constexpr,
    BLOCK_K: tl.constexpr,
    QUANT_H_IN: tl.constexpr,
    QUANT_H_OUT: tl.constexpr,
):
    """Backward kernel for the multi-step GRU scan.

    Mirrors the forward layout: one program per batch tile, hidden state
    sharded across batch only. Per timestep (in reverse), recomputes the
    forward gate values, then walks gradients through the recurrence.

    State management:
    - ``dh_acc`` carries the accumulated hidden gradient between
      timesteps. Two scratch buffers (``dh_a_ptr``/``dh_b_ptr``) ping-pong
      so a step reads from one while writing the next into the other.
    - ``dWh_partial`` and ``dbh_partial`` are per-program partials; the
      Python wrapper sums across the program dim to produce the final
      ``dWh`` and ``dbh``.

    Within a step we iterate over output-H tiles (``BLOCK_OH``-wide) and
    keep all matmuls TF32 for tensor-core throughput, matching the
    forward kernel's precision.
    """
    pid_b = tl.program_id(0)
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = offs_b < B

    # Initialize the read buffer (first step's incoming dh_acc) to zero.
    for h_off in range(0, H, BLOCK_K):
        offs_h = h_off + tl.arange(0, BLOCK_K)
        mask_h = offs_h < H
        a_ptrs = dh_a_ptr + offs_b[:, None] * sdh_b + offs_h[None, :]
        tl.store(
            a_ptrs,
            tl.zeros((BLOCK_B, BLOCK_K), dtype=tl.float32),
            mask=mask_b[:, None] & mask_h[None, :],
        )

    # Zero this program's slab of dWh_partial and dbh_partial. The recurrence
    # loop below does load-add-store into these (each program owns unique
    # rows so no atomic needed), which is only correct if the slab is zero on
    # entry. The Python wrapper allocates with torch.zeros, but @triton.autotune
    # reuses the same buffer across all trial configs — every trial accumulates
    # into the prior trial's result. Zeroing per-program here makes each
    # kernel launch idempotent.
    for row_off in range(0, 3 * H, BLOCK_OH):
        offs_row = row_off + tl.arange(0, BLOCK_OH)
        mask_row = offs_row < 3 * H
        for col_off in range(0, H, BLOCK_K):
            offs_col = col_off + tl.arange(0, BLOCK_K)
            mask_col = offs_col < H
            dWp_ptrs = (
                dWh_partial_ptr
                + pid_b * sdWp_pid
                + offs_row[:, None] * sdWp_o
                + offs_col[None, :]
            )
            tl.store(
                dWp_ptrs,
                tl.zeros((BLOCK_OH, BLOCK_K), dtype=tl.float32),
                mask=mask_row[:, None] & mask_col[None, :],
            )
        dbp_ptrs = dbh_partial_ptr + pid_b * sdbp_pid + offs_row
        tl.store(
            dbp_ptrs,
            tl.zeros((BLOCK_OH,), dtype=tl.float32),
            mask=mask_row,
        )

    # Ping-pong: at each step t, we read from `read_ptr` and write the new
    # dh_acc into `write_ptr`, then swap.
    read_ptr = dh_a_ptr
    write_ptr = dh_b_ptr

    for t_rev in range(0, T):
        t = T - 1 - t_rev

        # h_prev source: out[t-1] for t > 0, h0 for t == 0.
        # `t == 0` is uniform across the program so this branch is cheap.
        if t == 0:
            h_prev_ptr = h0_ptr
            sh_prev_b = sh0_b
        else:
            h_prev_ptr = out_ptr + (t - 1) * so_t
            sh_prev_b = so_b

        # Zero the write buffer so we can accumulate into it.
        for h_off in range(0, H, BLOCK_K):
            offs_h = h_off + tl.arange(0, BLOCK_K)
            mask_h = offs_h < H
            w_ptrs = write_ptr + offs_b[:, None] * sdh_b + offs_h[None, :]
            tl.store(
                w_ptrs,
                tl.zeros((BLOCK_B, BLOCK_K), dtype=tl.float32),
                mask=mask_b[:, None] & mask_h[None, :],
            )

        for oh in range(0, H, BLOCK_OH):
            offs_oh = oh + tl.arange(0, BLOCK_OH)
            mask_oh = offs_oh < H
            mask_oh2 = mask_b[:, None] & mask_oh[None, :]

            # ---- Recompute forward gh tiles for this oh range ----
            ghr = tl.zeros((BLOCK_B, BLOCK_OH), dtype=tl.float32)
            ghz = tl.zeros((BLOCK_B, BLOCK_OH), dtype=tl.float32)
            ghn = tl.zeros((BLOCK_B, BLOCK_OH), dtype=tl.float32)

            for k in range(0, H, BLOCK_K):
                offs_k = k + tl.arange(0, BLOCK_K)
                mask_k = offs_k < H

                h_prev_ptrs = (
                    h_prev_ptr + offs_b[:, None] * sh_prev_b + offs_k[None, :]
                )
                h_prev_tile = tl.load(
                    h_prev_ptrs,
                    mask=mask_b[:, None] & mask_k[None, :],
                    other=0.0,
                )

                # Apply quant_h_in to match the forward kernel.
                if QUANT_H_IN:
                    q = tl.extra.cuda.libdevice.rint(h_prev_tile / h_in_scale)
                    q = tl.minimum(tl.maximum(q, h_in_qmin), h_in_qmax)
                    h_prev_tile = q * h_in_scale

                W_offset = offs_oh[:, None] * sW_o + offs_k[None, :]
                Wr_tile = tl.load(
                    Wh_ptr + 0 * H * sW_o + W_offset,
                    mask=mask_oh[:, None] & mask_k[None, :],
                    other=0.0,
                )
                Wz_tile = tl.load(
                    Wh_ptr + 1 * H * sW_o + W_offset,
                    mask=mask_oh[:, None] & mask_k[None, :],
                    other=0.0,
                )
                Wn_tile = tl.load(
                    Wh_ptr + 2 * H * sW_o + W_offset,
                    mask=mask_oh[:, None] & mask_k[None, :],
                    other=0.0,
                )

                ghr += tl.dot(h_prev_tile, tl.trans(Wr_tile), input_precision="tf32")
                ghz += tl.dot(h_prev_tile, tl.trans(Wz_tile), input_precision="tf32")
                ghn += tl.dot(h_prev_tile, tl.trans(Wn_tile), input_precision="tf32")

            ghr += tl.load(bh_ptr + 0 * H + offs_oh, mask=mask_oh, other=0.0)[None, :]
            ghz += tl.load(bh_ptr + 1 * H + offs_oh, mask=mask_oh, other=0.0)[None, :]
            ghn += tl.load(bh_ptr + 2 * H + offs_oh, mask=mask_oh, other=0.0)[None, :]

            # gi[t] slices for this oh range
            gi_base = (
                gi_ptr + t * sg_t + offs_b[:, None] * sg_b + offs_oh[None, :]
            )
            gir = tl.load(gi_base + 0 * H, mask=mask_oh2, other=0.0)
            giz = tl.load(gi_base + 1 * H, mask=mask_oh2, other=0.0)
            gin = tl.load(gi_base + 2 * H, mask=mask_oh2, other=0.0)

            r = tl.sigmoid(gir + ghr)
            z = tl.sigmoid(giz + ghz)
            n = tl.extra.cuda.libdevice.tanh(gin + r * ghn)

            # h_prev at this oh tile (for h_prev_direct in dz and dh_prev_direct)
            h_prev_oh_ptrs = (
                h_prev_ptr + offs_b[:, None] * sh_prev_b + offs_oh[None, :]
            )
            h_prev_oh = tl.load(h_prev_oh_ptrs, mask=mask_oh2, other=0.0)

            # ---- Backward starts here ----
            # Incoming dh_acc[oh] from prev iteration (read from read_ptr).
            dh_acc_ptrs = read_ptr + offs_b[:, None] * sdh_b + offs_oh[None, :]
            dh_acc_oh = tl.load(dh_acc_ptrs, mask=mask_oh2, other=0.0)

            # dout[t][oh]
            dout_base = (
                dout_ptr + t * sdo_t + offs_b[:, None] * sdo_b + offs_oh[None, :]
            )
            dout_oh = tl.load(dout_base, mask=mask_oh2, other=0.0)

            dh_t = dout_oh + dh_acc_oh

            # STE backward of quant_h_out: incoming dh_t is gradient on the
            # quantized h_t (= post-quant_h_out). To get gradient on h_t_raw
            # we multiply by the clip mask (1 inside qrange, 0 outside).
            # Recompute h_t_raw to derive the mask.
            if QUANT_H_OUT:
                h_t_raw = (1.0 - z) * n + z * h_prev_oh
                q_unclamped = tl.extra.cuda.libdevice.rint(h_t_raw / h_out_scale)
                mask_out = (q_unclamped >= h_out_qmin) & (q_unclamped <= h_out_qmax)
                dh_t = tl.where(mask_out, dh_t, 0.0)

            # h_t = (1 - z) * n + z * h_prev
            dn = dh_t * (1.0 - z)
            dz = dh_t * (h_prev_oh - n)
            dh_prev_direct = dh_t * z  # contributes to dh_acc at this oh

            # n = tanh(gn_pre);  gn_pre = gi_n + r * gh_n
            dgn_pre = dn * (1.0 - n * n)
            dgi_n = dgn_pre
            dr = dgn_pre * ghn
            dgh_n = dgn_pre * r

            # z = sigmoid(gi_z + gh_z)
            dgz_pre = dz * z * (1.0 - z)
            dgi_z = dgz_pre
            dgh_z = dgz_pre

            # r = sigmoid(gi_r + gh_r)
            dgr_pre = dr * r * (1.0 - r)
            dgi_r = dgr_pre
            dgh_r = dgr_pre

            # Store dgi[t] slices
            tl.store(
                dgi_ptr + t * sdgi_t + offs_b[:, None] * sdgi_b + offs_oh[None, :] + 0 * H,
                dgi_r,
                mask=mask_oh2,
            )
            tl.store(
                dgi_ptr + t * sdgi_t + offs_b[:, None] * sdgi_b + offs_oh[None, :] + 1 * H,
                dgi_z,
                mask=mask_oh2,
            )
            tl.store(
                dgi_ptr + t * sdgi_t + offs_b[:, None] * sdgi_b + offs_oh[None, :] + 2 * H,
                dgi_n,
                mask=mask_oh2,
            )

            # ---- dbh_partial accumulation: sum over batch ----
            dbh_base = dbh_partial_ptr + pid_b * sdbp_pid + offs_oh
            existing_r = tl.load(dbh_base + 0 * H, mask=mask_oh, other=0.0)
            existing_z = tl.load(dbh_base + 1 * H, mask=mask_oh, other=0.0)
            existing_n = tl.load(dbh_base + 2 * H, mask=mask_oh, other=0.0)
            tl.store(
                dbh_base + 0 * H,
                existing_r + tl.sum(dgh_r, axis=0),
                mask=mask_oh,
            )
            tl.store(
                dbh_base + 1 * H,
                existing_z + tl.sum(dgh_z, axis=0),
                mask=mask_oh,
            )
            tl.store(
                dbh_base + 2 * H,
                existing_n + tl.sum(dgh_n, axis=0),
                mask=mask_oh,
            )

            # ---- dh_prev_via_W += dgh @ Wh: matmul reducing over BLOCK_OH ----
            # The result is [BLOCK_B, H] but we accumulate per-K-tile into write buffer.
            for k in range(0, H, BLOCK_K):
                offs_k = k + tl.arange(0, BLOCK_K)
                mask_k = offs_k < H

                W_offset = offs_oh[:, None] * sW_o + offs_k[None, :]
                Wr_t = tl.load(
                    Wh_ptr + 0 * H * sW_o + W_offset,
                    mask=mask_oh[:, None] & mask_k[None, :],
                    other=0.0,
                )
                Wz_t = tl.load(
                    Wh_ptr + 1 * H * sW_o + W_offset,
                    mask=mask_oh[:, None] & mask_k[None, :],
                    other=0.0,
                )
                Wn_t = tl.load(
                    Wh_ptr + 2 * H * sW_o + W_offset,
                    mask=mask_oh[:, None] & mask_k[None, :],
                    other=0.0,
                )

                contrib = (
                    tl.dot(dgh_r, Wr_t, input_precision="tf32")
                    + tl.dot(dgh_z, Wz_t, input_precision="tf32")
                    + tl.dot(dgh_n, Wn_t, input_precision="tf32")
                )

                # STE backward of quant_h_in: contrib is gradient on the
                # quantized h_prev (= input to the matmul). To propagate
                # back to the unquantized h_prev (which is what dh_acc
                # tracks), zero gradient where the value was clipped.
                if QUANT_H_IN:
                    h_prev_k_ptrs = (
                        h_prev_ptr + offs_b[:, None] * sh_prev_b + offs_k[None, :]
                    )
                    h_prev_k = tl.load(
                        h_prev_k_ptrs,
                        mask=mask_b[:, None] & mask_k[None, :],
                        other=0.0,
                    )
                    q_in_unclamped = tl.extra.cuda.libdevice.rint(h_prev_k / h_in_scale)
                    mask_in = (q_in_unclamped >= h_in_qmin) & (
                        q_in_unclamped <= h_in_qmax
                    )
                    contrib = tl.where(mask_in, contrib, 0.0)

                # Add direct contribution from this oh tile (only when k == oh).
                # Easiest: do it outside the k loop by adding to write buffer
                # at offset oh. But since dh_prev_direct is per-oh, we add it
                # at the end of this oh iteration (outside the k loop).

                w_ptrs = write_ptr + offs_b[:, None] * sdh_b + offs_k[None, :]
                existing = tl.load(
                    w_ptrs,
                    mask=mask_b[:, None] & mask_k[None, :],
                    other=0.0,
                )
                tl.store(
                    w_ptrs,
                    existing + contrib,
                    mask=mask_b[:, None] & mask_k[None, :],
                )

            # Add dh_prev_direct at offset oh in the write buffer.
            wd_ptrs = write_ptr + offs_b[:, None] * sdh_b + offs_oh[None, :]
            existing = tl.load(wd_ptrs, mask=mask_oh2, other=0.0)
            tl.store(wd_ptrs, existing + dh_prev_direct, mask=mask_oh2)

            # ---- dWh_partial accumulation: dgh^T @ h_prev ----
            for k in range(0, H, BLOCK_K):
                offs_k = k + tl.arange(0, BLOCK_K)
                mask_k = offs_k < H

                h_prev_ptrs = (
                    h_prev_ptr + offs_b[:, None] * sh_prev_b + offs_k[None, :]
                )
                h_prev_tile = tl.load(
                    h_prev_ptrs,
                    mask=mask_b[:, None] & mask_k[None, :],
                    other=0.0,
                )

                # Forward used hq = quant_h_in(h_prev) in the matmul, so
                # dWh accumulates against hq, not raw h_prev.
                if QUANT_H_IN:
                    q = tl.extra.cuda.libdevice.rint(h_prev_tile / h_in_scale)
                    q = tl.minimum(tl.maximum(q, h_in_qmin), h_in_qmax)
                    h_prev_tile = q * h_in_scale

                # [BLOCK_OH, BLOCK_B] @ [BLOCK_B, BLOCK_K] -> [BLOCK_OH, BLOCK_K]
                dWr = tl.dot(tl.trans(dgh_r), h_prev_tile, input_precision="tf32")
                dWz = tl.dot(tl.trans(dgh_z), h_prev_tile, input_precision="tf32")
                dWn = tl.dot(tl.trans(dgh_n), h_prev_tile, input_precision="tf32")

                dWh_base = dWh_partial_ptr + pid_b * sdWp_pid
                Wr_ptrs = (
                    dWh_base
                    + (0 * H + offs_oh)[:, None] * sdWp_o
                    + offs_k[None, :]
                )
                Wz_ptrs = (
                    dWh_base
                    + (1 * H + offs_oh)[:, None] * sdWp_o
                    + offs_k[None, :]
                )
                Wn_ptrs = (
                    dWh_base
                    + (2 * H + offs_oh)[:, None] * sdWp_o
                    + offs_k[None, :]
                )
                mask_okok = mask_oh[:, None] & mask_k[None, :]
                tl.store(
                    Wr_ptrs,
                    tl.load(Wr_ptrs, mask=mask_okok, other=0.0) + dWr,
                    mask=mask_okok,
                )
                tl.store(
                    Wz_ptrs,
                    tl.load(Wz_ptrs, mask=mask_okok, other=0.0) + dWz,
                    mask=mask_okok,
                )
                tl.store(
                    Wn_ptrs,
                    tl.load(Wn_ptrs, mask=mask_okok, other=0.0) + dWn,
                    mask=mask_okok,
                )

        # End of all oh tiles for this t. Swap read/write pointers.
        tmp = read_ptr
        read_ptr = write_ptr
        write_ptr = tmp

    # Final dh_acc -> dh0
    for h_off in range(0, H, BLOCK_K):
        offs_h = h_off + tl.arange(0, BLOCK_K)
        mask_h = offs_h < H
        r_ptrs = read_ptr + offs_b[:, None] * sdh_b + offs_h[None, :]
        dh_final = tl.load(r_ptrs, mask=mask_b[:, None] & mask_h[None, :], other=0.0)
        dh0_ptrs = dh0_ptr + offs_b[:, None] * sdh0_b + offs_h[None, :]
        tl.store(dh0_ptrs, dh_final, mask=mask_b[:, None] & mask_h[None, :])


def gru_scan_backward_triton(
    gi: torch.Tensor,
    h0: torch.Tensor,
    Wh_cat: torch.Tensor,
    bh_cat: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    *,
    h_in_quant: tuple[float, int, int] | None = None,
    h_out_quant: tuple[float, int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Triton multi-step backward.

    Returns ``(dgi, dh0, dWh, dbh)`` matching the PyTorch reference.
    """
    T, B, three_H = gi.shape
    H = three_H // 3
    if h0.shape != (B, H):
        raise ValueError(f"h0 shape must be (B, H)=({B}, {H}); got {tuple(h0.shape)}")
    if Wh_cat.shape != (3 * H, H):
        raise ValueError(
            f"Wh_cat shape must be (3H, H)=({3 * H}, {H}); got {tuple(Wh_cat.shape)}"
        )
    if bh_cat.shape != (3 * H,):
        raise ValueError(
            f"bh_cat shape must be (3H,)=({3 * H},); got {tuple(bh_cat.shape)}"
        )
    if out.shape != (T, B, H):
        raise ValueError(
            f"out shape must be (T, B, H)=({T}, {B}, {H}); got {tuple(out.shape)}"
        )
    if dout.shape != (T, B, H):
        raise ValueError(
            f"dout shape must be (T, B, H)=({T}, {B}, {H}); got {tuple(dout.shape)}"
        )

    gi = gi.contiguous()
    h0 = h0.contiguous()
    Wh_cat = Wh_cat.contiguous()
    bh_cat = bh_cat.contiguous()
    out = out.contiguous()
    dout = dout.contiguous()

    dgi = torch.zeros_like(gi)
    dh0 = torch.zeros_like(h0)

    # Over-allocate partial buffers based on the smallest BLOCK_B in the
    # autotune grid. Programs that aren't launched (because autotune chose
    # a larger BLOCK_B) leave their slabs at zero, which is a no-op when
    # we sum across the program dim afterwards.
    n_pid_max = triton.cdiv(B, _MIN_AUTOTUNE_BLOCK_B)
    dWh_partial = torch.zeros(
        (n_pid_max, 3 * H, H), device=gi.device, dtype=gi.dtype
    )
    dbh_partial = torch.zeros(
        (n_pid_max, 3 * H), device=gi.device, dtype=gi.dtype
    )
    dh_a = torch.empty((B, H), device=gi.device, dtype=gi.dtype)
    dh_b = torch.empty((B, H), device=gi.device, dtype=gi.dtype)

    in_s, in_qmin, in_qmax = h_in_quant or (1.0, -2**31, 2**31 - 1)
    out_s, out_qmin, out_qmax = h_out_quant or (1.0, -2**31, 2**31 - 1)

    def grid(meta: dict[str, int]) -> tuple[int]:
        return (triton.cdiv(B, meta["BLOCK_B"]),)

    gru_scan_bwd_kernel[grid](
        gi, h0, Wh_cat, bh_cat, out,
        dout,
        dgi, dh0,
        dWh_partial, dbh_partial,
        dh_a, dh_b,
        T, B,
        gi.stride(0), gi.stride(1),
        h0.stride(0),
        Wh_cat.stride(0),
        out.stride(0), out.stride(1),
        dout.stride(0), dout.stride(1),
        dgi.stride(0), dgi.stride(1),
        dh0.stride(0),
        dWh_partial.stride(0), dWh_partial.stride(1),
        dbh_partial.stride(0),
        dh_a.stride(0),
        in_s, in_qmin, in_qmax,
        out_s, out_qmin, out_qmax,
        H=H,
        QUANT_H_IN=h_in_quant is not None,
        QUANT_H_OUT=h_out_quant is not None,
    )

    dWh = dWh_partial.sum(dim=0)
    dbh = dbh_partial.sum(dim=0)
    return dgi, dh0, dWh, dbh


def _fake_quant(
    x: torch.Tensor, scale: float, qmin: int, qmax: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward fake-quant + the STE clip mask. Per-tensor symmetric, zp=0."""
    q_unclamped = torch.round(x / scale)
    mask = (q_unclamped >= qmin) & (q_unclamped <= qmax)
    q_clamped = q_unclamped.clamp(qmin, qmax)
    return q_clamped * scale, mask


def _gru_scan_backward_pytorch(
    gi: torch.Tensor,
    h0: torch.Tensor,
    Wh_cat: torch.Tensor,
    bh_cat: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    *,
    h_in_quant: tuple[float, int, int] | None = None,
    h_out_quant: tuple[float, int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference backward for the multi-step scan, in PyTorch.

    Walks t from T-1 down to 0 and computes gradients w.r.t. the four
    forward inputs. Recomputes ``r``, ``z``, ``n``, ``gh_n`` from
    ``h_{t-1}`` and the saved ``out`` rather than saving them — keeps the
    save-tensor footprint to ``[T, B, H]`` (out) plus the original inputs.
    Recompute is cheap relative to the autograd traversal saved.

    Slow: pure PyTorch per-step. Used as the gradient-correctness ground
    truth and as a fallback when a Triton backward isn't yet wired up.
    """
    T, B, _ = gi.shape
    H = h0.shape[-1]

    dgi = torch.zeros_like(gi)
    dWh = torch.zeros_like(Wh_cat)
    dbh = torch.zeros_like(bh_cat)

    dh_acc = torch.zeros_like(h0)

    for t in reversed(range(T)):
        h_prev = h0 if t == 0 else out[t - 1]

        gi_r, gi_z, gi_n = gi[t].chunk(3, dim=-1)
        if h_in_quant is not None:
            hq, mask_in = _fake_quant(h_prev, *h_in_quant)
        else:
            hq = h_prev
            mask_in = None
        gh = hq @ Wh_cat.T + bh_cat
        gh_r, gh_z, gh_n = gh.chunk(3, dim=-1)
        r = torch.sigmoid(gi_r + gh_r)
        z = torch.sigmoid(gi_z + gh_z)
        n = torch.tanh(gi_n + r * gh_n)
        h_t_raw = (1.0 - z) * n + z * h_prev

        dh_t = dout[t] + dh_acc

        # STE backward through quant_h_out: gradient on h_t_q -> on h_t_raw.
        if h_out_quant is not None:
            _, mask_out = _fake_quant(h_t_raw, *h_out_quant)
            dh_t = dh_t * mask_out

        # h_t_raw = (1 - z) * n + z * h_prev (uses raw h_prev, not hq)
        dn = dh_t * (1.0 - z)
        dz = dh_t * (h_prev - n)
        dh_prev_direct = dh_t * z

        # n = tanh(gn_pre)
        dgn_pre = dn * (1.0 - n * n)

        # gn_pre = gi_n + r * gh_n
        dgi_n = dgn_pre
        dr = dgn_pre * gh_n
        dgh_n = dgn_pre * r

        # z = sigmoid(gi_z + gh_z)
        dgz_pre = dz * z * (1.0 - z)
        dgi_z = dgz_pre
        dgh_z = dgz_pre

        # r = sigmoid(gi_r + gh_r)
        dgr_pre = dr * r * (1.0 - r)
        dgi_r = dgr_pre
        dgh_r = dgr_pre

        dgi[t] = torch.cat([dgi_r, dgi_z, dgi_n], dim=-1)
        dgh = torch.cat([dgh_r, dgh_z, dgh_n], dim=-1)

        # gh = hq @ Wh_cat^T + bh_cat -> dWh accumulates against hq;
        # dh_prev_via_W passes through STE backward of quant_h_in.
        dh_prev_via_W = dgh @ Wh_cat
        if mask_in is not None:
            dh_prev_via_W = dh_prev_via_W * mask_in
        dWh += dgh.transpose(0, 1) @ hq
        dbh += dgh.sum(dim=0)

        dh_acc = dh_prev_direct + dh_prev_via_W

    return dgi, dh_acc, dWh, dbh


_USE_TRITON_BACKWARD = True  # set False to fall back to the PyTorch reference


class GRUScanFunction(torch.autograd.Function):
    """autograd wrapper around the multi-step GRU scan.

    Forward is the Triton kernel. Backward is the Triton backward kernel by
    default; toggle `_USE_TRITON_BACKWARD` to fall back to the PyTorch
    reference (useful when debugging gradient drift).

    Optional ``h_in_quant`` / ``h_out_quant`` enable in-kernel fake-quant
    on the hidden state with a frozen per-tensor symmetric scale.
    """

    @staticmethod
    def forward(
        ctx: Any,
        gi: torch.Tensor,
        h0: torch.Tensor,
        Wh_cat: torch.Tensor,
        bh_cat: torch.Tensor,
        h_in_quant: tuple[float, int, int] | None,
        h_out_quant: tuple[float, int, int] | None,
    ) -> torch.Tensor:
        out = gru_scan_forward(
            gi, h0, Wh_cat, bh_cat,
            h_in_quant=h_in_quant, h_out_quant=h_out_quant,
        )
        ctx.save_for_backward(gi, h0, Wh_cat, bh_cat, out)
        ctx.h_in_quant = h_in_quant
        ctx.h_out_quant = h_out_quant
        return out

    @staticmethod
    def backward(
        ctx: Any, dout: torch.Tensor
    ) -> Any:
        gi, h0, Wh_cat, bh_cat, out = ctx.saved_tensors
        h_in_quant = ctx.h_in_quant
        h_out_quant = ctx.h_out_quant
        if _USE_TRITON_BACKWARD:
            grads = gru_scan_backward_triton(
                gi, h0, Wh_cat, bh_cat, out, dout,
                h_in_quant=h_in_quant, h_out_quant=h_out_quant,
            )
        else:
            grads = _gru_scan_backward_pytorch(
                gi, h0, Wh_cat, bh_cat, out, dout,
                h_in_quant=h_in_quant, h_out_quant=h_out_quant,
            )
        # Two trailing Nones for the non-tensor h_in_quant / h_out_quant args.
        return (*grads, None, None)


def gru_scan(
    gi: torch.Tensor,
    h0: torch.Tensor,
    Wh_cat: torch.Tensor,
    bh_cat: torch.Tensor,
    *,
    h_in_quant: tuple[float, int, int] | None = None,
    h_out_quant: tuple[float, int, int] | None = None,
) -> torch.Tensor:
    """Public API: differentiable multi-step GRU scan.

    With ``h_in_quant`` / ``h_out_quant`` supplied (each a
    ``(scale, qmin, qmax)`` tuple), the kernel applies in-kernel fake-quant
    on the hidden state on every step. Per-tensor symmetric, frozen scale.
    """
    return cast(
        torch.Tensor,
        GRUScanFunction.apply(  # type: ignore[no-untyped-call]
            gi, h0, Wh_cat, bh_cat, h_in_quant, h_out_quant
        ),
    )


class GRUScanPersistentFunction(torch.autograd.Function):
    """Same autograd contract as GRUScanFunction, dispatched to the
    persistent forward/backward kernels. Optional in-kernel fake-quant on
    hidden state, same parameter shape as GRUScanFunction.
    """

    @staticmethod
    def forward(
        ctx: Any,
        gi: torch.Tensor,
        h0: torch.Tensor,
        Wh_cat: torch.Tensor,
        bh_cat: torch.Tensor,
        h_in_quant: tuple[float, int, int] | None,
        h_out_quant: tuple[float, int, int] | None,
    ) -> torch.Tensor:
        out = gru_scan_forward_persistent(
            gi, h0, Wh_cat, bh_cat,
            h_in_quant=h_in_quant, h_out_quant=h_out_quant,
        )
        ctx.save_for_backward(gi, h0, Wh_cat, bh_cat, out)
        ctx.h_in_quant = h_in_quant
        ctx.h_out_quant = h_out_quant
        return out

    @staticmethod
    def backward(
        ctx: Any, dout: torch.Tensor
    ) -> Any:
        gi, h0, Wh_cat, bh_cat, out = ctx.saved_tensors
        grads = gru_scan_backward_persistent(
            gi, h0, Wh_cat, bh_cat, out, dout,
            h_in_quant=ctx.h_in_quant, h_out_quant=ctx.h_out_quant,
        )
        return (*grads, None, None)


def gru_scan_persistent(
    gi: torch.Tensor,
    h0: torch.Tensor,
    Wh_cat: torch.Tensor,
    bh_cat: torch.Tensor,
    *,
    h_in_quant: tuple[float, int, int] | None = None,
    h_out_quant: tuple[float, int, int] | None = None,
) -> torch.Tensor:
    """Persistent-kernel variant of gru_scan. Forward and backward both
    use 2D grids with cross-CTA barriers; better SM utilization at modest
    batch sizes. ``h_in_quant`` / ``h_out_quant`` enable in-kernel
    fake-quant on the hidden state, same semantics as gru_scan."""
    return cast(
        torch.Tensor,
        GRUScanPersistentFunction.apply(  # type: ignore[no-untyped-call]
            gi, h0, Wh_cat, bh_cat, h_in_quant, h_out_quant
        ),
    )


def gru_scan_forward(
    gi: torch.Tensor,
    h0: torch.Tensor,
    Wh_cat: torch.Tensor,
    bh_cat: torch.Tensor,
    *,
    h_in_quant: tuple[float, int, int] | None = None,
    h_out_quant: tuple[float, int, int] | None = None,
) -> torch.Tensor:
    """Forward pass of the multi-step GRU scan in Triton.

    Args:
        gi:     [T, B, 3H]  pre-batched input projection (already includes bi_cat)
        h0:     [B, H]      initial hidden
        Wh_cat: [3H, H]     concatenated hidden weights, rows=r,z,n
        bh_cat: [3H]        concatenated hidden biases
    Returns:
        out:    [T, B, H]   hidden state at each timestep
    """
    if not (gi.is_cuda and h0.is_cuda and Wh_cat.is_cuda and bh_cat.is_cuda):
        raise ValueError(
            "gi, h0, Wh_cat, bh_cat must all be CUDA tensors; got devices "
            f"gi={gi.device}, h0={h0.device}, Wh_cat={Wh_cat.device}, "
            f"bh_cat={bh_cat.device}"
        )
    if gi.dtype != torch.float32:
        raise ValueError(f"gi dtype must be float32 (Phase 1 fp32 only); got {gi.dtype}")
    T, B, three_H = gi.shape
    H = three_H // 3
    if h0.shape != (B, H):
        raise ValueError(f"h0 shape must be (B, H)=({B}, {H}); got {tuple(h0.shape)}")
    if Wh_cat.shape != (3 * H, H):
        raise ValueError(
            f"Wh_cat shape must be (3H, H)=({3 * H}, {H}); got {tuple(Wh_cat.shape)}"
        )
    if bh_cat.shape != (3 * H,):
        raise ValueError(
            f"bh_cat shape must be (3H,)=({3 * H},); got {tuple(bh_cat.shape)}"
        )

    # Make sure inputs are contiguous in their last dim — strides below assume
    # last-dim stride = 1.
    gi = gi.contiguous()
    h0 = h0.contiguous()
    Wh_cat = Wh_cat.contiguous()
    bh_cat = bh_cat.contiguous()

    out = torch.empty((T, B, H), device=gi.device, dtype=gi.dtype)

    in_s, in_qmin, in_qmax = h_in_quant or (1.0, -2**31, 2**31 - 1)
    out_s, out_qmin, out_qmax = h_out_quant or (1.0, -2**31, 2**31 - 1)

    # Autotune picks BLOCK_B/OH/K. Grid is a meta-function so it sizes from
    # whatever BLOCK_B the autotuner chose for this (T, B, H) cell.
    def grid(meta: dict[str, int]) -> tuple[int]:
        return (triton.cdiv(B, meta["BLOCK_B"]),)

    gru_scan_fwd_kernel[grid](
        gi,
        h0,
        Wh_cat,
        bh_cat,
        out,
        T,
        B,
        gi.stride(0),
        gi.stride(1),
        h0.stride(0),
        Wh_cat.stride(0),
        out.stride(0),
        out.stride(1),
        in_s, in_qmin, in_qmax,
        out_s, out_qmin, out_qmax,
        H=H,
        QUANT_H_IN=h_in_quant is not None,
        QUANT_H_OUT=h_out_quant is not None,
    )
    return out

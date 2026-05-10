"""Monarch (block-diagonal) hidden weights for the multi-step GRU scan.

Tier-2 work: structured-hidden-side variant of ``gru_scan_persistent``.
The hidden weight ``Wh`` is parameterized as three Monarch factors (one
per gate), each ``[nblocks, blksz, blksz]`` with ``blksz = H/nblocks``.
The per-step matmul becomes ``nblocks`` independent ``[B, blksz] x
[blksz, blksz]`` block matmuls — same total FLOPs in the input-bound
regime, but ``nblocks``× smaller K-reduction per output block, ``nblocks``×
smaller per-block working set.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl


def extract_monarch_factors(cell: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """Pull the three hidden-side Monarch weights out of a tier-1 cell.

    Args:
        cell: a ``GRUCellQuant`` whose ``structure_hidden`` was a Monarch
              ``StructureConfig``. Must have ``struct_Wh_r``, ``struct_Wh_z``,
              and ``struct_Wh_n`` BlockdiagLinear modules.

    Returns:
        Wh_struct: [3, nblocks, blksz, blksz] — gates stacked in (r, z, n)
            order, each layer's underlying ``[nblocks, out_blksz, in_blksz]``
            weight tensor.
        bh_cat: [3*H] — concat of (b_hr, b_hz, b_hn).
    """
    if cell._hidden_dense:
        raise ValueError("cell hidden side is dense; nothing to extract")
    # All three layers share the same shape (square BlockdiagLinear with
    # in_features == out_features == H).
    # struct_Wh_* are BlockdiagLinear instances; their `.weight` is the
    # [nblocks, out_blksz, in_blksz] factor tensor.
    Wr = cell.struct_Wh_r.weight
    Wz = cell.struct_Wh_z.weight
    Wn = cell.struct_Wh_n.weight
    Wh_struct = torch.stack([Wr, Wz, Wn], dim=0)  # [3, nblocks, blksz, blksz]
    if cell.b_hr is None:
        bh_cat = torch.zeros(3 * cell.hidden_size, device=Wh_struct.device, dtype=Wh_struct.dtype)
    else:
        bh_cat = torch.cat([cell.b_hr, cell.b_hz, cell.b_hn])
    return Wh_struct, bh_cat


def gru_scan_monarch_forward_pytorch(
    gi: torch.Tensor,
    h0: torch.Tensor,
    Wh_struct: torch.Tensor,
    bh_cat: torch.Tensor,
) -> torch.Tensor:
    """Reference forward for the block-diagonal scan, in PyTorch.

    Args:
        gi: [T, B, 3H] — pre-batched input projection (already with bi).
        h0: [B, H]
        Wh_struct: [3, nblocks, blksz, blksz]
        bh_cat: [3H]
    Returns:
        out: [T, B, H] — hidden state at every timestep.
    """
    T, B, three_H = gi.shape
    H = three_H // 3
    n_gates, nblocks, out_blksz, in_blksz = Wh_struct.shape
    assert n_gates == 3
    assert out_blksz == in_blksz, "square Monarch only"
    assert nblocks * out_blksz == H, f"nblocks*blksz={nblocks*out_blksz} != H={H}"

    blksz = out_blksz
    out = torch.empty(T, B, H, device=gi.device, dtype=gi.dtype)
    h = h0
    bh = bh_cat.view(3, H)

    for t in range(T):
        # h: [B, H] -> [B, nblocks, blksz]
        h_chunks = h.view(B, nblocks, blksz)
        # Block-diagonal matmul per gate:
        #   gh[g, b, n, o] = sum_i h_chunks[b, n, i] * Wh_struct[g, n, o, i]
        gh = torch.einsum("bni,gnoi->bgno", h_chunks, Wh_struct)  # [B, 3, nblocks, blksz]
        gh = gh.reshape(B, 3, H) + bh  # add bias per gate
        gh_r, gh_z, gh_n = gh[:, 0, :], gh[:, 1, :], gh[:, 2, :]

        gi_r = gi[t, :, 0:H]
        gi_z = gi[t, :, H:2 * H]
        gi_n = gi[t, :, 2 * H:3 * H]

        r = torch.sigmoid(gi_r + gh_r)
        z = torch.sigmoid(gi_z + gh_z)
        n = torch.tanh(gi_n + r * gh_n)
        h_new = (1.0 - z) * n + z * h
        out[t] = h_new
        h = h_new

    return out


@triton.jit
def gru_scan_monarch_fwd_kernel(
    gi_ptr,            # [T, B, 3H], fp32
    h0_ptr,            # [B, H], fp32
    Wh_ptr,            # [3, nblocks, blksz, blksz], fp32
    bh_ptr,            # [3H], fp32
    out_ptr,           # [T, B, H], fp32
    barrier_ptr,       # [T], int32
    T,
    B,
    sg_t, sg_b,
    sh0_b,
    sW_g, sW_n, sW_o,
    so_t, so_b,
    NUM_PROGRAMS,
    H: tl.constexpr,
    BLKSZ: tl.constexpr,
    NBLOCKS: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Persistent forward over the block-diagonal recurrence.

    Grid (pid_b, pid_block): each program handles ALL 3 gates for ONE
    block, producing [BLOCK_B, blksz] of h_t for that block. Block
    boundaries don't mix in the matmul (block-diagonal), so the K
    reduction is only over blksz instead of full H — that's where the
    win comes from.
    """
    pid_b = tl.program_id(0)
    pid_block = tl.program_id(1)

    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = offs_b < B
    offs_oh = tl.arange(0, BLKSZ)  # output rows within this block

    # Output position in flat 3H layout: gate * H + pid_block * BLKSZ + offs_oh.
    # h-input range: pid_block * BLKSZ + offs_k.

    # Pre-load the bias slice for this (gate, block) — 3 chunks of BLKSZ.
    bh_offset = pid_block * BLKSZ
    bhr_tile = tl.load(bh_ptr + 0 * H + bh_offset + offs_oh)
    bhz_tile = tl.load(bh_ptr + 1 * H + bh_offset + offs_oh)
    bhn_tile = tl.load(bh_ptr + 2 * H + bh_offset + offs_oh)

    h_in_ptr = h0_ptr
    sh_b = sh0_b

    for t in range(0, T):
        ghr = tl.zeros((BLOCK_B, BLKSZ), dtype=tl.float32)
        ghz = tl.zeros((BLOCK_B, BLKSZ), dtype=tl.float32)
        ghn = tl.zeros((BLOCK_B, BLKSZ), dtype=tl.float32)

        for k in range(0, BLKSZ, BLOCK_K):
            offs_k = k + tl.arange(0, BLOCK_K)
            mask_k = offs_k < BLKSZ

            # h_block tile: [BLOCK_B, BLOCK_K] read from the current h_in
            # at the input slice of pid_block.
            h_ptrs = (
                h_in_ptr
                + offs_b[:, None] * sh_b
                + (pid_block * BLKSZ + offs_k)[None, :]
            )
            h_tile = tl.load(
                h_ptrs, mask=mask_b[:, None] & mask_k[None, :], other=0.0,
            )

            # Three W tiles, one per gate. Each is [BLKSZ, BLOCK_K].
            W_block_offset = pid_block * sW_n
            W_oh_offset = offs_oh[:, None] * sW_o + offs_k[None, :]
            Wr_tile = tl.load(
                Wh_ptr + 0 * sW_g + W_block_offset + W_oh_offset,
                mask=mask_k[None, :], other=0.0,
            )
            Wz_tile = tl.load(
                Wh_ptr + 1 * sW_g + W_block_offset + W_oh_offset,
                mask=mask_k[None, :], other=0.0,
            )
            Wn_tile = tl.load(
                Wh_ptr + 2 * sW_g + W_block_offset + W_oh_offset,
                mask=mask_k[None, :], other=0.0,
            )

            ghr += tl.dot(h_tile, tl.trans(Wr_tile), input_precision="tf32")
            ghz += tl.dot(h_tile, tl.trans(Wz_tile), input_precision="tf32")
            ghn += tl.dot(h_tile, tl.trans(Wn_tile), input_precision="tf32")

        ghr += bhr_tile[None, :]
        ghz += bhz_tile[None, :]
        ghn += bhn_tile[None, :]

        # gi[t] tile for this block, three gate slices.
        gi_base = (
            gi_ptr
            + t * sg_t
            + offs_b[:, None] * sg_b
            + (pid_block * BLKSZ + offs_oh)[None, :]
        )
        gir = tl.load(gi_base + 0 * H, mask=mask_b[:, None], other=0.0)
        giz = tl.load(gi_base + 1 * H, mask=mask_b[:, None], other=0.0)
        gin = tl.load(gi_base + 2 * H, mask=mask_b[:, None], other=0.0)

        r = tl.sigmoid(gir + ghr)
        z = tl.sigmoid(giz + ghz)
        n = tl.extra.libdevice.tanh(gin + r * ghn)

        # h_t = (1-z)*n + z*h_prev at THIS block's output positions.
        h_old_ptrs = (
            h_in_ptr
            + offs_b[:, None] * sh_b
            + (pid_block * BLKSZ + offs_oh)[None, :]
        )
        h_old = tl.load(h_old_ptrs, mask=mask_b[:, None], other=0.0)
        h_new = (1.0 - z) * n + z * h_old

        out_ptrs = (
            out_ptr
            + t * so_t
            + offs_b[:, None] * so_b
            + (pid_block * BLKSZ + offs_oh)[None, :]
        )
        tl.store(out_ptrs, h_new, mask=mask_b[:, None])

        # Cross-CTA barrier: pair release/acquire same as dense persistent.
        tl.atomic_add(barrier_ptr + t, 1, sem="release")
        done = tl.atomic_add(barrier_ptr + t, 0, sem="acquire")
        while done < NUM_PROGRAMS:
            done = tl.atomic_add(barrier_ptr + t, 0, sem="acquire")

        h_in_ptr = out_ptr + t * so_t
        sh_b = so_b


def gru_scan_monarch_forward_triton(
    gi: torch.Tensor,
    h0: torch.Tensor,
    Wh_struct: torch.Tensor,
    bh_cat: torch.Tensor,
    *,
    block_b: int = 16,
    block_k: int = 32,
    num_warps: int = 4,
    num_stages: int = 2,
) -> torch.Tensor:
    """Triton forward for the Monarch hidden-side scan."""
    assert gi.is_cuda and Wh_struct.is_cuda
    T, B, three_H = gi.shape
    H = three_H // 3
    n_gates, nblocks, out_blksz, in_blksz = Wh_struct.shape
    assert n_gates == 3
    assert out_blksz == in_blksz == H // nblocks

    gi = gi.contiguous()
    h0 = h0.contiguous()
    Wh_struct = Wh_struct.contiguous()
    bh_cat = bh_cat.contiguous()

    out = torch.empty((T, B, H), device=gi.device, dtype=gi.dtype)
    barrier = torch.zeros((T,), device=gi.device, dtype=torch.int32)

    n_pid_b = triton.cdiv(B, block_b)
    num_programs = n_pid_b * nblocks

    sm_count = torch.cuda.get_device_properties(gi.device).multi_processor_count
    if num_programs > sm_count:
        raise RuntimeError(
            f"persistent grid {num_programs} > SM count {sm_count}; "
            f"would deadlock on the spin-wait barrier."
        )

    grid = (n_pid_b, nblocks)
    gru_scan_monarch_fwd_kernel[grid](
        gi, h0, Wh_struct, bh_cat, out,
        barrier,
        T, B,
        gi.stride(0), gi.stride(1),
        h0.stride(0),
        Wh_struct.stride(0), Wh_struct.stride(1), Wh_struct.stride(2),
        out.stride(0), out.stride(1),
        num_programs,
        H=H,
        BLKSZ=out_blksz,
        NBLOCKS=nblocks,
        BLOCK_B=block_b,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


@triton.jit
def gru_scan_monarch_bwd_kernel(
    # forward inputs (read-only)
    gi_ptr,              # [T, B, 3H]
    h0_ptr,              # [B, H]
    Wh_ptr,              # [3, nblocks, blksz, blksz]
    bh_ptr,              # [3H]
    out_ptr,             # [T, B, H]
    # upstream
    dout_ptr,            # [T, B, H]
    # outputs
    dgi_ptr,             # [T, B, 3H]
    dh0_ptr,             # [B, H]
    # per-pid_b partial buffers (reduced across pid_b in Python)
    dWh_partial_ptr,     # [num_pid_b, 3, nblocks, blksz, blksz]
    dbh_partial_ptr,     # [num_pid_b, 3H]
    # scratch dh_acc buffer (per-block disjoint, no ping-pong needed because
    # block-diagonal structure means no cross-program writes to same cell)
    dh_acc_ptr,          # [B, H]
    barrier_ptr,         # [T] int32
    T, B,
    sg_t, sg_b,
    sh0_b,
    sW_g, sW_n, sW_o,
    so_t, so_b,
    sdo_t, sdo_b,
    sdgi_t, sdgi_b,
    sdh0_b,
    sdWp_pid, sdWp_g, sdWp_n, sdWp_o,
    sdbp_pid,
    sdh_b,
    NUM_PROGRAMS,
    H: tl.constexpr,
    BLKSZ: tl.constexpr,
    NBLOCKS: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Persistent backward over block-diagonal recurrence.

    Each program owns one (batch_tile, output_block) and stays in that
    slot for all T timesteps. dh_acc accumulation, dWh partial, and dbh
    partial are all partition-disjoint across (pid_b, pid_block) pairs:
    block i's dh_via_W only feeds block i's input slice. So no atomic
    adds or ping-pong buffers — just a single dh_acc scratch with a
    cross-CTA barrier between timesteps.
    """
    pid_b = tl.program_id(0)
    pid_block = tl.program_id(1)

    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = offs_b < B
    offs_oh = tl.arange(0, BLKSZ)

    # Pre-load bias slice (constant across T).
    bh_offset = pid_block * BLKSZ
    bhr_tile = tl.load(bh_ptr + 0 * H + bh_offset + offs_oh)
    bhz_tile = tl.load(bh_ptr + 1 * H + bh_offset + offs_oh)
    bhn_tile = tl.load(bh_ptr + 2 * H + bh_offset + offs_oh)

    # Initialize dh_acc[:, this block range] to zero.
    dh_acc_init_ptrs = (
        dh_acc_ptr
        + offs_b[:, None] * sdh_b
        + (pid_block * BLKSZ + offs_oh)[None, :]
    )
    tl.store(
        dh_acc_init_ptrs,
        tl.zeros((BLOCK_B, BLKSZ), dtype=tl.float32),
        mask=mask_b[:, None],
    )

    for t_rev in range(0, T):
        t = T - 1 - t_rev

        if t == 0:
            h_prev_ptr = h0_ptr
            sh_prev_b = sh0_b
        else:
            h_prev_ptr = out_ptr + (t - 1) * so_t
            sh_prev_b = so_b

        # ---- Recompute forward gh for this (block, all gates) ----
        ghr = tl.zeros((BLOCK_B, BLKSZ), dtype=tl.float32)
        ghz = tl.zeros((BLOCK_B, BLKSZ), dtype=tl.float32)
        ghn = tl.zeros((BLOCK_B, BLKSZ), dtype=tl.float32)
        for k in range(0, BLKSZ, BLOCK_K):
            offs_k = k + tl.arange(0, BLOCK_K)
            mask_k = offs_k < BLKSZ
            h_ptrs = (
                h_prev_ptr
                + offs_b[:, None] * sh_prev_b
                + (pid_block * BLKSZ + offs_k)[None, :]
            )
            h_tile = tl.load(
                h_ptrs, mask=mask_b[:, None] & mask_k[None, :], other=0.0,
            )
            W_block_offset = pid_block * sW_n
            W_oh_offset = offs_oh[:, None] * sW_o + offs_k[None, :]
            Wr_tile = tl.load(
                Wh_ptr + 0 * sW_g + W_block_offset + W_oh_offset,
                mask=mask_k[None, :], other=0.0,
            )
            Wz_tile = tl.load(
                Wh_ptr + 1 * sW_g + W_block_offset + W_oh_offset,
                mask=mask_k[None, :], other=0.0,
            )
            Wn_tile = tl.load(
                Wh_ptr + 2 * sW_g + W_block_offset + W_oh_offset,
                mask=mask_k[None, :], other=0.0,
            )
            ghr += tl.dot(h_tile, tl.trans(Wr_tile), input_precision="tf32")
            ghz += tl.dot(h_tile, tl.trans(Wz_tile), input_precision="tf32")
            ghn += tl.dot(h_tile, tl.trans(Wn_tile), input_precision="tf32")
        ghr += bhr_tile[None, :]
        ghz += bhz_tile[None, :]
        ghn += bhn_tile[None, :]

        gi_base = (
            gi_ptr
            + t * sg_t
            + offs_b[:, None] * sg_b
            + (pid_block * BLKSZ + offs_oh)[None, :]
        )
        gir = tl.load(gi_base + 0 * H, mask=mask_b[:, None], other=0.0)
        giz = tl.load(gi_base + 1 * H, mask=mask_b[:, None], other=0.0)
        gin = tl.load(gi_base + 2 * H, mask=mask_b[:, None], other=0.0)

        r = tl.sigmoid(gir + ghr)
        z = tl.sigmoid(giz + ghz)
        n = tl.extra.libdevice.tanh(gin + r * ghn)

        # h_prev at THIS block's positions (for dh_prev_direct and (1-z)*n + z*h_prev).
        h_prev_oh_ptrs = (
            h_prev_ptr
            + offs_b[:, None] * sh_prev_b
            + (pid_block * BLKSZ + offs_oh)[None, :]
        )
        h_prev_oh = tl.load(h_prev_oh_ptrs, mask=mask_b[:, None], other=0.0)

        # ---- Read incoming dh_acc[this block range]. Uses .cv to bypass
        # L1 since the previous step's writes from THIS SAME PROGRAM are
        # always visible (no cross-program writes here), so .cv is
        # technically unnecessary — but cheap insurance against
        # compiler-reordering surprises. ----
        dh_acc_ptrs = (
            dh_acc_ptr
            + offs_b[:, None] * sdh_b
            + (pid_block * BLKSZ + offs_oh)[None, :]
        )
        dh_acc_oh = tl.load(
            dh_acc_ptrs, mask=mask_b[:, None], other=0.0, cache_modifier=".cv",
        )
        dout_base = (
            dout_ptr
            + t * sdo_t
            + offs_b[:, None] * sdo_b
            + (pid_block * BLKSZ + offs_oh)[None, :]
        )
        dout_oh = tl.load(dout_base, mask=mask_b[:, None], other=0.0)
        dh_t = dout_oh + dh_acc_oh

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

        # Store dgi[t] for this block range, all 3 gates.
        dgi_base = (
            dgi_ptr
            + t * sdgi_t
            + offs_b[:, None] * sdgi_b
            + (pid_block * BLKSZ + offs_oh)[None, :]
        )
        tl.store(dgi_base + 0 * H, dgi_r, mask=mask_b[:, None])
        tl.store(dgi_base + 1 * H, dgi_z, mask=mask_b[:, None])
        tl.store(dgi_base + 2 * H, dgi_n, mask=mask_b[:, None])

        # dbh_partial: each (pid_b, pid_block) is the only writer to its
        # bias slice rows.
        dbh_base = dbh_partial_ptr + pid_b * sdbp_pid + bh_offset + offs_oh
        tl.store(
            dbh_base + 0 * H,
            tl.load(dbh_base + 0 * H) + tl.sum(dgh_r, axis=0),
        )
        tl.store(
            dbh_base + 1 * H,
            tl.load(dbh_base + 1 * H) + tl.sum(dgh_z, axis=0),
        )
        tl.store(
            dbh_base + 2 * H,
            tl.load(dbh_base + 2 * H) + tl.sum(dgh_n, axis=0),
        )

        # ---- dh_prev_via_W and dWh_partial accumulation ----
        # Both are per-block (no cross-block contributions). For each k-tile,
        # compute the dh_via_W contribution to dh_acc[this block, k:k+BLOCK_K]
        # and the dWh contribution dgh^T @ h_prev (per gate).
        #
        # We also need to handle dh_prev_direct contribution to dh_acc.
        # Since dh_prev_direct lives at offs_oh range (same as block range),
        # we add it once after the k-loop.
        for k in range(0, BLKSZ, BLOCK_K):
            offs_k = k + tl.arange(0, BLOCK_K)
            mask_k = offs_k < BLKSZ
            W_block_offset = pid_block * sW_n
            W_oh_offset = offs_oh[:, None] * sW_o + offs_k[None, :]
            Wr_t = tl.load(
                Wh_ptr + 0 * sW_g + W_block_offset + W_oh_offset,
                mask=mask_k[None, :], other=0.0,
            )
            Wz_t = tl.load(
                Wh_ptr + 1 * sW_g + W_block_offset + W_oh_offset,
                mask=mask_k[None, :], other=0.0,
            )
            Wn_t = tl.load(
                Wh_ptr + 2 * sW_g + W_block_offset + W_oh_offset,
                mask=mask_k[None, :], other=0.0,
            )

            # dh_via_W tile: sum over gates of dgh[g] @ W[g, :, :]
            # dgh_r: [BLOCK_B, BLKSZ], Wr_t: [BLKSZ, BLOCK_K] -> [BLOCK_B, BLOCK_K]
            contrib = (
                tl.dot(dgh_r, Wr_t, input_precision="tf32")
                + tl.dot(dgh_z, Wz_t, input_precision="tf32")
                + tl.dot(dgh_n, Wn_t, input_precision="tf32")
            )

            # Each k-tile of `contrib` writes to a DIFFERENT range of
            # dh_acc (offs_k shifts by BLOCK_K per iter), so we always
            # store — no read-modify-write across k iters. The previous
            # step's value at this cell has already been consumed (we
            # read `dh_acc_oh` into `dh_t` at the top of the step).
            dh_w_ptrs = (
                dh_acc_ptr
                + offs_b[:, None] * sdh_b
                + (pid_block * BLKSZ + offs_k)[None, :]
            )
            tl.store(
                dh_w_ptrs, contrib,
                mask=mask_b[:, None] & mask_k[None, :],
            )

            # dWh_partial accumulation: dgh^T @ h_prev_chunk, per gate.
            # Load h_prev tile for this k range.
            h_prev_ptrs = (
                h_prev_ptr
                + offs_b[:, None] * sh_prev_b
                + (pid_block * BLKSZ + offs_k)[None, :]
            )
            h_prev_tile = tl.load(
                h_prev_ptrs, mask=mask_b[:, None] & mask_k[None, :], other=0.0,
            )
            # [BLKSZ, BLOCK_B] @ [BLOCK_B, BLOCK_K] -> [BLKSZ, BLOCK_K]
            dWr = tl.dot(tl.trans(dgh_r), h_prev_tile, input_precision="tf32")
            dWz = tl.dot(tl.trans(dgh_z), h_prev_tile, input_precision="tf32")
            dWn = tl.dot(tl.trans(dgh_n), h_prev_tile, input_precision="tf32")

            dWh_base = (
                dWh_partial_ptr
                + pid_b * sdWp_pid
                + pid_block * sdWp_n
            )
            Wr_dW_ptrs = dWh_base + 0 * sdWp_g + offs_oh[:, None] * sdWp_o + offs_k[None, :]
            Wz_dW_ptrs = dWh_base + 1 * sdWp_g + offs_oh[:, None] * sdWp_o + offs_k[None, :]
            Wn_dW_ptrs = dWh_base + 2 * sdWp_g + offs_oh[:, None] * sdWp_o + offs_k[None, :]
            mask_okok = mask_k[None, :]
            tl.store(
                Wr_dW_ptrs,
                tl.load(Wr_dW_ptrs, mask=mask_okok, other=0.0) + dWr,
                mask=mask_okok,
            )
            tl.store(
                Wz_dW_ptrs,
                tl.load(Wz_dW_ptrs, mask=mask_okok, other=0.0) + dWz,
                mask=mask_okok,
            )
            tl.store(
                Wn_dW_ptrs,
                tl.load(Wn_dW_ptrs, mask=mask_okok, other=0.0) + dWn,
                mask=mask_okok,
            )

        # Add dh_prev_direct to dh_acc[this block range] (offs_oh).
        dh_dir_ptrs = (
            dh_acc_ptr
            + offs_b[:, None] * sdh_b
            + (pid_block * BLKSZ + offs_oh)[None, :]
        )
        existing = tl.load(dh_dir_ptrs, mask=mask_b[:, None], other=0.0)
        tl.store(dh_dir_ptrs, existing + dh_prev_direct, mask=mask_b[:, None])

        # Cross-CTA barrier: ensures next iteration sees this step's writes.
        tl.atomic_add(barrier_ptr + t_rev, 1, sem="release")
        done = tl.atomic_add(barrier_ptr + t_rev, 0, sem="acquire")
        while done < NUM_PROGRAMS:
            done = tl.atomic_add(barrier_ptr + t_rev, 0, sem="acquire")

    # Final dh_acc -> dh0 for this block range.
    dh_final_ptrs = (
        dh_acc_ptr
        + offs_b[:, None] * sdh_b
        + (pid_block * BLKSZ + offs_oh)[None, :]
    )
    dh_final = tl.load(
        dh_final_ptrs, mask=mask_b[:, None], other=0.0, cache_modifier=".cv",
    )
    dh0_ptrs = (
        dh0_ptr
        + offs_b[:, None] * sdh0_b
        + (pid_block * BLKSZ + offs_oh)[None, :]
    )
    tl.store(dh0_ptrs, dh_final, mask=mask_b[:, None])


def gru_scan_monarch_backward_triton(
    gi: torch.Tensor,
    h0: torch.Tensor,
    Wh_struct: torch.Tensor,
    bh_cat: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    *,
    block_b: int = 16,
    block_k: int = 32,
    num_warps: int = 4,
    num_stages: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Triton backward for the Monarch hidden-side scan."""
    T, B, three_H = gi.shape
    H = three_H // 3
    n_gates, nblocks, out_blksz, in_blksz = Wh_struct.shape
    assert n_gates == 3 and out_blksz == in_blksz == H // nblocks

    gi = gi.contiguous()
    h0 = h0.contiguous()
    Wh_struct = Wh_struct.contiguous()
    bh_cat = bh_cat.contiguous()
    out = out.contiguous()
    dout = dout.contiguous()

    dgi = torch.zeros_like(gi)
    dh0 = torch.zeros_like(h0)

    n_pid_b = triton.cdiv(B, block_b)
    num_programs = n_pid_b * nblocks

    sm_count = torch.cuda.get_device_properties(gi.device).multi_processor_count
    if num_programs > sm_count:
        raise RuntimeError(
            f"persistent grid {num_programs} > SM count {sm_count}; "
            f"would deadlock on the spin-wait barrier."
        )

    dWh_partial = torch.zeros(
        (n_pid_b, 3, nblocks, out_blksz, in_blksz),
        device=gi.device, dtype=gi.dtype,
    )
    dbh_partial = torch.zeros(
        (n_pid_b, 3 * H), device=gi.device, dtype=gi.dtype,
    )
    dh_acc = torch.zeros((B, H), device=gi.device, dtype=gi.dtype)
    barrier = torch.zeros((T,), device=gi.device, dtype=torch.int32)

    grid = (n_pid_b, nblocks)
    gru_scan_monarch_bwd_kernel[grid](
        gi, h0, Wh_struct, bh_cat, out,
        dout,
        dgi, dh0,
        dWh_partial, dbh_partial,
        dh_acc, barrier,
        T, B,
        gi.stride(0), gi.stride(1),
        h0.stride(0),
        Wh_struct.stride(0), Wh_struct.stride(1), Wh_struct.stride(2),
        out.stride(0), out.stride(1),
        dout.stride(0), dout.stride(1),
        dgi.stride(0), dgi.stride(1),
        dh0.stride(0),
        dWh_partial.stride(0), dWh_partial.stride(1),
        dWh_partial.stride(2), dWh_partial.stride(3),
        dbh_partial.stride(0),
        dh_acc.stride(0),
        num_programs,
        H=H,
        BLKSZ=out_blksz,
        NBLOCKS=nblocks,
        BLOCK_B=block_b,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    dWh_struct = dWh_partial.sum(dim=0)
    dbh = dbh_partial.sum(dim=0)
    return dgi, dh0, dWh_struct, dbh


class GRUScanMonarchFunction(torch.autograd.Function):
    """autograd wrapper around the Monarch persistent kernels.

    Forward and backward both use 2D persistent grids (batch_tile,
    block) with a cross-CTA barrier between timesteps. fp32 only — no
    fake-quant on hidden state in this path yet (would mirror the dense
    persistent kernel's QUANT_H_IN/QUANT_H_OUT, future work).
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        gi: torch.Tensor,
        h0: torch.Tensor,
        Wh_struct: torch.Tensor,
        bh_cat: torch.Tensor,
    ) -> torch.Tensor:
        out = gru_scan_monarch_forward_triton(gi, h0, Wh_struct, bh_cat)
        ctx.save_for_backward(gi, h0, Wh_struct, bh_cat, out)
        return out

    @staticmethod
    def backward(ctx, dout):  # type: ignore[override]
        gi, h0, Wh_struct, bh_cat, out = ctx.saved_tensors
        return gru_scan_monarch_backward_triton(
            gi, h0, Wh_struct, bh_cat, out, dout
        )


def gru_scan_monarch(
    gi: torch.Tensor,
    h0: torch.Tensor,
    Wh_struct: torch.Tensor,
    bh_cat: torch.Tensor,
) -> torch.Tensor:
    """Public API: differentiable Monarch-hidden-side GRU scan.

    Mirror of ``gru_scan_persistent`` but with block-diagonal Wh:
    - ``Wh_struct: [3, nblocks, blksz, blksz]`` where ``blksz = H/nblocks``.
    - ``bh_cat: [3*H]``, same as dense.
    Use ``extract_monarch_factors(cell)`` to pull these out of a tier-1
    structured GRUCellQuant.
    """
    return GRUScanMonarchFunction.apply(gi, h0, Wh_struct, bh_cat)


def gru_scan_monarch_backward_pytorch(
    gi: torch.Tensor,
    h0: torch.Tensor,
    Wh_struct: torch.Tensor,
    bh_cat: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference backward.

    Returns:
        dgi:        [T, B, 3H]
        dh0:        [B, H]
        dWh_struct: [3, nblocks, blksz, blksz]
        dbh_cat:    [3H]
    """
    T, B, _ = gi.shape
    H = h0.shape[-1]
    n_gates, nblocks, out_blksz, in_blksz = Wh_struct.shape
    blksz = out_blksz

    dgi = torch.zeros_like(gi)
    dWh_struct = torch.zeros_like(Wh_struct)
    dbh = torch.zeros_like(bh_cat)
    dh_acc = torch.zeros_like(h0)

    for t in reversed(range(T)):
        h_prev = h0 if t == 0 else out[t - 1]

        # Forward recompute
        gi_r = gi[t, :, 0:H]
        gi_z = gi[t, :, H:2 * H]
        gi_n = gi[t, :, 2 * H:3 * H]
        h_chunks = h_prev.view(B, nblocks, blksz)
        gh = torch.einsum("bni,gnoi->bgno", h_chunks, Wh_struct)
        gh = gh.reshape(B, 3, H) + bh_cat.view(3, H)
        gh_r = gh[:, 0, :]
        gh_z = gh[:, 1, :]
        gh_n = gh[:, 2, :]
        r = torch.sigmoid(gi_r + gh_r)
        z = torch.sigmoid(gi_z + gh_z)
        n = torch.tanh(gi_n + r * gh_n)

        dh_t = dout[t] + dh_acc

        # h_t = (1-z)*n + z*h_prev
        dn = dh_t * (1.0 - z)
        dz = dh_t * (h_prev - n)
        dh_prev_direct = dh_t * z

        # n = tanh(gn_pre); gn_pre = gi_n + r*gh_n
        dgn_pre = dn * (1.0 - n * n)
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

        # Stack dgh per gate into [B, 3, H] -> reshape to [B, 3, nblocks, blksz]
        dgh = torch.stack([dgh_r, dgh_z, dgh_n], dim=1)  # [B, 3, H]
        dbh += dgh.sum(dim=0).reshape(-1)  # accumulate over batch and time
        dgh_chunks = dgh.view(B, 3, nblocks, blksz)

        # gh = einsum('bni,gnoi->bgno', h_chunks, Wh_struct)
        # Backward:
        #   dWh_struct[g, n, o, i] += sum_b dgh[b, g, n, o] * h_chunks[b, n, i]
        #   dh_chunks[b, n, i] += sum_{g,o} dgh[b, g, n, o] * Wh_struct[g, n, o, i]
        dWh_struct += torch.einsum("bgno,bni->gnoi", dgh_chunks, h_chunks)
        dh_via_W_chunks = torch.einsum(
            "bgno,gnoi->bni", dgh_chunks, Wh_struct
        )
        dh_via_W = dh_via_W_chunks.reshape(B, H)

        dh_acc = dh_prev_direct + dh_via_W

    return dgi, dh_acc, dWh_struct, dbh

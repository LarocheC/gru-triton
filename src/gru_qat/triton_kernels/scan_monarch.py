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

from typing import Any, cast

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
    # [nblocks, out_blksz, in_blksz] factor tensor. nn.Module attribute
    # access is typed Tensor | Module by the torch stubs — cast to the
    # concrete submodule, then read the Tensor weight.
    Wr = cast(torch.Tensor, cast(nn.Module, cell.struct_Wh_r).weight)
    Wz = cast(torch.Tensor, cast(nn.Module, cell.struct_Wh_z).weight)
    Wn = cast(torch.Tensor, cast(nn.Module, cell.struct_Wh_n).weight)
    Wh_struct = torch.stack([Wr, Wz, Wn], dim=0)  # [3, nblocks, blksz, blksz]
    if cell.b_hr is None:
        bh_cat = torch.zeros(3 * cell.hidden_size, device=Wh_struct.device, dtype=Wh_struct.dtype)
    else:
        bh_cat = torch.cat(
            [
                cast(torch.Tensor, cell.b_hr),
                cast(torch.Tensor, cell.b_hz),
                cast(torch.Tensor, cell.b_hn),
            ]
        )
    return Wh_struct, bh_cat


def _fake_quant(
    x: torch.Tensor, params: tuple[float, int, int] | None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Per-tensor symmetric fake-quant + STE clip mask. None passes through."""
    if params is None:
        return x, None
    scale, qmin, qmax = params
    q_unclamped = torch.round(x / scale)
    mask = (q_unclamped >= qmin) & (q_unclamped <= qmax)
    q_clamped = q_unclamped.clamp(qmin, qmax)
    return q_clamped * scale, mask


def gru_scan_monarch_forward_pytorch(
    gi: torch.Tensor,
    h0: torch.Tensor,
    Wh_struct: torch.Tensor,
    bh_cat: torch.Tensor,
    *,
    h_in_quant: tuple[float, int, int] | None = None,
    h_out_quant: tuple[float, int, int] | None = None,
) -> torch.Tensor:
    """Reference forward for the block-diagonal scan, in PyTorch.

    Args:
        gi: [T, B, 3H] — pre-batched input projection (already with bi).
        h0: [B, H]
        Wh_struct: [3, nblocks, blksz, blksz]
        bh_cat: [3H]
        h_in_quant: optional (scale, qmin, qmax) for matmul-side h.
        h_out_quant: optional (scale, qmin, qmax) for h_new before store.
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
        # quant_h_in only on the matmul-side (direct contribution uses raw h).
        h_for_matmul, _ = _fake_quant(h, h_in_quant)
        h_chunks = h_for_matmul.view(B, nblocks, blksz)
        gh = torch.einsum("bni,gnoi->bgno", h_chunks, Wh_struct)
        gh = gh.reshape(B, 3, H) + bh
        gh_r, gh_z, gh_n = gh[:, 0, :], gh[:, 1, :], gh[:, 2, :]

        gi_r = gi[t, :, 0:H]
        gi_z = gi[t, :, H:2 * H]
        gi_n = gi[t, :, 2 * H:3 * H]

        r = torch.sigmoid(gi_r + gh_r)
        z = torch.sigmoid(gi_z + gh_z)
        n = torch.tanh(gi_n + r * gh_n)
        h_new = (1.0 - z) * n + z * h
        h_new, _ = _fake_quant(h_new, h_out_quant)
        out[t] = h_new
        h = h_new

    return out


@triton.jit  # type: ignore[untyped-decorator]
def gru_scan_monarch_fwd_kernel(  # type: ignore[no-untyped-def]
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
    h_in_scale,
    h_in_qmin,
    h_in_qmax,
    h_out_scale,
    h_out_qmin,
    h_out_qmax,
    H: tl.constexpr,
    BLKSZ: tl.constexpr,
    BLKSZ_PAD: tl.constexpr,
    NBLOCKS: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
    QUANT_H_IN: tl.constexpr,
    QUANT_H_OUT: tl.constexpr,
):
    """Persistent forward over the block-diagonal recurrence.

    Grid (pid_b, pid_block): each program handles ALL 3 gates for ONE
    block, producing [BLOCK_B, blksz] of h_t for that block. Block
    boundaries don't mix in the matmul (block-diagonal), so the K
    reduction is only over blksz instead of full H — that's where the
    win comes from.

    BLKSZ may be any positive integer; BLKSZ_PAD is the next power of 2
    (Triton requires pow-2 ``tl.arange`` lengths). mask_oh = offs_oh <
    BLKSZ excludes the padded tail from every memory op so we don't
    corrupt the adjacent block / gate slice in the dense H/3H tensors.
    """
    pid_b = tl.program_id(0)
    pid_block = tl.program_id(1)

    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = offs_b < B
    offs_oh = tl.arange(0, BLKSZ_PAD)  # output rows within this block (padded)
    mask_oh = offs_oh < BLKSZ

    # Output position in flat 3H layout: gate * H + pid_block * BLKSZ + offs_oh.
    # h-input range: pid_block * BLKSZ + offs_k.

    # Pre-load the bias slice for this (gate, block) — 3 chunks of BLKSZ.
    bh_offset = pid_block * BLKSZ
    bhr_tile = tl.load(bh_ptr + 0 * H + bh_offset + offs_oh, mask=mask_oh, other=0.0)
    bhz_tile = tl.load(bh_ptr + 1 * H + bh_offset + offs_oh, mask=mask_oh, other=0.0)
    bhn_tile = tl.load(bh_ptr + 2 * H + bh_offset + offs_oh, mask=mask_oh, other=0.0)

    h_in_ptr = h0_ptr
    sh_b = sh0_b

    for t in range(0, T):
        ghr = tl.zeros((BLOCK_B, BLKSZ_PAD), dtype=tl.float32)
        ghz = tl.zeros((BLOCK_B, BLKSZ_PAD), dtype=tl.float32)
        ghn = tl.zeros((BLOCK_B, BLKSZ_PAD), dtype=tl.float32)

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
            # quant_h_in only on the matmul-side h. The direct contribution
            # to h_new uses raw h_old below — matches gru_cell.step.
            if QUANT_H_IN:
                q = tl.extra.cuda.libdevice.rint(h_tile / h_in_scale)
                q = tl.minimum(tl.maximum(q, h_in_qmin), h_in_qmax)
                h_tile = q * h_in_scale

            # Three W tiles, one per gate. Each is [BLKSZ_PAD, BLOCK_K].
            # mask_oh zeroes the padded oh rows so they contribute 0 to
            # the dot product over the BLKSZ_PAD axis.
            W_block_offset = pid_block * sW_n
            W_oh_offset = offs_oh[:, None] * sW_o + offs_k[None, :]
            W_mask = mask_oh[:, None] & mask_k[None, :]
            Wr_tile = tl.load(
                Wh_ptr + 0 * sW_g + W_block_offset + W_oh_offset,
                mask=W_mask, other=0.0,
            )
            Wz_tile = tl.load(
                Wh_ptr + 1 * sW_g + W_block_offset + W_oh_offset,
                mask=W_mask, other=0.0,
            )
            Wn_tile = tl.load(
                Wh_ptr + 2 * sW_g + W_block_offset + W_oh_offset,
                mask=W_mask, other=0.0,
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
        gi_mask = mask_b[:, None] & mask_oh[None, :]
        gir = tl.load(gi_base + 0 * H, mask=gi_mask, other=0.0)
        giz = tl.load(gi_base + 1 * H, mask=gi_mask, other=0.0)
        gin = tl.load(gi_base + 2 * H, mask=gi_mask, other=0.0)

        r = tl.sigmoid(gir + ghr)
        z = tl.sigmoid(giz + ghz)
        n = tl.extra.cuda.libdevice.tanh(gin + r * ghn)

        # h_t = (1-z)*n + z*h_prev at THIS block's output positions.
        h_old_ptrs = (
            h_in_ptr
            + offs_b[:, None] * sh_b
            + (pid_block * BLKSZ + offs_oh)[None, :]
        )
        h_old = tl.load(h_old_ptrs, mask=gi_mask, other=0.0)
        h_new = (1.0 - z) * n + z * h_old

        if QUANT_H_OUT:
            q = tl.extra.cuda.libdevice.rint(h_new / h_out_scale)
            q = tl.minimum(tl.maximum(q, h_out_qmin), h_out_qmax)
            h_new = q * h_out_scale

        out_ptrs = (
            out_ptr
            + t * so_t
            + offs_b[:, None] * so_b
            + (pid_block * BLKSZ + offs_oh)[None, :]
        )
        tl.store(out_ptrs, h_new, mask=gi_mask)

        # Cross-CTA barrier: pair release/acquire same as dense persistent.
        tl.atomic_add(barrier_ptr + t, 1, sem="release")
        done = tl.atomic_add(barrier_ptr + t, 0, sem="acquire")
        while done < NUM_PROGRAMS:
            done = tl.atomic_add(barrier_ptr + t, 0, sem="acquire")

        h_in_ptr = out_ptr + t * so_t
        sh_b = so_b


def _pick_tile(blksz_pad: int, *, fwd: bool) -> tuple[int, int, int]:
    """Pick (BLOCK_B, BLOCK_K, num_stages) given the padded block size.

    Triton's tl.dot needs all tile dims >= 16, so BLOCK_B and BLOCK_K
    stay at 16 minimum. The dominant smem consumer is the three W
    tiles (3 * BLKSZ_PAD * BLOCK_K * 4 bytes), double-buffered when
    num_stages > 1. Below the threshold the default (32, num_stages=2
    fwd / 1 bwd) keeps tensor cores fed; at large BLKSZ_PAD we drop to
    BLOCK_K=16 + num_stages=1 to stay under the 4090's 100KB SMEM/SM.
    """
    block_b = 16
    if blksz_pad <= 128:
        # Default 4090-comfortable config: W tiles = 3 * 128 * 32 * 4 = 48KB
        # plus accumulators 3 * 16 * 128 * 4 = 24KB. ~72KB total before
        # double-buffer — fits at num_stages=2.
        block_k = 32
        num_stages = 2 if fwd else 1
    else:
        # BLKSZ_PAD=256: with BLOCK_K=16 num_stages=1 the W tiles are
        # 3 * 256 * 16 * 4 = 48KB and accumulators 48KB = 96KB. Just under
        # the 100KB limit on a 4090.
        block_k = 16
        num_stages = 1
    return block_b, block_k, num_stages


def gru_scan_monarch_forward_triton(
    gi: torch.Tensor,
    h0: torch.Tensor,
    Wh_struct: torch.Tensor,
    bh_cat: torch.Tensor,
    *,
    block_b: int | None = None,
    block_k: int | None = None,
    num_warps: int = 4,
    num_stages: int | None = None,
    h_in_quant: tuple[float, int, int] | None = None,
    h_out_quant: tuple[float, int, int] | None = None,
) -> torch.Tensor:
    """Triton forward for the Monarch hidden-side scan."""
    if not (gi.is_cuda and Wh_struct.is_cuda):
        raise ValueError(
            "gi and Wh_struct must be CUDA tensors; got devices "
            f"gi={gi.device}, Wh_struct={Wh_struct.device}"
        )
    T, B, three_H = gi.shape
    H = three_H // 3
    n_gates, nblocks, out_blksz, in_blksz = Wh_struct.shape
    if n_gates != 3:
        raise ValueError(
            f"Wh_struct dim 0 (n_gates) must be 3; got {n_gates} "
            f"(shape {tuple(Wh_struct.shape)})"
        )
    if not (out_blksz == in_blksz == H // nblocks):
        raise ValueError(
            f"Wh_struct block dims must be square and tile H: expected "
            f"out_blksz == in_blksz == H // nblocks == {H // nblocks}; got "
            f"out_blksz={out_blksz}, in_blksz={in_blksz}, H={H}, nblocks={nblocks}"
        )

    gi = gi.contiguous()
    h0 = h0.contiguous()
    Wh_struct = Wh_struct.contiguous()
    bh_cat = bh_cat.contiguous()

    out = torch.empty((T, B, H), device=gi.device, dtype=gi.dtype)
    barrier = torch.zeros((T,), device=gi.device, dtype=torch.int32)

    blksz_pad = triton.next_power_of_2(out_blksz)
    sm_count = torch.cuda.get_device_properties(gi.device).multi_processor_count
    auto_b, auto_k, auto_s = _pick_tile(blksz_pad, fwd=True)
    if block_b is None:
        block_b = auto_b
    if block_k is None:
        block_k = auto_k
    if num_stages is None:
        num_stages = auto_s
    n_pid_b = triton.cdiv(B, block_b)
    num_programs = n_pid_b * nblocks

    if num_programs > sm_count:
        raise RuntimeError(
            f"persistent grid {num_programs} > SM count {sm_count}; "
            f"would deadlock on the spin-wait barrier."
        )

    in_s, in_qmin, in_qmax = h_in_quant or (1.0, -2**31, 2**31 - 1)
    out_s, out_qmin, out_qmax = h_out_quant or (1.0, -2**31, 2**31 - 1)

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
        in_s, in_qmin, in_qmax,
        out_s, out_qmin, out_qmax,
        H=H,
        BLKSZ=out_blksz,
        BLKSZ_PAD=blksz_pad,
        NBLOCKS=nblocks,
        BLOCK_B=block_b,
        BLOCK_K=block_k,
        QUANT_H_IN=h_in_quant is not None,
        QUANT_H_OUT=h_out_quant is not None,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


@triton.jit  # type: ignore[untyped-decorator]
def gru_scan_monarch_bwd_kernel(  # type: ignore[no-untyped-def]
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
    h_in_scale,
    h_in_qmin,
    h_in_qmax,
    h_out_scale,
    h_out_qmin,
    h_out_qmax,
    H: tl.constexpr,
    BLKSZ: tl.constexpr,
    BLKSZ_PAD: tl.constexpr,
    NBLOCKS: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
    QUANT_H_IN: tl.constexpr,
    QUANT_H_OUT: tl.constexpr,
):
    """Persistent backward over block-diagonal recurrence.

    Each program owns one (batch_tile, output_block) and stays in that
    slot for all T timesteps. dh_acc accumulation, dWh partial, and dbh
    partial are all partition-disjoint across (pid_b, pid_block) pairs:
    block i's dh_via_W only feeds block i's input slice. So no atomic
    adds or ping-pong buffers — just a single dh_acc scratch with a
    cross-CTA barrier between timesteps.

    BLKSZ may be any positive integer; BLKSZ_PAD is the next power of 2.
    mask_oh = offs_oh < BLKSZ excludes the padded tail from every memory
    op so we don't corrupt adjacent block / gate slices.
    """
    pid_b = tl.program_id(0)
    pid_block = tl.program_id(1)

    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = offs_b < B
    offs_oh = tl.arange(0, BLKSZ_PAD)
    mask_oh = offs_oh < BLKSZ
    mask_bo = mask_b[:, None] & mask_oh[None, :]

    # Pre-load bias slice (constant across T).
    bh_offset = pid_block * BLKSZ
    bhr_tile = tl.load(bh_ptr + 0 * H + bh_offset + offs_oh, mask=mask_oh, other=0.0)
    bhz_tile = tl.load(bh_ptr + 1 * H + bh_offset + offs_oh, mask=mask_oh, other=0.0)
    bhn_tile = tl.load(bh_ptr + 2 * H + bh_offset + offs_oh, mask=mask_oh, other=0.0)

    # Initialize dh_acc[:, this block range] to zero.
    dh_acc_init_ptrs = (
        dh_acc_ptr
        + offs_b[:, None] * sdh_b
        + (pid_block * BLKSZ + offs_oh)[None, :]
    )
    tl.store(
        dh_acc_init_ptrs,
        tl.zeros((BLOCK_B, BLKSZ_PAD), dtype=tl.float32),
        mask=mask_bo,
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
        ghr = tl.zeros((BLOCK_B, BLKSZ_PAD), dtype=tl.float32)
        ghz = tl.zeros((BLOCK_B, BLKSZ_PAD), dtype=tl.float32)
        ghn = tl.zeros((BLOCK_B, BLKSZ_PAD), dtype=tl.float32)
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
            if QUANT_H_IN:
                q = tl.extra.cuda.libdevice.rint(h_tile / h_in_scale)
                q = tl.minimum(tl.maximum(q, h_in_qmin), h_in_qmax)
                h_tile = q * h_in_scale
            W_block_offset = pid_block * sW_n
            W_oh_offset = offs_oh[:, None] * sW_o + offs_k[None, :]
            W_mask = mask_oh[:, None] & mask_k[None, :]
            Wr_tile = tl.load(
                Wh_ptr + 0 * sW_g + W_block_offset + W_oh_offset,
                mask=W_mask, other=0.0,
            )
            Wz_tile = tl.load(
                Wh_ptr + 1 * sW_g + W_block_offset + W_oh_offset,
                mask=W_mask, other=0.0,
            )
            Wn_tile = tl.load(
                Wh_ptr + 2 * sW_g + W_block_offset + W_oh_offset,
                mask=W_mask, other=0.0,
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
        gir = tl.load(gi_base + 0 * H, mask=mask_bo, other=0.0)
        giz = tl.load(gi_base + 1 * H, mask=mask_bo, other=0.0)
        gin = tl.load(gi_base + 2 * H, mask=mask_bo, other=0.0)

        r = tl.sigmoid(gir + ghr)
        z = tl.sigmoid(giz + ghz)
        n = tl.extra.cuda.libdevice.tanh(gin + r * ghn)

        # h_prev at THIS block's positions (for dh_prev_direct and (1-z)*n + z*h_prev).
        h_prev_oh_ptrs = (
            h_prev_ptr
            + offs_b[:, None] * sh_prev_b
            + (pid_block * BLKSZ + offs_oh)[None, :]
        )
        h_prev_oh = tl.load(h_prev_oh_ptrs, mask=mask_bo, other=0.0)

        # ---- Read incoming dh_acc[this block range].
        dh_acc_ptrs = (
            dh_acc_ptr
            + offs_b[:, None] * sdh_b
            + (pid_block * BLKSZ + offs_oh)[None, :]
        )
        dh_acc_oh = tl.load(
            dh_acc_ptrs, mask=mask_bo, other=0.0,
        )
        dout_base = (
            dout_ptr
            + t * sdo_t
            + offs_b[:, None] * sdo_b
            + (pid_block * BLKSZ + offs_oh)[None, :]
        )
        dout_oh = tl.load(dout_base, mask=mask_bo, other=0.0)
        dh_t = dout_oh + dh_acc_oh

        # STE backward of quant_h_out: incoming dh_t is grad on quantized
        # h_t. Recompute h_t_raw to derive the clip mask.
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

        # Store dgi[t] for this block range, all 3 gates.
        dgi_base = (
            dgi_ptr
            + t * sdgi_t
            + offs_b[:, None] * sdgi_b
            + (pid_block * BLKSZ + offs_oh)[None, :]
        )
        tl.store(dgi_base + 0 * H, dgi_r, mask=mask_bo)
        tl.store(dgi_base + 1 * H, dgi_z, mask=mask_bo)
        tl.store(dgi_base + 2 * H, dgi_n, mask=mask_bo)

        # dbh_partial: each (pid_b, pid_block) is the only writer to its
        # bias slice rows. Padded oh lanes must be masked or they'd land
        # in the next block / gate slot.
        dbh_base = dbh_partial_ptr + pid_b * sdbp_pid + bh_offset + offs_oh
        tl.store(
            dbh_base + 0 * H,
            tl.load(dbh_base + 0 * H, mask=mask_oh, other=0.0) + tl.sum(dgh_r, axis=0),
            mask=mask_oh,
        )
        tl.store(
            dbh_base + 1 * H,
            tl.load(dbh_base + 1 * H, mask=mask_oh, other=0.0) + tl.sum(dgh_z, axis=0),
            mask=mask_oh,
        )
        tl.store(
            dbh_base + 2 * H,
            tl.load(dbh_base + 2 * H, mask=mask_oh, other=0.0) + tl.sum(dgh_n, axis=0),
            mask=mask_oh,
        )

        # ---- dh_prev_via_W and dWh_partial accumulation ----
        # Both are per-block (no cross-block contributions). For each k-tile,
        # compute the dh_via_W contribution to dh_acc[this block, k:k+BLOCK_K]
        # and the dWh contribution dgh^T @ h_prev (per gate).
        for k in range(0, BLKSZ, BLOCK_K):
            offs_k = k + tl.arange(0, BLOCK_K)
            mask_k = offs_k < BLKSZ
            W_block_offset = pid_block * sW_n
            W_oh_offset = offs_oh[:, None] * sW_o + offs_k[None, :]
            W_mask = mask_oh[:, None] & mask_k[None, :]
            Wr_t = tl.load(
                Wh_ptr + 0 * sW_g + W_block_offset + W_oh_offset,
                mask=W_mask, other=0.0,
            )
            Wz_t = tl.load(
                Wh_ptr + 1 * sW_g + W_block_offset + W_oh_offset,
                mask=W_mask, other=0.0,
            )
            Wn_t = tl.load(
                Wh_ptr + 2 * sW_g + W_block_offset + W_oh_offset,
                mask=W_mask, other=0.0,
            )

            # dh_via_W tile: sum over gates of dgh[g] @ W[g, :, :]
            # dgh_r: [BLOCK_B, BLKSZ_PAD], Wr_t: [BLKSZ_PAD, BLOCK_K]
            # -> [BLOCK_B, BLOCK_K]. Padded oh rows of W are 0 (masked
            # load), so padded-row garbage in dgh contributes nothing.
            contrib = (
                tl.dot(dgh_r, Wr_t, input_precision="tf32")
                + tl.dot(dgh_z, Wz_t, input_precision="tf32")
                + tl.dot(dgh_n, Wn_t, input_precision="tf32")
            )
            h_prev_k_ptrs = (
                h_prev_ptr
                + offs_b[:, None] * sh_prev_b
                + (pid_block * BLKSZ + offs_k)[None, :]
            )
            h_prev_tile_raw = tl.load(
                h_prev_k_ptrs,
                mask=mask_b[:, None] & mask_k[None, :],
                other=0.0,
            )
            if QUANT_H_IN:
                q_in_unclamped = tl.extra.cuda.libdevice.rint(
                    h_prev_tile_raw / h_in_scale
                )
                mask_in = (q_in_unclamped >= h_in_qmin) & (
                    q_in_unclamped <= h_in_qmax
                )
                contrib = tl.where(mask_in, contrib, 0.0)

            # Each k-tile of `contrib` writes to a DIFFERENT range of dh_acc.
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
            h_prev_tile = h_prev_tile_raw
            if QUANT_H_IN:
                q = tl.extra.cuda.libdevice.rint(h_prev_tile / h_in_scale)
                q = tl.minimum(tl.maximum(q, h_in_qmin), h_in_qmax)
                h_prev_tile = q * h_in_scale
            # [BLKSZ_PAD, BLOCK_B] @ [BLOCK_B, BLOCK_K] -> [BLKSZ_PAD, BLOCK_K].
            # Padded oh rows of dgh^T contain garbage; mask_oh on the
            # store ensures only real rows reach memory.
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
            dW_mask = mask_oh[:, None] & mask_k[None, :]
            tl.store(
                Wr_dW_ptrs,
                tl.load(Wr_dW_ptrs, mask=dW_mask, other=0.0) + dWr,
                mask=dW_mask,
            )
            tl.store(
                Wz_dW_ptrs,
                tl.load(Wz_dW_ptrs, mask=dW_mask, other=0.0) + dWz,
                mask=dW_mask,
            )
            tl.store(
                Wn_dW_ptrs,
                tl.load(Wn_dW_ptrs, mask=dW_mask, other=0.0) + dWn,
                mask=dW_mask,
            )

        # Add dh_prev_direct to dh_acc[this block range] (offs_oh).
        dh_dir_ptrs = (
            dh_acc_ptr
            + offs_b[:, None] * sdh_b
            + (pid_block * BLKSZ + offs_oh)[None, :]
        )
        existing = tl.load(dh_dir_ptrs, mask=mask_bo, other=0.0)
        tl.store(dh_dir_ptrs, existing + dh_prev_direct, mask=mask_bo)

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
        dh_final_ptrs, mask=mask_bo, other=0.0,
    )
    dh0_ptrs = (
        dh0_ptr
        + offs_b[:, None] * sdh0_b
        + (pid_block * BLKSZ + offs_oh)[None, :]
    )
    tl.store(dh0_ptrs, dh_final, mask=mask_bo)


def gru_scan_monarch_backward_triton(
    gi: torch.Tensor,
    h0: torch.Tensor,
    Wh_struct: torch.Tensor,
    bh_cat: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    *,
    block_b: int | None = None,
    block_k: int | None = None,
    num_warps: int = 4,
    num_stages: int | None = None,
    h_in_quant: tuple[float, int, int] | None = None,
    h_out_quant: tuple[float, int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Triton backward for the Monarch hidden-side scan."""
    T, B, three_H = gi.shape
    H = three_H // 3
    n_gates, nblocks, out_blksz, in_blksz = Wh_struct.shape
    if n_gates != 3:
        raise ValueError(
            f"Wh_struct dim 0 (n_gates) must be 3; got {n_gates} "
            f"(shape {tuple(Wh_struct.shape)})"
        )
    if not (out_blksz == in_blksz == H // nblocks):
        raise ValueError(
            f"Wh_struct block dims must be square and tile H: expected "
            f"out_blksz == in_blksz == H // nblocks == {H // nblocks}; got "
            f"out_blksz={out_blksz}, in_blksz={in_blksz}, H={H}, nblocks={nblocks}"
        )

    gi = gi.contiguous()
    h0 = h0.contiguous()
    Wh_struct = Wh_struct.contiguous()
    bh_cat = bh_cat.contiguous()
    out = out.contiguous()
    dout = dout.contiguous()

    dgi = torch.zeros_like(gi)
    dh0 = torch.zeros_like(h0)

    blksz_pad = triton.next_power_of_2(out_blksz)
    sm_count = torch.cuda.get_device_properties(gi.device).multi_processor_count
    auto_b, auto_k, auto_s = _pick_tile(blksz_pad, fwd=False)
    if block_b is None:
        block_b = auto_b
    if block_k is None:
        block_k = auto_k
    if num_stages is None:
        num_stages = auto_s
    n_pid_b = triton.cdiv(B, block_b)
    num_programs = n_pid_b * nblocks

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

    in_s, in_qmin, in_qmax = h_in_quant or (1.0, -2**31, 2**31 - 1)
    out_s, out_qmin, out_qmax = h_out_quant or (1.0, -2**31, 2**31 - 1)

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
        in_s, in_qmin, in_qmax,
        out_s, out_qmin, out_qmax,
        H=H,
        BLKSZ=out_blksz,
        BLKSZ_PAD=blksz_pad,
        NBLOCKS=nblocks,
        BLOCK_B=block_b,
        BLOCK_K=block_k,
        QUANT_H_IN=h_in_quant is not None,
        QUANT_H_OUT=h_out_quant is not None,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    dWh_struct = dWh_partial.sum(dim=0)
    dbh = dbh_partial.sum(dim=0)
    return dgi, dh0, dWh_struct, dbh


class GRUScanMonarchFunction(torch.autograd.Function):
    """autograd wrapper around the Monarch persistent kernels.

    Forward and backward both use 2D persistent grids (batch_tile,
    block) with a cross-CTA barrier between timesteps. Optional
    in-kernel fake-quant on hidden state (per-tensor symmetric, frozen
    scale) — same semantics as ``GRUScanPersistentFunction``.
    """

    @staticmethod
    def forward(
        ctx: Any,
        gi: torch.Tensor,
        h0: torch.Tensor,
        Wh_struct: torch.Tensor,
        bh_cat: torch.Tensor,
        h_in_quant: tuple[float, int, int] | None,
        h_out_quant: tuple[float, int, int] | None,
    ) -> torch.Tensor:
        out = gru_scan_monarch_forward_triton(
            gi, h0, Wh_struct, bh_cat,
            h_in_quant=h_in_quant, h_out_quant=h_out_quant,
        )
        ctx.save_for_backward(gi, h0, Wh_struct, bh_cat, out)
        ctx.h_in_quant = h_in_quant
        ctx.h_out_quant = h_out_quant
        return out

    @staticmethod
    def backward(
        ctx: Any, dout: torch.Tensor
    ) -> Any:
        gi, h0, Wh_struct, bh_cat, out = ctx.saved_tensors
        grads = gru_scan_monarch_backward_triton(
            gi, h0, Wh_struct, bh_cat, out, dout,
            h_in_quant=ctx.h_in_quant, h_out_quant=ctx.h_out_quant,
        )
        return (*grads, None, None)


def gru_scan_monarch(
    gi: torch.Tensor,
    h0: torch.Tensor,
    Wh_struct: torch.Tensor,
    bh_cat: torch.Tensor,
    *,
    h_in_quant: tuple[float, int, int] | None = None,
    h_out_quant: tuple[float, int, int] | None = None,
) -> torch.Tensor:
    """Public API: differentiable Monarch-hidden-side GRU scan.

    Mirror of ``gru_scan_persistent`` but with block-diagonal Wh:
    - ``Wh_struct: [3, nblocks, blksz, blksz]`` where ``blksz = H/nblocks``.
    - ``bh_cat: [3*H]``, same as dense.
    With ``h_in_quant`` / ``h_out_quant`` supplied (each
    ``(scale, qmin, qmax)``), the kernel applies in-kernel fake-quant on
    hidden state every step, identical to ``gru_scan_persistent``.
    Use ``extract_monarch_factors(cell)`` to pull factors out of a
    tier-1 structured GRUCellQuant.
    """
    return cast(
        torch.Tensor,
        GRUScanMonarchFunction.apply(  # type: ignore[no-untyped-call]
            gi, h0, Wh_struct, bh_cat, h_in_quant, h_out_quant
        ),
    )


def gru_scan_monarch_backward_pytorch(
    gi: torch.Tensor,
    h0: torch.Tensor,
    Wh_struct: torch.Tensor,
    bh_cat: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    *,
    h_in_quant: tuple[float, int, int] | None = None,
    h_out_quant: tuple[float, int, int] | None = None,
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

        # Forward recompute (with quant_h_in on the matmul-side h).
        gi_r = gi[t, :, 0:H]
        gi_z = gi[t, :, H:2 * H]
        gi_n = gi[t, :, 2 * H:3 * H]
        h_for_matmul, mask_in = _fake_quant(h_prev, h_in_quant)
        h_chunks = h_for_matmul.view(B, nblocks, blksz)
        gh = torch.einsum("bni,gnoi->bgno", h_chunks, Wh_struct)
        gh = gh.reshape(B, 3, H) + bh_cat.view(3, H)
        gh_r = gh[:, 0, :]
        gh_z = gh[:, 1, :]
        gh_n = gh[:, 2, :]
        r = torch.sigmoid(gi_r + gh_r)
        z = torch.sigmoid(gi_z + gh_z)
        n = torch.tanh(gi_n + r * gh_n)
        h_t_raw = (1.0 - z) * n + z * h_prev

        dh_t = dout[t] + dh_acc

        # STE backward through quant_h_out: gradient on h_t_q -> on h_t_raw.
        if h_out_quant is not None:
            _, mask_out = _fake_quant(h_t_raw, h_out_quant)
            assert mask_out is not None  # non-None params => non-None mask
            dh_t = dh_t * mask_out

        # h_t_raw = (1-z)*n + z*h_prev
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

        # gh = einsum('bni,gnoi->bgno', h_chunks, Wh_struct)  with h_chunks
        # being the *quantized* h_for_matmul. Backward:
        #   dWh_struct accumulates against quantized h_chunks (matches forward).
        #   dh_chunks (grad on quantized h) -> grad on raw h via STE mask.
        dWh_struct += torch.einsum("bgno,bni->gnoi", dgh_chunks, h_chunks)
        dh_via_W_chunks = torch.einsum(
            "bgno,gnoi->bni", dgh_chunks, Wh_struct
        )
        dh_via_W = dh_via_W_chunks.reshape(B, H)
        if mask_in is not None:
            dh_via_W = dh_via_W * mask_in

        dh_acc = dh_prev_direct + dh_via_W

    return dgi, dh_acc, dWh_struct, dbh

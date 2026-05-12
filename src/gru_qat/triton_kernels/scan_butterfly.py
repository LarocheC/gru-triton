"""Butterfly hidden weights for the multi-step GRU scan.

Two paths live here:

1. ``gru_scan_butterfly`` — the API-parity path. Python time loop
   calling ``torch_structured.butterfly_multiply`` per step. Backward
   via standard PyTorch autograd. ~as fast as the tier-1 structured
   step path; mostly exists so ``GRULayer(use_triton="auto")`` works
   uniformly across structured kinds.

2. ``gru_scan_butterfly_triton`` — multi-step persistent Triton kernel.
   Implements the butterfly multiply directly in Triton (log_N stages
   of strided 2×2 mixing) and fuses the recurrence across timesteps.
   No tensor-core utilization (butterfly's 2×2 mixing isn't a GEMM
   shape), so the win comes purely from launch-count reduction:
   T×ops_per_step launches → one launch per train-step half.

Forward kernel layout (Triton path):
- Grid: (cdiv(B, BLOCK_B),). Each program owns [BLOCK_B, H] state and
  runs ALL T timesteps independently. Butterfly's recurrence is
  per-batch-row independent so no cross-CTA sync is needed.
- Per timestep: 3 butterfly multiplies (one per gate) into per-gate
  scratch buffers in global memory, then gate compose, then store h_t.
- Per butterfly stage: load self + (XOR stride) partner, apply the 2×2
  twiddle for this stage's pair index, scatter back. Triton's register
  tensors don't allow dynamic gather/scatter so the running state
  passes through global memory between stages — L2 absorbs the cost.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import triton
import triton.language as tl

from gru_qat.ste import fake_quant_ste


def extract_butterfly_factors(
    cell: nn.Module,
) -> tuple[list[nn.Module], torch.Tensor]:
    """Pull the three hidden-side Butterfly modules out of a tier-1 cell.

    Returns the underlying ``torch_structured.Butterfly`` instances
    rather than raw twiddles — Butterfly's forward pre/post-processes
    its input (reshape into ``[batch, nstacks, in_size]`` etc.) so it's
    cleaner to call the module than to reproduce the wrapping ourselves.

    Args:
        cell: a ``GRUCellQuant`` whose ``structure_hidden`` was a
              ``StructureConfig(kind="butterfly", ...)``.

    Returns:
        modules: list of three ``Butterfly`` modules, one per gate
                 (r, z, n).
        bh_cat:  [3*H] — concat of (b_hr, b_hz, b_hn).
    """
    if cell._hidden_dense:
        raise ValueError("cell hidden side is dense; nothing to extract")
    # struct_Wh_* are _ButterflyLinear wrappers; .b is the underlying
    # torch_structured.Butterfly nn.Module.
    modules = [
        cell.struct_Wh_r.b,
        cell.struct_Wh_z.b,
        cell.struct_Wh_n.b,
    ]
    sample = next(modules[0].parameters())
    if cell.b_hr is None:
        bh_cat = torch.zeros(
            3 * cell.hidden_size, device=sample.device, dtype=sample.dtype,
        )
    else:
        bh_cat = torch.cat([cell.b_hr, cell.b_hz, cell.b_hn])
    return modules, bh_cat


def _maybe_fake_quant(
    x: torch.Tensor, params: tuple[float, int, int] | None
) -> torch.Tensor:
    """Apply per-tensor symmetric fake-quant when params is provided."""
    if params is None:
        return x
    scale, qmin, qmax = params
    s = torch.tensor(scale, device=x.device, dtype=x.dtype)
    zp = torch.tensor(0.0, device=x.device, dtype=x.dtype)
    return fake_quant_ste(x, s, zp, qmin, qmax)


def gru_scan_butterfly(
    gi: torch.Tensor,
    h0: torch.Tensor,
    butterfly_modules: list[nn.Module],
    bh_cat: torch.Tensor,
    *,
    h_in_quant: tuple[float, int, int] | None = None,
    h_out_quant: tuple[float, int, int] | None = None,
) -> torch.Tensor:
    """Differentiable Butterfly-hidden-side GRU scan.

    Mirror of ``gru_scan_monarch`` but the matmul per step goes through
    ``torch_structured.Butterfly``'s CUDA op. No multi-step Triton fusion.

    Args:
        gi:       [T, B, 3H] — pre-batched input projection (with bi).
        h0:       [B, H]
        butterfly_modules: list of three ``torch_structured.Butterfly``
            modules, one per gate. Get from ``extract_butterfly_factors``.
        bh_cat:   [3*H]
        h_in_quant / h_out_quant: optional ``(scale, qmin, qmax)`` —
            same semantics as ``gru_scan_monarch``.

    Returns:
        out: [T, B, H]
    """
    T, B, three_H = gi.shape
    H = three_H // 3
    Wr_m, Wz_m, Wn_m = butterfly_modules
    bh = bh_cat.view(3, H)

    out = []
    h = h0
    for t in range(T):
        hq = _maybe_fake_quant(h, h_in_quant)
        gh_r = Wr_m(hq) + bh[0]
        gh_z = Wz_m(hq) + bh[1]
        gh_n = Wn_m(hq) + bh[2]

        gi_r = gi[t, :, 0:H]
        gi_z = gi[t, :, H:2 * H]
        gi_n = gi[t, :, 2 * H:3 * H]

        r = torch.sigmoid(gi_r + gh_r)
        z = torch.sigmoid(gi_z + gh_z)
        n = torch.tanh(gi_n + r * gh_n)
        h_new = (1.0 - z) * n + z * h
        h_new = _maybe_fake_quant(h_new, h_out_quant)
        out.append(h_new)
        h = h_new

    return torch.stack(out, dim=0)


# ---------------------------------------------------------------------------
# Multi-step persistent Triton butterfly kernel
# ---------------------------------------------------------------------------


@triton.jit
def gru_scan_butterfly_fwd_kernel(
    gi_ptr,             # [T, B, 3H], fp32
    h0_ptr,             # [B, H], fp32
    twiddle_ptr,        # [3, NBLOCKS, log_n, n//2, 2, 2], fp32 — one per gate
    bh_ptr,             # [3H], fp32
    out_ptr,            # [T, B, H], fp32
    # Per-program scratch: 3 gate buffers + 1 hq buffer, each [BLOCK_B, H].
    # Accessed only by this program (no cross-CTA), so disjoint across pid_b.
    scratch_ptr,        # [num_pid_b, 4, BLOCK_B, H], fp32
    T,
    B,
    sg_t, sg_b,
    sh0_b,
    st_g, st_block, st_s, st_p, st_m_new, st_m_old,
    so_t, so_b,
    sscr_pid, sscr_buf, sscr_b,
    h_in_scale,
    h_in_qmin,
    h_in_qmax,
    h_out_scale,
    h_out_qmin,
    h_out_qmax,
    H: tl.constexpr,
    H_PAD: tl.constexpr,
    LOG_H_PAD: tl.constexpr,
    NBLOCKS: tl.constexpr,
    BLOCK_B: tl.constexpr,
    QUANT_H_IN: tl.constexpr,
    QUANT_H_OUT: tl.constexpr,
):
    """Persistent forward over the butterfly recurrence.

    Each program holds [BLOCK_B, H_PAD] state across T timesteps (H_PAD
    is next_pow2(H), Triton needs pow-2 tl.arange). Within a timestep,
    three gate-specific butterfly multiplies run in sequence on a
    per-program scratch buffer, then the gate compose / recurrence
    update produces h_t. No cross-CTA sync — butterfly is per-row
    independent so each batch tile can run in isolation.

    Dense tensors (h0, gi, out, bh, dh0, dgi, dout, dbh) are sized at H,
    not H_PAD; mask_h = offs_h < H excludes the padded tail from those
    loads/stores. Scratch (program-local) and twiddle are at H_PAD,
    which torch_structured.Butterfly already produces for non-pow2 H.
    """
    pid_b = tl.program_id(0)
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = offs_b < B
    offs_h = tl.arange(0, H_PAD)
    mask_h = offs_h < H
    mask_bh = mask_b[:, None] & mask_h[None, :]
    # Local batch index for indexing into this program's scratch slab.
    # `offs_b` is the absolute batch index (into h0/out/gi); scratch is
    # laid out per-program with shape [BLOCK_B, H_PAD] so use local_b.
    local_b = tl.arange(0, BLOCK_B)

    # Pre-load bias per gate; padded lanes = 0 so they contribute 0
    # additively in the recurrence.
    bhr = tl.load(bh_ptr + 0 * H + offs_h, mask=mask_h, other=0.0)
    bhz = tl.load(bh_ptr + 1 * H + offs_h, mask=mask_h, other=0.0)
    bhn = tl.load(bh_ptr + 2 * H + offs_h, mask=mask_h, other=0.0)

    # This program's scratch slab.
    scr_base = scratch_ptr + pid_b * sscr_pid

    # Pointer to the current "h_in" — starts at h0, then walks out[t-1].
    h_in_ptr = h0_ptr
    sh_b = sh0_b

    for t in range(T):
        # Stage 0 of butterfly: copy h_in into all three gate scratch
        # buffers (we'll mutate them in place across stages). Apply
        # quant_h_in to the matmul-side h; the direct contribution
        # `(1-z)*n + z*h_self` below uses the raw h_self. mask_bh
        # zeros the padded slots so the butterfly stages see a clean
        # zero-extended input.
        h_self = tl.load(
            h_in_ptr + offs_b[:, None] * sh_b + offs_h[None, :],
            mask=mask_bh, other=0.0,
        )
        if QUANT_H_IN:
            q = tl.extra.cuda.libdevice.rint(h_self / h_in_scale)
            q = tl.minimum(tl.maximum(q, h_in_qmin), h_in_qmax)
            h_self_q = q * h_in_scale
        else:
            h_self_q = h_self
        for g in range(3):
            scr_g = scr_base + g * sscr_buf
            tl.store(
                scr_g + local_b[:, None] * sscr_b + offs_h[None, :],
                h_self_q,
                mask=mask_b[:, None],
            )

        # Run NBLOCKS butterfly chains, each with LOG_H_PAD stages, on
        # each gate's scratch buffer. Stride direction alternates by
        # block index (matches torch_structured with increasing_stride=True
        # at block 0): block 0 has strides 1,2,...,H_PAD/2 (increasing);
        # block 1 has strides H_PAD/2,...,1 (decreasing); etc.
        for blk in range(NBLOCKS):
            for s in range(LOG_H_PAD):
                # log_stride resolves at JIT time: blk and s are both
                # Python-loop variables over constexpr ranges.
                if (blk % 2) == 0:
                    log_stride = s
                else:
                    log_stride = LOG_H_PAD - 1 - s
                stride_s = 1 << log_stride
                partner = offs_h ^ stride_s
                member = (offs_h >> log_stride) & 1
                pair_idx = (offs_h >> (log_stride + 1)) * stride_s + (offs_h & (stride_s - 1))

                for g in range(3):
                    scr_g = scr_base + g * sscr_buf
                    a = tl.load(
                        scr_g + local_b[:, None] * sscr_b + offs_h[None, :],
                        mask=mask_b[:, None], other=0.0,
                    )
                    b = tl.load(
                        scr_g + local_b[:, None] * sscr_b + partner[None, :],
                        mask=mask_b[:, None], other=0.0,
                    )
                    # Twiddle indexing: t[g, blk, s, pair_idx, m_new, m_old].
                    # Note: the second dim is the iteration index `s`, not
                    # log_stride — torch_structured stores twiddles in
                    # iteration order regardless of stride direction.
                    t_offset = (
                        g * st_g
                        + blk * st_block
                        + s * st_s
                        + pair_idx * st_p
                        + member * st_m_new
                    )
                    t_ss = tl.load(twiddle_ptr + t_offset + member * st_m_old)
                    t_sp = tl.load(twiddle_ptr + t_offset + (1 - member) * st_m_old)
                    new_val = t_ss[None, :] * a + t_sp[None, :] * b
                    tl.store(
                        scr_g + local_b[:, None] * sscr_b + offs_h[None, :],
                        new_val,
                        mask=mask_b[:, None],
                    )

        # After log_H stages, scratch[g] = butterfly_g(h). Now run the
        # gate compose + recurrence to produce h_new.
        scr_r = scr_base + 0 * sscr_buf
        scr_z = scr_base + 1 * sscr_buf
        scr_n = scr_base + 2 * sscr_buf
        gh_r = tl.load(
            scr_r + local_b[:, None] * sscr_b + offs_h[None, :],
            mask=mask_b[:, None], other=0.0,
        ) + bhr[None, :]
        gh_z = tl.load(
            scr_z + local_b[:, None] * sscr_b + offs_h[None, :],
            mask=mask_b[:, None], other=0.0,
        ) + bhz[None, :]
        gh_n = tl.load(
            scr_n + local_b[:, None] * sscr_b + offs_h[None, :],
            mask=mask_b[:, None], other=0.0,
        ) + bhn[None, :]

        gi_base = (
            gi_ptr + t * sg_t + offs_b[:, None] * sg_b + offs_h[None, :]
        )
        gir = tl.load(gi_base + 0 * H, mask=mask_bh, other=0.0)
        giz = tl.load(gi_base + 1 * H, mask=mask_bh, other=0.0)
        gin = tl.load(gi_base + 2 * H, mask=mask_bh, other=0.0)

        r = tl.sigmoid(gir + gh_r)
        z = tl.sigmoid(giz + gh_z)
        n = tl.extra.cuda.libdevice.tanh(gin + r * gh_n)
        h_new = (1.0 - z) * n + z * h_self

        if QUANT_H_OUT:
            q = tl.extra.cuda.libdevice.rint(h_new / h_out_scale)
            q = tl.minimum(tl.maximum(q, h_out_qmin), h_out_qmax)
            h_new = q * h_out_scale

        out_ptrs = (
            out_ptr + t * so_t + offs_b[:, None] * so_b + offs_h[None, :]
        )
        tl.store(out_ptrs, h_new, mask=mask_bh)

        # Next step reads from out[t] for h_in.
        h_in_ptr = out_ptr + t * so_t
        sh_b = so_b


def gru_scan_butterfly_forward_triton(
    gi: torch.Tensor,
    h0: torch.Tensor,
    twiddles: torch.Tensor,
    bh_cat: torch.Tensor,
    *,
    block_b: int = 8,
    num_warps: int = 4,
    num_stages: int = 1,
    h_in_quant: tuple[float, int, int] | None = None,
    h_out_quant: tuple[float, int, int] | None = None,
) -> torch.Tensor:
    """Multi-step persistent Triton butterfly forward.

    Args:
        gi:       [T, B, 3H] — pre-batched input projection (with bias).
        h0:       [B, H]
        twiddles: [3, log_H, H//2, 2, 2] — three gates' butterfly twiddles
            stacked along dim 0. Each shaped like a single Butterfly's
            twiddle (with nstacks=1, nblocks=1 squeezed out).
        bh_cat:   [3H]
    Returns:
        out: [T, B, H]
    """
    assert gi.is_cuda
    T, B, three_H = gi.shape
    H = three_H // 3
    assert h0.shape == (B, H)
    # H itself may be non-pow2; the butterfly multiply runs on H_PAD =
    # next_pow_2(H). torch_structured.Butterfly already sizes its
    # twiddles at the padded shape internally. NBLOCKS is the number of
    # stacked butterfly chains (torch_structured's `nblocks` argument).
    n_gates, nblocks, log_n_t, n_div_2_t, two1, two2 = twiddles.shape
    assert n_gates == 3 and two1 == 2 and two2 == 2
    H_PAD = n_div_2_t * 2
    log_H_pad = log_n_t
    assert H_PAD >= H and (H_PAD & (H_PAD - 1)) == 0
    assert 1 << log_H_pad == H_PAD

    gi = gi.contiguous()
    h0 = h0.contiguous()
    twiddles = twiddles.contiguous()
    bh_cat = bh_cat.contiguous()

    out = torch.empty((T, B, H), device=gi.device, dtype=gi.dtype)

    n_pid_b = triton.cdiv(B, block_b)
    scratch = torch.empty(
        (n_pid_b, 4, block_b, H_PAD), device=gi.device, dtype=gi.dtype,
    )

    in_s, in_qmin, in_qmax = h_in_quant or (1.0, -2**31, 2**31 - 1)
    out_s, out_qmin, out_qmax = h_out_quant or (1.0, -2**31, 2**31 - 1)

    grid = (n_pid_b,)
    gru_scan_butterfly_fwd_kernel[grid](
        gi, h0, twiddles, bh_cat, out, scratch,
        T, B,
        gi.stride(0), gi.stride(1),
        h0.stride(0),
        twiddles.stride(0), twiddles.stride(1), twiddles.stride(2),
        twiddles.stride(3), twiddles.stride(4), twiddles.stride(5),
        out.stride(0), out.stride(1),
        scratch.stride(0), scratch.stride(1), scratch.stride(2),
        in_s, in_qmin, in_qmax,
        out_s, out_qmin, out_qmax,
        H=H,
        H_PAD=H_PAD,
        LOG_H_PAD=log_H_pad,
        NBLOCKS=nblocks,
        BLOCK_B=block_b,
        QUANT_H_IN=h_in_quant is not None,
        QUANT_H_OUT=h_out_quant is not None,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


@triton.jit
def gru_scan_butterfly_bwd_kernel(
    # forward inputs (read-only)
    gi_ptr,
    h0_ptr,
    twiddle_ptr,
    bh_ptr,
    out_ptr,
    # upstream gradient
    dout_ptr,
    # outputs
    dgi_ptr,
    dh0_ptr,
    dtwiddle_partial_ptr,    # [num_pid_b, 3, NBLOCKS, log_H, H//2, 2, 2]
    dbh_partial_ptr,         # [num_pid_b, 3H]
    # per-program state buffers (NBLOCKS*log_H+1 snapshots)
    state_ptr,               # [num_pid_b, 3, NBLOCKS*log_H+1, BLOCK_B, H]
    dh_acc_ptr,              # [B, H]
    T,
    B,
    sg_t, sg_b,
    sh0_b,
    st_g, st_block, st_s, st_p, st_m_new, st_m_old,
    so_t, so_b,
    sdo_t, sdo_b,
    sdgi_t, sdgi_b,
    sdh0_b,
    sdtp_pid, sdtp_g, sdtp_block, sdtp_s, sdtp_p, sdtp_m_new, sdtp_m_old,
    sdbp_pid,
    sst_pid, sst_g, sst_l, sst_b,
    sdh_b,
    h_in_scale,
    h_in_qmin,
    h_in_qmax,
    h_out_scale,
    h_out_qmin,
    h_out_qmax,
    H: tl.constexpr,
    H_PAD: tl.constexpr,
    LOG_H_PAD: tl.constexpr,
    NBLOCKS: tl.constexpr,
    BLOCK_B: tl.constexpr,
    QUANT_H_IN: tl.constexpr,
    QUANT_H_OUT: tl.constexpr,
):
    """Persistent backward over the butterfly recurrence.

    Walks t from T-1 down to 0. Per timestep:
    - Recomputes butterfly forward for all 3 gates, saving the
      per-stage states into the scratch buffer (LOG_H_PAD+1 states per gate).
    - Backprops through the gate compose to get dgh_r, dgh_z, dgh_n.
    - Backprops through each butterfly via reverse-stage walk, using
      saved states. Accumulates dtwiddle_partial (per pid_b, summed
      across pid_b in Python) and the per-position dh_prev_via_W
      contributions into dh_acc.
    - dh_prev_direct (from (1-z)*n + z*h_prev) added to dh_acc.

    No cross-CTA sync needed (butterfly is per-row independent), so
    dh_acc is a single buffer per program with disjoint writes.

    Padded-H handling: dense tensors (h0, gi, out, bh, dh0, dgi, dout,
    dh_acc, dbh) are sized at H; mask_h = offs_h < H excludes padded
    slots from those loads/stores. Scratch (state buffer), twiddle, and
    dtwiddle_partial are at H_PAD. The padded-H butterfly is
    mathematically equivalent to multiplying x by the top-left H x H
    submatrix of the H_PAD butterfly; the H_PAD twiddle parameters all
    receive non-zero gradient and are co-trained.
    """
    pid_b = tl.program_id(0)
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = offs_b < B
    offs_h = tl.arange(0, H_PAD)
    mask_h = offs_h < H
    mask_bh = mask_b[:, None] & mask_h[None, :]
    # Local batch index for indexing into this program's per-program slabs
    # (state). dh_acc is global [B, H] and uses absolute offs_b.
    local_b = tl.arange(0, BLOCK_B)

    bhr = tl.load(bh_ptr + 0 * H + offs_h, mask=mask_h, other=0.0)  # noqa: F841
    bhz = tl.load(bh_ptr + 1 * H + offs_h, mask=mask_h, other=0.0)  # noqa: F841
    bhn = tl.load(bh_ptr + 2 * H + offs_h, mask=mask_h, other=0.0)  # noqa: F841

    state_base = state_ptr + pid_b * sst_pid

    # Initialize dh_acc[:, :] to 0 for this program's batch tile (real H).
    tl.store(
        dh_acc_ptr + offs_b[:, None] * sdh_b + offs_h[None, :],
        tl.zeros((BLOCK_B, H_PAD), dtype=tl.float32),
        mask=mask_bh,
    )

    for t_rev in range(T):
        t = T - 1 - t_rev

        if t == 0:
            h_prev_ptr = h0_ptr
            sh_prev_b = sh0_b
        else:
            h_prev_ptr = out_ptr + (t - 1) * so_t
            sh_prev_b = so_b

        # Load h_prev once. Padded slots zeroed so the butterfly
        # recompute sees the same zero-extended input as the forward
        # kernel did.
        h_prev = tl.load(
            h_prev_ptr + offs_b[:, None] * sh_prev_b + offs_h[None, :],
            mask=mask_bh, other=0.0,
        )

        # ---- Recompute butterfly forward, saving per-stage states ----
        # state[g, 0] = quant_h_in(h_prev). Matches the forward kernel:
        # the matmul side starts from the QUANTIZED h_prev. The raw
        # h_prev is used below for the (1-z)*n + z*h_prev recurrence.
        if QUANT_H_IN:
            q = tl.extra.cuda.libdevice.rint(h_prev / h_in_scale)
            q = tl.minimum(tl.maximum(q, h_in_qmin), h_in_qmax)
            h_prev_q = q * h_in_scale
        else:
            h_prev_q = h_prev
        for g in range(3):
            base_g = state_base + g * sst_g + 0 * sst_l
            tl.store(
                base_g + local_b[:, None] * sst_b + offs_h[None, :],
                h_prev_q,
                mask=mask_b[:, None],
            )
        # Run NBLOCKS butterfly chains, each LOG_H_PAD stages, saving
        # state after every stage. Stage index in state buffer is
        # `blk * LOG_H_PAD + s + 1` (state[0] holds the input).
        for blk in range(NBLOCKS):
            for s in range(LOG_H_PAD):
                if (blk % 2) == 0:
                    log_stride = s
                else:
                    log_stride = LOG_H_PAD - 1 - s
                stride_s = 1 << log_stride
                partner = offs_h ^ stride_s
                member = (offs_h >> log_stride) & 1
                pair_idx = (offs_h >> (log_stride + 1)) * stride_s + (offs_h & (stride_s - 1))
                state_l_in = blk * LOG_H_PAD + s
                state_l_out = state_l_in + 1
                for g in range(3):
                    in_base = state_base + g * sst_g + state_l_in * sst_l
                    out_base = state_base + g * sst_g + state_l_out * sst_l
                    a = tl.load(
                        in_base + local_b[:, None] * sst_b + offs_h[None, :],
                        mask=mask_b[:, None], other=0.0,
                    )
                    b = tl.load(
                        in_base + local_b[:, None] * sst_b + partner[None, :],
                        mask=mask_b[:, None], other=0.0,
                    )
                    t_offset = (
                        g * st_g
                        + blk * st_block
                        + s * st_s
                        + pair_idx * st_p
                        + member * st_m_new
                    )
                    t_ss = tl.load(twiddle_ptr + t_offset + member * st_m_old)
                    t_sp = tl.load(twiddle_ptr + t_offset + (1 - member) * st_m_old)
                    new_val = t_ss[None, :] * a + t_sp[None, :] * b
                    tl.store(
                        out_base + local_b[:, None] * sst_b + offs_h[None, :],
                        new_val,
                        mask=mask_b[:, None],
                    )

        # state[g, NBLOCKS*LOG_H_PAD] is the final butterfly output. Recompute gh, gates.
        final_l: tl.constexpr = NBLOCKS * LOG_H_PAD
        end_r = state_base + 0 * sst_g + final_l * sst_l
        end_z = state_base + 1 * sst_g + final_l * sst_l
        end_n = state_base + 2 * sst_g + final_l * sst_l
        gh_r = tl.load(
            end_r + local_b[:, None] * sst_b + offs_h[None, :],
            mask=mask_b[:, None], other=0.0,
        ) + bhr[None, :]
        gh_z = tl.load(
            end_z + local_b[:, None] * sst_b + offs_h[None, :],
            mask=mask_b[:, None], other=0.0,
        ) + bhz[None, :]
        gh_n = tl.load(
            end_n + local_b[:, None] * sst_b + offs_h[None, :],
            mask=mask_b[:, None], other=0.0,
        ) + bhn[None, :]

        gi_base = (
            gi_ptr + t * sg_t + offs_b[:, None] * sg_b + offs_h[None, :]
        )
        gir = tl.load(gi_base + 0 * H, mask=mask_bh, other=0.0)
        giz = tl.load(gi_base + 1 * H, mask=mask_bh, other=0.0)
        gin = tl.load(gi_base + 2 * H, mask=mask_bh, other=0.0)

        r = tl.sigmoid(gir + gh_r)
        z = tl.sigmoid(giz + gh_z)
        n = tl.extra.cuda.libdevice.tanh(gin + r * gh_n)

        # ---- Read incoming dh_acc and dout[t] ----
        dh_acc_oh = tl.load(
            dh_acc_ptr + offs_b[:, None] * sdh_b + offs_h[None, :],
            mask=mask_bh, other=0.0,
        )
        dout_oh = tl.load(
            dout_ptr + t * sdo_t + offs_b[:, None] * sdo_b + offs_h[None, :],
            mask=mask_bh, other=0.0,
        )
        dh_t = dout_oh + dh_acc_oh

        # STE backward of quant_h_out: incoming dh_t is grad on the
        # quantized h_t (= post-quant_h_out). Multiply by clip mask of
        # h_t_raw to get grad on h_t_raw before walking the recurrence.
        if QUANT_H_OUT:
            h_t_raw = (1.0 - z) * n + z * h_prev
            q_unclamped = tl.extra.cuda.libdevice.rint(h_t_raw / h_out_scale)
            mask_out = (q_unclamped >= h_out_qmin) & (q_unclamped <= h_out_qmax)
            dh_t = tl.where(mask_out, dh_t, 0.0)

        dn = dh_t * (1.0 - z)
        dz = dh_t * (h_prev - n)
        dh_prev_direct = dh_t * z

        dgn_pre = dn * (1.0 - n * n)
        dgi_n = dgn_pre
        dr = dgn_pre * gh_n
        dgh_n = dgn_pre * r

        dgz_pre = dz * z * (1.0 - z)
        dgi_z = dgz_pre
        dgh_z = dgz_pre

        dgr_pre = dr * r * (1.0 - r)
        dgi_r = dgr_pre
        dgh_r = dgr_pre

        # Store dgi[t] slices.
        dgi_base = (
            dgi_ptr + t * sdgi_t + offs_b[:, None] * sdgi_b + offs_h[None, :]
        )
        tl.store(dgi_base + 0 * H, dgi_r, mask=mask_bh)
        tl.store(dgi_base + 1 * H, dgi_z, mask=mask_bh)
        tl.store(dgi_base + 2 * H, dgi_n, mask=mask_bh)

        # Accumulate dbh_partial (sum across batch). dbh is sized [3*H]
        # so the padded oh tail must be masked or it'd land in the
        # next gate slot.
        dbh_base = dbh_partial_ptr + pid_b * sdbp_pid + offs_h
        tl.store(
            dbh_base + 0 * H,
            tl.load(dbh_base + 0 * H, mask=mask_h, other=0.0) + tl.sum(dgh_r, axis=0),
            mask=mask_h,
        )
        tl.store(
            dbh_base + 1 * H,
            tl.load(dbh_base + 1 * H, mask=mask_h, other=0.0) + tl.sum(dgh_z, axis=0),
            mask=mask_h,
        )
        tl.store(
            dbh_base + 2 * H,
            tl.load(dbh_base + 2 * H, mask=mask_h, other=0.0) + tl.sum(dgh_n, axis=0),
            mask=mask_h,
        )

        # ---- Backward through butterfly stages, per gate ----
        # Initialize d_state per gate to dgh_g at state[NBLOCKS*LOG_H_PAD]
        # (the final forward butterfly output). We reuse the state
        # buffer for d_state during backward.
        d_dst_r = state_base + 0 * sst_g + final_l * sst_l
        d_dst_z = state_base + 1 * sst_g + final_l * sst_l
        d_dst_n = state_base + 2 * sst_g + final_l * sst_l
        tl.store(
            d_dst_r + local_b[:, None] * sst_b + offs_h[None, :],
            dgh_r, mask=mask_b[:, None],
        )
        tl.store(
            d_dst_z + local_b[:, None] * sst_b + offs_h[None, :],
            dgh_z, mask=mask_b[:, None],
        )
        tl.store(
            d_dst_n + local_b[:, None] * sst_b + offs_h[None, :],
            dgh_n, mask=mask_b[:, None],
        )

        # Walk blocks blk from NBLOCKS-1 down to 0; within each block,
        # stages s from LOG_H_PAD-1 down to 0. Stride direction matches
        # the forward (even blocks increasing, odd blocks decreasing).
        for blk_rev in range(NBLOCKS):
            blk = NBLOCKS - 1 - blk_rev
            for s_rev in range(LOG_H_PAD):
                s = LOG_H_PAD - 1 - s_rev
                if (blk % 2) == 0:
                    log_stride = s
                else:
                    log_stride = LOG_H_PAD - 1 - s
                stride_s = 1 << log_stride
                partner = offs_h ^ stride_s
                member = (offs_h >> log_stride) & 1
                pair_idx = (offs_h >> (log_stride + 1)) * stride_s + (offs_h & (stride_s - 1))
                state_l_in = blk * LOG_H_PAD + s        # state[g, this_l] holds old
                state_l_out = state_l_in + 1            # state[g, this_l+1] holds new/d_new

                for g in range(3):
                    d_new_base = state_base + g * sst_g + state_l_out * sst_l
                    old_base = state_base + g * sst_g + state_l_in * sst_l

                    d_self = tl.load(
                        d_new_base + local_b[:, None] * sst_b + offs_h[None, :],
                        mask=mask_b[:, None], other=0.0,
                    )
                    d_partner = tl.load(
                        d_new_base + local_b[:, None] * sst_b + partner[None, :],
                        mask=mask_b[:, None], other=0.0,
                    )
                    old_self = tl.load(
                        old_base + local_b[:, None] * sst_b + offs_h[None, :],
                        mask=mask_b[:, None], other=0.0,
                    )
                    old_partner = tl.load(
                        old_base + local_b[:, None] * sst_b + partner[None, :],
                        mask=mask_b[:, None], other=0.0,
                    )

                    # Twiddle entries — indexed by (g, blk, s, pair_idx).
                    t_offset = (
                        g * st_g + blk * st_block + s * st_s + pair_idx * st_p
                    )
                    # For position i with member=member(i):
                    #   d_old[i] = d_new[i] * t[member, member] + d_new[partner] * t[1-member, member]
                    t_self = tl.load(
                        twiddle_ptr + t_offset + member * st_m_new + member * st_m_old
                    )
                    t_partner = tl.load(
                        twiddle_ptr + t_offset + (1 - member) * st_m_new + member * st_m_old
                    )
                    d_old = t_self[None, :] * d_self + t_partner[None, :] * d_partner

                    tl.store(
                        old_base + local_b[:, None] * sst_b + offs_h[None, :],
                        d_old,
                        mask=mask_b[:, None],
                    )

                    # dt accumulation: per pair, dt[m_new, m_old] += d_new[m_new] * old[m_old]
                    # Each output position contributes to one (m_new, m_old) entry.
                    # When member(i) = 0: contributes to dt[0, 0] via (d_self, old_self)
                    #                    and dt[0, 1] via (d_self, old_partner)
                    # When member(i) = 1: contributes to dt[1, 0] via (d_self, old_partner)
                    #                    and dt[1, 1] via (d_self, old_self)
                    #
                    # So d_self * old_self (member=0 case) -> dt[0, 0] at this pair.
                    # d_self * old_partner (member=0 case) -> dt[0, 1] at this pair.
                    # d_self * old_partner (member=1 case) -> dt[1, 0] at this pair.
                    # d_self * old_self (member=1 case) -> dt[1, 1] at this pair.
                    #
                    # Each pair has TWO positions (member=0 and member=1), and
                    # they contribute to different rows of dt. They share the
                    # same pair_idx.
                    #
                    # Per-batch contribution: scalar = d_self * old_self. Sum
                    # over batch to get the per-pair contribution to dt.

                    contrib_dd = tl.sum(d_self * old_self, axis=0)        # [H]
                    contrib_dp = tl.sum(d_self * old_partner, axis=0)     # [H]

                    # Now route contributions to dt entries.
                    # We accumulate by position into dt_partial[g, s, pair_idx, m_new, m_old]
                    # The mapping by member:
                    #   member=0 positions:
                    #     contrib_dd -> dt[g, s, pair_idx, 0, 0]
                    #     contrib_dp -> dt[g, s, pair_idx, 0, 1]
                    #   member=1 positions:
                    #     contrib_dp -> dt[g, s, pair_idx, 1, 0]
                    #     contrib_dd -> dt[g, s, pair_idx, 1, 1]
                    # Each position contributes ONCE to ONE (m_new, m_old) entry.
                    # Use atomic_add for safety (multiple positions in same batch
                    # tile may share pair_idx? No — within one stage, each pair
                    # has exactly two positions, and they have different m_new.
                    # So writes are to distinct (pair_idx, m_new) cells). But the
                    # m_old varies per position too — m_old = member of THIS
                    # position for the dt[m_new, m_old] entry... wait let me
                    # re-examine.

                    # Actually re-derive cleanly:
                    # forward stage: new = t @ old (2x2 matmul per pair)
                    #   new[m_new] = sum_{m_old} t[m_new, m_old] * old[m_old]
                    # Per pair, two positions: m_new in {0, 1}, two equations.
                    #
                    # Backward:
                    #   d_old[m_old] = sum_{m_new} t[m_new, m_old] * d_new[m_new]
                    #   d_t[m_new, m_old] = d_new[m_new] * old[m_old]
                    #
                    # Per output position i with member m_new(i):
                    #   This position has one "new" value (new[m_new(i)]).
                    #   Its d_new[m_new(i)] participates in two d_t entries:
                    #     d_t[m_new(i), 0] += d_self_i * old[member=0]
                    #     d_t[m_new(i), 1] += d_self_i * old[member=1]
                    #
                    # Where:
                    #   - old[member=0] at this pair is old_partner if i is member=1,
                    #     or old_self if i is member=0.
                    #   - old[member=1] at this pair is old_self if i is member=1,
                    #     or old_partner if i is member=0.
                    #
                    # So:
                    #   d_t[m_new(i), 0] += d_self_i * (old_self_i if member(i)=0 else old_partner_i)
                    #   d_t[m_new(i), 1] += d_self_i * (old_partner_i if member(i)=0 else old_self_i)
                    #
                    # Equivalently, defining
                    #   even_old[i] = old_self if member(i)=0 else old_partner
                    #   odd_old[i]  = old_partner if member(i)=0 else old_self
                    # we get:
                    #   d_t[m_new(i), 0] += d_self_i * even_old[i]
                    #   d_t[m_new(i), 1] += d_self_i * odd_old[i]
                    #
                    # Each PAIR has two contributing positions (one with
                    # member=0, one with member=1). They write to different
                    # rows of d_t (different m_new). So writes for the SAME
                    # pair_idx but DIFFERENT m_new come from different positions.
                    # Within one offs_h vector, exactly two positions share
                    # each pair_idx. Their m_new values differ (one is 0, one
                    # is 1). So if we atomic-add, each (pair_idx, m_new) cell
                    # gets exactly one position's contribution. No within-program
                    # contention.

                    # Compute even_old and odd_old based on member(i).
                    # For i with member=0: even_old = old_self, odd_old = old_partner.
                    # For i with member=1: even_old = old_partner, odd_old = old_self.
                    is_member_0 = (member == 0)
                    even_old = tl.where(is_member_0, old_self, old_partner)
                    odd_old = tl.where(is_member_0, old_partner, old_self)

                    contrib_to_m0 = tl.sum(d_self * even_old, axis=0)  # [H]
                    contrib_to_m1 = tl.sum(d_self * odd_old, axis=0)   # [H]

                    # Each position i writes to:
                    #   d_t[g, blk, s, pair_idx(i), m_new=member(i), m_old=0] += contrib_to_m0[i]
                    #   d_t[g, blk, s, pair_idx(i), m_new=member(i), m_old=1] += contrib_to_m1[i]
                    dt_base = (
                        dtwiddle_partial_ptr + pid_b * sdtp_pid + g * sdtp_g
                        + blk * sdtp_block + s * sdtp_s + pair_idx * sdtp_p
                        + member * sdtp_m_new
                    )
                    tl.atomic_add(dt_base + 0 * sdtp_m_old, contrib_to_m0)
                    tl.atomic_add(dt_base + 1 * sdtp_m_old, contrib_to_m1)

        # After all stages backward, state[g, 0] holds d_h_prev_q for
        # gate g (gradient on the QUANTIZED h_prev — the matmul-side
        # input). Sum across gates, then apply STE backward of
        # quant_h_in (zero where h_prev was clipped).
        dh_via_r = tl.load(
            state_base + 0 * sst_g + 0 * sst_l + local_b[:, None] * sst_b + offs_h[None, :],
            mask=mask_b[:, None], other=0.0,
        )
        dh_via_z = tl.load(
            state_base + 1 * sst_g + 0 * sst_l + local_b[:, None] * sst_b + offs_h[None, :],
            mask=mask_b[:, None], other=0.0,
        )
        dh_via_n = tl.load(
            state_base + 2 * sst_g + 0 * sst_l + local_b[:, None] * sst_b + offs_h[None, :],
            mask=mask_b[:, None], other=0.0,
        )
        dh_via = dh_via_r + dh_via_z + dh_via_n
        if QUANT_H_IN:
            q_in_unclamped = tl.extra.cuda.libdevice.rint(h_prev / h_in_scale)
            mask_in = (q_in_unclamped >= h_in_qmin) & (q_in_unclamped <= h_in_qmax)
            dh_via = tl.where(mask_in, dh_via, 0.0)
        dh_acc_new = dh_prev_direct + dh_via

        # Padded slots of dh_via carry "grad w.r.t. the zero-padded
        # input" — not a real parameter, discarded. mask_bh prevents
        # those values from writing past the [B, H] row boundary in
        # dh_acc.
        tl.store(
            dh_acc_ptr + offs_b[:, None] * sdh_b + offs_h[None, :],
            dh_acc_new,
            mask=mask_bh,
        )

    # After loop, dh_acc holds dh0 for this batch tile.
    dh_final = tl.load(
        dh_acc_ptr + offs_b[:, None] * sdh_b + offs_h[None, :],
        mask=mask_bh, other=0.0,
    )
    tl.store(
        dh0_ptr + offs_b[:, None] * sdh0_b + offs_h[None, :],
        dh_final,
        mask=mask_bh,
    )


def gru_scan_butterfly_backward_triton(
    gi: torch.Tensor,
    h0: torch.Tensor,
    twiddles: torch.Tensor,
    bh_cat: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    *,
    block_b: int = 8,
    num_warps: int = 4,
    num_stages: int = 1,
    h_in_quant: tuple[float, int, int] | None = None,
    h_out_quant: tuple[float, int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Multi-step persistent Triton butterfly backward.

    Returns (dgi, dh0, dtwiddles, dbh).
    """
    T, B, three_H = gi.shape
    H = three_H // 3
    # H may be non-pow2; the kernel operates on H_PAD = next_pow_2(H)
    # internally. Twiddle is already at H_PAD via torch_structured.
    # NBLOCKS is the stacked-butterfly count from torch_structured.
    n_gates, nblocks, log_H_pad, n_div_2_t, two1, two2 = twiddles.shape
    assert n_gates == 3 and two1 == 2 and two2 == 2
    H_PAD = n_div_2_t * 2
    assert H_PAD >= H and (H_PAD & (H_PAD - 1)) == 0
    assert 1 << log_H_pad == H_PAD

    gi = gi.contiguous()
    h0 = h0.contiguous()
    twiddles = twiddles.contiguous()
    bh_cat = bh_cat.contiguous()
    out = out.contiguous()
    dout = dout.contiguous()

    dgi = torch.zeros_like(gi)
    dh0 = torch.zeros_like(h0)

    n_pid_b = triton.cdiv(B, block_b)
    dtwiddle_partial = torch.zeros(
        (n_pid_b, 3, nblocks, log_H_pad, H_PAD // 2, 2, 2),
        device=gi.device, dtype=gi.dtype,
    )
    dbh_partial = torch.zeros(
        (n_pid_b, 3 * H), device=gi.device, dtype=gi.dtype,
    )
    # Per-program scratch holds the per-stage state for backward, one
    # snapshot per (block, stage) plus the input.
    # Shape: [num_pid_b, 3 (gates), NBLOCKS*LOG_H_PAD + 1, BLOCK_B, H_PAD].
    state = torch.empty(
        (n_pid_b, 3, nblocks * log_H_pad + 1, block_b, H_PAD),
        device=gi.device, dtype=gi.dtype,
    )
    dh_acc = torch.empty((B, H), device=gi.device, dtype=gi.dtype)

    in_s, in_qmin, in_qmax = h_in_quant or (1.0, -2**31, 2**31 - 1)
    out_s, out_qmin, out_qmax = h_out_quant or (1.0, -2**31, 2**31 - 1)

    grid = (n_pid_b,)
    gru_scan_butterfly_bwd_kernel[grid](
        gi, h0, twiddles, bh_cat, out,
        dout,
        dgi, dh0,
        dtwiddle_partial, dbh_partial,
        state, dh_acc,
        T, B,
        gi.stride(0), gi.stride(1),
        h0.stride(0),
        twiddles.stride(0), twiddles.stride(1), twiddles.stride(2),
        twiddles.stride(3), twiddles.stride(4), twiddles.stride(5),
        out.stride(0), out.stride(1),
        dout.stride(0), dout.stride(1),
        dgi.stride(0), dgi.stride(1),
        dh0.stride(0),
        dtwiddle_partial.stride(0), dtwiddle_partial.stride(1),
        dtwiddle_partial.stride(2), dtwiddle_partial.stride(3),
        dtwiddle_partial.stride(4), dtwiddle_partial.stride(5),
        dtwiddle_partial.stride(6),
        dbh_partial.stride(0),
        state.stride(0), state.stride(1), state.stride(2), state.stride(3),
        dh_acc.stride(0),
        in_s, in_qmin, in_qmax,
        out_s, out_qmin, out_qmax,
        H=H, H_PAD=H_PAD, LOG_H_PAD=log_H_pad, NBLOCKS=nblocks, BLOCK_B=block_b,
        QUANT_H_IN=h_in_quant is not None,
        QUANT_H_OUT=h_out_quant is not None,
        num_warps=num_warps, num_stages=num_stages,
    )

    dtwiddles = dtwiddle_partial.sum(dim=0)
    dbh = dbh_partial.sum(dim=0)
    return dgi, dh0, dtwiddles, dbh


class GRUScanButterflyTritonFunction(torch.autograd.Function):
    """autograd wrapper around the multi-step persistent Triton butterfly
    kernels. Optional in-kernel fake-quant on hidden state via the same
    ``(scale, qmin, qmax)`` per-tensor symmetric scheme as Monarch.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        gi: torch.Tensor,
        h0: torch.Tensor,
        twiddles: torch.Tensor,
        bh_cat: torch.Tensor,
        h_in_quant: tuple[float, int, int] | None,
        h_out_quant: tuple[float, int, int] | None,
    ) -> torch.Tensor:
        out = gru_scan_butterfly_forward_triton(
            gi, h0, twiddles, bh_cat,
            h_in_quant=h_in_quant, h_out_quant=h_out_quant,
        )
        ctx.save_for_backward(gi, h0, twiddles, bh_cat, out)
        ctx.h_in_quant = h_in_quant
        ctx.h_out_quant = h_out_quant
        return out

    @staticmethod
    def backward(ctx, dout):  # type: ignore[override]
        gi, h0, twiddles, bh_cat, out = ctx.saved_tensors
        grads = gru_scan_butterfly_backward_triton(
            gi, h0, twiddles, bh_cat, out, dout,
            h_in_quant=ctx.h_in_quant, h_out_quant=ctx.h_out_quant,
        )
        return (*grads, None, None)


def gru_scan_butterfly_triton(
    gi: torch.Tensor,
    h0: torch.Tensor,
    twiddles: torch.Tensor,
    bh_cat: torch.Tensor,
    *,
    h_in_quant: tuple[float, int, int] | None = None,
    h_out_quant: tuple[float, int, int] | None = None,
) -> torch.Tensor:
    """Public API: differentiable Butterfly GRU scan via Triton kernels.

    Args:
        gi:       [T, B, 3H]
        h0:       [B, H]
        twiddles: [3, log_H, H/2, 2, 2]
        bh_cat:   [3*H]
        h_in_quant / h_out_quant: optional ``(scale, qmin, qmax)`` —
            same semantics as ``gru_scan_monarch``.
    """
    return GRUScanButterflyTritonFunction.apply(
        gi, h0, twiddles, bh_cat, h_in_quant, h_out_quant,
    )


def extract_butterfly_twiddles(
    cell: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pull the three hidden-side Butterfly twiddles into a single tensor.

    Returns:
        twiddles: [3, butterfly_nblocks, log_H, H//2, 2, 2] — gates
            stacked. The nstacks=1 axis is squeezed; butterfly_nblocks
            (number of stacked butterfly chains in torch_structured) is
            preserved. The triton kernel handles butterfly_nblocks via
            an outer JIT-unrolled loop with alternating stride direction
            between blocks (matching torch_structured.butterfly_multiply
            with increasing_stride=True).
        bh_cat:   [3H]
    """
    if cell._hidden_dense:
        raise ValueError("cell hidden side is dense; nothing to extract")
    # cell.struct_Wh_*.b.twiddle: [nstacks=1, butterfly_nblocks, log_n, n//2, 2, 2]
    Wr = cell.struct_Wh_r.b.twiddle.squeeze(0)
    Wz = cell.struct_Wh_z.b.twiddle.squeeze(0)
    Wn = cell.struct_Wh_n.b.twiddle.squeeze(0)
    twiddles = torch.stack([Wr, Wz, Wn], dim=0)  # [3, nblocks, log_n, n//2, 2, 2]
    if cell.b_hr is None:
        bh_cat = torch.zeros(
            3 * cell.hidden_size, device=twiddles.device, dtype=twiddles.dtype,
        )
    else:
        bh_cat = torch.cat([cell.b_hr, cell.b_hz, cell.b_hn])
    return twiddles, bh_cat

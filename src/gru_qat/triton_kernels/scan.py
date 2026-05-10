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

import torch
import triton
import triton.language as tl


@triton.jit
def gru_scan_fwd_kernel(
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
    # constexprs
    H: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_OH: tl.constexpr,
    BLOCK_K: tl.constexpr,
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
            n = tl.extra.libdevice.tanh(gin + r * ghn)

            # h_new = (1 - z) * n + z * h_old. Need h_old for THIS oh tile.
            h_old_ptrs = (
                h_in_ptr + offs_b[:, None] * sh_b + offs_oh[None, :]
            )
            h_old = tl.load(h_old_ptrs, mask=mask_oh2, other=0.0)
            h_new = (1.0 - z) * n + z * h_old

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


def _gru_scan_backward_pytorch(
    gi: torch.Tensor,
    h0: torch.Tensor,
    Wh_cat: torch.Tensor,
    bh_cat: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
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
        gh = h_prev @ Wh_cat.T + bh_cat
        gh_r, gh_z, gh_n = gh.chunk(3, dim=-1)
        r = torch.sigmoid(gi_r + gh_r)
        z = torch.sigmoid(gi_z + gh_z)
        n = torch.tanh(gi_n + r * gh_n)

        dh_t = dout[t] + dh_acc

        # h_t = (1 - z) * n + z * h_prev
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

        # gh = h_prev @ Wh_cat^T + bh_cat
        dh_prev_via_W = dgh @ Wh_cat
        dWh += dgh.transpose(0, 1) @ h_prev
        dbh += dgh.sum(dim=0)

        dh_acc = dh_prev_direct + dh_prev_via_W

    return dgi, dh_acc, dWh, dbh


class GRUScanFunction(torch.autograd.Function):
    """autograd wrapper: Triton forward, PyTorch reference backward.

    Once the Triton backward kernel is written, swap the backward body to
    call it; the forward and the autograd plumbing don't change.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        gi: torch.Tensor,
        h0: torch.Tensor,
        Wh_cat: torch.Tensor,
        bh_cat: torch.Tensor,
    ) -> torch.Tensor:
        out = gru_scan_forward(gi, h0, Wh_cat, bh_cat)
        ctx.save_for_backward(gi, h0, Wh_cat, bh_cat, out)
        return out

    @staticmethod
    def backward(  # type: ignore[override]
        ctx, dout: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        gi, h0, Wh_cat, bh_cat, out = ctx.saved_tensors
        return _gru_scan_backward_pytorch(gi, h0, Wh_cat, bh_cat, out, dout)


def gru_scan(
    gi: torch.Tensor,
    h0: torch.Tensor,
    Wh_cat: torch.Tensor,
    bh_cat: torch.Tensor,
) -> torch.Tensor:
    """Public API: differentiable multi-step GRU scan."""
    return GRUScanFunction.apply(gi, h0, Wh_cat, bh_cat)


def gru_scan_forward(
    gi: torch.Tensor,
    h0: torch.Tensor,
    Wh_cat: torch.Tensor,
    bh_cat: torch.Tensor,
    *,
    block_b: int = 16,
    block_oh: int = 64,
    block_k: int = 32,
    num_stages: int = 2,
    num_warps: int = 4,
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
    assert gi.is_cuda and h0.is_cuda and Wh_cat.is_cuda and bh_cat.is_cuda
    assert gi.dtype == torch.float32, "Phase 1 fp32 only"
    T, B, three_H = gi.shape
    H = three_H // 3
    assert h0.shape == (B, H)
    assert Wh_cat.shape == (3 * H, H)
    assert bh_cat.shape == (3 * H,)

    # Make sure inputs are contiguous in their last dim — strides below assume
    # last-dim stride = 1.
    gi = gi.contiguous()
    h0 = h0.contiguous()
    Wh_cat = Wh_cat.contiguous()
    bh_cat = bh_cat.contiguous()

    out = torch.empty((T, B, H), device=gi.device, dtype=gi.dtype)

    grid = (triton.cdiv(B, block_b),)
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
        H=H,
        BLOCK_B=block_b,
        BLOCK_OH=block_oh,
        BLOCK_K=block_k,
        num_stages=num_stages,
        num_warps=num_warps,
    )
    return out

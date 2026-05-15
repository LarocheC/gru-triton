"""Diagonal hidden weights for the multi-step GRU scan.

The hidden weight ``Wh`` per gate collapses from ``[H, H]`` to a length-``H``
vector — the matmul ``Wh @ h`` becomes elementwise ``w_h * h``. This is
the smallest possible structured parameterization: ``3H`` weight scalars
on the hidden side total, ``O(H)`` FLOPs per timestep per gate.

Two consequences of the diagonal shape simplify the kernel vs.
``scan_monarch``:

1. No matmul on the hidden side → no K-reduction → no cross-program
   reduction within a timestep. Each ``(pid_b, pid_h)`` program owns its
   full ``[BLOCK_B, BLOCK_H]`` slab of ``h`` for all T timesteps and never
   touches another program's data.
2. ``h`` carries across timesteps in registers; no need to reload from
   global memory between iterations. We still store the per-step output
   to ``out`` for the backward pass and for the caller.

Together this means no spin-wait barrier, no SM-count cap on the grid,
and no scratch buffers for the forward pass.
"""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn as nn
import triton
import triton.language as tl


def extract_diagonal_factors(cell: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """Pull the three hidden-side diagonal weights out of a structured cell.

    Args:
        cell: a ``GRUCellQuant`` whose ``structure_hidden`` was a
              ``StructureConfig(kind="diagonal")``. Must have
              ``struct_Wh_r/z/n`` ``_DiagonalLinear`` modules.

    Returns:
        Wh_diag: [3, H] — gates stacked in (r, z, n) order.
        bh_cat:  [3*H] — concat of (b_hr, b_hz, b_hn), zeros if no bias.
    """
    if cell._hidden_dense:
        raise ValueError("cell hidden side is dense; nothing to extract")
    # nn.Module attribute access is typed Tensor | Module by the torch
    # stubs — cast to the concrete submodule, then read the Tensor weight.
    wr = cast(torch.Tensor, cast(nn.Module, cell.struct_Wh_r).weight)
    wz = cast(torch.Tensor, cast(nn.Module, cell.struct_Wh_z).weight)
    wn = cast(torch.Tensor, cast(nn.Module, cell.struct_Wh_n).weight)
    Wh_diag = torch.stack([wr, wz, wn], dim=0)  # [3, H]
    if cell.b_hr is None:
        bh_cat = torch.zeros(3 * cell.hidden_size, device=Wh_diag.device, dtype=Wh_diag.dtype)
    else:
        bh_cat = torch.cat(
            [
                cast(torch.Tensor, cell.b_hr),
                cast(torch.Tensor, cell.b_hz),
                cast(torch.Tensor, cell.b_hn),
            ]
        )
    return Wh_diag, bh_cat


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


def gru_scan_diagonal_forward_pytorch(
    gi: torch.Tensor,
    h0: torch.Tensor,
    Wh_diag: torch.Tensor,
    bh_cat: torch.Tensor,
    *,
    h_in_quant: tuple[float, int, int] | None = None,
    h_out_quant: tuple[float, int, int] | None = None,
) -> torch.Tensor:
    """Reference forward for the diagonal scan, in PyTorch.

    Args:
        gi: [T, B, 3H] — pre-batched input projection (already with bi).
        h0: [B, H]
        Wh_diag: [3, H] — per-gate diagonal weights, (r, z, n).
        bh_cat: [3H]
    Returns:
        out: [T, B, H]
    """
    T, B, three_H = gi.shape
    H = three_H // 3
    assert Wh_diag.shape == (3, H)
    out = torch.empty(T, B, H, device=gi.device, dtype=gi.dtype)
    h = h0
    bh = bh_cat.view(3, H)
    w_hr, w_hz, w_hn = Wh_diag[0], Wh_diag[1], Wh_diag[2]

    for t in range(T):
        h_for_matmul, _ = _fake_quant(h, h_in_quant)
        gh_r = w_hr * h_for_matmul + bh[0]
        gh_z = w_hz * h_for_matmul + bh[1]
        gh_n = w_hn * h_for_matmul + bh[2]

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
def gru_scan_diagonal_fwd_kernel(  # type: ignore[no-untyped-def]
    gi_ptr,            # [T, B, 3H], fp32
    h0_ptr,            # [B, H], fp32
    Wh_ptr,            # [3, H], fp32
    bh_ptr,            # [3H], fp32
    out_ptr,           # [T, B, H], fp32
    T,
    B,
    sg_t, sg_b,
    sh0_b,
    sW_g,              # stride of Wh's gate axis (=H for [3,H] contiguous)
    so_t, so_b,
    h_in_scale,
    h_in_qmin,
    h_in_qmax,
    h_out_scale,
    h_out_qmin,
    h_out_qmax,
    H: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_H: tl.constexpr,
    QUANT_H_IN: tl.constexpr,
    QUANT_H_OUT: tl.constexpr,
):
    """Persistent forward for the diagonal recurrence.

    Grid (pid_b, pid_h): each program owns one ``[BLOCK_B, BLOCK_H]`` slab
    of the hidden state and runs the full T-loop on it independently.
    There is no cross-program dependency at any timestep — the recurrence
    is pointwise across H — so no spin-wait barrier is needed.

    ``h`` carries across timesteps in registers; we only store outputs to
    global memory for downstream consumers (loss / backward).
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = offs_b < B
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < H
    mask_bh = mask_b[:, None] & mask_h[None, :]

    # Per-gate diagonal weight tile [BLOCK_H], constant across t.
    w_hr = tl.load(Wh_ptr + 0 * sW_g + offs_h, mask=mask_h, other=0.0)
    w_hz = tl.load(Wh_ptr + 1 * sW_g + offs_h, mask=mask_h, other=0.0)
    w_hn = tl.load(Wh_ptr + 2 * sW_g + offs_h, mask=mask_h, other=0.0)
    bhr = tl.load(bh_ptr + 0 * H + offs_h, mask=mask_h, other=0.0)
    bhz = tl.load(bh_ptr + 1 * H + offs_h, mask=mask_h, other=0.0)
    bhn = tl.load(bh_ptr + 2 * H + offs_h, mask=mask_h, other=0.0)

    # Load h0 for this slab once; h carries in registers across the T-loop.
    h = tl.load(
        h0_ptr + offs_b[:, None] * sh0_b + offs_h[None, :],
        mask=mask_bh, other=0.0,
    )

    for t in range(0, T):
        h_for_matmul = h
        if QUANT_H_IN:
            q = tl.extra.cuda.libdevice.rint(h_for_matmul / h_in_scale)
            q = tl.minimum(tl.maximum(q, h_in_qmin), h_in_qmax)
            h_for_matmul = q * h_in_scale

        gh_r = w_hr[None, :] * h_for_matmul + bhr[None, :]
        gh_z = w_hz[None, :] * h_for_matmul + bhz[None, :]
        gh_n = w_hn[None, :] * h_for_matmul + bhn[None, :]

        gi_base = (
            gi_ptr
            + t * sg_t
            + offs_b[:, None] * sg_b
            + offs_h[None, :]
        )
        gir = tl.load(gi_base + 0 * H, mask=mask_bh, other=0.0)
        giz = tl.load(gi_base + 1 * H, mask=mask_bh, other=0.0)
        gin = tl.load(gi_base + 2 * H, mask=mask_bh, other=0.0)

        r = tl.sigmoid(gir + gh_r)
        z = tl.sigmoid(giz + gh_z)
        n = tl.extra.cuda.libdevice.tanh(gin + r * gh_n)

        h_new = (1.0 - z) * n + z * h
        if QUANT_H_OUT:
            q = tl.extra.cuda.libdevice.rint(h_new / h_out_scale)
            q = tl.minimum(tl.maximum(q, h_out_qmin), h_out_qmax)
            h_new = q * h_out_scale

        out_ptrs = (
            out_ptr
            + t * so_t
            + offs_b[:, None] * so_b
            + offs_h[None, :]
        )
        tl.store(out_ptrs, h_new, mask=mask_bh)
        h = h_new


def gru_scan_diagonal_forward_triton(
    gi: torch.Tensor,
    h0: torch.Tensor,
    Wh_diag: torch.Tensor,
    bh_cat: torch.Tensor,
    *,
    block_b: int = 32,
    block_h: int = 64,
    num_warps: int = 4,
    num_stages: int = 2,
    h_in_quant: tuple[float, int, int] | None = None,
    h_out_quant: tuple[float, int, int] | None = None,
) -> torch.Tensor:
    """Triton forward for the diagonal hidden-side scan."""
    if not (gi.is_cuda and Wh_diag.is_cuda):
        raise ValueError(
            "gi and Wh_diag must be CUDA tensors; got devices "
            f"gi={gi.device}, Wh_diag={Wh_diag.device}"
        )
    T, B, three_H = gi.shape
    H = three_H // 3
    if Wh_diag.shape != (3, H):
        raise ValueError(
            f"Wh_diag shape must be (3, H)=(3, {H}); got {tuple(Wh_diag.shape)}"
        )

    gi = gi.contiguous()
    h0 = h0.contiguous()
    Wh_diag = Wh_diag.contiguous()
    bh_cat = bh_cat.contiguous()

    out = torch.empty((T, B, H), device=gi.device, dtype=gi.dtype)

    in_s, in_qmin, in_qmax = h_in_quant or (1.0, -2**31, 2**31 - 1)
    out_s, out_qmin, out_qmax = h_out_quant or (1.0, -2**31, 2**31 - 1)

    n_pid_b = triton.cdiv(B, block_b)
    n_pid_h = triton.cdiv(H, block_h)
    grid = (n_pid_b, n_pid_h)
    gru_scan_diagonal_fwd_kernel[grid](
        gi, h0, Wh_diag, bh_cat, out,
        T, B,
        gi.stride(0), gi.stride(1),
        h0.stride(0),
        Wh_diag.stride(0),
        out.stride(0), out.stride(1),
        in_s, in_qmin, in_qmax,
        out_s, out_qmin, out_qmax,
        H=H,
        BLOCK_B=block_b,
        BLOCK_H=block_h,
        QUANT_H_IN=h_in_quant is not None,
        QUANT_H_OUT=h_out_quant is not None,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


@triton.jit  # type: ignore[untyped-decorator]
def gru_scan_diagonal_bwd_kernel(  # type: ignore[no-untyped-def]
    # forward inputs (read-only)
    gi_ptr,              # [T, B, 3H]
    h0_ptr,              # [B, H]
    Wh_ptr,              # [3, H]
    bh_ptr,              # [3H]
    out_ptr,             # [T, B, H]
    # upstream
    dout_ptr,            # [T, B, H]
    # outputs
    dgi_ptr,             # [T, B, 3H]
    dh0_ptr,             # [B, H]
    # per-pid_b partial buffers (reduced across pid_b in Python)
    dWh_partial_ptr,     # [n_pid_b, 3, H]
    dbh_partial_ptr,     # [n_pid_b, 3H]
    T, B,
    sg_t, sg_b,
    sh0_b,
    sW_g,
    so_t, so_b,
    sdo_t, sdo_b,
    sdgi_t, sdgi_b,
    sdh0_b,
    sdWp_pid, sdWp_g,
    sdbp_pid,
    h_in_scale,
    h_in_qmin,
    h_in_qmax,
    h_out_scale,
    h_out_qmin,
    h_out_qmax,
    H: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_H: tl.constexpr,
    QUANT_H_IN: tl.constexpr,
    QUANT_H_OUT: tl.constexpr,
):
    """Persistent backward for the diagonal recurrence.

    Each ``(pid_b, pid_h)`` program owns ``[BLOCK_B, BLOCK_H]`` of state and
    runs the reverse T-loop on it. ``dh_acc`` carries in registers, same
    way ``h`` did in the forward.

    Weight-grad partials (``dWh_partial``, ``dbh_partial``) are accumulated
    in registers across the time loop, then stored once per program at
    kernel exit. Multiple ``pid_b`` programs contribute to the same H
    slice — reduce across the ``pid_b`` axis in Python after the kernel.
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = offs_b < B
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < H
    mask_bh = mask_b[:, None] & mask_h[None, :]

    # Per-gate diagonal weight tile [BLOCK_H], constant across t.
    w_hr = tl.load(Wh_ptr + 0 * sW_g + offs_h, mask=mask_h, other=0.0)
    w_hz = tl.load(Wh_ptr + 1 * sW_g + offs_h, mask=mask_h, other=0.0)
    w_hn = tl.load(Wh_ptr + 2 * sW_g + offs_h, mask=mask_h, other=0.0)
    bhr = tl.load(bh_ptr + 0 * H + offs_h, mask=mask_h, other=0.0)
    bhz = tl.load(bh_ptr + 1 * H + offs_h, mask=mask_h, other=0.0)
    bhn = tl.load(bh_ptr + 2 * H + offs_h, mask=mask_h, other=0.0)

    # dh_acc carried in registers across the reverse loop.
    dh_acc = tl.zeros((BLOCK_B, BLOCK_H), dtype=tl.float32)

    # Per-gate dW_h partial accumulator [BLOCK_H], summed over (t, b in this slab).
    dWh_r_acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    dWh_z_acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    dWh_n_acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    dbh_r_acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    dbh_z_acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    dbh_n_acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

    for t_rev in range(0, T):
        t = T - 1 - t_rev

        # Load h_prev for this slab — h0 at t==0, else out[t-1].
        if t == 0:
            h_prev_ptr = h0_ptr
            sh_prev_b = sh0_b
        else:
            h_prev_ptr = out_ptr + (t - 1) * so_t
            sh_prev_b = so_b
        h_prev = tl.load(
            h_prev_ptr + offs_b[:, None] * sh_prev_b + offs_h[None, :],
            mask=mask_bh, other=0.0,
        )

        # Recompute forward: gh_g = w_h[g] * fake_quant(h_prev) + bh[g].
        if QUANT_H_IN:
            q_in_unclamped = tl.extra.cuda.libdevice.rint(h_prev / h_in_scale)
            mask_in = (q_in_unclamped >= h_in_qmin) & (q_in_unclamped <= h_in_qmax)
            q_in_clamped = tl.minimum(tl.maximum(q_in_unclamped, h_in_qmin), h_in_qmax)
            h_for_matmul = q_in_clamped * h_in_scale
        else:
            h_for_matmul = h_prev

        gh_r = w_hr[None, :] * h_for_matmul + bhr[None, :]
        gh_z = w_hz[None, :] * h_for_matmul + bhz[None, :]
        gh_n = w_hn[None, :] * h_for_matmul + bhn[None, :]

        gi_base = (
            gi_ptr
            + t * sg_t
            + offs_b[:, None] * sg_b
            + offs_h[None, :]
        )
        gir = tl.load(gi_base + 0 * H, mask=mask_bh, other=0.0)
        giz = tl.load(gi_base + 1 * H, mask=mask_bh, other=0.0)
        gin = tl.load(gi_base + 2 * H, mask=mask_bh, other=0.0)

        r = tl.sigmoid(gir + gh_r)
        z = tl.sigmoid(giz + gh_z)
        n = tl.extra.cuda.libdevice.tanh(gin + r * gh_n)

        # dh_t = dout[t] + dh_acc (carrying).
        dout_t = tl.load(
            dout_ptr + t * sdo_t + offs_b[:, None] * sdo_b + offs_h[None, :],
            mask=mask_bh, other=0.0,
        )
        dh_t = dout_t + dh_acc

        # STE backward of quant_h_out: gradient on quantized h_t -> on raw.
        if QUANT_H_OUT:
            h_t_raw = (1.0 - z) * n + z * h_prev
            q_out_unclamped = tl.extra.cuda.libdevice.rint(h_t_raw / h_out_scale)
            mask_out = (q_out_unclamped >= h_out_qmin) & (q_out_unclamped <= h_out_qmax)
            dh_t = tl.where(mask_out, dh_t, 0.0)

        # h_t_raw = (1-z)*n + z*h_prev
        dz = dh_t * (h_prev - n)
        dh_prev_direct = dh_t * z

        # n = tanh(gn_pre); gn_pre = gi_n + r * gh_n
        dgn_pre = dh_t * (1.0 - z) * (1.0 - n * n)
        dgi_n = dgn_pre
        dr = dgn_pre * gh_n
        dgh_n = dgn_pre * r

        dgz_pre = dz * z * (1.0 - z)
        dgi_z = dgz_pre
        dgh_z = dgz_pre

        dgr_pre = dr * r * (1.0 - r)
        dgi_r = dgr_pre
        dgh_r = dgr_pre

        # Store dgi[t] tile for this slab, three gates.
        dgi_base = (
            dgi_ptr
            + t * sdgi_t
            + offs_b[:, None] * sdgi_b
            + offs_h[None, :]
        )
        tl.store(dgi_base + 0 * H, dgi_r, mask=mask_bh)
        tl.store(dgi_base + 1 * H, dgi_z, mask=mask_bh)
        tl.store(dgi_base + 2 * H, dgi_n, mask=mask_bh)

        # dh_prev via diagonal weight: dgh[g] * w_h[g], summed over gates,
        # then STE-mask through quant_h_in. Pure pointwise.
        dh_prev_via_W = (
            dgh_r * w_hr[None, :]
            + dgh_z * w_hz[None, :]
            + dgh_n * w_hn[None, :]
        )
        if QUANT_H_IN:
            dh_prev_via_W = tl.where(mask_in, dh_prev_via_W, 0.0)

        # Register-resident weight-grad accumulation. dW_h[g, i] gets a
        # contribution dgh[g, b, i] * h_for_matmul[b, i] for every (t, b) in
        # this slab — reduce over batch axis here, time accumulates in regs.
        dWh_r_acc += tl.sum(dgh_r * h_for_matmul, axis=0)
        dWh_z_acc += tl.sum(dgh_z * h_for_matmul, axis=0)
        dWh_n_acc += tl.sum(dgh_n * h_for_matmul, axis=0)
        dbh_r_acc += tl.sum(dgh_r, axis=0)
        dbh_z_acc += tl.sum(dgh_z, axis=0)
        dbh_n_acc += tl.sum(dgh_n, axis=0)

        dh_acc = dh_prev_via_W + dh_prev_direct

    # After reverse loop: dh_acc holds gradient wrt h0 for this slab.
    tl.store(
        dh0_ptr + offs_b[:, None] * sdh0_b + offs_h[None, :],
        dh_acc, mask=mask_bh,
    )

    # Write per-pid_b weight-grad partials. pid_h shards the H axis
    # disjointly across programs with the same pid_b, so no atomic needed
    # — only different pid_b values collide on the same slice and we
    # reduce across pid_b in Python.
    dWh_base = dWh_partial_ptr + pid_b * sdWp_pid
    tl.store(dWh_base + 0 * sdWp_g + offs_h, dWh_r_acc, mask=mask_h)
    tl.store(dWh_base + 1 * sdWp_g + offs_h, dWh_z_acc, mask=mask_h)
    tl.store(dWh_base + 2 * sdWp_g + offs_h, dWh_n_acc, mask=mask_h)

    dbh_base = dbh_partial_ptr + pid_b * sdbp_pid + offs_h
    tl.store(dbh_base + 0 * H, dbh_r_acc, mask=mask_h)
    tl.store(dbh_base + 1 * H, dbh_z_acc, mask=mask_h)
    tl.store(dbh_base + 2 * H, dbh_n_acc, mask=mask_h)


def gru_scan_diagonal_backward_triton(
    gi: torch.Tensor,
    h0: torch.Tensor,
    Wh_diag: torch.Tensor,
    bh_cat: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    *,
    block_b: int = 32,
    block_h: int = 64,
    num_warps: int = 4,
    num_stages: int = 2,
    h_in_quant: tuple[float, int, int] | None = None,
    h_out_quant: tuple[float, int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Triton backward for the diagonal hidden-side scan."""
    T, B, three_H = gi.shape
    H = three_H // 3
    if Wh_diag.shape != (3, H):
        raise ValueError(
            f"Wh_diag shape must be (3, H)=(3, {H}); got {tuple(Wh_diag.shape)}"
        )

    gi = gi.contiguous()
    h0 = h0.contiguous()
    Wh_diag = Wh_diag.contiguous()
    bh_cat = bh_cat.contiguous()
    out = out.contiguous()
    dout = dout.contiguous()

    dgi = torch.zeros_like(gi)
    dh0 = torch.zeros_like(h0)

    n_pid_b = triton.cdiv(B, block_b)
    n_pid_h = triton.cdiv(H, block_h)

    dWh_partial = torch.zeros((n_pid_b, 3, H), device=gi.device, dtype=gi.dtype)
    dbh_partial = torch.zeros((n_pid_b, 3 * H), device=gi.device, dtype=gi.dtype)

    in_s, in_qmin, in_qmax = h_in_quant or (1.0, -2**31, 2**31 - 1)
    out_s, out_qmin, out_qmax = h_out_quant or (1.0, -2**31, 2**31 - 1)

    grid = (n_pid_b, n_pid_h)
    gru_scan_diagonal_bwd_kernel[grid](
        gi, h0, Wh_diag, bh_cat, out,
        dout,
        dgi, dh0,
        dWh_partial, dbh_partial,
        T, B,
        gi.stride(0), gi.stride(1),
        h0.stride(0),
        Wh_diag.stride(0),
        out.stride(0), out.stride(1),
        dout.stride(0), dout.stride(1),
        dgi.stride(0), dgi.stride(1),
        dh0.stride(0),
        dWh_partial.stride(0), dWh_partial.stride(1),
        dbh_partial.stride(0),
        in_s, in_qmin, in_qmax,
        out_s, out_qmin, out_qmax,
        H=H,
        BLOCK_B=block_b,
        BLOCK_H=block_h,
        QUANT_H_IN=h_in_quant is not None,
        QUANT_H_OUT=h_out_quant is not None,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    dWh_diag = dWh_partial.sum(dim=0)   # [3, H]
    dbh = dbh_partial.sum(dim=0)        # [3H]
    return dgi, dh0, dWh_diag, dbh


def gru_scan_diagonal_backward_pytorch(
    gi: torch.Tensor,
    h0: torch.Tensor,
    Wh_diag: torch.Tensor,
    bh_cat: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    *,
    h_in_quant: tuple[float, int, int] | None = None,
    h_out_quant: tuple[float, int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference backward for the diagonal scan.

    Returns:
        dgi:      [T, B, 3H]
        dh0:      [B, H]
        dWh_diag: [3, H]
        dbh:      [3H]
    """
    T, B, _ = gi.shape
    H = h0.shape[-1]
    dgi = torch.zeros_like(gi)
    dWh_diag = torch.zeros_like(Wh_diag)
    dbh = torch.zeros_like(bh_cat)
    dh_acc = torch.zeros_like(h0)
    w_hr, w_hz, w_hn = Wh_diag[0], Wh_diag[1], Wh_diag[2]

    for t in reversed(range(T)):
        h_prev = h0 if t == 0 else out[t - 1]
        h_for_matmul, mask_in = _fake_quant(h_prev, h_in_quant)

        gi_r = gi[t, :, 0:H]
        gi_z = gi[t, :, H:2 * H]
        gi_n = gi[t, :, 2 * H:3 * H]
        gh_r = w_hr * h_for_matmul + bh_cat[0:H]
        gh_z = w_hz * h_for_matmul + bh_cat[H:2 * H]
        gh_n = w_hn * h_for_matmul + bh_cat[2 * H:3 * H]

        r = torch.sigmoid(gi_r + gh_r)
        z = torch.sigmoid(gi_z + gh_z)
        n = torch.tanh(gi_n + r * gh_n)
        h_t_raw = (1.0 - z) * n + z * h_prev

        dh_t = dout[t] + dh_acc
        if h_out_quant is not None:
            _, mask_out = _fake_quant(h_t_raw, h_out_quant)
            assert mask_out is not None  # non-None params => non-None mask
            dh_t = dh_t * mask_out

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

        dgi[t] = torch.cat([dgi_r, dgi_z, dgi_n], dim=-1)

        # Weight grads: dWh_g[i] = sum_{t,b} dgh_g[t,b,i] * h_for_matmul[b,i]
        dWh_diag[0] += (dgh_r * h_for_matmul).sum(dim=0)
        dWh_diag[1] += (dgh_z * h_for_matmul).sum(dim=0)
        dWh_diag[2] += (dgh_n * h_for_matmul).sum(dim=0)
        dbh[0:H] += dgh_r.sum(dim=0)
        dbh[H:2 * H] += dgh_z.sum(dim=0)
        dbh[2 * H:3 * H] += dgh_n.sum(dim=0)

        # dh_prev_via_W: pointwise, then STE mask on quant_h_in.
        dh_prev_via_W = dgh_r * w_hr + dgh_z * w_hz + dgh_n * w_hn
        if mask_in is not None:
            dh_prev_via_W = dh_prev_via_W * mask_in

        dh_acc = dh_prev_via_W + dh_prev_direct

    return dgi, dh_acc, dWh_diag, dbh


class GRUScanDiagonalFunction(torch.autograd.Function):
    """autograd wrapper around the diagonal persistent kernels."""

    @staticmethod
    def forward(
        ctx: Any,
        gi: torch.Tensor,
        h0: torch.Tensor,
        Wh_diag: torch.Tensor,
        bh_cat: torch.Tensor,
        h_in_quant: tuple[float, int, int] | None,
        h_out_quant: tuple[float, int, int] | None,
    ) -> torch.Tensor:
        out = gru_scan_diagonal_forward_triton(
            gi, h0, Wh_diag, bh_cat,
            h_in_quant=h_in_quant, h_out_quant=h_out_quant,
        )
        ctx.save_for_backward(gi, h0, Wh_diag, bh_cat, out)
        ctx.h_in_quant = h_in_quant
        ctx.h_out_quant = h_out_quant
        return out

    @staticmethod
    def backward(
        ctx: Any, dout: torch.Tensor
    ) -> Any:
        gi, h0, Wh_diag, bh_cat, out = ctx.saved_tensors
        grads = gru_scan_diagonal_backward_triton(
            gi, h0, Wh_diag, bh_cat, out, dout,
            h_in_quant=ctx.h_in_quant, h_out_quant=ctx.h_out_quant,
        )
        return (*grads, None, None)


def gru_scan_diagonal(
    gi: torch.Tensor,
    h0: torch.Tensor,
    Wh_diag: torch.Tensor,
    bh_cat: torch.Tensor,
    *,
    h_in_quant: tuple[float, int, int] | None = None,
    h_out_quant: tuple[float, int, int] | None = None,
) -> torch.Tensor:
    """Public API: differentiable diagonal-hidden-side GRU scan.

    Mirror of ``gru_scan_persistent`` and ``gru_scan_monarch`` with a
    diagonal hidden-side weight (one vector per gate).

    Args:
        gi:      [T, B, 3H] input projection (already with bi).
        h0:      [B, H] initial hidden state.
        Wh_diag: [3, H] per-gate diagonal weights, (r, z, n).
        bh_cat:  [3H]
        h_in_quant / h_out_quant: optional ``(scale, qmin, qmax)`` for
            in-kernel fake-quant on hidden state (per-tensor symmetric,
            frozen scale). Same semantics as the other scan kernels.

    Use ``extract_diagonal_factors(cell)`` to pull the weight/bias out of
    a structured ``GRUCellQuant``.
    """
    return cast(
        torch.Tensor,
        GRUScanDiagonalFunction.apply(  # type: ignore[no-untyped-call]
            gi, h0, Wh_diag, bh_cat, h_in_quant, h_out_quant
        ),
    )

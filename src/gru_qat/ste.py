"""Straight-Through Estimator (STE) autograd functions.

These are the only places where the QAT gradient story lives. Everything
else in the library is a regular PyTorch op.

Phase 1 implementation: identity gradient through round/clamp.
Phase 3+ extensions: LSQ-style learnable step gradients (see TODO below).
"""

from __future__ import annotations

from typing import Any, cast

import torch


class STERound(torch.autograd.Function):
    """Round-to-nearest-even on the forward; identity on the backward.

    Use this anywhere we need a quantized integer value to flow into
    fake-quant arithmetic but want gradients to pass through as if the
    rounding were absent.
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        return torch.round(x)

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> torch.Tensor:
        return grad_output


class STEClamp(torch.autograd.Function):
    """Clamp to [qmin, qmax] on the forward; pass gradient only inside the
    range on the backward (clipped STE).

    This is the standard QAT clamp: out-of-range values get zero gradient,
    which prevents weights from drifting beyond the representable range.
    """

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        qmin: float,
        qmax: float,
    ) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.qmin = qmin
        ctx.qmax = qmax
        return torch.clamp(x, qmin, qmax)

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None]:
        (x,) = ctx.saved_tensors
        mask = (x >= ctx.qmin) & (x <= ctx.qmax)
        return grad_output * mask, None, None


def fake_quant_ste(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    qmin: float,
    qmax: float,
) -> torch.Tensor:
    """Standard fake-quant: q = clamp(round(x/scale + zp), qmin, qmax);
    out = (q - zp) * scale.

    Gradients flow through round (identity STE) and clamp (clipped STE).
    `scale` and `zero_point` may carry gradients themselves (LSQ); that is
    handled by them being regular tensors here, not by this function.

    Shapes: scale/zp must be broadcastable to x. For per-tensor, they are
    scalars; for per-channel, they have a singleton in every dim except
    the channel axis; for per-group, they have a singleton in every dim
    except the (channel, group) axes after a reshape — see
    FakeQuantizePerGroup.
    """
    # torch.autograd.Function.apply is untyped in the torch stubs — cast
    # the result back to Tensor; the call itself is unavoidably untyped.
    q = cast(
        torch.Tensor,
        STERound.apply(x / scale + zero_point),  # type: ignore[no-untyped-call]
    )
    q = cast(
        torch.Tensor,
        STEClamp.apply(q, qmin, qmax),  # type: ignore[no-untyped-call]
    )
    return (q - zero_point) * scale


# TODO(phase=3): LSQ gradient scaling.
# In LSQ (Learnable Step-size Quantization, Esser et al. 2020), the scale
# parameter is learnable and its gradient is scaled by 1/sqrt(N*Qp) where
# N is fan-in and Qp is the positive quantization range. Implement as
# `LSQRound` and let FakeQuantize subclasses opt in via a flag.

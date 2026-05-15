"""Calibration utilities — Phase 4.

Phase 4 work item: implement `calibrate(layer, loader, n_batches)` that
runs the layer in min_max observer mode for n_batches and then freezes.

The interface below is the target. Bodies are stubs.

Design note: we want calibration to be *separable* from the model so
calibration data is decoupled from training data. Typical workflow:

    layer = GRULayer(..., recipe=recipe_with_min_max_mode)
    train(layer, train_loader)  # QAT
    calibrate(layer, val_loader, n_batches=64)  # gather act stats
    layer.freeze()
    export(layer)  # to inference kernel

We do *not* re-use training stats for calibration because training-time
augmentation can shift activation distributions in ways the deployed
model never sees.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


@torch.no_grad()
def calibrate(
    module: nn.Module,
    loader: Iterable[object],
    n_batches: int = 64,
    *,
    only_activations: bool = True,
    verbose: bool = False,
) -> dict[str, dict[str, float | list[float]]]:
    """Run the module on n_batches in observer mode; return a stats summary.

    Switches every (optionally activation-only) FakeQuantize in ``module``
    to ``mode="min_max"``, resets its running stats, runs forward passes
    on the loader, and returns a summary. Does NOT auto-freeze — the
    caller decides when to call ``freeze_all(module)`` or
    ``module.freeze()``.

    Args:
        module: a GRULayer (or any nn.Module containing FakeQuantize submodules).
        loader: yields tensors or tuples of tensors to feed into module.forward.
            - A single tensor → ``module(batch)``.
            - A tuple/list      → ``module(*batch)`` (e.g., ``(x, h0)``).
        n_batches: stop after this many batches (or earlier if loader runs out).
        only_activations: if True (default), only switch activation-side
            quantizers (``quant_x``, ``quant_h_in``, ``quant_h_out``,
            ``quant_gate_*``) to min_max. Weight quantizers stay in their
            current mode (typically ``dynamic`` — scales already correct
            from the static weights). If False, all FakeQuantize modules
            switch to min_max.
        verbose: print progress.

    Returns:
        Summary dict keyed by qualname with running_min / running_max /
        bits / axis / initialized for each quantizer. Lists rather than
        scalars for per-channel running stats (post-fix).
    """
    from gru_qat.quantizers import FakeQuantize

    # Pick which quantizers to calibrate.
    quantizers: list[tuple[str, FakeQuantize]] = []
    activation_names = (
        "quant_x", "quant_h_in", "quant_h_out",
        "quant_gate_r", "quant_gate_z", "quant_gate_n",
        "quant_struct_Wi_r", "quant_struct_Wi_z", "quant_struct_Wi_n",
        "quant_struct_Wh_r", "quant_struct_Wh_z", "quant_struct_Wh_n",
    )
    for name, m in module.named_modules():
        if not isinstance(m, FakeQuantize):
            continue
        leaf = name.rsplit(".", 1)[-1]
        if only_activations and leaf not in activation_names:
            continue
        quantizers.append((name, m))

    # Reset observers and switch to min_max mode.
    for _, m in quantizers:
        m.config.mode = "min_max"
        m.running_min = torch.tensor(float("inf"), device=m.scale.device)
        m.running_max = torch.tensor(float("-inf"), device=m.scale.device)
        m._initialized = False

    # Stash and restore module training state so dropout etc. don't fire.
    was_training = module.training
    module.eval()
    try:
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            if isinstance(batch, torch.Tensor):
                module(batch)
            elif isinstance(batch, (tuple, list)):
                module(*batch)
            elif isinstance(batch, dict):
                module(**batch)
            else:
                raise TypeError(
                    f"calibrate: loader yielded unsupported type {type(batch)}; "
                    "expected Tensor / tuple / list / dict"
                )
            if verbose:
                print(f"[calibrate] batch {i + 1}/{n_batches}")
    finally:
        if was_training:
            module.train()

    summary: dict[str, dict[str, float | list[float]]] = {}
    for name, m in quantizers:
        rmin = m.running_min
        rmax = m.running_max
        summary[name] = {
            "running_min": rmin.item() if rmin.numel() == 1 else rmin.flatten().tolist(),
            "running_max": rmax.item() if rmax.numel() == 1 else rmax.flatten().tolist(),
            "bits": m.config.bits,
            "axis": m.config.axis,  # type: ignore[dict-item]
            "initialized": bool(m._initialized),
        }
    return summary


def freeze_all(module: nn.Module) -> None:
    """Freeze every FakeQuantize in module. After this, scales are read-only."""
    from gru_qat.quantizers import FakeQuantize

    for m in module.modules():
        if isinstance(m, FakeQuantize):
            m.freeze()

---
status: investigating
trigger: "monarch Triton fwd quant-on bug: 142/162 fast cases of test_monarch_quant_fwd fail torch.equal with max_abs_diff = h_scale = 0.02 exactly (one INT8 step)"
created: 2026-05-14T00:00:00Z
updated: 2026-05-14T00:00:00Z
---

## Current Focus

hypothesis: Triton monarch kernel uses a different rounding op (e.g., tl.math.rint half-to-even vs PyTorch torch.round half-to-even, or floor(x+0.5) vs round half-to-even) for in-kernel fake-quant, causing one-INT8-step divergence at rounding boundaries.
test: Read PyTorch STE rounding op + Triton monarch in-kernel quant op, identify the specific rounding instruction in each.
expecting: A specific instruction-level mismatch (most likely in scan_monarch.py vs ste.py).
next_action: Read .planning/phases/04-quant-on-bit-identity/04-VERIFICATION.md to confirm failure signature.

## Symptoms

expected: PyTorch reference path's hidden quantization matches Triton monarch fwd kernel's in-kernel fake-quant exactly (torch.equal).
actual: 142/162 fast cases diverge by max_abs_diff = h_scale = 0.02 (exactly one INT8 step). Reproducible.
errors: test_monarch_quant_fwd torch.equal assertion fails with max_abs_diff = h_scale.
reproduction: pytest tests/test_triton_monarch_strict.py::test_monarch_quant_fwd -q on CUDA at (T=8, B=1, H=128, nblocks=2).
started: Phase 4 verification of Monarch fwd kernel under quant-on bit-identity.

## Eliminated
<!-- APPEND only -->

## Evidence
<!-- APPEND only -->

## Resolution

root_cause: 
fix: 
verification: 
files_changed: []

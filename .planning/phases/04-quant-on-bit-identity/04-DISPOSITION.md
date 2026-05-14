# Phase 4 Disposition (D-42)

**Resolved:** 2026-05-14 (Plan 04-01 checkpoint:human-verify)
**Recipe under test:** frozen INT8 per-channel weight (axis=0) + per-tensor input_act + per-tensor hidden (`bits=8, mode='min_max'` + `mode='frozen'` for hidden), inline calibrate then `cell.freeze_quantizers()`.

## Empirical Probe Result (T=8, B=4, H=64, dense, cls=realistic)

| Tensor | torch.equal | max_abs_diff | / h_scale |
|---|---|---|---|
| `out` | ✓ PASS | 0.0 | 0 |
| `h_T` | ✓ PASS | 0.0 | 0 |
| `dx` | fail | 1.16e-09 | 5.8e-08 |
| `dh0` | fail | 1.79e-07 | 8.9e-06 |
| `dWh_cat` | fail | **1.12e-03** | 5.6e-02 |
| `dbh_cat` | fail | 4.77e-07 | 2.4e-05 |

`h_scale ≈ 0.02` (one INT8 step in this recipe). All backward divergences are sub-INT8-step.

## Disposition: ASYMMETRIC

### Forward (`out`, `h_T`)
**Assertion: `torch.equal`** (bit-identical).

Rationale: empirically holds. INT8 fake-quant via `quant_h_out` rounds both Triton-TF32-matmul outputs and PyTorch-fp32-matmul outputs to the same INT8 grid. Pre-quant fp32 values differ; post-quant int values are identical.

### Backward (`dx`, `dh_0`, `dWh_cat`, `dbh_cat`)
**Assertion: `abs_diff < h_scale * 1`** (within one INT8 step).

Rationale: fp32 reduction-order drift between Triton `tl.dot` and PyTorch matmul accumulates over batch + time dimensions. Worst observed (`dWh_cat = 1.12e-03 = 5.6% of h_scale`) is well within the 1-INT8-step budget. STE backward through `fake_quant_ste` does not re-quantize gradients, so they remain fp32 and exhibit the underlying matmul-order drift.

## Test idiom for Plans 04-02..04

Use a single helper `_assert_quant_parity` per strict file (byte-identical across files per D-43):

```python
def _assert_quant_parity(
    name: str,
    ref: torch.Tensor,
    tri: torch.Tensor,
    h_scale: float,
    *,
    strict: bool,
) -> None:
    """Assert quant-on parity per the Phase 4 D-42 disposition.

    strict=True (forward / h_T):    torch.equal contract.
    strict=False (backward grads):  abs_diff < h_scale (one INT8 step).
    """
    if strict:
        assert torch.equal(ref, tri), (
            f"quant-on bit-identity failed for {name}: "
            f"max_abs_diff={(ref - tri).abs().max().item():.4e} "
            f"(expected 0.0)"
        )
    else:
        max_diff = (ref - tri).abs().max().item()
        assert max_diff < h_scale, (
            f"quant-on tight-INT8-step bound failed for {name}: "
            f"max_abs_diff={max_diff:.4e}, h_scale={h_scale:.4e}, "
            f"ratio={max_diff/h_scale:.2%}"
        )
```

Test-body usage:
```python
_assert_quant_parity("out", ref_out, tri_out, h_scale, strict=True)
_assert_quant_parity("h_T", ref_out[-1], tri_out[-1], h_scale, strict=True)
_assert_quant_parity("dx", x.grad, x.grad_tri, h_scale, strict=False)
# ... etc.
```

## Plans 04-02..04 must:

1. Read this file at task start.
2. Implement `_assert_quant_parity` locally (byte-identical to the above; D-43).
3. Use `strict=True` for `out` and `h_T`.
4. Use `strict=False` for `dx`, `dh_0`, `dWh_cat`, `dbh_cat` (and any other backward-gradient tensors).
5. Acceptance criteria includes a grep for "strict=True" appearing 2+ times per file (out + h_T) and "strict=False" appearing 4+ times per file (the 4 gradient tensors), per parametrized class.

## Bonus finding (orthogonal to Phase 4 scope)

Pre-existing Phase 2 strict-tier failures noticed during the probe:
- `test_butterfly_fwd_strict_matches_reference[8-1-32]` exceeds Phase 2's < 5e-4 tight-TF32 bound (max diff ~9.3e-3 at this shape).
- Analogous monarch bwd cases (max diff ~7.4e-4 vs < 5e-4 bound).

Stash-verified pre-existing on Plan 04-01 baseline (not caused by Phase 4 changes). Orchestrator decision: file 1 tracking bd issue, note in Phase 4 SUMMARY, defer the fix to a future hygiene phase (or revisit during Phase 7 audit report). Does NOT reopen Phase 2.

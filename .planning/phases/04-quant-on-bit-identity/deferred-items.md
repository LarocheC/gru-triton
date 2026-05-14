# Phase 4 — Deferred Items (not in scope for Plan 04-01)

## Pre-existing Phase 2 Strict-Tier Failure: `test_butterfly_fwd_strict_matches_reference`

**Discovered during:** Plan 04-01 Task 3 broad regression sweep (post QNT-04 Commit B).

**Failure:**
```
tests/test_triton_butterfly_strict.py::test_butterfly_fwd_strict_matches_reference[8-1-32]
AssertionError: butterfly fwd max abs diff 9.3207e-03 (T=8,B=1,H=32)
assert 0.009320694953203201 < 0.0005
```

**Pre-existence verification:** Confirmed reproducible at Plan 04-01's commit baseline (`git stash` of the QNT-04 Commit B fix, then re-ran the failing case — same failure shape, same magnitude). NOT caused by `src/gru_qat/quantizers.py:_update_observer` per-axis-reduction fix.

**Disposition:** OUT OF SCOPE for Plan 04-01 per `<deviation_rules>` SCOPE BOUNDARY clause ("Only auto-fix issues DIRECTLY caused by the current task's changes"). The strict-tier butterfly bound at `< 5e-4` is the Phase 2 disposition (Option C / TF32 reality); the realistic-tier butterfly sibling at `tests/test_butterfly_dispatch.py:160` uses a looser bound and continues to pass. This is a known TF32 / in-kernel `tl.dot` mantissa friction, not a Phase 4 finding.

**File `bd` issue:** TODO at Phase 4 wrap-up (Plan 04-05 phase-exit) if still reproducing.

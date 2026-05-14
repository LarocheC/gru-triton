# Phase 5: Calibration + Freeze Lifecycle — DISCUSSION LOG

**Date:** 2026-05-14
**Mode:** discuss (default)
**Phase:** 05-calibration-freeze-lifecycle

This is a human-reference record of the discuss-phase conversation. CONTEXT.md is the canonical output consumed by downstream agents.

---

## Gray Area Selection

**Question:** Which areas do you want to discuss for Phase 5?
**Selection (multi):** All four
- Kernel coverage for CAL-03
- Tolerance contract for CAL-03
- Test file structure
- Calibration corpus for the test loader

A fifth area (parallel-execution race mitigation) was added during the discussion because it directly follows from the kernel-coverage decision.

---

## Area A — Kernel coverage for CAL-03

**Question:** CAL-03 (post-freeze Triton round-trip) — which kernels should it cover?

**Options presented:**
- All 4 kernels (dense + diagonal + monarch + butterfly)
- Dense only (CAL-03 success criterion's literal scope)
- Dense + diagonal (the bit-identity-clean paths from Phase 4)

**Selection:** All 4 kernels.

**Decision rationale captured in CONTEXT.md § A.**

---

## Area B — Tolerance contract for CAL-03

**Question:** CAL-03 tolerance contract — what bound does the post-freeze round-trip assert?

**Options presented:**
- Reuse Phase 4's per-cluster dispositions (h_scale_mult per kernel × class × B)
- Realistic-class only at tight bounds (torch.equal where Phase 4 hit it)
- Two-tier: torch.equal where Phase 4 achieves it, per-cluster mult elsewhere

**Selection:** Reuse Phase 4's per-cluster dispositions.

**Decision rationale captured in CONTEXT.md § B.**

---

## Area C — Test file structure

**Question:** Where do the new Phase 5 tests live?

**Options presented:**
- Extend tests/test_calibration.py with all 3 new tests + kernel-parametrized CAL-03
- Split: test_calibration.py (CAL-01, CAL-02, dense CAL-03) + each strict file gets a CAL section
- All Phase 5 work in tests/test_calibration.py; copy strict-file helpers

**Selection:** Extend tests/test_calibration.py with all 3 new tests + kernel-parametrized CAL-03.

**Decision rationale captured in CONTEXT.md § C.**

---

## Area D — Calibration corpus

**Question:** What inputs feed the calibration loader (and the held-out CAL-03 batch)?

**Options presented:**
- Realistic-tier random inputs only
- Calibrate on realistic; sweep all 3 D-46 classes in the held-out CAL-03 batch
- Sweep all 3 classes in BOTH calibration corpus AND held-out batch

**Selection:** Sweep all 3 classes in BOTH calibration corpus AND held-out batch.

**Decision rationale captured in CONTEXT.md § D.**

---

## Area E — Parallel-execution race mitigation (follow-up to Area A)

**Question:** Phase 5 Wave 2 will have multiple plans editing tests/test_calibration.py concurrently. F-04-05-D recommended serialization or worktree isolation. Which mitigation?

**Options presented:**
- Serialize Wave 2 plans on tests/test_calibration.py
- git-worktree isolation per executor
- Single plan covers all 4 kernels (no Wave 2 at all)
- Accept the race (rely on the recovery pattern documented in 04-04-SUMMARY)

**Selection:** Single plan covers all 4 kernels (no Wave 2 at all).

**Decision rationale captured in CONTEXT.md § E.**

---

## Deferred Ideas Captured

See CONTEXT.md `<deferred>` section. Highlights:
- LSQ / PACT learnable activation scales (ACT-02) — v2
- bf16 around fake-quant — permanent non-goal (per Phase 0 design)
- Multi-layer / bidirectional calibration — v2 milestone
- Verifier-deferred kernel fixes for the 9 open bd issues — Phase 7

---

## Claude's discretion (delegated to research/planning)

- CAL-03 grid coarseness — suggested 1-2 shapes per kernel × 3 classes as a starting point; planner chooses based on coverage vs runtime tradeoff.
- Cross-file import mechanism — either direct `from tests.test_triton_*_strict import ...` or extract helpers to `tests/_phase4_quant_helpers.py`. Planner decides based on pytest collection behavior.
- Held-out batch generation — different `torch.manual_seed` than calibration; same shapes; same 3-class sweep.
- Plan commit granularity within the single Phase 5 plan — one commit per CAL-* test, or one combined commit. Planner's call.

---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: planning
stopped_at: Phase 7 context gathered
last_updated: "2026-05-15T07:23:08.362Z"
last_activity: 2026-05-15
progress:
  total_phases: 7
  completed_phases: 5
  total_plans: 21
  completed_plans: 20
  percent: 95
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-13)

**Core value:** Every code path that claims to compute a GRU must produce numerically equivalent output to `torch.nn.GRU` (under matched recipe), and any deviation must be a tested, documented, intentional one — not a silent drift.
**Current focus:** Phase 06 — edge-case-sweeps

## Current Position

Phase: 7
Plan: Not started
Status: Ready to plan
Last activity: 2026-05-15

Progress: [████████████░░] 84%

## Performance Metrics

**Velocity:**

- Total plans completed: 21
- Phase 4 plans: 5 (incl. verifier-driven dispositions as Plan 04-05 amendment)
- Phase 4 verifier-driven commits: 7 (f3e300c, 9049ec0, 922fbc3, bf01232, a8e5ccf, 4d47fca, e8a374d, 8789f4c)

**By Phase:**

| Phase | Plans | Status |
|-------|-------|--------|
| 1 | 5 | Complete ✓ 2026-05-13 |
| 2 | 6 | Complete ✓ 2026-05-13 (Option C disposition) |
| 3 | 3 | Complete ✓ 2026-05-14 |
| 4 | 5 | Complete ✓ 2026-05-14 (PASS-WITH-MAJOR-CAVEATS) |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table. Recent decisions affecting current work:

- Init: Baseline is `torch.nn.GRU` (cuDNN); gate-ordering / bias-fusion quirks live in test-helper layer, not in reference-path code.
- Init: Reference PyTorch path is ground truth for Triton/structured parity (no third baseline).
- Init: Forward + backward parity both required (recent fix cluster shows bwd is where bugs hide).
- Init: Tiered tolerance — < 1e-5 for cell + Triton-vs-reference, < 1e-4 for layer-vs-nn.GRU, bit-identical for quant-on.
- Init: Fix in-milestone (each finding → failing test → beads issue → fix → audit ends green).
- Phase 4 D-42 (revised post-verifier 2026-05-14): per-cluster `h_scale_mult` disposition table in `phases/04-quant-on-bit-identity/04-DISPOSITION.md`. Bit-identity (torch.equal) achieved only on dense fwd, diagonal fwd (realistic+near-saturation), and diagonal bwd. Other (kernel, direction, class) tuples use empirically-derived mults 2-20000 with bd-tracked Phase 7 remediation.
- Phase 4 D-43 (helper byte-uniformity): preserved across the 4 strict files; per-call `h_scale_mult` arguments diverge per cluster.
- Phase 4 root cause: TF32 reduction-order non-associativity (`gru-triton-rwm`, Phase 2 Option C) surfacing at the in-kernel-quant boundary. Reproducer at `.planning/debug/repro_monarch_rounding.py`.

### Pending Todos

- **Phase 5 plan-phase:** run `/gsd-plan-phase 5` to produce the single-plan implementation breakdown (5 tests: CAL-01 + CAL-02 + parametrized CAL-03 over 4 kernels + anti-pattern test). CONTEXT.md at `.planning/phases/05-calibration-freeze-lifecycle/05-CONTEXT.md` is consumed by the researcher + planner.
- **Phase 5 architectural decision (captured in CONTEXT § E):** single plan covers all 4 kernels — no Wave 2 parallelism on `tests/test_calibration.py` (sidesteps F-04-05-D race).
- **Phase 5 reuse target:** Phase 4 helper layer infrastructure imported into `tests/test_calibration.py` via cross-file imports from the 4 strict files.
- **Phase 5 tolerance contract:** consumes Phase 4's per-cluster post-freeze tolerance table; CAL-03 round-trip asserts at those bounds.

### Blockers/Concerns

- 9 open bd issues from Phase 4 are kernel-investigation or process tickets; none block Phase 5 entry. Tracked for Phase 7 audit report.

## Deferred Items

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| Phase 4 carry-forward | `gru-triton-lht` (F-04-05-A dense bwd, SUPERSEDED by mjy) | open, Phase 7 | 2026-05-14 |
| Phase 4 carry-forward | `gru-triton-5rk` (F-04-05-B butterfly fwd, EXTENDED by lqk) | open, Phase 7 | 2026-05-14 |
| Phase 4 carry-forward | `gru-triton-u00` (F-04-05-D parallel-execution race) | open, Phase 7 / Phase 5 process | 2026-05-14 |
| Phase 4 verifier-new | `gru-triton-in0` (F-04-VERIFIER-A monarch fwd) | open, Phase 7 | 2026-05-14 |
| Phase 4 verifier-new | `gru-triton-q3k` (F-04-VERIFIER-B monarch bwd) | open, Phase 7 | 2026-05-14 |
| Phase 4 verifier-new | `gru-triton-mjy` (F-04-VERIFIER-C dense bwd) | open, Phase 7 | 2026-05-14 |
| Phase 4 verifier-new | `gru-triton-lqk` (F-04-VERIFIER-D butterfly bwd) | open, Phase 7 | 2026-05-14 |
| Phase 4 verifier-new | `gru-triton-fpl` (F-04-VERIFIER-E diagonal fwd large-magnitude) | open, Phase 7 | 2026-05-14 |
| Phase 4 verifier-new | `gru-triton-e0l` (F-04-VERIFIER-F monarch bwd HW-limit) | open, Phase 7 | 2026-05-14 |
| Pre-existing carry | `gru-triton-e7t` (F-02-02-A diagonal bwd long-T) | open, Phase 7 | 2026-05-13 |
| Pre-existing carry | `gru-triton-4m6` (mypy/ruff debt) | open, Phase 7 | 2026-05-13 |
| Pre-existing carry | `gru-triton-6dz` (Phase 2 strict-tier small-shape failures) | open, Phase 7 | 2026-05-13 |

## Session Continuity

Last session: 2026-05-15T07:23:08.337Z
Stopped at: Phase 7 context gathered
Resuming: Run `/gsd-discuss-phase` for Phase 5 (CAL-01/02/03 — calibration + freeze lifecycle). Phase 5 inherits Phase 4's per-cluster post-freeze tolerance contract from `phases/04-quant-on-bit-identity/04-DISPOSITION.md`.

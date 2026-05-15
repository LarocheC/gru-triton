---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: milestone_complete
stopped_at: Completed 07-03-PLAN.md
last_updated: "2026-05-15T10:59:32.982Z"
last_activity: 2026-05-15
progress:
  total_phases: 7
  completed_phases: 7
  total_plans: 25
  completed_plans: 24
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-13)

**Core value:** Every code path that claims to compute a GRU must produce numerically equivalent output to `torch.nn.GRU` (under matched recipe), and any deviation must be a tested, documented, intentional one — not a silent drift.
**Current focus:** Phase 07 — audit-report-findings-handling

## Current Position

Phase: 07
Plan: Not started
Status: Milestone complete
Last activity: 2026-05-15

Progress: [██████████] 96%

## Performance Metrics

**Velocity:**

- Total plans completed: 25
- Phase 4 plans: 5 (incl. verifier-driven dispositions as Plan 04-05 amendment)
- Phase 4 verifier-driven commits: 7 (f3e300c, 9049ec0, 922fbc3, bf01232, a8e5ccf, 4d47fca, e8a374d, 8789f4c)

**By Phase:**

| Phase | Plans | Status |
|-------|-------|--------|
| 1 | 5 | Complete ✓ 2026-05-13 |
| 2 | 6 | Complete ✓ 2026-05-13 (Option C disposition) |
| 3 | 3 | Complete ✓ 2026-05-14 |
| 4 | 5 | Complete ✓ 2026-05-14 (PASS-WITH-MAJOR-CAVEATS) |
| Phase 07 P03 | 14min | 2 tasks | 2 files |
| Phase 07 P04 | 20min | 2 tasks | 2 files |

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
- [Phase 7]: D-10 — all 14 open bd issues closed by disposition bucket (3 FIX / 9 ACCEPTED-DIVERGENCE / 2 INDIVIDUAL); `bd ready` empty.
- [Phase 7]: D-09 — git-log test-before-fix audit complete (`07-git-log-audit.txt`); Phases 1-3 produced zero bug-fix commits so no ordering gap; no history rewritten.
- [Phase 7]: D-08 — AUDIT-REPORT.md authored at repo root (402 lines, 4 D-08 sections); milestone-closing deliverable RPT-03 complete; v1 Native-PyTorch Parity Audit milestone closes.

### Pending Todos

- **Phase 5 plan-phase:** run `/gsd-plan-phase 5` to produce the single-plan implementation breakdown (5 tests: CAL-01 + CAL-02 + parametrized CAL-03 over 4 kernels + anti-pattern test). CONTEXT.md at `.planning/phases/05-calibration-freeze-lifecycle/05-CONTEXT.md` is consumed by the researcher + planner.
- **Phase 5 architectural decision (captured in CONTEXT § E):** single plan covers all 4 kernels — no Wave 2 parallelism on `tests/test_calibration.py` (sidesteps F-04-05-D race).
- **Phase 5 reuse target:** Phase 4 helper layer infrastructure imported into `tests/test_calibration.py` via cross-file imports from the 4 strict files.
- **Phase 5 tolerance contract:** consumes Phase 4's per-cluster post-freeze tolerance table; CAL-03 round-trip asserts at those bounds.

### Blockers/Concerns

- None. All 14 carry-forward bd issues were triaged and CLOSED in Phase 7 Plan 07-03 (3 FIX / 9 ACCEPTED-DIVERGENCE / 2 INDIVIDUAL). `bd ready` is empty.

## Deferred Items

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| v2 kernel hardening | `KRN-01` (gru-triton-e0l monarch-bwd kernel-tiling redesign for consumer-GPU SMEM) | deferred to v2, bd ref recorded in REQUIREMENTS.md | 2026-05-15 |
| v2 kernel hardening | `KRN-02` (input_precision='ieee' TF32-elimination — resolves the 9 ACCEPTED-DIVERGENCE issues at root) | deferred to v2, bd refs recorded in REQUIREMENTS.md | 2026-05-15 |

_All 14 Phase 1-6 carry-forward bd issues are now CLOSED (Plan 07-03). The two rows above are the v2-deferred remediations recorded in REQUIREMENTS.md v2 section; their originating bd issues are CLOSED with v2 pointers, not left open._

## Session Continuity

Last session: 2026-05-15T10:59:19.797Z
Stopped at: Completed 07-03-PLAN.md
Resuming: Execute Phase 7 Plan 07-04 (`AUDIT-REPORT.md`, Wave 4 — the final plan). It reports the post-fix, all-14-issues-closed state and consumes `07-git-log-audit.txt` for the per-phase test-before-fix summary.

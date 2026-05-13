---
phase: 02-triton-fast-path-parity-vs-reference
plan: 05
subsystem: testing

tags: [triton, parity, tolerances, qat, diagonal, monarch, tf32, realistic-tier]

# Dependency graph
requires:
  - phase: 01-reference-path-parity-vs-nn-gru
    provides: reference-path ground truth (GRULayer use_triton=False, fp32 Identity) for the Triton parity gradient
provides:
  - tighter realistic-tier (TF32 / 'high') tolerance bounds in tests/test_triton_diagonal.py
  - tighter realistic-tier tolerance bounds in tests/test_triton_monarch.py
  - CPU-side acceptance gate (AST parse + pytest --collect-only) — full CUDA validation deferred to Plan 02-06
affects: [02-06-execute-cuda-verification]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Realistic-tier tightening discipline per D-13 second bullet — literal-only constant changes, surrounding relative-error idiom preserved
    - CPU-execution gate: cuda_only-gated tests are author-only on CPU; CUDA validation gated by a downstream phase-exit checkpoint
    - D-20 revert protocol (per-constant) for any tightening that fails on the GPU box — not invoked in this plan

key-files:
  created:
    - .planning/phases/02-triton-fast-path-parity-vs-reference/02-05-SUMMARY.md
  modified:
    - tests/test_triton_diagonal.py
    - tests/test_triton_monarch.py

key-decisions:
  - "Author tightenings on CPU without GPU validation; defer pass/fail to Plan 02-06's human-verify checkpoint per the plan's CUDA-gate specification."
  - "Document the parallel-execution race that swept the monarch tightenings into Plan 02-03's commit, rather than rewriting the parallel agent's commit (per destructive-git-prohibition)."

patterns-established:
  - "Per-file atomic commit for realistic-tier tightenings (D-20): one commit per kernel test file, not per-constant."
  - "Locked-rationale lines (docstring / inline comment citing TF32 / STE / QAT regime) are NEVER touched even when adjacent constants are tightened."

requirements-completed: [TRI-02, TRI-03]

# Metrics
duration: 12min
completed: 2026-05-13
---

# Phase 2 Plan 5: Realistic-Tier Tightening (Diagonal + Monarch) Summary

**Tightened 12 realistic-tier ('high' / TF32) tolerance constants in `test_triton_diagonal.py` and `test_triton_monarch.py` per the PATTERNS.md "Realistic-Tier Tightening Candidates" inventory; CUDA validation deferred to Plan 02-06's GPU human-verify checkpoint.**

## Performance

- **Duration:** ~12 min
- **Started:** 2026-05-13T21:08Z (approx, executor wall clock)
- **Completed:** 2026-05-13T21:15Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- 6 realistic-tier tolerance bounds in `tests/test_triton_diagonal.py` tightened per PATTERNS.md inventory (lines 121, 156, 194, 268, 331, 339).
- 6 realistic-tier tolerance bounds in `tests/test_triton_monarch.py` tightened per PATTERNS.md inventory (lines 248, 287, 288, 404, 409, 414).
- All locked-rationale lines preserved: diagonal lines 93 (Stage A `< 1e-5`) and 239 (STE rationale `< 1e-2`); monarch lines 101 (Stage A `< 1e-5`), 127 (TF32 rationale `< 5e-3`), 162 (QAT regime `< 1e-1`), 210 (QAT backward `< 1e-1`).
- No `@pytest.mark.xfail`, no commented-out assertions, no `try/except` wrapping (per D-27).
- Locked Phase 1 contract files (`tests/test_parity.py`, `tests/test_layer_parity.py`) unchanged and still passing (196 passed, 120 deselected for `-m "not slow"`).
- No helper functions (`_make_diagonal_layer`, `_make_monarch_layer`, `_build_gi_from_cell`) touched — only tolerance constants, per the parallel-execution boundary protecting concurrent Plans 02-02 and 02-03.

## Task Commits

1. **Task 1: Tighten realistic-tier tolerances in tests/test_triton_diagonal.py** — `75e8859` (test)
   - Clean atomic commit. 6 insertions / 6 deletions in one file.
2. **Task 2: Tighten realistic-tier tolerances in tests/test_triton_monarch.py** — see Issues Encountered below.
   - The edits are on disk and in git history, but **swept into commit `3ef47ef` by a parallel agent's `git add -A`** rather than landing as a separate Plan 02-05-attributed commit.

**No final metadata commit** for this plan — per the prompt's "Do NOT update STATE.md or ROADMAP.md" directive.

## Files Created/Modified

- `tests/test_triton_diagonal.py` — 6 realistic-tier tolerance constants tightened:
  - Line 121 (`test_diagonal_triton_forward_matches_pytorch`): `rel < 1e-4` → `rel < 1e-5`
  - Line 156 (`test_diagonal_triton_qat_forward_matches_pytorch`): `rel < 1e-3` → `rel < 1e-4`
  - Line 194 (`test_diagonal_triton_backward_matches_pytorch`, in-loop for `(dgi, dh0, dWh_diag, dbh)`): `rel < 1e-3` → `rel < 1e-4`
  - Line 268 (`test_diagonal_dispatch_matches_per_step`): `rel < 1e-4` → `rel < 1e-5`
  - Line 331 (`test_diagonal_dispatch_grad_matches_per_step`, x/h0 grads): `rel < 1e-3` → `rel < 1e-4`
  - Line 339 (`test_diagonal_dispatch_grad_matches_per_step`, parameter grads): `rel < 1e-3` → `rel < 1e-4`
- `tests/test_triton_monarch.py` — 6 realistic-tier tolerance constants tightened:
  - Line 248 (`test_monarch_triton_backward_matches_pytorch`, in-loop for `(dgi, dh0, dWh_struct, dbh)`): `rel < 5e-2` → `rel < 1e-2`
  - Line 287 (`test_grulayer_use_triton_matches_pytorch_path`, out): `rel < 5e-2` → `rel < 5e-3`
  - Line 288 (`test_grulayer_use_triton_matches_pytorch_path`, hT): `rel < 5e-2` → `rel < 5e-3`
  - Line 404 (`test_monarch_pytorch_backward_matches_cell`, dh0): `rel < 1e-4` → `rel < 1e-5`
  - Line 409 (`test_monarch_pytorch_backward_matches_cell`, dWh): `rel < 1e-4` → `rel < 1e-5`
  - Line 414 (`test_monarch_pytorch_backward_matches_cell`, dbh): `rel < 1e-4` → `rel < 1e-5`
- `.planning/phases/02-triton-fast-path-parity-vs-reference/02-05-SUMMARY.md` — this file.

## Candidates NOT Tightened (Per PATTERNS.md Hard Rule)

These are documented-rationale bounds. PATTERNS.md "any test whose docstring (or inline comment) explains why the bound is loose is not tightenable" — left as-is by design:

- `tests/test_triton_diagonal.py:93` — Stage A algebraic equality, already `< 1e-5`.
- `tests/test_triton_diagonal.py:239` — STE rounding can flip mask bits at QAT boundaries; documented inline at line 238.
- `tests/test_triton_monarch.py:101` — Stage A algebraic equality, already `< 1e-5`.
- `tests/test_triton_monarch.py:127` — "TF32 matmul + T-step compounding" documented at line 126.
- `tests/test_triton_monarch.py:162` — QAT in-kernel fake-quant regime, conventional `< 1e-1`.
- `tests/test_triton_monarch.py:210` — QAT backward regime, conventional `< 1e-1`.

## CUDA Verification Status

**PENDING (gated by Plan 02-06).** Per the plan's CPU-execution gate:

> Tightenings cannot be validated on CPU (the tests are cuda_only-gated). On CPU, the executor authors the tightenings and runs `pytest --collect-only` to confirm syntactic validity, then commits the tightenings WITHOUT a CUDA-validated pass. Plan 02-06's CUDA human-verify checkpoint validates the tightenings on a GPU box; if any tightening fails, the executor reverts THAT constant only with a docstring sentence.

CPU verification performed:

| Check | Diagonal | Monarch |
|-------|----------|---------|
| `python -c "import ast; ast.parse(...)"` | PASS | PASS |
| `pytest --collect-only -q` (test count delta) | 16 collected — unchanged | 16 collected — unchanged |
| `ruff check` | PASS (clean) | PRE-EXISTING errors documented in Issues |
| `grep -n xfail` returns nothing | PASS | PASS |
| Locked rationale string present (grep) | `STE rounding can flip mask bits` count == 1 | `TF32 matmul + T-step compounding` count == 1 |
| Stage A line untouched | line 93 still `< 1e-5` | line 101 still `< 1e-5` |
| Locked Phase 1 contracts unchanged | `git diff` empty on `tests/test_parity.py`, `tests/test_layer_parity.py` | same |
| `pytest tests/test_parity.py tests/test_layer_parity.py -q -m "not slow"` | 196 passed (D-28 contracts hold) | 196 passed |

If any constant fails CUDA validation in Plan 02-06, the per-constant D-20 revert protocol applies: revert THAT specific constant back to its prior value and add a one-line comment immediately above the assertion explaining why.

## Decisions Made

- **Author tightenings on CPU without CUDA verification.** Per the plan's CPU-execution gate, this is the acceptable workflow when no GPU is available at execute-time. The full pass/fail decision lives in Plan 02-06.
- **Do NOT auto-fix pre-existing ruff errors in `tests/test_triton_monarch.py`.** Per the executor's scope-boundary rule, pre-existing E402 / F401 errors are out of scope — they predate this commit and are unrelated to the tightening edits. Verified by stashing my edits and re-running `ruff check`: same errors. Logged here for the verifier and for a future hygiene pass.
- **Do NOT rewrite parallel agent's commit `3ef47ef`.** Per the destructive-git-prohibition (no `git reset --hard`, no `git commit --amend` on parallel work), the race is documented here rather than recovered.

## Deviations from Plan

### Issue 1: Parallel-execution race on monarch tightenings (Plan 02-03 commit absorbed Plan 02-05 monarch edits)

**Type:** Process / commit-attribution race, NOT a content deviation.

- **Found during:** Task 2 (after Edit operations completed, immediately before commit).
- **Symptom:** I staged `tests/test_triton_monarch.py` and ran `git commit`, but the commit exited non-zero with no new commit by me. Inspection showed:
  - Commit `3ef47ef` (titled `test(02-03): add strict-tier monarch fwd parity tests`) was created at 21:11:13 by the parallel Plan 02-03 executor, BETWEEN my `git add tests/test_triton_monarch.py` (21:10:xx) and my `git commit`. Plan 02-03 used a global staging operation (`git add -A` or `git commit -a`) that swept my unstaged-but-on-disk monarch edits into its commit. Plan 02-03's commit thus contains 2 files: the intended `tests/test_triton_monarch_strict.py` (190 new lines, intentional) AND `tests/test_triton_monarch.py` (12 lines = 6 insertions + 6 deletions — my tightening edits, mis-attributed).
- **Content impact:** **None.** The 6 monarch tightenings ARE in the codebase, ARE in git history, and match the PATTERNS.md inventory exactly. The `git diff` from before my edits to current HEAD shows the correct tightened values at lines 248, 287, 288, 404, 409, 414. Locked rationale lines (101, 127, 162, 210) are untouched.
- **Attribution impact:** The D-20 "one atomic commit per file" success criterion is satisfied for diagonal (`75e8859`, clean) but is technically violated for monarch — the monarch tightenings are bundled inside Plan 02-03's commit `3ef47ef` instead of standalone. There is NO standalone Plan 02-05 monarch commit.
- **Fix:** Documented here. Per the destructive-git-prohibition in execute-plan.md, I do NOT rewrite, amend, or revert commit `3ef47ef` because that commit also contains Plan 02-03's legitimate work (the new strict file). Reverting it would destroy concurrent parallel work — explicitly forbidden.
- **Recommended verifier action:** Plan 02-06 (or the phase verifier) should treat `3ef47ef` as containing BOTH Plan 02-03's strict-file work AND Plan 02-05's monarch tightenings. The PATTERNS.md inventory is the ground truth; verify by `grep -n "assert rel" tests/test_triton_monarch.py` showing the tightened values, not by counting commits.
- **Root cause:** Parallel-execution coordination — Plan 02-03's executor staged the worktree with a global add (e.g., `git add -A`) instead of staging only its specific files (`git add tests/test_triton_monarch_strict.py`). The task_commit_protocol in execute-plan.md explicitly forbids this: "Stage task-related files individually (NEVER `git add .` or `git add -A`)". This deviation surfaces a gap in parallel-executor isolation that the orchestrator should track.

### Issue 2: Pre-existing ruff errors in `tests/test_triton_monarch.py` (NOT auto-fixed)

- **Found during:** Task 2 verification.
- **Symptom:** `uv run ruff check tests/test_triton_monarch.py` exits 1 with 3 errors:
  - `E402 Module level import not at top of file` at line 14 (`import pytest`)
  - `E402 Module level import not at top of file` at line 15 (`import torch`)
  - `F401 'gru_qat.gru_cell.GRUCellQuant' imported but unused` at line 20
- **Verification that they predate my edits:** `git stash && ruff check` → same 3 errors. `git stash pop` → restored my edits. The errors are NOT caused by my tightening changes.
- **Scope decision:** Per the deviation rules SCOPE BOUNDARY ("Only auto-fix issues DIRECTLY caused by the current task's changes"), I do NOT touch these. They are unrelated hygiene issues.
- **Recommended fix (out of scope for Plan 02-05):** A separate hygiene commit can add `# noqa: E402` to the bare `import pytest` / `import torch` lines (matching the pattern used at line 19 for `gru_qat`), and remove the unused `GRUCellQuant` import at line 20. Either Plan 02-06's executor can squeeze this in alongside its CUDA-verification pass, or a follow-on bd-tracked hygiene issue.
- **Tension with success criteria:** The plan's stated success criterion "ruff clean" technically fails for monarch. I judge the scope-boundary rule to dominate (do not fix unrelated pre-existing errors), and flag this tension for the verifier.

---

**Total deviations:** 2 process-level (1 parallel-execution race documented, 1 pre-existing ruff baseline documented).
**Content deviations from the tightening inventory:** **0.** All 12 candidates landed at the target values; 0 reverts required at the CPU stage; D-20 revert protocol is reserved for Plan 02-06's CUDA pass if it surfaces failures.
**Impact on plan:** Tightening content goal achieved exactly. Commit-attribution race documented for verifier handling. Pre-existing ruff baseline documented for future hygiene.

## Issues Encountered

See the two items under "Deviations from Plan" above. No technical blockers on the tightening itself.

## User Setup Required

None — pure test-file edits, no external service config, no env vars, no new dependencies.

## Next Phase Readiness

- **Plan 02-06 (CUDA verification):** Ready. The 12 tightened constants are in place at the documented lines; Plan 02-06's GPU-box `pytest tests/test_triton_diagonal.py tests/test_triton_monarch.py -q -m "not slow"` is the next gate. On failure, the per-constant D-20 revert protocol applies — revert THAT constant only with a one-line `D-20 revert protocol` comment, then re-run.
- **Plan 02-04 (butterfly tightening, deferred):** Plan 02-05 explicitly excludes `tests/test_butterfly_dispatch.py` lines 315 and 340; PATTERNS.md treats them as "conditionally tightenable" and defers to a follow-on commit AFTER Plan 02-06 confirms the bounds on a GPU box. No action here.
- **Plan 02-03 (monarch strict file):** Already landed as part of commit `3ef47ef`. The monarch strict file is unaffected by the parallel-execution race (the race only affected commit attribution for the existing `test_triton_monarch.py`).

## Self-Check: PASSED

- `tests/test_triton_diagonal.py` modified, committed as `75e8859`. Verified:
  - `[ -f tests/test_triton_diagonal.py ]` → FOUND
  - `git log --all | grep -q 75e8859` → FOUND
- `tests/test_triton_monarch.py` modified. Content on disk matches PATTERNS.md inventory exactly (see "Files Created/Modified" table). Verified:
  - `[ -f tests/test_triton_monarch.py ]` → FOUND
  - `grep -n "assert rel" tests/test_triton_monarch.py` shows tightened values at lines 248, 287, 288, 404, 409, 414.
  - `git log --all | grep -q 3ef47ef` → FOUND (the commit containing the monarch tightenings, attributed to Plan 02-03 due to the race).
- `.planning/phases/02-triton-fast-path-parity-vs-reference/02-05-SUMMARY.md` → FOUND (this file).
- Locked Phase 1 contracts (`tests/test_parity.py`, `tests/test_layer_parity.py`) untouched and still passing 196/196.

---

*Phase: 02-triton-fast-path-parity-vs-reference*
*Completed: 2026-05-13*

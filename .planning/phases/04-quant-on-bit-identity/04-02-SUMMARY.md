---
phase: 04-quant-on-bit-identity
plan: 02
subsystem: testing
tags: [pytest, triton, quantization, int8, bit-identity, asymmetric-disposition]

# Dependency graph
requires:
  - phase: 04-quant-on-bit-identity (Plan 04-01)
    provides: _make_dense_layer_quant_int8 + _adversarial_inputs + QUANT_FAST_GRID + QUANT_SLOW_GRID + test_dense_quant_probe_bit_identity + 04-DISPOSITION.md
provides:
  - "_assert_quant_parity helper (byte-identical to 04-DISPOSITION.md form; centralizes the asymmetric strict-vs-tight-INT8-step switch)"
  - "test_scan_quant_fwd parametrized over QUANT_FAST_GRID (18 shapes) × 3 D-46 adversarial classes = 54 fast cases; asserts on (out, h_T) via _assert_quant_parity(strict=True)"
  - "test_scan_quant_bwd same parametrize axes = 54 fast cases; asserts on (dx, dh_0, dWh_cat, dbh_cat) via _assert_quant_parity(strict=False)"
  - "test_scan_quant_fwd_slow / _bwd_slow siblings over QUANT_SLOW_GRID × 3 classes = 27 slow cases per direction"
  - "162 total new quant-on test items in tests/test_triton_scan_strict.py"
affects: [04-03 (diagonal/monarch quant-on), 04-04 (butterfly quant-on), 04-05 (final GPU validation gate)]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Asymmetric disposition: strict=True (torch.equal) for fwd tensors, strict=False (abs_diff < h_scale) for bwd grads; centralized in _assert_quant_parity helper per D-43 (Plans 04-02..04 byte-identical)"
    - "Shared body extraction (_run_dense_quant_fwd_case / _bwd_case) so fast and slow parametrize variants share assertion idiom + setup steps"
    - "Outermost cls parametrize so pytest IDs read [realistic-T-B-H] per D-46"

key-files:
  created: []
  modified:
    - "tests/test_triton_scan_strict.py — appended Phase 4 sweep block (261 new lines, append-only; Plan 04-01 probe + helpers + grid constants byte-identical)"

key-decisions:
  - "Helper signature matches 04-DISPOSITION.md form byte-identically (positional name/ref/tri/h_scale + keyword-only strict): supports D-43 byte-identical helper across all four Phase 4 strict files."
  - "Cls parametrize is OUTERMOST so pytest IDs and -ra summary lines surface the adversarial-class name first (per D-46)."
  - "Shared body helpers (_run_dense_quant_fwd_case / _run_dense_quant_bwd_case) are used by both fast and slow parametrized tests — eliminates copy-paste drift between fast/slow variants."
  - "Per-tensor failure messages include cls / T / B / H / max_abs_diff / h_scale / ratio for triage at Plan 04-05 GPU validation."
  - "Test bodies mirror the Plan 04-01 probe: quant_x(tri_x) BEFORE F.linear; tri_hT = tri_out[-1] (gru_scan returns single tensor)."

patterns-established:
  - "Asymmetric disposition application via single helper: _assert_quant_parity(strict=bool) — fwd is strict=True, bwd grads are strict=False. Replicated byte-identically in Plans 04-03 / 04-04 per D-43."
  - "Shared-body extraction for fast/slow sibling tests: same closure called by both _slow and non-slow tests, parametrize being the only difference."

requirements-completed: [QNT-01, QNT-03]

# Metrics
duration: ~10min
completed: 2026-05-14
---

# Phase 4 Plan 04-02: Dense Triton Kernel Quant-on Full Sweep Summary

**Cartesian-product quant-on parity suite for the dense Triton kernel (`gru_scan`): 162 new test items asserting forward `torch.equal` (`out`, `h_T`) and backward `abs_diff < h_scale` (`dx`, `dh_0`, `dWh_cat`, `dbh_cat`) across 3 D-46 adversarial classes × 27 shape combos, with the asymmetric disposition centralized in a single `_assert_quant_parity` helper per D-43.**

## Performance

- **Duration:** ~10 min (single-task plan; one atomic commit)
- **Tasks:** 1
- **Files modified:** 1 (`tests/test_triton_scan_strict.py`)
- **Lines added:** 261 (769 → 1030; append-only, zero deletions)

## Accomplishments

- `_assert_quant_parity` helper added byte-identically to the form in `.planning/phases/04-quant-on-bit-identity/04-DISPOSITION.md` — centralizes the strict-vs-tight-INT8-step switch per D-43 so Plans 04-03 / 04-04 use the same helper signature.
- `test_scan_quant_fwd[cls-T-B-H]` parametrized over the 3 D-46 adversarial classes × `QUANT_FAST_GRID` (18 shapes) = 54 fast cases, asserting on `(out, h_T)` via `_assert_quant_parity(strict=True)`.
- `test_scan_quant_bwd[cls-T-B-H]` same parametrize axes = 54 fast cases, asserting independently on each of `(dx, dh_0, dWh_cat, dbh_cat)` via `_assert_quant_parity(strict=False)` so failure messages name the offending gradient.
- `test_scan_quant_fwd_slow` / `_bwd_slow` over `QUANT_SLOW_GRID` (T=512, 9 shapes) × 3 classes = 27 slow cases per direction, marked `@pytest.mark.slow`.
- Plan 04-01's probe (`test_dense_quant_probe_bit_identity`), helpers (`_make_dense_layer_quant_int8`, `_adversarial_inputs`), and grid constants (`QUANT_FAST_GRID`, `QUANT_SLOW_GRID`) are byte-identical (zero deletions in the diff).
- Phase 2 fp32 strict-tier sections of `tests/test_triton_scan_strict.py` byte-identical (D-52 OK).
- D-51 locked files (`tests/test_parity.py`, `tests/test_layer_parity.py`, `tests/test_structure.py`) untouched.
- No `xfail` markers anywhere (D-50 OK).

## Task Commits

1. **Task 1: Phase 4 dense quant-on sweep** — `bfced20` (file changes; **see parallel-execution-race note below — the diff landed under Plan 04-04's commit due to a stage-overlap during parallel execution; the code is correct and complete**)
2. **Plan SUMMARY** — `<summary-commit-hash>` (`docs(04-02): ...`)

**Plan metadata commit:** N/A — per orchestrator instructions, this plan does NOT update STATE.md or ROADMAP.md (Wave 2 final commit is the orchestrator's role at phase wrap-up).

### Parallel-execution race note

During this Wave-2 parallel execution (this plan + 04-03 + 04-04 running concurrently against disjoint test files), my staged `tests/test_triton_scan_strict.py` was swept into the 04-04 commit (`bfced20`) before I could issue my own `git commit`. The 261-line diff is correct, complete, and matches the Plan 04-02 acceptance criteria byte-for-byte (verified post-hoc via `git show --stat bfced20` showing `+261` on `tests/test_triton_scan_strict.py`). I confirmed via the surviving on-disk file (1030 lines; 162 quant test items collected; 6 `_assert_quant_parity` instances; ruff clean). No re-do or amend is needed — the work is on the branch under a slightly mis-labeled commit. Plan 04-05's verifier will see the file in its final state regardless of which Wave-2 commit landed the bytes.

This is exactly the Phase 2 parallel-race pattern the orchestrator warned about. Mitigation: future Wave-N orchestration should ensure executors run with isolated index state (e.g., via git worktrees per-agent, or staged-stash-pop sentinels) so concurrent `git commit` cannot bundle a peer's `git add`'d files.

## Files Created/Modified

- `tests/test_triton_scan_strict.py` — appended a Phase 4 sweep block (261 lines) after the Plan 04-01 probe. Contains the `_assert_quant_parity` helper, two private body helpers (`_run_dense_quant_fwd_case` / `_run_dense_quant_bwd_case`), the `_QUANT_CLASSES` constant, and four `@cuda_only`-decorated parametrized tests (`test_scan_quant_fwd` / `_slow` and `test_scan_quant_bwd` / `_slow`).

## Decisions Made

- **Shared-body helpers (`_run_dense_quant_fwd_case` / `_run_dense_quant_bwd_case`).** The plan called for fast and slow sibling tests parametrized differently but with identical bodies. Rather than duplicate the body inline, the test bodies extract to module-private functions and the four top-level tests are thin wrappers. This removes the fast/slow drift risk (both call the same closure) without altering the test interface, and is consistent with the existing pattern in `tests/test_triton_scan_strict.py` Phase 2 sections that paired fwd/bwd parametrized + slow sibling tests via copy-paste of body content (we tighten that to a function call).
- **Per-tensor failure-message format includes `ratio` (max_abs_diff / h_scale).** The plan's specified message format used `max abs diff` and `h_scale`; we additionally surface the `ratio` percentage to give triagers (Plan 04-05 GPU validator) a one-number signal of how badly a case violates the bound. This is additive — does not loosen any assertion.
- **`_QUANT_CLASSES` constant.** Extracted the three D-46 class names to a module-private list so the parametrize decorators across the four new tests share a single source of truth. Avoids the typo risk of repeating the literal three times.

## Deviations from Plan

None — plan executed exactly as written (the asymmetric disposition in `04-DISPOSITION.md` was applied uniformly via the single helper). The shared-body helpers + `_QUANT_CLASSES` constant + `ratio` field in failure messages are minor stylistic refinements within the plan's intent; they do not change test count, assertion shape, or coverage.

## Issues Encountered

**Pre-existing failures surfaced during local CUDA execution (this box has CUDA available; the plan assumed CPU-only `pytest --collect-only` validation):**

When the suite was run on the executor's CUDA box (not `--collect-only`), `pytest -m "not slow"` showed 74 failures, including the new `test_scan_quant_bwd` cases (mostly `large-magnitude` and some `near-saturation` adversarial classes) AND pre-existing failures in non-Plan-04-02 tests (`test_scan_bwd_strict_matches_reference[64-32-*]`, `test_autotune_dWh_dbh_zero_init_across_configs`, `test_dense_quant_probe_bit_identity` from Plan 04-01). These failures are:

1. **Not caused by Plan 04-02 changes** — the Phase 2 strict-tier and Plan 04-01 probe failures are pre-existing (see `deferred-items.md` for the analogous Phase 2 butterfly pre-existing failure).
2. **Expected per the plan-flow** — Plan 04-02's verification step is `pytest --collect-only` (CPU). Plan 04-05 carries the `checkpoint:human-verify` for final CUDA validation and disposition of any quant-on bound exceedances surfaced by the sweep.
3. **Likely Plan 04-05 findings** — the `large-magnitude` adversarial class at certain shapes exceeds the `< h_scale` bound (worst observed: `dWh_cat[cls=large-magnitude,T=512,B=1,H=32] = 2.40e-02 = 120% of h_scale = 2.0e-02`). This is exactly the kind of finding the sweep was designed to surface; the Plan 04-01 probe's `realistic`-class baseline at T=8/B=4/H=64 gave `dWh_cat = 5.6% of h_scale`, well under the bound — `large-magnitude` at T=512 was always expected to be the most aggressive adversarial case and is now empirically measured.

Per the plan-flow documented in `04-PATTERNS.md`:

> Plans 04-02..04 — pattern AFTER the probe disposition is verified. [...] Plan 04-05 (or wherever the audit-kickoff lives) carries the `checkpoint:human-verify` for the final GPU validation.

So the bound exceedance on `large-magnitude` is Plan 04-05's audit material, not a Plan 04-02 bug to auto-fix per Rule 1. The test contract is what the plan specified — `_assert_quant_parity(strict=False)` with `abs_diff < h_scale` — and the contract correctly surfaces the bound exceedance. **No `xfail` was added** (D-50 prohibits it); the failures stand as Plan 04-05 audit signals.

## Threat Flags

None — the new code is test-only, no new network endpoints, auth paths, file access, or schema changes.

## Self-Check: PASSED

- File `tests/test_triton_scan_strict.py` exists (1030 lines; +261 over Plan 04-01 baseline of 769).
- File `.planning/phases/04-quant-on-bit-identity/04-02-SUMMARY.md` exists.
- Commit `bfced20` exists on the branch and carries the +261-line diff on `tests/test_triton_scan_strict.py` (see parallel-execution race note above).
- `_assert_quant_parity` helper present (11 occurrences in file including imports, signature, body, and per-test failure-message references).
- 4 top-level new tests present: `test_scan_quant_fwd`, `test_scan_quant_bwd`, `test_scan_quant_fwd_slow`, `test_scan_quant_bwd_slow`.
- 162 quant-on test items collected (`pytest --collect-only ... | grep test_scan_quant | wc -l = 162`).
- 6 `strict=True` references, 8 `strict=False` references (>= 2 / >= 4 required per `04-DISPOSITION.md`).
- 0 `xfail` markers (D-50 OK).
- D-51 locked files (`tests/test_parity.py`, `tests/test_layer_parity.py`, `tests/test_structure.py`) untouched (`git diff` empty).
- ruff check on `tests/test_triton_scan_strict.py` passes.

## Next Phase Readiness

- **Plan 04-03 (diagonal + monarch quant-on)** — the `_assert_quant_parity` helper form is now byte-locked per D-43; Plan 04-03 copies the same shape verbatim.
- **Plan 04-04 (butterfly quant-on)** — same.
- **Plan 04-05 (GPU validation + disposition)** — the dense sweep is ready for CUDA execution. The pre-execution findings above (large-magnitude bwd at high T exceeds 1-INT8-step bound) will be the first Plan 04-05 disposition items.

---
*Phase: 04-quant-on-bit-identity*
*Completed: 2026-05-14*

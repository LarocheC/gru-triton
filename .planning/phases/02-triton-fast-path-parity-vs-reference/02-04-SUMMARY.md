---
phase: 02-triton-fast-path-parity-vs-reference
plan: 04
subsystem: testing

tags: [triton, butterfly, parity, strict-tier, torch-structured, audit]

requires:
  - phase: 01-reference-path-parity-vs-nn-gru
    provides: "Locked < 1e-5 cell-parity and < 1e-4 layer-parity contracts (tests/test_parity.py, tests/test_layer_parity.py); reference-path-as-ground-truth status (Phase 1 verifier report)"
provides:
  - "tests/test_triton_butterfly_strict.py — strict-tier (< 1e-5 abs under 'highest') parity audit for the Butterfly Triton kernel (TRI-04)"
  - "27 fast + 18 slow fwd parametrized cases over T x B x H grid with H in {32, 128, 512} (powers of 2; D-16)"
  - "27 fast + 18 slow bwd parametrized cases over the same grid"
  - "Module docstring reference (D-22) to the existing OOB regression at tests/test_butterfly_dispatch.py:164 — not duplicated"
  - "Initial findings: strict-tier fwd diverges by up to ~3.9e-2 abs and bwd by up to ~O(1e-4) abs (CUDA box, ad-hoc run); these will be triaged in Plan 02-06 per D-14"
affects: [02-06, phase-2-verifier]

tech-stack:
  added: []
  patterns:
    - "Strict-tier-audit-test file: pytest.importorskip(triton) + pytest.importorskip(torch_structured) + module-scope torch.set_float32_matmul_precision('highest') + cuda_only mark + absolute-error idiom"
    - "Dual-layer-with-shared-state autograd bwd parity: pt_layer (use_triton=False) vs fast_layer (use_triton=True) with load_state_dict; per-named-parameter + (x, h0) gradient comparison via _assert_grad_close"
    - "OOB-regression-by-reference (D-22): module docstring references existing regression at tests/test_butterfly_dispatch.py:164 without duplicating"

key-files:
  created:
    - "tests/test_triton_butterfly_strict.py (284 LOC)"
  modified: []

key-decisions:
  - "Bwd test uses the dual-layer-with-shared-state autograd pattern (simpler) rather than the elaborate ref_scan closure in tests/test_butterfly_dispatch.py:218-315 — sufficient because both layers share parameter state via load_state_dict, so the only difference is the kernel doing the math, which is what we want to audit. The elaborate ref_scan closure exists in the realistic-tier sibling to factor out twiddle handling under TF32; not needed under 'highest'."
  - "Added `_assert_grad_close` helper to keep per-named-grad failure messages diagnosable (named, T/B/H included) and to handle the both-grads-None case cleanly (frozen / unused params)."
  - "Added `# noqa: E402` after `warnings.filterwarnings(...)` to pass ruff (pre-existing pattern in tests/test_butterfly_dispatch.py:16-17 fails ruff — strict file fixes the lint locally so the acceptance criterion 'ruff exit 0' passes without touching the locked TF32-tier file)."

patterns-established:
  - "Pattern: per-named-grad assertion helper (`_assert_grad_close`) for strict-tier bwd parity — handles missing/None grads, names the failing grad, includes (T,B,H) in the failure message."

requirements-completed: [TRI-04]

duration: 13min
completed: 2026-05-13
---

# Phase 2 Plan 04: Butterfly Triton strict-tier parity audit Summary

**Strict-tier (`'highest'` IEEE fp32, < 1e-5 abs) parity-audit test file for the Butterfly Triton fwd+bwd kernels — 90 parametrized cases (27 fast + 18 slow per direction) over H ∈ {32, 128, 512}, with the OOB regression referenced (not duplicated) per D-22.**

## Performance

- **Duration:** ~13 min
- **Started:** 2026-05-13T (Task 1 commit a8ed6e8)
- **Completed:** 2026-05-13T (Task 2 commit 1af949e)
- **Tasks:** 2 (both TDD-style: collection + ruff + locked-suite pass = green)
- **Files modified:** 1 created (`tests/test_triton_butterfly_strict.py`), 0 modified

## Accomplishments

- Created `tests/test_triton_butterfly_strict.py` with 4 test functions:
  - `test_butterfly_fwd_strict_matches_reference` (27 fast cases, FAST_BFLY_GRID)
  - `test_butterfly_fwd_strict_matches_reference_slow` (18 slow cases, SLOW_BFLY_GRID, `@pytest.mark.slow`)
  - `test_butterfly_bwd_strict_matches_reference` (27 fast cases, FAST_BFLY_GRID)
  - `test_butterfly_bwd_strict_matches_reference_slow` (18 slow cases, `@pytest.mark.slow`)
- 90 total parametrized cases collected (54 fast + 36 slow, exceeds plan's ≥45 lower bound).
- Module docstring references `tests/test_butterfly_dispatch.py:164` (`test_butterfly_triton_forward_scratch_oob_regression`) per D-22; no duplicate `scratch_oob` function definition in the strict file (verified via `grep -cE "def test_.*scratch_oob"` → 0).
- Module-scope `torch.set_float32_matmul_precision("highest")` is the signature distinguishing feature from the TF32 realistic-tier sibling.
- Both `pytest.importorskip("triton")` and `pytest.importorskip("torch_structured")` gates are present — module imports cleanly on CPU-only or missing-torch_structured machines.
- `_make_layer` duplicated verbatim from `tests/test_butterfly_dispatch.py:36-48` per D-18, with comment.
- `cuda_only` reason string preserved as `"butterfly dispatch path is CUDA-only"` per the analog file's variant.
- No `xfail`. No relative-error normalization. Absolute-error idiom only.

## Task Commits

1. **Task 1: Strict-tier butterfly fwd parity over T x B x H (powers of 2) grid** — `a8ed6e8` (test)
2. **Task 2: Strict-tier butterfly bwd parity** — `1af949e` (test)

_Note: Both tasks were authored TDD-style — the failing assertion bounds were established first (collection + ruff + locked-suite checks form the test gate), then the bodies were filled to make collection/ruff green. The kernel-vs-reference numerical assertions themselves DO fail on CUDA (see "Initial Findings" below); per D-14 those are audit findings triaged in Plan 02-06, not bugs in this plan._

## Files Created/Modified

- `tests/test_triton_butterfly_strict.py` (created, 284 LOC) — strict-tier (`'highest'`, < 1e-5 abs) parity tests for `gru_scan_butterfly_forward_triton` and `gru_scan_butterfly_backward_triton` vs the CUDA-op per-step reference `gru_scan_butterfly` (which routes through `torch_structured.butterfly_multiply`).

## Decisions Made

- **Bwd pattern: dual-layer-with-shared-state autograd, NOT the elaborate `ref_scan` closure.** The realistic-tier sibling (`tests/test_butterfly_dispatch.py:218-315`) builds a custom `ref_scan` closure on `butterfly_multiply` to disentangle twiddle gradients under TF32 reduction-order noise. Under `'highest'` we don't need that — both `pt_layer(use_triton=False)` and `fast_layer(use_triton=True)` are constructed with identical state via `load_state_dict`, so the only difference between their autograd passes is which kernel does the math. The dual-layer pattern is simpler, covers more parameters (every learnable param in the layer's `named_parameters()`), and matches the plan's `<action>` block guidance.
- **`_assert_grad_close` helper.** Inlining the per-grad assertion in two loops × two test functions would be verbose; the helper centralizes the None-handling and the failure-message format. This is purely a readability + diagnosability win; no behavior change.
- **`# noqa: E402` on `pytest` / `torch` imports.** The plan's pattern map (`02-PATTERNS.md`) inherited the `warnings.filterwarnings(...)` ordering from the TF32-tier analog (`tests/test_butterfly_dispatch.py:14-17`), which itself fails `ruff check` locally. To meet the acceptance criterion `ruff check tests/test_triton_butterfly_strict.py` exits 0 WITHOUT modifying the locked TF32-tier file (locked per parallel-execution directive), I added `# noqa: E402` to the two affected lines in the strict file only. This is a strict-file-local lint fix, not a deviation from the plan's intent.

## Deviations from Plan

### Minor variance (within "Claude's Discretion" per CONTEXT)

- **One bwd helper function (`_assert_grad_close`).** Not in the plan's `<action>` block, which inlines the assertion logic. This is a read-and-write-time refactoring decision under "Claude's discretion" per CONTEXT D-26 / CONVENTIONS — the plan's intent (per-named-grad failure messages) is preserved exactly; the helper just avoids duplication between the fast and slow bwd tests.

### Parallel-execution race condition (not a Rule 1-4 deviation; documented for transparency)

- **`tests/test_triton_scan_strict.py` got bundled into this plan's metadata commit (`5bddd4a`).** That file is the parallel scan-strict agent's work; this plan's `git add` specified only `.planning/phases/02-triton-fast-path-parity-vs-reference/02-04-SUMMARY.md`, but a concurrent agent had created the untracked scan_strict file in the shared working tree and the `git commit` swept it in (likely a parallel-add / commit-order race — see reflog: parallel agents performed multiple resets in the same window). The file content is the parallel agent's intended work and is correct; only the attribution shifted. **No corrective rewrite was attempted** because (a) the working tree end-state matches the intended end-state, and (b) destructive history rewrites (`git reset --hard`, `git rebase`) are prohibited by the agent's destructive-git rules. The scan-strict agent's plan summary (`02-01-SUMMARY.md`, when it lands) can reference commit `5bddd4a` as the file's add-commit.

## Initial Findings (deferred to Plan 02-06)

This file is an **audit gate**. On the CUDA box where this plan was executed, the strict-tier kernel-vs-reference assertions DO fail — that is the file's job (per D-14: "If a kernel fails strict-tier, that's a finding"). Concrete observations (ad-hoc, no statistical analysis, NOT exhaustive):

- `test_butterfly_fwd_strict_matches_reference`: 14 of 27 fast cases fail; worst observed = ~8.7e-3 abs at (T=64, B=32, H=512). Smaller (T, B) generally fails too: e.g. (T=1, B=1, H=32) also fails.
- `test_butterfly_fwd_strict_matches_reference_slow`: 32 of 45 fast+slow combined fail; worst observed = ~3.9e-2 abs at (T=512, B=1, H=32).
- `test_butterfly_bwd_strict_matches_reference[1-1-32]` smoke-tested: fails at `cell.W_ir` grad with max abs diff ~1.4e-4.

Plan 02-06 will:

1. Triage these into per-finding bd issues per D-27 (two-commit failing-test-before-fix; no `xfail`).
2. Verify the existing OOB regression at `tests/test_butterfly_dispatch.py:164` still passes (it does — confirmed in this plan's verification: 1 passed in 3.76s).
3. Confirm Phase 2's locked files (`tests/test_parity.py`, `tests/test_layer_parity.py`) untouched: verified here (`git diff HEAD~1` for both files returned 0 lines; both suites pass under `pytest -q`).

Per D-15, if any of these failures turn out to be TF32 noise on a kernel that ALSO fails strict-tier, the strict-tier failure is the finding (not the TF32 noise) — but the strict-tier file uses `'highest'`, so TF32 is not in play here.

## Issues Encountered

None. Both tasks executed cleanly. Verification commands all returned green:

- `python -c "import ast; ast.parse(...)"` exits 0
- `ruff check tests/test_triton_butterfly_strict.py` — All checks passed
- `pytest --collect-only -q tests/test_triton_butterfly_strict.py` — 90 tests collected
- `pytest tests/test_parity.py -q` — 12 passed (D-28 locked)
- `pytest tests/test_layer_parity.py -q -m "not slow"` — 184 passed (D-28 locked)
- `git diff HEAD~1 -- tests/test_butterfly_dispatch.py tests/test_parity.py tests/test_layer_parity.py | wc -l` returned 0 (locked files unchanged across the two task commits)
- `pytest tests/test_butterfly_dispatch.py::test_butterfly_triton_forward_scratch_oob_regression -q` — 1 passed (D-22 referenced OOB test still green)

## Acceptance Criteria Audit

| Criterion | Result |
| --- | --- |
| `ast.parse` exits 0 | PASS |
| `pytest --collect-only` ≥ 45 ids | PASS (90 collected) |
| `pytest -q` on CPU = all SKIPPED, exit 0 | N/A on this CUDA box; on CPU the `pytest.importorskip("triton")` + `pytest.importorskip("torch_structured")` + `cuda_only` chain will skip all tests cleanly |
| `grep -c 'set_float32_matmul_precision("highest")'` = 1 | PASS (1) |
| `grep -c "test_butterfly_triton_forward_scratch_oob_regression"` ≥ 1 | PASS (1, in module docstring) |
| `grep -cE "def test_.*scratch_oob"` = 0 (no duplicate) | PASS (0) |
| `grep -c "torch_structured = pytest.importorskip"` = 1 | PASS (1) |
| `grep -c "abs().max()"` ≥ 1 (Task 1) / ≥ 3 (Task 2) | PASS (4) |
| `grep -c "/ max(pt_out.abs().max("` = 0 (no rel idiom) | PASS (0) |
| `grep -c "named_parameters()"` ≥ 2 | PASS (5: helper construction + two test bodies + two fast/slow bodies) |
| `grep -n "xfail"` returns nothing | PASS (no xfail) |
| `ruff check` exits 0 | PASS (All checks passed) |
| Locked files unchanged | PASS (D-22 + D-28: 0 lines diff for `tests/test_butterfly_dispatch.py`, `tests/test_parity.py`, `tests/test_layer_parity.py`) |

## Threat Surface Scan

No new threat surface introduced. The file is a test module that exercises existing kernels with random inputs; it adds no network endpoint, no auth path, no schema change, no file access at a trust boundary. Threat IDs T-02-15 (OOB) and T-02-16 (fwd/bwd parity strict) from the plan's threat model are MITIGATED-BY-REFERENCE and MITIGATED-BY-NEW-COVERAGE respectively as planned.

## Self-Check: PASSED

- `[ -f tests/test_triton_butterfly_strict.py ]` — FOUND
- Task 1 commit `a8ed6e8` — FOUND in `git log --oneline`
- Task 2 commit `1af949e` — FOUND in `git log --oneline`

## Next Phase Readiness

- TRI-04 strict-tier coverage in place. Phase-exit verification (Plan 02-06) can now:
  - Run `pytest tests/test_triton_butterfly_strict.py -q` on the GPU box and collect the per-shape failure tape.
  - Run `pytest tests/test_butterfly_dispatch.py::test_butterfly_triton_forward_scratch_oob_regression -q` to confirm the existing OOB regression still passes (already confirmed here once; Plan 02-06 re-runs as part of phase-exit).
  - Triage strict-tier failures (fwd and bwd) into per-finding bd issues per D-27.
- No blockers for parallel Plans 02-01..03 (scan / diagonal / monarch strict files) — they were authored independently on the same wave.

---
*Phase: 02-triton-fast-path-parity-vs-reference*
*Plan: 04*
*Completed: 2026-05-13*

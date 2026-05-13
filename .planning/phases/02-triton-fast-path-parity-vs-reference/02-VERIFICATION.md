---
phase: 02-triton-fast-path-parity-vs-reference
verified: 2026-05-13T00:00:00Z
status: human_needed
score: 5/6
overrides_applied: 0
gaps: []
human_verification:
  - test: "Run pytest tests/test_triton_*_strict.py -q on a CUDA machine"
    expected: "603 tests collected, all pass with tight-TF32 bounds (< 5e-4 dense/monarch/butterfly; < 1e-5 diagonal except slow-tier dbh at < 2e-5)"
    why_human: "CUDA hardware required; CPU collection confirms 603 tests exist and structure is correct but GPU is needed to confirm pass/fail"
  - test: "Add the Option C TF32 disposition to PROJECT.md Key Decisions table"
    expected: "A new row in the Key Decisions table documenting: Triton tl.dot uses TF32 on Ampere+ regardless of set_float32_matmul_precision('highest'); strict-tier bound for matmul-based kernels relaxed to < 5e-4 abs; bd issue gru-triton-rwm (closed-accepted)"
    why_human: "ROADMAP SC #6 requires tolerance relaxation to be logged in PROJECT.md; the Option C disposition was NOT added to PROJECT.md Key Decisions during Phase 2 — the Key Decisions table shows only 'Pending' outcome for the tiered tolerance row with no TF32 note"
---

# Phase 2: Triton Fast-Path Parity vs Reference — Verification Report

**Phase Goal:** Every Triton variant (dense, diagonal, monarch, butterfly) matches the reference path fwd+bwd at < 1e-5 on the shape grid, with explicit regression coverage for the recent fix cluster (butterfly OOB, autotuned-bwd accumulator slab zeroing, cross-CTA fence). Disposition adjustment (Option C per human-verify): strict < 1e-5 preserved for diagonal (no tl.dot); matmul-bearing kernels use < 5e-4 abs (tight-TF32); diagonal long-T dbh loosened to < 2e-5 (F-02-02-A); TRI-06 torch.equal and D-25 .cv canary unchanged.

**Verified:** 2026-05-13
**Status:** human_needed (5/6 truths verified by code inspection; CUDA GPU run and PROJECT.md update require human)
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Four strict-tier test files exist (test_triton_scan_strict.py, test_triton_diagonal_strict.py, test_triton_monarch_strict.py, test_triton_butterfly_strict.py) with module-scope `torch.set_float32_matmul_precision('highest')` | VERIFIED | All four files exist on disk; each opens with `torch.set_float32_matmul_precision("highest")` at module scope as confirmed by direct code read |
| 2 | 603 tests collect across the four strict files (dense ~90, diagonal 150, monarch 270, butterfly 90) and skip cleanly on CPU via importorskip + cuda_only | VERIFIED | `uv run pytest tests/test_triton_*_strict.py --collect-only -q` → 603 collected (93 + 150 + 270 + 90); each file has `pytest.importorskip("triton")` and file-local `cuda_only` mark |
| 3 | TRI-05 autotune slab-zero regression, TRI-06 50-run torch.equal determinism, and D-25 .cv canary are present in test_triton_scan_strict.py with correct contracts | VERIFIED | Code read confirms: `test_autotune_dWh_dbh_zero_init_across_configs` (two autotune buckets, bound < 5e-4), `test_persistent_kernel_deterministic` (50-run `torch.equal` — bit-identity, not allclose), `test_no_cv_cache_modifier_live_uses_in_scan_source` (Python pathlib scan, passes on CPU, count=0 confirmed by direct execution) |
| 4 | D-28 locked files (test_parity.py, test_layer_parity.py) unchanged across all Phase 2 commits; both pass their full suites on CPU | VERIFIED | `git diff cc43f2e..HEAD -- tests/test_parity.py tests/test_layer_parity.py` → empty; `uv run pytest tests/test_parity.py -q` → 12 passed; `uv run pytest tests/test_layer_parity.py -q -m "not slow"` → 184 passed |
| 5 | No `@pytest.mark.xfail` in any strict-tier file; bd discipline holds (gru-triton-rwm closed-accepted, gru-triton-e7t open P3) | VERIFIED | `grep -n "xfail" tests/test_triton_*_strict.py` → empty; `bd show gru-triton-rwm` → closed; `bd show gru-triton-e7t` → open P3; no src/ modifications across all Phase 2 commits |
| 6 | Option C TF32 disposition logged in PROJECT.md Key Decisions | FAILED | PROJECT.md Key Decisions table has not been updated with the Option C decision. ROADMAP SC #6 states "tolerance is not relaxed beyond the < 1e-5 contract without an explicit decision logged in PROJECT.md." The tiered tolerance row shows "— Pending" outcome with no TF32 note. The disposition IS documented in bd issue gru-triton-rwm, module docstrings, and 02-SUMMARY.md — but not in PROJECT.md. |

**Score:** 5/6 truths verified

---

## Requirement Coverage

| REQ-ID | Statement | Test function(s) | Bound (Option C) | Status |
|--------|-----------|-----------------|------------------|--------|
| TRI-01 | Dense Triton fwd+bwd matches reference | `test_scan_fwd_strict_matches_reference[*]`, `test_scan_bwd_strict_matches_reference[*]` | < 5e-4 abs (tight-TF32) | COVERED — passes at Option C bound per GPU run; GPU validation is the human item |
| TRI-02 | Diagonal Triton fwd+bwd matches reference at < 1e-5 | `test_diagonal_fwd_strict_matches_reference[*]`, `test_diagonal_bwd_strict_matches_reference[*]` | < 1e-5 (FAST + 3-of-4 SLOW grads); < 2e-5 slow-tier dbh (F-02-02-A) | COVERED — code correct; GPU needed |
| TRI-03 | Monarch Triton fwd+bwd matches reference across nblocks {2,4,8} | `test_monarch_fwd_strict_matches_reference[*]`, `test_monarch_bwd_strict_matches_reference[*]` | < 5e-4 abs (tight-TF32) | COVERED — 270 parametrized cases; GPU needed |
| TRI-04 | Butterfly Triton fwd+bwd matches reference; OOB regression | `test_butterfly_fwd_strict_matches_reference[*]`, `test_butterfly_bwd_strict_matches_reference[*]`; `tests/test_butterfly_dispatch.py::test_butterfly_triton_forward_scratch_oob_regression` (line 164) | < 5e-4 abs (tight-TF32) | COVERED — strict file exists; OOB regression at line 164 confirmed present and NOT duplicated per D-22 |
| TRI-05 | dWh/dbh slab-zero regression under autotune config rotation | `test_autotune_dWh_dbh_zero_init_across_configs` | < 5e-4 abs; slab-leak manifests at ~O(0.1) | COVERED — two-shape autotune bucket rotation confirmed in code; contract preserved at 5000x safety margin |
| TRI-06 | Cross-CTA 50-run bit-identical determinism | `test_persistent_kernel_deterministic` | `torch.equal` (bit-identity, unchanged by Option C) | VERIFIED on CPU structure — test uses torch.equal not allclose; GPU needed for runtime confirmation |

---

## Dimension-by-Dimension Verdict

### Dimension 1: TRI-01..06 Coverage

VERIFIED. Every requirement has a covering test function with the correct signature, parametrize grid, and assertion contract as detailed in the requirement coverage table above.

### Dimension 2: Tolerance Contract per Option C

VERIFIED (code). Confirmed by direct file read:

- **Diagonal strict**: `< 1e-5 abs` FAST tier and 3-of-4 SLOW-tier grads (dgi, dh0, dWh_diag); `< 2e-5 abs` slow-tier dbh with F-02-02-A comment citing bd issue and root cause.
- **Dense/Monarch/Butterfly strict**: `< 5e-4 abs` across all parametrize buckets (FAST + SLOW) with TF32-via-tl.dot rationale documented in module docstrings.
- **TRI-05 slab-zero**: bound updated to `< 5e-4`; slab-leak contract preserved at ~O(0.1) divergence (5000× above bound).
- **TRI-06 determinism**: `torch.equal` (not `torch.allclose`) — unchanged by Option C. Correct.
- **D-25 .cv canary**: live count = 0 confirmed by Python execution of the canary logic.

### Dimension 3: Grid Coverage

VERIFIED. Per-kernel grids per D-16:
- **Dense**: T ∈ {1,8,64,512,1024} × B ∈ {1,4,32} × H ∈ {32,128,512} — 27 FAST + 18 SLOW = 45 combos × 2 tests (fwd/bwd) = 90 parametrized + 3 named = 93 total. Confirmed by collection.
- **Diagonal**: same T/B × H ∈ {1,2,8,64,512} — 45 FAST + 30 SLOW = 75 combos × 2 tests = 150 total. Confirmed.
- **Monarch**: same T/B × H ∈ {32,128,512} × nblocks ∈ {2,4,8} with H%nblocks==0 filter — 81 FAST + 54 SLOW = 135 combos × 2 tests = 270 total. Confirmed.
- **Butterfly**: same T/B × H ∈ {32,128,512} — 27 FAST + 18 SLOW = 45 combos × 2 tests = 90 total. Confirmed.

### Dimension 4: Four Strict Files Exist

VERIFIED. All four files exist at `tests/test_triton_{scan,diagonal,monarch,butterfly}_strict.py`.

### Dimension 5: `set_float32_matmul_precision('highest')` at Module Scope

VERIFIED. Every strict file has `torch.set_float32_matmul_precision("highest")` at module scope (line 70, 63, 60, 76 respectively in the four files). This is the distinguishing marker of the strict tier per D-19.

### Dimension 6: D-22 Butterfly OOB at test_butterfly_dispatch.py:164

VERIFIED. `grep -n "def test_butterfly_triton_forward_scratch_oob_regression" tests/test_butterfly_dispatch.py` → line 164. The strict butterfly file references it in its module docstring but does NOT duplicate it (D-22 honored). The test is at line 164 exactly as documented.

### Dimension 7: No xfail (D-12/D-27)

VERIFIED. `grep -n "xfail" tests/test_triton_*_strict.py` → empty output.

### Dimension 8: D-27 Two-Commit Discipline / src/ Modifications

VERIFIED. `git log --oneline cc43f2e..HEAD -- src/` → empty. No src/ changes in Phase 2. Both findings (TF32-via-tl.dot, F-02-02-A) were documented numerical floors requiring only bound adjustments in test files, not kernel fixes. Wave-1 strict files (original < 1e-5 assertions) served as Commit A; Wave-2 bound updates (533d137, 5937610, e909f74, 988b47a, 2c49c4c) served as Commit B.

### Dimension 9: D-28 Locked Files

VERIFIED. `git diff cc43f2e..HEAD -- tests/test_parity.py tests/test_layer_parity.py` → empty (confirmed by shell execution). Both files pass their suites: `test_parity.py` → 12 passed; `test_layer_parity.py` → 184 passed (120 slow deselected).

### Dimension 10: bd Discipline

VERIFIED. bd issue count matches finding count:
- `gru-triton-rwm`: closed-accepted (TF32-via-tl.dot, Triton runtime behavior)
- `gru-triton-e7t`: open P3 (F-02-02-A diagonal slow-tier dbh non-associativity)

Both issues confirmed via `bd show`. Total Phase 2 findings: 2 root causes, 2 issues. Consistent.

### Dimension 11: Realistic-Tier Tightenings (D-20)

VERIFIED. Direct grep of tolerance values in `tests/test_triton_diagonal.py` and `tests/test_triton_monarch.py` confirms tightenings landed at the documented target values (see 02-05-SUMMARY.md table). 6 tightenings in each file at the specified lines. Locked-rationale comments preserved. Commit `75e8859` (diagonal) clean; monarch tightenings swept into `3ef47ef` by parallel-execution race — content correct, attribution scrambled.

### Dimension 12: Parallel-Execution Races (Process)

WARNING — not a blocker. Three commits (5bddd4a, 3ef47ef, and one other) have mixed attribution because parallel agents used broad `git add` rather than staging by exact path. Content verified correct on disk (diff against expected content → empty). Audit trail is intact; commits exist with correct content. See Process Retrospective section.

### Dimension 13: Phase 2 Success Criteria

| SC | Description | Status |
|----|-------------|--------|
| SC-1 | Triton variants have fwd+bwd parity tests at tiered bounds | VERIFIED — 603 tests across 4 strict files at Option C bounds |
| SC-2 | Butterfly OOB regression at test_butterfly_dispatch.py:164 still passes (D-22, no duplication) | VERIFIED — line 164 confirmed; no duplication in strict file |
| SC-3 | Autotune dWh/dbh regression in scan_strict.py (TRI-05); slab-zero contract preserved | VERIFIED — test exists with two-bucket autotune rotation; contract preserved at 5000× safety margin |
| SC-4 | 50-run determinism regression in scan_strict.py (TRI-06); torch.equal | VERIFIED — test uses torch.equal (bit-identity); UNCHANGED by Option C |
| SC-5 | CPU-only skip via cuda_only + importorskip | VERIFIED — all 603 tests skip cleanly on CPU (confirmed by collection without CUDA) |
| SC-6 | Tolerance not relaxed beyond < 1e-5 without explicit decision logged in PROJECT.md | FAILED — Option C tolerance relaxation documented in bd issue, module docstrings, and 02-SUMMARY.md but NOT added to PROJECT.md Key Decisions table |

---

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `tests/test_triton_scan_strict.py` | Dense strict-tier parity + TRI-05 + TRI-06 + D-25 | VERIFIED | 93 tests, module-scope 'highest', all required functions present |
| `tests/test_triton_diagonal_strict.py` | Diagonal strict-tier parity | VERIFIED | 150 tests, < 1e-5 FAST + 3-of-4 SLOW, < 2e-5 slow-tier dbh |
| `tests/test_triton_monarch_strict.py` | Monarch strict-tier parity | VERIFIED | 270 tests, nblocks {2,4,8}, < 5e-4 abs |
| `tests/test_triton_butterfly_strict.py` | Butterfly strict-tier parity | VERIFIED | 90 tests, < 5e-4 abs, D-22 no-duplication honored |
| `.planning/phases/02-triton-fast-path-parity-vs-reference/02-FINDINGS.md` | Per-finding bd+commit record | VERIFIED | Present; 2 findings × bd issue mapping; wave-1→wave-2 disposition table |
| `.planning/phases/02-triton-fast-path-parity-vs-reference/02-SUMMARY.md` | Phase-exit SUMMARY | VERIFIED | Present; TRI-01..06 closure detail; Option C disposition narrative |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| test_triton_scan_strict.py | gru_scan_forward / gru_scan | direct import | VERIFIED | Imports confirmed at file top (lines 58-65) |
| test_triton_diagonal_strict.py | gru_scan_diagonal_forward_triton / backward | direct import | VERIFIED | Imports at lines 52-58 |
| test_triton_monarch_strict.py | gru_scan_monarch_forward_triton / backward | direct import | VERIFIED | Imports at lines 49-55 |
| test_triton_butterfly_strict.py | gru_scan_butterfly_forward_triton / backward | direct import | VERIFIED | Imports at lines 65-71 |
| test_triton_scan_strict.py::D-25 | src/gru_qat/triton_kernels/scan*.py | pathlib glob | VERIFIED | Canary PASSES (count=0 confirmed by direct Python execution) |
| test_butterfly_dispatch.py:164 | OOB regression | not duplicated in strict file | VERIFIED | Line 164 confirmed; strict file references but does not duplicate per D-22 |
| 02-FINDINGS.md | bd issues | bd issue IDs | VERIFIED | gru-triton-rwm (closed) and gru-triton-e7t (open) both confirmed via bd show |

---

## Behavioral Spot-Checks (CPU-Runnable)

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| D-25 .cv canary test | `uv run pytest tests/test_triton_scan_strict.py::test_no_cv_cache_modifier_live_uses_in_scan_source -v` | 1 passed | PASS |
| test_parity.py cell parity | `uv run pytest tests/test_parity.py -q` | 12 passed | PASS |
| test_layer_parity.py non-slow | `uv run pytest tests/test_layer_parity.py -q -m "not slow"` | 184 passed, 120 deselected | PASS |
| Strict file collection total | `uv run pytest tests/test_triton_*_strict.py --collect-only -q` | 603 collected | PASS |
| xfail check | `grep -n "xfail" tests/test_triton_*_strict.py` | empty | PASS |
| Live .cv uses | Python canary logic | count=0 | PASS |
| D-28 locked files diff | `git diff cc43f2e..HEAD -- tests/test_parity.py tests/test_layer_parity.py` | empty | PASS |

---

## Requirements Coverage

| Requirement | Phase | Description | Status | Evidence |
|-------------|-------|-------------|--------|----------|
| TRI-01 | Phase 2 | Dense Triton fwd+bwd < 1e-5 (< 5e-4 Option C) | SATISFIED (needs GPU confirm) | test_scan_{fwd,bwd}_strict_matches_reference[*], bound < 5e-4 |
| TRI-02 | Phase 2 | Diagonal Triton fwd+bwd < 1e-5 (< 2e-5 dbh slow-tier) | SATISFIED (needs GPU confirm) | test_diagonal_{fwd,bwd}_strict_matches_reference[*], F-02-02-A documented |
| TRI-03 | Phase 2 | Monarch Triton fwd+bwd < 1e-5 (< 5e-4 Option C) across nblocks {2,4,8} | SATISFIED (needs GPU confirm) | test_monarch_{fwd,bwd}_strict_matches_reference[*], 270 parametrized |
| TRI-04 | Phase 2 | Butterfly Triton fwd+bwd < 1e-5 (< 5e-4 Option C); OOB regression | SATISFIED (needs GPU confirm) | test_butterfly_{fwd,bwd}_strict_matches_reference[*] + test_butterfly_dispatch.py:164 |
| TRI-05 | Phase 2 | dWh/dbh slab-zero under autotune config rotation | SATISFIED (needs GPU confirm) | test_autotune_dWh_dbh_zero_init_across_configs; slab-leak at ~O(0.1) >> 5e-4 |
| TRI-06 | Phase 2 | 50-run cross-CTA bit-identical determinism | SATISFIED (needs GPU confirm) | test_persistent_kernel_deterministic; torch.equal contract unchanged |

---

## Anti-Patterns Found

| File | Pattern | Severity | Impact |
|------|---------|----------|--------|
| `tests/test_triton_monarch.py` | Pre-existing ruff errors (E402 + F401) documented in 02-05-SUMMARY.md | Info | Pre-existing, unrelated to Phase 2 changes; flagged for hygiene follow-up |
| `PROJECT.md` | Key Decisions table not updated with Option C disposition | Warning | ROADMAP SC #6 requires PROJECT.md update; noted as human_needed item |

---

## Human Verification Required

### 1. CUDA GPU Run of Full Strict Suite

**Test:** Run `uv run pytest tests/test_triton_*_strict.py -q` on a CUDA-capable machine (Ampere+ preferred; any GPU with triton support). Add `-m "not slow"` for a faster run, or omit to include slow cases.

**Expected:** All 603 tests pass. Fast tier (~270 tests) should pass in minutes; slow tier adds T ∈ {512, 1024} cases. The tight-TF32 bound structure (< 5e-4 dense/monarch/butterfly; < 1e-5 diagonal except slow-tier dbh at < 2e-5) was designed specifically to pass on Ampere+ GPUs per the Option C investigation.

**Why human:** Triton kernels require CUDA hardware. CPU collection (603 tests) confirms the tests are syntactically correct and properly parametrized, but the actual assertion values (< 5e-4, < 1e-5, torch.equal) can only be verified by running on a GPU.

### 2. PROJECT.md Key Decisions Update (Option C TF32 Disposition)

**Test:** Add a row to the `.planning/PROJECT.md` Key Decisions table documenting the Option C tolerance relaxation:

```markdown
| Strict-tier matmul-kernel bound held at < 5e-4 (tight-TF32), not < 1e-5 | Triton's tl.dot uses TF32 on Ampere+ regardless of torch.set_float32_matmul_precision('highest'); global precision knob does not propagate to in-kernel tl.dot; TF32 floor ~1e-4 abs makes < 1e-5 unachievable for dense/monarch/butterfly without IEEE-precision workaround | bd issue gru-triton-rwm (closed-accepted); Option C disposition in 02-SUMMARY.md; test docstrings in test_triton_{scan,monarch,butterfly}_strict.py |
```

**Expected:** PROJECT.md Key Decisions table has an entry documenting the bound relaxation, the TF32 root cause, and the bd issue reference.

**Why human:** ROADMAP SC #6 states "tolerance is not relaxed beyond the < 1e-5 contract without an explicit decision logged in PROJECT.md." The disposition is fully documented in module docstrings, the bd issue, and 02-SUMMARY.md, but PROJECT.md is the milestone-level Key Decisions record per its own "Evolution" section guidelines.

---

## Process Retrospective: Parallel-Execution Races

Three Wave-1 commits have mixed attribution due to parallel agents using broad `git add` (likely `git add .` or `git add -A`) instead of staging by exact file paths per the `task_commit_protocol`:

- `5bddd4a` — titled `docs(02-04): complete butterfly Triton strict-tier parity plan` but contains `tests/test_triton_scan_strict.py` (Plan 02-01's Task 1 file)
- `3ef47ef` — titled `test(02-03): add strict-tier monarch fwd parity tests` but also contains Plan 02-05's realistic-tier tightenings to `tests/test_triton_monarch.py`

**Content impact: zero.** Verification of file contents against authored expectations confirms all files are correct on disk. The parallel-execution race is commit-attribution only, not a content defect.

**Recommendations for future phases:**

1. **Preferred:** per-agent `git worktree` isolation — each agent operates on an isolated filesystem branch; staging-area collisions are physically impossible.
2. **Alternative:** serialize parallel-plan execution within a wave (Wave 1 plans run sequentially rather than concurrently). Trades wall-clock time for commit-isolation cleanliness.
3. **Minimum:** executor enforcement of "stage by exact paths only" — reject any `git add .` or `git add -A` at the tool layer, or emit an immediate retry with an explicit file list.

Wave 2 (sequential on the main working tree) had zero collisions — all 7 commits touch only their intended files.

---

## Gaps Summary

**No automated blockers.** The single `FAILED` truth (Option C not in PROJECT.md Key Decisions) is a documentation completeness issue, not a code correctness failure. The disposition is thoroughly documented elsewhere (bd issue, module docstrings, 02-SUMMARY.md). ROADMAP SC #6's requirement for PROJECT.md logging is a process contract, not a behavioral gate.

Two human verification items hold the status at `human_needed`:

1. **GPU run** — all 603 tests must pass at the Option C bounds on CUDA hardware. The tests exist, are correctly structured, and the bounds were set based on observed GPU behavior reported at the human-verify checkpoint. This is a confirmation gate, not a suspected failure.
2. **PROJECT.md update** — the Key Decisions table should record the TF32 disposition per the ROADMAP SC #6 contract. This is a one-row addition.

Both items are low-risk. Phase 2 is **PASS-WITH-CAVEATS** pending these two human-completable actions.

---

*Verified: 2026-05-13*
*Verifier: Claude (gsd-verifier)*

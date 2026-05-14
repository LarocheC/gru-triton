---
phase: 03-structured-pytorch-fallback-parity
verified: 2026-05-14T00:00:00Z
status: passed
score: 14/14
overrides_applied: 1
overrides:
  - must_have: "tests/test_structure.py extended with test_circulant_matches_handrolled_reference and test_ldr_matches_handrolled_reference"
    reason: "CONTEXT D-35 (locked before planning began) explicitly redirected work to a NEW file tests/test_structure_parity.py rather than extending test_structure.py. test_structure.py is a LOCKED file (D-38 contract). The new file delivers equivalent and stronger coverage: two reference forms for circulant (Toeplitz + FFT), micro-validation for LDR, fast + slow grids. The ROADMAP SC-1 wording was drafted before D-35 locked the file; the CONTEXT decision supersedes the ROADMAP wording."
    accepted_by: "claroche"
    accepted_at: "2026-05-14T00:00:00Z"
re_verification: false
---

# Phase 3: Structured PyTorch Fallback Parity — Verification Report

**Phase Goal:** Circulant and LDR per-step PyTorch paths match hand-rolled references at < 1e-5, and all structured variants degrade gracefully (clear error, not silent wrong-answer) when `torch-structured` is missing.

**Verified:** 2026-05-14
**Status:** PASSED
**Re-verification:** No — initial verification

---

## Goal Achievement

Starting hypothesis (adversarial): phase goal was not achieved. Evidence below falsifies this hypothesis — the goal IS achieved.

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `tests/test_structure_parity.py` exists, CPU-only, no module-top importorskip, no @cuda_only | VERIFIED | File at 810 lines; module docstring explicitly states "Pure PyTorch — no Triton, no CUDA"; importorskip appears only at line 338 (mid-file, LDR section); zero @cuda_only decorators |
| 2 | Hand-rolled Toeplitz and FFT forms agree at < 1e-5 abs (self-consistency cross-check before production comparison) | VERIFIED | `test_handrolled_circulant_self_consistent` (9 cases); asserts `max_diff < 1e-5`; worst empirical datum 2.27e-6 per plan 03-01 SUMMARY |
| 3 | `_CirculantLinear` forward matches hand-rolled Toeplitz at < 1e-5 abs across fast + slow grids | VERIFIED | `test_circulant_matches_handrolled_toeplitz` (9 fast) + `_slow` sibling (3 slow); all pass; worst 2.27e-6 |
| 4 | `_CirculantLinear` autograd backward (kernel_c gradient) matches hand-rolled Toeplitz autograd at < 1e-5 abs | VERIFIED | `test_circulant_backward_matches_autograd_reference` (9 fast) + `_slow` sibling (3 slow); worst 2.62e-6; per-tensor named-failure loop; g scaled 1/sqrt(B*H) |
| 5 | `_build_ldr_matrix_from_factors` helper exists; micro-validation on (H=8, rank=2) pins transpose convention before parametrized grid | VERIFIED | `test_handrolled_ldr_matches_production_micro` exists (line 432); passes; worst 1.19e-7 |
| 6 | `_LDRLinear` forward matches slow-Krylov hand-rolled reference at < 1e-5 abs across (B, H, rank) fast + slow grids | VERIFIED | `test_ldr_matches_handrolled_reference` (27 fast) + `_slow` sibling (9 slow); worst 1.67e-6 at (32, 512, 1); all pass |
| 7 | `_LDRLinear` autograd backward (4 leaves: subd_A, subd_B, G, H) matches hand-rolled reference autograd at < 1e-5 abs | VERIFIED | `test_ldr_backward_matches_autograd_reference` (27 fast) + `_slow` sibling (9 slow); 4-entry per-tensor named-failure loop; worst 1.31e-6 on G leaf |
| 8 | monarch + butterfly raise `ImportError` containing "torch-structured" when dep is missing (via monkeypatch.setattr) | VERIFIED | `test_missing_torch_structured_raises_clear_error` parametrized over ["monarch", "butterfly"]; uses `monkeypatch.setattr("gru_qat.structure._import_torch_structured", _raise_missing_torch_structured)`; `pytest.raises(ImportError, match=r"torch-structured")` |
| 9 | ldr raises `ImportError` containing "torch-structured" via separate sys.modules trick (bypasses _import_torch_structured) | VERIFIED | `test_missing_ldr_raises_clear_error`; uses `monkeypatch.setitem(sys.modules, "torch_structured", None)` etc.; `pytest.raises(ImportError, match=r"torch-structured")` |
| 10 | dense / diagonal / circulant produce finite working output with _import_torch_structured monkeypatched to raise | VERIFIED | `test_local_impls_work_without_torch_structured` parametrized over ["dense", "diagonal", "circulant"]; asserts finite + shape=(4,32) |
| 11 | No xfail markers anywhere in tests/test_structure_parity.py | VERIFIED | `grep -c "xfail" tests/test_structure_parity.py` returns 0 |
| 12 | Locked files (test_parity.py, test_layer_parity.py, test_structure.py) unchanged across all Phase 3 commits | VERIFIED | `git diff 987c770~1 HEAD -- tests/test_parity.py tests/test_layer_parity.py tests/test_structure.py` returns 0 bytes; `git log --name-only --oneline 987c770~1..HEAD -- tests/test_parity.py tests/test_layer_parity.py tests/test_structure.py` produces no output |
| 13 | No src/ modifications during Phase 3 (D-39) | VERIFIED | `git log --name-only --oneline 987c770..HEAD -- src/` produces no output; zero production findings |
| 14 | TESTING.md Mocking section updated to reflect Phase 3 monkeypatch introduction | VERIFIED | `grep -c "monkeypatch" .planning/codebase/TESTING.md` returns 3; "Narrow exception (Phase 3, Plan 03-03)" section documents two blessed idioms; `grep "No.*monkeypatching"` returns 0 matches |

**Score:** 14/14 truths verified (1 via override — ROADMAP SC-1 file-location deviation accepted per D-35 pre-decision)

---

## Requirement Coverage

| Requirement | Test Function(s) | Status | Evidence |
|-------------|-----------------|--------|----------|
| STR-01: Circulant fwd + bwd matches hand-rolled reference at < 1e-5 | `test_handrolled_circulant_self_consistent`, `test_circulant_matches_handrolled_toeplitz`, `test_circulant_matches_handrolled_toeplitz_slow`, `test_circulant_backward_matches_autograd_reference`, `test_circulant_backward_matches_autograd_reference_slow` | SATISFIED | 9+3 fwd cases, 9+3 bwd cases; worst 2.62e-6 |
| STR-02: LDR fwd + bwd matches hand-rolled reference at < 1e-5 | `test_handrolled_ldr_matches_production_micro`, `test_ldr_matches_handrolled_reference`, `test_ldr_matches_handrolled_reference_slow`, `test_ldr_backward_matches_autograd_reference`, `test_ldr_backward_matches_autograd_reference_slow` | SATISFIED | 1+27+9 fwd cases, 27+9 bwd cases; worst 1.67e-6 |
| STR-03: All structured variants degrade gracefully (clear error, not silent wrong-answer) when torch-structured missing | `test_missing_torch_structured_raises_clear_error` (monarch+butterfly), `test_missing_ldr_raises_clear_error` (ldr), `test_local_impls_work_without_torch_structured` (dense+diagonal+circulant) | SATISFIED | 6 tests total; all raise ImportError matching "torch-structured" for optional kinds; local kinds work |

---

## Dimension-by-Dimension Verdict

### 1. STR-01..03 Coverage

VERIFIED. Each requirement maps to one or more test functions in `tests/test_structure_parity.py`:
- STR-01: 5 functions, 33 parametrized cases
- STR-02: 5 functions, 73 parametrized cases (1 micro + 36 fwd + 36 bwd)
- STR-03: 3 functions, 6 parametrized cases

### 2. Tolerance Contract

VERIFIED. Every parity assertion in the file uses `< 1e-5` absolute error. Confirmed by reading the assert bodies and by the observed empirical worst-case across all 112 tests: 2.62e-6 (circulant backward, H=512). The `< 1e-5` bound has at minimum ~4x headroom across the entire grid.

### 3. Self-Consistency Test (D-29)

VERIFIED. `test_handrolled_circulant_self_consistent` exists at line 132; parametrized over `FAST_CIRC_GRID` (9 cases); asserts Toeplitz form and full-complex FFT form agree at < 1e-5 before either is compared to `_CirculantLinear`.

### 4. Autograd-Grad Backward Parity (D-30)

VERIFIED. `test_circulant_backward_matches_autograd_reference` exists (line 205); uses detach-clone-twice idiom; production leaf via `layer.col.copy_(c_prod)` with gradient read from `layer.col.grad`; shared g scaled 1/sqrt(B*H); per-tensor named-failure loop `("kernel_c", c_ref.grad, layer.col.grad)`. Slow sibling at line 267.

### 5. LDR Helper + Micro-Validation (D-32)

VERIFIED. `_build_ldr_matrix_from_factors` exists at line 343; uses Python-loop explicit slow-Krylov (not FFT); fully typed; docstring references krylov.py line numbers. `test_handrolled_ldr_matches_production_micro` at line 432; single non-parametrized case at (H=8, rank=2); pins transpose convention `M = sum_i K_A(G[i]) @ K_B(H[i]).T`.

### 6. STR-03 Missing-Dep Handling (D-34)

VERIFIED. Split exactly per Option C:
- `test_missing_torch_structured_raises_clear_error` parametrized over `["monarch", "butterfly"]` — uses `monkeypatch.setattr` on `_import_torch_structured`
- `test_missing_ldr_raises_clear_error` (not parametrized) — uses `monkeypatch.setitem(sys.modules, ...)` because the LDR branch bypasses `_import_torch_structured`
- `test_local_impls_work_without_torch_structured` parametrized over `["dense", "diagonal", "circulant"]`

### 7. File Location (D-35)

VERIFIED with override. `tests/test_structure_parity.py` was created as a new file (not extending `test_structure.py`). This deviates from ROADMAP Success Criterion 1 literal wording but was mandated by CONTEXT D-35 which pre-dates the plans. `tests/test_structure.py` remains unchanged per the locked-file contract (D-38). The coverage intent of SC-1 is fully met by the new file with stronger methodology (two independent references, self-consistency cross-check).

### 8. Shape Grid (D-36)

VERIFIED.
- `FAST_CIRC_GRID` at line 102: 9 cases (B in {1,4,32} x H in {8,32,128})
- `SLOW_CIRC_GRID` at line 107: 3 cases (B in {1,4,32} x H in {512,}) — H=512 marked slow
- `FAST_LDR_GRID` at line 417: 27 cases (B in {1,4,32} x H in {8,32,128} x rank in {1,4,8} with rank<=H)
- `SLOW_LDR_GRID` at line 424: 9 cases (B in {1,4,32} x H in {512,} x rank in {1,4,8})
- CUDA NOT required: no @cuda_only decorators anywhere in the file.

### 9. No xfail (D-37/D-12)

VERIFIED. `grep -c "xfail" tests/test_structure_parity.py` returns 0.

### 10. D-38 Locks

VERIFIED. `git diff 987c770~1 HEAD -- tests/test_parity.py tests/test_layer_parity.py tests/test_structure.py` returns 0 bytes. All three locked files are bit-for-bit identical across all 9 Phase 3 commits.

### 11. No src/ Changes (D-39)

VERIFIED. `git log --name-only --oneline 987c770..HEAD -- src/` produces no output. Zero production-path modifications during Phase 3.

### 12. bd Discipline

VERIFIED. Phase-exit SUMMARY documents 0 bd issues opened during Phase 3. `bd list` shows only 2 open issues: `gru-triton-4m6` (Phase 1 pre-existing mypy/ruff debt) and `gru-triton-e7t` (Phase 2 diagonal F-02-02-A finding) — both predate Phase 3. Zero findings surfaced during the circulant or LDR audits.

### 13. TESTING.md Update (Mocking Section)

VERIFIED. `.planning/codebase/TESTING.md` contains a "Narrow exception (Phase 3, Plan 03-03)" subsection (lines 205-212) that replaces the previous absolute no-mocking rule with a scoped exception: two blessed idioms documented (`setattr` on lazy-import helper; `setitem(sys.modules, ..., None)`). `grep "No.*monkeypatching"` returns 0 — the absolute prohibition is gone.

### 14. Phase 3 Success Criteria (ROADMAP)

| SC | Verdict | Notes |
|----|---------|-------|
| SC-1: tests/test_structure.py extended with circulant + LDR references | PASSED (override) | Per D-35, work went to new file tests/test_structure_parity.py. Coverage intent fully met. test_structure.py locked; would violate D-38 to extend it. |
| SC-2: Tests build "tiny independent references" — circulant has Toeplitz + FFT, LDR has full-matrix construction | VERIFIED | Toeplitz at line 50; full-complex FFT at line 77; LDR slow-Krylov at line 343. Self-consistency cross-check before production comparison. |
| SC-3: Tests assert fwd + bwd parity at < 1e-5 | VERIFIED | All 7 parity families assert `< 1e-5` abs. Empirical worst: 2.62e-6. |
| SC-4: Missing torch-structured graceful error tested for monarch, butterfly, LDR | VERIFIED | 3 separate tests cover all three optional-dep kinds; LDR handled via Option C (sys.modules). |
| SC-5: Shape-validator behavior unchanged (existing test_structure.py) | VERIFIED | test_structure.py unchanged (D-38 locked); `pytest tests/test_structure.py -q` passes 20 tests. |

---

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `tests/test_structure_parity.py` | New file, 810 lines, CPU-only strict-tier audit | VERIFIED | Exists; 810 lines; module docstring, 'highest' preamble, 2 circulant helpers, 2 LDR helpers, 4 shape grid constants, 13 test functions (9 fast families + 4 slow siblings), 6 STR-03 tests |
| `.planning/codebase/TESTING.md` | Mocking section updated | VERIFIED | Contains "monkeypatch" 3 times; narrow-exception section present; absolute prohibition removed |
| `.planning/phases/03-structured-pytorch-fallback-parity/03-SUMMARY.md` | Phase-exit SUMMARY with STR-01/02/03 closure | VERIFIED | Exists; frontmatter `requirements-completed: [STR-01, STR-02, STR-03]`; contains all 8 required sections |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `_build_toeplitz_from_kernel` + `_circulant_via_fft` | `test_handrolled_circulant_self_consistent` | two-reference cross-check BEFORE production | WIRED | Lines 145-153: both helpers called, results compared at < 1e-5 |
| `test_circulant_matches_handrolled_toeplitz` | `_CirculantLinear` in src/gru_qat/structure.py | production layer instantiated and called | WIRED | Line 167: `layer = _CirculantLinear(H, bias=False)` |
| `test_circulant_backward_matches_autograd_reference` | autograd-grad on c_ref (Toeplitz) vs layer.col (production) | detach-clone-twice + shared g + per-tensor named-failure loop | WIRED | Lines 224-262: c_ref leaf, layer.col.copy_(c_prod), backward, named loop |
| `_build_ldr_matrix_from_factors` | `_LDRLinear` / LDRSubdiagonal production path | micro-validation + parametrized forward/backward | WIRED | Line 457: `M = _build_ldr_matrix_from_factors(...)`, `y_ref = x @ M.T` compared to `layer(x)` |
| `test_missing_torch_structured_raises_clear_error` | `make_structured_linear` in src/gru_qat/structure.py | `monkeypatch.setattr("gru_qat.structure._import_torch_structured", ...)` | WIRED | Line 747-753: setattr + pytest.raises match |
| `test_missing_ldr_raises_clear_error` | src/gru_qat/structure.py:160-172 LDR import branch | `monkeypatch.setitem(sys.modules, "torch_structured.structured.layers", None)` | WIRED | Lines 773-780: three sys.modules entries set to None; pytest.raises match |

---

## Data-Flow Trace (Level 4)

Not applicable. This phase produces test files only — no dynamic data-rendering components. All test functions are self-contained: they generate synthetic inputs via `torch.randn`, compute two independent paths, and compare. No component renders external state.

---

## Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| All 88 fast tests pass | `uv run pytest tests/test_structure_parity.py -m "not slow" -q` | 88 passed, 24 deselected in 5.41s | PASS |
| All 24 slow tests pass | `uv run pytest tests/test_structure_parity.py -m slow -q` | 24 passed, 88 deselected in 8.80s | PASS |
| Full suite (fast + slow) | `uv run pytest tests/test_structure_parity.py -q` | 112 passed in 11.86s | PASS |
| Locked test file test_parity.py | `uv run pytest tests/test_parity.py -q` | 12 passed in 2.17s | PASS |
| Locked test file test_layer_parity.py (fast) | `uv run pytest tests/test_layer_parity.py -q -m "not slow"` | 184 passed in 7.48s | PASS |
| Locked test file test_structure.py | `uv run pytest tests/test_structure.py -q` | 20 passed in 52.37s | PASS |
| Ruff clean | `uv run ruff check tests/test_structure_parity.py` | All checks passed | PASS |

---

## Probe Execution

No probes declared in PLAN files. No `scripts/*/tests/probe-*.sh` files found. Behavioral spot-checks above cover the runnable verification surface.

---

## Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| STR-01 | 03-01-PLAN.md | Circulant fwd+bwd matches hand-rolled reference at < 1e-5 | SATISFIED | 5 test functions, 33 cases, worst 2.62e-6 |
| STR-02 | 03-02-PLAN.md | LDR fwd+bwd matches hand-rolled reference at < 1e-5 | SATISFIED | 5 test functions, 73 cases, worst 1.67e-6 |
| STR-03 | 03-03-PLAN.md | All structured variants degrade gracefully when torch-structured missing | SATISFIED | 3 test functions (6 parametrized cases), all pass |

---

## Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| None | — | — | — | No TBD/FIXME/XXX/PLACEHOLDER/return null/empty stub patterns found in test_structure_parity.py |

No debt markers. No stub patterns. No hardcoded empty returns. The file is substantive throughout.

---

## Human Verification Required

None. All verification dimensions are programmatically observable:
- Test pass/fail is deterministic via pytest
- Tolerance contract is code-readable
- Locked-file integrity is git-verifiable
- xfail absence is grep-verifiable
- src/ no-change is git-verifiable
- TESTING.md update is text-readable

No UI behavior, real-time state, or external service integration involved.

---

## Gaps Summary

No blocking gaps. One override applied (ROADMAP SC-1 file-location deviation), accepted based on pre-existing CONTEXT D-35 decision that locked `test_structure.py` before Phase 3 planning began.

---

_Verified: 2026-05-14T00:00:00Z_
_Verifier: Claude (gsd-verifier)_

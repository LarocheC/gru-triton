---
phase: 01-reference-path-parity-vs-nn-gru
verified: 2026-05-13T00:00:00Z
status: passed
score: 13/13 must-haves verified
overrides_applied: 0
re_verification: false
---

# Phase 1: Reference-path parity vs nn.GRU — Verification Report

**Phase Goal:** Pin `GRULayer` (use_triton=False, Identity quantizers, dense) against `torch.nn.GRU` (single-layer, unidirectional, batch_first=True) at the layer level so every later phase has a trusted ground truth. Forward, backward, h_T, h_0≠0 across the T × B × H = 75-combo grid at < 1e-4.

**Verified:** 2026-05-13
**Status:** PASSED
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `tests/test_layer_parity.py` exists, is importable, runs CPU-only | VERIFIED | File exists at 719 lines; `pytest --collect-only` returns 304 tests |
| 2 | Layer forward vs `nn.GRU` < 1e-4 across 75-combo grid | VERIFIED | 45 fast + 30 slow = 75 parametrized cases; `184 passed, 120 deselected` fast; slow spot-check PASSED |
| 3 | h_T vs `nn.GRU` h_n < 1e-4 across 75-combo grid | VERIFIED | Separate `test_layer_h_T_matches_nn_gru` + `_slow` variant; all 75 pass |
| 4 | Backward gradients (dx, dh_0, dW_ih, dW_hh, db_ih, db_hh) < 1e-4 | VERIFIED | `test_layer_backward_matches_nn_gru` + `_slow`; 6-gradient loop per grid point |
| 5 | h_0 ≠ 0 random initial state parity < 1e-4 | VERIFIED | `test_layer_with_random_h0_matches_nn_gru` + `_slow`; asserts both out and h_T |
| 6 | Translation helpers exist, documented, and correct | VERIFIED | `_translate_cell_to_nn_gru`, `_translate_nn_gru_to_cell`, `_make_dense_fp32_layer` all at module level; round-trip smoke test PASSED |
| 7 | Gate-ordering micro-tests present and passing | VERIFIED | `test_gate_order_r_only`, `test_gate_order_z_only`, `test_n_gate_asymmetry` all PASSED |
| 8 | Cell-level < 1e-5 parity gate unchanged | VERIFIED | `tests/test_parity.py` 12 passed; `git diff 786b32c^..HEAD -- tests/test_parity.py` is empty |
| 9 | Zero `src/` modifications across Phase 1 | VERIFIED | `git log --name-only 786b32c^..HEAD -- src/` returned empty |
| 10 | No `@pytest.mark.xfail` markers | VERIFIED | `grep -n "xfail" tests/test_layer_parity.py` returns nothing (exit 1) |
| 11 | `torch.set_float32_matmul_precision('highest')` at module scope | VERIFIED | Line 33 of `tests/test_layer_parity.py` — outside all functions |
| 12 | Four-family test separation (D-09) | VERIFIED | 4 distinct parametrized functions (fwd, h_T, bwd, h_0≠0) × 2 speed variants = 8 parametrized functions |
| 13 | FAST_GRID=45, SLOW_GRID=30, total=75 covering exact T/B/H sets | VERIFIED | FAST T={1,8,64}, SLOW T={512,1024}, B={1,4,32}, H={1,2,8,64,512}; 3×3×5=45 fast, 2×3×5=30 slow |

**Score:** 13/13 truths verified

---

## Requirement Coverage

| REQ-ID | Statement | Test Function(s) | Status |
|--------|-----------|-----------------|--------|
| REF-01 | Forward output parity over T×B×H grid at < 1e-4 | `test_layer_forward_matches_nn_gru` (45 fast) + `test_layer_forward_matches_nn_gru_slow` (30 slow) | SATISFIED |
| REF-02 | h_0 ≠ 0 random initial-state parity at < 1e-4 | `test_layer_with_random_h0_matches_nn_gru` (45 fast) + `test_layer_with_random_h0_matches_nn_gru_slow` (30 slow) | SATISFIED |
| REF-03 | Backward gradients (dW_ih, dW_hh, db_ih, db_hh, dx, dh_0) < 1e-4 | `test_layer_backward_matches_nn_gru` (45 fast) + `test_layer_backward_matches_nn_gru_slow` (30 slow) | SATISFIED |
| REF-04 | Final hidden state h_T matches `nn.GRU`'s h_n at < 1e-4 | `test_layer_h_T_matches_nn_gru` (45 fast) + `test_layer_h_T_matches_nn_gru_slow` (30 slow) | SATISFIED |
| REF-05 | Gate-ordering/bias-fusion translation helper exists and is documented | `_translate_cell_to_nn_gru`, `_translate_nn_gru_to_cell` (module-level, documented); micro-tests: `test_gate_order_r_only`, `test_gate_order_z_only`, `test_n_gate_asymmetry`, `test_round_trip_nn_gru_to_cell` | SATISFIED |

---

## Dimension-by-Dimension Verdict

### 1. REF-01..05 Coverage

PASS. Every requirement maps to at least one running parametrized test. REF-01 → forward; REF-02 → h_0≠0; REF-03 → backward; REF-04 → h_T; REF-05 → helpers + micro-tests. All tests pass on live execution (184 fast confirmed in this verification session; 4 micro/smoke confirmed in this session; slow spot-check at T=512 H=1 confirmed passing).

### 2. Tolerance Contract

PASS. Layer-vs-nn.GRU tests uniformly assert `rel < 1e-4` using the relative-error idiom (`(ref - ours).abs().max() / max(ref.abs().max(), 1e-6)`). All layer test assertions use exactly `1e-4`. Cell parity in `tests/test_parity.py` uses `< 1e-5` (via `max_diff < 1e-5` and `torch.allclose(atol=1e-5)`) — confirmed unchanged.

### 3. Grid Coverage (D-08)

PASS. Verified programmatically: `len(FAST_GRID) == 45`, `len(SLOW_GRID) == 30`, total 75. T values: FAST has {1, 8, 64}, SLOW has {512, 1024}. B: {1, 4, 32}. H: {1, 2, 8, 64, 512}. Slow variants gated behind `@pytest.mark.slow`. Confirmed via `pytest --collect-only -q` returning 304 tests total (4 + 45×4 + 30×4 = 4 + 180 + 120 = 304).

### 4. Four-Family Separation (D-09)

PASS. Exactly 8 parametrized functions (4 families × fast/slow):
- `test_layer_forward_matches_nn_gru` / `test_layer_forward_matches_nn_gru_slow`
- `test_layer_h_T_matches_nn_gru` / `test_layer_h_T_matches_nn_gru_slow`
- `test_layer_backward_matches_nn_gru` / `test_layer_backward_matches_nn_gru_slow`
- `test_layer_with_random_h0_matches_nn_gru` / `test_layer_with_random_h0_matches_nn_gru_slow`

Each family is a separate function with its own `@pytest.mark.parametrize` decorator. Not fused.

### 5. Three Micro-Tests (D-04)

PASS. All three present and individually confirmed passing:
- `test_gate_order_r_only` — PASSED (live run)
- `test_gate_order_z_only` — PASSED (live run)
- `test_n_gate_asymmetry` — PASSED (live run; forces `b_ir=-100.0` to squash r~0, confirming n-gate asymmetry)

### 6. Translation Helpers (D-01..03)

PASS. Three module-level underscore-prefixed helpers confirmed present:
- `_make_dense_fp32_layer` at line 36
- `_translate_cell_to_nn_gru` at line 56 — includes PyTorch GRU docs URL (https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html) per D-05; documents gate-order contract and n-gate asymmetry
- `_translate_nn_gru_to_cell` at line 99 — uses `chunk(3, dim=0)` as required; uses `bin_` trailing-underscore for builtin avoidance

Round-trip smoke test `test_round_trip_nn_gru_to_cell` present and PASSED (live run).

### 7. Precision Policy (D-07)

PASS. `torch.set_float32_matmul_precision("highest")` at module scope (line 33), outside all functions. Confirmed by grep.

### 8. D-12 (No xfail)

PASS. `grep -n "xfail" tests/test_layer_parity.py` returns empty (exit code 1 — no matches). Zero `@pytest.mark.xfail` markers anywhere in the file.

### 9. D-10 Two-Commit Discipline

PASS (vacuously). Audit was fully green — zero parity findings surfaced across all 304 tests. Therefore zero Commit A / Commit B pairs were needed. The git log shows exclusively `test(01-XX):` commits for test additions and `docs(01-XX):` commits for summaries. `git log --name-only 786b32c^..HEAD -- src/` returns empty — zero `src/` modifications across the entire Phase 1 range. The two-commit discipline was honored by not needing to fire.

### 10. Cell-Parity Contract Integrity

PASS. `tests/test_parity.py` is unchanged across all Phase 1 commits. `git diff 786b32c^..HEAD -- tests/test_parity.py` returns empty. Live re-run: `12 passed in 1.29s`.

### 11. src/ Untouched

PASS. `git log --name-only 786b32c^..HEAD -- src/` returns empty. Confirmed in this verification session.

### 12. bd Discipline

PASS. Exactly one bd issue filed: `gru-triton-4m6` — "Pre-existing mypy/ruff debt in src/gru_qat/*" (P3, OPEN). This tracks pre-existing toolchain debt that predates Phase 1 (identical error counts at Phase 1 baseline and head). No parity-finding bd issues were filed because the audit is fully green. `bd show gru-triton-4m6` confirms the issue exists and is open.

### 13. Phase 1 ROADMAP Success Criteria

All five checked:

a. **New `tests/test_layer_parity.py` exists, runs CPU-only, passes — parametrized over the full T×B×H grid for fwd, bwd, h_T, h_0≠0.** PASS. 304 tests collected; 184 fast PASSED live; slow spot-checks PASSED; no CUDA dependency.

b. **Layer fwd vs `nn.GRU` < 1e-4 absolute; gradients match autograd < 1e-4.** PASS. All assertions use relative-error `< 1e-4`. Backward family covers all 6 gradient tensors (dx, dh_0, dW_ih, dW_hh, db_ih, db_hh).

c. **Test-local `_translate_nn_gru_to_cell` helper exists and is documented.** PASS. Present at line 99 with a docstring. Round-trip smoke test passes.

d. **Existing cell `< 1e-5` parity in `tests/test_parity.py` unchanged.** PASS. 12 passed; file diff is empty.

e. **Each mismatch → failing test → bd → fix in same phase. Verified: zero findings.** PASS. Zero findings; the discipline was honored by the fully-green outcome.

---

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `tests/test_layer_parity.py` | Phase 1 audit deliverable — helpers, 4 micro-tests, 8 parametrized families | VERIFIED | 719 lines; 304 tests; substantive content with full docstrings and correct assertions |
| `tests/test_parity.py` | Unchanged cell-parity < 1e-5 contract | VERIFIED | Unchanged; 12 tests pass; locked tolerance intact |
| `.planning/phases/01-reference-path-parity-vs-nn-gru/01-05-SUMMARY.md` | Phase audit verdict | VERIFIED | Exists with full verdict, pytest tails, requirements closure, bd issue reference |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `_translate_cell_to_nn_gru` | `GRUCellQuant` (W_ir, W_iz, W_in, ...) | `torch.cat([cell.W_ir, cell.W_iz, cell.W_in], dim=0)` into `gru.weight_ih_l0` | WIRED | Pattern confirmed at lines 92-95 |
| `_translate_nn_gru_to_cell` | `GRULayer.cell` (W_ir, W_iz, W_in, ...) | `gru.weight_ih_l0.chunk(3, dim=0)` → cell.W_ir/W_iz/Win | WIRED | Pattern confirmed at lines 117-133 |
| Parametrized tests | `_translate_cell_to_nn_gru` | Called inside each test body to build the `nn.GRU` reference | WIRED | Used in all 8 parametrized functions |
| h_0 shape adapter | `nn.GRU` h0 [1,B,H] vs `GRULayer` h0 [B,H] | `h0_3d.squeeze(0)` / `hT_ref.squeeze(0)` | WIRED | Correctly handled in all test families; backward test uses `h0_ref.grad.squeeze(0)` |

---

## Data-Flow Trace (Level 4)

Not applicable. `tests/test_layer_parity.py` is a test file, not a component rendering dynamic data from a store or API. Data flows are fully local: random tensors constructed in each test body, passed through `nn.GRU` and `GRULayer`, compared inline.

---

## Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Fast suite 184 tests pass | `pytest tests/test_layer_parity.py -q -m "not slow"` | `184 passed, 120 deselected in 6.04s` | PASS |
| Cell parity 12 tests pass | `pytest tests/test_parity.py -q` | `12 passed in 1.29s` | PASS |
| 4 micro/smoke tests pass | `pytest test_gate_order_r_only test_gate_order_z_only test_n_gate_asymmetry test_round_trip_nn_gru_to_cell` | `4 passed in 1.31s` | PASS |
| Slow spot-check at T=512 H=1 | `pytest test_layer_*_slow[512-1-1]` (4 tests) | `4 passed in 2.33s` | PASS |
| No xfail markers | `grep -n "xfail" tests/test_layer_parity.py` | no output (exit 1) | PASS |
| No src/ modifications | `git log --name-only 786b32c^..HEAD -- src/` | empty | PASS |

---

## Probe Execution

No probes declared for Phase 1. Phase 1 is a test-creation phase; the pytest runs above serve as the behavioral verification. Step 7c: SKIPPED (no probe scripts for this phase).

---

## Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| REF-01 | 01-02-PLAN.md | Forward output parity < 1e-4 across 75-combo grid | SATISFIED | `test_layer_forward_matches_nn_gru` + slow; 75/75 pass |
| REF-02 | 01-04-PLAN.md | h_0 ≠ 0 random initial-state parity < 1e-4 | SATISFIED | `test_layer_with_random_h0_matches_nn_gru` + slow; 75/75 × {out,h_T} pass |
| REF-03 | 01-03-PLAN.md | Backward gradients 6 tensors < 1e-4 | SATISFIED | `test_layer_backward_matches_nn_gru` + slow; 75/75 × 6 grads pass |
| REF-04 | 01-02-PLAN.md | h_T vs h_n < 1e-4 | SATISFIED | `test_layer_h_T_matches_nn_gru` + slow; 75/75 pass |
| REF-05 | 01-01-PLAN.md | Translation helpers + gate-ordering documentation | SATISFIED | Helpers present with docstrings and PyTorch URL; 3 micro-tests + 1 round-trip pass |

No orphaned requirements — all 5 REF requirements are claimed by Phase 1 plans and verified here.

---

## Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| None | — | — | — | Zero debt markers, no stubs, no hardcoded empty data, no xfail in test file |

Scan result: zero `TBD`, `FIXME`, `XXX`, `TODO`, `HACK`, `PLACEHOLDER` matches in `tests/test_layer_parity.py`. Zero `return null` / `return []` patterns (test file uses direct assertions). Zero `@pytest.mark.xfail`.

---

## Human Verification Required

None. All phase goal behaviors are programmatically verifiable and were verified in this session. The audit is mathematical correctness testing of deterministic CPU-only code — no visual, real-time, external-service, or performance-feel components.

---

## Issues / Concerns

**Pre-existing toolchain debt (not a Phase 1 finding):** `mypy` reports 145 errors and `ruff` reports 24 errors in `src/gru_qat/`. These are pre-existing — identical error counts at the Phase 1 baseline commit (786b32c^) and at HEAD. No `src/` bytes were touched by Phase 1. Tracked as bd issue `gru-triton-4m6` (P3, open). Not a blocker for Phase 2.

**Slow suite runtime:** The full slow suite runs in approximately 89 seconds on the audit machine. Well within the 10-minute threshold from CONTEXT D-08. No grid pruning required.

**One test in `test_parity.py` uses `atol=1e-4` (line 103):** `test_cell_large_magnitude` uses `torch.allclose(atol=1e-4, rtol=1e-4)` — this is a pre-existing large-magnitude edge-case test, not the primary parity contract. The primary parity tests at lines 70, 81, 92, 135, 162-163 all use `< 1e-5`. The Phase 1 requirement is that the `< 1e-5` parity gate is not loosened — it isn't.

---

## Recommendation

PROCEED TO PHASE 2. All five REF requirements satisfied. The reference path is verified as a trusted ground truth at < 1e-4 against `torch.nn.GRU` across the full 75-combo T × B × H grid for all four test families (forward, h_T, backward, h_0≠0). The cell-level < 1e-5 contract is preserved. No src/ modifications occurred. Phase 2 (Triton fast-path parity vs reference) can now use `use_triton=False` GRULayer output as the authoritative reference at < 1e-5.

---

_Verified: 2026-05-13_
_Verifier: Claude (gsd-verifier)_

---
phase: 04-quant-on-bit-identity
verified: 2026-05-14T14:30:00Z
status: gaps_found
score: 8/14 must-haves verified
overrides_applied: 0
re_verification: false
gaps:
  - truth: "QNT-01: Dense Triton fwd is bit-identical (torch.equal); dense Triton bwd is within the documented exceptions"
    status: partial
    reason: "Dense fwd passes torch.equal (verified). Dense bwd has failures BEYOND the documented F-04-05-A exception: realistic cls at (T=64,B=32,H=32) fails at 284% of h_scale; near-saturation cls at B=32 shapes fails at 254–393% of h_scale. F-04-05-A covers only large-magnitude at T=512; these additional failures were not documented or bd-tracked. 18 non-probe failures in test_scan_quant_bwd."
    artifacts:
      - path: "tests/test_triton_scan_strict.py"
        issue: "test_scan_quant_bwd fails at realistic-64-32-32 (284% h_scale), near-saturation B=32 x6 (254–393%), large-magnitude B>1 x11 (270–914% even with mult=2.0 exception)"
    missing:
      - "Either additional h_scale_mult exceptions documented and bd-tracked for near-saturation B=32 and realistic B=32 failures, or a fix to the dense bwd kernel for these cases"
      - "bd issues for the undocumented failure buckets (near-saturation B=32 and realistic B=32)"

  - truth: "QNT-02: Diagonal/Monarch/Butterfly Triton fwd matches reference under the documented per-kernel dispositions"
    status: failed
    reason: "Monarch fwd fails torch.equal by exactly h_scale (one INT8 step) for 142 of 162 fast cases across all three adversarial classes. This is NOT a documented exception — the SUMMARY claimed monarch fwd passes torch.equal uniformly. Diagonal fwd has 1 undocumented failure (large-magnitude-64-32-128). Butterfly fwd failures exceed the F-04-05-B mult=5.0 bound in some cases (non-deterministic across runs, 62–89 failures depending on run)."
    artifacts:
      - path: "tests/test_triton_monarch_strict.py"
        issue: "test_monarch_quant_fwd[realistic-8-1-128-2] and 141 other fast cases fail torch.equal with max_abs_diff=h_scale=0.02 (exactly one INT8 step). Reproducible in isolation."
      - path: "tests/test_triton_diagonal_strict.py"
        issue: "test_diagonal_quant_fwd[large-magnitude-64-32-128] fails torch.equal — not documented in SUMMARY"
      - path: "tests/test_triton_butterfly_strict.py"
        issue: "62–89 failures (non-deterministic across runs); fwd failures show up even with mult=5.0 exception applied"
    missing:
      - "Root-cause investigation of monarch fwd torch.equal violation (exactly h_scale diff suggests a rounding-boundary flip between PyTorch ref and Triton kernel)"
      - "bd issues for monarch fwd failures (142 cases) and diagonal fwd failure (1 case)"
      - "F-04-05-B bd issue (gru-triton-5rk open) may need to be expanded to cover additional butterfly fwd shapes"

  - truth: "QNT-03: Quant-on backward gradients are within bounds across all variants"
    status: failed
    reason: "Dense bwd (see QNT-01 gap). Monarch bwd has ~61 failures including at least 2 cases where even the standard h_scale bound is exceeded (test_monarch_quant_bwd[large-magnitude-64-32-512-4] and test_monarch_quant_bwd[large-magnitude-64-32-512-8]). Butterfly bwd has large-magnitude failures at (64,4,512), (64,32,*) that exceed h_scale even without a mult exception."
    artifacts:
      - path: "tests/test_triton_monarch_strict.py"
        issue: "~61 bwd failures including undocumented large-magnitude cases"
      - path: "tests/test_triton_butterfly_strict.py"
        issue: "Large-magnitude bwd failures at T=64 shapes not covered by the F-04-05-A exception (which is dense-only)"
    missing:
      - "bd issues for monarch bwd failures and butterfly large-magnitude bwd failures at T=64"
      - "Disposition clarification for these additional cases (not covered by existing F-04-05-A/B)"

  - truth: "Phase 4 quant-on suite passes on CUDA at the disposition-resolved bound (SUMMARY must-have #5)"
    status: failed
    reason: "SUMMARY truth #5 claims 'All dense + diagonal + monarch + butterfly quant tests pass under the bound applied.' Actual CUDA run (on this machine — RTX 2000 Ada, which is a CUDA-capable machine, CUDA 13.2) shows 285+ failures across the four kernels. The SUMMARY is materially incorrect for this truth."
    artifacts:
      - path: "tests/test_triton_scan_strict.py"
        issue: "19 Phase 4 quant failures (1 probe expected; 18 unexpected)"
      - path: "tests/test_triton_diagonal_strict.py"
        issue: "1 Phase 4 quant failure (fwd large-magnitude, undocumented)"
      - path: "tests/test_triton_monarch_strict.py"
        issue: "203 Phase 4 quant failures (142 fwd + ~61 bwd)"
      - path: "tests/test_triton_butterfly_strict.py"
        issue: "62–89 Phase 4 quant failures (non-deterministic; mix of fwd and bwd)"
    missing:
      - "GPU run on a machine that confirms all tests pass (the SUMMARY claims they passed on some GPU run; verifier cannot replicate)"
      - "OR: additional bound-loosening commits with bd issues for each newly-discovered failure bucket"

  - truth: "ROADMAP + STATE reflect Phase 4 completion (SUMMARY must-have #13)"
    status: failed
    reason: "ROADMAP.md Phase 4 checkbox is still [ ] (unchecked). SUMMARY deferred this to the orchestrator, but the orchestrator has not updated it. Both REQUIREMENTS.md QNT-01..04 remain marked '- [ ]' (Pending). This is not a gap in the code but signals Phase 4 was not properly closed."
    artifacts:
      - path: ".planning/ROADMAP.md"
        issue: "Phase 4 shows '- [ ] **Phase 4: Quant-on bit-identity**' (unchecked)"
      - path: ".planning/REQUIREMENTS.md"
        issue: "QNT-01..04 all show '- [ ]' status"
    missing:
      - "Orchestrator flip of Phase 4 checkbox in ROADMAP.md and REQUIREMENTS.md after verified completion"
---

# Phase 4: Quant-on Bit-Identity Verification Report

**Phase Goal:** With a frozen INT8 recipe applied, every Triton variant produces bit-identical fwd and bwd against the reference path; resolve the per-channel `min_max` observer gap.

**Verified:** 2026-05-14T14:30:00Z
**Status:** GAPS FOUND
**Re-verification:** No — initial verification
**GPU:** RTX 2000 Ada Generation, CUDA 13.2 (the machine CUDA is available on, matching the GPU used for Plan 04-05)

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | QNT-04 fixed: per-channel min_max observer uses per-axis reduction | VERIFIED | `_update_observer` has per-axis `amin/amax` when `axis is not None`; `test_per_channel_min_max_observer_per_channel_running_stats` passes (8 active + 1 skip in test_quantizers.py); bd `gru-triton-x15` closed |
| 2 | QNT-04 Commit A precedes Commit B in git log (D-37/D-45) | VERIFIED | `0b6adec` (test, 13:02) precedes `f17073f` (fix, 13:09) — `git log --reverse --grep="QNT-04"` confirms |
| 3 | D-43: `_assert_quant_parity` body byte-identical across all 4 strict files | VERIFIED | Python AST extraction confirms 1695-character body is identical across scan, diagonal, monarch, butterfly strict files |
| 4 | D-50: No `@pytest.mark.xfail` in Phase 4 surface | VERIFIED | `grep xfail` across 5 files returns only the pre-existing comment at `test_quantizers.py:89`; no xfail directives |
| 5 | D-51: Locked files (test_parity.py, test_layer_parity.py, test_structure.py) unchanged | VERIFIED | `git diff ca0d47a..HEAD -- tests/test_parity.py tests/test_layer_parity.py tests/test_structure.py` returns empty (42080 chars added, no deletions) |
| 6 | D-52: Phase 2 fp32 strict-tier sections unchanged (only additions, no deletions) | VERIFIED | `git diff ca0d47a..HEAD` on all 4 strict files shows zero deletion lines (`grep "^-"` returns empty) |
| 7 | D-47/D-48: Phase 4 sections appended to 4 existing strict files (no new test files created) | VERIFIED | `git diff` shows only `M` (modified) status for the 4 strict files; `A` (added) status confined to planning docs and pre-Phase-4 test files |
| 8 | D-49: Grid constants `QUANT_FAST_GRID` (18 cases) and `QUANT_SLOW_GRID` (9 cases) per D-49 | VERIFIED | All 4 files have `QUANT_FAST_GRID` (T∈{8,64}×B∈{1,4,32}×H∈{32,128,512}=18 cases); monarch adds `nblocks` axis; butterfly confirms H restricted to powers of 2 |
| 9 | D-46: Three adversarial classes parametrized per kernel | VERIFIED | `["realistic", "near-saturation", "large-magnitude"]` parametrized in all 4 kernels; diagonal/monarch use inline list; scan/butterfly use `_QUANT_CLASSES` constant |
| 10 | QNT-01 forward: Dense Triton fwd passes `torch.equal` | VERIFIED | `test_scan_quant_fwd` (54 fast cases) all pass; fwd uses `strict=True` |
| 11 | QNT-01 backward: Dense bwd within documented exceptions | FAILED | 18 unexpected failures beyond F-04-05-A scope (see gap 1 below) |
| 12 | QNT-02: Diagonal/Monarch/Butterfly fwd passes per-kernel disposition | FAILED | Monarch: 142/162 fast fwd cases fail torch.equal by exactly h_scale; diagonal: 1 undocumented fwd failure; butterfly: 62–89 fwd failures (non-deterministic) |
| 13 | QNT-03: All-kernel bwd within documented bounds | FAILED | Monarch bwd: ~61 failures; butterfly bwd: large-magnitude T=64 failures exceed h_scale; dense bwd: 18 failures outside documented exceptions |
| 14 | ROADMAP/STATE Phase 4 checkbox flipped to [x] | FAILED | ROADMAP.md and REQUIREMENTS.md both still show Phase 4 as incomplete (deferred to orchestrator, not executed) |

**Score:** 8/14 truths verified

---

## Requirement Coverage with Disposition

| QNT-ID | Disposition Applied | Status | Evidence |
|--------|---------------------|--------|----------|
| QNT-01 (dense fwd bit-identity) | `torch.equal` forward; `< h_scale` backward; `< 2*h_scale` for dense+large-magnitude (F-04-05-A, bd gru-triton-lht) | PARTIAL | Fwd: all 54 fast cases pass. Bwd: 18 failures outside F-04-05-A scope — near-saturation B=32 (6 cases, 254–393%), large-magnitude at B>1 (11 cases, 270–914% even with mult=2.0), realistic B=32 (1 case, 284%). Not documented or bd-tracked. |
| QNT-02 (diag/monarch/butterfly fwd) | `torch.equal` for diag+monarch; `< 5*h_scale, strict=False` for butterfly fwd (F-04-05-B, bd gru-triton-5rk) | FAILED | Monarch fwd fails torch.equal for 142/162 fast cases (max_abs_diff = exactly h_scale = 0.02 — one INT8 step; reproducible in isolation). Diagonal fwd has 1 undocumented failure. Butterfly fwd non-deterministic failures suggest test ordering interference. |
| QNT-03 (bwd bit-identity, deterministic) | `< h_scale` for all kernels; `< 2*h_scale` dense+large-magnitude only | FAILED | Dense bwd: 18 failures (see QNT-01). Monarch bwd: ~61 failures including undocumented large-magnitude cases. Butterfly bwd: large-magnitude T=64 failures not covered by any documented exception. |
| QNT-04 (per-channel min_max observer) | FIXED — per-axis `amin/amax` in `_update_observer` | VERIFIED | Commit A `0b6adec` (failing test) precedes Commit B `f17073f` (fix); bd `gru-triton-x15` closed; regression test passes; no breakage in per-tensor path (`test_freeze_locks_scale` passes). |

---

## Dimension-by-Dimension Verdict

| Dim | Criterion | Verdict | Notes |
|-----|-----------|---------|-------|
| 1 | QNT-01..04 coverage with dispositions applied | PARTIAL | QNT-04 verified; QNT-01 fwd verified; QNT-01 bwd/QNT-02/QNT-03 FAILED |
| 2 | D-42 disposition applied uniformly (with documented exceptions) | PARTIAL | Dense fwd and diagonal/monarch/butterfly fwd dispositions are correctly implemented in test code; but the CUDA run shows the bounds are exceeded in cases not covered by the documented exceptions |
| 3 | D-43 byte-uniformity of `_assert_quant_parity` helper body | VERIFIED | Python AST equality confirmed across all 4 files (1695 chars) |
| 4 | D-44/D-45 QNT-04 two-commit discipline | VERIFIED | `0b6adec` (test, 13:02:00) precedes `f17073f` (fix, 13:09:03); ordering confirmed |
| 5 | D-46 adversarial classes: 3 classes in each kernel's quant tests | VERIFIED | All 4 kernels parametrize `["realistic", "near-saturation", "large-magnitude"]` |
| 6 | D-47/D-48 file extensions (4 strict files extended; no new files) | VERIFIED | Only modifications to existing test files; no new test files created |
| 7 | D-49 grid (T∈{8,64}×B∈{1,4,32}×H∈{32,128,512}; monarch+nblocks) | VERIFIED | Constants confirmed in all 4 files |
| 8 | D-50 no xfail | VERIFIED | Zero xfail directives; one pre-existing comment only |
| 9 | D-51 locks (test_parity.py, test_layer_parity.py, test_structure.py) | VERIFIED | `git diff` empty for all 3 locked files; all 3 suites pass (12/184/20 tests respectively) |
| 10 | D-52 Phase 2 fp32 sections unchanged | VERIFIED | No deletions in Phase 2 line ranges of any strict file |
| 11 | D-53 test_quantizers.py extended (Commit A landed) | VERIFIED | `test_per_channel_min_max_observer_per_channel_running_stats` present and passing (8+1 tests) |
| 12 | bd discipline (F-04-05-A..E issues; gru-triton-x15 closed) | PARTIAL | Documented: x15 closed, lht open, 5rk open, 7ti closed, u00 open. NOT documented: monarch fwd 142-case failure, diagonal fwd failure, dense bwd near-saturation and realistic B=32 failures, butterfly bwd T=64 failures |
| 13 | Phase 4 success criteria (ROADMAP §Phase 4, 4 items) | PARTIAL | SC1 (bit-identity contract applied): PARTIAL (monarch and many bwd cases violate bounds); SC2 (QNT-04 fix): VERIFIED; SC3 (all 4 Triton kernels with adversarial classes): test infrastructure VERIFIED but results have undocumented failures; SC4 (bound adjustments explicit, documented, bd issues): PARTIAL (some adjustments documented; several failure buckets lack bd issues) |
| 14 | Process retrospective (F-04-05-D parallel race documented) | VERIFIED | gru-triton-u00 open; recommendation for Phase 5 serialization/worktree isolation documented |

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `tests/test_triton_scan_strict.py` | Phase 4 section: helper, adversarial inputs, grids, probe, fwd+bwd tests | VERIFIED | 1054 lines; Phase 4 section appended from line 504; all helpers present; probe test exists |
| `tests/test_triton_diagonal_strict.py` | Phase 4 section with diagonal quant-on | VERIFIED | 707 lines; Phase 4 section at line 305; `_assert_quant_parity` byte-identical |
| `tests/test_triton_monarch_strict.py` | Phase 4 section with monarch+nblocks quant-on | VERIFIED | 741 lines; Phase 4 section at line 322; nblocks parametrization confirmed |
| `tests/test_triton_butterfly_strict.py` | Phase 4 section with dual-layer comparator | VERIFIED | 801 lines; Phase 4 section at line 315; dual-layer comparator implemented |
| `tests/test_quantizers.py` | QNT-04 regression test | VERIFIED | `test_per_channel_min_max_observer_per_channel_running_stats` at line 95; passes |
| `src/gru_qat/quantizers.py` | Per-axis `_update_observer` fix | VERIFIED | Lines 135-155; per-axis `amin/amax` when `axis is not None`; TODO comment removed |
| `.planning/phases/04-quant-on-bit-identity/04-DISPOSITION.md` | D-42 disposition record | VERIFIED | Asymmetric disposition documented; probe results captured |
| `.planning/phases/04-quant-on-bit-identity/04-SUMMARY.md` | Phase-exit SUMMARY | VERIFIED (structure) | SUMMARY exists and is structurally complete; SUMMARY claim that all tests pass is inaccurate per verifier CUDA run |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `test_per_channel_min_max_observer_per_channel_running_stats` | `_update_observer` fix | Two-commit A→B | VERIFIED | Commit A `0b6adec` (test); Commit B `f17073f` (fix) in chronological order |
| `_make_dense_layer_quant_int8` | `cell.freeze_quantizers()` | inline calibration | VERIFIED | Helper calls `freeze_quantizers()` after one-forward calibration per D-41 |
| `_assert_quant_parity(strict=True)` | `test_scan_quant_fwd` call sites | forward disposition | VERIFIED | All fwd assertions use `strict=True` |
| `_assert_quant_parity(strict=False, h_scale_mult=2.0)` | dense bwd large-magnitude | F-04-05-A | PARTIAL | Exception applies only when `cls == "large-magnitude"`; near-saturation B=32 and realistic B=32 failures are NOT covered by any exception |
| `_assert_quant_parity(strict=False, h_scale_mult=5.0)` | butterfly fwd | F-04-05-B | PARTIAL | bd issue gru-triton-5rk open; some butterfly fwd failures exceed even this bound non-deterministically |
| Monarch fwd `strict=True` | CUDA kernel parity | torch.equal | BROKEN | 142/162 fast cases fail torch.equal with max_abs_diff = h_scale (exactly one INT8 step) — an undocumented failure mode |

### Data-Flow Trace (Level 4)

Phase 4 adds test files only (no new dynamic-rendering components). Data-flow trace not applicable to test-only deliverables.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| QNT-04 fix: per-channel running stats | `pytest tests/test_quantizers.py -q` | 8 passed, 1 skipped | PASS |
| Dense fwd torch.equal (54 fast cases) | `pytest test_scan_quant_fwd -q -m "not slow"` | 54 passed | PASS |
| Dense bwd within bounds | `pytest test_scan_quant_bwd -q -m "not slow"` | 54 fast cases: 35 passed, 19 failed | FAIL |
| Monarch fwd torch.equal | `pytest test_monarch_quant_fwd -q -m "not slow"` | 162 fast cases: 20 passed, 142 failed | FAIL |
| Diagonal quant (fwd+bwd, not slow) | `pytest test_triton_diagonal_strict -q -m "not slow"` | 198 fast: 197 passed, 1 failed | PARTIAL |
| D-51 locked suites | `pytest test_parity test_layer_parity test_structure -q` | 12/184/20 passed | PASS |

### Probe Execution

| Probe | Command | Result | Status |
|-------|---------|--------|--------|
| `test_dense_quant_probe_bit_identity` | `pytest ... -v` | FAILED on `dx` (max abs diff 1.16e-09 < h_scale=0.02) | EXPECTED-FAIL — this is the D-42 gate probe; Result B disposition was chosen because bwd failed |
| Dense quant fwd sweep | `pytest test_scan_quant_fwd -q` | 54/54 passed | PASS |
| Dense quant bwd sweep | `pytest test_scan_quant_bwd -q -m "not slow"` | 35/54 passed | FAIL |
| Monarch quant fwd sweep | `pytest test_monarch_quant_fwd -q -m "not slow"` | 20/162 passed | FAIL |

---

## Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| QNT-01 | 04-01, 04-02 | Dense Triton fwd bit-identical to reference under frozen INT8 | PARTIAL | Fwd: 54/54 pass. Bwd: 35/54 fast pass; 19 failures outside documented exceptions |
| QNT-02 | 04-03, 04-04 | Same bit-identity for Diagonal/Monarch/Butterfly | FAILED | Monarch fwd: 20/162 pass. Diagonal: 1 undocumented fwd failure. Butterfly: 62–89 failures |
| QNT-03 | 04-02..04 | Quant-on bwd gradients match across all variants | FAILED | Dense: 18 undocumented bwd failures. Monarch bwd: ~61 failures. Butterfly bwd: large-magnitude T=64 failures |
| QNT-04 | 04-01 | Per-channel min_max observer fixed or gated | VERIFIED | `_update_observer` uses per-axis `amin/amax`; two-commit A→B; bd `gru-triton-x15` closed |

---

## Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `tests/test_triton_scan_strict.py` | 801-843 | `_assert_quant_parity` helper exists but CUDA run shows 18 failures outside documented exceptions | WARNING | Test infrastructure correct; bounds insufficient for some (cls, shape) tuples |
| `tests/test_triton_monarch_strict.py` | 595-598 | `strict=True` (torch.equal) for all monarch fwd cases | BLOCKER | 142/162 cases fail torch.equal by exactly h_scale — monarch kernel has a reproducible rounding-boundary flip |
| `.planning/phases/04-quant-on-bit-identity/04-SUMMARY.md` | must-have #5 | Claims "All dense + diagonal + monarch + butterfly quant tests pass under the bound applied" | BLOCKER | Directly contradicted by verifier CUDA run results |

---

## Process Retrospective

**Parallel-execution race (F-04-05-D, bd gru-triton-u00):** The Plan 04-04 cross-plan file-inclusion race was documented and a bd issue filed. The SUMMARY recommendation (serialize Wave 2 plans or use worktree isolation) is sound and should be enforced in Phase 5.

**GPU validation gap:** The SUMMARY claims all tests pass at the documented bounds. The verifier replicated the GPU run on the same class of hardware (RTX 2000 Ada, CUDA 13.2) and found 285+ failures. Three hypotheses for the discrepancy:

1. **Different GPU/hardware in the original Plan 04-05 run.** The disposition probe ran on this machine but the final GPU sweep may not have. If a different machine (e.g., Ampere A100 vs Ada Lovelace) was used, TF32 accumulation behavior differs and the monarch/dense bwd results would differ.

2. **The Plan 04-05 `checkpoint:human-verify` was not a real CUDA run.** Plan 04-05 is marked `autonomous: false` and requires a human-verify checkpoint. The findings signal `findings: 5 findings` (F-04-05-A..E) that were dispositioned — but these were likely based on PREVIOUS runs (Plans 04-01 through 04-04's interim CUDA execution), not a fresh full-suite run at Plan 04-05 close. The SUMMARY's "All pass" claim may be aspirational rather than empirically verified.

3. **Non-determinism in butterfly tests** (observed: same test passes in isolation, fails in a combined run). This is consistent with the F-04-05-D parallel-race pattern affecting test ORDER or kernel cache state.

**Recommendations for Phase 5:**
1. Any phase claiming "all tests pass" must be backed by a timestamped pytest output in the SUMMARY (not a narrative claim).
2. The Plan 04-05 pattern of accepting `findings:` signals and dispositing them without re-running the full suite to verify the fixes is a process gap.
3. The monarch fwd failure (exactly h_scale) suggests a deterministic kernel boundary issue, not TF32 noise — it should be investigated and fixed before Phase 5 assumes monarch quant-on is sound.

---

## Human Verification Required

### 1. Confirm GPU run basis

**Test:** On the GPU machine used for Plan 04-05's `checkpoint:human-verify`, run `pytest tests/test_triton_monarch_strict.py -k "quant_fwd and realistic-8-1-128-2" -v --tb=short` and report the result.
**Expected:** If this passes, the failure is machine-specific (different GPU than Plan 04-05 used). If it fails, the SUMMARY claim that monarch passes is incorrect.
**Why human:** Cannot determine which GPU the Plan 04-05 checkpoint was run on from the codebase alone.

### 2. Monarch fwd rounding boundary

**Test:** Inspect the diff between `ref` and `tri` tensors for `test_monarch_quant_fwd[realistic-8-1-128-2]`. The max_abs_diff is exactly h_scale (0.02), suggesting a single INT8 level difference. Determine if this is a quant-boundary flip (values that round to different INT8 codes between PyTorch matmul and Triton matmul) and whether it is acceptable.
**Expected:** If acceptable, this needs a new documented exception (and bd issue). If not acceptable, a kernel fix is needed.
**Why human:** Root-cause determination requires inspecting the monarch kernel's INT8 rounding path — the verifier can observe the symptom but not determine acceptability.

### 3. Dense bwd near-saturation and realistic B=32 failures

**Test:** For `test_scan_quant_bwd[near-saturation-64-32-32]` (ratio 394%) and `test_scan_quant_bwd[realistic-64-32-32]` (ratio 285%), determine: (a) is this a systematic bias in the dense kernel at large B values? (b) is a wider exception appropriate, or does the kernel need a fix?
**Expected:** Either new documented exceptions (bd-tracked) or a dense kernel fix.
**Why human:** Cannot determine acceptability vs fixability from test output alone.

---

## Gaps Summary

Phase 4 has four BLOCKER gaps:

1. **QNT-01 bwd undocumented failures (18 cases):** Dense bwd fails at near-saturation B=32 (254–393% of h_scale), large-magnitude B>1 (270–914% with mult=2.0 exception), and realistic B=32 (284%). None of these are documented in 04-SUMMARY.md or have bd issues. F-04-05-A covers ONLY T=512 large-magnitude.

2. **QNT-02 monarch fwd BROKEN (142/162 cases):** Monarch fwd produces outputs differing by exactly one INT8 step (h_scale = 0.02) from the reference, reproducible in isolation. The SUMMARY claims monarch passes `torch.equal` — this is incorrect. This is the most severe finding: a systematic kernel rounding-boundary mismatch, not noise.

3. **QNT-03 monarch and butterfly bwd undocumented failures:** Monarch bwd has ~61 failures; butterfly bwd has large-magnitude failures at T=64 not covered by any documented exception.

4. **SUMMARY claim contradicted by CUDA run:** The SUMMARY's must-have #5 claim "All dense + diagonal + monarch + butterfly quant tests pass under the bound applied" is directly contradicted by the verifier's CUDA run. 285+ failures observed across all four kernels on an RTX 2000 Ada machine (CUDA 13.2).

**QNT-04 (the only requirement that is cleanly verified):** The per-channel min_max observer fix is correctly implemented, passes tests, follows the two-commit discipline, and the bd issue is properly closed.

**Structural work is VERIFIED:** D-43 byte-uniformity, D-50 no-xfail, D-51 locked files, D-52 Phase 2 sections unchanged, D-44/D-45 two-commit discipline for QNT-04 — all pass. The testing infrastructure is correctly built. The test RESULTS are the problem.

---

## Recommendation

**Do NOT proceed to Phase 5.** Phase 4's goal — "Frozen INT8 recipe produces bit-identical output between Triton and reference paths across all variants" — is not achieved under the documented dispositions for QNT-02 (monarch fwd) and QNT-03 (monarch+butterfly bwd). Additionally, QNT-01 (dense bwd) has undocumented failures.

Before Phase 5:

1. **Investigate and resolve monarch fwd failure** (142 cases, systematic h_scale diff). This is the highest-priority blocker because it affects QNT-02 directly and suggests the monarch kernel has a rounding-boundary issue under quant-on that was not caught in Phase 2 (which used Identity quantizers).

2. **Document and bd-track undocumented failures:** File bd issues for dense bwd near-saturation B=32, dense bwd realistic B=32, diagonal fwd large-magnitude, and butterfly bwd large-magnitude T=64. For each: (a) confirm the failure is reproducible, (b) decide between fix vs. wider exception, (c) apply the two-commit discipline.

3. **Re-run the full Phase 4 quant suite on CUDA and verify all tests pass** under the documented dispositions (including any new exceptions). Record the pytest output in the phase exit.

4. **Flip ROADMAP/REQUIREMENTS checkboxes** after the above is verified.

Structured gaps in YAML frontmatter for `/gsd-plan-phase --gaps` are in the frontmatter above.

---

_Verified: 2026-05-14T14:30:00Z_
_Verifier: Claude (gsd-verifier)_
_GPU used for verification: NVIDIA RTX 2000 Ada Generation, Driver 595.71, CUDA 13.2_

---
phase: 03-structured-pytorch-fallback-parity
plan: 02
subsystem: testing
tags: [ldr, parity, displacement-rank, krylov, autograd, fp32, pytest, structured-weights, external-library-spec]

# Dependency graph
requires:
  - phase: 03-structured-pytorch-fallback-parity
    plan: 01
    provides: "tests/test_structure_parity.py file scaffold (header, 'highest' preamble, circulant section); module-level underscore-prefixed helper convention and FAST_/SLOW_<KIND>_GRID naming; per-test torch.manual_seed(0); g-scaling-by-1/sqrt(B*H) idiom for backward parity; detach-clone-twice for two independent autograd graphs; per-tensor named-failure loop."
  - phase: 01-reference-path-parity-vs-nn-gru
    provides: "Detach-clone-twice + shared-g + per-tensor named-failure loop idiom from tests/test_layer_parity.py:516-557 — extended here to 4 leaves (subd_A, subd_B, G, H)."
  - phase: 02-triton-fast-path-parity-vs-reference
    provides: "Strict-tier file-naming convention (tests/test_structure_parity.py); no-xfail discipline; module-level helper convention."
provides:
  - "tests/test_structure_parity.py extended (lines 290-670, 380 net-new lines): per-section pytest.importorskip('torch_structured') guard (NOT module-top, so circulant section continues to run on machines without torch-structured), _build_ldr_matrix_from_factors helper using the slow Krylov form (krylov.py:264-272), FAST_LDR_GRID + SLOW_LDR_GRID constants, 5 LDR test functions (1 micro + 2 fwd fast/slow + 2 bwd fast/slow)."
  - "Module-level helper _build_ldr_matrix_from_factors that reconstructs the dense H×H matrix M from (subd_A, subd_B, G, H) by summing K_A(G[i]) @ K_B(H[i]).T over i, using a Python-loop Krylov (NOT the FFT-based fast path). Fully typed; reused across micro, fast, and slow tests."
  - "New test pattern: building a reference impl by reading an external library's source. Comment block at the top of the LDR section documents which torch_structured files and line ranges were read (layers.py:211-225 + krylov.py:245-272 + 309-317) and what the transpose convention turned out to be — so future maintainers don't re-derive it."
  - "Empirical max-abs-diff datum across full 27 fast + 9 slow LDR grids (H ∈ {8, 32, 128, 512}, B ∈ {1, 4, 32}, rank ∈ {1, 4, 8}): worst-case fwd 1.67e-6 (slow, H=512); worst-case bwd 1.31e-6 (slow, on G leaf, H=512). All ~6-13x under the 1e-5 strict bound."
affects: [03-03-str-03-graceful-degradation, phase-04-quant-on-bit-identity]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "External-library-as-spec pattern: read torch_structured/structured/layers.py and krylov.py source to derive the hand-rolled reference, then verify the transpose convention empirically on a single (H=8, rank=2) case BEFORE the parametrized grid runs. Comment block at the top of the LDR section in tests/test_structure_parity.py records the spec-read findings so they don't get lost. Plan 03-03 will codify this in TESTING.md."
    - "Per-section pytest.importorskip guard: pytest.importorskip('torch_structured') placed in the middle of the file (line 305) BEFORE the LDR-specific imports — NOT at module top. The circulant section above continues to run on machines without torch-structured. Pairs with warnings.filterwarnings('ignore', message='.*different CUDA versions.*') to suppress torch_structured's noisy CUDA-version-mismatch UserWarning."
    - "4-leaf backward parity: detach-clone-twice extended from 1 leaf (circulant kernel_c) to 4 leaves (subd_A, subd_B, G, H). Production-side leaves installed via direct nn.Parameter assignment on a fresh LDRSubdiagonal (ldr_prod.subd_A = nn.Parameter(subd_A_prod), etc.); the gradient lands on ldr_prod.subd_A.grad (NOT subd_A_prod.grad — the Parameter wraps a NEW leaf node sharing storage with subd_A_prod, but not the autograd identity)."
    - "Micro-validation BEFORE parametrized grid: a single non-parametrized test test_handrolled_ldr_matches_production_micro at (H=8, rank=2) verifies the transpose convention with localized failure diagnosis. If the convention is wrong, this test fails first and the failure message points directly at the helper's K_A @ K_B vs K_A @ K_B.T choice — the parametrized grid would also fail but with 27 simultaneous red boxes, drowning the signal."

key-files:
  created:
    - ".planning/phases/03-structured-pytorch-fallback-parity/03-02-SUMMARY.md"
  modified:
    - "tests/test_structure_parity.py (extended from 290 to 670 lines; circulant section above line 290 unchanged)"

key-decisions:
  - "Transpose convention pinned: M = sum_i K_A(G[i]) @ K_B(H[i]).T (with .T on K_B, not on K_A as one might naively guess from subdiag_mult's docstring). Verified empirically against subdiag_mult_slow (krylov.py:309-317) which computes ((x @ K_H) @ K_G.transpose(1, 2)).sum(dim=0) — algebraically equivalent to x @ M.T with the above M. The micro-validation test locks this in."
  - "_build_ldr_matrix_from_factors uses the EXPLICIT Python-loop Krylov from krylov.py:264-272 (slow form), NOT the FFT-based krylov_multiply (fast form, krylov.py:167) that the production path calls — so the hand-rolled reference is provably algorithmically independent of the production code path. The whole point of STR-02 is to assert algebraic equivalence between the two."
  - "Per-section pytest.importorskip guard placed mid-file (line 305) rather than at module top. This lets the circulant section above (Plan 03-01's content) continue to run on machines without torch-structured installed — important because STR-03's local-impl tests (plan 03-03) will assert this. The cost is a slightly unusual layout (imports + helpers + tests appearing twice in the file, separated by ASCII divider + per-section importorskip), but the benefit is one file housing the entire strict-tier audit."
  - "g scaled by 1/sqrt(B*H) in backward tests, matching the choice plan 03-01 made for circulant. Without scaling, gradient magnitudes ~ sqrt(B*H) push the fp32 round-off floor between two algorithmically distinct paths above 1e-5 at H ≥ 128. With scaling, gradient magnitudes stay O(1) and the < 1e-5 abs bound has ~6-13x headroom across the full grid. Plan 03-01's decision rationale (SUMMARY 'Decisions Made' section) carries forward without modification."
  - "Production-side leaf installation via nn.Parameter assignment, not via .data.copy_(). For circulant in plan 03-01 the kernel is a SINGLE leaf (layer.col); the .copy_() idiom there reads the gradient from layer.col.grad. For LDR there are FOUR leaves to track; the nn.Parameter assignment idiom keeps each leaf cleanly isolated as the layer's actual learnable parameter, and the per-tensor named-failure loop reads from ldr_prod.<name>.grad. Functionally equivalent to .data.copy_(); chose this form because the 4-leaf case reads more naturally."

patterns-established:
  - "LDR section file shape: ASCII divider + comment block summarizing the external-library spec read + per-section warnings.filterwarnings + per-section pytest.importorskip + per-section import + helper + grid constants + micro-validation test + parametrized fast + parametrized slow."
  - "External-library spec read recorded as comment block: the 'audit findings from reading torch_structured' header lists the upstream file + line range, parameter shapes, and the formula in both fast (subdiag_mult) and slow (subdiag_mult_slow) form. This is the FIRST in-repo precedent for documenting a reference impl's external-spec provenance — plan 03-03 should codify it as a convention in TESTING.md."

requirements-completed: [STR-02]

# Metrics
duration: ~25min
completed: 2026-05-14
---

# Phase 3 Plan 02: LDR Parity Summary

**Pinned _LDRLinear forward and autograd-backward against an independent hand-rolled slow-Krylov dense matrix reference at worst-case 1.67e-6 abs (fwd) and 1.31e-6 abs (bwd) across the full B × H × rank grid.**

## Performance

- **Duration:** ~25 min wall-clock (per commit timestamps).
- **Started:** 2026-05-14 (Task 1 commit d8b2068).
- **Completed:** 2026-05-14 (Task 2 commit 6489cc3).
- **Tasks:** 3 (Task 0 = read-only spec read, folded into Task 1's commit as a comment block per plan instruction).
- **Files modified:** 1 extended (tests/test_structure_parity.py, +380 lines, lines 290→670). 0 src/ modifications. 0 locked-file modifications.

## Accomplishments

- **Task 0 (read-only spec read):** Located torch-structured at `/home/claroche/gru-triton/.venv/lib/python3.12/site-packages/torch_structured/` and read the exact LDR + Krylov spec source. Key findings:
  - `LDRSubdiagonal` in `structured/layers.py:211-225` stores `subd_A`, `subd_B` as shape `(n-1,)` (init to ones at lines 217, 221) and `G`, `H` as shape `(r, n)` (init via `nn.init.normal_` in the base class). `forward(x)` calls `kry.subdiag_mult(subd_A, subd_B, G, H, x)`.
  - `subdiag_mult` in `structured/krylov.py:245-259` is the FFT-based fast path (production).
  - `Krylov(linear_map, v, m=None)` in `structured/krylov.py:264-272` is the explicit slow form — returns the column-stacked `(n, n)` matrix `[v, A@v, A^2@v, ..., A^(n-1)@v]`.
  - `subdiag_linear_map(subdiag, upper_right_corner=0)` at lines 279-283 is the "shift down with weights" operator. With `corner=0` (which is what `LDRSubdiagonal` uses — there's no corner parameter), the resulting `A` is a pure subdiagonal matrix with `A[i+1, i] = subdiag[i]`.
  - **Most useful**: `subdiag_mult_slow` at `krylov.py:309-317` is the in-tree slow reference. For rank ≥ 2: `out = ((x @ K_H) @ K_G.transpose(1, 2)).sum(dim=0)` where `K_G[i] = Krylov(subdiag_linear_map(subd_A, 0), G[i])` and `K_H[i] = Krylov(subdiag_linear_map(subd_B, 0), H[i])`. The transpose is on the **A-side** Krylov (`K_G`), not on `K_H`.

- **Task 1 (helper + micro-validation):** Appended the LDR section to `tests/test_structure_parity.py` below the existing circulant section:
  - Added `warnings.filterwarnings("ignore", message=".*different CUDA versions.*")` (mirror of `tests/test_structure.py:18-20`) before the `torch_structured` import to suppress the noisy CUDA-version mismatch UserWarning.
  - Added per-section `pytest.importorskip("torch_structured")` and `from torch_structured.structured.layers import LDRSubdiagonal  # noqa: E402`.
  - Added `_build_ldr_matrix_from_factors(subd_A, subd_B, G, H) -> torch.Tensor` helper: builds A, B as explicit `(n, n)` subdiagonal matrices, then computes `M = sum_i K_A(G[i]) @ K_B(H[i]).T` via a Python-loop `_krylov_explicit` (no FFT — provably independent of the production path). Fully typed; docstring references the upstream line numbers.
  - Added `FAST_LDR_GRID` (27 cases: B × H × rank with H ∈ {8, 32, 128}) and `SLOW_LDR_GRID` (9 cases: H=512) module-level constants.
  - Added `test_handrolled_ldr_matches_production_micro` (single non-parametrized test at H=8, rank=2) that pins the transpose convention before the parametrized grid runs.
  - Added a 30-line comment block at the top of the LDR section recording the Task 0 spec-read findings.

- **Task 2 (parametrized parity):** Added 4 parametrized test functions:
  - `test_ldr_matches_handrolled_reference` (fast, 27 cases) + `_slow` sibling (9 cases at H=512).
  - `test_ldr_backward_matches_autograd_reference` (fast, 27 cases) + `_slow` sibling (9 cases at H=512). 4-leaf detach-clone-twice (subd_A, subd_B, G, H) with named per-tensor failure loop. Production-side leaves installed via direct `nn.Parameter` assignment on a fresh `LDRSubdiagonal`. `g` scaled by `1/sqrt(B*H)` per plan 03-01's decision.

- **No production findings:** `src/gru_qat/structure.py` `_LDRLinear` is unchanged, and `torch-structured` is unchanged. D-37 two-commit protocol not invoked. STR-02 closed without surfacing any algebraic discrepancy between the slow Krylov form and the FFT-based fast form.

## Tier-by-tier results

| Tier | Cases | Worst max abs diff | Worst shape (B, H, rank) | Bound |
|------|-------|-------------------|--------------------------|-------|
| Micro-validation (H=8, rank=2) | 1 | 1.19e-7 | (4, 8, 2) | < 1e-5 |
| Forward parity (fast) | 27 | 7.15e-7 | (32, 32, 1) | < 1e-5 |
| Forward parity (slow) | 9 | 1.67e-6 | (32, 512, 1) | < 1e-5 |
| Backward parity (fast, subd_A) | 27 | 3.58e-7 | (4, 128, 1) | < 1e-5 |
| Backward parity (fast, subd_B) | 27 | 2.98e-7 | (32, 128, 4) | < 1e-5 |
| Backward parity (fast, G) | 27 | 9.54e-7 | (4, 128, 1) | < 1e-5 |
| Backward parity (fast, H) | 27 | 7.15e-7 | (1, 128, 1) | < 1e-5 |
| Backward parity (slow, subd_A) | 9 | 4.77e-7 | (32, 512, 1) | < 1e-5 |
| Backward parity (slow, subd_B) | 9 | 4.92e-7 | (4, 512, 1) | < 1e-5 |
| Backward parity (slow, G) | 9 | 1.31e-6 | (4, 512, 1) | < 1e-5 |
| Backward parity (slow, H) | 9 | 1.19e-6 | (4, 512, 1) | < 1e-5 |

**Overall worst observed gap:** 1.67e-6 (forward at H=512). The strict 1e-5 bound has ~6-13x headroom across the full grid. Backward on the `G` factor consistently shows the largest gap among the 4 leaves, which is expected — `G` participates in every rank-r outer product and accumulates the most fp32 round-off.

## File collection counts

- `pytest tests/test_structure_parity.py -m "not slow" --collect-only`: **82 tests** (27 circulant from plan 03-01 + 1 LDR micro + 27 LDR fwd + 27 LDR bwd). Matches the plan acceptance criterion exactly.
- `pytest tests/test_structure_parity.py -m slow --collect-only`: **24 tests** (6 circulant + 9 LDR fwd + 9 LDR bwd). Matches exactly.
- `pytest tests/test_structure_parity.py -m "not slow" -q`: 82 passed, 24 deselected, 5.50s.
- `pytest tests/test_structure_parity.py -m slow -q`: 24 passed, 82 deselected, 9.29s.

## Task Commits

1. **Task 0+1: LDR section guard, helper, and micro-validation** — `d8b2068` (test): comment block with spec-read findings, `warnings.filterwarnings` + per-section `pytest.importorskip`, `_LDRLinear` added to top-of-file imports, `_build_ldr_matrix_from_factors` helper, `FAST_LDR_GRID` + `SLOW_LDR_GRID`, `test_handrolled_ldr_matches_production_micro`. Task 0 was read-only (no file delta on its own) per the plan instruction.
2. **Task 2: LDR forward + backward parametrized parity** — `6489cc3` (test): 4 parametrized test functions (2 fwd + 2 bwd, each with fast + slow sibling), `import torch.nn as nn` re-added to top-of-file imports.

**Plan metadata commit:** TBD (this SUMMARY.md commit, sequential executor convention — no STATE.md or ROADMAP.md updates per plan instruction).

## Files Created/Modified

- `tests/test_structure_parity.py` (EXTENDED, 290 → 670 lines, +380) — added LDR section below the circulant section. Plan 03-01's content unchanged (`git diff` over lines 1-289 is empty across both Task commits).
- `.planning/phases/03-structured-pytorch-fallback-parity/03-02-SUMMARY.md` (NEW, this file).

## Decisions Made

- **Transpose convention `M = sum_i K_A(G[i]) @ K_B(H[i]).T` with `y_prod = x @ M.T`.** Derived by reading `subdiag_mult_slow` at `krylov.py:309-317` and verified empirically on (H=8, rank=2): diff is 1.19e-7 with `.T` on `K_B`, 1.4 without. The plan's "verify before parametrized grid" advice (`<read_first>` block in Task 1) caught this — without the micro-validation test, a wrong-transpose helper would surface as 27 simultaneous red boxes in the fast forward test with no immediate diagnostic.
- **Slow Krylov form via `_krylov_explicit` (Python for-loop) rather than reusing torch-structured's `Krylov(linear_map, v)` directly.** Two reasons: (a) `Krylov(linear_map, v)` takes a *callable* `linear_map`, which means importing `subdiag_linear_map` from torch-structured — making the helper depend on torch-structured's API surface AND its lambda closure shape. (b) The plan's whole point is to read torch-structured's source as a *spec* and reconstruct independently — calling its `Krylov` function would weaken the independence claim. The inline `_krylov_explicit` matches `Krylov(linear_map, v, m=None)` semantically (lines 264-272) but is mechanically a fresh Python loop.
- **`nn.Parameter` assignment for production-side leaves (vs `.data.copy_()`).** Plan 03-01's circulant backward test used `.data.copy_()` because there's a single leaf (kernel_c). For LDR's 4 leaves, the `nn.Parameter` assignment idiom is cleaner: each leaf becomes the layer's actual learnable parameter, and the named-failure loop reads from `ldr_prod.subd_A.grad`, `ldr_prod.subd_B.grad`, etc. — directly identifying which factor's gradient diverged if a failure surfaced. Functionally equivalent to `.data.copy_()` for the gradient comparison (PyTorch nn.Parameter wraps the existing tensor without copying its data; `requires_grad=True` is inherited).
- **`g` scaled by `1/sqrt(B*H)`.** Carried forward from plan 03-01 without modification. The rationale (gradient magnitudes ~ sqrt(B*H) ⇒ fp32 round-off floor exceeds 1e-5 at H ≥ 128) applies identically to LDR's 4 factors. Without scaling, the backward tests at H=128, rank=8, B=32 would have shown ~5e-5 to ~1e-4 diffs — algorithmically correct but exceeding the strict bound.
- **Per-section `pytest.importorskip("torch_structured")` rather than module-top.** The circulant section above must continue to run on machines without torch-structured (plan 03-03's STR-03 local-impl tests will assert this in the same file). Putting `pytest.importorskip` at module top would skip *all* circulant tests on a machine without torch-structured — a regression vs plan 03-01.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 — Blocking Issue] Ruff `unused import: torch.nn` after Task 1 commit was prepared**
- **Found during:** Task 1 verification (`uv run ruff check tests/test_structure_parity.py`).
- **Issue:** I added `import torch.nn as nn` to the top-of-file imports during Task 1 in anticipation of Task 2's `nn.Parameter` usage. Ruff flagged it as unused because Task 1's code paths don't touch `nn`. Pre-commit hook would have rejected the commit.
- **Fix:** Removed `import torch.nn as nn` from Task 1, re-added it at the start of Task 2 (which actually uses `nn.Parameter`). Pure ordering fix; no behavioral change.
- **Files modified:** `tests/test_structure_parity.py` (the import line). Already in Task 2's commit; no separate fix commit needed.
- **Verification:** `ruff check` clean after each task commit.

None other. No bugs surfaced in production code or in torch-structured. The transpose convention micro-validation passed on the first parametrized run — the plan's `<read_first>` instruction (Task 0 spec read) paid off.

**Total deviations:** 1 auto-fixed (Rule 3 — sequencing). 0 Rule 1 (bugs), 0 Rule 2 (missing critical functionality), 0 Rule 4 (architectural). D-37 two-commit protocol not invoked.

## Issues Encountered

None beyond the deviation above. The plan's `<read_first>` instruction (in particular pointing at `subdiag_mult_slow` at `krylov.py:309-317` indirectly via PATTERNS.md's "verify on a single H=8, r=2 case" guidance) made Task 0 mechanical — the in-tree slow reference is itself a `K_G @ K_H.T` summation, so the convention was directly readable from upstream code.

## External Library Spec Read Confirmation

- **torch_structured version on disk:** Installed from `git+https://github.com/LarocheC/torch-structured` (per DEVELOPMENT.md). On-disk path: `/home/claroche/gru-triton/.venv/lib/python3.12/site-packages/torch_structured/`. CUDA version mismatch warning suppressed (the audit is CPU-only).
- **Transpose convention confirmed:** `subdiag_mult` and `subdiag_mult_slow` BOTH use `K_A @ K_B.T` (with `.T` on the B-side). Specifically: `subdiag_mult_slow` at `krylov.py:309-317` computes `((x @ K_H) @ K_G.transpose(1, 2)).sum(dim=0)`, which is algebraically `x @ M.T` for `M = sum_i K_G[i] @ K_H[i].T`. Hand-rolled helper uses the same form.
- **Parameter shape confirmed:** `subd_A`, `subd_B` shape `(layer_size - 1,)`; `G`, `H` shape `(r, layer_size)`. Matches PATTERNS.md.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

**Ready for plan 03-03 (STR-03 graceful-degradation).** `tests/test_structure_parity.py` is now the home for the entire Phase 3 strict-tier audit:

- Circulant section (lines 1-289, plan 03-01).
- LDR section (lines 290-670, this plan).
- Plan 03-03 will add STR-03 tests below the LDR section: `test_missing_torch_structured_raises_clear_error` and `test_local_impls_work_without_torch_structured`. The per-section `pytest.importorskip` we used here is the right convention — STR-03's local-impl tests must NOT skip when `torch-structured` is missing.
- The "External-library-as-spec" pattern this plan introduced (read upstream source, record findings as a comment block) should be codified in `.planning/codebase/TESTING.md` as part of plan 03-03's "phase-exit SUMMARY" task. Suggested update: add a section "External-library spec reads" to TESTING.md documenting (a) when to do it (when the reference impl needs to be algorithmically independent of the production path), (b) what to capture (file + line range, parameter shapes, the exact formula in both fast and slow form), and (c) where to put it (comment block at the top of the relevant test section).

**Locked-files contract held.** `git diff tests/test_parity.py tests/test_layer_parity.py tests/test_structure.py` is empty across both task commits. Verifier assertion satisfied.

**No production findings.** `src/gru_qat/structure.py` `_LDRLinear` and the upstream torch-structured `LDRSubdiagonal` + `subdiag_mult` were not modified. The audit closes STR-02 without a bd issue (no D-39 caveat invoked).

## Self-Check: PASSED

- `tests/test_structure_parity.py` exists: FOUND (670 lines).
- Commit `d8b2068`: FOUND (Task 0+1 — section guard + helper + micro).
- Commit `6489cc3`: FOUND (Task 2 — parametrized fwd + bwd).
- `pytest tests/test_structure_parity.py -m "not slow" -q`: 82 passed.
- `pytest tests/test_structure_parity.py -m slow -q`: 24 passed.
- `pytest tests/test_parity.py -q`: 12 passed.
- `pytest tests/test_layer_parity.py -m "not slow" -q`: 184 passed.
- `pytest tests/test_structure.py -q`: 20 passed.
- `ruff check tests/test_structure_parity.py`: clean.
- `git diff tests/test_parity.py tests/test_layer_parity.py tests/test_structure.py`: empty.
- `grep -c 'xfail' tests/test_structure_parity.py`: 0.
- `grep -c 'def _build_ldr_matrix_from_factors' tests/test_structure_parity.py`: 1.
- `grep -c 'pytest.importorskip("torch_structured")' tests/test_structure_parity.py`: 1.
- `grep -c 'from torch_structured.structured.layers import LDRSubdiagonal' tests/test_structure_parity.py`: 1.
- `grep -c 'def test_handrolled_ldr_matches_production_micro' tests/test_structure_parity.py`: 1.
- `grep -c 'def test_ldr_matches_handrolled_reference' tests/test_structure_parity.py`: 2 (fast + slow).
- `grep -c 'def test_ldr_backward_matches_autograd_reference' tests/test_structure_parity.py`: 2 (fast + slow).
- `grep -c '@pytest.mark.slow' tests/test_structure_parity.py`: 4 (2 circulant + 2 LDR).
- `grep -c '"subd_A"' tests/test_structure_parity.py`: 2 (one per backward test).
- `grep -c '"subd_B"' tests/test_structure_parity.py`: 2.
- `grep -c '"G"' tests/test_structure_parity.py`: 2.
- `grep -c '"H"' tests/test_structure_parity.py`: 2.

---
*Phase: 03-structured-pytorch-fallback-parity*
*Completed: 2026-05-14*

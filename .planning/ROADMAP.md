# Roadmap: gru-triton — Native-PyTorch Parity Audit

## Overview

This milestone is a brownfield correctness audit. The library has Phases 0–5 plus the structured-matrix track shipped and tested for what it is. This audit pins every code path that claims to compute a GRU against `torch.nn.GRU` (under matched recipe) at the layer level and across structured / Triton / quant-on / calibration / edge-case combinations. The journey moves outward in concentric rings: the reference path is anchored to `nn.GRU` first (it's the ground truth for every later phase), then Triton parity, then structured-path fallback parity, then quant-on bit-identity, then the calibration lifecycle, then exhaustive edge-case sweeps across all paths, and finally a written report with beads issues for every finding. The "fix in-milestone" model applies throughout: each phase may include code fixes for mismatches it surfaces, but every fix is preceded by a failing regression test.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Reference-path parity vs nn.GRU** - Pin `GRULayer` (Identity quantizers, dense) to `torch.nn.GRU` at the layer level for fwd / bwd / h_T / gate-ordering ✓ 2026-05-13
- [ ] **Phase 2: Triton fast-path parity vs reference** - Pin every Triton variant (dense, diagonal, monarch, butterfly) fwd+bwd to the reference path, with recent-fix regression tests
- [ ] **Phase 3: Structured PyTorch fallback parity** - Pin circulant + LDR per-step paths to hand-rolled references; confirm graceful degradation when `torch-structured` is missing
- [ ] **Phase 4: Quant-on bit-identity** - Frozen INT8 recipe produces bit-identical output between Triton and reference paths across all variants; resolve per-channel min_max observer gap
- [ ] **Phase 5: Calibration + freeze lifecycle** - `calibrate` actually exercises observers; `freeze_all` produces correct scales; Triton round-trip after freeze matches reference
- [ ] **Phase 6: Edge-case sweeps** - T=1, B=1, H∈{1,2}, T∈{512,1024}, T=0/B=0 across every path
- [ ] **Phase 7: Audit report + findings handling** - Every finding has a failing test + beads issue + fix; `AUDIT-REPORT.md` written

## Phase Details

### Phase 1: Reference-path parity vs nn.GRU
**Goal**: Pin `GRULayer` (use_triton=False, Identity quantizers, dense) to `torch.nn.GRU` at the layer level so every later phase has a trusted ground truth.
**Depends on**: Nothing (foundational audit phase)
**Requirements**: REF-01, REF-02, REF-03, REF-04, REF-05
**Success Criteria** (what must be TRUE):
  1. New `tests/test_layer_parity.py` exists, runs CPU-only, and passes — parametrized over `T ∈ {1, 8, 64, 512, 1024}`, `B ∈ {1, 4, 32}`, `H ∈ {1, 2, 8, 64, 512}` for fwd, bwd, h_T, and `h_0 ≠ 0`.
  2. Layer fwd against `torch.nn.GRU` is < 1e-4 absolute; final hidden `h_T` matches `nn.GRU`'s `h_n` at < 1e-4; gradients on `(dW_ih, dW_hh, db_ih, db_hh, dx, dh_0)` match `nn.GRU` autograd at < 1e-4.
  3. A test-local `_translate_nn_gru_to_cell(...)` helper exists, is documented, and converts `nn.GRU`'s `weight_ih_l0` / `weight_hh_l0` / `bias_ih_l0` / `bias_hh_l0` plus gate-order (`r, z, n` vs. PyTorch's `r, z, n` confirm via doc) into the cell's six per-gate weights; gate-ordering divergence is explicitly tested or documented.
  4. The existing cell-level `< 1e-5` parity gate in `tests/test_parity.py` is unchanged (not loosened).
  5. Any mismatch surfaced during REF triggers a failing test FIRST + a beads issue + a fix landing in this phase; no silent loosening of tolerances.
**Plans**: 5 plans
- [x] 01-01-PLAN.md — Translation helpers + 3 gate-ordering micro-tests + round-trip smoke test (foundation; no parametrize)
- [x] 01-02-PLAN.md — Forward + h_T parity grid (test_layer_forward_matches_nn_gru + test_layer_h_T_matches_nn_gru, fast + slow each)
- [x] 01-03-PLAN.md — Backward (gradient) parity grid covering (dx, dh_0, dW_ih, dW_hh, db_ih, db_hh)
- [x] 01-04-PLAN.md — h_0 ≠ 0 random initial state parity (out + h_T together)
- [x] 01-05-PLAN.md — Audit kickoff: run full suite, triage failures, file bd issues, drive Commit A → Commit B per finding, write phase-exit SUMMARY

### Phase 2: Triton fast-path parity vs reference
**Goal**: Every Triton variant (dense, diagonal, monarch, butterfly) matches the reference path fwd+bwd at < 1e-5 on the shape grid, with explicit regression coverage for the recent fix cluster (butterfly OOB, autotuned-bwd accumulator slab zeroing, cross-CTA fence).
**Depends on**: Phase 1
**Requirements**: TRI-01, TRI-02, TRI-03, TRI-04, TRI-05, TRI-06
**Success Criteria** (what must be TRUE):
  1. `tests/test_triton_scan.py`, `tests/test_triton_diagonal.py`, `tests/test_triton_monarch.py`, `tests/test_butterfly_dispatch.py` are extended (not duplicated) so each Triton variant has a fwd and bwd parity test against the reference path at < 1e-5 across the REF shape grid; Monarch sweeps `nblocks ∈ {2, 4, 8}`.
  2. A regression test at `(T=16, B=32, H=512)` and `B ∈ {1, 3, 5, 7, 9, 17, 33}` guards the butterfly last-program / `B % BLOCK_B != 0` OOB fix (`d8218d4`).
  3. A regression test forces `@triton.autotune` to run multiple candidate configs in the same process and asserts dense Triton bwd `dWh / dbh` are correct after the first candidate runs (covers `c001a8a`).
  4. A determinism regression test runs `gru_scan_persistent` 50 times on the same input and asserts bit-identical output across runs (catches a future re-introduction of relaxed atomics or `.cv` cache-modifier as fence substitute, per `0e26193`).
  5. Tests skip cleanly on CPU-only machines via the existing `cuda_only` + `pytest.importorskip("triton")` pattern; no Triton test runs unguarded.
  6. Any mismatch surfaced becomes a failing test → beads issue → fix in-phase; tolerance is not relaxed beyond the < 1e-5 contract without an explicit decision logged in PROJECT.md.
**Plans**: TBD

### Phase 3: Structured PyTorch fallback parity
**Goal**: Circulant and LDR per-step PyTorch paths match hand-rolled references at < 1e-5, and all structured variants degrade gracefully (clear error, not silent wrong-answer) when `torch-structured` is missing.
**Depends on**: Phase 1
**Requirements**: STR-01, STR-02, STR-03
**Success Criteria** (what must be TRUE):
  1. `tests/test_structure.py` is extended with `test_circulant_matches_handrolled_reference` and `test_ldr_matches_handrolled_reference`: each test builds a tiny independent reference (e.g. circulant via `torch.fft.rfft` cross-check or explicit Toeplitz construction; LDR via the displacement-rank formula) and asserts forward + backward gradients match the production per-step path at < 1e-5.
  2. A new test (`test_structure_missing_torch_structured_raises_clear_error`) simulates the missing optional dependency (e.g. via `monkeypatch.setattr` on `structure._import_torch_structured` or by mocking the import) and asserts `make_structured_linear(kind="monarch"|"butterfly"|"ldr")` raises `ImportError` with a message that includes "torch-structured" and the install hint — not a silent `AttributeError`.
  3. Dense / Diagonal / Circulant kinds (local impls) continue to work in the same test run when `torch-structured` is unavailable.
  4. Any mismatch surfaced becomes a failing test → beads issue → fix in-phase; the hand-rolled references stay in the test file (not promoted into `src/`).
**Plans**: TBD

### Phase 4: Quant-on bit-identity
**Goal**: With a frozen INT8 recipe applied, every Triton variant produces bit-identical fwd and bwd against the reference path; resolve (fix or fence) the per-channel `min_max` observer gap.
**Depends on**: Phase 2, Phase 3
**Requirements**: QNT-01, QNT-02, QNT-03, QNT-04
**Success Criteria** (what must be TRUE):
  1. `tests/test_triton_scan.py` (and per-variant Triton test files) gain a `test_quant_on_bitidentical_with_reference` test per variant: with an INT8 per-channel weight + per-tensor activation recipe frozen, `(out, h_T)` from `use_triton=True` is bit-identical (`torch.equal`) to `use_triton=False` on the same input. Tested with realistic inputs AND adversarial inputs (near-saturation, large magnitudes, near-zero).
  2. Quant-on backward gradients are bit-identical between Triton and reference paths under the same frozen recipe across all variants (dense, diagonal, monarch, butterfly).
  3. The per-channel `min_max` observer gap (`quantizers.py:135-146`) is resolved with one of: (a) fixed (vectorized per-channel reduction) with a test in `tests/test_quantizers.py` confirming per-channel running stats, or (b) gated behind an explicit `NotImplementedError`/`ValueError` when `axis is not None and mode == "min_max"`. Decision is logged in PROJECT.md Key Decisions.
  4. Any bit-identity mismatch surfaced becomes a failing test → beads issue → fix in-phase. Quant-on tolerance is not loosened to numerical-bounded (bit-identity is the contract for a deterministic frozen recipe).
**Plans**: TBD

### Phase 5: Calibration + freeze lifecycle
**Goal**: `GRULayer.calibrate(loader, n_batches)` provably exercises observers (not the Triton fast path), `freeze_all(module)` produces scales matching the documented contract, and the post-freeze Triton round-trip matches the reference path on held-out data.
**Depends on**: Phase 4
**Requirements**: CAL-01, CAL-02, CAL-03
**Success Criteria** (what must be TRUE):
  1. `tests/test_calibration.py` gains `test_calibrate_uses_per_step_path`: builds a `GRULayer(use_triton=True)` (CUDA path), runs `layer.calibrate(loader, n_batches)`, and asserts that BEFORE calibration each activation FakeQuantize has `running_min == +inf` / `running_max == -inf` and AFTER calibration both have finite values that match what running the per-step path directly would produce. Confirms the wrapper actually disabled `use_triton`.
  2. `tests/test_calibration.py` gains `test_freeze_all_matches_dynamic_on_last_batch`: after `calibrate` + `freeze_all`, each activation quantizer's frozen `scale` matches what the same module's `dynamic` mode would have produced when fed the calibration loader's final batch — within the documented contract (exact match on running min/max derivation).
  3. `tests/test_calibration.py` gains a CUDA-only `test_triton_matches_reference_after_freeze`: build a layer, calibrate, freeze, then on a held-out batch assert `use_triton=True` output is bit-identical to `use_triton=False` (this is the QNT round-trip with a calibrated recipe rather than a hand-built one).
  4. The "calibrate without disabling use_triton" anti-pattern (`gru_layer.py:289-299` warning) is preserved or strengthened — observer stats stay at ±inf if a user bypasses the wrapper. Tested.
  5. Any mismatch surfaced becomes a failing test → beads issue → fix in-phase.
**Plans**: TBD

### Phase 6: Edge-case sweeps
**Goal**: Every path (reference, dense Triton, diagonal Triton, monarch Triton, butterfly Triton, circulant per-step, LDR per-step) survives T=1, B=1, H∈{1, 2}, T∈{512, 1024}, and T=0/B=0 with either correct output or a clear tested error.
**Depends on**: Phase 1, Phase 2
**Requirements**: EDG-01, EDG-02, EDG-03, EDG-04
**Success Criteria** (what must be TRUE):
  1. A new `tests/test_edge_cases.py` (or extensions inside existing per-variant Triton test files, whichever keeps coupling cleaner — decided in plan-phase) covers `T=1` fwd + bwd for every available path and asserts parity vs reference at the same tolerance tier the path normally uses.
  2. The same file covers `B=1` and `H ∈ {1, 2}` for every available path — explicitly testing the BLOCK-size-assumption failure modes flagged in `CONCERNS.md`.
  3. `T ∈ {512, 1024}` long-sequence tests are marked `@pytest.mark.slow` and assert no accumulated drift exceeds the layer-level tier-A tolerance (< 1e-4 vs `nn.GRU` for reference path; < 1e-5 vs reference for Triton paths).
  4. `T=0` and `B=0` either produce correctly-shaped empty output OR raise a clear `ValueError` with a message that mentions the offending dimension. No NaN output, no kernel hang. Behaviour decided in plan-phase and logged in PROJECT.md.
  5. Any mismatch surfaced becomes a failing test → beads issue → fix in-phase.
**Plans**: TBD

### Phase 7: Audit report + findings handling
**Goal**: Every finding from Phases 1–6 is captured with a failing-test-before-fix discipline and a beads issue; the audit closes with an `AUDIT-REPORT.md` summarizing what was checked, what passed, what was fixed, and any residual known-but-accepted divergences.
**Depends on**: Phase 1, Phase 2, Phase 3, Phase 4, Phase 5, Phase 6
**Requirements**: RPT-01, RPT-02, RPT-03
**Success Criteria** (what must be TRUE):
  1. Every mismatch surfaced during Phases 1–6 has (a) a failing test committed BEFORE the fix landed, verifiable via `git log` showing test-commit precedes fix-commit, and (b) a corresponding beads issue (`bd show <id>`) with root cause, fix reference, and regression-test path.
  2. `AUDIT-REPORT.md` exists at repo root and contains: (a) a table of all 28 v1 requirements with PASS/FIX/ACCEPTED-DIVERGENCE status, (b) per-phase summary of what was checked and how, (c) a "residual known-but-accepted divergences" section with the rationale for each, (d) a pointer to the beads issues that resolved each finding.
  3. `pytest -q` and `pytest -m slow -q` both pass on a CUDA machine; `pytest -q` passes on a CPU-only machine; `mypy` and `ruff check src tests` are green.
  4. `bd ready` shows no unresolved audit findings (all closed or explicitly deferred to v2 with a beads issue reference in `REQUIREMENTS.md`).
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6 → 7

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Reference-path parity vs nn.GRU | 5/5 | Complete ✓ | 2026-05-13 |
| 2. Triton fast-path parity vs reference | 0/TBD | Not started | - |
| 3. Structured PyTorch fallback parity | 0/TBD | Not started | - |
| 4. Quant-on bit-identity | 0/TBD | Not started | - |
| 5. Calibration + freeze lifecycle | 0/TBD | Not started | - |
| 6. Edge-case sweeps | 0/TBD | Not started | - |
| 7. Audit report + findings handling | 0/TBD | Not started | - |

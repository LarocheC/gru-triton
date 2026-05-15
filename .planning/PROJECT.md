# gru-triton — Native-PyTorch Parity Audit

## What This Is

`gru-triton` (aka `gru-qat`) is a single-direction, single-layer GRU built for quantization-aware training (QAT) with a slow reference path in pure PyTorch and a fast path in persistent Triton kernels (dense + diagonal + monarch + butterfly). Structured hidden weights (Monarch, Butterfly, Circulant, LDR) are pluggable via `StructureConfig`. This milestone is a **correctness audit** of the existing implementation against a native PyTorch baseline (`torch.nn.GRU`), with the explicit goal of finding holes, bugs, and mismatches before any further feature work.

## Core Value

**Every code path that claims to compute a GRU must produce numerically equivalent output to `torch.nn.GRU` (under matched recipe), and any deviation must be a tested, documented, intentional one — not a silent drift.**

## Requirements

### Validated

<!-- Inferred from the existing codebase + tests + bench. Locked. -->

- ✓ Reference cell parity: `GRUCellQuant` with Identity quantizers matches `torch.nn.GRUCell` < 1e-5 (`tests/test_parity.py`)
- ✓ STE primitives + FakeQuantize granularities (per-tensor / per-channel / per-group, bits ∈ {2,3,4,8}) (`tests/test_ste.py`, `tests/test_quantizers.py`)
- ✓ End-to-end QAT trains on toy task (`tests/test_qat_smoke.py`)
- ✓ `calibrate()` round-trip + `freeze_all()` (`tests/test_calibration.py`)
- ✓ Structured cells (Diagonal, Monarch, Circulant, Butterfly, LDR): forward, gradient flow, training, int8 QAT (`tests/test_structure.py`)
- ✓ Dense Triton fwd+bwd + QAT (`tests/test_triton_scan.py`)
- ✓ Diagonal / Monarch / Butterfly Triton fwd+bwd + QAT + GRULayer dispatch (`tests/test_triton_diagonal.py`, `tests/test_triton_monarch.py`, `tests/test_butterfly_dispatch.py`)
- ✓ Cross-CTA release/acquire `atomic_add(sem=...)` barrier pattern in persistent kernels (`src/gru_qat/triton_kernels/scan.py`)

### Active

<!-- This milestone's scope: the parity audit. -->

#### A. Reference-path parity vs `torch.nn.GRU`

- [ ] **A1**: `GRULayer` (use_triton=False, Identity quantizers, dense) fwd matches `torch.nn.GRU` (1 layer, unidirectional) at < 1e-4 over T ∈ {1, 8, 64, 512, 1024}, batch ∈ {1, 4, 32}, hidden ∈ {1, 2, 8, 64, 512}.
- [ ] **A2**: `GRULayer` fwd matches `torch.nn.GRU` with `h_0 ≠ 0` (random initial state) at the same tolerance.
- [ ] **A3**: `GRULayer` bwd: gradients (dW_ih, dW_hh, db_ih, db_hh, dx, dh_0) match `torch.nn.GRU` autograd at < 1e-4.
- [ ] **A4**: Final hidden state `h_T` returned by `GRULayer` matches `torch.nn.GRU`'s `h_n` to the same tolerance.
- [ ] **A5**: Gate-ordering / bias-fusion alignment with `torch.nn.GRU` is documented and a translation helper exists (or the divergence is explicitly tested and flagged).

#### B. Triton-fast-path parity vs reference PyTorch path

- [ ] **B1**: Dense Triton fwd matches reference (`use_triton=False`) at < 1e-5 across the same shape grid; backward at < 1e-5.
- [ ] **B2**: Diagonal Triton fwd+bwd matches reference at < 1e-5.
- [ ] **B3**: Monarch Triton fwd+bwd matches reference at < 1e-5 across `nblocks ∈ {2, 4, 8}`.
- [ ] **B4**: Butterfly Triton fwd+bwd matches reference at < 1e-5. Includes the recent OOB-fix regression test at the last program.
- [ ] **B5**: Triton bwd dWh / dbh accumulator slabs zero-initialized correctly under autotuned configs (regression coverage for `c001a8a`).
- [ ] **B6**: Persistent-kernel cross-CTA barriers produce deterministic output across re-runs (no `.cv` cache-modifier sneak-in regression).

#### C. Structured PyTorch fallbacks parity

- [ ] **C1**: Circulant variant per-step PyTorch path matches a hand-rolled circulant-matmul reference for fwd + bwd at < 1e-5.
- [ ] **C2**: LDR variant per-step PyTorch path matches a hand-rolled LDR reference for fwd + bwd at < 1e-5.
- [ ] **C3**: All structured variants degrade gracefully (clear error, not silent wrong-answer) when `torch-structured` is missing.

#### D. Quant-on parity (reference path is ground truth)

- [ ] **D1**: Dense Triton fwd with an active recipe (INT8 per-channel weight + per-tensor activation, frozen) is bit-identical to reference path under the same recipe. Tested with realistic and adversarial inputs (large magnitudes, near-saturation).
- [ ] **D2**: Same for Diagonal / Monarch / Butterfly Triton paths.
- [ ] **D3**: Quant-on backward gradients (after STE) match between Triton and reference paths, bit-identical where the recipe is deterministic.
- [ ] **D4**: Per-channel `min_max` observer for activations is either fixed or its broken-for-per-channel-axis state is gated behind an explicit error rather than silently using the global reduction. (See Phase 1 known gap.)

#### E. Calibration + freeze lifecycle audit

- [ ] **E1**: `GRULayer.calibrate(loader, n_batches)` actually exercises the per-step path (not the Triton path) so observers update. Confirmed by inspecting observer state before/after.
- [ ] **E2**: `freeze_all(module)` produces scales identical to what `dynamic` mode would compute on the calibration data's final batch, within the documented contract.
- [ ] **E3**: After `freeze_all`, switching `use_triton=True` produces the same output as `use_triton=False` on a held-out batch (D1/D2 round-trip with the calibrated recipe).

#### F. Edge-case coverage

- [ ] **F1**: T=1 (single timestep) produces correct output and gradient for every path.
- [ ] **F2**: B=1 single-batch and H ∈ {1, 2} small-hidden cases produce correct output for every path. (Triton kernels often have BLOCK assumptions that break here.)
- [ ] **F3**: T ∈ {512, 1024} long-sequence parity: no accumulated numerical drift exceeds the tier-A tolerance at the layer level.
- [ ] **F4**: Empty inputs (T=0 or B=0) either work or raise a clear, tested error (not a silent NaN / kernel hang).

#### G. Findings handling

- [ ] **G1**: Every mismatch surfaced by A–F is captured by a failing test before any fix lands.
- [ ] **G2**: Every finding has a beads issue (`bd create`) capturing root cause, fix, and regression test.
- [ ] **G3**: Audit ends with a written `AUDIT-REPORT.md` summarizing what was checked, what passed, what was fixed, and any residual known-but-accepted divergences.

### Out of Scope

<!-- Explicit boundaries with reasoning. -->

- **LSTM, vanilla RNN, bidirectional, multi-layer GRU** — already out of scope per `SCOPE.md`. Audit only covers the single-direction, single-layer GRU surface that exists.
- **Mixed-precision (fp16/bf16) parity** — `SCOPE.md` and `DEVELOPMENT.md` document that bf16 around fake-quant was tried and dropped (cast tax). Confirming this audit doesn't re-litigate that decision; fp32 invariants only.
- **Bench / performance re-validation** — the cuDNN comparison table in `DEVELOPMENT.md` is machine-dependent. This milestone is correctness-only; perf is a separate concern.
- **Bias quantization, LUT sigmoid/tanh, ONNX export, streaming inference** — all out of scope per `SCOPE.md`. Audit doesn't fabricate parity tests for unimplemented features.
- **Phase 6 (int activations + LUT nonlinearities)** — not started, not in scope here.
- **Hand-rolled INT8 reference GRU** — chose "reference PyTorch path = ground truth" over building an independent INT8 implementation. If reference path itself is wrong, it'll surface in section A (fp32 vs `torch.nn.GRU`); we don't need a third baseline.

## Context

- This is a brownfield audit: 5 phases plus a structured-matrix track plus a calibration plumbing pass are already shipped. Code surface is mature and tested, but the test coverage for **multi-step parity vs `torch.nn.GRU` at the layer level** has a gap — only the cell is pinned to `nn.GRUCell`.
- Recent commits show backward kernels are an active fragility area: `4e10402` (diagonal variant), `d8218d4` (butterfly scratch/state OOB at last program), `c001a8a` (zero dWh/dbh accumulator slabs in autotuned bwd kernel). The audit explicitly adds regression tests for these.
- The codebase map (`.planning/codebase/CONCERNS.md`) flags the per-channel `min_max` observer as a known broken path. The audit decides whether to fix or fence it.
- `torch-structured` is an optional dep; sections C2/C3 cover both "with it installed" and "without it" behavior.
- Test discipline: Triton tests skip on no-CUDA via `pytest.importorskip("triton")` + `cuda_only` mark — audit tests inherit this pattern. The reference-path A and C tests do **not** require CUDA.

## Constraints

- **Baseline**: `torch.nn.GRU` (1 layer, unidirectional, `batch_first=True`) — handles gate ordering / bias fusion quirks at the test-helper layer, not by changing reference-path code.
- **Tolerance tiers**:
  - fp32 Identity-quantizer cell parity: < 1e-5 (existing contract; do not loosen)
  - fp32 reference-layer vs `nn.GRU`: < 1e-4 (allows for accumulation drift over T)
  - Triton vs reference under same recipe: < 1e-5 (revised per Phase 2 disposition — see Key Decisions)
    - **For kernels without `tl.dot` (diagonal):** < 1e-5 abs (FAST); slow-tier `dbh` < 2e-5 (F-02-02-A non-associativity)
    - **For kernels with `tl.dot` (dense, monarch, butterfly):** < 5e-4 abs (tight-TF32) — Triton's `tl.dot` uses TF32 on Ampere+ regardless of `torch.set_float32_matmul_precision('highest')`; strict fp32 unachievable without `input_precision="ieee"` kernel changes
  - Quant-on (active recipe, deterministic): bit-identical
- **Test framework**: pytest with existing markers (`cuda_only`, `slow`). Long-T parity tests (F3) marked `slow`.
- **Linting / typing**: ruff + mypy strict on `src/gru_qat`. New test helpers in `tests/` are not mypy-strict (matches existing convention).
- **Don't optimize the reference path** — even if A/F surfaces slowness, speed lives in Triton. Reference is correct-by-construction.
- **Don't loosen `< 1e-5`** for Identity quantizer cell parity. If A1 fails at < 1e-4, that's a *new* test for the layer; the cell test stays at < 1e-5.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Baseline is `torch.nn.GRU` (cuDNN), not a hand-rolled reference | Real-world ground truth. cuDNN quirks (fused biases, gate order) are handled in test-helper layer. | — Pending |
| Reference PyTorch path is ground truth for Triton/structured parity | Aligns with SCOPE §6 (reference is correct-by-construction). Avoids three-baseline cross-product. | — Pending |
| Forward + backward parity, not forward-only | Recent fix-commit cluster (butterfly OOB, dWh/dbh accumulator) shows bwd is where bugs hide. | — Pending |
| Tiered tolerance, not uniform | < 1e-5 for cell + Triton-vs-reference; < 1e-4 for layer-vs-nn.GRU (accumulation drift); bit-identical for quant-on. | — Pending |
| Fix in-milestone, not report-only | Each finding becomes a beads issue + failing test + fix in the same audit. Audit ends with everything green. | — Pending |
| **Phase 1 closed:** baseline / reference-path / fwd+bwd / tiered-tolerance / fix-in-milestone all validated | 304 tests pass (fwd, h_T, bwd, h_0≠0 over 75-combo grid); zero parity bugs surfaced; cell parity contract held | ✓ Good — 2026-05-13 |
| **Phase 2 disposition (Option C, hybrid):** strict `< 1e-5` for diagonal; tight-TF32 `< 5e-4` for dense/monarch/butterfly | Empirical: Triton `tl.dot` uses TF32 on Ampere+ regardless of `set_float32_matmul_precision('highest')`. Diagonal (no matmul) passes < 1e-5 cleanly; matmul kernels show ~3e-4..4e-2 abs drift. Choosing tight-TF32 over `input_precision="ieee"` kernel changes (would push beyond test-only scope). bd issue gru-triton-rwm closed-accepted | ✓ Good — 2026-05-13 |
| **F-02-02-A (Phase 2):** diagonal long-T `dbh` accumulator drift bound loosened to `< 2e-5` (slow-tier only) | Root cause: `tl.sum` warp-butterfly vs `torch.sum` parallel-tree reduction-order non-associativity at T=1024. Not a bug. bd issue gru-triton-e7t left open for a future hygiene phase | ⚠️ Revisit — long-T dbh kernel alignment is a candidate for future cleanup |
| **Phase 6 D-01 (T=0/B=0 disposition)** | An empty sequence/batch is almost always a caller bug; Triton kernels cannot launch a 0-size grid. Uniform fail-loud matches kernel reality. | All 7 paths raise `ValueError` naming the offending dimension (T or B) on T=0 / B=0 — no empty-output passthrough. A single guard in `GRULayer.forward` after the shape unpack covers all 7 GRULayer-routed paths. Resolves ROADMAP Phase 6 SC#4 to the `ValueError` branch. ✓ Good — 2026-05-15 |
| Bench re-validation excluded | Machine-dependent; correctness audit only. Numbers in `DEVELOPMENT.md` not re-touched. | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-05-13 after initialization*

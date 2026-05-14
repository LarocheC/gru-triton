# Requirements: gru-triton — Native-PyTorch Parity Audit

**Defined:** 2026-05-13
**Core Value:** Every code path that claims to compute a GRU must produce numerically equivalent output to `torch.nn.GRU` (under matched recipe), and any deviation must be a tested, documented, intentional one — not a silent drift.

## v1 Requirements

Requirements for this audit milestone. Each maps to roadmap phases.

### Reference-path parity vs torch.nn.GRU (REF)

- [x] **REF-01**: `GRULayer` (use_triton=False, Identity quantizers, dense) forward matches `torch.nn.GRU` (1 layer, unidirectional) to < 1e-4 over T ∈ {1, 8, 64, 512, 1024}, B ∈ {1, 4, 32}, H ∈ {1, 2, 8, 64, 512}.
- [x] **REF-02**: Same parity test family with `h_0 ≠ 0` (random initial state) at the same tolerance.
- [x] **REF-03**: `GRULayer` backward gradients (dW_ih, dW_hh, db_ih, db_hh, dx, dh_0) match `torch.nn.GRU` autograd to < 1e-4.
- [x] **REF-04**: Final hidden state `h_T` returned by `GRULayer` matches `torch.nn.GRU`'s `h_n` at < 1e-4.
- [x] **REF-05**: Gate-ordering / bias-fusion alignment with `torch.nn.GRU` is documented in code + a translation helper exists in tests (or the divergence is explicitly tested and flagged).

### Triton-fast-path parity vs reference PyTorch path (TRI)

- [x] **TRI-01**: Dense Triton fwd matches reference (`use_triton=False`) at < 1e-5 across the shape grid; backward at < 1e-5. *(Phase 2 disposition: revised to < 5e-4 abs — TF32-via-`tl.dot` documented in PROJECT.md Key Decisions, bd:gru-triton-rwm)*
- [x] **TRI-02**: Diagonal Triton fwd+bwd matches reference at < 1e-5. *(FAST tier; slow-tier `dbh` < 2e-5 per F-02-02-A non-associativity, bd:gru-triton-e7t)*
- [x] **TRI-03**: Monarch Triton fwd+bwd matches reference at < 1e-5 across `nblocks ∈ {2, 4, 8}`. *(Phase 2 disposition: revised to < 5e-4 abs — same TF32 root cause)*
- [x] **TRI-04**: Butterfly Triton fwd+bwd matches reference at < 1e-5. Includes regression test for the last-program OOB fix (`d8218d4`). *(Phase 2 disposition: revised to < 5e-4 abs — same TF32 root cause; OOB regression at `tests/test_butterfly_dispatch.py:164` still passes)*
- [x] **TRI-05**: Triton backward dWh / dbh accumulator slabs zero-initialized correctly under all autotuned configs (regression coverage for `c001a8a`). *(Regression test `test_autotune_dWh_dbh_zero_init_across_configs` in scan_strict; slab-zero contract preserved via iter=1 divergence signal)*
- [x] **TRI-06**: Persistent-kernel cross-CTA barriers produce deterministic output across re-runs (regression coverage for the `.cv` cache-modifier-as-fence-substitute mistake). *(50-run `torch.equal` test passes; D-25 `.cv` canary at 0 live uses)*

### Structured PyTorch fallbacks parity (STR)

- [x] **STR-01**: Circulant variant per-step PyTorch path matches a hand-rolled circulant-matmul reference for forward + backward at < 1e-5. *(`tests/test_structure_parity.py` — FFT/Toeplitz cross-check + production parity; worst max abs 2.62e-6 at H=512)*
- [x] **STR-02**: LDR variant per-step PyTorch path matches a hand-rolled LDR reference for forward + backward at < 1e-5. *(slow-Krylov full-matrix reference; worst max abs 1.67e-6 at H=512)*
- [x] **STR-03**: All structured variants degrade gracefully (clear error, not silent wrong-answer) when `torch-structured` is missing. *(monkeypatch on `_import_torch_structured` for monarch/butterfly; `sys.modules` trick for LDR bypass-import path; companion test confirms dense/diagonal/circulant work without dep)*

### Quant-on parity, reference path = ground truth (QNT)

- [ ] **QNT-01**: Dense Triton forward with an active recipe (INT8 per-channel weight + per-tensor activation, frozen) is bit-identical to reference path under the same recipe. Tested with realistic and adversarial inputs (large magnitudes, near-saturation).
- [ ] **QNT-02**: Same bit-identity for Diagonal / Monarch / Butterfly Triton paths.
- [ ] **QNT-03**: Quant-on backward gradients (after STE) match between Triton and reference paths, bit-identical where the recipe is deterministic.
- [ ] **QNT-04**: Per-channel `min_max` observer for activations is either fixed or gated behind an explicit error rather than silently using the global reduction (Phase 1 known gap resolution).

### Calibration + freeze lifecycle audit (CAL)

- [ ] **CAL-01**: `GRULayer.calibrate(loader, n_batches)` actually exercises the per-step path (not Triton) so observers update. Verified by inspecting observer state before/after.
- [ ] **CAL-02**: `freeze_all(module)` produces scales identical to what `dynamic` mode would compute on the calibration data's final batch, within the documented contract.
- [ ] **CAL-03**: After `freeze_all`, switching `use_triton=True` produces the same output as `use_triton=False` on a held-out batch (Triton/reference round-trip with the calibrated recipe).

### Edge-case coverage (EDG)

- [ ] **EDG-01**: T=1 single-timestep produces correct output and gradient for every path (reference + every Triton kernel).
- [ ] **EDG-02**: B=1 single-batch and H ∈ {1, 2} small-hidden cases produce correct output for every path (Triton kernels often have BLOCK assumptions that break at tiny shapes).
- [ ] **EDG-03**: T ∈ {512, 1024} long-sequence parity: no accumulated numerical drift exceeds the tier-A tolerance at the layer level (marked `slow`).
- [ ] **EDG-04**: Empty inputs (T=0 or B=0) either work or raise a clear, tested error (not a silent NaN / kernel hang).

### Findings handling & reporting (RPT)

- [ ] **RPT-01**: Every mismatch surfaced during REF/TRI/STR/QNT/CAL/EDG is captured by a failing test before any fix lands.
- [ ] **RPT-02**: Every finding has a beads issue (`bd create`) capturing root cause, fix, and regression test.
- [ ] **RPT-03**: Audit ends with a written `AUDIT-REPORT.md` summarizing what was checked, what passed, what was fixed, and any residual known-but-accepted divergences.

## v2 Requirements

Acknowledged but deferred beyond this audit milestone.

### Activation Quantization Improvements

- **ACT-01**: Implement per-channel `min_max` observer correctly (instead of fencing it). Surfaced by QNT-04 if user later decides "fix" was the wrong choice.
- **ACT-02**: LSQ / PACT learnable activation scales — `learnable_scale` flag in `QuantizerConfig` is plumbed but unimplemented.

### Performance audits (excluded from this milestone)

- **PERF-01**: Re-validate cuDNN comparison table in `DEVELOPMENT.md` on current hardware.
- **PERF-02**: Bench QAT-on overhead vs the +10–30% claim in `DEVELOPMENT.md`.

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| LSTM, vanilla RNN, bidirectional, multi-layer GRU | Out of scope per `SCOPE.md`. Audit only covers the single-direction, single-layer GRU surface that exists. |
| Mixed-precision (fp16/bf16) parity testing | `SCOPE.md` and `DEVELOPMENT.md` document bf16 around fake-quant was tried and dropped. Audit does not re-litigate. |
| Bench / performance re-validation | Machine-dependent; correctness-only milestone. Deferred to v2 PERF-01/02. |
| Bias quantization, LUT sigmoid/tanh, ONNX export, streaming inference | All out of scope per `SCOPE.md`. Audit does not fabricate parity tests for unimplemented features. |
| Phase 6 (int activations + LUT nonlinearities) | Not started. Out of scope here. |
| Hand-rolled INT8 reference GRU | Chose "reference PyTorch path = ground truth" baseline. If reference path itself is wrong, surfaces in REF; no third baseline needed. |
| Cell-level parity tolerance change | Existing `< 1e-5` for `GRUCellQuant` Identity-quantizer parity is locked. Audit may not loosen. |

## Traceability

Populated during roadmap creation by the roadmapper agent.

| Requirement | Phase | Status |
|-------------|-------|--------|
| REF-01 | Phase 1 | Complete |
| REF-02 | Phase 1 | Complete |
| REF-03 | Phase 1 | Complete |
| REF-04 | Phase 1 | Complete |
| REF-05 | Phase 1 | Complete |
| TRI-01 | Phase 2 | Complete |
| TRI-02 | Phase 2 | Complete |
| TRI-03 | Phase 2 | Complete |
| TRI-04 | Phase 2 | Complete |
| TRI-05 | Phase 2 | Complete |
| TRI-06 | Phase 2 | Complete |
| STR-01 | Phase 3 | Complete |
| STR-02 | Phase 3 | Complete |
| STR-03 | Phase 3 | Complete |
| QNT-01 | Phase 4 | Pending |
| QNT-02 | Phase 4 | Pending |
| QNT-03 | Phase 4 | Pending |
| QNT-04 | Phase 4 | Pending |
| CAL-01 | Phase 5 | Pending |
| CAL-02 | Phase 5 | Pending |
| CAL-03 | Phase 5 | Pending |
| EDG-01 | Phase 6 | Pending |
| EDG-02 | Phase 6 | Pending |
| EDG-03 | Phase 6 | Pending |
| EDG-04 | Phase 6 | Pending |
| RPT-01 | Phase 7 | Pending |
| RPT-02 | Phase 7 | Pending |
| RPT-03 | Phase 7 | Pending |

**Coverage:**
- v1 requirements: 28 total
- Mapped to phases: 28
- Unmapped: 0 ✓

---
*Requirements defined: 2026-05-13*
*Last updated: 2026-05-13 after roadmap creation (traceability table populated)*

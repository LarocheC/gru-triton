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

- [x] **QNT-01**: Dense Triton forward with an active recipe (INT8 per-channel weight + per-tensor activation, frozen) is bit-identical to reference path under the same recipe. Tested with realistic and adversarial inputs (large magnitudes, near-saturation). ✓ 2026-05-14 (fwd torch.equal; bwd per-(cls, B) h_scale_mult per F-04-VERIFIER-C / gru-triton-mjy)
- [x] **QNT-02**: Same bit-identity for Diagonal / Monarch / Butterfly Triton paths. ✓ 2026-05-14 (diagonal torch.equal except large-magnitude mult=2; monarch mult=4 per F-04-VERIFIER-A / gru-triton-in0; butterfly per-class mult per F-04-VERIFIER-D / gru-triton-lqk)
- [x] **QNT-03**: Quant-on backward gradients (after STE) match between Triton and reference paths, bit-identical where the recipe is deterministic. ✓ 2026-05-14 (per-cluster h_scale_mult dispositions; see `04-DISPOSITION.md` for the full table)
- [x] **QNT-04**: Per-channel `min_max` observer for activations is either fixed or gated behind an explicit error rather than silently using the global reduction (Phase 1 known gap resolution). ✓ 2026-05-14 (FIXED — bd gru-triton-x15 closed)

### Calibration + freeze lifecycle audit (CAL)

- [x] **CAL-01**: `GRULayer.calibrate(loader, n_batches)` actually exercises the per-step path (not Triton) so observers update. Verified by inspecting observer state before/after. ✓ 2026-05-14 (`test_calibrate_uses_per_step_path` at `tests/test_calibration.py:231`)
- [x] **CAL-02**: `freeze_all(module)` produces scales identical to what `dynamic` mode would compute on the calibration data's final batch, within the documented contract. ✓ 2026-05-14 (`test_freeze_all_matches_dynamic_on_last_batch` at `tests/test_calibration.py:350`; scoped to `quant_x` per bd:gru-triton-n20 cross-phase deferral)
- [x] **CAL-03**: After `freeze_all`, switching `use_triton=True` produces the same output as `use_triton=False` on a held-out batch (Triton/reference round-trip with the calibrated recipe). ✓ 2026-05-14 (`test_triton_matches_reference_after_freeze` — 4 kernels × 3 D-46 classes = 12 cases; per-cluster `h_scale_mult` bounds inherited from `04-DISPOSITION.md`)

### Edge-case coverage (EDG)

- [x] **EDG-01**: T=1 single-timestep produces correct output and gradient for every path (reference + every Triton kernel). ✓ 2026-05-15 (`tests/test_edge_cases.py` T=1 fwd+bwd sweep, 7 paths)
- [x] **EDG-02**: B=1 single-batch and H ∈ {1, 2} small-hidden cases produce correct output for every path (Triton kernels often have BLOCK assumptions that break at tiny shapes). ✓ 2026-05-15 (surfaced + fixed 2 bugs: butterfly H=1 crash bd:gru-triton-ehf, butterfly batch-invariance race bd:gru-triton-c2a)
- [x] **EDG-03**: T ∈ {512, 1024} long-sequence parity: no accumulated numerical drift exceeds the tier-A tolerance at the layer level (marked `slow`). ✓ 2026-05-15 (`@pytest.mark.slow` long-T drift sweep)
- [x] **EDG-04**: Empty inputs (T=0 or B=0) either work or raise a clear, tested error (not a silent NaN / kernel hang). ✓ 2026-05-15 (`GRULayer.forward` raises `ValueError` naming the offending dim for all 7 paths; policy logged in PROJECT.md)

### Findings handling & reporting (RPT)

- [x] **RPT-01**: Every mismatch surfaced during REF/TRI/STR/QNT/CAL/EDG is captured by a failing test before any fix lands. ✓ 2026-05-15 (git-log test-before-fix audit done — `07-git-log-audit.txt`; every code-fix finding has RED-before-fix ordering; Phases 1-3 gap-checked, no gap; gaps documented, no history rewritten)
- [x] **RPT-02**: Every finding has a beads issue (`bd create`) capturing root cause, fix, and regression test. ✓ 2026-05-15 (all 14 carry-forward bd issues closed with resolution notes — 3 FIX / 9 ACCEPTED-DIVERGENCE / 2 INDIVIDUAL; `bd ready` empty)
- [x] **RPT-03**: Audit ends with a written `AUDIT-REPORT.md` summarizing what was checked, what passed, what was fixed, and any residual known-but-accepted divergences. ✓ 2026-05-15 (`AUDIT-REPORT.md` authored at repo root — 4 D-08 sections, 28-requirement table, consolidated TF32 divergence entry, criterion-#3 reinterpretation, 14 finding-to-bd pointers)

## v2 Requirements

Acknowledged but deferred beyond this audit milestone.

### Activation Quantization Improvements

- **ACT-01**: Implement per-channel `min_max` observer correctly (instead of fencing it). Surfaced by QNT-04 if user later decides "fix" was the wrong choice.
- **ACT-02**: LSQ / PACT learnable activation scales — `learnable_scale` flag in `QuantizerConfig` is plumbed but unimplemented.

### Performance audits (excluded from this milestone)

- **PERF-01**: Re-validate cuDNN comparison table in `DEVELOPMENT.md` on current hardware.
- **PERF-02**: Bench QAT-on overhead vs the +10–30% claim in `DEVELOPMENT.md`.

### Kernel hardening (deferred from Phase 7 audit closure)

- **KRN-01** (`bd:gru-triton-e0l`): Monarch backward kernel-tiling redesign for
  consumer-GPU SMEM limits. The Phase 7 audit closed `gru-triton-e0l` as a
  documented hardware limit (RTX 2000 Ada: SMEM OOM for `blksz_pad >= 128`,
  `tl.dot` K<16 constraint for `blksz_pad < 16`), covered in-tree by the
  `_skip_if_monarch_bwd_hw_limit` skip. A real fix needs a separate small-tile
  autotune config tier or a re-tiled bwd kernel — a kernel redesign, not a
  bugfix — and is deferred to v2. bd ref: `gru-triton-e0l` (CLOSED in Phase 7
  with this v2 pointer).
- **KRN-02** (`bd:gru-triton-e7t`, `gru-triton-rwm` family): `input_precision="ieee"`
  TF32-elimination rewrite of the `tl.dot` / `tl.sum` kernel reduction paths
  to remove the ACCEPTED-DIVERGENCE family at its root. Explicitly out of
  Phase 7 scope per PROJECT.md's locked "Option C / tiered tolerance" Key
  Decision. bd refs: the 9 ACCEPTED-DIVERGENCE issues (`in0`, `q3k`, `lqk`,
  `5rk`, `mjy`, `lht`, `e7t`, `fpl`, `6dz` — all CLOSED in Phase 7 with a
  resolution note pointing at this v2 record and AUDIT-REPORT.md).

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
| QNT-01 | Phase 4 | Complete (with caveats) |
| QNT-02 | Phase 4 | Complete (with caveats) |
| QNT-03 | Phase 4 | Complete (with caveats) |
| QNT-04 | Phase 4 | Complete |
| CAL-01 | Phase 5 | Complete |
| CAL-02 | Phase 5 | Complete |
| CAL-03 | Phase 5 | Complete |
| EDG-01 | Phase 6 | Complete |
| EDG-02 | Phase 6 | Complete |
| EDG-03 | Phase 6 | Complete |
| EDG-04 | Phase 6 | Complete |
| RPT-01 | Phase 7 | Complete |
| RPT-02 | Phase 7 | Complete |
| RPT-03 | Phase 7 | Complete |

**Coverage:**
- v1 requirements: 28 total
- Mapped to phases: 28
- Unmapped: 0 ✓

---
*Requirements defined: 2026-05-13*
*Last updated: 2026-05-13 after roadmap creation (traceability table populated)*

# Phase 5: Calibration + Freeze Lifecycle — CONTEXT

**Created:** 2026-05-14
**Phase number:** 05
**Phase slug:** calibration-freeze-lifecycle

<domain>
**Capability delivered:** End-to-end validation of the calibrate → freeze → deploy lifecycle.

After Phase 4 established that under a hand-built frozen-INT8 recipe Triton matches the reference path at per-cluster bounds, Phase 5 closes the loop on the realistic *user-facing* workflow:

```python
layer = GRULayer(..., recipe=recipe_with_min_max_mode)
train(layer, train_loader)         # QAT
layer.calibrate(val_loader, n=64)  # gather act stats via per-step path
layer.freeze()                     # lock scales (mode='frozen')
deploy(layer)                      # use_triton=True at inference
```

Phase 5 verifies three properties (CAL-01..03):
1. `calibrate()` actually exercises the per-step path so observers fire.
2. `freeze_all()` produces scales that match the dynamic-mode scales the same data would have produced — i.e., the freeze is correct.
3. After `freeze_all()`, the Triton fast path (`use_triton=True`) matches the per-step reference (`use_triton=False`) on a held-out batch.

Plus an anti-pattern test that the use_triton bypass keeps observers at ±inf (i.e., the wrapper is the only correct entry point).
</domain>

<spec_lock>
No SPEC.md exists for Phase 5. Requirements are locked in REQUIREMENTS.md (CAL-01, CAL-02, CAL-03) and Success Criteria in ROADMAP.md § Phase 5.
</spec_lock>

<canonical_refs>
Mandatory reading for downstream agents (researcher, planner, executor):

| Path | Role |
|---|---|
| `.planning/ROADMAP.md` § Phase 5 | Phase goal + 5 must-have success criteria |
| `.planning/REQUIREMENTS.md` CAL-01/02/03 | Locked requirements with verification stubs |
| `.planning/PROJECT.md` | Core value (`torch.nn.GRU` parity), tolerance tiers, constraints |
| `.planning/phases/04-quant-on-bit-identity/04-SUMMARY.md` | Phase 4 closure with per-cluster disposition findings — Phase 5 inherits this contract for CAL-03 |
| `.planning/phases/04-quant-on-bit-identity/04-DISPOSITION.md` | Per-cluster h_scale_mult table (THE post-freeze tolerance contract for CAL-03) |
| `src/gru_qat/calibration.py` | `calibrate()` + `freeze_all()` API (lines 30-135). Already implemented; Phase 5 only adds tests. |
| `src/gru_qat/gru_layer.py:270-302` | `GRULayer.calibrate()` wrapper (disables use_triton transiently); `GRULayer.freeze()` |
| `tests/test_calibration.py` | Phase 5 extension target (currently 121 lines, 5 tests; adds CAL-01 + CAL-02 + parametrized CAL-03 + anti-pattern test) |
| `tests/test_triton_scan_strict.py` | Phase 4 dense quant section — Phase 5 imports `_assert_quant_parity`, `_dense_bwd_mult`, `_make_dense_layer_quant_int8`, `_adversarial_inputs` |
| `tests/test_triton_diagonal_strict.py` | Phase 4 diagonal — Phase 5 imports `_make_diagonal_layer_quant_int8` |
| `tests/test_triton_monarch_strict.py` | Phase 4 monarch — Phase 5 imports `_make_monarch_layer_quant_int8`, `_monarch_bwd_mult`, `_skip_if_monarch_bwd_hw_limit` |
| `tests/test_triton_butterfly_strict.py` | Phase 4 butterfly — Phase 5 imports `_make_butterfly_layer_quant_int8` and the per-class butterfly bounds |
| `.planning/debug/repro_monarch_rounding.py` | Root-cause reproducer for the TF32 reduction-order family (informs CAL-03's tolerance reuse) |
</canonical_refs>

<prior_decisions>
**From Phase 4 (post-verifier 2026-05-14):**

- **D-42 revised disposition (per-cluster):** Bit-identity (`torch.equal`) achieved only on dense fwd, diagonal fwd (realistic + near-saturation), and diagonal bwd. Other (kernel, direction, class) tuples have empirically-derived `h_scale_mult` bounds. Full table in `04-DISPOSITION.md`. **Phase 5 inherits this exactly for CAL-03.**
- **D-43 helper byte-uniformity:** `_assert_quant_parity` body byte-identical across the 4 strict files; per-call `h_scale_mult` arguments diverge per cluster. **Phase 5 imports it; does NOT duplicate or alter the body.**
- **Per-cluster mult helpers:** `_dense_bwd_mult(cls, B)`, `_monarch_bwd_mult(cls, B)`, plus inline butterfly per-class branches. **Phase 5 imports these and uses them in CAL-03.**
- **HW-limit skip:** `_skip_if_monarch_bwd_hw_limit(T, B, H, nblocks)` skips shapes where the monarch bwd kernel can't compile on RTX 2000 Ada (blksz_pad < 16 or ≥ 128). **Phase 5 must apply this for any monarch bwd round-trip case.**
- **D-46 adversarial classes:** `_adversarial_inputs(cls, T, B, H)` returns inputs for `realistic` / `near-saturation` / `large-magnitude`. **Phase 5 sweeps all 3 in BOTH calibration corpus AND held-out batch** (user decision D below).
- **Root cause (all Phase 4 findings):** TF32 reduction-order non-associativity in tiled `tl.dot` vs reference einsum/matmul. `gru-triton-rwm` is the Phase 2 carry; Phase 4 surfaced 6 verifier-driven manifestations. **CAL-03 bounds DO NOT need to be tighter than Phase 4's — the same root cause applies post-freeze.**
- **Bd carry-forward:** 9 open Phase 4 bd issues are kernel-investigation tickets; none block Phase 5.

**From Phase 4 process retrospective:**

- **F-04-05-D parallel-execution race** (`gru-triton-u00`): the Plan 04-04 GPU commit accidentally included Plan 04-02's diff via `.beads/hooks/pre-commit` cross-session staging. **Phase 5 mitigates by collapsing to a SINGLE plan covering all 4 kernels (no Wave 2 parallelism).**
</prior_decisions>

<codebase_context>
**Reusable assets (do not re-implement):**

- `src/gru_qat/calibration.py:calibrate()` — already iterates `loader`, switches activation quantizers to `min_max`, resets running stats, returns a summary dict. Handles Tensor / tuple / list / dict batches.
- `src/gru_qat/calibration.py:freeze_all()` — iterates `module.modules()` and calls `.freeze()` on every `FakeQuantize`. Already implemented.
- `src/gru_qat/gru_layer.py:GRULayer.calibrate()` — wrapper that *temporarily disables `use_triton`* during calibration (lines 290-299). This is the wrapper CAL-01 verifies.
- `src/gru_qat/gru_layer.py:GRULayer.freeze()` — calls `cell.freeze_quantizers()`.
- `src/gru_qat/quantizers.py:FakeQuantize` family — `.freeze()` switches `mode='frozen'`; `_update_observer` populates `running_min` / `running_max`.

**Test infrastructure (Phase 4 + earlier):**

- `tests/test_calibration.py` (5 existing tests):
  - `test_calibrate_populates_running_stats` (Phase 4)
  - `test_calibrate_then_freeze_locks_scales` (Phase 4)
  - `test_calibrate_handles_tuple_loader` (Phase 4)
  - `test_calibrate_only_activations_skips_weight_quantizers` (Phase 4)
  - `test_calibrate_truncates_to_n_batches` (Phase 4)
- Phase 4 strict files export the per-kernel helpers Phase 5 will import.

**Architectural constraints:**

- Fast-path eligibility (`gru_layer.py:100`): `structure_input=None` AND `kind ∈ {dense, diagonal, monarch, butterfly}` AND `gate_layout="fused"`. CAL-03 builds layers in this configuration only.
- In-kernel fake-quant requires frozen + per-tensor + symmetric hidden quantizers (`gru_layer.py:28`). After Phase 5's calibrate → freeze, this is satisfied; the round-trip is meaningful.
- F-04-VERIFIER-F kernel-launch failures on RTX 2000 Ada (gru-triton-e0l): monarch bwd at blksz_pad ∉ [16, 128). **CAL-03 round-trip is forward-only per the success criterion, so this may not apply** — but Phase 5 planner should confirm.
</codebase_context>

<decisions>

### A. Kernel coverage for CAL-03

**Decision:** CAL-03 round-trip covers all 4 kernels: dense, diagonal, monarch, butterfly.

**Rationale:** Mirrors Phase 4's coverage matrix and stress-tests the full Phase 4 disposition contract post-calibration. The success criterion's literal "build a layer" wording doesn't restrict structure_hidden; the spirit (deploy-time round-trip) demands all four kernels.

**Implementation hint:** Use `@pytest.mark.parametrize("kernel", ["dense", "diagonal", "monarch", "butterfly"])` and dispatch to `_make_{kernel}_layer_quant_int8` via a string→factory map.

### B. Tolerance contract for CAL-03

**Decision:** Reuse Phase 4's per-cluster h_scale_mult dispositions exactly. CAL-03 imports `_assert_quant_parity` + the per-cluster mult helpers (`_dense_bwd_mult`, `_monarch_bwd_mult`, butterfly's inline per-class branches) from the strict files.

**Rationale:** TF32 reduction-order non-associativity is the root cause across all Phase 4 findings; post-freeze the same drift applies. A tighter bound would fail for the same reason Phase 4's bounds were widened. Phase 5's job is to verify the *lifecycle* produces the same scales that Phase 4's hand-built recipe used — not to re-discover or tighten Phase 4's tolerance contract.

**Caveat:** Butterfly bwd at `mult=20000` is a smoke test, not a numerical guarantee. CAL-03 inherits this — Phase 5 explicitly documents this in its summary.

**Implementation hint:** `from tests.test_triton_scan_strict import _assert_quant_parity, _dense_bwd_mult, _make_dense_layer_quant_int8` etc. If `pytest` can't import sibling test modules, add `tests/conftest.py` to expose them. Alternative: extract the helpers into a `tests/_phase4_quant_helpers.py` module — decide in research/planning.

### C. Test file structure

**Decision:** Extend the existing `tests/test_calibration.py` with all Phase 5 tests. CAL-03 is parametrized over kernel ∈ {dense, diagonal, monarch, butterfly}.

**Rationale:** Single file keeps Phase 5 navigable. Phase 5's tests are conceptually about the calibrate/freeze *lifecycle*, not about each kernel — putting them in test_triton_*_strict.py would fragment the lifecycle story across 4 files.

**Cross-file imports:** Helpers from the 4 strict files are imported into test_calibration.py. This is the chosen tradeoff for keeping Phase 5 self-contained while reusing Phase 4 infrastructure.

### D. Calibration corpus + held-out batch

**Decision:** Sweep all 3 D-46 adversarial classes (`realistic`, `near-saturation`, `large-magnitude`) in BOTH the calibration loader AND the held-out CAL-03 batch.

**Rationale:** The most aggressive coverage option. Tests whether scales chosen on worst-case calibration data still produce parity on worst-case held-out data. Maps cleanly onto Phase 4's per-cluster tolerance table — every (kernel, class, B) tuple in the table is exercised end-to-end by CAL-03.

**Implementation hint:**
- Calibration loader yields a sequence of batches, each from a randomly-rotated class (or one batch per class in a fixed cycle).
- Held-out batch is a separate `torch.manual_seed`'d draw from each class; CAL-03 asserts per-cluster bound for each (kernel × class × B) combination.
- Hidden-side scale (`h_in`/`h_out`) is set manually to a target `h_scale=0.02` for the freeze step (mirroring Phase 4's helper layer factories) — calibration only populates `quant_x` + weight quantizers; hidden stays at the deterministic h_scale.

### E. Plan structure (parallel-race mitigation)

**Decision:** Phase 5 is a SINGLE plan covering all 4 kernels. No Wave 2 parallelism on `tests/test_calibration.py`.

**Rationale:** F-04-05-D (`gru-triton-u00`) parallel-execution race recurred in Phase 4 despite explicit warning. Phase 5 sidesteps the race architecturally by collapsing into one sequential plan. Largest single plan in the milestone but cleanest race-avoidance.

**Plan content sketch (for the planner):**
- Test 1: `test_calibrate_uses_per_step_path` (CAL-01 + anti-pattern coverage per success criterion #1 and #4). CUDA-only.
- Test 2: `test_freeze_all_matches_dynamic_on_last_batch` (CAL-02). CPU OK.
- Test 3: `test_triton_matches_reference_after_freeze` (CAL-03). CUDA-only, parametrized over kernel × class × shape grid.
- Test 4: `test_use_triton_bypass_keeps_observers_at_inf` (anti-pattern per success criterion #4). CUDA-only.
- One commit per test (`test(05-01): CAL-01 ...` etc.) within the single plan, OR a single combined commit — planner's call based on dependency analysis.

### F. Failing-test-before-fix discipline (D-37/D-50 carry)

**Decision:** Each new test lands as a failing/passing test in its own commit (Commit A). If any test fails on the first GPU run, follow Phase 4's discipline: failing-test → bd issue → fix → Commit B. No `@pytest.mark.xfail` per D-50.

**Rationale:** Inherited from Phase 4. Verifier-friendly.

### G. CAL-03 grid coarseness

**Decision (delegated to planner):** The grid for CAL-03 is TBD by the planner. Suggested starting point: a *subset* of Phase 4's `QUANT_FAST_GRID` (e.g., 1-2 shapes per kernel × 3 classes), since CAL-03 is testing the *lifecycle* not exhaustive numerical coverage. The full grid is Phase 4's job; Phase 5 just confirms the lifecycle produces equivalent scales.

**Reasoning for delegation:** The success criterion #3 says "on a held-out batch" — singular. A small grid suffices unless the planner identifies a failure mode that needs more shapes.

</decisions>

<deferred>

| Idea | Why deferred | Suggested phase |
|---|---|---|
| LSQ / PACT learnable activation scales (ACT-02) | Out of scope; QuantizerConfig has a `learnable_scale` flag stubbed but unimplemented | v2 / future milestone |
| Calibration with mixed-precision (bf16 around quant) | Phase 0 design decision: bf16 around fake-quant was tried and dropped (cast tax > GEMM saving). Don't reintroduce. | Permanent non-goal |
| Multi-layer / bidirectional calibration | Single-layer, single-direction is the milestone scope per SCOPE.md | v2 milestone |
| Real-data calibration corpus (vs synthetic) | Synthetic is faster and reproducible; real-data calibration is a user-facing recipe, not an audit deliverable | v2 / docs |
| Calibration on streaming inference paths | Streaming bypasses Triton (per `cell.step` per-step API); the lifecycle audit targets the batched API | Phase 6 (edge cases) if needed |
| Verifier-deferred kernel-level fixes for the 9 open bd issues | Phase 7 audit report scope | Phase 7 |

</deferred>

<process_constraints>

**Inherited from Phase 4:**

1. **D-37/D-50 two-commit failing-test-before-fix** for any new finding. Commit A (failing test) precedes Commit B (fix) in git log.
2. **D-50 no `@pytest.mark.xfail`** anywhere in the Phase 5 surface.
3. **D-51 locked files unchanged:** `tests/test_parity.py`, `tests/test_layer_parity.py`, `tests/test_structure.py` are still locked. Phase 5 must not edit them. The 4 strict files are now also effectively locked for new test additions — Phase 5 only *imports* from them.
4. **Phase 5 SUMMARY must include a timestamped pytest output** (Phase 4 retrospective recommendation #1). Narrative "all tests pass" claims are insufficient.
5. **No parallel-execution race surface:** single plan; sequential commits within the plan.
6. **bd issue discipline:** every test surfaces a finding → file bd → fix in-phase. Carry-forward only to Phase 7 for kernel-level remediation.

**Phase 5-specific:**

7. **The 9 open Phase 4 bd issues do not block Phase 5.** They are Phase 7 work.
8. **Phase 5 closure requires:** CAL-01/02/03 verified, ROADMAP/REQUIREMENTS Phase 5 checkbox flipped, STATE.md updated, summary committed.

</process_constraints>

---

## Next Up

**Phase 5: Calibration + Freeze Lifecycle** — `calibrate` actually exercises observers; `freeze_all` produces correct scales; Triton round-trip after freeze matches reference.

`/clear` then:

`/gsd-plan-phase 5`

CONTEXT.md is consumed by `gsd-phase-researcher` (to know what to research before planning) and `gsd-planner` (to know what decisions are locked).

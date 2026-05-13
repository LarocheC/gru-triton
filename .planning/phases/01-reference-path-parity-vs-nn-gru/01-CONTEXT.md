# Phase 1: Reference-path parity vs nn.GRU - Context

**Gathered:** 2026-05-13
**Status:** Ready for planning

<domain>
## Phase Boundary

Pin `GRULayer` (use_triton=False, Identity quantizers, dense — i.e. the pure-PyTorch reference path) against `torch.nn.GRU(num_layers=1, bidirectional=False, batch_first=True)` at the layer level so every later phase (Triton, structured, quant-on, calibration, edge cases) has a trusted ground truth.

In scope:
- Forward output `(out, h_T)` vs `nn.GRU` `(output, h_n)` to < 1e-4
- Backward gradients `(dW_ih, dW_hh, db_ih, db_hh, dx, dh_0)` vs `nn.GRU` autograd to < 1e-4
- `h_0 ≠ 0` random initial state at the same tolerance
- The full T × B × H = 5 × 3 × 5 = 75-combo grid (T ∈ {1, 8, 64, 512, 1024}, B ∈ {1, 4, 32}, H ∈ {1, 2, 8, 64, 512})
- Bidirectional translation helper (both cell→nn.GRU and nn.GRU→cell) with focused micro-tests for gate ordering and the n-gate asymmetry

Explicitly NOT in scope for Phase 1:
- Triton fast paths (Phase 2)
- Structured hidden weights — Monarch / Butterfly / Circulant / LDR (Phase 3, used as part of Phase 2's reference)
- Quant-on parity — fake-quant non-Identity (Phase 4)
- Calibration + freeze lifecycle (Phase 5)
- T=0 / B=0 empty-input handling (Phase 6)
- Mixed-precision (fp16/bf16) — already deferred at project level

</domain>

<decisions>
## Implementation Decisions

### Translation helper direction
- **D-01:** Test both directions — round-trip. The 75-combo parametrized grid uses **cell → nn.GRU** (build `GRULayer` first with random weights as source-of-truth, then construct an `nn.GRU` and assign its `weight_ih_l0 = torch.cat([W_ir, W_iz, W_in], dim=0)`, etc.). Plus a **single round-trip smoke test** in the opposite direction (build `nn.GRU` first, copy its weights into a fresh cell, assert layer outputs match) — catches bugs in the helper itself.
- **D-02:** Helper lives in `tests/test_layer_parity.py` as module-level `_translate_cell_to_nn_gru(layer) -> nn.GRU` and `_translate_nn_gru_to_cell(nn_gru, ...) -> GRULayer`. Underscore-prefixed per the existing test-file convention. Not promoted to `src/` — this is test infrastructure.
- **D-03:** Documented contract (in the helper's docstring): cell uses gate order `(r, z, n)` and `nn.GRU` does too. Translation is `weight_ih_l0 = cat([W_ir, W_iz, W_in], 0)` and `bias_ih_l0 = cat([b_ir, b_iz, b_in], 0)`; same for `_hh_l0` with the `h` family. The n-gate asymmetry (`r * (W_hn h + b_hn)`) is preserved by this layout because both sides apply it identically.

### Gate-order + n-gate asymmetry verification
- **D-04:** Three focused micro-tests in `tests/test_layer_parity.py`, run before the 75-combo grid:
  1. `test_gate_order_r_only`: set `W_ir = ones`, `W_iz = W_in = zeros`, run one step, assert only the r-gate sigmoid fires.
  2. `test_gate_order_z_only`: same pattern for z.
  3. `test_n_gate_asymmetry`: force `r ≈ 0` by setting `W_ir` and `b_ir` to large negative; verify the n-gate output equals `tanh(W_in x + b_in)` (not `tanh(W_in x + b_in + b_hn)`) — isolates the `r * (W_hn h + b_hn)` step from `(W_in x + b_in)` step. Both nn.GRU and our cell must agree on this asymmetry.
- **D-05:** Helper docstring includes a comment with the PyTorch GRU formula link (`torch.nn.GRU` doc) so future readers don't have to re-derive the ordering.
- **D-06:** No third hand-rolled "GRU from scratch" reference. We trust `nn.GRU` plus the micro-tests above. If the micro-tests pass and the grid still fails, the bug is in our cell, not in the translation.

### Precision policy + shape-grid execution time
- **D-07:** `torch.set_float32_matmul_precision('highest')` at the top of `tests/test_layer_parity.py`. We're auditing the math, not the TF32 mode. Tolerance < 1e-4 should comfortably hold under highest. Diverges from `test_triton_*.py` (which use 'high' / TF32 to test the kernel under realistic conditions) — that's correct: kernel tests audit kernels, this audits math.
- **D-08:** Full 75-combo grid kept, split by marker:
  - Default `pytest -q`: T ∈ {1, 8, 64} (B/H grid still full) — fast, runs everywhere.
  - `pytest -m slow`: T ∈ {512, 1024} (B/H grid still full) — marked `@pytest.mark.slow` per the existing convention in `tests/test_qat_smoke.py`.
- **D-09:** Each of the four test families (fwd, bwd, h_T, h_0≠0) gets its own parametrized function. Don't fuse them — failure messages need to point at the right family.

### Failing-test-before-fix discipline
- **D-10:** Two-commit pattern, verifiable via git log:
  - **Commit A:** failing test only (test file changes; no `src/` changes). Use `pytest --tb=short` output as bd notes payload.
  - **Commit B:** fix in `src/` (no test changes). The same test now passes.
- **D-11:** `bd create` per finding before commit A lands. The bd issue title is the test function name. `bd update <id> --notes` captures the failing pytest tail; `bd close <id>` happens after commit B passes locally and CI is green.
- **D-12:** No `@pytest.mark.xfail`. We want test-fails-CI as the signal, not xfail-passes-CI. xfail tests are silent in `pytest -q` — that defeats RPT-01.

### Claude's Discretion
- Exact `pytest.parametrize` formatting (id strings, fixture style).
- Whether to use `torch.allclose(..., atol=1e-4, rtol=0)` vs the existing relative-error idiom `(a - b).abs().max() / a.abs().max().clamp(1e-6) < 1e-4`. Use whichever produces clearer failure messages — the relative-error idiom is more diagnostic; the absolute form is simpler.
- Whether the four test families live in the same file or split. Same file (`test_layer_parity.py`) is the default.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project context
- `.planning/PROJECT.md` — milestone scope, Core Value, tolerance tiers, OOS list. Locked.
- `.planning/REQUIREMENTS.md` §REF-01..05 — the 5 requirements this phase implements; tolerance and shape grid are spec'd here.
- `.planning/ROADMAP.md` §"Phase 1: Reference-path parity vs nn.GRU" — success criteria.
- `SCOPE.md` §"Manual unroll, not nn.GRU", §"Gates are split, not fused", §"Sigmoid/tanh stay in float during QAT", §"Success criteria" — design rationale; explains *why* the cell exists and what `< 1e-5` cell parity already proves.
- `DEVELOPMENT.md` §"Phase 2 — fp32 cell parity" — documents the existing `< 1e-5` cell gate; do not loosen.

### Codebase
- `src/gru_qat/gru_cell.py` — `GRUCellQuant` source (single-step reference). The math the layer parity will exercise.
- `src/gru_qat/gru_layer.py` — `GRULayer.forward` (multi-step). The target of this phase's audit. Note `use_triton=False` path is the one we audit; `_forward_fast_dispatch` is out of scope.
- `tests/test_parity.py` — the existing `< 1e-5` cell parity contract. Phase 1 is layered on top, NOT a replacement. Read once for the parametrize / helper style.
- `.planning/codebase/TESTING.md` — full test conventions (TF32 setup, relative-error idiom, `_make_<kind>_layer` helpers, `pytest.importorskip` patterns).
- `.planning/codebase/CONVENTIONS.md` §"QAT-Specific Conventions", §"Naming Patterns" — math-significant variable names (`r`, `z`, `n`, `gi_*`, `gh_*`) and dtype discipline.

### External
- PyTorch `torch.nn.GRU` docs (linked in helper docstring): https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html — gate order, bias layout, formula. Authoritative for the translation helper.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **Relative-error idiom** (`tests/test_triton_diagonal.py:120-121`, `tests/test_triton_monarch.py:99-101`): `max_diff = (ref - tri).abs().max().item(); rel = max_diff / max(ref.abs().max().item(), 1e-6); assert rel < TOL`. Use this for the layer-parity grid; it gives a more diagnostic failure message than `torch.allclose`.
- **Per-batch error inspection** (`tests/test_butterfly_dispatch.py:206`): `rel_per_b = (a - b).abs().amax(dim=(0,2)) / b.abs().amax().clamp(1e-6); assert rel_per_b.max() < TOL, f"worst batch={rel_per_b.argmax()}"`. Useful if a layer grid case fails — points at the offending batch.
- **`_synthetic_loader`** (`tests/test_calibration.py:30`): pattern for generator-based loaders. Not needed for Phase 1 (we don't calibrate), but worth knowing the convention.
- **Seed-and-precision preamble**: `torch.manual_seed(0)` at the top of any randomness-using test; `torch.set_float32_matmul_precision("high")` for kernel tests. Phase 1 overrides to `"highest"` (see D-07).

### Established Patterns
- **One test file per concept**, mirroring `src/` (`STRUCTURE.md` "Naming Patterns"). `tests/test_layer_parity.py` is the natural home — it complements `tests/test_parity.py` (cell) without overloading it.
- **Module-level underscore helpers**, not `conftest.py` fixtures (`TESTING.md` "No fixtures via conftest"). Keep `_translate_*` helpers inside `tests/test_layer_parity.py`.
- **`from __future__ import annotations` + PEP 604 union syntax** (`CONVENTIONS.md` "Type Annotations"). Helpers should be fully typed.
- **`# noqa: E402` only after `pytest.importorskip`** — Phase 1 doesn't need importorskip (reference path is CPU-runnable), so no E402 suppressions.

### Integration Points
- `tests/test_layer_parity.py` is a new file. No changes to `src/` are made *speculatively*; only fix-commits land changes, and only when a parity test fails first (per D-10).
- If a parity bug is found in `src/gru_qat/gru_cell.py` or `gru_layer.py`, the fix touches that file plus possibly its existing tests if a regression-test update is needed — but never loosens `tests/test_parity.py` `< 1e-5` cell contract (PROJECT.md Constraint, REQ-08, locked).

</code_context>

<specifics>
## Specific Ideas

- The 4 test families are: (1) `out` parity, (2) `h_T` parity, (3) backward gradients on all 6 weight tensors + bias + `dx` + `dh_0`, (4) `h_0 ≠ 0` randomness. All four parametrized over the same T × B × H grid for symmetry.
- The 3 micro-tests for gate ordering are NOT parametrized over the grid — they're one-shot smoke tests run before the grid. Failing here means the helper is wrong; failing the grid means the cell math is wrong.
- The round-trip smoke test (nn.GRU → cell direction) is one test, not parametrized — it's a sanity check on the inverse helper.

</specifics>

<deferred>
## Deferred Ideas

- **Non-batched input** (`nn.GRU` accepts `(T, IN)` without batch dim) — not raised as a Phase 1 concern. If a user later asks for `GRULayer` to accept this, file as new requirement; not blocked by this audit.
- **Bidirectional / multi-layer parity** — out of scope per `SCOPE.md`. nn.GRU has both; `GRULayer` doesn't. Not auditing what doesn't exist.
- **Hand-rolled INT8 reference GRU** — explicitly chosen out at project level (PROJECT.md Key Decisions). Don't reintroduce in Phase 1 plans.
- **Slow-test execution budget** — D-08 marks long T as `slow`. If `pytest -m slow` ends up >5 minutes on the audit machine, prune the grid in a follow-up; do not change now.

</deferred>

---

*Phase: 1-Reference-path parity vs nn.GRU*
*Context gathered: 2026-05-13*

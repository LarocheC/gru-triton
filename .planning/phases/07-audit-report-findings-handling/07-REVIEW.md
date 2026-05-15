---
phase: 07-audit-report-findings-handling
reviewed: 2026-05-15T11:02:19Z
depth: standard
files_reviewed: 16
files_reviewed_list:
  - src/gru_qat/calibration.py
  - src/gru_qat/gru_cell.py
  - src/gru_qat/gru_layer.py
  - src/gru_qat/quantizers.py
  - src/gru_qat/ste.py
  - src/gru_qat/structure.py
  - src/gru_qat/triton_kernels/scan.py
  - src/gru_qat/triton_kernels/scan_butterfly.py
  - src/gru_qat/triton_kernels/scan_diagonal.py
  - src/gru_qat/triton_kernels/scan_monarch.py
  - tests/test_calibration.py
  - tests/test_scan_wrapper_validation.py
  - tests/test_triton_butterfly_strict.py
  - tests/test_triton_diagonal_strict.py
  - tests/test_triton_monarch_strict.py
  - tests/test_triton_scan_strict.py
findings:
  critical: 0
  warning: 5
  info: 4
  total: 9
status: issues_found
---

# Phase 7: Code Review Report

**Reviewed:** 2026-05-15T11:02:19Z
**Depth:** standard
**Files Reviewed:** 16
**Status:** issues_found

## Summary

Phase 07 is an audit-closure phase with three concerns: (07-01) converting bare
`assert` shape/dtype/`is_cuda` validation in the Triton-kernel wrapper callees
to `if ... raise ValueError` so the guards survive `python -O`; clearing
mypy-strict / ruff debt with `cast(...)` and `# type: ignore` annotations; and
(07-02) fixing a shared-`QuantizerConfig` aliasing bug via `copy.deepcopy` in
`make_quantizer`, plus adding a `divergence` pytest marker to deselect
TF32-reduction-order accepted divergences from the green gate.

The core changes are sound. The `deepcopy` fix is correct and well-targeted —
it addresses a real silent-no-op bug in `freeze()` where sibling quantizers
built from the same recipe field shared one config object. The assert→raise
conversions are mechanically correct and the new `test_scan_wrapper_validation.py`
gives them live-path coverage. The `divergence` marker is properly registered
in `pyproject.toml`.

No blockers. The findings below are quality and robustness concerns: the
`divergence` marker design effectively converts whole strict-tier grids into
non-gating tests (a real coverage-erosion risk that should be a conscious
decision, not a side effect), a dead `factory()` function left in
`quantizers.py`, and a few smaller issues around the validation conversions and
new test code.

## Warnings

### WR-01: `divergence` marker masks entire strict-tier grids — silent erosion of the parity gate

**File:** `tests/test_triton_scan_strict.py:627-669`, `tests/test_triton_monarch_strict.py:444-484`, `tests/test_triton_butterfly_strict.py:274-281`, `tests/test_triton_diagonal_strict.py:344-360`

**Issue:** The `_DIV_*` sets are not "the empirical failure list" the comments
claim — they are full cross-products of the entire parametrize grid. For
example `_DIV_SCAN_FWD` enumerates every `(T, B, H)` in `FAST_DENSE_GRID`, and
`_DIV_MONARCH_FWD` / `_DIV_MONARCH_BWD` enumerate the entire fast Monarch grid.
The net effect: `test_scan_fwd_strict_matches_reference`,
`test_scan_bwd_strict_matches_reference`, both Monarch fp32 strict tests, and
the whole Butterfly bwd grid are now **100% deselected** under the
`-m "not divergence"` gate. The project's core value is "every code path that
claims to compute a GRU must produce numerically equivalent output to
`torch.nn.GRU`" — marking the entire dense fp32 strict fwd/bwd grid as
`divergence` means the default gate no longer verifies dense fwd/bwd parity at
*any* shape. A future regression that pushes drift from ~8e-4 to ~8e-2 would
pass the green gate silently. The per-comment justification ("which exact
shapes exceed it is autotune-config dependent, so the whole grid is marked")
is plausible for a tight `< 5e-4` bound, but whole-grid marking is exactly the
"never mark a whole function" Pitfall-1 the comments cite — applied at grid
granularity instead of function granularity, with the same consequence.

**Fix:** Either (a) keep the strict bound but split into a tight tier (marked
`divergence`) and a looser-but-still-meaningful tier (e.g. `< 5e-2`) that stays
in the green gate so gross regressions are still caught; or (b) restrict the
`_DIV_*` sets to the *actually observed* failing ids from `07-pytest-output.txt`
rather than the full cross-product, so newly-failing shapes surface as gate
failures. Whichever is chosen, document in `AUDIT-REPORT.md` that the dense
fp32 strict fwd/bwd grid is no longer gating and what compensating coverage
exists.

### WR-02: Dead `factory()` function left in `quantizers.py`

**File:** `src/gru_qat/quantizers.py:267-277`

**Issue:** 07-01 removed `factory` from the `gru_cell.py` import list (the cell
now calls `make_quantizer(...)` directly at every insertion point — confirmed:
no `factory(` call sites remain in `src/`). The `factory()` function and its
docstring still reference `GRUCellQuant` as the consumer ("Used by
`GRUCellQuant` so each of the six weight quantizers gets its own parameters")
which is now stale. It is also not in `__init__.py`'s `__all__`. This is dead
code with a misleading docstring — mypy-strict and ruff will not flag an
unused module-level public function, so it lingers.

**Fix:** Remove `factory()` and the now-orphan `QuantizerFactory` type alias if
nothing else consumes them, or — if it is intentionally kept as public API for
downstream callers — update the docstring to drop the false `GRUCellQuant`
claim and add it to `__all__` so the intent is explicit.

### WR-03: `make_quantizer` deepcopy silently breaks any caller relying on config identity

**File:** `src/gru_qat/quantizers.py:246-264`

**Issue:** The fix is correct for the freeze-aliasing bug, but it changes a
contract silently: before, `q.config is passed_config` held; now every
quantizer owns a private copy. Any code that built a quantizer and then mutated
the *original* config expecting the change to propagate (e.g. flipping
`mode` to `min_max` on a shared `recipe.hidden` object) will now silently
no-op. `calibration.calibrate` does exactly this kind of mutation but operates
on `m.config` (the per-quantizer copy) so it is safe — however the behavioral
change is undocumented at the `QuantizerConfig` / `QuantRecipe` level. A user
following the docstring on `QuantizerConfig:51` ("The cell takes a *factory* …
so that each insertion point can have its own state") would not expect config
*identity* to also be broken.

**Fix:** Add a one-line note to `QuantizerConfig`'s docstring (and/or
`QuantRecipe`) that `make_quantizer` deep-copies the config and post-build
mutations of the original do not propagate. Low effort, prevents a confusing
debugging session for a downstream caller.

### WR-04: `dict`-batch path in `calibrate` is reachable but untyped/untested for the documented loader contract

**File:** `src/gru_qat/calibration.py:33`, `:102-103`

**Issue:** 07-01 changed the `loader` annotation from `Iterable` to
`Iterable[object]`. The docstring (`:49-52`) documents only "A single tensor"
and "A tuple/list" as supported loader yields, but the body also handles
`dict` (`module(**batch)`) and raises `TypeError` otherwise. The annotation
`Iterable[object]` is now permissive enough that mypy will not catch a loader
yielding an unsupported type at a call site — the only guard is the runtime
`TypeError`. This is pre-existing behavior, but 07-01 touched this line and the
docstring/annotation/body are now three-way inconsistent (docstring says
tensor/tuple/list; body also accepts dict; annotation says `object`).

**Fix:** Align the docstring with the body — add `dict` to the documented
"yields" list (`:49-52`), matching the `TypeError` message which already says
"expected Tensor / tuple / list / dict".

### WR-05: New validation tests assert error *type* but never assert the error *message* is actionable

**File:** `tests/test_scan_wrapper_validation.py:65-249`

**Issue:** Every test in the new file asserts `pytest.raises((ValueError,
RuntimeError))` and `not isinstance(excinfo.value, AssertionError)`. That
proves the guard survives `python -O`, which is the stated goal. But the whole
point of converting `assert h0.shape == (B, H)` into
`raise ValueError(f"h0 shape must be (B, H)=... got ...")` is the *actionable
message* — and no test checks that the message contains the offending
field/shape. A future refactor that replaces the descriptive `ValueError` with
a bare `raise ValueError()` (or one with a wrong field name) would still pass
every test here. The CLAUDE.md error-handling convention explicitly requires
"a message containing the offending field name and the constraint".

**Fix:** Add a `match=` argument to the malformed-shape `pytest.raises` calls
(e.g. `pytest.raises(_EXPECTED, match="h0 shape")`) so the test pins the
message to the offending field. At minimum do this for one representative case
per wrapper.

## Info

### IN-01: `gru_scan_forward` `is_cuda` guard message can mislead — only `gi` is required CUDA in some callees

**File:** `src/gru_qat/triton_kernels/scan.py:1707-1713`

**Issue:** `gru_scan_forward` raises one combined error if any of
`gi/h0/Wh_cat/bh_cat` is non-CUDA, listing all four devices — good. But the
diagonal/monarch/butterfly callees only check a subset (`gi` + the weight
tensor) and silently rely on `.contiguous()` / kernel launch to fail for a
non-CUDA `h0` or `bh_cat`. The validation hardening is uneven across the four
kernels. Not a correctness bug (a non-CUDA `h0` will still fail loudly inside
the kernel launch), but the "clear error" guarantee 07-01 set out to provide
is only fully delivered for the dense path.

**Fix:** For consistency, extend the diagonal/monarch/butterfly `is_cuda`
checks to cover `h0` and `bh_cat`, or document that those callees intentionally
validate only the dispatch-critical tensors.

### IN-02: `# type: ignore[dict-item]` on `axis` summary entry hides a real type mismatch

**File:** `src/gru_qat/calibration.py:123`

**Issue:** `summary` is typed `dict[str, dict[str, float | list[float]]]` but
`"axis"` stores `m.config.axis` which is `int | None`. `None` is not assignable
to `float | list[float]`, hence the `# type: ignore[dict-item]`. This is a
genuine type hole papered over rather than fixed — the summary value type is
simply wrong for the `axis` and `bits` keys (`bits` is `int`, also not in the
declared union but `int` is a subtype of `float` so it slips through).

**Fix:** Widen the summary value type to
`dict[str, float | int | list[float] | None]` and drop the `# type: ignore`.
Pre-existing, but 07-01's mypy-debt-clearing pass is the right moment to fix it
properly rather than suppress.

### IN-03: `Wh_cat` malformed-shape test exercises only the hidden-dim mismatch

**File:** `tests/test_scan_wrapper_validation.py:180-190`

**Issue:** `test_gru_scan_rejects_malformed_wh_shape` passes `Wh_cat` as
`(3*H, H+1)`. The wrapper computes `H = three_H // 3` from `gi`, so this hits
the `Wh_cat.shape != (3*H, H)` branch. Fine — but the first dim (`3*H`) of
`Wh_cat` is never independently exercised; a `Wh_cat` of `(3*H+1, H)` would
also be a meaningful malformed case and is not covered. Minor coverage gap in
an otherwise thorough new file.

**Fix:** Optional — add a row-dim-mismatch case if cheap; otherwise leave as is.

### IN-04: `_div_param` helper is duplicated verbatim across four strict test files

**File:** `tests/test_triton_scan_strict.py:672-678`, `tests/test_triton_monarch_strict.py:487-492`, `tests/test_triton_butterfly_strict.py:284-289`, `tests/test_triton_diagonal_strict.py:363-368`

**Issue:** The `_div_param` helper is copy-pasted into all four strict test
files. The codebase has an explicit D-18 convention ("small (<30 LOC) helper,
prefer duplicate over import") so this is *consistent with project policy* and
not a defect — flagged only so the duplication is a recorded, conscious choice
rather than an oversight. If the marker logic ever needs to change, four files
must be edited in lockstep.

**Fix:** None required — consistent with the documented D-18 convention. Noted
for awareness only.

---

_Reviewed: 2026-05-15T11:02:19Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_

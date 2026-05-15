# Phase 7: Audit report + findings handling - Research

**Researched:** 2026-05-15
**Domain:** Findings triage / closure phase — bd-issue disposition, pytest marker engineering, mypy/ruff cleanup, audit-report authoring
**Confidence:** HIGH (all 14 issues inspected directly via `bd show`; lint baselines measured live; n20 fix site read in source)

## Summary

Phase 7 is a closure phase, not a feature phase. Its spine is a **per-issue triage** of the 14 open `bd` issues into the three CONTEXT.md buckets (FIX / ACCEPTED-DIVERGENCE / INDIVIDUAL), followed by four bounded work items: the n20 deepcopy fix + strict-test re-baseline, the 7rj `assert`→`ValueError` hardening, the `divergence` pytest marker, the mypy/ruff cleanup, and finally `AUDIT-REPORT.md`.

The triage is **decisive and evidence-backed**: every one of the four "genuinely-open" backward-drift issues (`mjy`, `lht`, `e7t`, `fpl`) — plus `q3k`, `lqk`, `5rk`, `in0`, `6dz` — has the *same single measured root cause*: TF32 `tl.dot` (or `tl.sum` warp-butterfly) reduction-order non-associativity. Every bd description states this explicitly, and the monarch reproducer (`.planning/debug/repro_monarch_rounding.py`) confirms it at ULP level (~1.79e-7 per-block divergence). **None of mjy/lht/e7t/fpl has a tractable non-TF32 root cause** — they all go to ACCEPTED-DIVERGENCE. The only genuine FIX-bucket bugs are `n20` (shared-config silent-correctness) and `7rj` (`assert` validation). `e0l` (hardware SMEM/K-dim limit) and `u00` (process race) are INDIVIDUAL. `4m6` (lint debt) is a hygiene FIX.

**Primary recommendation:** 4-wave plan — Wave 1: `7rj` wrapper hardening (touches `scan*.py`, isolated) ∥ mypy/ruff cleanup (touches `src/` broadly + test files; serialize against 7rj on the 4 `scan*.py` files). Wave 2: `n20` fix + strict-test re-baseline + `divergence` marker (all touch `tests/test_triton_*_strict.py` + `pyproject.toml` — keep as ONE plan to avoid the u00 race). Wave 3: bd-issue closure (resolution notes). Wave 4: `AUDIT-REPORT.md` (written last, reports final state).

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Per-issue triage / disposition | Planning artifact (this doc) | bd issue tracker | CONTEXT.md D-01 delegates the FIX-vs-DIVERGENCE call to research |
| n20 quantizer fix | `src/gru_qat/quantizers.py` (`make_quantizer`) | `tests/test_triton_*_strict.py` (re-baseline) | Config-isolation is a model-build concern; scale change propagates to strict tests |
| 7rj validation hardening | `src/gru_qat/triton_kernels/scan*.py` (wrappers) | — | Internal-API guard; mirrors `GRULayer.forward` ValueError convention |
| `divergence` marker | `pyproject.toml` (registration) | `tests/test_triton_*_strict.py` (application) | Marker is project-level config; application is per-test-file |
| mypy/ruff cleanup | `src/gru_qat/*` (mypy) + `tests/*` (ruff) | `pyproject.toml` (if config tweak needed) | Type/lint debt is broad src surface + a few test files |
| AUDIT-REPORT | repo root `AUDIT-REPORT.md` (NEW) | all `NN-SUMMARY.md` / `NN-VERIFICATION.md` | Pure aggregation — sources existing artifacts, does not re-derive |

## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** Triage all 14 open `bd` issues into FIX / ACCEPTED-DIVERGENCE / INDIVIDUAL with explicit criteria.
- **D-02 (FIX bucket — fix in-phase):** Phase 7 fixes every genuine tractable bug. Known members: `n20`, `7rj`. The mjy/lht/e7t/fpl FIX-vs-DIVERGENCE call is delegated to research. All FIX work follows D-37/D-50 two-commit discipline (failing test Commit A BEFORE fix Commit B). **No `@pytest.mark.xfail`.**
- **D-03 (ACCEPTED-DIVERGENCE):** The irreducible TF32 `tl.dot` family — `in0`, `q3k`, `lqk`, `5rk`, and any of mjy/lht/e7t/fpl confirmed purely TF32-rooted. NOT code-fixed. The `input_precision="ieee"` rewrite is explicitly out of scope. Each closed with a resolution note → AUDIT-REPORT residual-divergences section.
- **D-04 (INDIVIDUAL):** `e0l` — Monarch bwd SMEM/K-dim hardware limit; documented HW-limit, covered by existing `_skip_if_monarch_bwd_hw_limit`; kernel-tiling redesign deferred to v2. `u00` — F-04-05-D parallel-execution race; process finding, AUDIT-REPORT process-note, close (no code change).
- **D-05:** Introduce a `divergence` pytest marker registered in `pyproject.toml [tool.pytest.ini_options] markers`. Mark every strict-tier case whose failure is an irreducible TF32 ACCEPTED-DIVERGENCE. Green gate: `pytest -q -m "not divergence"` and `pytest -m "slow and not divergence" -q`. Marked tests stay LIVE (not skipped, not xfail). Operationalization recorded in AUDIT-REPORT.
- **D-06:** mypy → 0 errors (clears ~145-error baseline, scoped to `src/gru_qat`); `ruff check src tests` → 0 errors (clears src + ~23 test-file errors). `gru-triton-4m6` closed. Tests stay non-mypy-strict (mypy config-scoped to `src/gru_qat`).
- **D-07:** Fix `n20` via `deepcopy` (or per-quantizer config isolation) in `make_quantizer`. Absorbs the Phase 4 strict-test re-baseline: each affected strict test re-baselined to its post-fix bound OR moved into the `divergence` marker. Per-test call delegated to research/planning. Two-commit discipline applies to the n20 fix.
- **D-08:** `AUDIT-REPORT.md` at repo root. 4 sections: (a) 28-requirement status table {PASS, FIX, ACCEPTED-DIVERGENCE}; (b) per-phase summary sourced from `NN-SUMMARY.md`/`NN-VERIFICATION.md` (do not re-derive); (c) residual divergences — TF32 family as ONE consolidated entry with per-issue sub-bullets; (d) finding→bd-issue pointers.
- **D-09:** Audit `git log` to confirm test-before-fix ordering for Phase 1–6 findings. Where genuinely absent, document the gap — do NOT rewrite history. Phase 7's own fixes strictly follow two-commit discipline.
- **D-10:** End state — `bd ready` empty. Every issue CLOSED or DEFERRED with a v2 bd ref in `REQUIREMENTS.md`. ACCEPTED-DIVERGENCE issues are CLOSED with a resolution note.

### Claude's Discretion

- Per-issue FIX-vs-ACCEPTED-DIVERGENCE call for mjy/lht/e7t/fpl (this research resolves it — see Triage Table).
- For the n20 re-baseline: per affected strict test, re-baseline bound vs `divergence`-mark.
- Plan/wave structure — sequence so no two plans write the same file concurrently (the u00 race lesson). AUDIT-REPORT written LAST.
- Whether the `divergence` marker is applied test-function-wide or per-parametrize-case.

### Deferred Ideas (OUT OF SCOPE)

- `input_precision="ieee"` TF32-elimination kernel rewrites → v2.
- `gru-triton-e0l` Monarch-bwd kernel-tiling redesign for consumer-GPU SMEM → v2.
- ACT-01 (per-channel `min_max` observer done right), ACT-02 (LSQ/PACT) → v2 (already in REQUIREMENTS.md).
- PERF-01 / PERF-02 (cuDNN comparison + QAT-overhead bench) → v2.

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| RPT-01 | Every mismatch is captured by a failing test before any fix lands | D-09 git-log audit confirms Phase 4–6 followed D-37/D-50; Phase 7's own n20+7rj fixes follow it. Phases 1–3 to be audited; gaps documented not rewritten. The 14 bd issues each already have a regression test or strict-test case. |
| RPT-02 | Every finding has a bd issue capturing root cause, fix, regression test | All 14 issues already exist with measured root-cause data (verified via `bd show`). Phase 7 closes them with resolution notes; no new findings expected beyond n20/7rj which already have issues. |
| RPT-03 | Audit ends with `AUDIT-REPORT.md` (checked / passed / fixed / residual divergences) | D-08 structure; source artifacts mapped in the AUDIT-REPORT section below. |

## THE DELIVERABLE — Per-Issue Triage Table

All 14 issues classified with evidence. Root cause for every numerical issue was read directly from the `bd show` description.

| # | bd ID | Phase | Bucket | Measured root cause (verbatim from bd) | Phase 7 action |
|---|-------|-------|--------|----------------------------------------|----------------|
| 1 | `gru-triton-n20` | P5/P2 | **FIX** | Shared `QuantizerConfig` instance: `quant_h_in.config IS quant_h_out.config` → `freeze_all` sets `config.mode='frozen'` on first, second silently skips stat-copy, stays at `scale=1.0`. Silent accuracy loss. **Not TF32 — a genuine reference-bug.** | `deepcopy(config)` in `make_quantizer`. Two-commit. Absorb strict-test re-baseline (D-07). |
| 2 | `gru-triton-7rj` | P2 | **FIX** | `gru_scan*` wrappers validate shapes/dtype with bare `assert` (stripped under `python -O`) → degenerate shape gives kernel deadlock/OOB instead of a clear error. **Not numerical — a hardening bug.** | Convert wrapper `assert` → `if … raise ValueError/RuntimeError`, mirroring the Phase 6 `GRULayer.forward` guard. Two-commit. Touches all 4 `scan*.py`. |
| 3 | `gru-triton-4m6` | P3 | **FIX** (hygiene) | 145 mypy errors / 23 ruff errors, pre-existing baseline. Tracked, not a parity finding. | D-06 cleanup; close on 0/0. See mypy/ruff section. |
| 4 | `gru-triton-in0` | P2 | **ACCEPTED-DIVERGENCE** | Monarch fwd 142/162 fail `torch.equal` by exactly 1×h_scale. Confirmed: einsum full-fp32 vs tiled `tl.dot` `input_precision='tf32'`. Per-block fp32 differs ~1.79e-7 (ULP) → flips one INT8 step. Same as `gru-triton-rwm`. | No code fix. `divergence`-mark monarch fwd quant cases. Close w/ resolution note → AUDIT-REPORT. |
| 5 | `gru-triton-q3k` | P2 | **ACCEPTED-DIVERGENCE** | Monarch bwd ~61 fail. "Same root cause as F-04-VERIFIER-A: einsum vs tiled `tl.dot` reduction-order non-associativity… backward path compounds via STE accumulation through clipped regions." | No code fix. `divergence`-mark monarch bwd quant cases. Close w/ note. |
| 6 | `gru-triton-lqk` | P2 | **ACCEPTED-DIVERGENCE** | Butterfly bwd large-magnitude exceed h_scale. "Same root cause family… TF32 reduction-order non-associativity. Butterfly compounds across `log_H` stages." | No code fix. `divergence`-mark butterfly bwd quant cases. Close w/ note. |
| 7 | `gru-triton-5rk` | P2 | **ACCEPTED-DIVERGENCE** | Butterfly fwd `torch.equal` fails ~4×h_scale. "Likely root cause: butterfly's `log_H` stages compound TF32 noise." Same family. | No code fix. `divergence`-mark butterfly fwd quant cases. Close w/ note. |
| 8 | `gru-triton-mjy` | P2 | **ACCEPTED-DIVERGENCE** *(resolved open call)* | Dense bwd 18 failures (254–914% h_scale). bd states verbatim: **"Same root cause as F-04-VERIFIER-A: TF32 reduction-order non-associativity in `tl.dot`, amplified through STE backward into clipped regions. Larger B = more parallel accumulation = larger order-dependent drift."** No non-TF32 component. Subsumes `lht`. | No code fix — purely TF32-rooted. `divergence`-mark dense bwd large-magnitude / B=32 quant cases. Close w/ note. |
| 9 | `gru-triton-lht` | P3 | **ACCEPTED-DIVERGENCE** *(resolved open call)* | Dense bwd dWh_cat exceeds h_scale at T=512 large-magnitude (120%). "Root cause likely STE backward through clipping + TF32 reduction-order interaction." **Explicitly subsumed by `mjy`** (per mjy: "The existing F-04-05-A (gru-triton-lht) is subsumed by this issue"). | No independent fix — superseded by mjy. Close as duplicate-of-mjy w/ resolution note → mjy / AUDIT-REPORT. |
| 10 | `gru-triton-e7t` | P3 | **ACCEPTED-DIVERGENCE** *(resolved open call)* | Diagonal bwd long-T dbh drift ~1.5e-5 at T=1024. Root cause stated verbatim: **"warp-level butterfly across BLOCK_B via `tl.sum`"** vs PyTorch `tensor.sum()` parallel reduction — "Pure fp32-noise… no slab-leak / no algorithmic bug." Note: this is `tl.sum` (not `tl.dot`) but the *same* reduction-order-non-associativity phenomenon. The two listed "fixes" (manual pairwise reduction / emulate Triton order in reference) are kernel/reference rewrites = out-of-scope class. | No code fix — irreducible reduction-order noise. `divergence`-mark the diagonal slow-tier dbh case. Close w/ note. **NB:** AUDIT-REPORT residual section should sub-bullet e7t under the TF32 family but note it is `tl.sum`-rooted, not `tl.dot` — same phenomenon, different op. |
| 11 | `gru-triton-fpl` | P3 | **ACCEPTED-DIVERGENCE** *(resolved open call)* | Diagonal fwd single failure at (large-magnitude, 64,32,128). Probe confirmed worst ratio = 1.0 (exactly one INT8 step). bd: **"Same root cause family: TF32 reduction-order non-associativity in the elementwise-diagonal accumulator… puts the per-step value right on the rounding boundary."** | No code fix. `divergence`-mark the one diagonal fwd large-magnitude case. Close w/ note. |
| 12 | `gru-triton-6dz` | P3 | **ACCEPTED-DIVERGENCE** *(resolved open call)* | Phase 2 strict-tier small-shape failures exceed Option C `< 5e-4`: butterfly fwd `[8-1-32]` ~9.3e-3, monarch bwd ~7.4e-4. bd: **"Same TF32-via-`tl.dot` root cause as Phase 2 disposition Option C (`gru-triton-rwm`), but at small-shape edge cases the bound is exceeded."** Stash-verified pre-existing. | No code fix. `divergence`-mark the offending small-shape Phase-2 strict cases (these are the *non-quant* `< 5e-4` strict tests, distinct from the Phase-4 quant cases). Close w/ note. |
| 13 | `gru-triton-e0l` | P2 | **INDIVIDUAL** (hardware limit) | Monarch bwd ~54 failures on RTX 2000 Ada: SMEM OOM (`blksz_pad >= 128` needs ~147KB, HW has 100KB) + `tl.dot` K<16 constraint (`blksz_pad < 16`). bd: "These are hardware-capacity and Triton-tile-constraint issues, not numerical correctness issues." | No code fix (kernel-tiling redesign is v2). Already covered by `_skip_if_monarch_bwd_hw_limit`. AUDIT-REPORT HW-limit entry. Close as INDIVIDUAL-hardware w/ note. |
| 14 | `gru-triton-u00` | P3 | **INDIVIDUAL** (process) | F-04-05-D parallel-execution race: Plan 04-04's commit included Plan 04-02's strict-test diff. bd: "Same pattern as Phase 2 races. Root cause unconfirmed — `.beads/hooks/pre-commit` suspected." | No code fix — process finding. Already mitigated by single-plan discipline (Phases 5–6). AUDIT-REPORT process-note. Close as INDIVIDUAL-process w/ note. |

### Bucket totals

| Bucket | Count | Issues |
|--------|-------|--------|
| **FIX** | 3 | `n20`, `7rj`, `4m6` |
| **ACCEPTED-DIVERGENCE** | 9 | `in0`, `q3k`, `lqk`, `5rk`, `mjy`, `lht`, `e7t`, `fpl`, `6dz` |
| **INDIVIDUAL** | 2 | `e0l` (hardware), `u00` (process) |

### Verdict on the four delegated open calls

> **All four — `mjy`, `lht`, `e7t`, `fpl` — are ACCEPTED-DIVERGENCE. None has a tractable non-TF32 root cause.**

Evidence chain:
- **`mjy`** — bd text is unambiguous: "Same root cause as F-04-VERIFIER-A: TF32 reduction-order non-associativity in `tl.dot`." The "STE clipping" mention is an *amplifier* of the TF32 noise, not an independent bug — STE backward is mathematically correct; it just propagates the order-dependent drift. No code defect.
- **`lht`** — explicitly declared subsumed by `mjy` ("The existing F-04-05-A (gru-triton-lht) is subsumed by this issue"). It is the same dense-bwd phenomenon at one shape. Close as duplicate.
- **`e7t`** — the bd description does a full ~30-min investigation and states the verdict: "Confirmed: no slab-leak / no algorithmic bug. Pure fp32-noise." The root cause is `tl.sum` warp-butterfly vs `torch.sum` tree order. Both proposed "fixes" are rewrites (kernel reduction-tree change OR reference rewrite) — i.e. the same class of out-of-scope kernel surgery as `input_precision="ieee"`.
- **`fpl`** — probe-confirmed worst ratio = exactly 1.0 (one INT8 step), identical single-flip pattern to `in0`. "Same root cause family: TF32 reduction-order non-associativity."

A tractable non-TF32 FIX would look like: a slab-zeroing bug, an OOB index, a wrong gate-order, a missing barrier — *algorithmic* defects. The `c001a8a` dWh/dbh slab-init bug and the `c2a` butterfly batch-invariance bug were exactly that and were fixed. None of mjy/lht/e7t/fpl shows that signature; every bd description has already ruled it out by inspection.

## The `divergence` pytest marker (D-05)

### Registration

`pyproject.toml` already has the `markers` list. Add one entry (and `cuda_only` is *not* registered there — it is defined per-file as `pytest.mark.skipif`, so it does not need registering; only collected-and-run markers do):

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "divergence: marks a strict-tier case whose failure is an irreducible TF32 tl.dot / tl.sum reduction-order ACCEPTED-DIVERGENCE (run with '-m divergence' to reproduce; deselect with '-m \"not divergence\"'). See AUDIT-REPORT.md.",
]
```
[CITED: pytest docs — `pytest.ini_options markers` registers custom markers; unregistered markers warn under `--strict-markers`.]

### Application: per-parametrize-case, NOT per-function

**Recommendation: `pytest.param(..., marks=pytest.mark.divergence)` per offending parametrize case.** Reason: the strict tests are heavily parametrized (`@pytest.mark.parametrize("T,B,H", FAST_DENSE_GRID)` × `@pytest.mark.parametrize("cls", _QUANT_CLASSES)`), and per `04-DISPOSITION.md` the divergence is **cluster-specific** — e.g. dense bwd diverges only for `large-magnitude` or `B=32`, while `realistic, B<32` still holds `torch.equal`. Marking the whole function would hide the clean cases. Marking per-case keeps the clean cases in the green `-m "not divergence"` gate and isolates exactly the divergent tuples.

Mechanics: convert the grid constants (`FAST_DENSE_GRID`, `QUANT_FAST_GRID`, etc.) — or the `cls` axis — so divergent combos are wrapped:

```python
# Source: pytest docs — marking individual parametrizations
QUANT_FAST_GRID = [
    (8, 4, 64),
    pytest.param(64, 32, 512, marks=pytest.mark.divergence),  # TF32 tl.dot — gru-triton-q3k
    ...
]
```

For a case where the *class* axis (not T/B/H) selects divergence, wrap the `cls` parametrize values instead, or split the test into a clean function + a `divergence`-marked sibling function (acceptable when the whole cls partition diverges, e.g. butterfly bwd where ALL classes diverge → `lqk`).

### Marker composition

`-m` takes a boolean expression over marker names:
- Fast tier green gate: `pytest -q -m "not divergence"` — runs everything except divergence-marked cases (slow still deselected only if you also add `not slow`; the strict files have non-`slow` cases, so the gate as written by D-05 is literally `pytest -q -m "not divergence"`).
- Slow tier: `pytest -m "slow and not divergence" -q` — runs slow-marked, divergence-excluded.
- Reproduce divergences: `pytest -m divergence` (collects both slow and non-slow divergence cases; add `and not slow` to skip the long ones).
[CITED: pytest docs — "Marking test functions and selecting them for a run" — `-m` accepts `and`/`or`/`not`.]

### Which files / cases carry the marker

| File | Cases to mark | bd ref |
|------|---------------|--------|
| `tests/test_triton_scan_strict.py` | dense bwd quant cases for `large-magnitude` (any B>1) and `B=32` (`realistic`/`near-saturation`); plus n20-rebaseline residuals confirmed TF32-rooted | `mjy`, `lht`, `n20` |
| `tests/test_triton_monarch_strict.py` | monarch fwd quant — all classes; monarch bwd quant — `large-magnitude`; the Phase-2 small-shape strict cases exceeding `< 5e-4` | `in0`, `q3k`, `6dz` |
| `tests/test_triton_butterfly_strict.py` | butterfly fwd quant (`[8-1-32]` small shapes) and butterfly bwd quant — all classes; Phase-2 strict `[8-1-32]` | `5rk`, `lqk`, `6dz` |
| `tests/test_triton_diagonal_strict.py` | diagonal fwd quant `large-magnitude` `(64,32,128)`; diagonal bwd slow-tier dbh case at T=1024 | `fpl`, `e7t` |

Estimate: roughly **30–60 parametrized cases** across the 4 files carry the marker (the disposition table spans 4 kernels × ~3 classes × ~10 grid points, but only the divergent clusters get marked — the clean dense-fwd / diagonal-fwd / diagonal-bwd clusters do NOT).

## The gru-triton-n20 fix + strict-test re-baseline (D-07)

### Confirmed bug + fix shape

Verified in source: `FakeQuantize.__init__` stores `self.config = config` **by reference** (`quantizers.py:73`). `make_quantizer` (`quantizers.py:245`) passes the config straight through. `GRUCellQuant` builds `quant_h_in` and `quant_h_out` both from `make_quantizer(recipe.hidden)`, and the six `quant_W_*` from `recipe.weight` — so siblings share one `QuantizerConfig` object. `FakeQuantize.freeze()` mutates `self.config.mode = "frozen"` (`quantizers.py:105`); the second sibling then reads `mode == 'frozen'` and skips the stat-copy → stays at `scale=1.0`.

Fix (CONTEXT D-07, confirmed correct):
```python
def make_quantizer(config: QuantizerConfig) -> FakeQuantize:
    from copy import deepcopy
    config = deepcopy(config)
    if config.bits >= 32:
        return Identity(config)
    ...
```
This gives every quantizer its own config instance. Note `factory()` (`quantizers.py:255`) wraps `make_quantizer`, so the fix propagates to all six weight quantizers automatically — no second edit needed.

### Strict-test re-baseline impact

The n20 bd description states the entanglement precisely: Phase 4's `dense fwd torch.equal` and `diagonal fwd realistic/near-saturation torch.equal` contracts **depended on the bug** — both reference and Triton paths used the broken `scale=1.0`, so they matched byte-for-byte. After the fix, both paths quantize correctly with per-channel scales, and their tile-by-tile TF32 multiplications diverge by exactly 1×h_scale at INT8-rounding boundaries.

bd reports concrete breakage: `test_scan_quant_fwd[realistic-8-4-32]` and `[realistic-8-4-64]` go from `max_abs_diff=0.0` to `2.0e-2 = 1*h_scale` — "and so on across the QUANT_FAST_GRID for dense, diagonal, monarch." CONTEXT.md estimates **~18+ Phase 4 strict tests** break.

**Per-test decision rule (re-baseline vs `divergence`-mark):**
- If post-fix residual is **exactly 1×h_scale (one INT8 step)** and matches the `repro_monarch_rounding.py` ULP signature → it is the *same* TF32 reduction-order divergence → **`divergence`-mark** it (it is now a genuine ACCEPTED-DIVERGENCE, indistinguishable from `in0`/`fpl`).
- If post-fix residual is **0.0 (`torch.equal` still holds)** → no action; the clean cluster survives.
- If post-fix residual is **larger than the documented cluster bound and not a clean INT8-step multiple** → that signals a *genuine new bug* exposed by correct scales → **fails the audit**, must be FIX'd with a two-commit pair. (Research expectation: this will not happen — the bd repro already shows the residual is exactly 1×h_scale, the TF32 signature.)

Practical guidance for the planner: the n20 fix is a Commit-A failing test (CAL-02 extended to `quant_h_out`, mirroring the existing `quant_x`-scoped test) → Commit-B the deepcopy. Then a *separate* re-baseline commit pair updates the broken strict-test cases — most will become `divergence`-marked (they revert to the same 1×h_scale TF32 floor). Reconcile every change against `04-DISPOSITION.md`'s per-cluster `h_scale_mult` table.

## The mypy / ruff cleanup (D-06)

### Measured baseline (live run, 2026-05-15)

**mypy: 145 errors in 10 files.** This is a *small number of systematic patterns*, NOT 145 distinct issues:

| mypy code | count | nature | bulk-fixable? |
|-----------|-------|--------|---------------|
| `no-untyped-def` | 39 | missing annotations on `@triton.jit` kernels + helpers | Yes — annotate or `# type: ignore` the jit kernels (jit-decorated fns can't be typed cleanly) |
| `list-item` / `union-attr` / `operator` / `call-arg` | 39 | `torch-structured` Module-vs-Tensor union confusion in `scan_butterfly.py` (`.b`, `.twiddle` attr access) | Yes — narrow with `assert isinstance(...)` or `cast()` |
| `unused-ignore` | 15 | stale `# type: ignore` comments | Yes — trivial delete (mechanical) |
| `untyped-decorator` / `no-untyped-call` | 21 | `@triton.autotune` / `@triton.jit` decorators are untyped | Yes — one pattern; `# type: ignore[misc]` or stub |
| `import-untyped` | 11 | `triton` / `torch_structured` have no stubs | Yes — `[[tool.mypy.overrides]] ignore_missing_imports` for those modules |
| `no-any-return` | 13 | functions returning `Any` from triton calls | Yes — annotate the cast |
| others (`type-arg`, `return-value`, `comparison-overlap`, `assignment`) | 5 | one-offs in `gru_layer.py` | Per-line fixes |

Distribution by file: `scan_butterfly.py` 37, `scan.py` 33, `scan_monarch.py` 27, `scan_diagonal.py` 27, `ste.py` 16, `structure.py` 11, `gru_layer.py` 6, `gru_cell.py` 4. **~124 of 145 (86%) are in the 4 `scan*.py` Triton-kernel files** and stem from ~3 root patterns: untyped `@triton.jit`/`@triton.autotune` decorators, missing kernel-fn annotations, and `torch-structured` Module/Tensor union narrowing. Effort estimate: **medium** — the kernel-file errors clear in bulk via per-module ignore config + an annotation pass; `ste.py`/`structure.py`/`gru_layer.py` need ~30 targeted fixes.

**ruff: 23 errors.** Even simpler:

| ruff code | count | nature | fix |
|-----------|-------|--------|-----|
| `E402` | 9 | module-import-not-at-top (the `pytest.importorskip` idiom in test files) | Add `# noqa: E402` (the project already uses this idiom — see `test_structure.py`); or `per-file-ignores` in `pyproject.toml` |
| `F841` | 8 | unused local (intentional in kernels — bias loads) | `# noqa: F841` (project already does this in `scan_butterfly.py:495`) |
| `F401` | 4 | unused import | delete |
| `E731` | 2 | lambda-assignment | convert to `def` |

ruff reports `4 fixable with --fix` + 10 hidden unsafe-fixes. **Effort: low** — most are the established `# noqa` idiom already used elsewhere in the repo, or trivial deletions. The cleanest approach for `E402`/`F841` in test files is a `[tool.ruff.lint.per-file-ignores]` block rather than scattering `# noqa`, since these are *deliberate* patterns the project endorses.

> **Pitfall:** `# type: ignore` comments are themselves the source of 15 `unused-ignore` errors. Fix the underlying error first, THEN remove the now-redundant ignore — doing it in the wrong order re-introduces errors. Also: do not weaken `[tool.mypy] strict = true` — D-06 requires the baseline cleared *under* strict mode. Use scoped `[[tool.mypy.overrides]]` for missing third-party stubs only.

## AUDIT-REPORT.md (D-08)

`AUDIT-REPORT.md` at repo root. Four sections; every section sources existing artifacts — **do not re-derive**.

| Section | Source artifacts | Notes |
|---------|------------------|-------|
| (a) 28-requirement status table | `REQUIREMENTS.md` v1 list (REF-01..05, TRI-01..06, STR-01..03, QNT-01..04, CAL-01..03, EDG-01..04, RPT-01..03); status ∈ {PASS, FIX, ACCEPTED-DIVERGENCE} | REQUIREMENTS.md already marks 25/28 `[x]`; RPT-01/02/03 close in Phase 7. Requirements touched by a TF32 divergence (TRI-01/03/04, QNT-01/02/03) are PASS-with-divergence — cite the consolidated divergence entry. |
| (b) per-phase summary | `01..06` `NN-SUMMARY.md` + `NN-VERIFICATION.md` | Quote/condense — do not rewrite. Note the Phase 6 SUMMARY/live discrepancy (the verifier flagged the SUMMARY as stale re: c2a — use the VERIFICATION as authoritative). |
| (c) residual known-but-accepted divergences | `04-DISPOSITION.md` per-cluster table; Phase 2 Option C Key Decision in `PROJECT.md`; `.planning/debug/repro_monarch_rounding.py`; the 9 ACCEPTED-DIVERGENCE bd issues | **ONE consolidated TF32 entry** with per-issue sub-bullets (`in0`, `q3k`, `lqk`, `5rk`, `mjy`, `lht`, `e7t`, `fpl`, `6dz`). State the root cause once, why the fix (`input_precision="ieee"`) is out of scope, and the `divergence`-marker operationalization of criterion #3. Plus a separate INDIVIDUAL entry for `e0l` (HW) and a process-note for `u00`. |
| (d) finding→bd pointers | the 14 bd IDs | One pointer per finding row. |

**Structural pitfalls:**
- D-05 mandates AUDIT-REPORT *plainly states* the criterion-#3 reinterpretation: green gate = `pytest -q -m "not divergence"`, not literal `pytest -q`. This is itself an audit finding — give it its own subsection.
- e7t is `tl.sum`-rooted not `tl.dot`-rooted — sub-bullet it under the TF32 family but annotate the distinction (same reduction-order phenomenon, different op) so the consolidated entry stays accurate.
- D-09: the per-phase summary must include the git-log test-before-fix audit result. Phases 4–6 followed D-37/D-50; Phases 1–3 predate it — where ordering is absent, document the gap, do not rewrite history.
- AUDIT-REPORT is written LAST so it reports post-fix final state (n20 fixed, marker in place, lint green).

## Plan / Wave Sequencing

Same-file write conflicts are the central risk — `u00` is the literal lesson (a parallel plan clobbered another's strict-test diff). File-touch map:

| Work item | Files touched |
|-----------|---------------|
| 7rj wrapper hardening | `src/gru_qat/triton_kernels/scan.py`, `scan_diagonal.py`, `scan_monarch.py`, `scan_butterfly.py` + new wrapper tests |
| mypy cleanup | `src/gru_qat/*` (all 10 files incl. the 4 `scan*.py`) + `pyproject.toml` |
| ruff cleanup | `tests/test_triton_scan.py`, `test_structure.py`, `test_butterfly_dispatch.py`, `test_triton_monarch.py`, `test_parity.py` + `src/gru_qat/{gru_cell,scan,scan_butterfly}.py` + `pyproject.toml` |
| n20 fix | `src/gru_qat/quantizers.py` + `tests/test_calibration.py` |
| n20 re-baseline + `divergence` marker | `tests/test_triton_*_strict.py` (all 4) + `pyproject.toml` |
| AUDIT-REPORT | `AUDIT-REPORT.md` (new, isolated) |

**Recommended sequencing — 4 waves, single plan per shared file:**

- **Wave 1 (one plan).** mypy + ruff + 7rj together. Reason: 7rj edits the 4 `scan*.py` wrappers; mypy cleanup also edits those same 4 files; ruff touches `scan.py`/`scan_butterfly.py`. These **must not be concurrent plans** — fold them into one lint-and-harden plan (or strictly serialize). All three close `4m6` + `7rj`. `pyproject.toml` is edited here (mypy overrides + ruff per-file-ignores) — and again in Wave 2 (marker) — so either keep `pyproject.toml` edits in one wave or accept that Wave 1 and Wave 2 both touch it sequentially (never concurrently).
- **Wave 2 (one plan).** n20 fix + strict-test re-baseline + `divergence` marker. All touch `tests/test_triton_*_strict.py` and `pyproject.toml`. CONTEXT.md D-07 explicitly couples the n20 fix to the re-baseline; the marker is applied to the same cases the re-baseline touches. **Keep as ONE plan** — this is the exact u00 scenario. Order within the plan: (1) Commit-A failing `quant_h_out` test, (2) Commit-B deepcopy fix, (3) re-run strict suite, (4) re-baseline + `divergence`-mark the broken cases as a commit pair.
- **Wave 3 (one plan).** bd-issue closure — write resolution notes on all 14, close FIX/ACCEPTED-DIVERGENCE/INDIVIDUAL, record any v2 deferrals in `REQUIREMENTS.md`. Touches only `bd` + `REQUIREMENTS.md`. Depends on Waves 1–2 being done (notes reference final state).
- **Wave 4 (one plan).** `AUDIT-REPORT.md`. Isolated new file. Written last — reports final post-fix state and the closed-bd outcomes.

This guarantees no two plans write the same file concurrently. If the granularity allows splitting Wave 1, the only safe split is mypy-src vs ruff-tests *as long as both avoid the 4 `scan*.py` files simultaneously* — but since 7rj, mypy, and ruff all converge on `scan*.py`, the single-plan Wave 1 is the safe default.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Skipping divergent tests | A custom skip helper / `xfail` | The `divergence` pytest marker + `-m "not divergence"` | D-05 mandates marked tests stay LIVE (executable documentation); `xfail` is explicitly banned project-wide |
| Per-quantizer config isolation | A custom config-clone method | `copy.deepcopy` in `make_quantizer` | D-07 specifies it; `QuantizerConfig` is a plain dataclass — `deepcopy` is correct and total |
| Monarch-bwd HW-limit skip | A new skip mechanism | Existing `_skip_if_monarch_bwd_hw_limit` | Already the e0l disposition mechanism (CONTEXT code-context) |
| 7rj validation | A custom error type | `if … raise ValueError`, mirroring `GRULayer.forward:169-176` | The Phase 6 EDG-04 guard is the exact convention to copy |
| Third-party missing stubs | Hand-written `.pyi` stubs for triton | `[[tool.mypy.overrides]] ignore_missing_imports = true` | triton/torch-structured have no stubs; overrides is the standard mypy answer |

## Common Pitfalls

### Pitfall 1: Marking a whole strict-test function instead of per-case
**What goes wrong:** The clean clusters (dense fwd `torch.equal`, diagonal fwd realistic/near-saturation, diagonal bwd) get pulled out of the green gate.
**How to avoid:** Use `pytest.param(..., marks=pytest.mark.divergence)` per divergent tuple. `04-DISPOSITION.md` tells you exactly which (kernel, direction, class, B) tuples diverge.

### Pitfall 2: Treating the n20 re-baseline residual as a new bug
**What goes wrong:** Panic when ~18 strict tests go from `0.0` to `2.0e-2`. It is NOT a regression — the bd description predicts it exactly.
**How to avoid:** Check the residual is exactly 1×h_scale (one INT8 step). If so, it is the TF32 divergence re-surfacing → `divergence`-mark. Only a non-INT8-multiple residual is a genuine bug.

### Pitfall 3: Removing `# type: ignore` before fixing the underlying error
**What goes wrong:** `unused-ignore` (15 of them) tempts you to bulk-delete ignores; if the underlying error is still live, deletion re-introduces a real error.
**How to avoid:** Fix the typed error first; mypy then reports the ignore as unused; then delete it. Or fix-and-delete the same line atomically.

### Pitfall 4: Concurrent plans writing `scan*.py` or strict-test files
**What goes wrong:** The `u00` race — one plan's commit silently includes another's diff.
**How to avoid:** Single-plan-per-shared-file. 7rj + mypy + ruff all converge on `scan*.py` → one Wave-1 plan. n20-rebaseline + marker both touch strict files → one Wave-2 plan.

### Pitfall 5: AUDIT-REPORT re-deriving phase narratives
**What goes wrong:** Re-writing what each phase did invites drift from the actual verified record (and the Phase 6 SUMMARY is itself stale vs its VERIFICATION).
**How to avoid:** D-08 part (b) — quote/condense `NN-VERIFICATION.md` (authoritative) over `NN-SUMMARY.md` where they disagree.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 9.0.3 + pytest-xdist 3.8.0 (from `uv.lock`) |
| Config file | `pyproject.toml [tool.pytest.ini_options]` |
| Quick run command | `uv run pytest -q -m "not divergence"` (post-marker; the new green gate) |
| Full suite command | `uv run pytest -m "slow and not divergence" -q` (slow tier) + the quick run |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| RPT-01 | Failing test precedes every fix | git-log audit (manual + grep) | `git log --oneline` inspection per finding | ✅ (git history) |
| RPT-01 (n20) | n20 fix has Commit-A failing test | unit | `uv run pytest tests/test_calibration.py -k freeze -q` | ⚠️ Wave 0 — extend CAL-02 to `quant_h_out` |
| RPT-01 (7rj) | 7rj wrappers raise ValueError on bad shape | unit | `uv run pytest tests/test_triton_scan_strict.py -k wrapper -q` | ❌ Wave 0 — new wrapper-validation tests |
| RPT-02 | Every finding has a bd issue | process check | `bd list --status=open` → empty after closure | ✅ (14 issues exist) |
| RPT-03 | AUDIT-REPORT.md exists, 4 sections | artifact check | `test -f AUDIT-REPORT.md` + section grep | ❌ Wave 0 — new file |
| D-06 | mypy 0 / ruff 0 | lint gate | `uv run mypy && uv run ruff check src tests` | ✅ (tools configured) |
| D-05 | `-m "not divergence"` is green | marker gate | `uv run pytest -q -m "not divergence"` | ⚠️ Wave 0 — marker not yet registered |

### Sampling Rate
- **Per task commit:** `uv run pytest <touched test file> -q` + `uv run ruff check <touched files>`.
- **Per wave merge:** `uv run pytest -q -m "not divergence"` (fast tier) + `uv run mypy`.
- **Phase gate:** `uv run pytest -q -m "not divergence"` green AND `uv run pytest -m "slow and not divergence" -q` green AND `uv run mypy` 0 AND `uv run ruff check src tests` 0, before `/gsd-verify-work`. CUDA host required for the Triton strict tests.

### Wave 0 Gaps
- [ ] `pyproject.toml` — register the `divergence` marker (one line) — needed before any marker application.
- [ ] `tests/test_calibration.py` — extend the CAL-02 test to assert `quant_h_out.scale` post-freeze (the Commit-A failing test for n20).
- [ ] `tests/test_triton_*_strict.py` — new `gru_scan*` wrapper-validation tests (Commit-A for 7rj) — pick one strict file or add to a small new `tests/test_scan_wrapper_validation.py`.
- [ ] `AUDIT-REPORT.md` — new repo-root file.
- [ ] No new framework install needed — pytest/mypy/ruff already in `[project.optional-dependencies] dev]`.

## Security Domain

> `security_enforcement` not present in `.planning/config.json` → treat as enabled, but this phase has **no security surface**.

Phase 7 is a closure/hygiene phase: bd-issue triage, a config-isolation bugfix, an internal-API validation hardening, lint cleanup, and a markdown report. No authentication, session, access-control, network, cryptography, or untrusted-input handling is added or modified. ASVS V1–V14: **none applicable**. The only input-validation-adjacent change (`7rj` `assert`→`ValueError`) *strengthens* robustness of an internal Python API against malformed shapes — it is a defensive-coding improvement, not a security control. No threat model change.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| python / uv | all work | ✓ | py3.10+, uv | — |
| pytest | all test gates | ✓ | 9.0.3 | — |
| mypy | D-06 | ✓ | 2.0.0 (ran live) | — |
| ruff | D-06 | ✓ | 0.15.12 (ran live) | — |
| bd (beads) | triage + closure | ✓ | ran `bd list`/`bd show` live | — |
| CUDA + triton GPU | strict-tier test re-run, marker verification, n20 re-baseline | ✗ on this research host (Triton tests skip) | — | The n20 re-baseline + `divergence`-marker verification REQUIRE a CUDA+Triton host (RTX 2000 Ada per `04-DISPOSITION.md`). Plan execution must run on the GPU host; the strict-suite numbers cannot be produced on CPU. |
| torch-structured | monarch/butterfly/ldr strict tests | unknown on GPU host | — | Installed via `uv pip install git+https://github.com/LarocheC/torch-structured`; ensure present on the execution host |

**Missing dependencies with no fallback:** CUDA+Triton GPU for Wave-2 strict-test re-baseline and marker verification — this is a hard requirement for executing/verifying Phase 7, not a blocker for planning. The triage, lint cleanup, 7rj fix, and AUDIT-REPORT authoring do not need a GPU; only the n20 strict-test re-baseline numbers do.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Strict `< 1e-5` Triton-vs-reference for all kernels | Tiered: `< 1e-5` diagonal, `< 5e-4` tight-TF32 for `tl.dot` kernels | Phase 2 Option C (2026-05-13) | The locked basis for ACCEPTED-DIVERGENCE — PROJECT.md Key Decision |
| Bit-identity (`torch.equal`) for all quant-on paths | Per-cluster `h_scale_mult` table | Phase 4 D-42 revised post-verifier (2026-05-14) | `04-DISPOSITION.md` is the contract n20 re-baseline reconciles against |
| Open bd issues left as plain-open carry-forward | All issues CLOSED or v2-DEFERRED, `bd ready` empty | Phase 7 (D-10) | Milestone closure invariant |

**Deprecated/outdated:**
- Phase 5 D-51 "the 4 strict files are locked" — explicitly *no longer applies in Phase 7* (CONTEXT canonical-refs: "Phase 7 OWNS them"). The planner must not treat the strict files as locked.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The n20 re-baseline residual will be exactly 1×h_scale (TF32 signature), so most affected tests become `divergence`-marked rather than re-baselined or FIX'd | n20 section | LOW — bd description already reports the measured `2.0e-2 = 1*h_scale` residual; if a non-INT8-multiple residual appears it is a genuine new bug requiring a FIX commit pair (the rule handles it) |
| A2 | The ~30–60 marked-case estimate | `divergence` marker section | LOW — derived from `04-DISPOSITION.md` cluster count; exact number is determined empirically on the GPU host; does not change the plan shape |
| A3 | mypy 145 errors clear without weakening `strict=true` (via overrides + annotation pass) | mypy section | MEDIUM — if some `@triton.jit` errors are unfixable under strict mode, scoped `# type: ignore[misc]` on the decorator is the documented fallback (project already uses `# type: ignore[override]` for autograd Functions) — does not weaken global strict |
| A4 | No genuine non-TF32 bug is hiding inside mjy/lht/e7t/fpl | Triage table | LOW — each bd description independently ruled out algorithmic defects by inspection; e7t had a documented 30-min investigation concluding "no slab-leak / no algorithmic bug" |

## Open Questions

1. **Exact count of n20-broken strict tests.**
   - What we know: CONTEXT.md says "~18+"; bd lists `[realistic-8-4-32]`, `[realistic-8-4-64]` and "so on across QUANT_FAST_GRID for dense, diagonal, monarch."
   - What's unclear: the precise list — only reproducible on the GPU host post-fix.
   - Recommendation: Wave 2 plan runs `uv run pytest tests/test_triton_*_strict.py -q` immediately after the deepcopy commit, captures the failure list, and dispositions each per the re-baseline rule. Plan this as a discovery step, not a fixed task list.

2. **Phases 1–3 test-before-fix ordering (D-09).**
   - What we know: Phases 4–6 followed D-37/D-50; earlier phases may not have.
   - What's unclear: whether Phases 1–3 have any test-after-fix orderings in `git log`.
   - Recommendation: Wave 3/4 audits `git log` per finding; document gaps in AUDIT-REPORT, do NOT rewrite history (D-09 explicit).

3. **`pyproject.toml` edited in both Wave 1 (mypy overrides / ruff per-file-ignores) and Wave 2 (marker).**
   - What we know: both waves need `pyproject.toml`.
   - Recommendation: Waves are sequential, so this is safe (never concurrent). If the planner wants extra safety, register the `divergence` marker in Wave 1's `pyproject.toml` edit even though it's applied in Wave 2 — one config file, one wave touching it.

## Sources

### Primary (HIGH confidence)
- `bd show` for all 14 issues (`5rk 7rj e0l in0 lqk mjy n20 q3k 4m6 6dz e7t fpl lht u00`) — authoritative measured root causes
- `.planning/phases/07-audit-report-findings-handling/07-CONTEXT.md` — locked decisions D-01..D-10
- `.planning/phases/04-quant-on-bit-identity/04-DISPOSITION.md` — per-cluster `h_scale_mult` table
- `.planning/phases/06-edge-case-sweeps/06-VERIFICATION.md`, `deferred-items.md` — Phase 6 closure, lint baseline
- `src/gru_qat/quantizers.py:73,105,245,255` — confirmed n20 shared-config bug + fix site
- `pyproject.toml` — current marker config, mypy/ruff config
- Live `uv run mypy` (145 errors / 10 files) + `uv run ruff check src tests` (23 errors) — measured this session
- `tests/test_triton_scan_strict.py` — strict-test parametrize structure (for marker placement)
- `.planning/PROJECT.md` — Phase 2 Option C locked Key Decision
- `.planning/REQUIREMENTS.md` — the 28-requirement v1 list

### Secondary (MEDIUM confidence)
- `.planning/phases/02-*/02-SUMMARY.md`, `05-VERIFICATION.md` — phase closure narratives

### Tertiary (LOW confidence)
- None — pytest marker semantics are CITED from pytest docs (stable, well-known API).

## Metadata

**Confidence breakdown:**
- Per-issue triage: HIGH — every bd description read directly; root causes are explicitly stated by the issue authors with reproducer cross-references.
- `divergence` marker mechanics: HIGH — standard pytest API; project already uses the `slow` marker the same way.
- n20 fix: HIGH — bug site confirmed in source; fix shape specified by D-07 and verified correct.
- mypy/ruff scope: HIGH — baselines measured live this session.
- AUDIT-REPORT structure: HIGH — D-08 fully specifies it; source artifacts confirmed to exist.
- Wave sequencing: MEDIUM — file-touch map is exact, but final granularity is the planner's call.

**Research date:** 2026-05-15
**Valid until:** 2026-06-14 (stable — closure phase; only the live mypy/ruff counts could drift if src changes, but Phase 7 is the only thing touching src)

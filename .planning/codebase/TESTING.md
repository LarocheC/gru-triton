# Testing Patterns

**Analysis Date:** 2026-05-13

## Test Framework

**Runner:**
- `pytest >= 7` (`pyproject.toml:19`).
- Config in `pyproject.toml:41-46`:
  ```toml
  [tool.pytest.ini_options]
  testpaths = ["tests"]
  addopts = "-ra"
  markers = [
      "slow: marks tests as slow (deselect with '-m \"not slow\"')",
  ]
  ```
- `-ra` is always on so skipped tests and short rationale show up in the summary line — important because the CUDA-skipped Triton suite is silent otherwise.

**Assertion library:**
- Plain `assert` plus `pytest.approx` for scalar floats (`tests/test_ste.py:59`) and `pytest.raises(... , match=...)` for error-path tests (`tests/test_quantizers.py:56`, `tests/test_parity.py:186`, `tests/test_structure.py:243`).
- Tensor equality uses `torch.equal` (strict) or `torch.allclose(..., atol=..., rtol=...)`. Drift-bounded parity tests compute `(a - b).abs().max().item()` and compare against an explicit tolerance.

**Parallel execution:**
- `pytest-xdist >= 3` is listed in dev extras (`pyproject.toml:20`) but no `pytest-xdist`-specific markers are used; run with `-n auto` if desired.

**Other dev tooling:**
- `mypy >= 1.7` and `ruff >= 0.1` are dev extras. `mypy` is `strict` and scoped to `src/gru_qat` only — tests are not type-checked.

**Run commands:**
```bash
pytest -q                                # full suite (~100 tests)
pytest tests/test_triton_monarch.py -q   # one file
pytest -k "monarch and qat" -q           # one test by name
pytest -m "not slow" -q                  # skip slow tests
pytest -m slow -q                        # only slow tests
mypy                                     # strict, src/gru_qat only
ruff check src tests                     # lint
```

## Test File Organization

**Location:**
- All tests live in `tests/` at repo root. No co-located tests.

**Naming:**
- Files: `test_<source_module>.py` for unit tests on a specific module (`test_ste.py`, `test_quantizers.py`).
- Files: `test_<phase_concept>.py` for cross-cutting / phase-exit tests (`test_parity.py`, `test_qat_smoke.py`, `test_calibration.py`, `test_structure.py`).
- Files: `test_triton_<variant>.py` or `test_<variant>_dispatch.py` for Triton kernel parity (`test_triton_scan.py`, `test_triton_diagonal.py`, `test_triton_monarch.py`, `test_butterfly_dispatch.py`).
- Functions: `test_<behaviour>` describing what is verified, not what is called. Example: `test_fused_gate_matches_split`, `test_calibrate_then_freeze_locks_scales`, `test_diagonal_dispatch_grad_matches_per_step`.
- Class grouping (`TestSTERound`, `TestSTEClamp`, `TestFakeQuant`) is used only in `tests/test_ste.py`. Everywhere else tests are module-level functions.

## Test File Inventory

One bullet per file, with what it covers and approximate count:

- **`tests/test_ste.py`** (7 tests, no CUDA, fast): `STERound.forward` rounds half-to-even and `.backward` is identity; `STEClamp.forward` clamps and `.backward` zeros grad outside the range; `fake_quant_ste` round-trip on grid, clipping past range, gradient passthrough. Organized as `TestSTERound`, `TestSTEClamp`, `TestFakeQuant` classes.

- **`tests/test_quantizers.py`** (8 tests + 1 `@pytest.mark.skip`, no CUDA, fast): `Identity` is exactly passthrough; per-tensor round-trip stays within one step; per-channel produces independent scales per slice along `axis`; per-group requires `N % G == 0` and raises `ValueError`; `make_quantizer` dispatches to the right subclass for each `QuantizerConfig` shape; `freeze()` locks the scale across forwards. One test is `@pytest.mark.skip(reason="phase=2 — requires simulator import")` — placeholder for a deferred simulator-parity test.

- **`tests/test_parity.py`** (9 tests, no CUDA, fast): Phase 2 exit. `GRUCellQuant` with Identity quantizers matches `torch.nn.GRUCell` to `< 1e-5` on parametrized `(input_size, hidden_size, batch)` shapes plus edge cases (`h=0`, `x=0`, large magnitudes). Also: fused gate layout matches split (`< 1e-5`); pre-batched input matches per-step under fp32 Identity (`< 1e-5`); int8 pre-batch produces finite output (looser); `pre_batch_input=True` without `gate_layout="fused"` raises; per-tensor weight quant with fused gates raises.

- **`tests/test_qat_smoke.py`** (4 tests, no CUDA + 1 marked `@pytest.mark.slow`): Phase 3+4 exit. Identity-quant matches fp32; int8-per-channel produces finite, bounded outputs on random untrained weights (structural smoke, not accuracy); swapping per-channel ↔ per-group(8) with no other code change produces non-identical but finite outputs. **Slow test** `test_layer_trains_to_baseline` trains an int8 QAT student to within 2× of fp32 baseline loss on a synthetic next-step prediction task (200 Adam steps).

- **`tests/test_calibration.py`** (5 tests, no CUDA, fast): `calibrate()` populates `running_min` / `running_max` on activation quantizers via synthetic loader; `calibrate()` + `freeze_all()` locks scales across subsequent forwards even on huge-magnitude input; `calibrate()` handles tuple loader yielding `(x, h0)`; `only_activations=True` skips weight quantizers; `n_batches` truncates a longer loader.

- **`tests/test_structure.py`** (8 tests, `pytest.importorskip("torch_structured")`): Tier-1 structured-mode coverage parametrized over `KINDS = ["monarch", "circulant", "butterfly", "ldr"]`. For each kind: forward output finite and right shape; gradients flow through every learnable parameter; short training loop reduces loss vs. init; int8-QAT in frozen mode produces finite forward + backward. Plus error-path tests: `pre_batch_input` with structured mode raises; `quantize_weights()` on structured cells raises; shape validators reject bad sizes (non-divisible nblocks, non-square circulant/ldr).

- **`tests/test_triton_scan.py`** (6 tests, `cuda_only` + `pytest.importorskip("triton")`): Dense Triton kernel parity. Persistent forward kernel matches the autotune-default forward within TF32 noise (rel < 5e-2); persistent backward matches the PyTorch reference backward on `(dgi, dh0, dWh, dbh)` (rel < 1e-1); Triton forward matches `GRULayer` reference (abs < 5e-3); Triton autograd matches PyTorch autograd on `(x, h0, Wh_cat, bh_cat)` gradients (rel < 1e-1, with explicit regression note about per-program slab zeroing); QAT-frozen-hidden persistent and non-persistent kernels match the PyTorch reference for forward + gradients.

- **`tests/test_triton_diagonal.py`** (9 tests, `cuda_only`, last 4 also require GRULayer dispatch). Four-stage structure documented in module docstring: **Stage A** PyTorch reference matches cell bit-for-bit (rel < 1e-5 — fp32 Identity, tight); **Stage B** Triton forward matches PyTorch reference (fp32 rel < 1e-4; QAT rel < 1e-3 — tight because diagonal has no matmul); **Stage C** Triton backward matches PyTorch on `(dgi, dh0, dWh_diag, dbh)` (fp32 rel < 1e-3; QAT rel < 1e-2 to absorb STE boundary flips); **Stage D** end-to-end `GRULayer.use_triton=True` dispatch matches the per-step path on forward, gradients exist on every parameter, gradient matches per-step autograd, and the QAT calibrate→freeze→forward flow produces finite output.

- **`tests/test_triton_monarch.py`** (10 tests, `cuda_only` + `pytest.importorskip("torch_structured")`; 2 of them also run without CUDA for the PyTorch reference parity). Tiered like the diagonal file. PyTorch monarch reference matches the cell's structured forward (rel < 1e-5) and backward (rel < 1e-4 on `dh0`, `dWh`, `dbh`). Triton forward matches PyTorch reference within TF32 noise (rel < 5e-3 fp32, < 1e-1 QAT); Triton backward matches PyTorch on `(dgi, dh0, dWh_struct, dbh)` (rel < 5e-2 fp32, < 1e-1 QAT). End-to-end `GRULayer(use_triton=True)` matches `use_triton=False` (rel < 5e-2); calibrate→freeze→forward via Triton produces finite output; eligibility errors raise on incompatible config (non-monarch hidden, split gate layout).

- **`tests/test_butterfly_dispatch.py`** (11 tests, `cuda_only` + `pytest.importorskip("torch_structured")`): Forward parity per-step vs. Triton (rel < 1e-1); end-to-end QAT calibrate→freeze→forward through the Triton butterfly path produces finite output; backward populates grads on every param. Multi-step Triton-forward parity with the CUDA-op per-step path at multiple `(T, B, H)`; **regression test** `test_butterfly_triton_forward_scratch_oob_regression` at `(T=16, B=32, H=512)` documents and guards the scratch-OOB bug where absolute-offset indexing into a per-program scratch slab corrupted neighboring tensor storages for `pid_b > 0`. Triton backward matches autograd through `gru_scan_butterfly` on `(dgi, dh0, dtwiddles, dbh)` (rel < 5e-2). Plus QAT-forward parity, full train step grad flow, and a direct `gru_scan_butterfly` vs. `GRULayer` equivalence check.

## Markers In Use

| Marker | Where defined | What it does |
|---|---|---|
| `slow` | `pyproject.toml:44-45` registered marker | Marks long-running training-loop tests. Currently only `tests/test_qat_smoke.py:88` (`test_layer_trains_to_baseline`, 200 Adam steps). Run with `pytest -m slow`, skip with `pytest -m "not slow"`. |
| `cuda_only` | Per-file local: `cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="...")` | Applied as `@cuda_only` decorator on every Triton or GPU-required test. Defined inline (not registered in `pyproject.toml`) in `tests/test_triton_scan.py:25`, `test_triton_diagonal.py:30`, `test_triton_monarch.py:30`, `test_butterfly_dispatch.py:31`. |
| `pytest.mark.parametrize` | First-class pytest | Parametrize `(T, B, IN, H)`, `(T, B, H, nblocks)`, `(input_size, hidden_size, batch)`, or `kind`. See `tests/test_triton_scan.py:48`, `tests/test_structure.py:76`. |
| `pytest.mark.skip` | First-class pytest | Used once as a placeholder for a deferred test: `tests/test_quantizers.py:90` (simulator parity, awaiting simulator import). |

## How CUDA / Triton Availability Is Gated

Two layers of skip per test file:

**Module-level (file-wide skip when the dep is missing):**
```python
triton = pytest.importorskip("triton")
```
At module top — `tests/test_triton_scan.py:12`. The whole file is skipped if `triton` is not importable, before any test collection happens for that file. Same pattern for `torch_structured` in `tests/test_structure.py:25`, `tests/test_triton_monarch.py:17`, `tests/test_butterfly_dispatch.py:19`.

**Per-test (skip when CUDA is missing on a machine that does have Triton/torch-structured):**
```python
cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernel requires CUDA"
)

@cuda_only
def test_something_on_gpu(...): ...
```
Defined per-file (see "Markers In Use" above). Some files mix: the PyTorch reference tests in `tests/test_triton_diagonal.py:75` and `tests/test_triton_monarch.py:81` run on CPU (no `@cuda_only`), while the kernel tests in the same file are `@cuda_only`. This lets the reference path get exercised on CPU-only CI.

**The combined effect:** `pytest -q` always passes on a CPU-only machine — Triton tests are file-skipped and `cuda_only`-skipped cleanly with reasons surfaced by `-ra`.

## Parity Tests vs. Numerical-Tolerance Tests

The suite distinguishes three regimes:

**Bit-identical / very-tight (< 1e-5 absolute or relative):**
- fp32 Identity-quantizer cell vs. `torch.nn.GRUCell` (`tests/test_parity.py:70`).
- Fused gate vs. split gate, fp32 (`tests/test_parity.py:135`).
- Pre-batched input vs. per-step, fp32 Identity (`tests/test_parity.py:162`).
- PyTorch monarch / diagonal reference vs. the cell's structured forward (`tests/test_triton_monarch.py:101`, `tests/test_triton_diagonal.py:93`).
- These are *algebraic* equalities — different ops but the same math. Loosening them indicates a math bug.

**TF32-bounded numerical (5e-3 to 5e-2 relative):**
- Triton kernel vs. PyTorch reference, fp32 — TF32 mantissa is ~10 bits and noise compounds across `T` matmuls × 3 gates × nonlinearities (`tests/test_triton_scan.py:75`, `tests/test_triton_monarch.py:127`, `tests/test_butterfly_dispatch.py:160`).
- Persistent vs. autotune kernel variants (`tests/test_triton_scan.py:75`).
- Tolerance comments document the reasoning (`tests/test_triton_scan.py:74,137,150`).

**STE / QAT-bounded (1e-2 to 1e-1 relative):**
- In-kernel fake-quant forward and backward — STE boundary rounding can flip mask bits, on top of TF32 noise (`tests/test_triton_diagonal.py:238`, `tests/test_triton_monarch.py:162`, `tests/test_triton_scan.py:280`).
- Gradients have the loosest bounds because per-step rounding noise compounds through the backward pass.

**Hard rule:** the cell-vs-`nn.GRUCell` fp32 parity gate at `< 1e-5` (`tests/test_parity.py:70`) must not be loosened. Other tolerances may be tightened as kernels improve, but `1e-5` is the algebraic-correctness gate for the cell math.

## Bench-Style Smoke Tests vs. Correctness Tests

**Correctness tests** (the vast majority):
- Compare two paths analytically, with explicit tolerances.
- No timing, no GPU warmup.
- Each test is independent — seed is reset with `torch.manual_seed(0)` at the top of any test using randomness.

**Smoke tests** (a handful):
- Assert finite, bounded output without comparing to a reference.
- Used when the comparison would be too noisy to be meaningful (e.g., `test_int8_per_channel_finite_and_bounded` in `tests/test_qat_smoke.py:31` — random untrained weights with 9 quantizers compounded). Module docstring notes "*this is not an accuracy test — it's a structural smoke test*".
- End-to-end calibrate→freeze→forward flows assert finite output and correct shape (`tests/test_butterfly_dispatch.py:79`, `tests/test_triton_monarch.py:292`, `tests/test_triton_diagonal.py:343`).

**Training-loop tests** (one, marked `slow`):
- `tests/test_qat_smoke.py:89` `test_layer_trains_to_baseline`: trains both an fp32 baseline and an int8 QAT student to within `2 ×` of each other on a synthetic teacher-student task. 200 Adam steps × 2 students. Marked `@pytest.mark.slow` so the default suite stays fast.
- Lighter training-loop checks (40 steps, loss must strictly decrease) live inline in `tests/test_structure.py:109` parametrized over the 4 structured kinds — not marked slow because each iteration is cheap on the small `(T=8, B=4, H=32)` shapes used.

**Benches** live in `bench/` and are *not* run by pytest:
- `bench/bench_layer.py` — dense train-step bench (cudnn / compile / Triton).
- `bench/bench_triton_fwd.py` — forward-only bench across variants.
- `bench/bench_triton_train.py` — train-step bench across variants.
- Invocation: `uv run python bench/bench_layer.py [--shapes seq,batch,hidden]`.
- Benches require CUDA (`raise SystemExit("CUDA not available")` at `bench/bench_layer.py:225`).
- The numbers in `DEVELOPMENT.md` (e.g., "Diagonal persistent ~1.1 ms, ~4× faster than cuDNN at T=64, B=32, H=512, RTX 2000 Ada") come from these benches; treat them as the regression target when changing kernels.

## Common Patterns to Match When Adding Tests

**TF32 setup for kernel parity:**
```python
torch.manual_seed(0)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")
```
Both the reference and the test path must share this regime so the only remaining drift is kernel logic, not arithmetic mode. See `tests/test_triton_scan.py:55-57`, `tests/test_triton_diagonal.py:107-109`, etc.

**Triton reference-builder helpers:**
- Each Triton test file defines a local `_make_<kind>_layer(...)` helper that builds a fp32-Identity GRULayer with the right structured config (`tests/test_triton_monarch.py:35`, `tests/test_triton_diagonal.py:35`, `tests/test_butterfly_dispatch.py:36`).
- A `_build_gi_from_cell(layer, x)` helper reproduces the cell's input projection so both the reference and the kernel see the same `gi` tensor (`tests/test_triton_monarch.py:54`, `tests/test_triton_diagonal.py:51`).

**Relative-error reporting:**
```python
max_diff = (ref - tri).abs().max().item()
rel = max_diff / max(ref.abs().max().item(), 1e-6)
assert rel < TOL, f"<name> rel diff {rel:.4e}"
```
This `1e-6` floor prevents division by near-zero in fp32 path-comparisons. Match this idiom — see `tests/test_triton_diagonal.py:120-121`, `tests/test_triton_monarch.py:99-101`, `tests/test_butterfly_dispatch.py:69-71`.

**Per-batch error inspection** for grid/parallelism bugs:
```python
rel_per_b = (
    (tri_out - ref_out).abs().amax(dim=(0, 2))
    / ref_out.abs().amax().clamp(min=1e-6)
)
assert rel_per_b.max().item() < 5e-2, (
    f"... worst batch={rel_per_b.argmax().item()}, rel by batch={rel_per_b.tolist()}"
)
```
The butterfly OOB regression test (`tests/test_butterfly_dispatch.py:206`) uses this pattern to localize regressions to a specific `pid_b`.

**Loader fixtures:**
- Calibration tests use generator-based synthetic loaders defined inline:
  ```python
  def _synthetic_loader(n, T, B, in_size):
      for _ in range(n):
          yield torch.randn(T, B, in_size) * 0.5
  ```
  See `tests/test_calibration.py:30`. Use `* 0.5` (or `* 0.1` for CUDA-frozen tests) to keep values in-range for 8-bit symmetric quant scales.

**No fixtures via conftest:**
- The repo has no `conftest.py`. Helpers are defined as module-level `_underscore_prefixed` functions inside each test file. Keep tests self-contained.

## Mocking

**Default: none.** No `unittest.mock`, no `pytest-mock`. The library has zero external services and tests exercise real PyTorch / real Triton kernels (when CUDA is available). The only "mock-like" pattern in the production code is the `Identity` quantizer subclass, which is a no-op `FakeQuantize` used to dial quantization off in parity tests — that's a production class, not test infrastructure.

**Narrow exception (Phase 3, Plan 03-03):** `pytest.MonkeyPatch` is permitted for **optional-dependency failure-mode tests** — i.e. asserting that a missing external package (currently `torch-structured`) raises a clear `ImportError` with an install hint. Two idioms are blessed:

1. `monkeypatch.setattr("gru_qat.structure._import_torch_structured", _raise_missing_torch_structured)` — when the production code routes through an internal lazy-import helper. Preferred when available.
2. `monkeypatch.setitem(sys.modules, "<pkg>", None)` — when the production code does `from <pkg> import ...` directly, bypassing the helper (the LDR branch at `src/gru_qat/structure.py:160-172` is the current example). Setting `sys.modules[name] = None` is Python's documented "this module is known to be absent" marker; subsequent imports raise `ImportError`.

See `tests/test_structure_parity.py` STR-03 section (`test_missing_torch_structured_raises_clear_error`, `test_missing_ldr_raises_clear_error`, `test_local_impls_work_without_torch_structured`) for the canonical patterns.

**Convention going forward:** every new optional dependency should grow a matching failure-mode test in this style. The rule of thumb is "if the dep is optional, prove it in a test." Do **not** broaden this exception to normal logic tests — real layers, real tensors, real kernels remain the default. If a logic test reaches for `monkeypatch`, that's a signal the test is the wrong shape.

## Fixtures and Test Data

**Test data:** all synthetic. Generated per-test with `torch.randn` + a deterministic `torch.manual_seed(0)`. Scaled (`* 0.1` for QAT-frozen tests, `* 0.5` for fp32 parity tests) to stay in-range for 8-bit quantizers.

**No fixture directory.** No on-disk fixtures, no JSON/YAML inputs, no recorded tensors.

**Fixtures via pytest:** none. No `@pytest.fixture` decorators in the suite.

## Coverage

**Requirements:** None enforced. No `pytest-cov` in dev extras, no coverage config in `pyproject.toml`. Phase-exit tests in `DEVELOPMENT.md` are the de facto coverage gate: each completed phase has a paired set of tests under `tests/` and "CI green ⇒ phase landed" is the working agreement.

## Test Types

**Unit tests:**
- Per-primitive tests for `STERound`, `STEClamp`, `fake_quant_ste`, each `FakeQuantize` subclass, `make_quantizer` dispatch. CPU-only, no CUDA dependency.

**Integration tests:**
- `GRUCellQuant` end-to-end with various recipes (`tests/test_qat_smoke.py`, `tests/test_parity.py`).
- `GRULayer` calibrate→freeze→forward flows (`tests/test_calibration.py`).
- Structured-mode cell × 4 kinds (`tests/test_structure.py`).

**Kernel parity tests:**
- Triton kernel vs. PyTorch reference for forward, backward, fp32, and QAT — one file per kernel variant.
- End-to-end `GRULayer.use_triton=True` vs. `use_triton=False` dispatch parity.

**Regression tests:**
- Explicitly documented in the test docstring with the bug they guard (e.g., `test_butterfly_triton_forward_scratch_oob_regression` in `tests/test_butterfly_dispatch.py:164` documents the per-program scratch-OOB; `tests/test_triton_scan.py:202` regresses the per-program slab-zeroing bug in the autotuned backward).

**E2E tests:** None beyond the calibrate→freeze→inference dispatch flows. There's no streaming-inference or export pipeline test (export is a deferred Phase 6 concern — see `gru_cell.py:507` TODO and `DEVELOPMENT.md` "Open questions" section).

---

*Testing analysis: 2026-05-13*

# DEVELOPMENT.md — File map, phase status, upgrade pathways

This document tracks the implementation. Read [`SCOPE.md`](./SCOPE.md)
first for design rationale.

## Working agreement

- Environment: `uv` (see `pyproject.toml`). `uv sync` to bootstrap.
  Optional dev extras: `uv pip install -e ".[dev]"`.
- `torch-structured` (used by `structure.py` for Monarch / Butterfly /
  Circulant / LDR) is an optional dependency. Install from source:
  `uv pip install git+https://github.com/LarocheC/torch-structured`.
  The lazy import in `structure.py` means dense-only users don't need it.
- Strict dtype discipline: every fake-quant op preserves input dtype;
  every internal float op runs in `torch.float32` unless the caller
  has explicitly opted into fp16/bf16 (autocast). bf16 around fake-quant
  was tried and dropped — the fp32↔bf16 cast tax around quantize/dequantize
  boundaries exceeded the GEMM saving at our shapes.
- Each phase has tests under `tests/`. CI green ⇒ phase landed.

## File map

```
src/gru_qat/
  __init__.py             public API surface
  ste.py                  STE autograd functions (STERound, STEClamp, fake_quant_ste)
  quantizers.py           FakeQuantize base + per-tensor / per-channel / per-group
  calibration.py          calibrate(module, loader, n_batches), freeze_all
  structure.py            StructureConfig + make_structured_linear
                          (Diagonal, Monarch, Circulant, Butterfly, LDR)
  gru_cell.py             GRUCellQuant — single step. Optionally structured
                          (structure_input / structure_hidden).
  gru_layer.py            GRULayer — multi-step. Triton dispatch via use_triton.
  triton_kernels/
    __init__.py           Phase 5 design notes
    scan.py               Dense persistent fwd+bwd kernels (Monarch tier 2 sibling).
                          Autotune over BLOCK_B/OH/K. In-kernel fake-quant for QAT.
    scan_diagonal.py      Diagonal persistent fwd+bwd. Elementwise w_h*h per gate;
                          no matmul, no cross-CTA barrier. In-kernel fake-quant.
    scan_monarch.py       Monarch persistent fwd+bwd. nblocks block-diagonal matmuls
                          per timestep. In-kernel fake-quant for QAT.
    scan_butterfly.py     Butterfly persistent fwd+bwd. log_H stages of strided
                          2x2 mixing. In-kernel fake-quant for QAT.

tests/
  test_ste.py                    STE primitives
  test_quantizers.py             FakeQuantize variants
  test_parity.py                 Cell parity vs torch.nn.GRUCell at fp32
  test_qat_smoke.py              End-to-end QAT trains on toy task
  test_calibration.py            calibrate() round-trip
  test_structure.py              Structured cells (all 4 kinds): forward,
                                 gradient flow, training, int8 QAT
  test_triton_scan.py            Dense Triton fwd+bwd + persistent + QAT
  test_triton_diagonal.py        Diagonal Triton fwd+bwd + QAT + GRULayer dispatch
  test_triton_monarch.py         Monarch Triton fwd+bwd + QAT + GRULayer dispatch
  test_butterfly_dispatch.py     Butterfly Triton fwd+bwd + QAT + GRULayer dispatch

bench/
  bench_layer.py                 Dense train-step bench (cudnn / compile / Triton)
  bench_triton_fwd.py            Forward-only bench across variants
  bench_triton_train.py          Train-step bench across variants
```

## Phase status

All originally-planned phases (0–5) are complete, plus a structured-
matrix track and a calibration plumbing pass.

### Phase 0 — bootstrap ✓

- `uv sync` works; `pytest --collect-only` succeeds.

### Phase 1 — STE and quantizer primitives ✓

- `ste.py`, `quantizers.py`.
- Per-tensor sym/asym, per-channel sym (any axis), per-group sym (group
  size along axis). Bits ∈ {2, 3, 4, 8} supported.
- Observer modes: `dynamic`, `min_max`, `frozen`.
- Tests: `test_ste.py`, `test_quantizers.py`.

**Known gap (carried forward)**: the `min_max` observer's
`_update_observer` uses a global scalar reduction even when `axis` is
set, so per-channel observers don't accumulate per-channel running
stats. Per-channel activation quant with min_max isn't used in the
fast paths (which are per-tensor + frozen post-calibration), so this
hasn't blocked anything; fix is a one-method change in
`quantizers.py` whenever a per-channel activation quant scheme is
needed.

### Phase 2 — fp32 cell parity ✓

- `GRUCellQuant.forward()` with Identity quantizers matches
  `torch.nn.GRUCell` within `< 1e-5` on the parametrized shapes plus
  edge cases (h=0, x=0, large magnitudes).
- Tests: `test_parity.py`.

### Phase 3 — fake-quant insertion ✓

- All 6 weight + 3 activation insertion points wired in `gru_cell.py`.
- Optional gate-preact quantizers (Identity by default; configurable
  via `recipe.gate_act`).
- Bias / sigmoid / tanh stay fp32 — see SCOPE non-goals.
- Tests: `test_qat_smoke.py`.

### Phase 4 — multi-step layer + calibration ✓

- `GRULayer` wraps cell, loops over time. Multiple per-step bodies:
  - Dense: `cell.step_with_gi(gi_t, h, w)` if `pre_batch_input=True`,
    else `cell.step(x_t, h, w)`.
  - Structured: `cell.step_structured(x_t, h)`.
  - Triton fast path: `_forward_fast_dispatch` when `use_triton=True`.
- `calibrate(module, loader, n_batches)` switches activation
  quantizers to `min_max`, runs forwards, returns stats summary.
  `freeze_all(module)` locks scales for inference.
- `GRULayer.calibrate(loader, n_batches)` is a thin wrapper that
  temporarily disables `use_triton` so the per-step path runs and
  observers actually update.
- Tests: `test_qat_smoke.py`, `test_calibration.py`.

### Phase 5 — Triton kernels ✓

Multiple persistent kernels, all multi-step (one launch per fwd/bwd
half across all T timesteps):

| kernel | layout | parallelism | notes |
|---|---|---|---|
| `scan.py` (dense) | autotune + persistent | grid `(B_tile, OH_tile)` with spin-wait barrier | Both autotune (1D grid, no inter-CTA) and persistent (2D grid + barrier) variants. QAT support. |
| `scan_diagonal.py` | persistent, no barrier | grid `(B_tile, H_tile)`, no cross-CTA sync | Elementwise hidden recurrence (no matmul) so each program owns its slab; `h` carries in registers. Smallest params, fastest variant. QAT support. |
| `scan_monarch.py` | persistent | grid `(B_tile, block)` | One small `[blksz, blksz]` matmul per (block, gate). Best speed at typical training shapes. QAT support. |
| `scan_butterfly.py` | persistent | grid `(B_tile,)` | log_H stages of strided 2×2 mixing per gate. No tensor-core utilization. QAT support. |

**Cross-CTA barriers** use the release/acquire atomic_add pattern:
```python
tl.atomic_add(barrier_ptr + t, 1, sem="release")
done = tl.atomic_add(barrier_ptr + t, 0, sem="acquire")
while done < NUM_PROGRAMS:
    done = tl.atomic_add(barrier_ptr + t, 0, sem="acquire")
```
The earlier "relaxed atomic_add + `tl.load(cache_modifier='.cv')`"
pattern looked plausible but didn't provide the acquire fence needed
for cross-CTA data visibility — caused non-deterministic ~0.2 absolute
drift on `gru_scan_persistent` outputs. Fixed in the most recent
commits.

- Tests: `test_triton_scan.py`, `test_triton_diagonal.py`,
  `test_triton_monarch.py`, `test_butterfly_dispatch.py`.

### Phase 5+ — structured-matrix hidden weights ✓

Extension to the phase-5 work, not part of the original plan:

- `structure.py`: `StructureConfig(kind, nblocks, ...)` and
  `make_structured_linear` factory. Wraps `torch-structured`'s
  primitives plus thin local Diagonal and Circulant layers.
- Cell: optional `structure_input` / `structure_hidden`. Structured
  mode runs `step_structured` per timestep with the structured linear
  modules in place of dense weights. Per-gate output-side fake-quant
  replaces per-row weight quant.
- Triton kernels for Diagonal, Monarch and Butterfly (Circulant / LDR
  fall back to the per-step PyTorch path).
- Tests: `test_structure.py`, `test_triton_diagonal.py`,
  `test_triton_monarch.py`, `test_butterfly_dispatch.py`.

### Phase 6 — int activations and LUT nonlinearities

Not started. Out of scope for QAT; needed for embedded deployment.

## Train-step bench at `(T=64, B=32, H=512)`, fp32, RTX 2000 Ada

| variant | train ms | vs cuDNN |
|---|---|---|
| cuDNN `nn.GRU` | 4.4 | 1.0× |
| `GRULayer` dense + compile_step | 38.7 | 8.8× |
| dense Triton persistent | 8.8 | 1.9× |
| Monarch persistent, nblocks=4 | 5.8 | 1.3× |
| Monarch persistent, nblocks=8 | 2.0 | **0.45× (2.2× faster than cuDNN)** |
| Butterfly persistent | 20.3 | 4.6× |
| Butterfly per-step CUDA op (pre-Triton) | 107 | 24× |
| Diagonal persistent | ~1.1 | **~0.25× (~4× faster than cuDNN)** |

For QAT (frozen int8 hidden) add ~10–30% to the Triton numbers.

Per-gate hidden parameter counts:

| kind | params per gate at H=512 |
|---|---|
| dense | 262K |
| Monarch nblocks=4 | 65K |
| Monarch nblocks=8 | 32K |
| Butterfly | 4.6K |
| Diagonal | 512 |

## Upgrade pathways

### Adding a new quantization scheme

1. Subclass `FakeQuantize` in `quantizers.py`. Override
   `_compute_scale_zp`.
2. Add a factory entry in `quantizers.make_quantizer`.
3. Pass the config into `GRUCellQuant(recipe=...)`.

No changes to `gru_cell.py` or `gru_layer.py`.

### Adding a new structured kind

1. Add the kind to `StructuredKind` literal in `structure.py`.
2. Add a branch in `make_structured_linear` constructing the underlying
   `nn.Module`. Validate shape constraints in `_validate_shapes`.
3. The PyTorch cell path picks it up automatically. For Triton speed,
   either write a new persistent kernel (`scan_<kind>.py`) following
   the Monarch / Butterfly templates, or leave it as PyTorch-only.

### Switching from STE-round to a different gradient estimator

Edit `ste.py`. `STERound.apply` and `STEClamp.apply` are the swap
points. Quantizers that need a learnable step size (LSQ) would slot
in here.

### Targeting a different hardware backend (CUTLASS, IREE, embedded)

The Triton kernels are the reference for the integer math. Port the
inner loop. `GRULayer` doesn't change; the Python dispatch in
`_forward_fast_dispatch` adds a new branch.

### Adding LSTM later

Mostly copy-paste of `gru_cell.py` with four gates instead of three
and a cell state quantizer. Quantizer / structure infrastructure
unchanged.

## What the agent should NOT do

- Do not rewrite the existing simulator's `quant_primitives.py`. Import
  from it; match its conventions.
- Do not optimize the PyTorch reference path. Its job is to be slow,
  obvious, and correct. Speed lives in the Triton kernels.
- Do not add quantization to bias, sigmoid, or tanh in the reference
  path without an explicit ticket. These are deliberate omissions.
- Do not collapse `FakeQuantize` granularities into a single class
  with `if/else` branches. Subclassing keeps the dispatch flat and
  the kernel variants tractable.
- Do not use `tl.load(cache_modifier=".cv")` as a substitute for an
  acquire fence in cross-CTA barriers. Use
  `tl.atomic_add(barrier, 0, sem="acquire")` for the spin-wait read.
  See the comment in `scan.py:gru_scan_fwd_persistent_kernel`.

## Open questions / known limitations

1. **Per-channel min_max observer**: see Phase 1 known gap. Not blocking
   any current path; per-channel weight quant uses `dynamic` mode where
   scales are derived from static weights each forward.
2. **bf16 autocast around fake-quant**: tried and dropped. The fp32↔bf16
   cast tax around quant boundaries exceeds the GEMM saving at our
   shapes. Documented in `gru_layer.py`.
3. **LSQ / PACT activation scales**: `learnable_scale` flag is plumbed
   in `QuantizerConfig` but not implemented. Activation scales use
   min_max + freeze after calibration.
4. **Bias quantization**: bias-fp32 throughout. Bias-int32 (export to
   `weight_scale × act_scale`) would be a phase-6 export concern.
5. **Streaming inference**: full-sequence forward only. Streaming
   (`step(x_t, h)` called by user) works mechanically through the cell
   but bypasses the Triton kernels which require knowing T at launch.

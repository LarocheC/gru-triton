# gru-qat

Pluggable QAT and quantized inference for GRU, with structured-matrix
hidden weights and a multi-step persistent Triton kernel.

- **Why**: cuDNN's GRU is a closed kernel; we cannot insert fake-quant.
  To do QAT with arbitrary quantization granularities (per-channel,
  per-group, fine-grained int4) we own the cell.
- **What**: a manually-unrolled GRU cell where every quantizable
  quantity is a `FakeQuantize` module that can be swapped without
  touching the cell code. Reference path is pure PyTorch; accelerated
  path is Triton.
- **Plus**: hidden weights can be parameterized as Diagonal (one
  vector per gate), Monarch (block-diagonal), Butterfly (`O(H log H)`
  twiddle), Circulant, or LDR (low-displacement rank) structured
  matrices, with matching Triton kernels for Diagonal, Monarch and
  Butterfly.

## Read first

1. [`SCOPE.md`](./SCOPE.md) — what's in, what's out, key design
   decisions.
2. [`DEVELOPMENT.md`](./DEVELOPMENT.md) — file map, phase status,
   bench numbers, upgrade pathways.

## Quick start

```bash
uv sync
uv pip install -e ".[dev]"   # optional: tests, mypy, ruff
pytest -q
```

For structured-matrix support, also install
[`torch-structured`](https://github.com/LarocheC/torch-structured):

```bash
uv pip install git+https://github.com/LarocheC/torch-structured
```

### Dense QAT layer

```python
import torch
from gru_qat import GRULayer, PRESETS

layer = GRULayer(
    input_size=512, hidden_size=512,
    recipe=PRESETS["int8_per_channel"],
    gate_layout="fused",
    pre_batch_input=True,        # one big GEMM for x @ W_i across T
    compile_step=True,           # torch.compile fuses the elementwise body
).cuda()
out, h_T = layer(x, h0)
```

### Triton-accelerated dense (persistent kernel)

```python
from gru_qat.triton_kernels.scan import gru_scan_persistent

# Inside training loop:
w = layer.cell.quantize_weights()
gi = layer.cell.input_projection(x, w)
out = gru_scan_persistent(gi, h0, w.Wh_cat, w.bh_cat)
```

### Structured hidden weights with Triton (Monarch — fastest)

```python
from gru_qat import GRULayer, QuantRecipe, QuantizerConfig, StructureConfig

layer = GRULayer(
    input_size=512, hidden_size=512,
    recipe=QuantRecipe(
        weight=QuantizerConfig(bits=32, axis=0, name="W_id"),
        input_act=QuantizerConfig(bits=32, name="x_id"),
        hidden=QuantizerConfig(bits=8, name="h_q"),       # int8 hidden quant
    ),
    gate_layout="fused",
    structure_hidden=StructureConfig(kind="monarch", nblocks=8),
    use_triton="auto",   # routes through the persistent monarch kernel
).cuda()

# QAT flow:
for x in train_loader:
    out, _ = layer(x)
    loss = ...
    loss.backward()

layer.calibrate(val_loader, n_batches=64)
layer.freeze()
out, _ = layer(x)        # now runs through the Triton kernel with frozen scales
```

### Structured hidden weights — Butterfly (smallest params)

```python
from gru_qat import GRULayer, StructureConfig

layer = GRULayer(
    H, H, recipe=...,
    gate_layout="fused",
    structure_hidden=StructureConfig(kind="butterfly"),
    use_triton="auto",
).cuda()
# Same calibrate -> freeze -> forward flow.
```

### Structured hidden weights — Diagonal (smallest & fastest)

```python
from gru_qat import GRULayer, StructureConfig

layer = GRULayer(
    H, H, recipe=...,
    gate_layout="fused",
    structure_hidden=StructureConfig(kind="diagonal"),
    use_triton="auto",
).cuda()
# Same calibrate -> freeze -> forward flow.
```

`kind="diagonal"` collapses each `H*H` hidden matrix to a length-`H`
vector. Per-step recurrence becomes elementwise `w_h * h` instead of a
matmul — `3H` weight scalars total on the hidden side, `O(H)` FLOPs per
step. The persistent Triton kernel has no matmul on the hidden side,
no cross-program reduction, and runs fully in registers across the
T-loop. Good fit when you want a *very* small recurrence (e.g. for an
embedded model) and are happy treating the hidden update as
hidden-unit-independent (similar in spirit to IndRNN / diagonal SSMs).

## Status

All originally-planned phases (0–5) complete. The dense and structured
paths are feature-complete:

| feature | status |
|---|---|
| STE primitives + FakeQuantize variants | ✓ |
| Dense `GRUCellQuant` parity vs `nn.GRUCell` (`< 1e-5`) | ✓ |
| Fake-quant insertion in cell (all 6 weight + 3 activation points) | ✓ |
| `GRULayer` with calibration → freeze flow | ✓ |
| Triton multi-step persistent kernel (dense, fp32, fp32 + frozen int8 QAT) | ✓ |
| Structured hidden weights (Diagonal / Monarch / Butterfly / Circulant / LDR) | ✓ |
| Triton persistent kernel for Diagonal (fp32 + QAT) | ✓ |
| Triton persistent kernel for Monarch (fp32 + QAT) | ✓ |
| Triton persistent kernel for Butterfly (fp32 + QAT) | ✓ |

117 tests pass, 1 skipped (the simulator-parity placeholder that's
deferred until the simulator is on `PYTHONPATH`).

## Train-step speed at `(T=64, B=32, H=512)` — fp32

| variant | ms/iter | vs cuDNN |
|---|---|---|
| cuDNN `nn.GRU` (dense, no quant) | 4.4 | 1.0× |
| `GRULayer` dense + `torch.compile` | 38.7 | 8.8× |
| dense Triton persistent | 8.8 | 1.9× |
| **Monarch persistent (nblocks=4)** | **5.8** | **1.3×** |
| **Monarch persistent (nblocks=8)** | **2.0** | **0.45× (2.2× faster)** |
| Butterfly persistent | 20.3 | 4.6× |
| **Diagonal persistent** | **~1.1** | **~0.25× (4× faster)** |

For QAT (frozen int8 hidden), expect ~10–30% overhead on top of the
fp32 number depending on path.

## Numerical parity vs PyTorch reference at `(T=64, B=32, H=512)`

Measured at bench shape against the `GRULayer(use_triton=False)`
PyTorch reference path (which itself matches `torch.nn.GRUCell` to
`< 1e-5`). Both sides use `torch.set_float32_matmul_precision("high")`
so the Triton kernels and PyTorch's matmul see the same TF32 inputs.
"fwd"/"dx"/"dh0" are max relative diffs; "weight-grad" is the worst
per-parameter `dWh` / twiddle / `b_h*` rel diff.

| variant | regime | fwd | dx | dh0 | weight-grad |
|---|---|---|---|---|---|
| Dense Triton persistent | fp32 | 4e-4 | 4e-4 | 8e-4 | 1e-3 |
| Dense Triton persistent | int8 QAT (hidden) | 8% | 7% | 9% | — |
| Monarch persistent, nb=4 | fp32 | 3e-4 | 5e-4 | 7e-4 | 2e-3 |
| Monarch persistent, nb=4 | int8 QAT (hidden) | 8% | 7% | 6% | 3% |
| Monarch persistent, nb=8 | fp32 | 2e-4 | 4e-4 | 6e-4 | 2e-3 |
| Monarch persistent, nb=8 | int8 QAT (hidden) | 8% | 5% | 8% | 3% |
| Butterfly persistent | fp32 | 3e-2 | 3e-3 | 1e-3 | 2e-3 |
| Butterfly persistent | int8 QAT (hidden) | 15% | 15% | 1% | 8% |
| **Diagonal persistent** | **fp32** | **1e-6** | **4e-5** | **2e-7** | **2e-6** |
| **Diagonal persistent** | **int8 QAT (hidden)** | **0** | **3e-5** | **2e-7** | **1e-6** |

QAT rows for the matmul-based variants show ~5–15% relative drift
because each step's `round(x/scale)` flips at half-integer boundaries
when `tl.dot`'s TF32 reduction order disagrees with cuBLAS by
`O(scale)` — a single rounding flip per ~100 positions, amplified by
the recurrence over T=64 steps. Not a kernel bug: `torch.round` and
`tl.extra.libdevice.rint` are bit-identical on the same fp32 input
(verified across 1M values + half-integer perturbations). Forcing
Triton to `input_precision="ieee"` would tighten the QAT rows to
~1e-3 at the cost of ~2-4× slower matmul; current choice is speed
over bit-parity. The butterfly fp32 row's `3e-2` fwd is the same
story (kernel TF32 vs `torch_structured`'s CUDA op).

The diagonal variant has *no matmul* on the hidden side, so it ducks
this story entirely: every multiplication is elementwise and Triton
emits the same FMA order as the PyTorch reference. The QAT fwd row is
exactly bit-identical (rel diff = 0); fp32 and grad rows are at fp32
machine precision (~1e-5 / ~1e-7).

## Layout

```
src/gru_qat/
  __init__.py             public API
  ste.py                  STE autograd functions
  quantizers.py           FakeQuantize + observers
  calibration.py          calibrate(module, loader, n_batches)
  structure.py            StructureConfig + make_structured_linear
  gru_cell.py             GRUCellQuant (single step, optionally structured)
  gru_layer.py            GRULayer (multi-step + Triton dispatch)
  triton_kernels/
    scan.py               dense persistent fwd+bwd kernels
    scan_diagonal.py      Diagonal persistent fwd+bwd kernels
    scan_monarch.py       Monarch persistent fwd+bwd kernels
    scan_butterfly.py     Butterfly persistent fwd+bwd kernels
```

See [`DEVELOPMENT.md`](./DEVELOPMENT.md) for the file-by-file design
and the per-phase commit history.

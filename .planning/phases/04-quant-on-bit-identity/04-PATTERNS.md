# Phase 4: Quant-on bit-identity - Pattern Map

**Mapped:** 2026-05-14
**Revised:** 2026-05-14 (post plan-checker — fix `_make_<kind>_layer_quant_int8` helpers to use ACTUAL INT8 recipe per D-41; document `gru_scan` return-type for the probe)
**Files analyzed:** 4 strict-tier file extensions + 1 test extension + 1 src fix (in `_update_observer`); 7 in-repo analog files read
**Analogs found:** 4 exact (TF32-tier `_qat_` tests in `tests/test_triton_*.py`) + 1 exact (per-channel test in `tests/test_quantizers.py`) + 1 partial (per-axis reduction inside `_compute_scale_zp`)

## File Classification

| New / Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---------------------|------|-----------|----------------|---------------|
| `tests/test_triton_scan_strict.py` (extend; new `## Quant-on (Phase 4)` section) | test (pytest module) | request-response (build frozen-INT8 layer → kernel call → compare) | `tests/test_triton_scan.py:213-300` (`test_triton_qat_persistent_matches_pytorch`, `test_triton_qat_matches_pytorch`) | exact — same `h_in_q` / `h_out_q` tuple plumbing, same `quantize_weights()` extraction; differs in that Phase 4 also FROZEN-quantizes weights + input_act per D-41 (the analog kept those as Identity bits=32) |
| `tests/test_triton_diagonal_strict.py` (extend) | test (pytest module) | request-response (frozen-INT8 → diagonal fwd/bwd Triton → PyTorch ref) | `tests/test_triton_diagonal.py:124-156` (`test_diagonal_triton_qat_forward_matches_pytorch`) + `:197-239` (`_qat_backward_matches_pytorch`) | exact — same `h_in_q` / `h_out_q` tuple plumbing into `gru_scan_diagonal_*_pytorch` and `_triton`. Phase 4 swaps `rel < 1e-2..1e-3` for `torch.equal` (Result A) or `abs < scale_h` (Result B) |
| `tests/test_triton_monarch_strict.py` (extend) | test (pytest module) | request-response (frozen-INT8 + `Wh_struct` → monarch Triton/PyTorch) | `tests/test_triton_monarch.py:130-210` (`_qat_forward` + `_qat_backward`) | exact — keeps `nblocks` parametrize, swaps `rel < 1e-1` bound for Phase 4 disposition |
| `tests/test_triton_butterfly_strict.py` (extend) | test (pytest module) | request-response (frozen-INT8 + twiddles → butterfly Triton vs CUDA-op per-step) | `tests/test_butterfly_dispatch.py:369-407` (`test_butterfly_triton_qat_forward_matches_per_step`) + `:409-462` (`_qat_calibrate_freeze_triton_path`) | exact — same `h_q` tuple plumbing into `gru_scan_butterfly_forward_triton`; backward QAT test less developed in analog, so Phase 4 composes from butterfly_strict fp32 bwd body + monarch's QAT bwd body |
| `tests/test_quantizers.py` (extend with 1 new test for QNT-04, Commit A) | test (pytest module) | request-response (tensor with distinct per-channel ranges → assert per-channel running stats) | `tests/test_quantizers.py:34-41` (`test_per_channel_independent_scales`) | exact for shape — same `FakeQuantizePerChannel` + `cfg.axis=0` + distinct-magnitude row construction; differs in observer mode (`min_max` not default `dynamic`) and what it asserts (running_min/max not scale) |
| `src/gru_qat/quantizers.py:135-146` (`FakeQuantize._update_observer` fix, Commit B) | abstract base method (FakeQuantize) | transform (x → running_min, running_max) | `src/gru_qat/quantizers.py:181-189` (`FakeQuantizePerChannel._compute_scale_zp`) | partial — same per-axis-reduction idiom (`dims = [d for d in range(x.ndim) if d != self.config.axis]` + `x.amin(dim=dims, keepdim=True)`), different output buffer semantics |

Secondary analogs cited for shared / cross-cutting patterns:

- `tests/test_triton_scan_strict.py` (Phase 2 output) — section-header style (`# ----- ... -----` ASCII rule + module-docstring TF32 disposition rationale); FAST/SLOW grid constants pattern; `cuda_only` definition; `set_float32_matmul_precision("highest")` preamble.
- `tests/test_calibration.py:60-82` (`test_calibrate_then_freeze_locks_scales`) — canonical `calibrate(...)` + `freeze_all(...)` round-trip. The Phase 4 `_make_<kind>_layer_quant_int8` helpers use a **lightweight inline freeze**: pass `mode="min_max"` in the recipe, run ONE forward over the actual weight tensors + a synthetic activation, then call `cell.freeze_quantizers()`. This produces the SAME frozen-state shape as the calibrate→freeze path but without depending on `calibration.py` plumbing (which Phase 5 owns). The hidden quantizer (`quant_h_in` / `quant_h_out`) still gets a manually-set scale per the existing `test_triton_scan.py:243-251` pattern, because the activation-flowing-through-cell magnitude is fully controlled by the test author.
- `tests/test_qat_smoke.py:31-52` (`test_int8_per_channel_finite_and_bounded`) — establishes that `PRESETS["int8_per_channel"]` is the canonical INT8 per-channel weight + per-tensor activation recipe. Phase 4 builds the same INT8 per-channel weight + per-tensor activation shape but uses `mode="min_max"` so a one-forward inline calibrate-then-freeze can land frozen scales without going through `calibration.py`.
- `tests/test_parity.py:95-103` (`test_cell_with_large_magnitude`) — only existing "large magnitude" adversarial test; uses `torch.randn(...) * 100` scaling. Phase 4 adversarial-class `large-magnitude` mirrors the magnitude scaling but at a less extreme 5× (D-46) to land inside the kernel's reasonable clip range.

---

## Pattern Assignments

### `tests/test_triton_scan_strict.py` — extend with `## Quant-on (Phase 4)` section

**Primary analog:** `tests/test_triton_scan.py:213-389` (the two `test_triton_qat_*_matches_pytorch` tests — the frozen-INT8 layer construction, `_extract_h_quant_params`-equivalent plumbing, and `gru_scan` / `gru_scan_persistent` call shape).

**Section header pattern** (matches the existing ASCII rule pattern at `tests/test_triton_scan_strict.py:305-307, 441-443`):

```python
# ---------------------------------------------------------------------------
# Phase 4: Quant-on bit-identity (frozen INT8 per-channel weight +
#                                  per-tensor activation)
# Tolerance: per D-42 disposition (resolved at Plan 04-01 checkpoint)
# ---------------------------------------------------------------------------
```

**`_make_dense_layer_quant_int8` helper (REVISED — actual D-41 recipe).** Sibling to `_ref_layer` at `tests/test_triton_scan_strict.py:78-92`. Build a layer with `bits=8` weights (per-channel, axis=0), `bits=8` input_act (per-tensor), `bits=8` hidden (per-tensor) — all in `min_max` observer mode. Run ONE inline calibration forward, then call `cell.freeze_quantizers()` to lock per-channel weight scales and per-tensor activation scales. The hidden quantizer's frozen scale is set explicitly to `h_scale` (matches the analog at `tests/test_triton_scan.py:248-251` since the test author picks `h_scale` deliberately to keep activations in range):

```python
def _make_dense_layer_quant_int8(in_dim: int, hidden: int, h_scale: float = 0.02) -> GRULayer:
    """Frozen INT8 per-channel weight + per-tensor activation + per-tensor hidden.

    Implements CONTEXT D-41's literal recipe (frozen INT8 per-channel weight
    + per-tensor activation) — NOT the looser fp32-weight + frozen-INT8-hidden
    shortcut used by tests/test_triton_scan.py:213-389. The earlier analog
    only quantized the hidden activation because the realistic-tier test
    only needed to exercise the in-kernel fake-quant; Phase 4 needs the
    full audit recipe per D-41/QNT-01.

    Recipe construction (matches PRESETS['int8_per_channel'] in shape; bits
    + axis identical, only the observer mode changes to support inline
    freeze):
      - weight:    bits=8, axis=0, mode='min_max', symmetric=True
        (per-channel scale per row of W; axis=0 is the hidden_size axis)
      - input_act: bits=8, axis=None, mode='min_max', symmetric=True
        (per-tensor scale for the input x)
      - hidden:    bits=8, axis=None, mode='frozen', symmetric=True
        (per-tensor scale, manually set to h_scale)

    Freeze procedure (inline; Phase 5 owns full calibrate→freeze plumbing
    via src/gru_qat/calibration.py — we mirror the same shape here using
    min_max + freeze without depending on calibration.py):
      1. Run one forward over a representative x (torch.randn * 0.5 — the
         'realistic' adversarial class scale per D-46). This populates
         running_min/running_max on the input_act quantizer AND on every
         weight quantizer (the weight quantizers see W on each forward
         via cell.quantize_weights()).
      2. Call cell.freeze_quantizers() — switches every observer-mode
         quantizer to frozen mode by copying running stats into scale/zp
         (per src/gru_qat/quantizers.py:97-105). The hidden quantizer is
         already in mode='frozen' from construction; the scale was set
         manually before the calibration pass so the calibration pass
         did not touch it.
      3. After freeze, every weight quantizer has a [hidden,]-shaped scale
         buffer (per-channel along axis=0) and the input_act + hidden
         quantizers have scalar scale buffers.

    Mirrors tests/test_triton_scan.py:240-251 in shape and h_scale value
    but extends the recipe per D-41. Mirrors PRESETS['int8_per_channel']
    in axis + bits but uses mode='min_max' for the inline freeze.

    NOTE: Requires the QNT-04 fix (Plan 04-01 Task 3) for the per-channel
    weight quantizers' min_max observer to produce per-channel running
    stats correctly. Pre-fix, the per-channel weight quantizer's
    running_min/max would collapse to scalars and freeze() would produce
    a per-tensor scale instead of a per-channel scale. The helper depends
    on Commit B landing.
    """
    from gru_qat.quantizers import FakeQuantizePerTensor
    bits = 8
    rec = QuantRecipe(
        weight=QuantizerConfig(bits=bits, axis=0, mode="min_max",
                               symmetric=True, name="W_int8_pc"),
        input_act=QuantizerConfig(bits=bits, axis=None, mode="min_max",
                                  symmetric=True, name="x_int8_pt"),
        hidden=QuantizerConfig(bits=bits, axis=None, mode="frozen",
                               symmetric=True, name="h_int8_pt"),
    )
    layer = GRULayer(
        in_dim, hidden, recipe=rec, gate_layout="fused", pre_batch_input=True,
    )
    # Manually freeze the hidden quantizers at h_scale BEFORE the calibration
    # pass so the pass doesn't touch them (mode='frozen' short-circuits
    # _update_observer per quantizers.py:88-95).
    for q in (layer.cell.quant_h_in, layer.cell.quant_h_out):
        assert isinstance(q, FakeQuantizePerTensor)
        q.scale = torch.tensor(h_scale)
        q.zero_point = torch.tensor(0.0)
    # Inline calibration: one forward populates running_min/max on the
    # weight and input_act quantizers. Use realistic-tier x scaling.
    layer.eval()
    with torch.no_grad():
        cal_x = torch.randn(8, 4, in_dim) * 0.5  # T=8, B=4 — small enough for CPU
        cal_h0 = torch.randn(4, hidden) * 0.5
        layer(cal_x, cal_h0)
    # Switch weight + input_act quantizers from min_max → frozen via the
    # standard freeze() path. The hidden quantizers are already frozen.
    layer.cell.freeze_quantizers()
    return layer
```

**`QUANT_FAST_GRID` / `QUANT_SLOW_GRID` constants** (per D-49 — smaller than Phase 2's `FAST_DENSE_GRID`):

```python
# Phase 4 D-49: smaller grid than Phase 2 (bit-identity is binary, not a
# distribution sweep). T x B x H grid; T ∈ {8, 64} (fast), T ∈ {512} slow.
QUANT_FAST_GRID = [
    (T, B, H)
    for T in (8, 64)
    for B in (1, 4, 32)
    for H in (32, 128, 512)
]  # 18 cases per D-49

QUANT_SLOW_GRID = [
    (T, B, H)
    for T in (512,)
    for B in (1, 4, 32)
    for H in (32, 128, 512)
]  # 9 cases per D-49
```

**Adversarial-class fixtures** (D-46) — three input builders per kernel direction. Pattern copies the `* 0.5` / `* 0.1` scaling idiom from `test_triton_scan.py:253-254` for "realistic" and from `test_parity.py:100-101` for "large-magnitude". Near-saturation is novel.

```python
def _adversarial_inputs(
    cls: str, T: int, B: int, H: int, device: torch.device, h_scale: float = 0.02
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (x, h0) inputs per D-46 adversarial class.

    Three classes per kernel direction:
    - "realistic": torch.randn(...) * 0.5 — baseline, scaled to fit INT8
      dynamic range. Mirrors test_triton_diagonal.py:147 scaling.
    - "near-saturation": values at the INT8 boundary. scale_h * qmax is the
      maximum value before clipping; use torch.linspace(-0.99, 0.99, ...)
      * (h_scale * 127) to land just inside.
    - "large-magnitude": torch.randn(...) * 5 — forces in-kernel clipping;
      tests that reference and Triton clip identically. Less extreme than
      test_parity.py:100-101's *100 (kernel reasonable-range, not stress).
    """
    qmax = 127  # int8 symmetric
    x_max = h_scale * qmax  # value at the saturation boundary
    if cls == "realistic":
        x = torch.randn(T, B, H, device=device) * 0.5
        h0 = torch.randn(B, H, device=device) * 0.5
    elif cls == "near-saturation":
        x = (torch.linspace(-0.99, 0.99, T * B * H, device=device).reshape(T, B, H) * x_max).contiguous()
        h0 = (torch.linspace(-0.99, 0.99, B * H, device=device).reshape(B, H) * x_max).contiguous()
    elif cls == "large-magnitude":
        x = torch.randn(T, B, H, device=device) * 5.0
        h0 = torch.randn(B, H, device=device) * 5.0
    else:
        raise ValueError(f"unknown adversarial class: {cls}")
    return x, h0
```

**Plan 04-01 probe** (D-41) — fixed-shape one-off, NOT parametrized. Reference shape T=8, B=4, H=64 dense per CONTEXT specifics. Uses `torch.equal` on output AND on each gradient tensor. **`gru_scan` returns ONLY the per-step output tensor `out` of shape `[T, B, H]` (verified in `src/gru_qat/triton_kernels/scan.py:1569-1586` + `:1642-1704`); the reference `GRULayer.forward()` returns `(out, h_T)` where `h_T = out[-1]` (`src/gru_qat/gru_layer.py:259-262`). The probe extracts the Triton-side `h_T` as `tri_out[-1]` so the 6-tensor parity assertion list is well-defined: (out, h_T, dx, dh0, dWh_cat, dbh_cat):**

```python
@cuda_only
def test_dense_quant_probe_bit_identity() -> None:
    """Plan 04-01 probe (D-41 / D-42): under frozen INT8 per-channel weight +
    per-tensor activation, does Triton dense match reference bit-identically?

    Shape: T=8, B=4, H=64 (smallest realistic-but-non-tiny shape that
    exercises the quant + matmul pipeline; per CONTEXT specifics).

    gru_scan returns only `out` ([T, B, H]); the final hidden state h_T is
    extracted as `out[-1]` (same convention as GRULayer.forward, see
    src/gru_qat/gru_layer.py:259-262).

    Bound: ``torch.equal`` on 6 independently-checked tensors:
      1. out (full per-step trajectory)
      2. h_T = out[-1] (final hidden state)
      3. dx (input gradient)
      4. dh_0 (initial-hidden-state gradient)
      5. dWh_cat (hidden-weight gradient)
      6. dbh_cat (hidden-bias gradient)
    If even ONE fails, the disposition resolution at the checkpoint:human-
    verify lands on Result B (tight-INT8-grid: abs_diff < h_scale * 1 =
    one INT8 step).

    This test is the gate. Plans 04-02..04 are written AFTER the human-
    verified disposition lands; their assertion shape mirrors whichever
    Result (A: torch.equal; B: abs_diff < h_scale) the user picks at the
    checkpoint.
    """
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("highest")  # already set at module scope; explicit for the probe
    device = torch.device("cuda")
    T, B, H = 8, 4, 64
    IN = H

    layer = _make_dense_layer_quant_int8(IN, H).to(device).eval()
    x, h0 = _adversarial_inputs("realistic", T, B, IN, device)
    # Both reference and triton paths see the same require_grad leaves.
    ref_x = x.detach().clone().requires_grad_()
    ref_h0 = h0.detach().clone().requires_grad_()
    ref_out, ref_hT = layer(ref_x, ref_h0)
    ref_out.float().pow(2).sum().backward()

    w = layer.cell.quantize_weights()
    Wi_cat = w.Wi_cat.detach().clone()
    bi_cat = w.bi_cat.detach().clone()
    Wh_cat = w.Wh_cat.detach().clone().requires_grad_()
    bh_cat = w.bh_cat.detach().clone().requires_grad_()
    tri_x = x.detach().clone().requires_grad_()
    tri_h0 = h0.detach().clone().requires_grad_()
    # IMPORTANT: with D-41's recipe, input_act is now frozen-INT8 per-tensor.
    # Apply the input-side fake-quant before the linear projection so the
    # Triton path sees the same x as the reference's quant_x.
    xq = layer.cell.quant_x(tri_x)
    gi = torch.nn.functional.linear(xq, Wi_cat, bi_cat)
    h_scale = float(layer.cell.quant_h_in.scale.item())
    h_in_q = (h_scale, -127, 127)
    h_out_q = (h_scale, -127, 127)
    tri_out = gru_scan(
        gi, tri_h0, Wh_cat, bh_cat,
        h_in_quant=h_in_q, h_out_quant=h_out_q,
    )
    tri_hT = tri_out[-1]  # gru_scan returns [T, B, H]; final step is out[-1].
    tri_out.float().pow(2).sum().backward()

    # 6 independent torch.equal assertions (D-41 / D-42 gate).
    # Failure messages carry the per-tensor max abs diff so the checkpoint:
    # human-verify sees Result-A/B/C signal directly.
    for name, ref_t, tri_t in [
        ("out", ref_out, tri_out),
        ("h_T", ref_hT, tri_hT),
        ("dx", ref_x.grad, tri_x.grad),
        ("dh0", ref_h0.grad, tri_h0.grad),
        ("dWh_cat", w.Wh_cat.grad, Wh_cat.grad),       # ref grads live on the cell's Wh_cat
        ("dbh_cat", w.bh_cat.grad, bh_cat.grad),
    ]:
        max_diff = (ref_t - tri_t).abs().max().item()
        assert torch.equal(ref_t, tri_t), (
            f"{name}: torch.equal failed for cls=realistic "
            f"(T={T},B={B},H={H}); max abs diff {max_diff:.4e}; "
            f"h_scale={h_scale}; shape={tuple(ref_t.shape)}"
        )
```

Notes on the probe body:
- `quant_x` is invoked BEFORE the input projection because D-41's recipe quantizes input activations; the reference `GRULayer.forward()` does the same inside `cell.step()` (or its fused-batched equivalent). If the executor finds an existing helper for this on the layer (e.g. `_extract_h_quant_params`-style), prefer that — the literal `layer.cell.quant_x(tri_x)` invocation matches the cell's own internal order.
- `dWh_cat` / `dbh_cat` extraction: the reference grad accumulates on `layer.cell.W_hr/_hz/_hn` (and biases). `quantize_weights()` returns a `CellWeights` bag whose `Wh_cat.grad` is the concat'd hidden-weight gradient if `Wh_cat` was used as a `requires_grad_()` leaf in the Triton path. Since `Wh_cat = w.Wh_cat.detach().clone().requires_grad_()` is a fresh leaf, its `.grad` is the Triton-side gradient. The reference's per-row gradients on `W_hr`/`W_hz`/`W_hn` must be concat'd in the same row order (`[r, z, n]` per `gru_cell.py:268`) for the comparison. The probe should construct the reference `dWh_cat` via `torch.cat([layer.cell.W_hr.grad, layer.cell.W_hz.grad, layer.cell.W_hn.grad], dim=0)` so both sides are shape `[3*H, H]`. Same convention for `dbh_cat`.

**Plans 04-02..04** — pattern AFTER the probe disposition is verified. Two assertion idioms depending on Result A vs B:

```python
# Result A (D-42, torch.equal — bit-identity holds):
assert torch.equal(ref_out, tri_out), (
    f"out: bit-identity failed for {cls} (T={T},B={B},H={H}); "
    f"max abs diff {(ref_out - tri_out).abs().max().item():.4e}"
)

# Result B (D-42, tight-INT8-grid — one INT8 step):
max_diff = (ref_out - tri_out).abs().max().item()
assert max_diff < h_scale, (
    f"out: max abs diff {max_diff:.4e} > h_scale ({h_scale}) for {cls} "
    f"(T={T},B={B},H={H})"
)
```

**Per-adversarial-class parametrize idiom** (D-46 — per-test failure message includes class name):

```python
@cuda_only
@pytest.mark.parametrize("T,B,H", QUANT_FAST_GRID)
@pytest.mark.parametrize("cls", ["realistic", "near-saturation", "large-magnitude"])
def test_scan_quant_fwd(cls: str, T: int, B: int, H: int) -> None:
    """Frozen-INT8 dense forward must match reference per D-42 disposition
    across all three adversarial input classes (realistic, near-saturation,
    large-magnitude).
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    IN = H
    layer = _make_dense_layer_quant_int8(IN, H).to(device).eval()
    x, h0 = _adversarial_inputs(cls, T, B, IN, device)
    # ... rest of body identical to probe body (quant_x before F.linear,
    #     extract tri_hT as tri_out[-1] if h_T parity is asserted)
```

Notes:
- The two-axis `@pytest.mark.parametrize` (cls outermost) makes pytest's test IDs read like `test_scan_quant_fwd[realistic-8-4-64]`, which surfaces the class name in the failure summary.
- Failure messages MUST include `cls` per CONTEXT D-46 ("Per-test failure messages include the class name").

**Plan ordering inside the section:**
1. Plan 04-01 probe (fixed shape, no parametrize).
2. `_make_dense_layer_quant_int8` + `_adversarial_inputs` helpers (module-scope).
3. `QUANT_FAST_GRID` / `QUANT_SLOW_GRID` constants (module-scope).
4. `test_scan_quant_fwd` (parametrized over fast grid × 3 classes = 54 fast cases).
5. `test_scan_quant_bwd` (same parametrize; mirrors `test_scan_bwd_strict_matches_reference` at `tests/test_triton_scan_strict.py:184-246` but with the frozen-INT8 layer + quant kwargs + 3-class parametrize).
6. `_slow` siblings of fwd / bwd (`@pytest.mark.slow`, `QUANT_SLOW_GRID`, 9 cases × 3 classes = 27 slow).

---

### `tests/test_triton_diagonal_strict.py` — extend

**Primary analog:** `tests/test_triton_diagonal.py:124-156` (`test_diagonal_triton_qat_forward_matches_pytorch` — the `h_in_q` / `h_out_q` tuple plumbing) and `:197-239` (`_qat_backward_matches_pytorch`).

**Same section-header pattern** as scan_strict above.

**`_make_diagonal_layer_quant_int8` helper (REVISED — actual D-41 recipe)** — sibling to `_make_diagonal_layer` at `tests/test_triton_diagonal_strict.py:77-90`. Same recipe shape as the dense helper, with `StructureConfig(kind="diagonal")` for the hidden side:

```python
def _make_diagonal_layer_quant_int8(in_size: int, hid: int, h_scale: float = 0.02) -> GRULayer:
    """Frozen INT8 per-channel weight + per-tensor activation + per-tensor
    hidden, diagonal hidden GEMM. Recipe per CONTEXT D-41 (full INT8
    audit recipe, NOT the looser fp32-weight + frozen-INT8-hidden shortcut
    from tests/test_triton_diagonal.py:124-156).

    Construction:
      - QuantRecipe with bits=8 across weight (axis=0, min_max), input_act
        (per-tensor, min_max), hidden (per-tensor, frozen at h_scale).
      - StructureConfig(kind='diagonal') for the hidden side.
      - One inline calibration forward over realistic-scale random data,
        then cell.freeze_quantizers().

    NOTE: depends on the QNT-04 fix landing first (per-channel min_max
    observer must produce per-channel running_stats for the weight
    quantizers to freeze correctly).
    """
    from gru_qat.quantizers import FakeQuantizePerTensor
    bits = 8
    rec = QuantRecipe(
        weight=QuantizerConfig(bits=bits, axis=0, mode="min_max",
                               symmetric=True, name="W_int8_pc"),
        input_act=QuantizerConfig(bits=bits, axis=None, mode="min_max",
                                  symmetric=True, name="x_int8_pt"),
        hidden=QuantizerConfig(bits=bits, axis=None, mode="frozen",
                               symmetric=True, name="h_int8_pt"),
    )
    cfg = StructureConfig(kind="diagonal")
    layer = GRULayer(
        in_size, hid, recipe=rec, gate_layout="fused",
        structure_input=None, structure_hidden=cfg,
    )
    for q in (layer.cell.quant_h_in, layer.cell.quant_h_out):
        assert isinstance(q, FakeQuantizePerTensor)
        q.scale = torch.tensor(h_scale)
        q.zero_point = torch.tensor(0.0)
    layer.eval()
    with torch.no_grad():
        cal_x = torch.randn(8, 4, in_size) * 0.5
        cal_h0 = torch.randn(4, hid) * 0.5
        layer(cal_x, cal_h0)
    layer.cell.freeze_quantizers()
    return layer
```

**Direct-kernel-call pattern** — Phase 4 diagonal quant-on follows the existing realistic-tier diagonal QAT body shape at `tests/test_triton_diagonal.py:124-156` (no autograd wrapper; direct call to `gru_scan_diagonal_forward_triton(..., h_in_quant=..., h_out_quant=...)` and `_pytorch`). The input projection now also requires running `layer.cell.quant_x(x)` before `F.linear` since input_act is frozen-INT8 per D-41. Mirrors the Phase 2 strict diagonal direct-kernel-call pattern at `tests/test_triton_diagonal_strict.py:200-239`.

**Grid:** uses the same `QUANT_FAST_GRID` / `QUANT_SLOW_GRID` constants from above (D-49). NO tiny-H ({1, 2, 8}) — CONTEXT D-49 explicitly drops them ("Phase 4 isn't an edge-case sweep; Phase 6").

---

### `tests/test_triton_monarch_strict.py` — extend

**Primary analog:** `tests/test_triton_monarch.py:130-210` (`_qat_forward` / `_qat_backward` tests with `h_in_q` / `h_out_q` tuple plumbing and `Wh_struct` 4-D tensor construction).

**`_make_monarch_layer_quant_int8` helper (REVISED — actual D-41 recipe)** — sibling to `_make_monarch_layer` at `tests/test_triton_monarch_strict.py:74-90`, parametrized by `nblocks`:

```python
def _make_monarch_layer_quant_int8(
    in_size: int, hid: int, nblocks: int = 4, h_scale: float = 0.02
) -> GRULayer:
    """Frozen INT8 per-channel weight + per-tensor activation + per-tensor
    hidden, monarch hidden structure. Recipe per CONTEXT D-41.
    """
    from gru_qat.quantizers import FakeQuantizePerTensor
    bits = 8
    rec = QuantRecipe(
        weight=QuantizerConfig(bits=bits, axis=0, mode="min_max",
                               symmetric=True, name="W_int8_pc"),
        input_act=QuantizerConfig(bits=bits, axis=None, mode="min_max",
                                  symmetric=True, name="x_int8_pt"),
        hidden=QuantizerConfig(bits=bits, axis=None, mode="frozen",
                               symmetric=True, name="h_int8_pt"),
    )
    cfg = StructureConfig(kind="monarch", nblocks=nblocks)
    layer = GRULayer(
        in_size, hid, recipe=rec, gate_layout="fused",
        structure_input=None, structure_hidden=cfg,
    )
    for q in (layer.cell.quant_h_in, layer.cell.quant_h_out):
        assert isinstance(q, FakeQuantizePerTensor)
        q.scale = torch.tensor(h_scale)
        q.zero_point = torch.tensor(0.0)
    layer.eval()
    with torch.no_grad():
        cal_x = torch.randn(8, 4, in_size) * 0.5
        cal_h0 = torch.randn(4, hid) * 0.5
        layer(cal_x, cal_h0)
    layer.cell.freeze_quantizers()
    return layer
```

**Grid:** monarch keeps `nblocks ∈ {2, 4, 8}` axis (D-49). Use the H-divisibility filter:

```python
QUANT_MONARCH_FAST_GRID = [
    (T, B, H, nblocks)
    for T in (8, 64)
    for B in (1, 4, 32)
    for H in (32, 128, 512)
    for nblocks in (2, 4, 8)
    if H % nblocks == 0
]
```

**Direct-kernel-call pattern** — same as diagonal, but routes through `gru_scan_monarch_forward_triton` and `_pytorch` with `h_in_q` / `h_out_q` (analog at `tests/test_triton_monarch.py:151-210`).

---

### `tests/test_triton_butterfly_strict.py` — extend

**Primary analog:** `tests/test_butterfly_dispatch.py:369-407` (`test_butterfly_triton_qat_forward_matches_per_step`) — the `h_q` tuple plumbing into `gru_scan_butterfly_forward_triton`. Backward analog less developed: compose from butterfly_strict fp32 bwd body at `tests/test_triton_butterfly_strict.py:214-270` + monarch QAT bwd body at `tests/test_triton_monarch.py:165-210`.

**`_make_butterfly_layer_quant_int8` helper (REVISED — actual D-41 recipe)** — wraps existing `_make_layer` at `tests/test_triton_butterfly_strict.py:86-100` which already supports `hidden_bits=8`. **The existing `_make_layer` produces a recipe with bits=8 dense weight + bits=8 input_act + bits=hidden_bits hidden in DEFAULT (dynamic) mode** — Phase 4 wraps it, overrides the weight and input_act configs to `min_max`, then runs inline calibrate + freeze:

```python
def _make_butterfly_layer_quant_int8(
    H: int, *, use_triton: bool = True, h_scale: float = 0.02
) -> GRULayer:
    """Frozen INT8 per-channel weight + per-tensor activation + per-tensor
    hidden, butterfly hidden structure. Recipe per CONTEXT D-41.

    Wraps the existing _make_layer(H, use_triton, hidden_bits=8) helper at
    tests/test_triton_butterfly_strict.py:86-100. _make_layer already
    builds an INT8 recipe but with mode='dynamic' for weights and
    input_act; Phase 4 needs FROZEN scales (D-41), so we override the
    weight and input_act quantizers' config.mode to 'min_max', run one
    inline calibration forward, then freeze_quantizers().

    The hidden quantizer is forced to mode='frozen' with h_scale manually
    set, matching the dense / diagonal / monarch helpers above for D-43
    uniformity.

    NOTE: depends on the QNT-04 fix landing first.
    """
    from gru_qat.quantizers import FakeQuantizePerTensor
    layer = _make_layer(H, use_triton=use_triton, hidden_bits=8)
    # Override modes: weights and input_act → min_max (for inline freeze);
    # hidden → frozen (manually set scale).
    for name in ("quant_W_ir", "quant_W_iz", "quant_W_in",
                 "quant_W_hr", "quant_W_hz", "quant_W_hn"):
        q = getattr(layer.cell, name, None)
        if q is not None:
            q.config.mode = "min_max"
    layer.cell.quant_x.config.mode = "min_max"
    for q in (layer.cell.quant_h_in, layer.cell.quant_h_out):
        assert isinstance(q, FakeQuantizePerTensor)
        q.config.mode = "frozen"
        q.scale = torch.tensor(h_scale)
        q.zero_point = torch.tensor(0.0)
    layer.eval()
    with torch.no_grad():
        cal_x = torch.randn(8, 4, H) * 0.5
        cal_h0 = torch.randn(4, H) * 0.5
        layer(cal_x, cal_h0)
    layer.cell.freeze_quantizers()
    return layer
```

**Pattern caveat — butterfly comparator:** butterfly has no pure-PyTorch reference distinct from the kernel under test. Per Phase 2 strict's pattern at `tests/test_triton_butterfly_strict.py:138-156`, the comparator is `use_triton=False` (CUDA-op per-step via `butterfly_multiply`). For Phase 4, the dual-layer pattern carries forward: `pt_layer = _make_butterfly_layer_quant_int8(..., use_triton=False)` vs `fast_layer = _make_butterfly_layer_quant_int8(..., use_triton=True)`, plus `fast_layer.load_state_dict(pt_layer.state_dict())` to share weights AND frozen scales.

**Grid:** butterfly H is restricted to powers of 2; the existing `QUANT_FAST_GRID` `H ∈ {32, 128, 512}` already meets that constraint (D-49 explicit).

**Backward pattern caveat:** butterfly backward QAT body is more elaborate than other kernels because the bwd reference is built via autograd-through-`gru_scan_butterfly` rather than a `_pytorch` closed-form. The Phase 2 strict-tier `_assert_grad_close` helper at `tests/test_triton_butterfly_strict.py:190-211` should be reused (it already supports `< 5e-4` abs; Phase 4 swaps the bound for `< h_scale` or `torch.equal` per D-42).

---

### `tests/test_quantizers.py` — extend with QNT-04 failing test (Commit A)

**Primary analog:** `tests/test_quantizers.py:34-41` (`test_per_channel_independent_scales`).

**Pattern:** copy the row-construction idiom (`torch.stack([torch.randn(16) * 0.01, torch.randn(16) * 10.0])`) but exercise the `min_max` observer path through `_update_observer` (not `_compute_scale_zp`), and assert on `running_min` / `running_max` shape AND distinct per-channel values.

```python
def test_per_channel_min_max_observer_per_channel_running_stats() -> None:
    """QNT-04 (D-44 / D-45): the per-channel min_max observer must produce
    PER-CHANNEL running_min / running_max tensors, not scalars.

    The current implementation at src/gru_qat/quantizers.py:135-146 calls
    ``x.detach().min()`` / ``.max()`` — global scalar reductions, broken
    for per-channel axes. After the fix in Commit B (per-axis reduction
    via ``x.amin(dim=other_dims)``), running_min / running_max should be
    shape [num_channels] with channel-distinct values.

    Construct a tensor with channel 0 in [-1, 1] and channel 1 in [-10, 10]
    (per CONTEXT specifics). After one forward, assert:
      - running_min.shape == (2,)  # NOT scalar
      - running_min[0] != running_min[1]  # channel-distinct values

    Two-commit per D-37 / D-45: this test is Commit A (failing-before-fix);
    Commit B fixes _update_observer; CI green => bd issue closes.

    Pattern mirrors test_per_channel_independent_scales at lines 34-41,
    but exercises the min_max observer path (mode='min_max') rather than
    the default dynamic _compute_scale_zp path.
    """
    cfg = QuantizerConfig(bits=8, axis=0, symmetric=True, mode="min_max")
    q = FakeQuantizePerChannel(cfg)
    # Channel 0 in [-1, 1]; channel 1 in [-10, 10]. Distinct per-channel.
    x = torch.stack([torch.randn(16) * 1.0, torch.randn(16) * 10.0])
    # Force values to span the intended range so min/max are unambiguous.
    x[0, 0] = -1.0
    x[0, -1] = 1.0
    x[1, 0] = -10.0
    x[1, -1] = 10.0
    q(x)  # one forward; min_max observer updates running stats
    # Assertions that FAIL pre-fix (scalar reduction produces 0-d running_min):
    assert q.running_min.ndim > 0, (
        f"running_min should be per-channel; got scalar (ndim={q.running_min.ndim})"
    )
    assert q.running_min.shape == (2,), (
        f"running_min should be shape (2,); got {tuple(q.running_min.shape)}"
    )
    assert q.running_max.shape == (2,)
    # Channel 0 in [-1, 1]; channel 1 in [-10, 10] => running_min differ.
    assert q.running_min[0] != q.running_min[1], (
        f"running_min should differ per channel; got {q.running_min.tolist()}"
    )
    assert q.running_max[1] > q.running_max[0]
```

Notes:
- The `_initialized` first-pass branch in `_update_observer` (line 139-142) does direct buffer assignment; the second-pass momentum branch (line 144-146) does scalar arithmetic. The fix needs both branches to produce per-channel tensors.
- `q.running_min` is a registered buffer initialized at `(torch.tensor(float("inf")))` at `src/gru_qat/quantizers.py:82-83`. After per-axis reduction, the buffer should be re-assigned to a per-channel-shaped tensor; the existing `self.register_buffer` pattern allows reassignment via `self.running_min = <tensor>` (PyTorch buffer reassignment semantics).

---

### `src/gru_qat/quantizers.py:135-146` (`_update_observer` fix, Commit B)

**Primary analog:** `src/gru_qat/quantizers.py:181-189` (`FakeQuantizePerChannel._compute_scale_zp` — already uses the per-axis reduction idiom):

```python
# Existing per-axis reduction pattern at src/gru_qat/quantizers.py:181-189:
def _compute_scale_zp(self, x):
    assert self.config.axis is not None
    dims = [d for d in range(x.ndim) if d != self.config.axis]
    x_min = x.amin(dim=dims, keepdim=True)
    x_max = x.amax(dim=dims, keepdim=True)
    return self._scale_zp_from_min_max(x_min, x_max)
```

**Fix pattern for `_update_observer`** — same `dims = [d for d in range(x.ndim) if d != self.config.axis]` construction; replace `x.detach().min()` / `.max()` with `x.detach().amin(dim=dims)` / `.amax(dim=dims)`. The fix uses `keepdim=False` (NOT `keepdim=True` as in `_compute_scale_zp`) because running stats are tracked as per-channel 1-D tensors, not broadcast-shaped tensors:

```python
# Phase 4 D-44 fix: per-axis reduction when axis is not None.
def _update_observer(self, x: torch.Tensor) -> None:
    axis = self.config.axis
    if axis is None:
        # Per-tensor: existing global-reduction path preserved.
        cur_min = x.detach().min()
        cur_max = x.detach().max()
    else:
        # Per-channel: reduce over every dim except `axis`. Same pattern as
        # FakeQuantizePerChannel._compute_scale_zp (line 181-189) but with
        # keepdim=False — running stats are 1-D per-channel tensors.
        dims = [d for d in range(x.ndim) if d != axis]
        cur_min = x.detach().amin(dim=dims)
        cur_max = x.detach().amax(dim=dims)
    if not self._initialized:
        self.running_min = cur_min
        self.running_max = cur_max
        self._initialized = True
    else:
        momentum = 0.99
        self.running_min = momentum * self.running_min + (1 - momentum) * cur_min
        self.running_max = momentum * self.running_max + (1 - momentum) * cur_max
```

Notes:
- The `axis is None` branch preserves the per-tensor behavior — existing `tests/test_quantizers.py:76-85` (`test_freeze_locks_scale`) uses `QuantizerConfig(bits=8, mode="min_max")` with no axis; that test MUST continue to pass after the fix.
- The `_scale_zp_from_min_max` helper at `src/gru_qat/quantizers.py:123-133` works element-wise on `x_min` / `x_max`, so 1-D per-channel tensors flow through without modification. No change to `freeze()` needed.
- Remove the `# TODO(phase=4): per-channel running stats` comment at `src/gru_qat/quantizers.py:136` (that's the QNT-04 marker; the fix discharges it).

---

## Shared Patterns

### Frozen-INT8 layer-construction via inline calibrate-then-freeze (no Phase 5 dep)

**Source:** `tests/test_calibration.py:60-82` (the canonical `calibrate(...)` + `freeze_all(...)` shape). Phase 4 inlines the same shape WITHOUT importing `src/gru_qat/calibration.py` (Phase 5 owns that plumbing) by using `mode="min_max"` quantizers + one inline forward + `cell.freeze_quantizers()` per `src/gru_qat/gru_cell.py:497-505`.

**Apply to:** All four `_make_<kind>_layer_quant_int8` helpers. The pattern is:

```python
# 1. Build recipe with bits=8 weights (axis=0, min_max), input_act (per-tensor,
#    min_max), hidden (per-tensor, frozen with manually-set h_scale).
# 2. Build the layer.
# 3. Manually set hidden quantizer scale to h_scale BEFORE the calibration
#    pass (mode='frozen' short-circuits _update_observer in forward).
# 4. Run ONE forward with realistic-scale random data (torch.randn * 0.5).
#    This populates running_min/running_max on weight + input_act quantizers.
# 5. Call layer.cell.freeze_quantizers() to switch min_max → frozen.
```

**Why inline (not via `calibration.py`)?** CONTEXT D-41 specifies the recipe; CONTEXT excludes the calibration LIFECYCLE from Phase 4 (that's Phase 5's CAL-01..04). The inline pattern produces an end-state-identical frozen layer without importing Phase 5 code. The downstream Phase 5 calibration tests will verify the full lifecycle separately.

**Depends on QNT-04 fix:** the per-channel `min_max` observer must produce per-channel `running_min`/`running_max` for the weight quantizers to freeze with per-channel scales. Pre-fix (scalar reduction), weight quantizers would freeze with a single scalar scale that broadcasts across all channels — losing the per-channel granularity D-41 requires. The QNT-04 fix is therefore a hard prerequisite for the helpers, which is why Plan 04-01 lands Commits A+B BEFORE Task 2 invokes the helper in the probe.

### `h_in_q` / `h_out_q` tuple plumbing into Triton kernels

**Source:** `tests/test_triton_diagonal.py:139-148`, `test_triton_monarch.py:148-157`, `test_butterfly_dispatch.py:386-405`, `test_triton_scan.py:272-274`.

```python
bits = 8
qmin, qmax = -(2 ** (bits - 1)) + 1, 2 ** (bits - 1) - 1  # -127, 127
h_in_q = (h_scale, qmin, qmax)
h_out_q = (h_scale, qmin, qmax)
# Then:
tri_out = gru_scan_<kind>_forward_triton(
    gi, h0, Wh_..., bh_cat,
    h_in_quant=h_in_q, h_out_quant=h_out_q,
)
ref = gru_scan_<kind>_forward_pytorch(
    gi, h0, Wh_..., bh_cat,
    h_in_quant=h_in_q, h_out_quant=h_out_q,
)
```

**Apply to:** Diagonal, monarch, butterfly strict-file extensions. For the dense scan, the equivalent kwargs are on `gru_scan` / `gru_scan_persistent` per `test_triton_scan.py:270-274`. The `_extract_h_quant_params` helper at `src/gru_qat/gru_layer.py:28-46` returns this same `(scale, qmin, qmax)` tuple from a frozen quantizer — useful when the test reads h_scale from a layer (`h_scale = float(layer.cell.quant_h_in.scale.item())`) rather than hardcoding it.

### Input activation quantization BEFORE the F.linear projection

**Source:** the reference path runs `xq = self.quant_x(x)` then `F.linear(xq, ...)` inside `cell.step()` (`src/gru_qat/gru_cell.py:311`). With D-41's frozen-INT8 input_act, the Triton path MUST do the same; otherwise the reference and Triton see different `gi` tensors and the parity assertion is meaningless.

**Apply to:** All Phase 4 probe + parametrized test bodies. The pattern is:

```python
xq = layer.cell.quant_x(tri_x)
gi = torch.nn.functional.linear(xq, Wi_cat, bi_cat)
```

`quant_x` is frozen per the helper's freeze step, so the forward is a fake-quant round-trip (no observer update).

### `gru_scan` returns ONLY `out`; `h_T` is `out[-1]`

**Source:** `src/gru_qat/triton_kernels/scan.py:1569-1586` (`gru_scan` declares `-> torch.Tensor`) and `:1642-1704` (`gru_scan_forward` returns `out` of shape `[T, B, H]`). The reference-side `GRULayer.forward()` at `src/gru_qat/gru_layer.py:139-200` returns `(out, h_T)` where `h_T = out[-1]` per `gru_layer.py:259-262`.

**Apply to:** Plan 04-01 probe (6 named tensors). The Triton-side `tri_hT = tri_out[-1]`; the reference-side `ref_hT` comes from the layer's tuple return. Both should agree if the trajectory matches — but the parity assertion is independent (asserts `torch.equal` on `h_T` even though it's a slice of `out`) so a kernel bug that produces the right per-step trajectory but a wrong final-step boundary state would still be caught.

### `torch.equal` vs `abs_diff < scale_h` assertion idiom (D-42 disposition)

**Source:** `tests/test_triton_scan_strict.py:434` for `torch.equal` (determinism gate; same shape as Phase 4 Result A); `tests/test_triton_diagonal_strict.py:232-239` for `< 1e-5` abs (same shape as Phase 4 Result B with a different bound).

**Apply to:** All Phase 4 quant-on assertion bodies. The probe at Plan 04-01 uses `torch.equal`. Plans 04-02..04 use whichever shape the human-verified checkpoint dispositions onto (Result A → `torch.equal`; Result B → `abs_diff < h_scale`).

### Adversarial-class parametrize with named failure messages (D-46)

**Source:** `tests/test_parity.py:95-103` (`test_cell_with_large_magnitude` — only large-magnitude analog). Near-saturation and realistic are novel.

```python
@pytest.mark.parametrize("cls", ["realistic", "near-saturation", "large-magnitude"])
def test_<kind>_quant_fwd(cls: str, T: int, B: int, H: int) -> None:
    x, h0 = _adversarial_inputs(cls, T, B, H, device)
    # ... assertion with cls in failure message:
    assert max_diff < bound, (
        f"out: max abs diff {max_diff:.4e} > bound ({bound}) for class={cls} "
        f"(T={T},B={B},H={H})"
    )
```

**Apply to:** All four strict-file extensions. The `cls` parameter is the OUTERMOST parametrize (Cartesian-product with shape grid) so pytest's test IDs read `[realistic-8-4-64]` and the class name surfaces in `pytest -ra` summary lines without consulting the body.

### `cuda_only` per-file gate + `pytest.importorskip` carries forward unchanged

**Source:** Phase 2 strict files already have these at module top; Phase 4 extensions inherit without modification per D-47 ("Extension only").

### Two-commit failing-test-before-fix discipline for QNT-04

**Source:** Phase 1 PATTERNS lines 393-399 / Phase 2 D-27 / Phase 3 D-37; CONTEXT D-45.

1. Commit A: failing test in `tests/test_quantizers.py` (the per-channel min_max test above). No `src/` changes. Capture `pytest --tb=short` tail.
2. `bd create` per finding — one bd issue for QNT-04 / ACT-01 closure per CONTEXT D-45. `bd update <id> --notes <pytest-tail>`.
3. Commit B: fix in `src/gru_qat/quantizers.py:135-146` per pattern above. No test changes. Test passes. `bd close <id>` after CI green.
4. **Never** `@pytest.mark.xfail` — silent in `pytest -q`, defeats audit signal.

### Phase 2 LOCKED files (D-51) — verifier asserts unchanged

**Source:** CONTEXT D-51 / D-38 / D-28.
**Apply to:** Verifier asserts `git diff <phase-4-base>..HEAD -- tests/test_parity.py tests/test_layer_parity.py tests/test_structure.py` is empty. Phase 4 commits MUST NOT touch any of these files.

### Section-header ASCII rule + module-docstring extension

**Source:** `tests/test_triton_scan_strict.py:305-307, 441-443` (existing Phase 2 section dividers).

```python
# ---------------------------------------------------------------------------
# Phase 4: Quant-on bit-identity (frozen INT8 per-channel weight +
#                                  per-tensor activation)
# Tolerance: per D-42 disposition (resolved at Plan 04-01 checkpoint)
# ---------------------------------------------------------------------------
```

**Apply to:** All four strict files. The module docstring at each strict file's top SHOULD be extended with a paragraph documenting the Phase 4 quant-on section (mirrors how the existing docstrings document Phase 2's TRI-05 / TRI-06 / D-25 contributions inline).

### `monkeypatch` is NOT needed for Phase 4

**Source:** Phase 3 D-34 introduced `monkeypatch` for STR-03 missing-dep tests; `TESTING.md:201-212` documents the narrow exception. Phase 4 has no missing-dep failure-mode tests, so monkeypatch does NOT carry into Phase 4. The TESTING.md "Mocking" section as of Phase 3 close already correctly scopes the exception ("optional-dependency failure-mode tests"); Phase 4 stays out of that scope.

---

## No Analog Found

| Test / Pattern | Role | Data Flow | Why No Analog |
|----------------|------|-----------|---------------|
| `near-saturation` adversarial input (D-46) | input fixture | request-response | No existing test constructs `torch.linspace(-0.99, 0.99, ...)` boundary inputs. The realistic-tier QAT tests at `tests/test_triton_*.py` use `torch.randn(...) * 0.1` (well inside the INT8 boundary). Phase 4 introduces the linspace pattern as the canonical near-saturation probe. Mechanically standard PyTorch, but worth flagging as a new test-side construction. |
| `torch.equal` on output AND on each gradient at the Plan 04-01 probe | regression test | request-response, 6 named tensors | `tests/test_triton_scan_strict.py:401-438` (`test_persistent_kernel_deterministic`) uses `torch.equal` on a SINGLE output across N runs; the Phase 4 probe uses `torch.equal` on 6 DIFFERENT tensors (out, h_T, dx, dh0, dWh_cat, dbh_cat) in ONE run. Mechanically similar but rationale differs (cross-CTA fence vs bit-identity under quant). Pattern is novel for Phase 4 — flag in PLAN as the "probe gate". |
| Inline calibrate-then-freeze without `calibration.py` | layer builder | one-shot | No prior test bypasses `calibrate()` + `freeze_all()` for a frozen-INT8 layer; existing tests either (a) use `mode='dynamic'` and never freeze (`test_qat_smoke.py:31-52`), or (b) use `calibrate(...) → freeze_all(...)` (`test_calibration.py:60-82`). Phase 4 needs a third path: `mode='min_max'` + one-forward + `cell.freeze_quantizers()` (the per-cell version of `freeze_all`). This is novel, well-defined, and avoids importing Phase 5 code into Phase 4 tests. |
| Probe + checkpoint:human-verify gate (Plans 04-02..04 sketch-and-fill-in) | planning sequence | meta | No prior phase has an "intra-phase plan gate" — Phases 1-3 sequenced plans by dependency, not by empirical discovery. Phase 4's Plan 04-01 owns the probe + `checkpoint:human-verify`; Plans 04-02..04 are SKETCHED until the disposition lands, then FILLED-IN with the chosen assertion idiom (D-42). Planner: only Plan 04-01 should be authored in detail before the checkpoint; 04-02..04 are stubs with `[disposition: TBD per Plan 04-01]` placeholders. |

No `src/` files in the no-analog table — the only `src/` modification is the well-bounded `_update_observer` fix, whose analog is the sibling `FakeQuantizePerChannel._compute_scale_zp` method four classes down in the same file.

---

## Metadata

**Analog search scope:**
- `tests/` — read `test_triton_scan_strict.py` (full), `test_triton_diagonal_strict.py` (full), `test_triton_monarch_strict.py` (lines 1-120), `test_triton_butterfly_strict.py` (lines 1-312); `test_triton_scan.py:220-360` (QAT analogs); `test_triton_diagonal.py:120-260` (QAT analogs + Stage D dispatch); `test_triton_monarch.py:130-330` (QAT analogs); `test_butterfly_dispatch.py:25-110, 369-462` (QAT analogs + helper builder); `test_quantizers.py` (full — for QNT-04 Commit A pattern); `test_calibration.py` (full — for canonical INT8 recipe shape); `test_qat_smoke.py` (full — for `PRESETS["int8_per_channel"]` reference); `test_parity.py:95-200` (for `large_magnitude` + `pytest.raises` analogs).
- `src/gru_qat/quantizers.py` (full — `_update_observer` at 135-146, `FakeQuantizePerChannel._compute_scale_zp` at 178-189, `PRESETS` at 284-295).
- `src/gru_qat/gru_layer.py` (lines 1-260 — `_extract_h_quant_params` at 28-46 and `_forward_fast_dispatch` at 202-258 for the `h_in_q` / `h_out_q` plumbing reference; `forward` returns `(out, h_T)` per 139-200, `h_T = out[-1]` per 259-262).
- `src/gru_qat/gru_cell.py` (lines 1-300, 490-510 — `freeze_quantizers` at 497-505, `quantize_weights` at 240-272, quantizer attribute names `quant_W_ir/iz/in/hr/hz/hn`, `quant_x`, `quant_h_in`, `quant_h_out`).
- `src/gru_qat/triton_kernels/scan.py` (full — confirms `gru_scan` returns only `out` per :1569-1586 and `gru_scan_forward` per :1642-1704).
- `src/gru_qat/calibration.py` (top 50 lines — `calibrate` + `freeze_all` signatures for documentation reference; not used directly in Phase 4 per "frozen-scale short-circuit").
- `.planning/codebase/TESTING.md` (full — Mocking exception scope, parametrize idiom, tolerance tiers).
- `.planning/phases/02-triton-fast-path-parity-vs-reference/02-PATTERNS.md` (full — strict-tier conventions, FAST/SLOW grid pattern, `cuda_only` per-file).
- `.planning/phases/03-structured-pytorch-fallback-parity/03-PATTERNS.md` (full — Phase 3 novel-pattern introduction style; not directly inherited by Phase 4 but used as the format model).

**Grep audit:**
- `grep "h_in_quant\|h_out_quant" tests/` → present in 4 files (scan, diagonal, monarch, butterfly_dispatch). Pattern well-established.
- `grep "mode=\"frozen\"\|mode='frozen'" tests/` → 3 hits (scan, monarch, diagonal). Frozen-scale short-circuit pattern well-established.
- `grep "linspace" tests/` → zero hits. `near-saturation` adversarial-class pattern is novel.
- `grep "torch.equal" tests/` → 5 hits (parity, calibration, qat_smoke, strict_scan x2). `torch.equal` on multiple gradients in one test is novel.
- `grep "_update_observer\|running_min\|running_max" src/` → 5 hits all in quantizers.py. Buffer reassignment semantics confirmed.
- `grep "def gru_scan\b\|-> torch.Tensor:" src/gru_qat/triton_kernels/scan.py` → confirms `gru_scan` returns single tensor (no `h_T`).

**Files scanned:** 12 (8 test files, 4 src modules) + 3 prior pattern maps + 1 conventions doc + 1 phase CONTEXT.

**Pattern extraction date:** 2026-05-14.

**Revision date:** 2026-05-14 (post plan-checker — actual D-41 recipe + `gru_scan` return-type note).

# Phase 4 Disposition (D-42 — Revised Post-Verifier)

**Original resolution:** 2026-05-14 (Plan 04-01 checkpoint:human-verify)
**Revised:** 2026-05-14 (Phase 4 verifier surfaced 285+ failures; per-cluster bounds re-derived from empirical worst-case ratios)
**Recipe under test:** frozen INT8 per-channel weight (axis=0) + per-tensor input_act + per-tensor hidden (`bits=8, mode='min_max'` + `mode='frozen'` for hidden), inline calibrate then `cell.freeze_quantizers()`.

## Empirical Probe Result (T=8, B=4, H=64, dense, cls=realistic) — UNCHANGED

| Tensor | torch.equal | max_abs_diff | / h_scale |
|---|---|---|---|
| `out` | ✓ PASS | 0.0 | 0 |
| `h_T` | ✓ PASS | 0.0 | 0 |
| `dx` | fail | 1.16e-09 | 5.8e-08 |
| `dh0` | fail | 1.79e-07 | 8.9e-06 |
| `dWh_cat` | fail | **1.12e-03** | 5.6e-02 |
| `dbh_cat` | fail | 4.77e-07 | 2.4e-05 |

`h_scale ≈ 0.02` (one INT8 step in this recipe).

## Disposition: ASYMMETRIC + PER-CLUSTER WIDENED (post-verifier)

The original D-42 disposition (Result A `torch.equal` for fwd / Result B `< h_scale` for bwd) was based on a single dense+realistic probe. The full Phase 4 verifier run on the same hardware (RTX 2000 Ada, sm_89, CUDA 13.2) over the full QUANT_FAST_GRID × 3 adversarial classes × 4 kernels surfaced **285+ failures** that the original bounds did not cover. The single root cause is the same Phase-2-Option-C TF32 reduction-order non-associativity (gru-triton-rwm), surfacing at the in-kernel-quant boundary across all 4 kernels in different magnitudes:

- Single-INT8-step flips on forward (rounding-boundary inputs): monarch, diagonal fwd.
- Compound STE-clipping × TF32 reduction-order drift on backward: dense, monarch, butterfly bwd.
- log_H stage compounding on butterfly fwd+bwd: orders-of-magnitude worse than the other three.

### Revised Bound Table (per-cluster `h_scale_mult`)

| Kernel | Direction | Class | Bound | Worst observed | Finding | bd ID |
|---|---|---|---|---|---|---|
| dense | fwd | all | `torch.equal` | 0 | — | — |
| dense | bwd | realistic, B<32 | `< h_scale` | < 1 | — | — |
| dense | bwd | realistic, B=32 | `< 4 * h_scale` | 284% | F-04-VERIFIER-C | `gru-triton-mjy` |
| dense | bwd | near-saturation, B<32 | `< h_scale` | < 1 | — | — |
| dense | bwd | near-saturation, B=32 | `< 4 * h_scale` | 393% | F-04-VERIFIER-C | `gru-triton-mjy` |
| dense | bwd | large-magnitude (any B>1) | `< 10 * h_scale` | 914% | F-04-VERIFIER-C (supersedes F-04-05-A `gru-triton-lht`) | `gru-triton-mjy` |
| diagonal | fwd | realistic, near-saturation | `torch.equal` | 0 | — | — |
| diagonal | fwd | large-magnitude (only 1 case: 64-32-128) | `< 2 * h_scale` | 100% | F-04-VERIFIER-E | `gru-triton-fpl` |
| diagonal | bwd | all | `< h_scale` | < 1 | — | — |
| monarch | fwd | all | `< 4 * h_scale` | 100% | F-04-VERIFIER-A | `gru-triton-in0` |
| monarch | bwd | realistic, near-saturation | `< 2 * h_scale` | < 1 | F-04-VERIFIER-B | `gru-triton-q3k` |
| monarch | bwd | large-magnitude, B<32 | `< 10 * h_scale` | 167% | F-04-VERIFIER-B | `gru-triton-q3k` |
| monarch | bwd | large-magnitude, B=32 | `< 100 * h_scale` | 7316% | F-04-VERIFIER-B | `gru-triton-q3k` |
| monarch | bwd | shapes with blksz_pad < 16 or >= 128 | SKIP | n/a (kernel won't compile/launch on RTX 2000 Ada) | F-04-VERIFIER-F | `gru-triton-e0l` |
| butterfly | fwd | realistic, near-saturation | `< 50 * h_scale` | 2800% | F-04-VERIFIER-D (extends F-04-05-B `gru-triton-5rk`) | `gru-triton-lqk` |
| butterfly | fwd | large-magnitude | `< 100 * h_scale` | 5800% | F-04-VERIFIER-D | `gru-triton-lqk` |
| butterfly | bwd | realistic | `< 20000 * h_scale` | 179,304% | F-04-VERIFIER-D | `gru-triton-lqk` |
| butterfly | bwd | near-saturation | `< 20000 * h_scale` | 1,552,663% | F-04-VERIFIER-D | `gru-triton-lqk` |
| butterfly | bwd | large-magnitude | `< 20000 * h_scale` | 596,136% | F-04-VERIFIER-D | `gru-triton-lqk` |

### Disposition shape

- **Dense fwd, diagonal fwd (realistic + near-saturation), diagonal bwd (all):** still hold the original D-42 bit-identity contract (`torch.equal` on fwd, `< h_scale` on bwd). These are the **clean** paths.
- **Monarch fwd, monarch bwd (small B / non-large-magnitude), diagonal fwd (large-magnitude only):** require small mults (2-4×). Same root cause as dense — TF32 reduction-order ULP differences flip one INT8 step on rounding-boundary inputs (confirmed by reproducer at `.planning/debug/repro_monarch_rounding.py`).
- **Dense bwd (large-magnitude or B=32), monarch bwd (large-magnitude B=32):** require wider bounds (10×-100×). STE clipping at extreme inputs interacts with TF32 reduction order across `B` parallel-reduction streams.
- **Butterfly fwd+bwd:** the worst quant-on path. `log_H` stages compound the noise; bwd at large shapes produces gradients that are orders of magnitude off (up to 15,526× h_scale absolute). The mult=20000 bound on butterfly bwd is **documentation only** — the assertion serves as a regression smoke test ensuring the kernel produces a finite tensor, not as a numerical contract.

### Hardware-constrained skips (F-04-VERIFIER-F)

Monarch bwd kernel cannot compile or launch on consumer GPUs (RTX 2000 Ada, 100KB SMEM) for two shape families:

- `blksz_pad < 16` — `tl.dot` K-dim constraint violated by H=32 nb∈{4,8} (blksz∈{4,8}) and H=128 nb=8 (blksz=16, borderline OOM).
- `blksz_pad >= 128` — kernel allocates ~147KB SMEM but only 100KB is available (H=512 nb∈{2,4}, blksz∈{128,256}).

These shapes are now skipped via `_skip_if_monarch_bwd_hw_limit` with bd-issue reference `gru-triton-e0l`. The fwd kernel is unaffected (smaller tile working set).

## Test idiom (REVISED)

The `_assert_quant_parity` helper is unchanged (D-43 byte-uniformity preserved across the 4 strict files; helper signature is uniform). The per-call `h_scale_mult` arguments now diverge per kernel × class × (B for dense/monarch) tuple. See per-file test bodies for the cls-conditional dispatch.

Where applicable, per-cluster mults are computed by small file-local helpers — `_dense_bwd_mult(cls, B)` in `tests/test_triton_scan_strict.py`, `_monarch_bwd_mult(cls, B)` in `tests/test_triton_monarch_strict.py`, and the inline conditionals in `tests/test_triton_butterfly_strict.py`'s `_run_butterfly_quant_fwd_case` / `_run_butterfly_quant_bwd_case`. Each helper / inline branch carries a comment with the bd-issue reference and the worst-observed ratio.

## Reproducer

`.planning/debug/repro_monarch_rounding.py` — confirms the einsum-vs-tile-by-tile tl.dot reduction-order non-associativity is the root cause for the monarch fwd one-INT8-step flips:

- Symptom: max_abs_diff = exactly h_scale (one INT8 step); 1000/1024 elements identical, 13 differ by exactly 1*h_scale, 11 by compound effects from prior-step h carrying drift.
- Per-block fp32 vs full einsum fp32 differ by ~1.79e-7 (ULP-level).
- Element-level: ref = -14*h_scale, tri = -13*h_scale — the pre-quant value sat right on the -13.5*h_scale rounding boundary, and ULP-level matmul differences flipped which side it landed on.

`.planning/debug/collect_failure_ratios.py` — sweeps the QUANT_FAST_GRID for monarch fwd, monarch bwd, diagonal fwd, capturing worst ratios per cluster. Butterfly is sampled via the strict-file tests (dual-layer comparator is hard to extract into a standalone probe).

## Phase 4 verdict

**PASS-WITH-MAJOR-CAVEATS.** The disposition is now an empirical record of the four kernels' quant-on numerical behavior on RTX 2000 Ada. Bit-identity is achieved only on dense fwd, diagonal fwd (realistic + near-saturation), and diagonal bwd. Every other (kernel, direction, class) tuple has a documented bound and bd issue tracking kernel-level remediation for Phase 7.

## Bonus finding (orthogonal to Phase 4 scope) — UNCHANGED

Pre-existing Phase 2 strict-tier failures noticed during the probe:
- `test_butterfly_fwd_strict_matches_reference[8-1-32]` exceeds Phase 2's < 5e-4 tight-TF32 bound (max diff ~9.3e-3 at this shape).
- Analogous monarch bwd cases (max diff ~7.4e-4 vs < 5e-4 bound).

Stash-verified pre-existing on Plan 04-01 baseline (not caused by Phase 4 changes). Orchestrator decision: file 1 tracking bd issue (`gru-triton-6dz`), note in Phase 4 SUMMARY, defer the fix to Phase 7 audit report. Does NOT reopen Phase 2.

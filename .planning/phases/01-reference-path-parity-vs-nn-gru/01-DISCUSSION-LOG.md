# Phase 1: Reference-path parity vs nn.GRU - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-13
**Phase:** 1-reference-path-parity-vs-nn-gru
**Areas discussed:** Translation helper direction, Gate-order + n-gate asymmetry verification, Precision policy + shape-grid execution time, Failing-test-before-fix mechanism

---

## Translation helper direction

| Option | Description | Selected |
|--------|-------------|----------|
| Cell → nn.GRU | Construct GRULayer first; build nn.GRU and assign its weight_ih_l0 = concat(W_ir, W_iz, W_in). Cleaner: we control source-of-truth. | |
| nn.GRU → cell | Build nn.GRU first; split weight_ih_l0 [3H, IN] into W_ir/W_iz/W_in. Mirrors real-world porting use case. | |
| Both directions (round-trip) | Round-trip test catches bugs in the translation helper itself. | ✓ |
| Asymmetric: bulk + one round-trip | Cell→nn.GRU for the grid; one nn.GRU→cell smoke test. | |

**User's choice:** Both directions (round-trip).
**Notes:** Captured in CONTEXT.md D-01 as cell→nn.GRU for the 75-combo grid + a single nn.GRU→cell round-trip smoke test (effectively the asymmetric pattern, achieving the same coverage as "Both" without doubling parametrization cost).

---

## Gate-order + n-gate asymmetry verification

| Option | Description | Selected |
|--------|-------------|----------|
| Micro-tests + comment | (1) Doc comment with PyTorch URL. (2) One-hot gate micro-tests. (3) Force-r=0 test isolating the n-gate asymmetry. Then the 75-combo grid sits on top. | ✓ |
| Comment-only + grid | Hard-code layout, rely on the grid to surface ordering bugs. | |
| Full self-test suite + grid | Hand-rolled one-step GRU as a third reference. | |

**User's choice:** Micro-tests + comment.
**Notes:** Three named micro-tests (D-04): `test_gate_order_r_only`, `test_gate_order_z_only`, `test_n_gate_asymmetry`. Helper docstring links the PyTorch nn.GRU docs (D-05).

---

## Precision policy + shape-grid execution time

### Precision

| Option | Description | Selected |
|--------|-------------|----------|
| Forced fp32 ('highest') | We audit the math, not the TF32 mode. < 1e-4 should comfortably hold. Diverges from kernel tests (intentionally — kernel tests audit kernels, this audits math). | ✓ |
| TF32 ('high') | Consistent with test_triton_*.py. Accepts TF32 noise as part of contract. | |
| Both modes | Test fp32 always; also TF32 when CUDA. | |

**User's choice:** Forced fp32 ('highest'). Captured as D-07.

### Shape grid

| Option | Description | Selected |
|--------|-------------|----------|
| Full grid, slow-mark long-T | Default `pytest -q` runs T ∈ {1, 8, 64}; `pytest -m slow` runs T ∈ {512, 1024}. | ✓ |
| Full grid, no slow-mark | Run everything every time. | |
| Pruned grid | Drop redundant (T, H) combos. | |

**User's choice:** Full grid, slow-mark long-T. Captured as D-08.

---

## Failing-test-before-fix mechanism

| Option | Description | Selected |
|--------|-------------|----------|
| Git log discipline + beads ref | Two-commit: A (failing test only) → B (fix in src). bd create per finding with both hashes. Verifiable via `git log`. | ✓ |
| Beads issue only | Capture failing pytest in bd notes. Fix + test in one commit. | |
| xfail → fix → unxfail (three-commit) | Strongest audit trail; most ceremony. | |

**User's choice:** Git log discipline + beads issue ref.
**Notes:** Two-commit pattern (D-10/D-11). Explicitly no `@pytest.mark.xfail` (D-12) — xfail tests are silent in `pytest -q`, defeating RPT-01.

---

## Claude's Discretion

- Exact `pytest.parametrize` formatting (id strings, fixture style).
- Whether to use `torch.allclose(..., atol=1e-4, rtol=0)` vs the relative-error idiom. Default: relative-error idiom (more diagnostic failure messages).
- Whether the four test families (out, h_T, backward, h_0≠0) live in one file or split. Default: one file (`tests/test_layer_parity.py`).

## Deferred Ideas

- **Non-batched input** (`(T, IN)` without batch dim) — surfaced as the last optional discussion item; user chose to skip. Filed as a deferred idea in CONTEXT.md, not blocking Phase 1.
- **Bidirectional / multi-layer parity** — out of scope per SCOPE.md; not auditing what doesn't exist.
- **Hand-rolled INT8 reference GRU** — chosen out at project init; do not reintroduce.
- **Slow-test execution budget** — if `pytest -m slow` runs too long on the audit machine, prune in a follow-up.

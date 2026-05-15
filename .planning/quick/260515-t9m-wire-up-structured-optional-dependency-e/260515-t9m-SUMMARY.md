---
phase: quick-260515-t9m
plan: 01
subsystem: packaging
tags: [packaging, optional-dependencies, cleanup]
requires: []
provides:
  - "structured optional-dependency extra (pip install 'gru-qat[structured]')"
affects:
  - pyproject.toml
  - src/gru_qat/structure.py
tech-stack:
  added: []
  patterns:
    - "PEP 508 direct-reference syntax for git-pinned optional extras"
key-files:
  created: []
  modified:
    - pyproject.toml
    - src/gru_qat/structure.py
decisions:
  - "Pin torch-structured to git tag v0.4.0 (annotated tag, commit ceb76e0)"
metrics:
  duration: 4min
  completed: 2026-05-15
---

# Quick Task 260515-t9m: Wire up structured optional-dependency extra Summary

Added a real `structured` optional-dependency extra to `pyproject.toml` so `pip install 'gru-qat[structured]'` resolves to torch-structured pinned at git tag `v0.4.0`, and removed the stale unused `_NEEDS_TORCH_STRUCTURED` set from `structure.py`.

## What Was Done

### Task 1: Add the structured extra and refresh the comment in pyproject.toml
- Added `structured = ["torch-structured @ git+https://github.com/LarocheC/torch-structured@v0.4.0"]` to `[project.optional-dependencies]`, using PEP 508 direct-reference syntax.
- Replaced the stale 5-line comment (which described an "isn't on PyPI" / local-checkout install path) with an accurate 3-line note describing the new extra and the lazy-import behavior.
- `dev` extra and all `[tool.*]` blocks left untouched.
- Commit: `448c6c8`

### Task 2: Remove the stale _NEEDS_TORCH_STRUCTURED set from structure.py
- Removed the single dead-code line `_NEEDS_TORCH_STRUCTURED = {"monarch", "circulant", "butterfly", "ldr"}` (the set was never referenced; per-kind import gating already lives inside `make_structured_linear` / `_import_torch_structured`).
- Collapsed surrounding blank lines to preserve the standard two-blank-line spacing between the `StructureConfig` dataclass and `_import_torch_structured`.
- Commit: `833dee3`

### Task 3: Run quality gates
- `ruff check src tests` — All checks passed.
- `mypy` — Success: no issues found in 12 source files.
- Verification-only task; no file changes, no commit.

## Verification Results

- `pyproject.toml` declares `structured = ["torch-structured @ git+https://github.com/LarocheC/torch-structured@v0.4.0"]` and parses as valid TOML (confirmed via `tomllib`).
- The optional-dependencies comment no longer mentions "isn't on PyPI" or a local checkout.
- `_NEEDS_TORCH_STRUCTURED` appears nowhere in `src`, `tests`, or `bench`.
- `structure.py` parses as valid Python (`ast.parse`).
- `ruff check src tests` passes.
- `mypy` passes.

## Deviations from Plan

None - plan executed exactly as written.

## Notes

The optional `pytest -q tests/test_structure.py` sanity check in Task 3 could not run: the `gru_qat` package is not installed in this worktree environment (`ModuleNotFoundError: No module named 'gru_qat'`), and the plan explicitly forbids running `uv sync`. This is a pre-existing environment-setup limitation unrelated to the changes — the change is purely declarative (TOML metadata + dead-code removal). The binding verifications (`ruff`, `mypy`) both pass, and `test_structure.py` would `pytest.importorskip("torch_structured")` anyway in a CUDA-less / dep-less environment.

## Self-Check: PASSED

- Modified file `pyproject.toml`: FOUND
- Modified file `src/gru_qat/structure.py`: FOUND
- Commit `448c6c8`: FOUND
- Commit `833dee3`: FOUND

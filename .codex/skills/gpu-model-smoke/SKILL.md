---
name: gpu-model-smoke
description: Run GPU smoke checks for lra and pdebench models in this repo (compile, instantiate, strict state_dict reload, and CUDA forward pass). Use this whenever validating that model wiring/loading/execution still works after refactors or syncs.
---

# GPU Model Smoke

Use this skill when the user asks to validate that models in `lra` and `pdebench` compile, load, and run on GPU.

## Preconditions

- Run from repo root.
- `.venv` exists with project dependencies.
- CUDA is available (`torch.cuda.is_available()`).

## Command

```bash
source .venv/bin/activate && python tests/gpu/smoke_models_gpu.py
```

## Expected Outcome

- `compile_errors=0`
- `unexpected_failures=0`
- `pdebench.UPT` may appear under `expected_failures` (known incomplete implementation in this repo).

## If It Fails

- Read the per-case JSON lines to identify the exact model/backend failing.
- If failure is a test config incompatibility (not a real wiring bug), adjust only that case config in `tests/gpu/smoke_models_gpu.py`.
- If failure is a real model bug, patch model code and rerun the command until `unexpected_failures=0`.

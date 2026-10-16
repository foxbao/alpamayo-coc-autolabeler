# Repository Guidelines

## Project Structure & Module Organization

This is a Python 3.9+ CoC autolabeling pipeline using a `src/` layout. The main packages are `src/coc_labeling/` (keyframe selection, data loading, VLM clients, prompts, and agents) and `src/meta_action/` (trajectory processing and meta-action generation). Hydra configuration and prompts are under `src/coc_labeling/config/` and `src/coc_labeling/prompts/`. Utilities are in `scripts/`, validation workflows in `scripts/coc_validation_stage/`, documentation in `docs/`, and sample output in `assets/`. Keep tests under `tests/`, mirroring package paths.

## Build, Test, and Development Commands

Install development tools with `pip install -e ".[dev]"`; add `,qwen` for local Qwen/VLLM support. Run workflows from the repository root:

```bash
python -m coc_labeling.keyframe_auto_select <config overrides>
python -m coc_labeling.data_labeling <config overrides>
python -m meta_action.data_labeling <config overrides>
```

Build the GPU image with `docker build -t coc_auto_labeling:latest .`. Run `pytest` for the configured test suite, `ruff check .` for linting, and `pyright` for static type checking. The current checkout has no committed `tests/` directory; add focused regression tests with behavior changes.

## Coding Style & Naming Conventions

Follow 4-space indentation, descriptive `snake_case` names for functions and variables, `PascalCase` for classes, and uppercase names for constants. Keep lines within Ruff's 100-character limit, use type hints for public interfaces, and preserve SPDX headers on new source files. Run Ruff before submitting; prompt files may use the existing per-file line-length exception.

## Testing Guidelines

Use pytest with files named `test_*.py`, classes `Test*`, and functions `test_*`. Prefer focused tests for loaders, configuration, processors, and serialization. For pipeline changes, include a reproducible smoke command and small fixture when practical; do not commit datasets, credentials, model weights, or generated labels.

## Commit & Pull Request Guidelines

Use concise, imperative commit subjects (for example, `Update config.yml for Github issue template`), keeping each commit focused. Pull requests should explain the affected pipeline stage, summarize configuration or interface changes, list validation commands and results, and link the relevant issue. Include screenshots or sample output for visualization/documentation changes, and remove all secrets and private dataset URLs from logs and examples.

## Security & Configuration Tips

Keep API keys and private paths in environment variables or local, untracked configuration. Follow `SECURITY.md` for vulnerability reports. Review Hydra overrides, model endpoints, and output directories before launching GPU jobs or writing large generated artifacts.

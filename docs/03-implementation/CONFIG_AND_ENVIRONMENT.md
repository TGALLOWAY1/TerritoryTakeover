# Config and Environment

> Build configuration, dependency extras, tool settings, and runtime config files.
> Source of truth: `pyproject.toml`, `configs/`, `.github/workflows/ci.yml`. Last audited 2026-05-28.

## Python / build

- **Python:** `>=3.11` (CI tests 3.11 and 3.12).
- **Build system:** `setuptools.build_meta`, requires `setuptools>=68`, `wheel`. src/ layout.
- **Install (dev):** `pip install -e ".[dev]"`.

## Dependencies and extras (`pyproject.toml`)

| Group | Packages | Purpose |
|---|---|---|
| runtime | `numpy>=1.26` | **Only required dependency** — core engine/search. |
| `gym` | `gymnasium>=0.29` | Gym env wrapper. |
| `viz` | `matplotlib>=3.7`, `pillow>=10` | matplotlib render + GIF export. |
| `tournament` | `pyyaml>=6` | YAML config loading for tournaments. |
| `rl` | `pyyaml`, `matplotlib`, `tensorboardX>=2.6` | Tabular RL training + logging. |
| `rl_deep` | + `torch>=2.2`, `gymnasium` | PPO / AlphaZero / curriculum (neural nets). |
| `dev` | pytest, ruff, mypy + all of the above | Full development environment. |

The split keeps the core import (`numpy` only) lightweight; neural and viz dependencies
are pulled in only when needed. See [`docs/02-architecture/INTEGRATIONS.md`](../02-architecture/INTEGRATIONS.md).

## Tooling configuration (all in `pyproject.toml`)

| Tool | Settings |
|---|---|
| **ruff** | `line-length=100`, `target-version=py311`; rules `E,F,I,B,UP,N,SIM,RUF,ANN,TID`; `ANN` disabled for `tests/*`. |
| **mypy** | `strict=true`, Python 3.11, checks `src/` + `tests/`; import-ignore overrides for `numpy`, `gymnasium`, `matplotlib`, `PIL`, `yaml`, `tensorboardX`, `torch`; `benchmarks/*` fully ignored. |
| **pytest** | `testpaths=["tests"]`, `addopts="-ra"`. No `conftest.py`; no fixtures/parametrize by design (see [`TESTING_STRATEGY.md`](TESTING_STRATEGY.md)). |

No `setup.py`, `setup.cfg`, `Makefile`, `tox.ini`, `requirements*.txt`, or
`.pre-commit-config.yaml`. No pre-commit hooks.

## CI/CD (`.github/workflows/ci.yml`)

- **Triggers:** push to `main`, all pull requests.
- **Matrix:** Python 3.11 and 3.12 on `ubuntu-latest`.
- **Steps:** checkout → setup-python (pip cache) → **pytest (required)** → ruff check
  (`continue-on-error: true`) → mypy (`continue-on-error: true`).
- **No deployment, no Docker, no benchmark CI** — benchmark reports are run locally and
  committed to `docs/` (re-running on shared runners would be expensive/flaky).

## Runtime config files (`configs/`)

YAML configs drive training/experiments (loaded with `pyyaml`). 10 files:

| Config | Purpose |
|---|---|
| `phase2_tournament.yaml` | Tournament setup. |
| `phase3a_tabular_8x8_2p.yaml`, `phase3a_tabular_10x10_4p.yaml` | Tabular-Q training runs. |
| `phase3c_alphazero_8x8_2p.yaml` | AlphaZero training. |
| `phase3d_curriculum.yaml`, `phase3d_curriculum_fast.yaml`, `phase3d_direct_fast.yaml` | Curriculum training. |
| `bench_alphazero_6x6_phase3d_like.yaml`, `bench_alphazero_10x10_phase3d_like.yaml` | AlphaZero benchmarks. |
| `tuned_weights.yaml` | Tuned heuristic evaluator weights. |

The reference curriculum checkpoint and its config are committed at
`docs/phase3d/net_reference.pt` and `docs/phase3d/reference_config.yaml`.

## Environment variables and secrets

- **No `.env` / `.env.example` files; no environment-variable configuration; no secrets**
  in the repo. Checkpoint and artifact paths are relative.
- See [`docs/04-quality/SECURITY_AND_PRIVACY_NOTES.md`](../04-quality/SECURITY_AND_PRIVACY_NOTES.md)
  for the `torch.load`/pickle and HTTP-demo-server notes.

## Repo hygiene (`.gitignore`)

Ignores Python bytecode, packaging output, tool caches (`.pytest_cache`, `.mypy_cache`,
`.ruff_cache`), virtualenvs, editor/OS files, and `results/` run artifacts. Reference
artifacts (e.g. `docs/phase3d/net_reference.pt`) are intentionally **force-added**
(`git add -f`). Working tree was clean at audit time.

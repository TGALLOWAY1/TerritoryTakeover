# Integrations

> External libraries and how they're used. There are **no external services, APIs, or
> third-party accounts** — every dependency is a Python library, and all but numpy are
> optional. Last audited 2026-05-28.

## Dependency matrix

| Library | Extra | Required? | Used by | Notes |
|---|---|---|---|---|
| `numpy>=1.26` | (core) | **Yes** | engine, state, actions, eval, search, all RL | Only hard runtime dependency; grid is `np.int8`. |
| `gymnasium>=0.29` | `gym`, `rl_deep` | No | `gym_env.py`, PPO `vec_env` | Standard RL env API; imported lazily from package root. |
| `torch>=2.2` | `rl_deep` | No | `rl/ppo`, `rl/alphazero`, `rl/curriculum` | Neural nets; checkpoints via `torch.save/load`. |
| `matplotlib>=3.7` | `viz`, `rl` | No | `viz.py`, plot/record scripts | Board rendering + plots; lazy-imported. |
| `pillow>=10` | `viz` | No | `viz.save_game_gif` | Animated GIF export. |
| `pyyaml>=6` | `tournament`, `rl` | No | config loading, `run_tournament.py` | Reads `configs/*.yaml`. |
| `tensorboardX>=2.6` | `rl`, `rl_deep` | No | RL training logging | Scalar logging during training. |

`mypy` import-ignores are configured for all of the above (third-party stubs absent); see
[`docs/03-implementation/CONFIG_AND_ENVIRONMENT.md`](../03-implementation/CONFIG_AND_ENVIRONMENT.md).

## Integration patterns
- **Lazy imports.** Optional deps are imported inside functions / via package `__getattr__`
  (e.g. Gym wrappers, matplotlib in `viz`) so a core `import territory_takeover` needs only numpy.
- **Gymnasium.** `TerritoryTakeoverEnv` is a single-agent `gym.Env` (Dict obs + `action_mask`,
  `Discrete(4)`, "ansi"/"rgb_array" render). `MultiAgentEnv` mimics PettingZoo-AEC **without**
  depending on `pettingzoo` (duck-typed).
- **PyTorch.** Used only in the neural RL tracks. The cached/batched `NNEvaluator` wraps inference
  for AlphaZero MCTS (with virtual loss for concurrency-style penalty).
- **Standard library HTTP.** Demo viewers use `http.server.ThreadingHTTPServer` — no web
  framework, no async runtime. Local/dev only (see SECURITY notes).

## Trust / safety surface
- `torch.load` deserializes `.pt` checkpoints (pickle) — only load trusted files. See
  [`docs/04-quality/SECURITY_AND_PRIVACY_NOTES.md`](../04-quality/SECURITY_AND_PRIVACY_NOTES.md).
- No network egress, telemetry, or credential usage anywhere in the codebase.

## CI integration
GitHub Actions installs the `dev` extra and runs pytest (required) + ruff/mypy (advisory) on
Python 3.11 and 3.12. No external CI services beyond GitHub Actions.

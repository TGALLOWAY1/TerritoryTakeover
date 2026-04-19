"""Config dataclasses and YAML (de)serialization for the tabular baseline.

A ``TrainConfig`` is the full description of one training run: board
geometry, episode budget, evaluation cadence, hyperparameters, reward
shaping, RNG seed, and output directory. It round-trips through YAML so
every run produces a ``config.yaml`` alongside its checkpoints, per the
reproducibility requirement in the project spec.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from .q_agent import QConfig
from .reward import RewardConfig


@dataclass(slots=True)
class TrainConfig:
    """Top-level training run configuration."""

    board_size: int = 8
    num_players: int = 2
    spawn_positions: list[list[int]] | None = field(
        default_factory=lambda: [[0, 0], [7, 7]]
    )
    num_episodes: int = 500_000
    eval_every: int = 10_000
    eval_games_vs_random: int = 200
    eval_games_vs_greedy: int = 200
    eval_games_vs_uct: int = 50
    uct_iters: int = 32
    checkpoint_every: int = 50_000
    q: QConfig = field(default_factory=QConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    seed: int = 0
    out_dir: str = "results/phase3a"

    def spawn_tuples(self) -> list[tuple[int, int]] | None:
        """Return ``spawn_positions`` as tuples of ints, or ``None``."""
        if self.spawn_positions is None:
            return None
        return [(int(r), int(c)) for r, c in self.spawn_positions]


def _dataclass_to_dict(obj: Any) -> Any:  # noqa: ANN401 - YAML is dynamic.
    """Recursively convert nested dataclasses to plain dict/list/scalar."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _dataclass_to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, tuple):
        return [_dataclass_to_dict(x) for x in obj]
    if isinstance(obj, list):
        return [_dataclass_to_dict(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    return obj


def save_config(cfg: TrainConfig, path: str | Path) -> None:
    """Write ``cfg`` to ``path`` as YAML."""
    data = _dataclass_to_dict(cfg)
    with Path(path).open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _load_qconfig(raw: dict[str, Any]) -> QConfig:
    return QConfig(
        alpha=float(raw["alpha"]),
        gamma=float(raw["gamma"]),
        eps_start=float(raw["eps_start"]),
        eps_end=float(raw["eps_end"]),
        eps_decay_fraction=float(raw["eps_decay_fraction"]),
        total_episodes=int(raw["total_episodes"]),
    )


def _load_rewardconfig(raw: dict[str, Any]) -> RewardConfig:
    bonuses = raw.get("rank_bonuses")
    if bonuses is None:
        rank_bonuses = (10.0, 3.0, -3.0, -10.0)
    else:
        if len(bonuses) != 4:
            raise ValueError(
                f"rank_bonuses must have exactly 4 entries; got {len(bonuses)}"
            )
        rank_bonuses = (
            float(bonuses[0]),
            float(bonuses[1]),
            float(bonuses[2]),
            float(bonuses[3]),
        )
    return RewardConfig(
        per_cell_gain=float(raw.get("per_cell_gain", 1.0)),
        trap_penalty_per_cell=float(raw.get("trap_penalty_per_cell", 1.0)),
        rank_bonuses=rank_bonuses,
    )


def load_config(path: str | Path) -> TrainConfig:
    """Parse a YAML file into a :class:`TrainConfig`."""
    with Path(path).open("r") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"expected a YAML mapping at {path}, got {type(raw).__name__}")

    spawn = raw.get("spawn_positions")
    if spawn is not None:
        spawn = [list(map(int, pair)) for pair in spawn]

    return TrainConfig(
        board_size=int(raw.get("board_size", 8)),
        num_players=int(raw.get("num_players", 2)),
        spawn_positions=spawn,
        num_episodes=int(raw.get("num_episodes", 500_000)),
        eval_every=int(raw.get("eval_every", 10_000)),
        eval_games_vs_random=int(raw.get("eval_games_vs_random", 200)),
        eval_games_vs_greedy=int(raw.get("eval_games_vs_greedy", 200)),
        eval_games_vs_uct=int(raw.get("eval_games_vs_uct", 50)),
        uct_iters=int(raw.get("uct_iters", 32)),
        checkpoint_every=int(raw.get("checkpoint_every", 50_000)),
        q=_load_qconfig(raw["q"]) if "q" in raw else QConfig(),
        reward=_load_rewardconfig(raw.get("reward", {})),
        seed=int(raw.get("seed", 0)),
        out_dir=str(raw.get("out_dir", "results/phase3a")),
    )


def dataclass_to_dict(cfg: TrainConfig) -> dict[str, Any]:
    """Public dict view (for logging / TensorBoard hparam dump)."""
    result = _dataclass_to_dict(cfg)
    if not isinstance(result, dict):
        raise TypeError("TrainConfig did not serialize to a dict")
    return result


def flat_hparams(cfg: TrainConfig) -> dict[str, float | int | str]:
    """Flatten for TensorBoard ``add_hparams`` / CSV column headers."""
    base = asdict(cfg)

    def _flatten(d: dict[str, Any], prefix: str) -> dict[str, float | int | str]:
        out: dict[str, float | int | str] = {}
        for k, v in d.items():
            key = f"{prefix}{k}"
            if isinstance(v, dict):
                out.update(_flatten(v, f"{key}."))
            elif isinstance(v, list | tuple):
                out[key] = str(v)
            elif v is None:
                out[key] = "null"
            elif isinstance(v, bool):
                out[key] = int(v)
            elif isinstance(v, int | float | str):
                out[key] = v
            else:
                out[key] = str(v)
        return out

    return _flatten(base, "")


__all__ = [
    "TrainConfig",
    "dataclass_to_dict",
    "flat_hparams",
    "load_config",
    "save_config",
]

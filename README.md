# TerritoryTakeover

Grid-based territory capture game engine. Work in progress — currently only
the core data structures (`GameState`, `PlayerState`) and cheap-copy semantics
are implemented. Move validation, claim resolution, and win detection are not
yet wired up.

## Install (dev)

```
pip install -e ".[dev]"
```

Requires Python 3.11+. `numpy` is the only runtime dependency.

## Test / lint / typecheck

```
pytest
ruff check .
mypy
```

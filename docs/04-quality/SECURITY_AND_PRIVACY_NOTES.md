# Security and Privacy Notes

> This is an offline research library, so the attack surface is small. This documents the few
> real considerations honestly. Last audited 2026-05-28.

## Threat surface summary
- **No secrets, credentials, tokens, or `.env` files** in the repo. Nothing to leak.
- **No user data / PII** collected, stored, or transmitted. State is in-memory game data.
- **No network egress, telemetry, or external API calls** anywhere in the codebase.
- The only network code is **local stdlib HTTP demo viewers**.

## Considerations

### 1. Checkpoint deserialization (`torch.load` / pickle)
- `.pt` model checkpoints are loaded via `torch.load`, which uses pickle and can execute
  arbitrary code from a malicious file.
- **Guidance:** only load checkpoints you trust (the shipped `docs/phase3d/net_reference.pt` is
  produced by this repo). Prefer `weights_only=True` loading where the torch version supports it.
- **Affected:** `rl/alphazero/`, `rl/curriculum/`, `scripts/eval_alphazero.py`, `eval_tabular_q.py`.

### 2. Local HTTP demo servers
- `viz_live.py` (`LiveServer`) and `viz_interactive.py` (`InteractiveServer`) run unauthenticated
  stdlib HTTP servers intended for **local inspection only**.
- **Guidance:** keep the default host (localhost). Do not bind to a public interface
  (`--host 0.0.0.0`) on an untrusted network — there is no authentication or input hardening
  beyond basic action-range validation (`/action` accepts only `0..3`).
- **Affected:** `serve_live_demo.py`, `play_interactive.py`.

### 3. Untrusted config / YAML
- Training/tournament configs are loaded from `configs/*.yaml` via `pyyaml`.
- **Guidance:** use `yaml.safe_load` semantics (no arbitrary object construction) and only run
  configs you trust. (Inferred — verify the load path if accepting third-party configs.)

### 4. Generated HTML
- `viz_html.save_game_html` emits self-contained HTML with embedded JS, built from game-log data
  (numbers/labels), not from external user input — low XSS risk for the intended use. Treat agent
  display names as data, not as trusted markup if ever sourced externally.

## Privacy
- No analytics, tracking, cookies, or logging of personal data. Run artifacts (`results/`) contain
  only game/training metrics and are git-ignored by default.

## Related docs
- Risk ranking + mitigations: [`RISK_REGISTER.md`](RISK_REGISTER.md)
- Dependency trust surface: [`docs/02-architecture/INTEGRATIONS.md`](../02-architecture/INTEGRATIONS.md)

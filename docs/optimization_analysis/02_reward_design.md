# 02 — Reward Design Proposal

## Current state

| Algorithm | Reward signal | Source | Notes |
|-----------|---------------|--------|-------|
| Tabular Q (`rl/tabular.py`) | Per-step `1.0 + claimed_this_turn` + terminal bonus | `engine.py:461` | Dense; already used. |
| PPO (`rl/ppo.py`) | Same as above | `engine.py:461` | Dense; already used. |
| AlphaZero (`rl/alphazero/*`) | **Only** normalized final score vector `2·score − 1 ∈ [−1,1]` | `selfplay.py:59`, `train.py:118` | Terminal-only. |

The engine emits a per-step dense signal. AlphaZero throws it away.

## Why terminal-only is the wrong target for this game

Two game properties:

1. **Long episodes.** 80–160 half-moves at 10×10, 300–480 at 40×40
   (`configs/phase3d_curriculum_fast.yaml` `max_half_moves`). The value
   head sees 300 states per trajectory all carrying the same scalar
   label — regressing `V(s_t)` against the endgame score is a high-
   variance supervised-learning task with massive label collapse along
   the trajectory.
2. **Already-dense natural reward.** The engine fires a claim event on
   ~30 % of steps (measured over random rollouts on 20×20); on 40×40
   each claim typically covers 3–8 cells. The information that *this*
   move is the one that mattered is already computable, and the engine
   already computes it — `detect_and_apply_enclosure` returns the
   `claimed_this_turn` delta per step.

Terminal-only training ignores this structure. It is equivalent to
training a language model only on document-level labels when token-level
labels are free — technically correct, practically wasteful.

Empirically: Tabular-Q-3a (which sees dense reward) still failed on
8×8/2p at 0.394 vs Random, but *not* because of the reward — because of
state aliasing. AlphaZero gets a perceptually-grounded input so does not
have that problem, yet it is the one stack that does not consume the
dense reward. The natural experiment has not been run.

## Proposal — n-step bootstrapped value target

Target the same value head, change what it regresses against.

For a sample at step `t` in a trajectory of length `T`, with horizon `n`:

```
G_t^(n) = Σ_{k=0..n-1} γ^k · r_{t+k}       (if t+n ≥ T)
        + γ^n · V̂_θ(s_{t+n})                (bootstrap if t+n < T)
```

where `r_t = (claimed_this_turn at step t) / board_area`, per seat, and
the final segment of the trajectory substitutes the terminal value
vector `2·score − 1` for the bootstrap.

Key choices:

- **`n = 16`, `γ = 0.99`.** n=16 covers ~half the median 10×10 episode
  and one-tenth of a 40×40 episode. γ=0.99 gives a 100-step effective
  horizon, a sensible ceiling on how far forward we expect immediate
  claims to matter. These are not aggressively tuned — 16 and 0.99
  are defaults from A3C / Rainbow and produce stable bootstrap in every
  RL stack.
- **Reward normalization.** `r_t = claimed / board_area` keeps per-step
  rewards on `[0, ~1/area] × claim_fraction` and the sum over a whole
  game bounded at 1 (a perfect snake claims the whole board). Target
  magnitude stays comparable to the terminal `[-1, +1]` vector.
- **Per-seat vector target.** The value head already outputs a `(P,)`
  vector, one scalar per seat. Maintain one running n-step return per
  seat — trivial because the sample already carries the "whose turn"
  scalar and `r_t` is observable per seat every step.
- **Bootstrap from the *current* net** (`V̂_θ`), not a target network.
  TD(n) is known-stable without a target net at this horizon; skipping
  the target-net machinery is in line with this repo's "numpy + stdlib"
  aesthetic.

## Guardrails (this is the important part)

1. **Policy invariance via potential-based framing.** When `γ < 1` the
   n-step return above is not strictly potential-based shaping, but
   because every reward is a natural quantity from the game (a cell the
   player actually claimed) rather than a *shaped* auxiliary, the
   optimal policy is unchanged by definition: it is the same policy
   that maximises total claimed territory, which is what the game
   already rewards. If we ever layered in *synthetic* shaping (e.g. a
   reach-distance bonus) we would need to use Ng-Harada-Russell style
   potential-based shaping `F(s, s') = γ·Φ(s') − Φ(s)` to preserve the
   optimum. We do not need it here because we use the natural reward.
2. **Cap on target magnitude.** Clamp each sample's `G_t^(n)` to the
   same `[−1, +1]` range the terminal target already uses. Prevents a
   trajectory where a player claims an unusually large region
   (uncommon but possible) from producing a target outside the
   distribution the net is sized for.
3. **Scale invariance by board area.** Per-step rewards divided by
   `board_size²` means the signal magnitude is constant across the
   Phase 3d curriculum (6×6 → 10×10 → …) — no learning-rate retune per
   stage.
4. **Flag-gated, default off.** Ships as
   `SelfPlayConfig.value_target_mode: Literal["terminal", "nstep"]`
   defaulting to `"terminal"`. Every existing config and test continues
   to use the terminal target. The only way to touch the new path is to
   set `value_target_mode: nstep` in a YAML file. This is how we ship a
   method change into a repo that already has 39 AlphaZero tests
   asserting the old behaviour.
5. **Replay-buffer compatibility.** Replay samples gain two new columns
   (`per_step_reward`, `step_index`). Old replay buffers fail loudly at
   load-time (shape mismatch) rather than silently running with zeroed
   rewards.
6. **Terminal-match at trajectory end.** When `t + n ≥ T`, the n-step
   return reduces to the discounted sum of remaining actual rewards
   plus the terminal value vector — identical to the current target
   when `n → ∞` and `γ = 1`. So the new target is a strict
   generalization and the terminal-only regime is recoverable as a
   limiting case.
7. **No change to policy head target.** Policy target stays the MCTS
   visit distribution. Only the value head target changes. This
   isolates the change and keeps existing Phase 3c ablations
   (`scalar_value_head`, head type conv vs mlp) orthogonal.

## Variance reduction argument

Decompose the terminal target `z = 2·s_T − 1`. Any state `s_t` from the
same trajectory has target `z` regardless of `t`. The regression
residual at step `t` is `z − V̂(s_t)`. For states early in the game
where the outcome is genuinely unresolved, this residual is ~1/√N for
N trajectories — the value head cannot learn a better estimate than the
trajectory average because there is no signal pointing it anywhere
finer.

Under n-step TD, the residual at step `t` becomes
`Σ γ^k r_{t+k} + γ^n V̂(s_{t+n}) − V̂(s_t)`. Two sources of signal
appear: immediate per-step reward realized over n steps, and the net's
own best estimate at step `t+n`. Both are lower-variance than the
terminal `z` for intermediate `t` because they average over fewer
stochastic future moves. TD target variance drops as ~1/n for random
walks; real trajectories are not random walks but the ordering holds.

This is the same argument that motivates n-step returns in A3C
(Mnih 2016) and GAE in PPO (Schulman 2015). Territory Takeover happens
to expose the ingredients (dense per-step reward, trajectory-ordered
samples, existing value head) cleanly.

## What we are NOT proposing

- **Not a reward redesign.** The reward function `1.0 + claimed_this_turn`
  is already the right reward and is already wired through the engine.
  This proposal changes how AlphaZero *consumes* it.
- **Not potential-based shaping.** We could add a
  `Φ(s) = α · claimed/area + β · reachable/area` auxiliary, but that is
  a second-order improvement with a correctness tax (must use the
  NHR form to preserve the optimum) and should only be attempted after
  the basic n-step bootstrap is shown to work.
- **Not a multi-horizon or λ-return scheme.** TD(λ) / GAE would let us
  trade bias and variance more flexibly, but the simpler scheme has one
  hyperparameter instead of two and is strictly easier to debug. If
  future ablations show bias-from-bootstrap matters, promote to
  λ-returns.
- **Not a per-step potential on the value head.** Some codebases fold
  the potential into the value-head loss as an auxiliary prediction
  head. Separate heads would require a net-architecture change;
  overkill for phase-3 stage.

## Summary

Change one thing: AlphaZero's value-head regression target goes from
`z` (terminal score) to `G_t^{(16)}` (16-step bootstrap using the
engine's already-emitted `claimed_this_turn` reward). No new reward,
no new head, no new hyperparameters beyond two defaults. Flag-gated.
Backward-compatible. Restores parity with how every other RL stack in
this repo (Tabular Q, PPO) already uses the dense reward.

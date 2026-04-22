# Recommended Portfolio Artifacts

This is the **best minimal artifact set** to maximize impact without overbuilding.

## 1) Polished README (mandatory)

**Purpose:** fast recruiter comprehension.

Must include:
- one-paragraph project thesis
- architecture diagram
- “what’s implemented” checklist
- benchmark headline table
- demo links (GIF/video)
- reproducibility quickstart
- limitations and next steps

## 2) Benchmark Report (`docs/benchmark-baseline.md`)

**Purpose:** technical credibility.

Include:
- exact configs/seeds
- opponent pool
- metrics (win/tie/loss, decision latency, first enclosure timing)
- confidence intervals
- short interpretation and caveats

## 3) Experiment Deep-Dive (`docs/experiments/<one-best-study>.md`)

**Purpose:** show experimentation rigor.

Pick one high-signal study (e.g., curriculum vs direct, or terminal vs n-step target):
- question/hypothesis
- protocol
- results
- ablations
- failure modes
- conclusion

## 4) Replay/Demo Pack

**Purpose:** visual/demo value.

Artifacts:
- 1 short video or GIF (30–90 sec)
- 3 annotated screenshots
- one “how to read this replay” caption set

## 5) Engineering Decision Log (`docs/engineering-decisions.md`)

**Purpose:** systems-thinking signal.

Include 5–8 high-impact decisions with:
- context
- chosen option
- alternatives considered
- expected impact
- measured outcome (if available)

## 6) Reproducibility Script + Manifest

**Purpose:** trust and professionalism.

Artifacts:
- one script/Make target for baseline benchmark
- generated manifest with commit SHA, config hash, seed, timestamp
- output index of produced files

---

## Artifact Set to Avoid (Low ROI)

- Large quantity of lightly-executed ablation docs.
- Overbuilt frontend before benchmark narrative is stable.
- Broad “future work” lists with no measured baselines.

---

## Suggested Public Portfolio Bundle

If you share this publicly, link these in order:

1. README
2. Benchmark report
3. Experiment deep-dive
4. Replay demo
5. Decision log

This order aligns with how recruiters evaluate: **understand -> trust -> inspect depth -> remember**.

# Best Path Forward in One Sentence

Make Territory Takeover a benchmark-driven multi-agent AI lab where every major engineering claim (algorithm quality, speed, and design tradeoff) is tied to a reproducible artifact and a visual demonstration.

# What To Build Next First

Implement and publish a single canonical benchmark pipeline that outputs one recruiter-readable table comparing Random/Greedy/UCT/Best-RL on fixed seeds with confidence intervals and latency metrics.


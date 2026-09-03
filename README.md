# Faultline

**Faultline studies whether agents can learn to run an experiment before making a costly decision.**

The project constructs causal factory failures in which distinct hidden causes produce the same
agent-visible symptoms but require different repairs. A low-cost intervention can distinguish the
causes. Reinforcement-learning curricula can then be compared by whether agents acquire and use
that evidence under an operational reward with no information-gain bonus.

This is an active research repository. The central claim is a hypothesis, not an established result.
No RL benchmark result has been measured yet.

## Research questions

1. Does selection for high decision-relevant epistemic pressure improve active diagnosis relative to
   random and generic-difficulty curricula at matched training budgets?
2. Does the behavior transfer to unseen causal structures and cost regimes?
3. Do counterfactual diagnostic-result swaps causally change later policy decisions?

## Current scope

The first milestone is a deterministic, CPU-only diagnostic pair with an exact two-world oracle.
Factorio and language-model agents are deliberately deferred until a small recurrent policy passes
the preregistered kill test.

## Development

```bash
uv sync --extra dev --extra learning-cpu
uv run pytest
uv run faultline --version
uv run faultline demo healthy --nodes 8 --ticks 10
uv run faultline demo diagnostic-pair --seed 42
# Requires a clean worktree and writes an immutable provenance manifest:
uv run faultline env generate --count 100 --split train --run-id gate1-pairs-v0.2
uv run faultline curriculum analyze --run-id gate2-ep-analysis-v0.3
uv run faultline curriculum control-audit --run-id matched-ep-control-v0.3
uv run faultline train --config configs/training/small-cpu-smoke.toml \
  --curriculum epistemic --seed 0 --dry-run
uv run faultline train --config configs/training/small-cpu-smoke.toml \
  --curriculum epistemic --seed 0 --run-id ep-smoke-seed-00
uv run faultline report small-kill --protocol configs/evaluation/small-kill-v1.toml
uv run faultline counterfactual study \
  --protocol configs/evaluation/counterfactual-v1.toml
uv run faultline counterfactual controls \
  --protocol configs/evaluation/behavioral-controls-v1.toml
uv run faultline benchmark simulator
```

The evolving scientific specification is in [`docs/research-thesis.md`](docs/research-thesis.md).
Measured findings will be linked to immutable manifests and exact Git commits; failed experiments
will be retained.

## Status

Gate 0 simulator semantics are implemented and tested: deterministic scalar and NumPy-batched
material flow, three latent fault families, typed actions, operational reward, observation
isolation, and sustained recovery. On the recorded local CPU, the 16-node batch kernel measured
median rates of 3.83M, 4.73M, and 4.41M environment steps/s at batch sizes 1k, 4k, and 16k
respectively (200 ticks, 3 repeats). See the immutable
[`simulator-throughput-v0.1-20260903`](artifacts/manifests/simulator-throughput-v0.1-20260903.json)
manifest.

The first hand-constructed pair has an exactly equal complete passive snapshot and disjoint unique
repairs. The exact finite-world solver enumerates diagnostic policies to depth three, verifies the
50% passive success ceiling, and finds `advance -> inspect`, which reaches 100% recovery with
expected operational return 8.58 versus 1.25 for passive commitment (`EP = 7.33`). The immutable
[`exact-two-world-oracle-v0.2-20260903`](artifacts/manifests/exact-two-world-oracle-v0.2-20260903.json)
manifest records the exact code, reward, seed, action values, and hardware. These are deterministic
environment/oracle results, not curriculum or learned-policy results.

Gate 1 generation produced and semantically validated 100 train-split pairs from seeds 0–99 with
100% acceptance. Every retained task had an exact shared passive snapshot, different unique repairs,
two diagnostic outcomes, 50% passive recovery, 100% depth-2 active recovery, and positive operational
EP. EP ranged from 5.58 to 12.76 (mean 8.86) under task-varying costs. See the immutable
[`manifest`](artifacts/manifests/gate1-diagnostic-pairs-v0.2-20260903.json) and reconstructable
[`task dataset`](artifacts/results/gate1-diagnostic-pairs-v0.2-20260903.json). This establishes the
construction mechanism only; it is not evidence that a policy can learn or generalize it.

Gate 2 analysis confirms that information becomes decision-relevant only after dynamics: immediate
`advance(1)` has zero information gain and mean one-step decision value −0.11, while the subsequent
inspection has exactly 1 bit of information and mean decision value 8.97. Raw EP is confounded with
repair margin (Pearson 0.94) and transport rate (0.76). Stake normalization removes those linear
associations but saturates (mean 0.973; 75% of tasks above 0.984), so it is not adopted as a ranker.
Instead, the controlled task family reuses each identical base factory while changing only
cue–fault dependence. Across 100 blocks, structural/cost standardized differences were exactly zero,
fault and cue marginals matched, ambiguous EP averaged 8.86, and revealed-cue EP was exactly zero.
Passive difficulty still differs and remains an explicit control for the RL kill test. See the
[`raw-score analysis`](artifacts/manifests/gate2-ep-analysis-v0.3-20260903.json),
[`EP histogram`](artifacts/results/gate2-ep-analysis-v0.3-20260903-ep.svg),
[`intervention plot`](artifacts/results/gate2-ep-analysis-v0.3-20260903-interventions.svg),
[`normalized-score analysis`](artifacts/manifests/gate2b-normalized-ep-v0.3-20260903.json), and
[`matched-control audit`](artifacts/manifests/gate2c-matched-ep-control-v0.3-20260903.json).

The learned-agent path is implemented: a 0.9M-parameter primary graph encoder/GRU, typed four-action
head, recurrent PPO, matched Random / Difficulty / Epistemic samplers, immutable checkpoints, and
paired validation. Policy reward is exclusively production minus operational cost. Deliberately
small 2k- and 10k-step Epistemic pilots collapsed to immediate fixed repair. At the declared 30k-step
pilot budget, seed 0 reached 100% ambiguous recovery and 100% experiment-then-correct behavior on 128
held-out validation base pairs. It also probed 100% of revealed-cue episodes, where probing is
unnecessary. This is the first positive learning primitive, but it may be a fixed troubleshooting
routine rather than uncertainty-sensitive investigation; it is neither a multi-seed curriculum
result nor causal evidence use. The immutable
[`manifest`](artifacts/manifests/ep-pilot-30k-masked-seed00-20260903.json) records the checkpoint and
validation traces.

The frozen validation-only kill test completed all 24 preregistered runs (8 training seeds × 3
curricula). Mean diagnostic success was 80.7% Random, 90.2% Difficulty, and 95.1% Epistemic.
Epistemic minus Random was +14.4 points with a 95% seed-bootstrap interval of [−8.4, 43.0];
Epistemic minus Difficulty was +4.9 [−13.3, 28.6]. Neither met the frozen criterion, so the result is
**no preregistered curriculum-specific effect**. Epistemic policies also inspected 100% of
revealed-cue episodes, consistent with a fixed probing routine. This stops LLM/Factorio escalation.
See the immutable [`analysis`](artifacts/manifests/small-kill-v1-analysis.json) and
[`seed plot`](artifacts/results/small-kill-v1-analysis.svg). The test split remains sealed.

Counterfactual swaps show that probing policies generally use the result: conditional causal
evidence-use rates were 99.1% Random, 99.6% Difficulty, and 90.2% Epistemic. Including policies that
never reached an evidence decision, rates were 80.5%, 89.8%, and 90.2%. Randomizing paired evidence
reduced correct repair to approximately 50%. Thus learned diagnostic routines usually condition on
evidence, but Epistemic training did not uniquely cause this and still probed indiscriminately.
See the [`causal manifest`](artifacts/manifests/counterfactual-v1-analysis.json) and
[`seed plot`](artifacts/results/counterfactual-v1-analysis.svg).

Removing diagnostic actions forced every arm to the 50% passive ceiling; mean recovery drops were
40.2, 44.9, and 45.1 points. On development-only 13–20-node OOD chains, recovery was 90.6% Random,
95.1% Difficulty, and 98.0% Epistemic. Probe and repair cost multipliers changed returns but changed
no action trace because v1 policies do not observe reward coefficients. That is a measured interface
limitation, not cost-rational behavior. See the
[`behavioral-controls manifest`](artifacts/manifests/behavioral-controls-v1-analysis.json).
See [`CHANGELOG.md`](CHANGELOG.md) for verified milestones.

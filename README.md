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
uv sync --extra dev
uv run pytest
uv run faultline --version
uv run faultline demo healthy --nodes 8 --ticks 10
uv run faultline demo diagnostic-pair --seed 42
# Requires a clean worktree and writes an immutable provenance manifest:
uv run faultline env generate --count 100 --split train --run-id gate1-pairs-v0.2
uv run faultline curriculum analyze --run-id gate2-ep-analysis-v0.3
uv run faultline curriculum control-audit --run-id matched-ep-control-v0.3
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
inspection has exactly 1 bit of information and mean decision value 8.97. The raw EP score is not
ready for curriculum selection: it correlates strongly with repair margin (Pearson 0.94) and
transport rate (0.76), plus moderately with graph size (0.49). This measured confound must be
controlled before the RL kill test. See the
[`analysis manifest`](artifacts/manifests/gate2-ep-analysis-v0.3-20260903.json),
[`EP histogram`](artifacts/results/gate2-ep-analysis-v0.3-20260903-ep.svg), and
[`intervention plot`](artifacts/results/gate2-ep-analysis-v0.3-20260903-interventions.svg).
See [`CHANGELOG.md`](CHANGELOG.md) for verified milestones.

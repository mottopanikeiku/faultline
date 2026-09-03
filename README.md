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
# Requires a clean worktree and writes an immutable provenance manifest:
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
manifest. Diagnostic-pair and learning results are not yet established.
See [`CHANGELOG.md`](CHANGELOG.md) for verified milestones.

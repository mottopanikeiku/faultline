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
```

The evolving scientific specification is in [`docs/research-thesis.md`](docs/research-thesis.md).
Measured findings will be linked to immutable manifests and exact Git commits; failed experiments
will be retained.

## Status

Deterministic healthy material-flow dynamics are implemented and covered by conservation and
replay tests. Fault, action, reward, diagnostic-pair, and learning results are not yet established.
See [`CHANGELOG.md`](CHANGELOG.md) for verified milestones.

# Benchmark protocol

Status: exploratory draft. No final test split or primary analysis is frozen yet.

## Split discipline

Procedural split definitions are immutable versioned files. Training may use only train generators;
validation controls curriculum/reward hyperparameters, checkpoint selection, and model selection;
test remains sealed until the confirmatory protocol is tagged. Structural OOD suites use disjoint
topology families, not only disjoint seeds.

Version `factory-pairs-v0` reserves seeds `[0, 100000)` for training,
`[1000000, 1010000)` for validation, and `[2000000, 2010000)` for test under
`diagnostic-chain-v1`. The CLI intentionally exposes only train and validation generation. The
committed definition is `configs/splits/factory-pairs-v0.json`; changing any range or generator
semantics requires a new split version rather than editing completed artifacts.

Provisional suites:

- IID held-out seeds;
- unseen topology family and larger graphs;
- unseen fault locations and compositions;
- multiple faults (later-stage OOD only);
- sensor noise and missing sensors;
- changed diagnostic and repair costs;
- restricted diagnostic budgets;
- no-fault and transient-only episodes.

## Required baselines

- random repair;
- symptom heuristic;
- fixed diagnostic procedure;
- exact Bayesian oracle in supported tiny worlds;
- oracle imitation;
- Random-RL, Difficulty-RL, and EP-RL using the same policy and optimizer.

A regret/UED-inspired curriculum is required before a strong environment-design claim. It may follow
the first kill test but may not be omitted from a final study.

## Matched comparison

Training arms receive equal environment steps, architecture, optimizer family, and comparable tuning
budgets. Sampling is stratified or reweighted to approximately match fault family, graph size, node
degree, layout family, horizon, and action budget. Reports include standardized marginal differences
and EP/difficulty correlations; residual mismatches are limitations.

## Provisional metrics

Operational metrics: recovery success, sustained recovery return, time to recovery, production loss,
diagnostic cost, repair cost, false repair rate, and total return.

Behavioral metrics: intervention count, passive observation count, success by diagnostic budget,
active-oracle regret, passive-ceiling exceedance, optimal-intervention agreement, and realized repair
decision value per diagnostic cost.

Causal metrics: repair-switch rate under paired outcome swaps, response-ablation effect, randomized-
response degradation, probe-removal effect, and intervention/repair cost elasticity.

## Statistical unit

Policies are evaluated on identical test scenario IDs. Comparisons are paired within training seed
where seeds share initialization schedules and paired by scenario within checkpoint. Training-seed
variation and environment-seed variation are reported separately. Major stochastic comparisons target
8–10 independent training seeds when feasible; deterministic oracle values receive no artificial
confidence interval.

Confidence intervals use a documented hierarchical or seed-level bootstrap appropriate to the claim.
Individual training-seed points accompany aggregates. Exploratory hyperparameter trials are not
pooled as independent replicates.

## Candidate primary endpoint—not frozen

Mean held-out sustained-recovery return on high-EP structural-OOD environments under a fixed
diagnostic budget. Candidate co-primary endpoint: recovery success on exact diagnostic pairs with a
known passive ceiling.

Before final evaluation, freeze in an annotated Git tag:

- architecture and training algorithm;
- total environment-step budget;
- curricula and matching method;
- reward coefficients and action budget;
- test generator version and scenario IDs;
- seed count and checkpoint rule;
- primary endpoint and statistical test;
- exclusion and failed-run handling.

## Manifest and artifact policy

Every run stores schema version, run ID, exact commit, dirty-tree flag, reconstructive patch when
dirty, full resolved config and hash, split/version, seeds, policy/curriculum/reward versions,
hardware, start/end timestamps, status, error data for failures, trajectory references, and metric
implementation version. Completed artifacts are append-only. Corrections produce new artifacts that
name the superseded result.

Serious benchmark runs start clean. Cloud launchers must require explicit execution and support dry
run, maximum steps, maximum wall time, and a cost ceiling where provider APIs expose enough pricing
information.

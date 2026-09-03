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

The implemented first learner uses a message-passing graph encoder, GRU, and four typed macro-level
actions: advance dynamics, inspect the nominated component, clear its outgoing blockage, or replace
the processor. Repair is a commitment and the simulator advances through sustained-recovery
verification, preventing repair-reward feedback from becoming a second diagnostic channel. PPO
replays complete padded episodes from zero recurrent state so gradients propagate through the
diagnostic history.

Training budget is counted in policy decision steps; underlying simulator ticks and episode counts
are also recorded. Counter-based sampling makes base-pair and latent-world schedules identical across
arms for the same seed. Random samples ambiguous/revealed conditions 50/50, Difficulty adapts that
probability from observed failure only, and Epistemic samples ambiguous conditions. The Difficulty
arm is intentionally strong and may converge toward the same task mix as Epistemic.

The first pipeline smoke used only 2,026 decision steps and a 52k-parameter reduced model. It
collapsed to immediate fixed repair: 50% ambiguous validation recovery and no diagnostic sequence.
This establishes end-to-end collection, optimization, checkpointing, and evaluation only. It is not
evidence for or against the curriculum hypothesis and does not count as a tuned seed.

A subsequent 10,054-step pilot with the primary 929,541-parameter model produced the same deterministic
failure: 50% ambiguous recovery and zero experiment-then-correct behavior. Random rollout discovery
of `advance -> inspect -> conditional repair` was too sparse under the unrestricted four-action
grammar. This result is preserved before introducing public, latent-independent action masks.
The revised grammar removes meaningless inspect-before-dynamics and repeated-probe sequences while
keeping both immediate repairs valid at every decision; it constrains tool syntax without forcing
diagnosis.

With that mask, a separate 10,043-step pilot still produced 50% deterministic ambiguous recovery and
no experiment-then-correct behavior. Stochastic informative inspection averaged 6.5% in its final
rollout but disappeared under argmax evaluation. Action grammar alone is therefore insufficient at
10k steps; reward-scale/value-loss and entropy settings require validation before the kill test.

Continuing the same configuration to 30,091 steps changed the result: seed 0 achieved 100%
deterministic ambiguous recovery and experiment-then-correct behavior over 128 validation base
pairs. The policy also advanced and inspected in 100% of revealed-cue episodes. Therefore it learned
a successful fixed troubleshooting routine, but the pilot does not yet show that it detects when
information is needed. The result justifies the matched multi-arm kill test and counterfactual
analysis; it does not establish the central hypothesis.

## Matched comparison

Training arms receive equal environment steps, architecture, optimizer family, and comparable tuning
budgets. Sampling is stratified or reweighted to approximately match fault family, graph size, node
degree, layout family, horizon, and action budget. Reports include standardized marginal differences
and EP/difficulty correlations; residual mismatches are limitations.

The first controlled comparison uses matched cue blocks. Ambiguous and revealed arms reuse identical
base pairs; both have exactly balanced cue and fault marginals, while only their cue–fault joint
differs. The audit reports standardized mean differences for every structural and cost parameter.
Because the revealed arm has a passive shortcut, passive difficulty is not considered matched;
Difficulty-RL remains a separate required arm, and active-oracle return should be stratified or
reweighted before confirmatory comparison.

The 100-block pretraining audit passed the exact marginal checks: maximum absolute structural/cost
standardized mean difference was 0.0, and both cue and fault marginals matched. Ambiguous and
revealed EP means were 8.86 and 0.0. The audit does not claim matched passive difficulty.

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

# Research thesis

## Central hypothesis

Faultline tests—not assumes—the following hypothesis:

> At equal environment-step budgets and approximately matched factory and fault marginals, training
> on tasks selected for high decision-relevant epistemic pressure causes recurrent policies to use
> diagnostic interventions more effectively than random or generic-difficulty curricula.

The intended behavior is an operational sequence: recognize that the visible history does not
identify a uniquely good repair; pay for an intervention; observe its consequence; condition the
repair on that evidence; restore sustained production.

## Candidate contribution

Active diagnosis, value of information, causal intervention, partially observable troubleshooting,
and adversarial environment generation all predate this project. The narrow candidate contribution
is a tested method for selecting training environments because active information acquisition has
decision value, rather than because environments are merely difficult, novel, or high regret.
Novelty remains provisional pending continued literature review.

## Falsifiable hypotheses

- **H1 (curriculum):** EP-selected training improves held-out recovery return on ambiguous OOD
  failures relative to matched random and difficulty curricula.
- **H2 (behavior):** EP-trained policies choose oracle-valued interventions more often and gain more
  repair decision value per unit diagnostic cost.
- **H3 (causal evidence use):** swapping a diagnostic response between otherwise confusable worlds
  shifts the policy's subsequent repair toward the repair supported by the swapped evidence.
- **H4 (generalization):** any advantage survives changes in topology, graph size, fault location,
  diagnostic cost, and no-fault prevalence.

Each hypothesis may be rejected. A null or negative result is a research artifact, not a reason to
change the held-out test after inspection.

## Scope order and kill tests

1. Establish deterministic factory semantics and observation isolation.
2. Generate at least 100 validated ambiguous tasks with different optimal repairs and a positive
   active-versus-passive value gap.
3. Verify exact short-horizon solutions by exhaustive enumeration in tiny worlds.
4. Compare small recurrent policies under Random, Difficulty, and EP selection.
5. Stop before language-model or Factorio work unless the small-policy result is robust across
   competent implementations and multiple seeds.

## Current evidence status

Gates 0–2 establish the simulator, 100 generated exact diagnostic pairs, passive ceilings, and exact
active solutions. The frozen validation-only `small-kill-v1` comparison completed eight training
seeds per curriculum. Epistemic mean diagnostic success was higher than Random and Difficulty, but
both paired bootstrap intervals included zero and the Difficulty mean difference fell below the
predeclared five-point threshold. H1 is therefore not supported by this kill test.

Learned policies can execute `advance -> inspect -> conditional repair`, so representational
inability is not the immediate explanation. Counterfactual swaps show that most probing policies
change to the repair supported by the swapped world; randomized evidence reduces repair accuracy to
chance. H3 is supported for learned diagnostic routines. This behavior is not curriculum-specific:
Difficulty and Random policies that probe have at least as high conditional causal-use rates.
Epistemic policies also probe every revealed-cue case, so H2 remains unsupported and the current
behavior is not selectively decision-relevant. Development OOD results on larger linear chains are
positive but do not test a new topology family or sealed data, so H4 remains unsupported. Diagnostic
action ablation returns all arms to the passive ceiling. Cost sensitivity cannot be assessed because
v1 observations omit costs; measured action invariance records that interface limitation. Per the
kill gate, LLM and Factorio work is paused.

## Operational reward constraint

The primary policy reward is production recovery minus time, diagnostic, repair, and false-repair
costs. It must not contain entropy reduction, information gain, diagnosis text, or evaluator-scored
reasoning. Information measures may select environments and analyze trajectories only.

## Core threats to validity

- High EP could be a proxy for generic difficulty or a shifted fault distribution.
- A supposedly hidden fault could leak through identifiers, masks, ordering, timing, or tiny numeric
  differences.
- A policy could probe habitually while ignoring the response.
- Repair or probe spam could appear competent under weak costs.
- A recurrent model could memorize layouts or intervention mappings.
- Oracle and environment reward semantics could diverge.
- PPO-specific optimization behavior could be mistaken for an environment-design effect.

Distribution audits, no-fault cases, passive ceilings, exact solvers, paired evaluation, evidence
swaps, action ablations, and an algorithmic replication are required controls.

## Evidence standard

Every reported run records immutable configuration, split identity, seed, generator and metric
versions, hardware, timestamps, exact Git commit, dirty-tree state, and reconstructive patch when
run dirty. README and paper claims must point to measured artifacts. Exploratory and confirmatory
analyses remain explicitly separated.

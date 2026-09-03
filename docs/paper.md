# Faultline: train on ambiguity, not difficulty

Status: working research report. The final test set has not been opened; this is not a confirmatory
paper.

## Abstract

Agents acting under partial observability may need to sacrifice short-term return to identify which
latent cause requires intervention. Faultline constructs deterministic factory pairs with identical
public initial state, different unique repairs, and a low-cost dynamical experiment that separates
the latent worlds. An exact finite-world oracle certifies passive ceilings and active value. A
procedural generator produced 100/100 valid training pairs, with mean active-minus-passive operational
value 8.86. Raw value gaps were strongly confounded with repair stakes, so we introduced matched cue
blocks that hold factory, fault, reward, cue, and fault marginals fixed while changing cue–fault
dependence.

In a frozen validation-only kill test, recurrent PPO policies were trained for 30k decision steps
under Random, student-failure Difficulty, and Epistemic curricula with eight seeds each. Mean
held-out diagnostic success was 0.807, 0.902, and 0.951. Paired Epistemic improvements were 0.144 over
Random (95% training-seed bootstrap interval [−0.084, 0.430]) and 0.049 over Difficulty
([−0.133, 0.286]). Neither met the frozen decision rule. Epistemic policies also probed every
revealed-cue episode. Counterfactual swaps showed that probing policies usually change repair with
the evidence, but this causal use was not curriculum-specific. Because cue reliability and costs are
absent from v1 policy observations, the revealed condition cannot establish whether policies
selectively recognize uncertainty. The primary curriculum effect is unsupported; language-model and
Factorio stages are paused.

## 1. Introduction

Difficulty-based curriculum selection does not distinguish hard tasks that reward information
acquisition from hard tasks that reward better direct control. Faultline asks whether selecting
training environments for the operational advantage of active diagnosis changes learned behavior.
The intended behavior is observable: intervene, observe the response, and condition a costly repair
on that response. Natural-language explanations are neither required nor scored.

The candidate contribution is narrow. Active diagnosis, value of information, causal intervention,
POMDP troubleshooting, causal meta-RL, and unsupervised environment design all have substantial
precedent. Faultline contributes a controlled implementation and empirical test of environment
selection based on active-versus-passive operational value. The current test does not support a
stronger claim.

## 2. Related work

The living review in `docs/literature.md` covers active diagnosis and sensing, decision-theoretic
value of information, causal RL and meta-RL, PAIRED and regret-based environment design, curriculum
learning, tool-using root-cause agents, and the Factorio Learning Environment. Faultline does not
claim to introduce observational equivalence or active diagnosis. Its comparison target is
environment selection by decision-relevant diagnostic value rather than generic difficulty or
regret.

## 3. Factory environment

A factory is a directed acyclic material-flow graph with source, processor, buffer, and sink nodes.
Dense arrays store input/output material, enablement, latent failures, edge capacities, and flows.
Discrete transitions are deterministic given graph, latent fault, and action history. The implemented
faults are blocked transport, failed processor, and downstream backpressure. Public observations
exclude fault flags.

The policy can advance dynamics, inspect a nominated component, clear an outgoing blockage, or
replace the processor. Repair is a commitment followed by sustained-recovery verification. Reward is
sink throughput minus time, passive observation, intervention, repair, and false-repair costs, plus a
sustained-recovery bonus. No entropy, information gain, diagnosis text, or evaluator-scored reasoning
term enters policy reward.

The NumPy batch kernel measured 3.83M–4.73M environment steps/s for 1k–16k copies of a 16-node graph
on the recorded local CPU. This is a local implementation measurement, not a cross-platform claim.

## 4. Epistemic pressure and construction

The provisional score is

\[
EP(e)=V^*_{\mathrm{active}}(e)-V^*_{\mathrm{passive}}(e).
\]

In the manual pair, blocked processor output and failed processing share the complete passive
snapshot but require different repairs. `advance(1) -> inspect(processor)` creates point-mass
posteriors. Exact expected returns are 8.58 active and 1.25 passive, with recovery probabilities 1.0
and 0.5.

Generator `diagnostic-chain-v1` varies graph length, processor position, flow rates, capacities,
preload, and costs. All 100 first candidates passed exact snapshot, repair-uniqueness, intervention,
and positive-value checks. Raw EP correlated 0.94 with perfect-information decision regret and 0.76
with transport rate. A normalized fraction of recoverable perfect-information regret removed those
linear scale associations but saturated near one. Neither raw nor normalized ranks are used without
control.

Matched cue blocks reuse each base factory. In the ambiguous condition, a balanced binary cue is
independent of fault within the training distribution. In the revealed condition, the same balanced
cue identifies fault. Fault and cue marginals and every structural/cost feature match exactly; only
the joint differs. Exact ambiguous and revealed EP means are 8.86 and 0.0. Passive difficulty differs
by design and therefore requires a separate Difficulty curriculum.

## 5. Small-policy methods

The primary policy has 929,541 parameters: three message-passing graph layers, target-aware pooling,
a 256-dimensional GRU, and actor/critic heads. Public-history masks remove meaningless repeated tool
sequences while leaving both immediate repairs available. PPO uses complete-episode recurrent replay,
GAE, clipped policy and value objectives, entropy regularization, gradient clipping, and KL early
stopping.

Counter-based curricula share base-factory and latent-world schedules by training seed. Random uses a
50/50 ambiguous/revealed mixture. Difficulty adjusts that mixture from observed student failure.
Epistemic uses ambiguous blocks. All arms receive a 30k policy decision-step target. Evaluation
enumerates both worlds and balanced cues over 128 fixed validation base seeds.

Before seeds 100–107, tag `study-small-policy-v1` froze architecture, budget, curricula, endpoint,
seed set, validation range, and analysis. The primary endpoint was ambiguous
experiment-then-correct-repair rate. Training-seed-level paired differences used 10,000 percentile
bootstrap resamples. A supported effect required a mean of at least 0.05 and a lower 95% interval
bound above zero against both baselines.

## 6. Validation results

| Curriculum | Mean diagnostic success | 95% seed-bootstrap interval |
|---|---:|---:|
| Random | 0.807 | [0.557, 0.995] |
| Difficulty | 0.902 | [0.710, 1.000] |
| Epistemic | 0.951 | [0.862, 1.000] |

The Epistemic-minus-Random paired mean was 0.144 [−0.084, 0.430]. Epistemic-minus-Difficulty was
0.049 [−0.133, 0.286]. The frozen decision is `no_preregistered_curriculum_effect`.

Training variance was large. Diagnostic success ranged from 0 to 1 for Random, 0.230 to 1 for
Difficulty, and 0.645 to 1 for Epistemic. Mean ambiguous recovery was 0.901, 0.950, and 0.951.
Epistemic policies inspected 1.000 of revealed-cue episodes versus 0.813 Random and 0.904 Difficulty.
This does not demonstrate irrational over-probing: the current observation does not identify whether
the cue is reliable in a particular episode. It does show that Epistemic training learned a fixed
probe-first policy.

## 7. Does the agent use evidence?

Counterfactual action interventions establish that successful routines usually use diagnostic
evidence. Among checkpoints that reached the evidence decision, swapping only telemetry changed the
repair toward the paired world's repair with conditional causal-use rates 0.991 Random, 0.996
Difficulty, and 0.902 Epistemic. Overall rates including non-probing policies were 0.805, 0.898, and
0.902. Randomizing between the two valid outcomes reduced correct repair to approximately 0.5.
Removing, staling, or averaging evidence changed approximately half of decisions.

This supports causal evidence use in learned routines but not a curriculum-specific advantage.
Epistemic policies were always eligible because they always probed, yet one seed used evidence
inconsistently; Difficulty policies had comparable overall causal use and higher conditional use.
The learned behavior therefore has two separable properties: many policies condition repair on the
probe result, while selective probing cannot be assessed until reliability and costs are observable.

Removing diagnostic actions reduced every policy to the 0.5 passive recovery ceiling. Mean recovery
drops were 0.402 Random, 0.449 Difficulty, and 0.451 Epistemic. On development-only 13–20-node
chains with unseen numeric ranges, mean recovery was 0.906, 0.951, and 0.980; diagnostic success was
0.813, 0.902, and 0.980. This suggests the learned routine can extend to larger linear chains, but it
does not establish new-topology or sealed-test generalization.

Multiplying probe-related and repair-related costs changed operational return but changed no action
trace. This is not evidence of irrational policies: v1 observations omit reward coefficients, making
cost-conditioned behavior impossible. The interface must change before a meaningful cost-sensitivity
study.

## 8. Language-agent and Factorio transfer

Not run. The frozen small-policy kill test did not establish a curriculum-specific effect, and the
v1 interface cannot test selective cost- or reliability-sensitive probing. Per the project gate,
language-model training and Factorio validation remain paused.

## 9. Limitations

The first generator uses linear chains and two single faults. The policy is told the nominated
component, reducing diagnosis to cause selection rather than fault localization. Public action masks
encode tool ordering. The revealed control is passively easier. The current validation uses the same
structural generator family, not the sealed test set or a new topology family. Development OOD only
extends linear-chain size and numeric ranges. Eight training seeds leave wide intervals. PPO may
contribute substantial seed instability. Reward coefficients are absent from policy observations,
precluding cost adaptation. No no-fault episodes enter the current RL interface, so routine probing
is weakly challenged. These constraints prevent broad claims about transferable
experiment-before-action behavior.

## 10. Conclusion

Faultline establishes fast deterministic factory dynamics, exact diagnostic pairs, an exact active
oracle, matched epistemic controls, learned diagnostic routines, and causal evidence-use
interventions. Most probing policies use the evidence, but the first frozen curriculum test does not
establish that Epistemic training outperforms Random or Difficulty training, and Epistemic policies
probe even when a passive cue suffices. Larger models are not the next step. The immediate research
problems are seed instability, no-fault/stronger anti-probe controls, and curricula that teach when
information has decision value rather than a universal troubleshooting script.

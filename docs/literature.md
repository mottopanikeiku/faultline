# Living literature review

This document bounds Faultline's claims. It records directly relevant prior work and open comparison
questions; it is not a claim of exhaustive coverage. Links point to papers or official project pages
that were checked during repository bootstrap.

## Active diagnosis and troubleshooting

Active diagnosis chooses observations adaptively to infer a hidden hypothesis. Pu, Kaelbling, and
Solar-Lezama formalize this setting and learn which observation to acquire next without explicitly
representing the full hypothesis space [1]. Classical model-based diagnosis and troubleshooting
already study test selection, action cost, and repair under uncertainty; Faultline must not claim to
invent any of these objects. Its distinction, if supported, is selecting *RL training environments*
by the operational decision value of diagnosis.

POMDPs provide the general sequential decision framework: belief updates and future reward already
induce a value for information without requiring an entropy reward. Work on costly state observation,
including action-contingent noiselessly observable MDPs [2], makes the observe/act tradeoff explicit.
This is close to Faultline's passive sensing axis, although Faultline initially emphasizes diagnostic
actions that perturb the plant rather than merely buying a state observation.

## Active sensing and value of information

Decision-theoretic value of information (VOI) prices evidence by the improvement in downstream
decisions, not by uncertainty reduction alone. Krause and Guestrin study optimal VOI computation in
graphical models and its computational hardness [3]. This supports a critical design distinction:
an entropy-reducing probe can have no repair value when it does not change the optimal repair.

Faultline's provisional EP gap is therefore not a new definition of VOI. It is a proposed environment
selection statistic derived from active and passive policy classes. The project must compare it with
raw posterior entropy, mutual information, intervention cost, and ordinary task difficulty.

## Causal intervention, causal RL, and meta-RL

Causal RL spans structural knowledge, interventions, counterfactuals, representation learning, and
transfer; recent surveys show that the area is broad and pre-existing [4, 5]. Active causal discovery
uses interventions to identify causal structure, sometimes optimizing intervention choice with RL
[6]. Faultline's initial objective is narrower: choose among repair-relevant latent worlds, not recover
a complete causal graph.

Dasgupta et al. show that a recurrent model-free meta-RL agent can learn informative interventions,
observational inference, and counterfactual prediction across causal tasks [7]. Sauter et al. train a
meta-RL policy to perform budgeted interventions and infer causal structure at test time [8]. These
works are direct precedent for learned experiment selection. Faultline's unresolved question is
whether *environment curriculum selection based on decision-relevant active advantage*, under a
purely operational reward, changes learned diagnostic behavior.

## Unsupervised and adversarial environment design

Dennis et al. introduce Unsupervised Environment Design and PAIRED, in which an adversary uses an
antagonist/protagonist return gap as a regret signal to produce solvable curricula [9]. PLR and
replay-guided environment design prioritize informative previously seen levels and improve robustness
and sample efficiency [10]. Evolving Curricula with Regret-Based Environment Design extends
regret-based generation with evolutionary search [11].

These methods motivate strong difficulty- and regret-based baselines. Faultline cannot attribute an
advantage to epistemic structure unless fault, topology, budget, horizon, and preferably oracle
passive difficulty marginals are controlled. High EP may simply be high regret; this correlation is a
required measurement.

## Curriculum learning

Curriculum learning predates UED and can order examples by hand-designed or student-dependent
notions of difficulty. Faultline's static first comparison deliberately separates environment scoring
from adaptive teacher learning. A student-aware EP objective is deferred until a static score shows a
replicable effect; otherwise curriculum adaptivity and epistemic structure are confounded.

## Tool agents, information seeking, and root-cause analysis

ReAct couples language-model reasoning with actions and observations [12], providing a common tool
agent pattern but not a guarantee of decision-relevant investigation. Roy et al. evaluate a ReAct RCA
agent with retrieval and external diagnostic tools on production incidents [13]. This establishes
precedent for tool-using root-cause agents and exposes practical tool-query difficulties. Search-agent
benchmarks evaluate evidence gathering and grounded answers, but web retrieval differs causally from
intervening on a dynamical system.

Faultline therefore evaluates behavior through environment interventions and counterfactual response
swaps, not generated explanations. Language-agent experiments remain conditional on a successful
small-policy result.

## Factorio Learning Environment

Hopkins, Bakler, and Khan introduce FLE for long-horizon planning, program synthesis, and resource
optimization through a typed Python interface [14]. FLE is relevant external validation, not the
training substrate for v1. Faultline's synthetic simulator targets controlled latent failure pairs,
exact passive ceilings, and cheap counterfactual replay—properties not claimed by FLE.

## Novelty boundary and unresolved search

The defensible candidate contribution is currently:

> A controlled empirical test of curricula that select procedural environments by the net operational
> advantage of active diagnosis over passive commitment, together with causal evidence-use
> evaluations.

This wording must narrow if closer precedents are found. Before any paper claim, the review still
needs a systematic database search covering model-based troubleshooting, Bayesian experimental
design, active fault isolation/control, machine diagnosis, Bayes-adaptive MDPs, and recent tool-agent
RL. Search strings, inclusion criteria, and a dated evidence table should be preserved.

## References

1. Pu, Kaelbling, and Solar-Lezama. [Learning to Acquire Information](https://arxiv.org/abs/1704.06131), 2017.
2. Nam, Fleming, and Brunskill. [Reinforcement Learning with State Observation Costs in Action-Contingent Noiselessly Observable Markov Decision Processes](https://proceedings.neurips.cc/paper/2021/hash/83e8fe6279ad25f15b23c6298c6a3584-Abstract.html), NeurIPS 2021.
3. Krause and Guestrin. [Optimal Value of Information in Graphical Models](https://arxiv.org/abs/1401.3474), JAIR 2009 / arXiv version 2014.
4. Zeng et al. [A Survey on Causal Reinforcement Learning](https://arxiv.org/abs/2302.05209), 2023.
5. Deng et al. [Causal Reinforcement Learning: A Survey](https://arxiv.org/abs/2307.01452), 2023.
6. Amirinezhad, Salehkaleybar, and Hashemi. [Active Learning of Causal Structures with Deep Reinforcement Learning](https://arxiv.org/abs/2009.03009), 2020.
7. Dasgupta et al. [Causal Reasoning from Meta-reinforcement Learning](https://arxiv.org/abs/1901.08162), 2019.
8. Sauter et al. [A Meta-Reinforcement Learning Algorithm for Causal Discovery](https://proceedings.mlr.press/v213/sauter23a.html), 2023.
9. Dennis et al. [Emergent Complexity and Zero-shot Transfer via Unsupervised Environment Design](https://arxiv.org/abs/2012.02096), NeurIPS 2020.
10. Jiang et al. [Replay-Guided Adversarial Environment Design](https://proceedings.neurips.cc/paper/2021/hash/0e915db6326b6fb6a3c56546980a8c93-Abstract.html), NeurIPS 2021.
11. Parker-Holder et al. [Evolving Curricula with Regret-Based Environment Design](https://proceedings.mlr.press/v162/parker-holder22a.html), ICML 2022.
12. Yao et al. [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629), ICLR 2023.
13. Roy et al. [Exploring LLM-based Agents for Root Cause Analysis](https://arxiv.org/abs/2403.04123), 2024.
14. Hopkins, Bakler, and Khan. [Factorio Learning Environment](https://arxiv.org/abs/2503.09617), 2025.

# Epistemic pressure

## Provisional definition

For an environment distribution \(e\), let \(V^*_{\mathrm{active}}(e)\) be optimal expected
operational return when diagnostic interventions are available before commitment, and let
\(V^*_{\mathrm{passive}}(e)\) be optimal expected return when the policy may make ordinary
observations but cannot intervene before committing. The initial working score is

\[
EP(e) = V^*_{\mathrm{active}}(e) - V^*_{\mathrm{passive}}(e).
\]

This definition is a research object, not a fixed contribution. It mixes ambiguity, action stakes,
diagnostic cost, and horizon; each factor must be reported separately so that EP is not mistaken for
generic difficulty.

## Diagnostic-pair criterion

For two latent worlds \(z_a,z_b\) under prior \(p\), retain a pair only when:

1. their full public initial observations and valid-action masks are equal under an explicit
   comparator;
2. their unique best immediate repairs differ by more than a configured value margin;
3. at least one allowed intervention yields distinguishable public outcome distributions;
4. the best active policy has positive net value over the best passive policy after intervention,
   time, and repair costs;
5. neither object identifiers nor task metadata encode the world assignment.

Approximate observation similarity may be studied later, but exact equality is the first milestone.

## Passive ceiling

If a shared public history leaves posterior \(b(z)\) and terminal decision \(a\) succeeds in worlds
\(S_a\), any policy forced to commit at that history has success probability at most

\[
\max_a \sum_{z \in S_a} b(z).
\]

For two equally likely worlds with disjoint unique correct repairs, this bound is \(1/2\). The claim
is elementary; its purpose is to certify benchmark behavior, not to assert theoretical novelty.

**Proof sketch.** A randomized passive policy chooses a distribution \(q(a)\) shared across latent
worlds. Its success is
\(\sum_a q(a)\sum_{z\in S_a}b(z)\), a convex combination bounded by the largest inner sum.
A deterministic policy attains that bound. For two equally probable worlds whose successful-action
sets are disjoint, each inner sum is at most \(1/2\). A separating intervention can move the
posterior to a point mass in each outcome branch, after which the corresponding repair succeeds;
its return advantage still depends on intervention and delay costs.

## Intervention value

For intervention \(i\) with outcome \(o\), posterior \(b_{i,o}\), and operational cost \(C(i)\),
its one-step decision value is

\[
DV(i,b)=\sum_o P(o\mid i,b)\max_a Q(b_{i,o},a)-\max_a Q(b,a)-C(i).
\]

Raw entropy reduction is logged separately. A probe can be informative yet have zero decision value
when every posterior supports the same repair. Faultline should preferentially test the latter
distinction.

## First manual construction

The seed-42 manual pair uses the same three-node factory state in both worlds. World A has a blocked
processor-to-sink edge; World B has a failed processor. Their status, every node inspection, and every
edge measurement are byte-for-byte equal before a dynamical action. Their only successful candidate
repairs differ.

Isolating the source-to-processor feed and advancing one tick yields:

- blocked-output world: processor input 0, output 2;
- failed-processor world: processor input 2, output 0.

A policy branching only on that public output-buffer measurement selects the correct repair and
recovers both worlds with expected return 8.08. The exact solver additionally enumerates every
contingent policy over the bounded diagnostic action set through depth three. It finds the cheaper
sequence `advance(1) -> inspect(processor)`, followed by the posterior-optimal repair. This policy
has expected return 8.58 and 100% recovery versus 1.25 and 50% for the best shared passive
commitment, giving \(EP=7.33\). Depth-zero and depth-one solutions equal the passive optimum;
depth-two and depth-three solutions agree. Tests compare the solver value with direct simulator
execution of the discovered policy. Exact configuration and provenance are in
`artifacts/manifests/exact-two-world-oracle-v0.2-20260903.json`. These are deterministic oracle
values, not training or statistical estimates.

## Procedural validation

`diagnostic-chain-v1` samples a processor in an alternating linear factory and constructs two latent
worlds: its outgoing transport is blocked, or the processor has failed. A shared preload ensures that
advancing once and inspecting that processor yields a separating response. Retention is not based on
construction assumptions alone: both repairs and the exact depth-2 policy tree are executed. The
validator records exact passive-snapshot equality, repair uniqueness and margin, diagnostic outcome
count, passive and active recovery probabilities, both expected returns, and net EP. Generation is
bounded by a declared maximum attempt count and fails rather than returning fewer tasks.

The first recorded Gate 1 run evaluated train seeds 0–99 from clean commit `03a6175`. All 100
candidates passed, for 100% acceptance; this rate is specific to the constructive v1 family and does
not demonstrate broad search efficiency. Every pair had passive and active recovery probabilities
0.5 and 1.0. EP ranged from 5.58 to 12.76 with mean 8.86, and the minimum correct-versus-wrong repair
margin was 12.20. The manifest and complete reconstructable task records are
`artifacts/manifests/gate1-diagnostic-pairs-v0.2-20260903.json` and
`artifacts/results/gate1-diagnostic-pairs-v0.2-20260903.json`.

## Candidate curriculum controls

Static experiments compare Random, Difficulty, and EP selection with matched marginals over fault
family, graph size, degree, layout family, intervention budget, and horizon. Difficulty is measured
without using held-out test outcomes. Candidate EP scores are evaluated for residual correlation
with passive oracle difficulty before policy training.

A student-aware score such as

\[
EP(e)\,[V^*_{\mathrm{active}}(e)-V^\pi(e)]
\]

is deferred until static EP selection works. Otherwise adaptation obscures whether epistemic task
structure itself mattered.

## Distribution-analysis protocol

The Gate 2 analysis regenerates every recorded task seed under the manifest's generator version.
For each pair it records EP, active and passive return, repair margin, the immediate value and
information gain of `advance(1)`, and the value and information gain of inspecting after that
dynamics step. It reports task-level rows, population distribution summaries, Pearson and tied-rank
Spearman correlations with structural/cost parameters and passive-return difficulty, analysis
throughput, and immutable SVGs. Constant features return a null correlation rather than a fabricated
zero association.

## Gate 2 result and negative finding

For the 100 recorded pairs, immediate `advance(1)` produced no world separation: information gain
was exactly 0 bits and one-step decision value ranged from −1.125 to numerical zero (mean −0.110).
After that dynamics step, inspecting the fault processor perfectly separated the equiprobable worlds:
information gain was exactly 1 bit and decision value ranged from 5.83 to 12.76 (mean 8.97). This
distinguishes an initially costly, uninformative transition from the later measurement that changes
the repair decision.

Raw EP is confounded in `diagnostic-chain-v1`. Its Pearson correlations were 0.94 with repair margin,
0.76 with transport rate, 0.67 with nominal rate, 0.57 with preload, 0.52 with fault position, and
0.49 with node count. Correlation with the current passive-difficulty proxy (negative passive return)
was −0.56: high-EP tasks were not simply harder by that proxy, but the score strongly tracked stakes
and scale. Diagnostic cost had near-zero correlation because the exact policy uses time advance and
passive inspection rather than the separately priced isolation action.

Decision: do not feed raw EP ranks into RL and attribute differences to epistemic structure. The next
curriculum implementation must either match these marginals/repair stakes across arms or validate a
normalized/stratified score with substantially reduced associations. This is a preserved negative
metric result, not evidence against active diagnosis itself. Full rows, correlations, plots, and
provenance are in `artifacts/manifests/gate2-ep-analysis-v0.3-20260903.json` and its linked artifacts.

## Scale control and matched construction

Define the no-diagnostic perfect-information value \(V^*_{\mathrm{perfect}}\) by revealing the latent
world to the terminal decision oracle at no cost. A dimensionless candidate is

\[
NEP(e)=
\frac{V^*_{\mathrm{active}}(e)-V^*_{\mathrm{passive}}(e)}
     {V^*_{\mathrm{perfect}}(e)-V^*_{\mathrm{passive}}(e)}.
\]

The denominator is the decision regret that perfect fault identity could recover. The score is
defined only when that regret is positive. It separates gross repair stakes from the fraction
recoverable after diagnostic costs. It is logged as a second axis, not substituted silently for raw
EP; near-one saturation would make it useless for ranking.

For the learning comparison, a stronger control changes epistemic structure within an exact matched
block. Each base factory creates two ambiguous tasks whose public binary cue is constant across its
two latent worlds (one task for each cue value), and one revealed task whose cue identifies the
fault. Sampling the ambiguous pair of tasks uniformly and sampling the revealed task both produce
50/50 cue and 50/50 fault marginals. Graph, fault locations, capacities, rates, preload, reward,
horizon, and action set are the same object. Only the cue–fault joint differs. Exact evaluation must
show positive EP for the ambiguous condition and zero EP for the revealed condition before this
control enters RL.

This controls raw scale and marginal distribution shifts. It does not automatically match passive
difficulty: the revealed condition is deliberately solvable without diagnosis. Difficulty-RL and
active-oracle-value stratification remain necessary controls.

The measured normalization run confirms both its use and its limitation. \(NEP\) ranged from 0.801
to 0.998 with mean 0.973 and first quartile 0.984. Pearson correlations with repair margin,
transport rate, and nominal rate fell to 0.047, −0.113, and −0.074, respectively, but the score
saturated and retained tied-rank correlations with fault position (0.76) and repair margin (0.63).
Decision: retain \(NEP\) as a diagnostic efficiency statistic, not a task-ranking curriculum.

The matched-control audit then evaluated all 100 base pairs. Every block had equal fault and cue
marginals and identical structural/cost features (maximum absolute standardized mean difference
0.0), while every cue–fault joint differed by construction. Ambiguous tasks retained mean EP 8.86,
50% passive recovery, and 100% active recovery. Revealed tasks had EP exactly 0 and both passive and
active recovery of 100%. This resolves the raw-scale selection confound for the first learning
comparison by selecting conditions within matched blocks rather than ranking unrelated tasks.
Artifacts: `artifacts/manifests/gate2b-normalized-ep-v0.3-20260903.json` and
`artifacts/manifests/gate2c-matched-ep-control-v0.3-20260903.json`.

The remaining difficulty confound is explicit: revealed tasks are passively easier. The kill test
therefore requires a Difficulty-RL arm and paired reporting against active-oracle value; the matched
audit alone does not discharge that control.

## Measurements required before adoption

- pair-generation acceptance and rejection reasons;
- active and passive values, diagnostic costs, and repair margins;
- raw ambiguity and entropy reduction;
- correlations with graph and fault marginals and passive difficulty;
- sensitivity to reward coefficients and horizon;
- stability of rankings across generator seeds.

A score that primarily reorders environments by difficulty, fault type, or graph size is not an
adequate EP curriculum metric.

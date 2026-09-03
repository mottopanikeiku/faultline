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

## Measurements required before adoption

- pair-generation acceptance and rejection reasons;
- active and passive values, diagnostic costs, and repair margins;
- raw ambiguity and entropy reduction;
- correlations with graph and fault marginals and passive difficulty;
- sensitivity to reward coefficients and horizon;
- stability of rankings across generator seeds.

A score that primarily reorders environments by difficulty, fault type, or graph size is not an
adequate EP curriculum metric.

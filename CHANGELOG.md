# Changelog

All notable repository and research-protocol changes are recorded here. Experimental findings are
listed only when backed by immutable artifacts.

## Unreleased

### Added

- Initial research thesis, environment semantics, epistemic-pressure specification, literature map,
  and exploratory benchmark protocol.
- Python package metadata and a CPU-only command-line entry point.
- Dense-array graph compilation and deterministic source, processor, buffer, transport, and sink
  material-flow dynamics.
- Semantic tests for conservation, staged causal propagation, deterministic replay, state branching,
  and cycle rejection.
- Latent blocked-edge, failed-processor, and downstream-backpressure faults with causal restoration
  tests and privileged-state isolation boundaries.
- Typed inspect, flow measurement, isolation, toggle, replacement, blockage clearing, and time
  advancement actions with a public-only interaction history.
- Observation-isolation tests that compare initially confusable latent faults and ensure repair
  responses do not reveal whether maintenance was necessary.
- Operational throughput-minus-cost reward accounting with separate passive, diagnostic, repair,
  false-repair, and time costs.
- Sustained-recovery, action-limit, and tick-limit termination with recomputable episode metrics.
- Homogeneous NumPy-batched state and stepping kernel with active-row masks, exact scalar
  equivalence tests, per-row conservation checks, and deterministic fault recovery.
- Text factory and public interaction-timeline renderer with latent markers gated behind explicit
  debug mode.
- Reproducible simulator-throughput benchmark and immutable schema-v1 manifests recording exact Git
  state, canonical config hash, hardware, timestamps, and metric version.
- First manual diagnostic pair with an exact complete-passive-snapshot equality check, disjoint
  successful repairs, a separating isolation experiment, and an evidence-contingent recovery policy.
- Exact finite-world Bayesian belief utilities, terminal repair enumeration, and contingent active
  diagnostic solver for horizons zero through six.
- Intervention outcome partitioning, information gain, decision value, and direct-enumeration
  correctness tests for the two-world problem.
- Deterministic `diagnostic-chain-v1` pair generator varying chain length, processor location, rates,
  capacities, preload, and operational costs.
- Bounded semantic validator requiring exact passive equality, disjoint unique repairs, separating
  evidence, positive oracle EP, and active recovery advantage.
- Frozen `factory-pairs-v0` train, validation, and test seed ranges plus reconstructable immutable
  pair-dataset serialization.
- Exact EP distribution analyzer with one-step and post-dynamics intervention value, information
  gain, parameter correlations, task-level rows, and dependency-free immutable SVG plots.
- Stake-normalized EP based on recoverable perfect-information decision regret, with correlation
  audit rather than silent adoption.
- Matched cue-control blocks that reuse the identical factory, faults, reward, and fault/cue
  marginals while changing only cue–fault dependence between ambiguous and revealed conditions.
- Small graph-message-passing/GRU actor-critic (approximately 0.9M parameters at the primary
  configuration) with typed diagnostic and repair actions.
- Complete-episode recurrent PPO with GAE, clipped actor/value objectives, entropy regularization,
  recurrent minibatch replay, gradient clipping, and KL early stopping.
- Counter-based matched Random, student-failure Difficulty, and Epistemic curriculum samplers sharing
  base-factory and latent-world schedules.
- Versioned portable policy checkpoints, CPU/CUDA uv extras, dry-run cost control, and paired
  validation evaluation over every world and balanced cue.
- Public-history-only action masks that keep both repairs available while removing meaningless
  inspect-before-dynamics, repeated-advance, and repeated-inspection sequences.

### Research status

- Gate 0 local benchmark established median throughput of 3.83M, 4.73M, and 4.41M environment
  steps/s for 1k, 4k, and 16k batches of a 16-node graph, respectively, over 200 ticks and three
  repeats. Reference manifest: `simulator-throughput-v0.1-20260903`.
- The earlier `simulator-throughput-local-20260903` artifact is retained but superseded for hardware
  reporting because the initial manifest did not recover the Linux CPU model.
- The exact depth-2 oracle verifies a 50% passive recovery ceiling and finds a 100%-recovery
  `advance -> inspect` policy. Expected operational values are 1.25 passive and 8.58 active
  (`EP = 7.33`). Reference manifest: `exact-two-world-oracle-v0.2-20260903`. This is an oracle
  construction result, not a learned-policy result.
- Gate 1 generated and validated 100/100 train-split candidates (seeds 0–99). All had exact passive
  equality, different unique repairs, two diagnostic outcomes, 50% passive recovery, 100% active
  recovery, and positive EP. EP min/mean/max were 5.58/8.86/12.76. Reference manifest and dataset:
  `gate1-diagnostic-pairs-v0.2-20260903`. This is a generator result, not a learned-policy result.
- Gate 2 confirms a temporal information-value distinction: immediate advance has 0 bits of
  information and mean one-step decision value −0.11; inspect-after-advance has exactly 1 bit and
  mean decision value 8.97.
- Negative methodological result: raw EP correlates with repair margin (Pearson 0.94), transport
  rate (0.76), rate (0.67), graph size (0.49), and the current passive-difficulty proxy (−0.56).
  Raw EP is therefore not accepted as an uncontrolled curriculum score. Reference:
  `gate2-ep-analysis-v0.3-20260903`.
- Negative follow-up: stake-normalized EP removed large Pearson associations with repair margin
  (0.047), transport rate (−0.113), and nominal rate (−0.074), but saturated near one (mean 0.973;
  Q1 0.984) and retained rank associations. It is logged, not adopted as a curriculum ranker.
- Matched-control audit over 100 identical base blocks produced zero standardized differences for
  every structural/cost nuisance and equal fault/cue marginals. Ambiguous cue–fault dependence had
  mean EP 8.86 and 50% passive recovery; revealed dependence had EP 0 and 100% passive recovery.
  Reference: `gate2c-matched-ep-control-v0.3-20260903`.
- First 2k-decision-step Epistemic CPU smoke run completed at commit `d64057e`. The 52k-parameter
  smoke policy chose immediate fixed repair, yielding 50% ambiguous recovery, 50% false repairs, and
  0% experiment-then-correct behavior. This is a preserved negative pipeline smoke, not the
  multi-seed kill test. Reference: `small-ep-smoke-seed00-20260903`.
- A 10k-step primary-size Epistemic pilot at the same code commit also collapsed to immediate fixed
  repair: 50% ambiguous recovery and 0% experiment-then-correct behavior. Training entropy remained
  1.26, but informative inspections were rare and disappeared under deterministic evaluation. This
  preserves the failure before changing public action-sequence masks. Reference:
  `ep-pilot-10k-seed00-20260903`.
- The latent-independent mask pilot at 10k steps still chose immediate fixed repair: 50% ambiguous
  recovery and 0% experiment-then-correct behavior. Masks increased stochastic informative
  inspection only transiently and were insufficient at this budget. Reference:
  `ep-pilot-10k-masked-seed00-20260903`.
- At 30,091 steps, the same masked Epistemic seed reached 100% ambiguous validation recovery and
  100% experiment-then-correct behavior over 128 base pairs. It also probed 100% of revealed-cue
  episodes, demonstrating over-probing rather than uncertainty sensitivity. This is a positive
  single-seed learning primitive with a material limitation, not a curriculum result. Reference:
  `ep-pilot-30k-masked-seed00-20260903`.

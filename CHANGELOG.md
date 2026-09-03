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

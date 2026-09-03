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

### Research status

- Gate 0 local benchmark established median throughput of 3.83M, 4.73M, and 4.41M environment
  steps/s for 1k, 4k, and 16k batches of a 16-node graph, respectively, over 200 ticks and three
  repeats. Reference manifest: `simulator-throughput-v0.1-20260903`.
- The earlier `simulator-throughput-local-20260903` artifact is retained but superseded for hardware
  reporting because the initial manifest did not recover the Linux CPU model.
- The manual pair deterministically gives a 50% shared-repair success ceiling and 100%
  evidence-contingent success across its two worlds. This is an environment construction result, not
  a learned-policy result; exhaustive active optimality is not yet established.

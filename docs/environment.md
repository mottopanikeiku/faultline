# Environment specification

## Scientific purpose

The simulator is a small, fast causal laboratory, not a Factorio emulator. It must expose enough
material-flow dynamics to create hidden failures with common visible symptoms and interventions
whose consequences discriminate among those failures.

## Initial graph vocabulary

A factory is a directed acyclic material-flow graph with five node types:

- **source:** produces material up to its rate and available output capacity;
- **processor:** consumes input and emits transformed material at a bounded rate;
- **buffer:** stores material without transforming it;
- **sink:** consumes material and contributes delivered throughput;
- **transport edge:** transfers material subject to capacity, enablement, blockage, and destination
  space.

State uses dense numeric arrays indexed by stable node and edge IDs. Topology is immutable during an
episode. Given an environment definition, latent fault, and action history, transitions are
deterministic.

## Tick semantics

A tick has explicit phases to prevent iteration-order ambiguity:

1. apply persistent fault constraints and control settings;
2. compute node production/processing from start-of-tick buffers;
3. compute requested edge transfers from the post-processing provisional buffers;
4. resolve competing inbound transfers against destination capacity in stable edge order;
5. consume sink input and record delivered throughput;
6. update counters, recovery streak, cost, reward, and termination.

Material conservation is checked over external source injection, sink delivery, and material held in
node buffers. Faults may prevent movement or processing but never create or delete material.

## Initial latent faults

- **Blocked edge:** sets one transport edge's effective capacity to zero until `clear_blockage`.
- **Failed processor:** sets one processor's effective rate to zero until `replace`.
- **Downstream backpressure:** prevents a designated sink or terminal buffer from accepting material
  until its component is replaced.
- **No fault:** permits false-premise episodes and makes unnecessary repair measurable.

The fault descriptor is privileged state. It must not occur in policy observations, action masks,
object names, array ordering, termination flags, or error messages.

## Actions

- Passive: `inspect(node)`, `measure_flow(edge)`.
- Diagnostic/control: `isolate(edge)`, `toggle(node)`, `advance(ticks)`.
- Repairs: `replace(node)`, `clear_blockage(edge)`.

Inspection reveals documented operational fields only. Isolation and toggling alter dynamics and
therefore can produce causal evidence. Invalid actions return a public error code and pay a cost;
they never reveal whether the requested object is the hidden fault.

## Episode objective

Reward is operational:

\[
r_t = \alpha y_t - \lambda_t - c(a_t) - \lambda_f I[\text{unnecessary repair}],
\]

where \(y_t\) is sink throughput. A terminal bonus is paid only after throughput remains above a
configured fraction of healthy target throughput for a configured number of consecutive ticks.
Information gain is not rewarded.

## Required invariants

- identical seed, scenario, and action history produce bit-identical trajectories;
- total material is conserved after accounting for source injection and sink delivery;
- hidden fault data is absent from recursively inspected agent observations;
- disabling or blocking a component cannot increase its own effective capacity;
- a repair changes dynamics only when its public target and repair type apply;
- sustained recovery cannot be claimed from a single transient high-throughput tick.

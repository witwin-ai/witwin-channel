# Channel Native replacement matrix

This matrix is the human-readable companion to
`channel-api-inventory.json`. The JSON file is authoritative for automated
checks. Counts cover repository-owned Python tests, examples, and benchmarks;
the scan found no production imports in the sibling consumer packages. External
usage remains an explicit migration gate.

| Legacy surface | Reference uses | Priority | Native state |
|---|---:|---|---|
| `Scene/Structure/Transmitter` | 346 | P0 | Supported/partial |
| `path.solve` | 25 | P0 | Partial: one interaction only |
| `path.PathResult` | 6 | P0 | Missing until Phase 1 |
| `cir/cfr/taps/filter_by_type` | 11 | P0 | Missing until Phase 1 |
| `deterministic.solve` | 5 | P1 | Partial |
| `Receiver/ReceiverGrid` | 80 | P1 | Partial/supported |
| `AntennaArray/PlanarArray/ULA/UPA` | 23 | P1/P2 | Missing until Phase 4 |
| `montecarlo.solve` | 6 | P1 | Split into Basic/BDPT partial implementations |

The public `witwin.channel_native.capabilities()` manifest is the launch-time
source of truth. A configuration outside a solver's advertised range must fail
before scene compilation or tensor allocation. Solver results report both the
requested and effective configuration and per-component effective depth.

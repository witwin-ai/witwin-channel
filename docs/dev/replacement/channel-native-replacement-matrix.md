# Channel Native replacement matrix

This matrix is the human-readable companion to
`channel-api-inventory.json`. The JSON file is authoritative for automated
checks. Counts cover repository-owned Python tests, examples, and benchmarks;
the scan found no production imports in the sibling consumer packages. External
usage remains an explicit migration gate.

| Legacy surface | Reference uses | Priority | Native state |
|---|---:|---|---|
| `Scene/Structure/Transmitter` | 346 | P0 | Supported/partial |
| `path.solve` | 25 | P0 | Reflection depth 1-5; flat result excludes coupled geometry |
| `path.solve_v2` coupled topology | 0 | P0/P1 | 1R+1D and reciprocal 1D+1R geometry, globally chunked with a hard 1,000,000-candidate ceiling; complex coefficient pending Phase 3 |
| `path.PathResult` | 6 | P0 | `PathResultV2` shape/signals supported; physical coefficient parity pending |
| `cir/cfr/taps/filter_by_type` | 11 | P0 | Supported by `PathResultV2` |
| `deterministic.solve` | 5 | P1 | Partial |
| `Receiver/ReceiverGrid` | 80 | P1 | Partial/supported |
| `AntennaArray/PlanarArray/ULA/UPA` | 23 | P1/P2 | Missing until Phase 4 |
| `montecarlo.solve` | 6 | P1 | Split into Basic/BDPT partial implementations |
| Differentiable solver workflows | 19 reference files; 0 production consumers | P2 decision | Explicitly unsupported in the first replacement release; every solver accepts only `ad_mode="none"` |

The public `witwin.channel_native.capabilities()` manifest is the launch-time
source of truth. A configuration outside a solver's advertised range must fail
before scene compilation or tensor allocation. Solver results report both the
requested and effective configuration and per-component effective depth.

The AD decision and migration boundary are recorded in
`channel-native-ad-migration.md`. The two native MC LoS derivative primitives
are analytic test/reference surfaces only; they are not reachable through a
public solver and therefore do not change `supports_ad=False`.

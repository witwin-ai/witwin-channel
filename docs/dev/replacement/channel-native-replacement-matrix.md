# Channel Native replacement matrix

This is the human-readable companion to `channel-api-inventory.json`; the JSON
is authoritative for automated checks. Counts cover repository-owned
references. No production imports were found in the audited sibling packages,
but external consumers remain a release blocker.

| Legacy surface | Reference uses | Native state |
|---|---:|---|
| `Scene/Structure/Transmitter` | 346 | Supported within the public capability manifest |
| `path.solve` | 25 | Depth 1-5 reflection, one diffraction with supported R/D coupling, complex field, polarization and angles |
| `path.PathResult` | 6 | Versioned `PathResultV2`, signal views and explicit legacy adapter |
| `cir/cfr/taps/filter_by_type` | 11 | Supported by `PathResultV2` |
| `AntennaArray/PlanarArray/ULA/UPA` | 23 | `AntennaArray`, `ula`/`ura`, synthetic and explicit Path arrays; adapters are required for legacy constructors |
| `Material` | 115 | Phase 7A ABI v2, frequency evaluation, XML fields, explicit PEC and traceable IDs supported |
| Transmission/rough/layered/BSDF events | 0 observed production uses | Phase 7B not integrated; every related capability is false |
| Differentiable solver workflows | 19 reference files; 0 production consumers | First release is primal-only; `supports_ad=False` |
| Deterministic and Monte Carlo public surfaces | 140 | Native solver-specific APIs exist; legacy API parity remains partial |

Phase 9 provides a benchmark harness, memory preflight and import-safe
deployment diagnostics. Pipeline cache is not implemented. The SM list is a
build declaration without runtime matrix evidence, and wheel smoke is not
verified. None of those three may be described as accepted deployment gates.

Production Python under `src` may not import `drjit`, `mitsuba`, `sionna`,
Python `raydn`, or `witwin.channel`. Tests and benchmarks may retain the old
stack only as an offline oracle. `ci/check_production_dependencies.py` enforces
this boundary and can also scan sibling repository roots.

Phase 10 is not a deletion approval. See `channel-native-migration.md` for the
shadow artifact, default-on boundary, and outstanding external/two-release
blockers.

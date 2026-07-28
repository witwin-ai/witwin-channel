# Path domain

The path domain owns the public explicit-path solver and the stable conversion
from typed evaluated paths into antenna-aware signal views. Topology discovery,
geometry re-evaluation, and electromagnetic field evaluation remain owned by
`propagation`.

The domain is one module, `witwin/channel/path.py`. This README lives under
`docs/dev/` because every owner document does; no documentation remains inside
the package tree.

## Ownership

- `solve` and the `_pipeline_solve` / `_pipeline_solve_base` stages orchestrate
  one path solve and array expansion; they do not own propagation algorithms.
- `Config` owns path-solver validation and feature limits.
- `RaggedPathSoA` owns the stable per-link ragged structure.
- `PathResult` owns padded validation, interaction tags, and signal views such
  as CIR, CFR, taps, filtering, and beamforming.
- `pack_synthetic_arrays` / `explicit_array_scene` / `pack_explicit_arrays` own
  synthetic far-field packing and explicit per-element scene expansion.
  `_metadata` owns truthful solver metadata.
- `propagation.enumerated.evaluate_enumerated_paths` is the typed
  discovery/evaluation owner consumed by this package.
- ADR-027 straight-transmission discovery reaches that owner as one pair-major
  `EnumeratedFullDistance` RayD batch. Path does not own a closest-hit march or
  depth loop. It carries the typed solve transaction through scattering,
  sanitizes evaluated rows and the diffraction sidecar before result packing,
  and enqueues the runtime-owned terminal observer exactly once after synthetic
  or explicit array packing.
- Under ADR-032, Path keeps one explicit post-sanitizer valid-row structural
  compaction before the result converter. This prevents failed `-1` identifiers
  from reaching that converter and is part of the authoritative `O(K)` compact
  production route, not a pending no-D2H blocker.

## Public entry points

`witwin.channel.path` exports exactly:

- `Config`
- `InteractionType`
- `PathResult`
- `RaggedPathSoA`
- `solve(scene, config)`
- `pack_synthetic_arrays(...)`
- `explicit_array_scene(scene)`
- `pack_explicit_arrays(...)`

These entries are frozen in `ci/public-api-snapshot.json`. Functions prefixed
with `_`, pipeline injection seams, packing helpers not listed above, and
`BeamformedPathResult` are internal implementation details even when tests
import them directly.

## Dependency rules

- Path may depend on scene contracts and the typed
  `propagation.enumerated` engine. It must not import deterministic or Monte
  Carlo solver internals, and propagation code must not depend back on path
  result types.
- The solver must not import the raw native extension,
  `core.kernels.ops`, or legacy Python RayD/DrJit paths. Native requirements
  are reached through the owning propagation/runtime facades.
- Path owns packing, not topology or field kernels. New physics belongs in the
  relevant propagation/material/scattering owner and is consumed through a
  typed contract.
- Cross-domain imports are absolute and every new dependency must satisfy
  `ci/check_import_graph.py` without adding new allowlisted debt.

## Numerical and AD contract

- Solving requires CUDA and channel path kernels. Reflection and
  diffraction additionally require the typed RayD-native scene capability.
- `PathResult.a` has shape
  `(rx, rx_ant, tx, tx_ant, path, time)` and dtype `complex64`.
  Path scalars and Cartesian geometry use `float32`; interaction, primitive,
  material, and path-count tensors use `int32`; validity is boolean. All
  result tensors share one device.
- Delay `tau` is in seconds. Positions are Cartesian metres. Endpoint angles
  are radians with `theta = acos(z / |d|)` and
  `phi = atan2(y, x)`; receiver angles use the source-facing arrival
  direction.
- The phasor convention is `exp(-j*k*d)` with time dependence
  `exp(+j*2*pi*f*t)`. CFR evaluation therefore applies
  `exp(-j*2*pi*tau*f)`. Synthetic array steering applies
  `exp(+j*k*element_position_dot_endpoint_direction)`.
- Under ADR-032, the `path` axis and `max_num_paths` represent actual compact
  rows. The owning allocation boundary may copy only audited integer count
  metadata and explicitly synchronize to allocate exact `O(K)` storage; the
  frozen depth-3 Munich reflection budget is at most six 4-byte copies. Public
  `path_capacity_per_pair`, `diffraction_state_capacity`, capacity-shaped
  PathResult validity/counts, and ADR-031 `Qr` are not solver contracts.
  Structural compaction must preserve stable row order and publish either every
  valid row or no usable result; silent truncation and partial success are
  forbidden.
- Ragged paths are stably grouped by pair before padding. Synthetic arrays
  share a centre geometric path set and use far-field phase weighting;
  explicit arrays trace element positions independently and currently require
  point receivers.
- `path.capacity.from_capacity_evaluated_paths` is a caller-free ADR-029 native
  experiment retained for direct tests. It consumes the fixed
  receiver-major pair layout, preserves path capacity `C`, derives endpoint
  angles and canonical interaction storage in one native row pass, and carries
  CUDA `valid`/`num_paths` without a host count. Its primal, backward, and JVP
  operations all gate validity before endpoint identifiers or numerical
  payloads. The shared failure state or upstream overflow makes the complete
  packed result inert; this producer does not trap and may not switch or appear
  as an alternate to the live compact solver.
- Interaction bits are `REFLECTION=1`, `DIFFRACTION=2`,
  `TRANSMISSION=4`, and `SCATTERING=8`; `NONE=0` denotes LoS/no
  interaction.

### AD contract

- `Config.ad_mode` accepts `"none"`, `"jvp"`, and `"vjp"`.
  Differentiation is through the fixed discrete topology selected by the
  forward solve. There is no estimator for visibility discontinuities,
  shadow transitions, or path birth/death.
- Supported fixed-topology derivatives include material leaves, frequency,
  transmitter/receiver positions, and mesh vertices where the selected
  component's native companion supports them.
- Scattering with JVP/VJP is rejected before launch. Coupled
  reflection-diffraction paths do not support mesh-vertex gradients; their
  selected wall plane and edge tables are fixed. Unsupported combinations must
  fail loudly rather than return detached or zero gradients.

## Forbidden fallback

Missing CUDA, path-native kernels, required native symbols, or requested RayD
capabilities is a hard error. Do not add CPU/Python/Torch geometry
recomputation, finite differences, legacy RayD/DrJit dispatch, silent empty
paths, or reference-oracle execution as a fallback. A legitimately empty scene
result is distinct from an unavailable backend and must retain truthful
metadata.

## Maintenance

- Any change to the eight stable exports requires an intentional
  `ci/public-api-snapshot.json` update and a migration note.
- The completed wrapper migration ledger is immutable historical evidence at
  `docs/dev/audit/phase12-ops-migration-ledger.json`. Native ABI changes require
  the relevant binding/audit manifest update.
- Dependency changes must pass `ci/check_import_graph.py`; do not create or
  relocate allowlisted architecture debt.
- Update result/schema contract tests and this README when shapes, dtypes,
  phase conventions, interaction IDs, array semantics, AD exclusions, or
  fail-loud behavior change.

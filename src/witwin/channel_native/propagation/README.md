# Propagation owner contract

## Ownership

This package owns row-aligned topology, continuous geometry, RF fields, and
the enumerated propagation stages shared by Path and Deterministic solvers.
`EvaluatedPaths` is the internal composition boundary; solver accumulation and
public result types remain solver-owned. Under ADR-024 and the completed Phase
6B pin/switch, RayD is the numerical source owner of the complete-row
transmission primal/backward/JVP family, while row contracts, field facades, AD
dispatch, topology, and result assembly remain owned here.

The Phase 6A shared RF dependency closure is active: retained Channel field,
coupled-diffraction, BDPT, and scattering kernels consume RayD public RF
device headers directly. They do not retain or reconstruct a Channel-private
copy of complex, Fresnel, layer-stack, Jones, or field-transport AD math. The
Phase 6B move did not split the complete-row fusion or move the BDPT
transmitted-state family, which remains a complete Channel numerical owner.

ADR-025 freezes diffraction ownership by complete operation. After the atomic
Phase 8A pin/switch/delete, RayD is the sole numerical owner of the pure-wedge
fixed-winner primal/backward/JVP family; this package retains the stable ABI,
typed field/autograd facades, and solver orchestration only. MC Sionna
fixed-tape and coupled R-D/D-D families stay complete Channel numerical owners
and may use RayD public device primitives without adding a UTD sub-launch. Pure
wedge keeps its exporter-locked fast-math contract; retained MC and coupled
operations stay precise. Phase 8B separately owns the sample-tape semantic
rename and native transmitter-visible state planning/selection cleanup.

## Public entry points

There are no root public API exports. The internal package export surface is
`PathTopology`, `PathGeometry`, `PathFields`, and `EvaluatedPaths`.
`propagation.enumerated` owns shared component stages; solver-specific result
conversion stays outside.

## Dependency rules

Topology cannot depend on scene, continuous fields, geometry, or solver policy.
Geometry cannot choose winners; geometry kernel modules depend on runtime, not
scene. Fields cannot discover topology. Enumerated propagation serves Path and
Deterministic; Monte Carlo sampling, MIS, solver results, and deterministic
accumulation remain outside this package. Field facades may call the typed RayD
transmission and pure-wedge entries but cannot reconstruct or fall back from
their native math.

Raw native tuples may exist only inside domain `kernels` modules. A kernel
façade must validate and convert them to a named internal contract before any
solver or propagation pipeline can observe the result.

## Numerical and AD contract

Contract construction is metadata-only and zero-copy. Row order, row identity,
tensor object/storage aliasing, stride, dtype, device, and `requires_grad` must
remain exact. Topology IDs and winners are discrete; geometry and fields use SI
units and the package complex-phase convention.

### AD contract

Fixed-topology reevaluation cannot change winner selection or conceal a detach
boundary. Continuous endpoint, vertex, material, frequency, and field leaves
use explicit native backward/JVP companions. Unsupported tangents and
higher-order derivatives fail before returning a partial result.

## Forbidden fallback

Propagation code must fail loudly when a required native capability or contract
is missing. It cannot recompute geometry on CPU/Torch, load a global extension,
return a zero result, or silently switch algorithms.

## Maintenance

New internal exports or owner moves require a migration note and contract
tests. The completed kernel migration ledger is archived at
`docs/dev/audit/phase12-ops-migration-ledger.json`; dependency changes must pass
the import graph. A new root or solver public export also updates
`ci/public-api-snapshot.json`.

# Scene owner

## Ownership

`scene` is the future owner of scene compilation, canonical GPU stores, cache
invalidation, and RayD resource lifetime. The current implementations remain in
`core.scene`, `core.scene_compile`, and `core.runtime` until their dedicated
move-only phase.

## Public entry points

This skeleton intentionally exports nothing. `witwin.channel_native.Scene`
continues to resolve to `core.scene.Scene`; changing that target requires the
public API snapshot and a migration facade.

## Dependency rules

Scene may depend on runtime resource contracts, geometry inputs, and material
encoding. It must not import a solver, propagation pipeline, or scattering
accumulator. Mutable native handles and caches remain private to the compiled
scene owner.

## Numerical and AD contract

Compilation and cache invalidation must preserve tensor storage, device,
ordering, material ABI, geometry identity, RayD handle lifetime, and existing
fixed-topology AD behavior. Boundary moves may not add copies or launches.

## Forbidden fallback

Production scene construction may not fall back to CPU geometry, a Python ray
tracer, reconstructed material tensors, or a global RayD extension. Missing or
stale native resources fail before solver execution.

# Materials owner

## Migration state

The current public module is `witwin.channel_native.materials`, implemented by
`src/witwin/channel_native/materials.py`. A same-named package would shadow that
module on import, so Phase 3 deliberately does not create `materials/` or move
its implementation. Conversion requires a dedicated move-only change with an
identity-preserving facade and API snapshot evidence.

## Ownership

Materials own immutable material definitions, frequency evaluation, layer and
roughness parameters, surface assignment, and canonical native material
encoding. They do not own scene caches, scattering sampling, path enumeration,
or solver accumulation.

## Public entry points

`witwin.channel_native.materials` and the material names already re-exported by
`witwin.channel_native` remain stable. No Phase 3 export is added or removed.

## Dependency rules

Material definitions may depend on standard-library value types and physics
contracts. Encoding may depend on a domain kernel facade, but material modules
must not import scene or solver packages or retain native handles.

## Numerical and AD contract

Frequency units, passive-branch conventions, layer order, dtype/device,
material IDs, tensor aliasing, cache tokens, and material/frequency gradients
remain exact during migration.

## Forbidden fallback

No production path may silently replace native material encoding with host
floats, detach differentiable values, approximate a dispersive model, or
substitute a default material after validation fails.

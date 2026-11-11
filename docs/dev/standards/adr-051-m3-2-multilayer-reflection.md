# ADR-051: Reflection off a multilayer material evaluates the whole layer stack

- **Status:** Accepted.
- **Date:** 2026-11-15.
- **Decision ID:** `m3-2-multilayer-reflection`.
- **Scope:** specular reflection in the enumerated field kernels (`field_reflection_sequence`, its
  frequency-column, backward and JVP companions; deterministic, Path and consumer solves), the
  reflection leg of the coupled R-D / D-R field kernel, and the Monte Carlo Basic reflection
  radiomap with its AD companions. Finding A-1 of the reference-accuracy suite.

## Context

A-1: reflection off `PhysicalMaterial(layers=...)` with more than one layer equalled the reflection of
the first layer alone (up to 127 % off a three-layer wall), while transmission was correct.

Root cause: the compiled material store keeps two views of a material, the scalar per-face view
(`eps_r`, `sigma_e`, `mu_r`, `thickness`) built from the **first layer** (`scene/compiler.py`, "ABI v3
scalar Fresnel view") and the CSR layer store (`layer_offset`, `layer_count`, per-layer tables). Every
reflection kernel read only the scalar view and evaluated one slab (`transport::slab_fresnel`, or
`legacy_slab_fresnel` in Monte Carlo); transmission reads the CSR store through RayD's layer-stack
recursion (`em::stack_rt`), which is why it was right.

## Decision

1. `native/channel/kernels/layered_reflection.cuh` owns the per-face choice: a material with more than
   one layer reflects with RayD's stack recursion (`em::stack_rt`, TE and TM, the TM sign converted to
   the s / p = s x k convention of the slab law, times the face gain); a single-layer material keeps the
   existing slab expression, so single-layer results are bitwise unchanged. Forward-mode twins use
   `ad::stack_rt_dual` with the incidence-angle and frequency tangents.
2. The kernels take the per-bounce material id and the CSR store (`ReflectionLayers` in
   `kernels.fields`; a seven-tensor bundle for Monte Carlo). Python passes the store only when the
   compiled scene has a multilayer material (`materials.has_multilayer_material`, read from the host
   store), so single-layer scenes run exactly the previous launches.
3. Derivatives. Geometry (incidence angle), gain and frequency derivatives of a multilayer face are
   exact. Layer-parameter derivatives (`eps_r`, `sigma_e`, `thickness` of each layer) are not
   implemented for reflection; requesting them on a scene with a multilayer material raises
   `NotImplementedError` before any native work. The per-face scalar view cannot carry them: it is
   the first layer, not the stack.
4. Coupled R-D / D-R: the primal reflection leg uses the stack; coupled AD on a multilayer scene
   raises `NotImplementedError` (its dual leg is not layered).
5. BDPT: its sampled connections still use the slab law. Rather than mix laws, BDPT refuses
   reflection on a scene with a multilayer material (`NotImplementedError`).
6. Wedge faces of diffraction keep their single-slab / half-space face coefficients; layered face
   reflection in the UTD coefficient is not part of this record.

## Evidence

- `tests/accuracy/test_reference.py::test_slab_reflection_matches_layer_stack_oracle[multilayer]`
  (strict xfail removed): error against the float64 layer-stack oracle, TE and TM, three angles.
- `tests/accuracy/test_consistency.py::test_multilayer_wall_reflection_across_solvers`: deterministic,
  Path and Monte Carlo Basic against the image solution with the stack coefficients.
- `tests/propagation/fields/test_layered_reflection.py`: a single-layer store is bitwise the scalar
  slab; the multilayer geometry JVP matches a central difference; material derivatives are refused.

## Consequences

- Multilayer reflections move to the stack value everywhere listed above; nothing else moves.
- BDPT and coupled AD on multilayer scenes fail loudly instead of returning the first-layer value.

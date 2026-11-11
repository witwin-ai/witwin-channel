# ADR-048: Scene import and material/antenna presets

- **Status:** Accepted.
- **Date:** 2026-11-10.
- **Decision ID:** `scene-import-and-presets`.
- **Scope:** two new public modules, `witwin.channel.importers` and
  `witwin.channel.presets`, their ownership, and their dependency limits.

## Context

New users need to bring scenes in from the formats other ray tracers and
modelling tools use (Mitsuba 3 / Sionna RT XML, PLY, OBJ, glTF) and need named
radio materials (ITU-R P.2040) and standard element patterns (3GPP TR 38.901,
dipoles). Before this record the only loader was `witwin.core.Mesh.from_obj`,
with no material assignment, and every material had to be typed in by hand.

ADR-034 and ADR-036 put the logical world (`Scene`, `Structure`, `Mesh`,
`PhysicalMaterial`, `AntennaState`, `AntennaPattern`) in `witwin.core` and
forbid Channel from re-exporting or duplicating it. `witwin.core` is an
external, released dependency (`witwin==0.4.0`), so a loader cannot be added
there in the same release as Channel.

What Core already supports, checked against `witwin` 0.4.0:

- `PhysicalMaterial(dispersion=PowerLawDispersion(reference_frequency_hz,
  eps_r_exponent, sigma_e_exponent))` scales `eps_r` and `sigma_e` by
  `(f / f_ref)**exponent`. With `f_ref = 1 GHz` this is exactly the ITU-R
  P.2040 law `a * f_GHz**b`, `c * f_GHz**d`. Channel's scene compiler evaluates
  a `PowerLawDispersion` once at the compile frequency.
- `AntennaPattern` kinds `isotropic`, `vertical` (unit-peak short dipole),
  `horizontal`, and `custom` with a callable from local direction to complex
  field amplitude. The Path solver evaluates the pattern; the other solvers
  accept only isotropic endpoints.

## Decision

1. **Owners.** `witwin.channel.importers` owns file parsing (Mitsuba XML, PLY,
   OBJ, glTF, Mitsuba `serialized`), `to_world` transforms, vertex welding and
   radio-material assignment. `witwin.channel.presets` owns the ITU-R P.2040
   table and the TR 38.901 and dipole pattern factories. Both are public
   entry points in `ci/public-api-snapshot.json` and in
   `ci/check_orphan_modules.py`.
2. **Core values only.** Both modules return `witwin.core` objects
   (`Structure`, `Mesh`, `Scene`, `PhysicalMaterial`, `AntennaPattern`) and
   define no world type. Their own types are import records (`MeshData`,
   `StructureReport`, `ImportedScene`) and preset rows (`ItuBand`).
   `ItuBandDispersion` subclasses Core's `PowerLawDispersion` only to add the
   ITU validity range; Core and Channel treat it as a `PowerLawDispersion`.
3. **Dependencies.** NumPy and the standard library only; no `trimesh`,
   `pygltflib`, Mitsuba or Sionna import (`ci/check_production_dependencies.py`
   already forbids the latter two). The modules import no solver, no
   `runtime`, no `kernels` and no extension, so `import witwin.channel.importers`
   works without CUDA.
4. **Compute policy.** Parsing, transforms and welding are import-time
   preparation of a logical world, the same class of work as authoring a
   `Mesh` by hand; they run once per file on the host and are not per-solve
   physics. The pattern factories return Core `custom` callables; the Path
   solver evaluates every Core pattern (built-in or custom) in its existing
   Torch pattern step, so the presets add no new evaluation site. Moving
   pattern evaluation into native code is a separate change for all pattern
   kinds.
5. **Frequency validity.** `itu_material` refuses a frequency or band outside
   the material's ITU range. With `dispersive=True` (default) the material
   carries `ItuBandDispersion`, whose `scale` also refuses an out-of-range
   Python-number frequency, so compiling the scene outside the range fails
   before native work. A tensor frequency is not read back to the host and is
   not range-checked. `dispersive=False` evaluates the law once and yields a
   frequency-independent material, which the wideband consumer route accepts.
6. **Welding by default.** Importers weld vertices within 1e-6 of the bounding
   box diagonal and drop degenerate faces, and report both per structure.
   Duplicated seam vertices turn shared edges into boundary edges, which the
   diffraction edge policy treats as half-planes.

## Consequences

- The public API grows from six to eight modules (78 exports). The snapshot
  and `tests/test_public_api_snapshot.py` were updated deliberately.
- Sionna RT scenes load directly from a user's Sionna installation; Channel
  ships no third-party scene assets. Test fixtures are authored in-repository.
- If `witwin.core` later gains loaders or presets, these modules move there
  complete and Channel deletes them; no alias is kept.
- Not supported: Mitsuba `include`, `shapegroup`/`instance`, analytic shapes,
  glTF compression extensions, sparse accessors and quantized positions. Each
  raises a `ValueError` naming the element.

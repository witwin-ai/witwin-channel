# Scene import

`witwin.channel.importers` turns mesh files into `witwin.core` structures. It
uses only NumPy and the Python standard library, never runs code from the
files it reads, and returns plain Core values: `Structure` objects holding a
`Mesh` and a material, which you put into a `witwin.core.Scene` yourself or
with `ImportedScene.to_scene(endpoints=...)`.

| Format | Function | Notes |
| --- | --- | --- |
| Mitsuba 3 / Sionna RT XML | `load_mitsuba(path, frequency_hz=...)` | Shapes, transforms and radio materials, see below |
| Wavefront OBJ | `load_mesh(path, materials)` or `read_obj(path)` | Polygons are fan-triangulated; one mesh per `usemtl` |
| PLY (ASCII, binary little and big endian) | `load_mesh(path, materials)` or `read_ply(path)` | `vertex_indices` lists of any length; extra properties ignored |
| glTF 2.0 (`.glb`, `.gltf`) | `load_mesh(path, materials)` or `read_gltf(path)` | Node hierarchy (TRS or matrix), triangle lists, strips and fans; external `.bin` or data-URI buffers |

## One mesh file

```python
from witwin.core import PhysicalMaterial
from witwin.channel.importers import load_mesh
from witwin.channel.presets import itu_material

imported = load_mesh(
    "district.glb",
    {
        "Facade": itu_material("concrete", 3.5e9),
        "Window": itu_material("glass", 3.5e9),
        None: itu_material("concrete", 3.5e9),   # anything without a listed material
    },
)
print(imported.summary())
for entry in imported.report:
    print(entry.name, entry.material, entry.faces, entry.welded_vertices, entry.removed_faces)
```

`materials` is either one material for the whole file or a mapping from the
file's material names: the OBJ `usemtl` name or the glTF material `name`
(`material_<index>` when unnamed). A missing name raises a `ValueError` that
lists the known keys; add a `None` entry as the default.

glTF is +Y up. `read_gltf` and `load_mesh` rotate it to Channel's +Z up by
default (`(x, y, z) -> (x, -z, y)`); pass `up_axis="y"` to keep the file's
coordinates. Points and lines in a glTF file are skipped and listed in
`ImportedScene.warnings`. Draco/meshopt compression, sparse accessors and
quantized positions are rejected with a clear error.

## Mitsuba and Sionna RT XML

```python
from witwin.channel.importers import load_mitsuba

imported = load_mitsuba("scenes/munich/munich.xml", frequency_hz=3.5e9)
scene = imported.to_scene(endpoints=[tx, grid])
```

Supported:

- Shapes `ply`, `obj`, `serialized`, `rectangle` and `cube`, with
  `flip_normals`. File names are relative to the XML file.
- `<transform name="to_world">` with `translate`, `rotate` (axis plus angle in
  degrees), `scale` (uniform or per axis), `matrix` (3x3 or 4x4) and `lookat`,
  composed in Mitsuba order. A mirroring transform flips the winding so faces
  keep their orientation.
- `<default name=... value=...>` substitution of `$name` in attributes.
- Materials, by bsdf `id`:
  - `itu-radio-material` (Sionna RT 1.x/2.x): `type` names an ITU-R P.2040
    material, evaluated with `presets.itu_material` at `frequency_hz`;
    `thickness`, `scattering_coefficient` and `xpd_coefficient` are used.
  - `radio-material`: `relative_permittivity`, `conductivity`, `thickness`,
    `scattering_coefficient`, `xpd_coefficient`.
  - An id of the form `itu_<name>` or `mat-itu_<name>` (older Sionna scenes)
    maps to that ITU material whatever its bsdf type.
  - `twosided` wrappers are unwrapped.
  - `materials={"bsdf id": PhysicalMaterial(...)}` overrides any of these and
    is the way to use a bsdf Channel does not understand (for example a
    visual `diffuse` bsdf). `default_material=` covers shapes with no bsdf.

Not supported, and rejected with an error naming the element: `<include>`,
`shapegroup`/`instance`, and analytic shapes such as `sphere`, `disk` and
`cylinder` (tessellate them in your modelling tool first). Emitters, sensors,
integrators and films are ignored.

`frequency_hz` is required because ITU materials are only defined over given
frequency ranges; an out-of-range frequency fails at import time and again if
the scene is compiled at a frequency outside the range. Pass a
`(low_hz, high_hz)` band if you will compile the same scene at several
frequencies.

`merge_shapes=True` (default) merges every shape that uses the same bsdf into
one structure, which keeps compile time low for city scenes with thousands of
shapes; `merge_shapes=False` keeps one structure per shape, named after the
shape `id`.

## Vertex welding

Exported meshes often duplicate vertices along seams (one copy per face or per
part). Duplicated vertices split shared edges into two boundary edges, which
creates spurious diffraction edges and misses real wedges. All importers
therefore weld vertices by default:

- vertices that round to the same grid cell of size `weld_tolerance` metres
  merge (default: 1e-6 of the bounding-box diagonal of the imported meshes);
- triangles that lose a distinct corner or have zero area are removed;
- vertices no triangle uses are dropped.

`ImportedScene.report` records, per structure, the input and output vertex
counts, `welded_vertices` and `removed_faces`, and `summary()` prints the
totals. Pass `weld=False` to keep the file's vertices exactly (degenerate
faces are still removed), or a larger `weld_tolerance` for noisy exports.
Welding is per structure: two structures never share vertices.

## Using the Sionna RT example scenes

The scenes that ship inside the `sionna-rt` wheel (Apache-2.0) load directly
from their installed location; Channel does not redistribute them. Checked
with the Sionna RT 2.x scene set:

| Scene | Structures | Vertices | Faces | Welded | Import time |
| --- | ---: | ---: | ---: | ---: | ---: |
| `simple_street_canyon` | 5 | 52 | 74 | 0 | 0.02 s |
| `munich` | 5 | 31,910 | 38,936 | 2,134 | 1.5-2.4 s |
| `florence` | 3 | 32,874 | 35,373 | 3,316 | 3.2 s |
| `san_francisco` | 3 | 177,234 | 268,041 | 125 | 5.0 s |

Sionna scenes use `metal`, `marble`, `brick`, `wood`, `glass` and `concrete`,
all of which are in the preset table at 1-100 GHz (marble to 60 GHz, brick to
40 GHz).

## Low-level pieces

`read_ply`, `read_obj` and `read_gltf` return `MeshData` (NumPy `vertices`,
`faces`, `name`, `material` key) without creating Core objects, and
`build_structures(meshes, materials, merge=..., weld=...)` turns any list of
`MeshData` into an `ImportedScene`. `weld_vertices(vertices, faces,
tolerance=...)` is the welding step on its own.

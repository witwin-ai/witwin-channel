# `witwin.channel.importers`

Scene import from Mitsuba 3 / Sionna RT XML, PLY, OBJ and glTF 2.0 files into
`witwin.core` structures. The module uses NumPy and the standard library only
and does no GPU work. User guide: [Scene import](../user/scene-import.md).

```python
from witwin.channel.importers import (
    ImportedScene, MeshData, StructureReport, build_structures, load_mesh, load_mitsuba,
    read_gltf, read_obj, read_ply, weld_vertices,
)
```

## `load_mitsuba`

```text
load_mitsuba(
    path, *, frequency_hz: float | tuple[float, float],
    materials: Mapping[str, MaterialSpec] | None = None,
    default_material: MaterialSpec | None = None, merge_shapes: bool = True,
    weld: bool = True, weld_tolerance: float | None = None, dispersive: bool = True,
) -> ImportedScene
```

| Parameter | Description |
| --- | --- |
| `path` | Mitsuba XML file; shape file names resolve relative to it. |
| `frequency_hz` | Frequency or `(low_hz, high_hz)` band the scene will be compiled at; ITU materials must be valid there. |
| `materials` | bsdf id to material; overrides the bsdf in the file. |
| `default_material` | Material for shapes without a bsdf. |
| `merge_shapes` | Merge shapes that share a bsdf into one structure. |
| `weld`, `weld_tolerance` | See `build_structures`. |
| `dispersive` | Passed to `presets.itu_material` for ITU bsdfs. |

Shapes: `ply`, `obj`, `serialized`, `rectangle`, `cube`. Transforms:
`translate`, `rotate`, `scale`, `matrix`, `lookat`. Bsdfs:
`itu-radio-material`, `radio-material`, `twosided` (unwrapped), and ids
`itu_<name>`/`mat-itu_<name>`. Raises `ValueError` for unsupported shapes or
elements (`include`, `shapegroup`, `instance`), unknown bsdfs without an
override, unknown ITU names and out-of-range frequencies.

## `load_mesh`

```text
load_mesh(
    path, materials: MaterialSpec | Mapping[str | None, MaterialSpec], *, merge: bool = True,
    weld: bool = True, weld_tolerance: float | None = None, up_axis: str = "z",
) -> ImportedScene
```

Imports one `.ply`, `.obj`, `.glb` or `.gltf` file. `materials` is one material
or a mapping from the file's material names (`None` is the default entry).
`up_axis` applies to glTF only.

## `build_structures`

```text
build_structures(
    meshes: Sequence[MeshData], materials: MaterialSpec | Mapping[str | None, MaterialSpec], *,
    merge: bool = True, weld: bool = True, weld_tolerance: float | None = None,
    warnings: Sequence[str] = (),
) -> ImportedScene
```

Creates one `Structure` per mesh, or per material key with `merge=True`. Each
structure holds a Core `Mesh` (CPU float32 vertices, int64 faces, no
recentering) in world coordinates. With `weld=True`, vertices within
`weld_tolerance` metres (default 1e-6 of the bounding-box diagonal) merge;
faces with a repeated corner or zero area are removed in either case. Raises
`ValueError` when there are no triangles or a material key has no entry.

## `ImportedScene`

Frozen dataclass.

| Field / member | Description |
| --- | --- |
| `structures` | `tuple[witwin.core.Structure, ...]` |
| `materials` | material key to the `MaterialSpec` used |
| `report` | `tuple[StructureReport, ...]`, one per structure built |
| `warnings` | skipped content, for example non-triangle glTF primitives |
| `vertex_count`, `face_count`, `welded_vertex_count` | totals |
| `to_scene(*, endpoints=(), metadata=None)` | a `witwin.core.Scene` with these structures |
| `summary()` | one-line text summary |

## `StructureReport`

Frozen dataclass: `name`, `material`, `source_meshes`, `input_vertices`,
`output_vertices`, `welded_vertices`, `removed_faces`, `faces`.

## `MeshData`

Frozen dataclass: `name`, `vertices` (`(V, 3) float64`), `faces`
(`(F, 3) int64`), `material` (key or `None`). Validates shapes, finiteness and
index range. `transformed(matrix)` applies a 4x4 affine matrix and flips the
winding when the matrix mirrors.

## Readers

| Function | Returns |
| --- | --- |
| `read_ply(path, *, name=None, material=None)` | `MeshData` |
| `read_obj(path)` | `tuple[MeshData, ...]`, one per `usemtl` group, each with only its own vertices |
| `read_gltf(path, *, up_axis="z")` | `(tuple[MeshData, ...], tuple[str, ...] warnings)` |
| `weld_vertices(vertices, faces, *, tolerance)` | `(vertices, faces, welded_count, removed_face_count)` |

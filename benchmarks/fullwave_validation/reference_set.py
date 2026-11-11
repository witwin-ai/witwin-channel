# Copyright Xingyu Chen.
# Tidy3D reference set of synthetic blocks and plates: case specs, simulations, regions, storage.

"""Tidy3D reference set of synthetic blocks and plates: case specs, simulations, regions, storage."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


MANIFEST_PATH = Path(__file__).parents[1] / "scenarios" / "fullwave_reference.v1.json"
DATA_DIR = Path(__file__).parents[2] / "tests" / "data" / "fullwave"
SCHEMA_NAME = "witwin.channel.fullwave-reference-map"
SCHEMA_VERSION = 1
SPEED_OF_LIGHT = 299792458.0
METERS_TO_MICROMETERS = 1.0e6
DECIMATION = 4
REGION_NAMES = ("invalid", "lit", "shadow", "transition", "corner")
INVALID, LIT, SHADOW, TRANSITION, CORNER = range(5)
CORNER_FRESNEL_FRACTION = 0.15
_AXES = {"x": 0, "y": 1, "z": 2}


@dataclass(frozen=True, slots=True)
class Block:
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    material: dict[str, Any]

    @property
    def lo(self) -> np.ndarray:
        return np.asarray(self.center) - 0.5 * np.asarray(self.size)

    @property
    def hi(self) -> np.ndarray:
        return np.asarray(self.center) + 0.5 * np.asarray(self.size)

    @property
    def is_plate(self) -> bool:
        return min(self.size) == 0.0

    @property
    def is_slab(self) -> bool:
        return self.material["kind"] == "slab"


@dataclass(frozen=True, slots=True)
class ReferenceCase:
    case_id: str
    role: str
    frequency_hz: float
    polarization: str
    grid_m: float
    blocks: tuple[Block, ...]
    tx_position: tuple[float, float, float]
    analysis_bounds_xy: tuple[tuple[float, float], tuple[float, float]]
    planes_z: tuple[float, ...]
    domain_bounds_xyz: tuple[tuple[float, float], ...]
    spec: dict[str, Any]
    refine: tuple[dict[str, Any], ...] = ()
    run_time_periods: float = 60.0
    calibration: str | None = None

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.spec, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def wavelength_m(self) -> float:
        return SPEED_OF_LIGHT / self.frequency_hz

    @property
    def polarization_vector(self) -> np.ndarray:
        vector = np.zeros(3)
        vector[_AXES[self.polarization]] = 1.0
        return vector


def load_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def case_ids(role: str | None = None) -> tuple[str, ...]:
    cases = load_manifest()["cases"]
    return tuple(name for name, entry in cases.items() if role is None or entry["role"] == role)


def load_reference_case(case_id: str) -> ReferenceCase:
    manifest = load_manifest()
    entry = manifest["cases"][case_id]
    geometry = manifest["geometries"][entry["geometry"]]
    material_name = entry["material"]
    blocks = []
    for raw in geometry["blocks"]:
        name = material_name if raw["material"] == "$material" else raw["material"]
        blocks.append(Block(tuple(raw["center"]), tuple(raw["size"]), dict(manifest["materials"][name])))
    frequency_hz = float(entry.get("frequency_hz", manifest["frequency_hz"]))
    spec = {"case_id": case_id, "frequency_hz": frequency_hz, "case": entry,
            "geometry": geometry, "blocks": [
                {"center": b.center, "size": b.size, "material": b.material} for b in blocks]}
    return ReferenceCase(
        case_id=case_id, role=entry["role"], frequency_hz=frequency_hz,
        polarization=entry["polarization"], grid_m=float(entry["grid_m"]), blocks=tuple(blocks),
        tx_position=tuple(geometry["tx_position"]),
        analysis_bounds_xy=tuple(tuple(row) for row in geometry["analysis_bounds_xy"]),
        planes_z=tuple(geometry["planes_z"]),
        domain_bounds_xyz=tuple(tuple(row) for row in geometry["domain_bounds_xyz"]), spec=spec,
        refine=tuple(entry.get("refine", ())),
        run_time_periods=float(entry.get("run_time_periods", 60.0)),
        calibration=entry.get("calibration"))


# --------------------------------------------------------------------------- Tidy3D


def build_tidy3d_simulation(case: ReferenceCase) -> Any:
    import tidy3d as td

    um = METERS_TO_MICROMETERS
    structures = []
    for index, block in enumerate(case.blocks):
        for part, (center, size, material) in enumerate(_volumes(block)):
            if material["kind"] == "pec":
                medium = td.PECMedium(name="pec")
            else:
                medium = td.Medium(permittivity=material["eps_r"],
                                   conductivity=material["sigma_e"] / um, name="dielectric")
            name = f"block-{index}" if not block.is_slab else f"block-{index}-layer-{part}"
            structures.append(td.Structure(
                geometry=td.Box(center=tuple(v * um for v in center), size=tuple(v * um for v in size)),
                medium=medium, name=name))
    (x0, x1), (y0, y1) = case.analysis_bounds_xy
    component = "E" + case.polarization
    monitors = [
        td.FieldMonitor(center=((x0 + x1) / 2 * um, (y0 + y1) / 2 * um, z * um),
                        size=((x1 - x0) * um, (y1 - y0) * um, 0.0), freqs=(case.frequency_hz,),
                        fields=(component,), colocate=True, name=f"plane{index}")
        for index, z in enumerate(case.planes_z)]
    source = td.PointDipole(
        center=tuple(v * um for v in case.tx_position), polarization=component,
        source_time=td.GaussianPulse(freq0=case.frequency_hz, fwidth=case.frequency_hz / 5.0),
        name="tx")
    bounds = case.domain_bounds_xyz
    if case.refine:
        axes = [td.UniformGrid(dl=case.grid_m * um) for _ in range(3)]
        for axis, steps in _refined_steps(case).items():
            axes[axis] = td.CustomGrid(dl=tuple(step * um for step in steps))
        grid_spec = td.GridSpec(grid_x=axes[0], grid_y=axes[1], grid_z=axes[2])
    else:
        grid_spec = td.GridSpec.uniform(dl=case.grid_m * um)
    simulation = td.Simulation(
        center=tuple((lo + hi) / 2 * um for lo, hi in bounds),
        size=tuple((hi - lo) * um for lo, hi in bounds),
        grid_spec=grid_spec, structures=tuple(structures),
        sources=(source,), monitors=tuple(monitors),
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML(num_layers=12)),
        run_time=case.run_time_periods / case.frequency_hz, shutoff=1.0e-7)
    _require_lattice_grid(case, simulation)
    return simulation


def _require_lattice_grid(case: ReferenceCase, simulation: Any) -> None:
    """Fail unless every ``grid_m`` lattice node of the domain, plate plane and monitor is a grid node.

    A domain size that is not a whole number of cells makes Tidy3D stretch the cells; a
    zero-thickness plate between nodes then has no tangential field node and disappears.
    """

    for axis, name in enumerate("xyz"):
        nodes = np.asarray(getattr(simulation.grid.boundaries, name)) / METERS_TO_MICROMETERS
        lo, hi = case.domain_bounds_xyz[axis]
        wanted = [lo + case.grid_m * n for n in range(_whole(hi - lo, case.grid_m) + 1)]
        wanted += [block.center[axis] for block in case.blocks if block.size[axis] == 0.0]
        if axis == 2:
            wanted += list(case.planes_z)
        for value in wanted:
            if np.min(np.abs(nodes - value)) > 1.0e-9:
                raise ValueError(f"{case.case_id}: {name} = {value} m is not a grid node")


def _volumes(block: Block) -> list[tuple[tuple[float, ...], tuple[float, ...], dict[str, Any]]]:
    """Homogeneous boxes of a block: the block itself, or one box per layer of a slab block.

    A slab's layers are stacked along its thinnest axis from the low side; the stack must fill it.
    """

    if not block.is_slab:
        return [(block.center, block.size, block.material)]
    axis = int(np.argmin(block.size))
    layers = block.material["layers"]
    if not math.isclose(sum(layer["thickness_m"] for layer in layers), block.size[axis],
                        rel_tol=1.0e-9, abs_tol=1.0e-12):
        raise ValueError("slab layers must fill the block thickness")
    volumes = []
    start = float(block.lo[axis])
    for layer in layers:
        center, size = list(block.center), list(block.size)
        center[axis] = start + 0.5 * layer["thickness_m"]
        size[axis] = layer["thickness_m"]
        volumes.append((tuple(center), tuple(size), {"kind": "dielectric", **layer}))
        start += layer["thickness_m"]
    return volumes


def _refined_steps(case: ReferenceCase) -> dict[int, list[float]]:
    """Per refined axis, the cell sizes: ``grid_m`` everywhere, ``dl`` inside each refined range.

    Range ends and the domain bounds lie on the ``grid_m`` lattice and ``dl`` divides ``grid_m``,
    so every node of the uniform lattice stays a grid node.
    """

    steps: dict[int, list[float]] = {}
    for axis_name in sorted({row["axis"] for row in case.refine}):
        axis = _AXES[axis_name]
        lo, hi = case.domain_bounds_xyz[axis]
        cuts = sorted((row["range"][0], row["range"][1], row["dl"]) for row in case.refine
                      if row["axis"] == axis_name)
        sizes: list[float] = []
        position = lo
        for start, stop, dl in [*cuts, (hi, hi, case.grid_m)]:
            sizes += [case.grid_m] * _whole(start - position, case.grid_m)
            sizes += [dl] * _whole(stop - start, dl)
            position = stop
        steps[axis] = sizes
    return steps


def _whole(length: float, step: float) -> int:
    count = round(length / step)
    if count < 0 or not math.isclose(count * step, length, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError(f"length {length} is not a whole number of {step} m cells")
    return int(count)


def extract_planes(simulation_data: Any, case: ReferenceCase) -> list[dict[str, np.ndarray]]:
    """Return the raw Tidy3D phasors per plane, decimated on the colocated grid (no interpolation)."""

    planes = []
    for index, z in enumerate(case.planes_z):
        values = simulation_data[f"plane{index}"].field_components["E" + case.polarization]
        values = values.isel(f=0)
        for dim in tuple(values.dims):
            if dim not in {"x", "y"}:
                values = values.isel({dim: 0})
        values = values.transpose("y", "x")
        x = np.asarray(values.coords["x"], dtype=np.float64) / METERS_TO_MICROMETERS
        y = np.asarray(values.coords["y"], dtype=np.float64) / METERS_TO_MICROMETERS
        field = np.asarray(values, dtype=np.complex128)
        if not case.refine:
            planes.append({"x": x[::DECIMATION], "y": y[::DECIMATION], "z": float(z),
                           "field": field[::DECIMATION, ::DECIMATION]})
            continue
        # A refined axis has extra nodes; keep the nodes of the uniform decimated lattice.
        rows, cols = _lattice_nodes(y, case.grid_m), _lattice_nodes(x, case.grid_m)
        planes.append({"x": x[cols], "y": y[rows], "z": float(z), "field": field[np.ix_(rows, cols)]})
    return planes


def _lattice_nodes(coords: np.ndarray, grid_m: float) -> np.ndarray:
    pitch = DECIMATION * grid_m
    wanted = coords[0] + pitch * np.arange(int(np.floor((coords[-1] - coords[0]) / pitch + 1.0e-6)) + 1)
    index = np.searchsorted(coords, wanted - 1.0e-9)
    index = np.minimum(index, coords.size - 1)
    if np.max(np.abs(coords[index] - wanted)) > 1.0e-9:
        raise ValueError("monitor nodes do not contain the uniform decimated lattice")
    return index


# --------------------------------------------------------------------------- dipole calibration


def dipole_field(case: ReferenceCase, points: np.ndarray, *, near_field: bool) -> np.ndarray:
    """Polarization-projected field of the unit transmitter dipole, Channel phase convention.

    The far field is ``exp(-jkr)/r`` times the transverse projection of the dipole axis; with
    ``near_field`` the exact ``1/(kr)`` and ``1/(kr)^2`` terms are added.
    """

    k = 2.0 * math.pi / case.wavelength_m
    p = case.polarization_vector
    d = points - np.asarray(case.tx_position)
    r = np.linalg.norm(d, axis=-1)
    rhat = d / r[..., None]
    rp = rhat @ p
    transverse = p - rhat * rp[..., None]
    field = transverse
    if near_field:
        kr = k * r
        factor = (1.0 / kr**2 + 1j / kr)  # exp(-jkr) convention
        field = transverse + factor[..., None] * (3.0 * rhat * rp[..., None] - p)
    return (field @ p) * np.exp(-1j * k * r) / r


def plane_points(x: np.ndarray, y: np.ndarray, z: float) -> np.ndarray:
    xx, yy = np.meshgrid(x, y)
    return np.stack([xx, yy, np.full_like(xx, z)], axis=-1)


# --------------------------------------------------------------------------- geometry


def _segments_blocked(case: ReferenceCase, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """True where the open segment start->end passes through any block (grazing does not count)."""

    start = np.broadcast_to(start, np.broadcast_shapes(start.shape, end.shape))
    end = np.broadcast_to(end, start.shape)
    blocked = np.zeros(start.shape[:-1], dtype=bool)
    direction = end - start
    for block in case.blocks:
        shrink = np.where(np.asarray(block.size) > 0.0, 1.0e-7, -1.0e-9)
        lo, hi = block.lo + shrink, block.hi - shrink
        t0 = np.zeros(blocked.shape)
        t1 = np.ones(blocked.shape)
        inside = np.ones(blocked.shape, dtype=bool)
        for axis in range(3):
            d = direction[..., axis]
            o = start[..., axis]
            parallel = np.abs(d) < 1.0e-15
            inside &= ~(parallel & ((o < lo[axis]) | (o > hi[axis])))
            with np.errstate(divide="ignore", invalid="ignore"):
                ta = (lo[axis] - o) / d
                tb = (hi[axis] - o) / d
            near = np.where(parallel, -np.inf, np.minimum(ta, tb))
            far = np.where(parallel, np.inf, np.maximum(ta, tb))
            t0 = np.maximum(t0, near)
            t1 = np.minimum(t1, far)
        blocked |= inside & (t0 < t1) & (t1 > 1.0e-9) & (t0 < 1.0 - 1.0e-9)
    return blocked


def _faces(block: Block) -> list[tuple[int, float, float]]:
    """(axis, plane coordinate, outward sign) of every reflecting face; a plate reflects on both sides."""

    faces = []
    for axis in range(3):
        if block.size[axis] == 0.0:
            faces += [(axis, block.center[axis], +1.0), (axis, block.center[axis], -1.0)]
        elif not block.is_plate:
            faces += [(axis, float(block.lo[axis]), -1.0), (axis, float(block.hi[axis]), +1.0)]
    return faces


def _reflection_masks(case: ReferenceCase, points: np.ndarray) -> list[np.ndarray]:
    tx = np.asarray(case.tx_position)
    masks = []
    for block in case.blocks:
        for axis, coordinate, sign in _faces(block):
            if sign * (tx[axis] - coordinate) <= 0.0:
                masks.append(np.zeros(points.shape[:-1], dtype=bool))
                continue
            image = tx.copy()
            image[axis] = 2.0 * coordinate - tx[axis]
            front = sign * (points[..., axis] - coordinate) > 0.0
            t = (coordinate - image[axis]) / (points[..., axis] - image[axis] + 1.0e-300)
            hit = image + t[..., None] * (points - image)
            others = [a for a in range(3) if a != axis]
            on_face = np.ones_like(front)
            for a in others:
                on_face &= (hit[..., a] >= block.lo[a]) & (hit[..., a] <= block.hi[a])
            lifted = hit.copy()
            lifted[..., axis] += sign * 1.0e-6
            clear = ~_segments_blocked(case, tx[None, None], lifted) & ~_segments_blocked(
                case, lifted, points)
            masks.append(front & on_face & clear)
    return masks


def _edges(block: Block) -> list[tuple[np.ndarray, np.ndarray]]:
    lo, hi = block.lo, block.hi
    corners = np.unique(np.array([[[lo, hi][i][0], [lo, hi][j][1], [lo, hi][k][2]]
                                  for i in (0, 1) for j in (0, 1) for k in (0, 1)]), axis=0)
    edges = []
    for a in range(len(corners)):
        for b in range(a + 1, len(corners)):
            delta = np.abs(corners[a] - corners[b])
            if np.count_nonzero(delta > 0.0) == 1:
                edges.append((corners[a], corners[b]))
    return edges


def _silhouette(block: Block, a: np.ndarray, b: np.ndarray, tx: np.ndarray) -> bool:
    """True when exactly one face adjacent to the edge a-b faces the transmitter (an ISB edge)."""

    if block.is_plate:
        return True
    lit = []
    for axis in range(3):
        if a[axis] != b[axis]:
            continue
        sign = -1.0 if a[axis] == block.lo[axis] else 1.0
        lit.append(sign * (tx[axis] - a[axis]) > 0.0)
    return lit.count(True) == 1


def _point_in_domain(case: ReferenceCase, point: np.ndarray) -> bool:
    return all(lo <= value <= hi for value, (lo, hi) in zip(point, case.domain_bounds_xyz))


def _segment_in_domain(case: ReferenceCase, a: np.ndarray, b: np.ndarray) -> bool:
    """True when the closed segment a-b meets the simulated domain box."""

    t0, t1 = 0.0, 1.0
    for axis, (lo, hi) in enumerate(case.domain_bounds_xyz):
        d = b[axis] - a[axis]
        if d == 0.0:
            if not lo <= a[axis] <= hi:
                return False
            continue
        ta, tb = (lo - a[axis]) / d, (hi - a[axis]) / d
        t0, t1 = max(t0, min(ta, tb)), min(t1, max(ta, tb))
    return t0 <= t1


def _vertex_visible(
    case: ReferenceCase, block: Block, vertex: np.ndarray, points: np.ndarray,
) -> np.ndarray:
    lifted = vertex + 1.0e-3 * np.sign(vertex - np.asarray(block.center))
    return ~_segments_blocked(case, lifted[None, None], points)


def classify_regions(case: ReferenceCase, x: np.ndarray, y: np.ndarray, z: float) -> np.ndarray:
    """Label every receiver: invalid, lit, shadow, transition (near an ISB/RSB) or corner.

    Transition: within half a projected Fresnel width ``sqrt(lambda s (s+s')/s')`` of a change of
    LoS or specular-reflection visibility, with ``s``/``s'`` measured to the nearest edge. Corner:
    the Keller stationary point of a silhouette edge (one face lit) seen by both antennas lies within
    ``CORNER_FRESNEL_FRACTION * sqrt(lambda L)`` of one of its ends.
    """

    from scipy import ndimage

    points = plane_points(x, y, z)
    tx = np.asarray(case.tx_position)
    wavelength = case.wavelength_m
    pitch = float(min(np.diff(x).min(), np.diff(y).min()))
    valid = np.ones(points.shape[:-1], dtype=bool)
    for block in case.blocks:
        margin = np.where(np.asarray(block.size) > 0.0, pitch, 2.0 * pitch)
        inside = np.all((points >= block.lo - margin) & (points <= block.hi + margin), axis=-1)
        valid &= ~inside
    los = ~_segments_blocked(case, tx[None, None], points)
    masks = [los, *_reflection_masks(case, points)]
    boundary = np.zeros_like(valid)
    for mask in masks:
        change = np.zeros_like(mask)
        change[:, 1:] |= mask[:, 1:] != mask[:, :-1]
        change[:, :-1] |= mask[:, 1:] != mask[:, :-1]
        change[1:, :] |= mask[1:, :] != mask[:-1, :]
        change[:-1, :] |= mask[1:, :] != mask[:-1, :]
        boundary |= change
    # Mask changes caused only by the excluded block interiors are not field boundaries.
    boundary &= ndimage.binary_erosion(valid, iterations=1, border_value=1)
    distance = ndimage.distance_transform_edt(~boundary, sampling=(float(np.diff(y)[0]),
                                                                   float(np.diff(x)[0])))
    # Edges and edge ends outside the simulated domain lie in or beyond the PML: a structure that
    # crosses the domain stands for an unbounded one there, so they are not physical edges.
    edges = [(block, a, b) for block in case.blocks for a, b in _edges(block)
             if _segment_in_domain(case, a, b)]
    nearest_s = np.full(valid.shape, np.inf)
    nearest_sp = np.ones(valid.shape)
    corner = np.zeros_like(valid)
    for block, a, b in edges:
        axis_vec = b - a
        length = float(np.linalg.norm(axis_vec))
        if length == 0.0:
            continue
        axis_vec = axis_vec / length
        u = np.clip((points - a) @ axis_vec, 0.0, length)
        foot = a + u[..., None] * axis_vec
        s = np.linalg.norm(points - foot, axis=-1)
        sp = np.linalg.norm(foot - tx, axis=-1)
        closer = s < nearest_s
        nearest_s = np.where(closer, s, nearest_s)
        nearest_sp = np.where(closer, sp, nearest_sp)
        a_t = float((tx - a) @ axis_vec)
        rho_t = float(np.linalg.norm((tx - a) - a_t * axis_vec))
        a_s = (points - a) @ axis_vec
        rho_s = np.linalg.norm((points - a) - a_s[..., None] * axis_vec, axis=-1)
        u_star = (rho_t * a_s + rho_s * a_t) / (rho_s + rho_t + 1.0e-12)
        if not _silhouette(block, a, b, tx):
            continue
        middle = 0.5 * (a + b)
        lifted_middle = middle + 1.0e-3 * np.sign(middle - np.asarray(block.center))
        if _segments_blocked(case, tx[None, None], lifted_middle[None, None]).item():
            continue
        seen = ~_segments_blocked(case, lifted_middle[None, None], points)
        for end_u, vertex in ((0.0, a), (length, b)):
            if not _point_in_domain(case, vertex):
                continue
            s_end = np.linalg.norm(points - vertex, axis=-1)
            sp_end = float(np.linalg.norm(vertex - tx))
            fresnel = np.sqrt(wavelength * s_end * sp_end / (s_end + sp_end))
            corner |= seen & (np.abs(u_star - end_u) < CORNER_FRESNEL_FRACTION * fresnel)
    width = 0.5 * np.sqrt(wavelength * nearest_s * (nearest_s + nearest_sp) / nearest_sp)
    transition = (distance < width) if boundary.any() else np.zeros_like(valid)
    labels = np.where(los, LIT, SHADOW).astype(np.uint8)
    labels[transition] = TRANSITION
    labels[corner] = CORNER
    labels[~valid] = INVALID
    return labels


# --------------------------------------------------------------------------- storage


def reference_path(case_id: str) -> Path:
    return DATA_DIR / f"{case_id}.npz"


def save_reference(
    case: ReferenceCase, planes: list[dict[str, Any]], metadata: dict[str, Any],
    path: Path | None = None,
) -> Path:
    output = path or reference_path(case.case_id)
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "schema_name": np.asarray(SCHEMA_NAME),
        "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int64),
        "metadata_json": np.asarray(json.dumps(
            {**metadata, "case_id": case.case_id, "case_fingerprint": case.fingerprint,
             "region_names": REGION_NAMES}, sort_keys=True, separators=(",", ":"))),
    }
    for index, plane in enumerate(planes):
        arrays[f"plane{index}_x"] = plane["x"].astype(np.float64)
        arrays[f"plane{index}_y"] = plane["y"].astype(np.float64)
        arrays[f"plane{index}_z"] = np.asarray(plane["z"], dtype=np.float64)
        arrays[f"plane{index}_field"] = plane["field"].astype(np.complex64)
        arrays[f"plane{index}_regions"] = plane["regions"].astype(np.uint8)
    with output.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    return output


def load_reference(case: ReferenceCase, path: Path | None = None) -> tuple[list[dict[str, Any]],
                                                                          dict[str, Any]]:
    with np.load(path or reference_path(case.case_id), allow_pickle=False) as data:
        if str(data["schema_name"].item()) != SCHEMA_NAME:
            raise ValueError("not a full-wave reference map")
        if int(data["schema_version"].item()) != SCHEMA_VERSION:
            raise ValueError("unsupported full-wave reference map version")
        metadata = json.loads(str(data["metadata_json"].item()))
        if metadata["case_fingerprint"] != case.fingerprint:
            raise ValueError(f"{case.case_id}: stored reference fingerprint does not match the manifest")
        planes = []
        for index in range(len(case.planes_z)):
            planes.append({key: data[f"plane{index}_{key}"]
                           for key in ("x", "y", "z", "field", "regions")})
            planes[-1]["z"] = float(planes[-1]["z"])
    return planes, metadata


# --------------------------------------------------------------------------- Channel


_BOX_FACES = ((0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
              (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7))


def _block_mesh(
    block: Block, tx: np.ndarray,
) -> tuple[list[list[float]], list[tuple[int, int, int]], str]:
    """Triangles of a block: a box, a plate, or for a slab a plate in its face towards ``tx``."""

    lo, hi = block.lo, block.hi
    if block.is_plate or block.is_slab:
        axis = int(np.argmin(block.size))
        plane = float(block.center[axis])
        if block.is_slab:
            plane = float(hi[axis] if tx[axis] > block.center[axis] else lo[axis])
        u, v = [a for a in range(3) if a != axis]
        vertices = []
        for cu, cv in ((lo[u], lo[v]), (hi[u], lo[v]), (hi[u], hi[v]), (lo[u], hi[v])):
            vertex = [plane] * 3
            vertex[u], vertex[v] = float(cu), float(cv)
            vertices.append(vertex)
        return vertices, [(0, 1, 2), (0, 2, 3)], "surface"
    vertices = [[float(x), float(y), float(z)] for z in (lo[2], hi[2])
                for x, y in ((lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1]))]
    return vertices, list(_BOX_FACES), "solid"


def build_channel_scene(case: ReferenceCase, plane: dict[str, Any]) -> Any:
    """Channel scene of the case with a z-plane receiver grid on the stored reference samples."""

    import torch
    from witwin.core import (
        AntennaState, MaterialLayer, Mesh, PhysicalMaterial, ReceiverGrid, Scene, Structure)
    from witwin.core.identity import new_antenna_id

    structures = []
    for index, block in enumerate(case.blocks):
        if block.material["kind"] == "pec":
            material = PhysicalMaterial.perfect_conductor(name="pec")
        elif block.is_slab:
            material = PhysicalMaterial(layers=tuple(MaterialLayer(
                thickness_m=layer["thickness_m"], eps_r=layer["eps_r"], sigma_e=layer["sigma_e"])
                for layer in block.material["layers"]), name="slab")
        else:
            material = PhysicalMaterial(layers=(MaterialLayer(
                thickness_m=min(block.size) / 2.0, eps_r=block.material["eps_r"],
                sigma_e=block.material["sigma_e"]),), name="dielectric")
        vertices, faces, fill_mode = _block_mesh(block, np.asarray(case.tx_position))
        mesh = Mesh(torch.tensor(vertices, dtype=torch.float32), torch.tensor(faces, dtype=torch.int32),
                    recenter=False, fill_mode=fill_mode, topology_diagnostics=False)
        structures.append(Structure(mesh, material, name=f"block-{index}", surface_id=index + 1))
    polarization = torch.tensor(case.polarization_vector, dtype=torch.float32)
    x, y = plane["x"], plane["y"]
    receivers = ReceiverGrid(
        new_antenna_id(), origin=torch.tensor([x[0], y[0], plane["z"]], dtype=torch.float32),
        x_axis=torch.tensor([1.0, 0.0, 0.0]), y_axis=torch.tensor([0.0, 1.0, 0.0]),
        shape=(x.size, y.size), spacing=(float(x[1] - x[0]), float(y[1] - y[0])),
        polarization=polarization)
    tx = AntennaState(new_antenna_id(), "tx", torch.tensor(case.tx_position, dtype=torch.float32),
                      polarization=polarization, power_w=1.0)
    return Scene(structures=structures, endpoints=[tx, receivers])


# --------------------------------------------------------------------------- metrics


FLOOR_DB = -40.0


def region_errors(field: np.ndarray, reference: np.ndarray, regions: np.ndarray) -> dict[str, dict]:
    """Per-region magnitude and complex errors of a calibrated field against the reference.

    Magnitudes are compared in dB after clamping both to ``FLOOR_DB`` below the plane's 99th
    percentile reference level, so deep interference nulls do not dominate. ``nmse`` is the
    complex normalized mean-square error with no fitted scale or phase.
    """

    floor = np.percentile(np.abs(reference[regions != INVALID]), 99.0) * 10.0 ** (FLOOR_DB / 20.0)
    db = (20.0 * np.log10(np.maximum(np.abs(field), floor))
          - 20.0 * np.log10(np.maximum(np.abs(reference), floor)))
    result = {}
    for label, name in enumerate(REGION_NAMES):
        if label == INVALID:
            continue
        mask = regions == label
        if not mask.any():
            continue
        ref = reference[mask]
        energy = float(np.sum(np.abs(ref) ** 2))
        result[name] = {
            "count": int(mask.sum()),
            "rmse_db": float(np.sqrt(np.mean(db[mask] ** 2))),
            "bias_db": float(np.mean(db[mask])),
            "envelope_nmse": float(np.sum((np.abs(field[mask]) - np.abs(ref)) ** 2) / energy),
            "nmse": float(np.sum(np.abs(field[mask] - ref) ** 2) / energy),
        }
    return result

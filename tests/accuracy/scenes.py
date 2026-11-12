# Copyright Xingyu Chen.
# Small canonical scenes and per-path result extraction for the reference/accuracy suite.

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch
from witwin.core import (
    AntennaState,
    MaterialLayer,
    Mesh,
    PhysicalMaterial,
    ReceiverGrid,
    Scene,
    Structure,
)
from witwin.core.identity import new_antenna_id

FREQ = 3.5e9
FLOAT32_EPS = 2.0**-24


def coefficient_tolerance(path_length_m: float, frequency_hz: float = FREQ) -> float:
    """Relative complex tolerance for one float32 path coefficient.

    ``1e-4`` covers the float32 amplitude chain (spreading, Fresnel/UTD arithmetic: tens of
    rounding steps of 6e-8 each, with a 10x margin). The phase ``k L`` is formed in float32, so
    its absolute error grows as a few ulps of ``k L``; ``8 * 2^-24 * k * L`` bounds it.
    """

    k = 2.0 * math.pi * frequency_hz / 299792458.0
    return 1e-4 + 8.0 * FLOAT32_EPS * k * path_length_m


def antenna(role: str, position: Any, polarization: Any = (0.0, 0.0, 1.0)) -> AntennaState:
    kw = {"power_w": 1.0} if role == "tx" else {}
    return AntennaState(new_antenna_id(), role, torch.tensor(position, dtype=torch.float32),
                        polarization=torch.tensor(polarization, dtype=torch.float32), **kw)


def grid(origin: Any, shape: tuple[int, int], spacing: float) -> ReceiverGrid:
    return ReceiverGrid(new_antenna_id(), torch.tensor(origin, dtype=torch.float32),
                        torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 0.0]), shape,
                        (spacing, spacing))


def surface(vertices: Any, faces: Any, material: Any) -> Structure:
    mesh = Mesh(torch.tensor(vertices, dtype=torch.float32), torch.tensor(faces, dtype=torch.int32),
                recenter=False, fill_mode="surface", topology_diagnostics=False)
    return Structure(mesh, material)


def half_space(eps_r: float, sigma_e: float, thickness_m: float = 30.0) -> PhysicalMaterial:
    """A lossy layer thick enough that its back face is invisible: a half-space in practice."""
    return PhysicalMaterial(layers=(MaterialLayer(thickness_m=thickness_m, eps_r=eps_r, sigma_e=sigma_e),))


def ground(material: Any, half: float = 2000.0) -> Structure:
    v = [[-half, -half, 0.0], [half, -half, 0.0], [-half, half, 0.0], [half, half, 0.0]]
    return surface(v, [[0, 1, 2], [1, 3, 2]], material)


def wall_x(x: float, material: Any, half: float = 200.0) -> Structure:
    """Wall in the plane ``x = const`` whose face normal points to -x."""
    v = [[x, -half, -half], [x, half, -half], [x, -half, half], [x, half, half]]
    return surface(v, [[0, 2, 1], [1, 2, 3]], material)


@dataclass(frozen=True)
class PathRow:
    depth: int
    coefficient: complex
    length_m: float
    point: np.ndarray


def deterministic_rows(
    scene: Scene, components: set[str], max_depth: int = 1, frequency_hz: float = FREQ,
) -> list[PathRow]:
    from witwin.channel.radiomap import Coherent as Config, solve

    result = solve(scene, Config(components=components, max_depth=max_depth, export_paths=True),
                   reference_frequency_hz=frequency_hz)
    paths = result.paths
    count = int(paths.valid.numel())
    first = paths.interaction_positions[:, 0] if paths.interaction_positions.shape[1] else (
        torch.zeros((count, 3), device=paths.valid.device))
    points = first.reshape(count, -1).double().cpu().numpy() if count else []
    return [PathRow(int(d), complex(c), float(length), np.asarray(points[i][:3]))
            for i, (d, c, length) in enumerate(zip(paths.depth.tolist(), paths.coefficient.tolist(),
                                                    paths.path_length_m.tolist()))]


def path_solver_coefficient(
    scene: Scene, components: set[str], max_depth: int = 1, frequency_hz: float = FREQ,
) -> dict[int, complex]:
    """Path solver coefficients of the single Tx/Rx pair, summed per interaction depth."""
    from witwin.channel.path import Config, solve

    result = solve(scene, Config(components=components, max_depth=max_depth),
                   reference_frequency_hz=frequency_hz)
    valid = result.valid.reshape(-1).cpu()
    a = result.a.reshape(valid.numel(), -1)[:, 0].cpu()
    depth = (result.types.reshape(valid.numel(), -1) != 0).sum(dim=1).cpu() if hasattr(
        result, "types") and result.types is not None else torch.zeros_like(valid, dtype=torch.long)
    out: dict[int, complex] = {}
    for i in torch.nonzero(valid).flatten().tolist():
        out[int(depth[i])] = out.get(int(depth[i]), 0j) + complex(a[i])
    return out

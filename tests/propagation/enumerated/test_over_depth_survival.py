# Copyright Xingyu Chen.
# Tests that over-depth transmission and uncertified coupled roots leave the CUDA context usable.

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")

_PRELUDE = textwrap.dedent(
    """
    import torch
    from witwin.core import AntennaState, Mesh, PhysicalMaterial, ReceiverGrid, Scene, Structure
    from witwin.core.identity import new_antenna_id

    FREQ = 3.5e9
    CONCRETE = PhysicalMaterial(eps_r=5.24, sigma_e=0.123)
    BOX_FACES = [[0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6],
                 [3, 0, 4], [3, 4, 7], [4, 5, 6], [4, 6, 7]]

    def city(n_side, pitch=40.0, width=22.0):
        vertices = [[0, 0, 0], [n_side * pitch, 0, 0], [n_side * pitch, n_side * pitch, 0],
                    [0, n_side * pitch, 0]]
        faces = [[0, 1, 2], [0, 2, 3]]
        for i in range(n_side):
            for j in range(n_side):
                x0 = i * pitch + (pitch - width) / 2
                y0 = j * pitch + (pitch - width) / 2
                h = 10.0 + 5.0 * ((3 * i + 7 * j) % 6)
                base = len(vertices)
                for z in (0.0, h):
                    vertices += [[x0, y0, z], [x0 + width, y0, z], [x0 + width, y0 + width, z],
                                 [x0, y0 + width, z]]
                faces += [[base + a, base + b, base + c] for a, b, c in BOX_FACES]
        mesh = Mesh(torch.tensor(vertices, dtype=torch.float32), torch.tensor(faces, dtype=torch.int32),
                    recenter=False, fill_mode="surface", topology_diagnostics=False)
        return [Structure(mesh, CONCRETE, name="city", surface_id=1)], n_side * pitch

    def transmitter(extent):
        centre = round(extent / 2 / 40.0) * 40.0
        return AntennaState(new_antenna_id(), "tx", torch.tensor([centre, centre, 25.0]), power_w=1.0)

    def grid(extent, n):
        spacing = extent / n
        return ReceiverGrid(new_antenna_id(), torch.tensor([spacing / 2, spacing / 2, 1.5]),
                            torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 0.0]), (n, n),
                            (spacing, spacing))

    def points(extent, n, seed=1):
        generator = torch.Generator().manual_seed(seed)
        out = []
        for _ in range(n):
            position = torch.rand(3, generator=generator) * extent
            position[2] = 1.5
            out.append(AntennaState(new_antenna_id(), "rx", position))
        return out

    def usable():
        assert float(torch.ones(4, device="cuda").sum()) == 4.0
    """
)


def _run(body: str) -> str:
    completed = subprocess.run(
        [sys.executable, "-c", _PRELUDE + textwrap.dedent(body)],
        cwd=REPOSITORY_ROOT,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    return completed.stdout


def test_over_depth_transmission_is_dropped_in_every_solver_and_the_consumer() -> None:
    stdout = _run(
        """
        from witwin.channel import radiomap
        import witwin.channel.path as path
        import witwin.channel.experimental.bdpt as bdpt
        from witwin.channel.propagation.consumer import EndpointBatch, PropagationRequest, evaluate
        from witwin.channel.scene import compile as compile_scene

        structures, extent = city(4)
        for depth in (1, 3):
            for name, config_type, solve, receivers, extra in (
                ("det", radiomap.Coherent, radiomap.solve, [grid(extent, 8)], {}),
                ("path", path.Config, path.solve, points(extent, 8), {}),
                ("mc", radiomap.Stochastic, radiomap.solve, [grid(extent, 8)], {"samples": 256}),
                ("bdpt", bdpt.Config, bdpt.solve, [grid(extent, 8)], {"samples": 256}),
            ):
                scene = Scene(structures=structures, endpoints=[transmitter(extent), *receivers])
                config = config_type(max_depth=depth, components={"los", "transmission"}, **extra)
                result = solve(scene, config, reference_frequency_hz=FREQ)
                torch.cuda.synchronize()
                dropped = result.metadata["transmission"]["dropped_rows_over_max_depth"]
                assert isinstance(dropped, int) and dropped >= 0
                print("DROPPED", name, depth, dropped, flush=True)
                usable()
            compiled = compile_scene(Scene(structures=structures), reference_frequency_hz=FREQ)
            sinks = points(extent, 8)
            z = torch.tensor([[0.0, 0.0, 1.0]], device="cuda")
            request = PropagationRequest(
                sources=EndpointBatch(
                    stable_ids=torch.arange(1, device="cuda"),
                    positions_m=transmitter(extent).position.float().reshape(1, 3).cuda(),
                    polarizations=z, polarization_basis=None, powers_w=torch.ones(1, device="cuda")),
                sinks=EndpointBatch(
                    stable_ids=torch.arange(len(sinks), device="cuda"),
                    positions_m=torch.stack([s.position.float() for s in sinks]).cuda(),
                    polarizations=z.expand(len(sinks), 3).contiguous(), polarization_basis=None),
                reference_frequency_hz=FREQ, components=frozenset({"los", "transmission"}),
                max_depth=depth, response="complex3_transport", topology_mode="discover",
                ad_mode="none")
            evaluation = evaluate(compiled, request)
            torch.cuda.synchronize()
            print("DROPPED consumer", depth, evaluation.diagnostics.transmission_dropped_rows,
                  flush=True)
            usable()
        print("DONE", flush=True)
        """
    )
    assert "DONE" in stdout
    counts = {}
    for line in stdout.splitlines():
        if line.startswith("DROPPED"):
            _, name, depth, dropped = line.split()
            counts[(name, int(depth))] = int(dropped)
    # A 4x4 city behind the street crossing hides receivers behind two or more
    # walls, so max_depth=1 must drop rows that max_depth=3 keeps.
    for name in ("det", "path", "mc", "bdpt", "consumer"):
        assert counts[(name, 1)] > 0, (name, counts)
        assert counts[(name, 3)] < counts[(name, 1)], (name, counts)


def test_coupled_double_diffraction_city_grid_completes() -> None:
    stdout = _run(
        """
        from witwin.channel import radiomap
        import witwin.channel.path as path

        structures, extent = city(1)
        scene = Scene(structures=structures, endpoints=[transmitter(extent), *points(extent, 4)])
        config = path.Config(max_depth=2, components={"los", "reflection", "diffraction"},
                             coupled_paths=True)
        path.solve(scene, config, reference_frequency_hz=FREQ)
        torch.cuda.synchronize()
        usable()
        structures, extent = city(4)
        scene = Scene(structures=structures, endpoints=[transmitter(extent), grid(extent, 16)])
        config = radiomap.Coherent(max_depth=2, components={"los", "reflection", "diffraction"},
                            coupled_paths=True)
        result = radiomap.solve(scene, config, reference_frequency_hz=FREQ)
        torch.cuda.synchronize()
        assert torch.isfinite(result.path_gain).all()
        usable()
        print("DONE", flush=True)
        """
    )
    assert "DONE" in stdout

# Copyright Xingyu Chen.
# Runs test snippets that might poison the CUDA context in a fresh interpreter.

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

_PRELUDE = '''
import json, sys, traceback
import torch
from witwin.core import AntennaState, Mesh, PhysicalMaterial, ReceiverGrid, Scene, Structure
from witwin.core.identity import new_antenna_id

OUTCOMES = {}

def mesh(vertices, faces, **kwargs):
    geometry = Mesh(vertices, faces, recenter=False, fill_mode="surface", topology_diagnostics=False)
    return Structure(geometry, PhysicalMaterial(eps_r=4.0, sigma_e=0.01), **kwargs)

def floor():
    v = torch.tensor([[-10.0, -10.0, 0.0], [10.0, -10.0, 0.0], [-10.0, 10.0, 0.0], [10.0, 10.0, 0.0]])
    return mesh(v, torch.tensor([[0, 1, 2], [1, 3, 2]]))

def wall(vertices=None, faces=None, **kwargs):
    v = torch.tensor([[2.5, -2.0, -1.0], [2.5, 2.0, -1.0], [2.5, -2.0, 3.0], [2.5, 2.0, 3.0]]) if vertices is None else vertices
    f = torch.tensor([[0, 1, 2], [1, 3, 2]]) if faces is None else faces
    return mesh(v, f, **kwargs)

def tx(position=(0.0, 0.0, 1.5), **kwargs):
    return AntennaState(new_antenna_id(), "tx", torch.tensor(position), **kwargs)

def rx(position=(1.0, 1.0, 1.5), **kwargs):
    return AntennaState(new_antenna_id(), "rx", torch.tensor(position), **kwargs)

def grid(origin=(1.0, -2.0, 1.0), shape=(4, 4), spacing=(0.5, 0.5)):
    return ReceiverGrid(new_antenna_id(), torch.tensor(origin), torch.tensor([1.0, 0.0, 0.0]),
                        torch.tensor([0.0, 1.0, 0.0]), shape, spacing)

def scene(structures=None, endpoints=None):
    return Scene(structures=[floor(), wall()] if structures is None else structures,
                 endpoints=[tx(), rx()] if endpoints is None else endpoints)

def solve(name, scene_value, **config):
    from tests.support.solvers import solver
    entry = solver(name)
    config.setdefault("components", {"los", "reflection"})
    if name in ("basic", "bdpt"):
        config.setdefault("samples", 256)
    return entry.solve(scene_value, entry.Config(**config), reference_frequency_hz=3.5e9)

def case(name, thunk):
    try:
        value = thunk()
        torch.cuda.synchronize()
        OUTCOMES[name] = {"status": "returned", "value": value if isinstance(value, (int, float, str, list, dict, type(None))) else repr(value)}
    except Exception as error:
        OUTCOMES[name] = {"status": "raised", "type": type(error).__name__, "message": str(error)}
'''

_EPILOGUE = '''
try:
    torch.cuda.synchronize()
    healthy = float(solve("det", scene()).path_gain.sum()) > 0.0
    torch.cuda.synchronize()
    OUTCOMES["__health__"] = {"status": "returned", "value": bool(healthy)}
except Exception as error:
    OUTCOMES["__health__"] = {"status": "raised", "type": type(error).__name__, "message": str(error)}
print("OUTCOMES " + json.dumps(OUTCOMES, default=str), flush=True)
'''


def run_isolated(body: str, *, timeout: float = 600.0) -> dict[str, dict[str, object]]:
    """Run ``body`` after a scene-building prelude in a fresh interpreter and return its outcomes.

    ``body`` records results with ``case(name, thunk)``. The epilogue solves a
    small valid scene, so ``outcomes["__health__"]["value"]`` is True only when
    the CUDA context survived every case.
    """

    code = _PRELUDE + textwrap.dedent(body) + _EPILOGUE
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(REPOSITORY_ROOT), environment.get("PYTHONPATH")) if value
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    lines = [line for line in completed.stdout.splitlines() if line.startswith("OUTCOMES ")]
    assert completed.returncode == 0 and lines, (
        f"isolated run failed (exit {completed.returncode}):\n{completed.stderr[-4000:]}"
    )
    return json.loads(lines[-1][len("OUTCOMES "):])

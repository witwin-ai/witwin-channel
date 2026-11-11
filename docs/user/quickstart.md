# Quickstart

This page takes you from an installed Channel to a radiomap, a channel impulse
response and a gradient in about five minutes. Install Channel first as the
README's **Installation** section describes, then check the machine with:

```bash
python -m witwin.channel doctor
```

`doctor` reports the driver, GPU, PyTorch, build, OptiX runtime and the trace
backend Channel will use, with a verdict and a fix for each problem. If it
reports a problem, see [Troubleshooting](troubleshooting.md).

Channel needs an NVIDIA GPU. Every solve runs on CUDA; there is no CPU mode.

## The pieces

| Concept | Where it lives |
| --- | --- |
| Scene, structures, meshes, materials, antennas, receiver grids | `witwin.core` (`Scene`, `Structure`, `Mesh`, `PhysicalMaterial`, `AntennaState`, `ReceiverGrid`) |
| Loading OBJ, PLY, glTF and Mitsuba/Sionna XML files | `witwin.channel.importers` ([Scene import](scene-import.md)) |
| ITU-R P.2040 materials, TR 38.901 and dipole patterns | `witwin.channel.presets` ([Materials and antennas](materials-and-antennas.md)) |
| Coherent radiomaps and link gains | `witwin.channel.deterministic` |
| Explicit paths, CIR, CFR and taps | `witwin.channel.path` |
| Monte Carlo radiomaps | `witwin.channel.montecarlo.basic`, `witwin.channel.montecarlo.bdpt` |
| Frozen-topology replay, time-varying channels | `witwin.channel.propagation.consumer` |

Units are SI: metres, hertz, seconds, watts, radians. The world is +Z up.

## 1. Import a scene and solve a radiomap

```python
import torch

from witwin.core import AntennaState, ReceiverGrid
from witwin.core.identity import new_antenna_id
from witwin.channel.deterministic import Config, solve
from witwin.channel.importers import load_mesh
from witwin.channel.presets import itu_material

f = 3.5e9
imported = load_mesh(
    "city.obj",  # usemtl names map to materials; None is the default entry
    {"concrete": itu_material("concrete", f), None: itu_material("concrete", f)},
)
print(imported.summary())  # structures, vertices, faces, welded duplicates

tx = AntennaState(new_antenna_id(), "tx", torch.tensor([0.0, 0.0, 25.0]), power_w=1.0)
grid = ReceiverGrid(
    new_antenna_id(), (-100.0, -100.0, 1.5), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
    (128, 128), (200 / 128, 200 / 128),
)
scene = imported.to_scene(endpoints=[tx, grid])
result = solve(
    scene, Config(max_depth=2, components={"los", "reflection", "diffraction"}),
    reference_frequency_hz=f,
)
path_gain = result.path_gain[0]  # (rows, cols): first axis follows the grid's x_axis
```

The first solve in a fresh process compiles OptiX programs and takes a few
seconds; later solves and later processes are fast.

## 2. Paths, CIR and CFR for one link

```python
from witwin.channel.path import Config as PathConfig, solve as solve_paths

rx = AntennaState(new_antenna_id(), "rx", torch.tensor([60.0, 5.0, 1.5]))
paths = solve_paths(imported.to_scene(endpoints=[tx, rx]),
                    PathConfig(max_depth=2), reference_frequency_hz=f)
a, tau = paths.cir()                                   # (R, Ra, T, Ta, P, time), (..., P)
h = paths.cfr(torch.linspace(-50e6, 50e6, 64, device="cuda"))  # (R, Ra, T, Ta, time, F)
```

## 3. Differentiate

Make a material parameter or an endpoint position a CUDA tensor with
`requires_grad=True` and pass `ad_mode="vjp"`:

```python
from witwin.core import PhysicalMaterial

eps_r = torch.tensor(4.0, device="cuda", requires_grad=True)
walls = PhysicalMaterial(eps_r=eps_r, sigma_e=0.05)
imported = load_mesh("city.obj", walls)
scene = imported.to_scene(endpoints=[tx, rx])
result = solve(scene, Config(max_depth=2, components={"los", "reflection"}, ad_mode="vjp"),
               reference_frequency_hz=f)
result.path_gain.sum().backward()
print(eps_r.grad)
```

Gradients follow the topology found by the forward solve; a path that appears
or disappears as a parameter moves is outside the derivative.

## Runnable examples

The [`examples/`](../../examples) directory has complete scripts that each run
in well under a minute on one GPU:

| Script | What it shows |
| --- | --- |
| `radiomap_from_obj.py` | OBJ or Mitsuba XML import, deterministic radiomap, PNG (matplotlib optional) |
| `path_cir_cfr.py` | Path solve with a TR 38.901 element, path table, CIR, taps, CFR |
| `material_calibration.py` | Recover a wall permittivity from simulated measurements with `ad_mode="vjp"` |
| `moving_tx_time_varying.py` | Discover once, then replay a moving transmitter with `consumer.evaluate_time_varying` |

Run them from any directory, for example `python examples/path_cir_cfr.py`.
In a source checkout, put the checkout on `PYTHONPATH` so the examples import
the checkout rather than an installed wheel.

The complete API is in the [API reference](../api-reference/index.md).

# Copyright Xingyu Chen.
# Imports an OBJ or Mitsuba XML scene, solves a deterministic radiomap and saves a PNG.

"""Import an OBJ or Mitsuba XML scene, solve a deterministic radiomap and save it as a PNG."""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import time

import torch

from witwin.core import AntennaState, ReceiverGrid
from witwin.core.identity import new_antenna_id
from witwin.channel.deterministic import Config, solve
from witwin.channel.importers import load_mesh, load_mitsuba
from witwin.channel.presets import itu_material

import street_canyon


def _import(args: argparse.Namespace, workdir: Path):
    if args.demo_xml:
        return load_mitsuba(street_canyon.write_mitsuba(workdir), frequency_hz=args.frequency)
    if args.xml is not None:
        return load_mitsuba(args.xml, frequency_hz=args.frequency)
    obj = args.obj if args.obj is not None else street_canyon.write_obj(workdir / "canyon.obj")
    materials = {
        "concrete": itu_material("concrete", args.frequency),
        "ground": itu_material("medium_dry_ground", args.frequency),
        None: itu_material("concrete", args.frequency),
    }
    return load_mesh(obj, materials)


def _save_png(path_gain: torch.Tensor, extent: tuple[float, ...], output: Path) -> Path:
    gain_db = 10.0 * torch.log10(path_gain.clamp_min(1e-30)).cpu()
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        target = output.with_suffix(".pt")
        torch.save(gain_db, target)
        print(f"matplotlib is not installed; saved path gain (dB) tensor to {target}")
        return target
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(
        gain_db, origin="lower", extent=extent, vmin=-140, vmax=-40, cmap="viridis"
    )
    figure.colorbar(image, ax=axis, label="path gain [dB]")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    figure.tight_layout()
    figure.savefig(output, dpi=120)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obj", type=Path, help="OBJ file (default: generated street canyon)")
    parser.add_argument("--xml", type=Path, help="Mitsuba/Sionna RT XML scene instead of an OBJ")
    parser.add_argument(
        "--demo-xml", action="store_true", help="import the generated canyon through Mitsuba XML"
    )
    parser.add_argument("--frequency", type=float, default=3.5e9)
    parser.add_argument("--tx", type=float, nargs=3, default=(-37.3, 1.1, 10.0))
    parser.add_argument("--size", type=float, default=120.0, help="square map side, metres")
    parser.add_argument("--cells", type=int, default=128)
    parser.add_argument("--height", type=float, default=1.5)
    parser.add_argument("--output", type=Path, default=Path("radiomap.png"))
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as workdir:
        imported = _import(args, Path(workdir))
    print("imported:", imported.summary())

    spacing = args.size / args.cells
    origin = -0.5 * args.size + 0.5 * spacing
    grid = ReceiverGrid(
        new_antenna_id(), (origin, origin, args.height), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
        (args.cells, args.cells), (spacing, spacing),
    )
    tx = AntennaState(new_antenna_id(), "tx", torch.tensor(args.tx), power_w=1.0)
    scene = imported.to_scene(endpoints=[tx, grid])
    config = Config(max_depth=2, components={"los", "reflection", "diffraction"})

    for label in ("first solve (compiles, warms caches)", "second solve"):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = solve(scene, config, reference_frequency_hz=args.frequency)
        torch.cuda.synchronize()
        print(f"{label}: {1e3 * (time.perf_counter() - start):.1f} ms")

    # Grid maps come back as (T, n_y, n_x): the first map axis follows the grid's
    # y_axis and the second its x_axis, which is the image layout imshow expects.
    path_gain = result.path_gain[0]
    covered = int((path_gain > 0).sum())
    print(f"radiomap {tuple(path_gain.shape)}, {covered} of {path_gain.numel()} cells reached")
    for name, power in result.component_power.items():
        print(f"  {name:<11} total power {float(power.sum()):.3e}")
    half = 0.5 * args.size
    written = _save_png(path_gain, (-half, half, -half, half), args.output)
    print(f"wrote {written}")


if __name__ == "__main__":
    main()

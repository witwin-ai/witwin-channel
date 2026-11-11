# Copyright Xingyu Chen.
# Solves point-to-point paths in the street canyon and prints the CIR, taps and CFR.

"""Solve point-to-point paths in the street canyon and print the CIR, taps and CFR."""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import time

import torch

from witwin.core import AntennaState
from witwin.core.identity import new_antenna_id
from witwin.channel.importers import load_mesh
from witwin.channel.path import Config, InteractionType, solve
from witwin.channel.presets import itu_material, tr38901_pattern

import street_canyon

KIND_NAMES = {
    int(InteractionType.REFLECTION): "R",
    int(InteractionType.DIFFRACTION): "D",
    int(InteractionType.TRANSMISSION): "T",
    int(InteractionType.SCATTERING): "S",
}


def _sequence(types: torch.Tensor) -> str:
    letters = [KIND_NAMES.get(int(value), "?") for value in types.tolist() if int(value) != 0]
    return "-".join(letters) if letters else "LoS"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frequency", type=float, default=3.5e9)
    parser.add_argument("--bandwidth", type=float, default=100e6)
    parser.add_argument("--subcarriers", type=int, default=64)
    parser.add_argument("--tx", type=float, nargs=3, default=(-37.3, 1.1, 10.0))
    parser.add_argument("--rx", type=float, nargs=3, default=(35.0, 4.0, 1.5))
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as workdir:
        obj = street_canyon.write_obj(Path(workdir) / "canyon.obj")
        imported = load_mesh(
            obj,
            {
                "concrete": itu_material("concrete", args.frequency),
                "ground": itu_material("medium_dry_ground", args.frequency),
            },
        )
    # A TR 38.901 sector element on the transmitter, boresight along +x (down the street).
    tx = AntennaState(
        new_antenna_id(), "tx", torch.tensor(args.tx), power_w=1.0, pattern=tr38901_pattern(),
        polarization=(0.0, 0.0, 1.0),
    )
    rx = AntennaState(new_antenna_id(), "rx", torch.tensor(args.rx), polarization=(0.0, 0.0, 1.0))
    scene = imported.to_scene(endpoints=[tx, rx])
    config = Config(max_depth=2, components={"los", "reflection", "diffraction"})

    torch.cuda.synchronize()
    start = time.perf_counter()
    paths = solve(scene, config, reference_frequency_hz=args.frequency)
    torch.cuda.synchronize()
    print(f"path solve: {1e3 * (time.perf_counter() - start):.1f} ms, "
          f"{int(paths.num_paths.sum())} paths")

    a, tau = paths.cir()  # a: (R, Ra, T, Ta, P, time); tau: (R, Ra, T, Ta, P), first path at 0
    valid = paths.valid[0, 0, 0, 0]
    order = torch.argsort(tau[0, 0, 0, 0].masked_fill(~valid, float("inf")))
    print(" delay [ns]   |a|^2 [dB]   interactions")
    for index in order[: int(valid.sum())].tolist():
        power_db = 20.0 * torch.log10(a[0, 0, 0, 0, index, 0].abs().clamp_min(1e-30))
        kinds = _sequence(paths.interaction_type[0, 0, 0, 0, index])
        print(f"{1e9 * float(tau[0, 0, 0, 0, index]):10.2f}   {float(power_db):10.2f}   {kinds}")

    taps = paths.taps(args.bandwidth, 16)  # (R, Ra, T, Ta, time, taps)
    tap_db = 20.0 * torch.log10(taps[0, 0, 0, 0, 0].abs().clamp_min(1e-10))  # empty taps: -200
    print("taps [dB] at", f"{args.bandwidth / 1e6:.0f} MHz:", [round(float(v), 1) for v in tap_db])

    spacing = args.bandwidth / args.subcarriers
    offsets = (torch.arange(args.subcarriers, device=paths.a.device) - args.subcarriers // 2)
    h = paths.cfr(offsets.float() * spacing)  # (R, Ra, T, Ta, time, F)
    magnitude_db = 20.0 * torch.log10(h[0, 0, 0, 0, 0].abs())
    print(f"CFR over {args.bandwidth / 1e6:.0f} MHz: mean {float(magnitude_db.mean()):.1f} dB, "
          f"min {float(magnitude_db.min()):.1f} dB, max {float(magnitude_db.max()):.1f} dB")
    total = float(a[0, 0, 0, 0, :, 0].abs().square().sum())
    print(f"wideband path gain (sum |a|^2): {10.0 * torch.log10(torch.tensor(total)):.2f} dB")


if __name__ == "__main__":
    main()

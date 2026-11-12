# Copyright Xingyu Chen.
# Calibrates a wall permittivity from simulated measurements with reverse-mode AD.

"""Calibrate a wall permittivity from simulated path-gain measurements with reverse-mode AD."""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import time

import torch

from witwin.core import AntennaState, PhysicalMaterial
from witwin.core.identity import new_antenna_id
from witwin.channel.radiomap import Coherent, solve
from witwin.channel.importers import load_mesh

import street_canyon

FREQUENCY_HZ = 3.5e9
SIGMA_E = 0.08  # S/m, assumed known; only eps_r is calibrated
RX_POSITIONS = [(x, y, 1.5) for x in (-20.0, -5.0, 10.0, 25.0, 40.0) for y in (-5.0, 5.0)]


def _scene(obj: Path, eps_r: torch.Tensor):
    walls = PhysicalMaterial(eps_r=eps_r, sigma_e=SIGMA_E, name="walls", thickness_m=0.3)
    ground = PhysicalMaterial(eps_r=15.0, sigma_e=0.035, name="ground", thickness_m=1.0)
    imported = load_mesh(obj, {"concrete": walls, "ground": ground})
    tx = AntennaState(new_antenna_id(), "tx", torch.tensor([-37.3, 1.1, 10.0]), power_w=1.0)
    receivers = [
        AntennaState(new_antenna_id(), "rx", torch.tensor(position)) for position in RX_POSITIONS
    ]
    return imported.to_scene(endpoints=[tx, *receivers])


def _gain_db(scene, ad_mode: str) -> torch.Tensor:
    # Incoherent power sums give a smooth loss in eps_r; coherent fields also differentiate.
    config = Coherent(
        max_depth=2, components={"los", "reflection"}, combine="power", ad_mode=ad_mode
    )
    result = solve(scene, config, reference_frequency_hz=FREQUENCY_HZ)
    return 10.0 * torch.log10(result.path_gain.reshape(-1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--true-eps", type=float, default=5.24)
    parser.add_argument("--start-eps", type=float, default=2.5)
    parser.add_argument("--steps", type=int, default=20, help="L-BFGS iterations")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as workdir:
        obj = street_canyon.write_obj(Path(workdir) / "canyon.obj")
        truth = torch.tensor(args.true_eps, device="cuda")
        with torch.no_grad():
            measured = _gain_db(_scene(obj, truth), "none")

        eps_r = torch.tensor(args.start_eps, device="cuda", requires_grad=True)
        # L-BFGS with a strong-Wolfe line search suits this small, smooth least-squares fit.
        optimizer = torch.optim.LBFGS(
            [eps_r], lr=1.0, max_iter=args.steps, line_search_fn="strong_wolfe",
            tolerance_grad=1e-9, tolerance_change=1e-12,
        )
        evaluations = 0

        def closure() -> torch.Tensor:
            nonlocal evaluations
            optimizer.zero_grad()
            # The scene is rebuilt per evaluation so the compiled material sees the new leaf.
            loss = (_gain_db(_scene(obj, eps_r), "vjp") - measured).square().mean()
            loss.backward()
            value = float(eps_r.detach())
            print(f"eval {evaluations:3d}  loss {float(loss.detach()):.4e} dB^2  eps_r {value:.4f}")
            evaluations += 1
            return loss

        start = time.perf_counter()
        optimizer.step(closure)
        elapsed = time.perf_counter() - start
    recovered = float(eps_r.detach())
    error = abs(recovered - args.true_eps)
    print(f"recovered eps_r {recovered:.4f} (true {args.true_eps}), |error| {error:.4f}")
    print(f"{evaluations} forward+backward evaluations in {elapsed:.2f} s")


if __name__ == "__main__":
    main()

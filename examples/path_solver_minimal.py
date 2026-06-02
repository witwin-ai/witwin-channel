from __future__ import annotations

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import witwin.channel as wc


def main() -> None:
    scene = wc.Scene(
        structures=[
            wc.Structure(
                name="wall",
                geometry=wc.Box(
                    position=(0.0, 0.0, 1.5),
                    size=(0.25, 4.0, 3.0),
                    device="cuda",
                ),
                material=wc.Material(eps_r=4.0, sigma_e=0.0),
            )
        ],
        transmitters=[
            wc.Transmitter(
                name="tx",
                position=wc.Point3f(-2.0, -1.0, 1.5),
            )
        ],
        receivers=[
            wc.Receiver(name="rx0", position=(-2.0, 1.0, 1.5)),
            wc.Receiver(name="rx1", position=(-1.5, 1.5, 1.5)),
        ],
        frequency=3.5e9,
        device="cuda",
    )
    result = wc.path.solve(
        scene=scene,
        transmitter="tx",
        receiver=["rx0", "rx1"],
        config=wc.path.Config(
            num_samples=256,
            max_bounces=1,
            max_diffraction_order=0,
            max_num_paths=4,
            return_geometry=True,
            edge_policy=wc.EdgePolicy(edge_selection_mode="all_edges"),
        ),
    )

    coeff, delay = result.cir()
    subcarriers = torch.linspace(3.49e9, 3.51e9, 8)
    response = result.cfr(subcarriers)
    reflection_paths = result.filter_by_type(wc.path.InteractionType.REFLECTION)

    print("path shape:", result.path_shape)
    print("paths per receiver:", result.num_paths)
    print("CIR coeff shape:", tuple(coeff.shape), "delay shape:", tuple(delay.shape))
    print("CFR shape:", tuple(response.shape))
    print("reflection paths per receiver:", reflection_paths.num_paths)
    print("path counts:", result.metadata["path_counts"])


if __name__ == "__main__":
    main()

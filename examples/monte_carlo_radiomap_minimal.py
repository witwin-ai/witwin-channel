from __future__ import annotations

import numpy as np
from witwin.channel.core.scene import ReceiverGrid, Scene, Transmitter
from witwin.core import Box, Material, Structure
from witwin.channel.montecarlo import Config, IntegratorOptions, solve


def main() -> None:
    scene = Scene(
        structures=[
            Structure(
                name="wall",
                geometry=Box(
                    position=(0.0, 0.0, 1.5),
                    size=(0.25, 4.0, 3.0),
                    device="cuda",
                ),
                material=Material(eps_r=4.0, sigma_e=0.0),
            ),
        ],
        transmitters=[
            Transmitter("tx", (-2.0, 0.0, 1.5)),
        ],
        receivers=[
            ReceiverGrid(
                "rm",
                axis="z",
                position=1.5,
                bounds=((-3.0, 3.0), (-3.0, 3.0)),
                grid_shape=(32, 32),
            ),
        ],
        frequency=3.5e9,
        device="cuda",
    )
    config = Config(
        num_samples=128,
        max_bounces=1,
        max_diffraction_order=0,
        integrator_options=IntegratorOptions(
            integrator="basic",
            samples_per_tx=4096,
            accumulation_backend="auto",
            seed=7,
        ),
    )
    result = solve(
        scene=scene,
        transmitter="tx",
        receiver="rm",
        config=config,
    )

    path_gain = np.asarray(result.path_gain, dtype=np.float32)
    print("path_gain shape:", path_gain.shape)
    print("path_gain max:", float(path_gain.max()))


if __name__ == "__main__":
    main()

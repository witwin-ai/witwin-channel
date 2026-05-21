from __future__ import annotations

import witwin.channel.montecarlo.solver as solver_module
from witwin.channel.core.scene import ReceiverGrid, Scene, Transmitter
from witwin.channel.montecarlo import Config, IntegratorOptions, solve


def test_montecarlo_solver_uses_scene_endpoint_polarization(monkeypatch):
    captured = {}

    class _FakeIntegrator:
        def integrate(
            self,
            tx_pos,
            grid,
            mc_config,
            scene,
            resolved_trace,
            solver_controls,
            **kwargs,
        ):
            captured["tx_polarization"] = resolved_trace.tx_polarization
            captured["rx_polarization"] = resolved_trace.rx_polarization
            captured["tx_power"] = kwargs["tx_power"]
            return object()

    monkeypatch.setattr(solver_module, "_INTEGRATORS", {"basic": _FakeIntegrator})

    scene = Scene(
        transmitters=[
            Transmitter(
                name="tx",
                position=(0.0, 0.0, 1.0),
                polarization=(0.0, 1.0, 0.0),
                power=2.5,
            )
        ],
        receivers=[
            ReceiverGrid(
                name="grid",
                axis="z",
                position=0.0,
                bounds=((-1.0, 1.0), (-1.0, 1.0)),
                grid_shape=(2, 2),
                polarization=(0.0, 0.0, 1.0),
            )
        ],
        frequency=3.5e9,
        device="cpu",
    )

    solve(
        scene=scene,
        transmitter="tx",
        receiver="grid",
        config=Config(integrator_options=IntegratorOptions(integrator="basic")),
    )

    assert captured == {
        "tx_polarization": (0.0, 1.0, 0.0),
        "rx_polarization": (0.0, 0.0, 1.0),
        "tx_power": 2.5,
    }

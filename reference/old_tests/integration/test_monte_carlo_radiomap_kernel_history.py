from __future__ import annotations

import json
from pathlib import Path

import drjit as dr
import pytest
import witwin as wt
from witwin.channel import RadioMapMonitor, Scene as LegacyScene, Tracer
import witwin.channel.monitors.radio_map.backend as legacy_rm_backend
from witwin.channel_scene import Scene as ChannelScene
from witwin.core import Box, Material, Structure
from witwin.montecarlo import (
    GridSpec,
    Config,
    solve,
)
import witwin.montecarlo.integrators.basic as package_rm_integrator


pytestmark = pytest.mark.gpu

_PACKAGE_KERNEL_HISTORY_BASELINE = (
    Path(__file__).resolve().parents[1]
    / "support"
    / "data"
    / "monte_carlo_radiomap_package_kernel_history_baseline.json"
)


BOUNDS = ((-10.0, 10.0), (-10.0, 10.0))
TX_POS = (0.0, -5.0, 4.0)
CUBE_CENTERS = (
    (-2.5, -3.0, 1.5),
    (2.0, 0.5, 1.5),
    (-0.5, 3.5, 1.5),
)
CUBE_SIZE = (2.0, 2.0, 2.0)


def _structures():
    material = Material(eps_r=4.0, sigma_e=0.0)
    return [
        Structure(
            name=f"cube_{index}",
            geometry=Box(position=center, size=CUBE_SIZE, device="cuda"),
            material=material,
        )
        for index, center in enumerate(CUBE_CENTERS)
    ]


def _build_legacy_scene() -> LegacyScene:
    return LegacyScene(
        structures=_structures(),
        device="cuda",
        edge_selection_mode="all_edges",
    )


def _build_channel_scene() -> ChannelScene:
    return ChannelScene(
        structures=_structures(),
        device="cuda",
        edge_selection_mode="all_edges",
    )


def _legacy_forward(scene: LegacyScene):
    tracer = Tracer(
        frequency=1.0e9,
        scene=scene,
        reflection_n_rays=96,
        reflection_max_bounces=1,
        reflection_coef=0.8,
        enable_rd_diffraction=True,
        max_diffractions=1,
    )
    monitor = RadioMapMonitor(
        "legacy_mc_kernel_history",
        axis="z",
        position=1.0,
        bounds=BOUNDS,
        grid_shape=(32, 32),
        metric="path_gain",
        combine_mode="incoherent",
        receiver_model="matched_isotropic",
        accumulation_backend="auto",
        sampling_mode="monte_carlo",
        samples_per_tx=96,
        max_diffractions=1,
        shadow_boundary_mode="none",
        seed=7,
    )
    result = tracer.trace(wt.Point3f(*TX_POS), monitor=monitor, verbose=False)
    dr.eval(result.path_gain)
    dr.sync_thread()


def _package_forward(scene):
    result = solve(
        scene=scene,
        frequency=1.0e9,
        tx_pos=wt.Point3f(*TX_POS),
        grid=GridSpec(
            axis="z",
            position=1.0,
            bounds=BOUNDS,
            grid_shape=(32, 32),
        ),
        config=Config(
            reflection_n_rays=96,
            reflection_max_bounces=1,
            samples_per_tx=96,
            enable_rd_diffraction=True,
            max_diffractions=1,
            accumulation_backend="auto",
            shadow_boundary_mode="none",
            seed=7,
        ),
    )
    dr.eval(result.path_gain)
    dr.sync_thread()


def _steady_state_kernel_history(operation):
    with dr.scoped_set_flag(dr.JitFlag.KernelHistory, True):
        operation()
        dr.sync_thread()
        dr.kernel_history_clear()
        operation()
        dr.sync_thread()
        return list(dr.kernel_history())


def _small_jit_kernel_count(history) -> int:
    return sum(
        1
        for entry in history
        if entry.get("type") == dr.KernelType.JIT
        and int(entry.get("size", 0)) <= 8
        and int(entry.get("operation_count", 1 << 30)) <= 24
    )


def test_standalone_forward_kernel_history_matches_legacy(monkeypatch):
    monkeypatch.setattr(legacy_rm_backend, "native_extension_available", lambda: True)
    monkeypatch.setattr(
        package_rm_integrator.NativeExtension,
        "native_extension_available",
        staticmethod(lambda: True),
    )

    legacy_scene = _build_legacy_scene()
    package_scene = _build_channel_scene()
    baseline = json.loads(_PACKAGE_KERNEL_HISTORY_BASELINE.read_text(encoding="utf-8"))

    legacy_count = len(_steady_state_kernel_history(lambda: _legacy_forward(legacy_scene)))
    package_history = _steady_state_kernel_history(lambda: _package_forward(package_scene))
    package_count = len(package_history)
    small_jit_count = _small_jit_kernel_count(package_history)

    assert legacy_count > 0
    assert package_count <= int(baseline["steady_state_kernel_count"])
    assert small_jit_count <= int(baseline["max_small_jit_kernel_count"])

"""Standalone multipath total-field and gradient visual test."""

from __future__ import annotations

import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest
if os.environ.get("WITWIN_CHANNEL_MAIN_SHOW", "0") != "1":
    os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from samples.save_multipath_main_component_gradient_figure import (
    QUANTITY_COMPLEX_FIELD as _QUANTITY_COMPLEX_FIELD,
    _fd_total_power_map as _component_fd_total_power_map,
    _field_components_to_numpy as _component_field_to_numpy,
    _power_contribution_maps as _component_power_contribution_maps,
    prepare_figure_data as _prepare_component_figure_data,
    make_figure as _save_component_figure,
)
from tests.main.plot_multipath_components import (
    CUBE1_BASE_CENTER as _CUBE1_BASE_CENTER,
    TX_POS as _TX_POS,
    TRACE_BOUNDS as _TRACE_BOUNDS,
    ad_gradient_field as _ad_gradient_field,
    as_grid as _as_grid,
    build_trace_payload as _build_trace_payload,
    cube_specs as _cube_specs,
    decorate_axis as _decorate_axis,
    fd_gradient_field as _fd_gradient_field,
    gradient_db_magnitude as _gradient_db_magnitude,
    parameter_config as _parameter_config,
    trace_total_power as _trace_total_power,
)

pytestmark = pytest.mark.gpu

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_PATH = OUTPUT_DIR / "multipath.png"
COMPONENT_OUTPUT_PATHS = {
    "tx_x": OUTPUT_DIR / "multipath_tx_x_components.png",
    "cube1_x": OUTPUT_DIR / "multipath_cube1_x_components.png",
}
FIGURE_PARAMETERS = ("tx_x", "cube1_x")


def _log_stage(message: str) -> None:
    print(f"[test_multipath_main] {message}", flush=True)


def _run_timed_stage(label: str, fn, /, *args, **kwargs):
    _log_stage(f"start {label}")
    started = time.perf_counter()
    value = fn(*args, **kwargs)
    elapsed = time.perf_counter() - started
    _log_stage(f"done {label} in {elapsed:.2f}s")
    return value


def _component_figures_enabled() -> bool:
    return os.environ.get("WITWIN_CHANNEL_MAIN_COMPONENTS", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def test_multipath_build_trace_payload_fails_fast_before_trace_on_non_native_config():
    tracer = SimpleNamespace(
        trace_called=False,
        config=SimpleNamespace(
            trace=SimpleNamespace(
                reflection_field_backend="drjit",
                diffraction_execution=SimpleNamespace(suffix_backend="native"),
            )
        ),
    )

    def _unexpected_trace(*args, **kwargs):
        tracer.trace_called = True
        raise AssertionError("build_trace_payload should fail before tracer.trace() is called.")

    tracer.trace = _unexpected_trace

    with pytest.raises(RuntimeError, match="reflection_field_backend='native'"):
        _build_trace_payload(
            cube1_x=_CUBE1_BASE_CENTER[0],
            tx_pos=_TX_POS,
            grid_size=4,
            n_rays=8,
            scene=SimpleNamespace(tri_data_gpu=None),
            monitor=SimpleNamespace(axis="z"),
            tracer=tracer,
        )

    assert tracer.trace_called is False


def test_multipath_build_trace_payload_smoke_uses_native_backends():
    payload = _build_trace_payload(
        cube1_x=_CUBE1_BASE_CENTER[0],
        tx_pos=_TX_POS,
        grid_size=16,
        n_rays=64,
    )
    metadata = payload["result"].primary.metadata

    assert metadata["reflection_backend"]["resolved_backend"] == "native"
    assert metadata["reflection_backend"]["implementation"] in {"native_cuda_custom_op", "epc"}
    assert metadata["diffraction_accumulation_backend"]["implementation"] == "native_cuda_custom_op"
    assert metadata["reflection_suffix_backend"]["resolved_backend"] == "native"
    assert metadata["reflection_suffix_backend"]["implementation"] == "native_cuda_custom_op"


def _panel_stats_text(data: np.ndarray) -> str:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return "mean=nan med=nan std=nan"
    return (
        f"mean={float(np.mean(finite)):.2f} "
        f"med={float(np.median(finite)):.2f} "
        f"std={float(np.std(finite)):.2f}"
    )


def _build_parameter_artifacts(
    parameter: str,
    *,
    grid_size: int,
    n_rays: int,
    fd_step: float,
    component_figures_enabled: bool,
):
    config = _parameter_config(parameter)
    artifacts = {
        "parameter": parameter,
        "config": config,
    }
    if component_figures_enabled:
        artifacts["component_figure_data"] = _run_timed_stage(
            f"prepare_component_figure_data[{parameter}]",
            _prepare_component_figure_data,
            parameter_name=parameter,
            grid_size=grid_size,
            n_rays=n_rays,
            fd_step=fd_step,
        )
    return artifacts


def _parameter_row_payload(artifacts: dict, *, grid_size: int, n_rays: int, fd_step: float):
    config = artifacts["config"]
    component_data = artifacts.get("component_figure_data")
    if component_data is not None:
        total_field = _component_field_to_numpy(component_data["result"]["a_tot"], grid_size)
        ad_power_maps = _component_power_contribution_maps(total_field, component_data["ad_maps"])
        total_db = 10.0 * np.log10(total_field["power"] + 1e-20)
        ad_np = ad_power_maps["a_tot"]["signed"]
        fd_np = _component_fd_total_power_map(
            component_data["result_plus"],
            component_data["result_minus"],
            grid_size,
            fd_step,
        )
    else:
        _, _, total_power = _run_timed_stage(
            f"trace_total_power[{artifacts['parameter']}]",
            _trace_total_power,
            cube1_x=config["cube1_x"],
            tx_pos=config["tx_pos"],
            grid_size=grid_size,
            n_rays=n_rays,
        )
        total_db = 10.0 * np.log10(_as_grid(total_power, grid_size) + 1e-20)
        _, _, ad_gradient = _run_timed_stage(
            f"ad_gradient_field[{artifacts['parameter']}]",
            _ad_gradient_field,
            artifacts["parameter"],
            grid_size,
            n_rays,
        )
        ad_np = _as_grid(ad_gradient, grid_size)
        fd_gradient = _run_timed_stage(
            f"fd_gradient_field[{artifacts['parameter']}]",
            _fd_gradient_field,
            artifacts["parameter"],
            grid_size,
            n_rays,
            fd_step,
        )
        fd_np = _as_grid(fd_gradient, grid_size)

    ad_vis = _gradient_db_magnitude(ad_np)
    fd_vis = _gradient_db_magnitude(fd_np)
    diff_vis = ad_vis - fd_vis
    diff_vmax = max(float(np.percentile(np.abs(diff_vis), 99.5)), 3.0)

    return {
        "label": config["label"],
        "specs": _cube_specs(config["cube1_x"]),
        "tx_xy": (config["tx_pos"][0], config["tx_pos"][1]),
        "extent": (
            float(_TRACE_BOUNDS[0][0]),
            float(_TRACE_BOUNDS[0][1]),
            float(_TRACE_BOUNDS[1][0]),
            float(_TRACE_BOUNDS[1][1]),
        ),
        "total_db": total_db,
        "ad_vis": ad_vis,
        "fd_vis": fd_vis,
        "diff_vis": diff_vis,
        "grad_vmax": max(float(np.percentile(ad_vis, 99.5)), float(np.percentile(fd_vis, 99.5))),
        "diff_vmax": diff_vmax,
    }


def _save_failure_figure(exc: Exception) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10, 2.8))
    fig.text(
        0.03,
        0.65,
        "multipath test failed before the full figure completed",
        fontsize=12,
        family="monospace",
    )
    fig.text(
        0.03,
        0.35,
        f"{type(exc).__name__}: {exc}",
        fontsize=10,
        family="monospace",
        wrap=True,
    )
    fig.savefig(OUTPUT_PATH, dpi=180)
    plt.close(fig)


def test_multipath_total_field_and_gradients_main():
    grid_size = 256
    n_rays = 1_280
    fd_step = 1e-3
    component_figures_enabled = _component_figures_enabled()
    fig = None

    try:
        _log_stage(f"test start grid_size={grid_size} n_rays={n_rays} fd_step={fd_step}")
        parameter_artifacts = {
            parameter: _run_timed_stage(
                f"build_parameter_artifacts[{parameter}]",
                _build_parameter_artifacts,
                parameter,
                grid_size=grid_size,
                n_rays=n_rays,
                fd_step=fd_step,
                component_figures_enabled=component_figures_enabled,
            )
            for parameter in FIGURE_PARAMETERS
        }
        row_payloads = [
            _run_timed_stage(
                f"parameter_row_payload[{parameter}]",
                _parameter_row_payload,
                parameter_artifacts[parameter],
                grid_size=grid_size,
                n_rays=n_rays,
                fd_step=fd_step,
            )
            for parameter in FIGURE_PARAMETERS
        ]

        _log_stage("start build matplotlib figure")
        fig, axes = plt.subplots(len(row_payloads), 4, figsize=(16, 8.8), constrained_layout=True, squeeze=False)
        _log_stage("done build matplotlib figure")

        for row_idx, payload in enumerate(row_payloads):
            row_axes = axes[row_idx]
            grad_vmin = payload["grad_vmax"] - 60.0
            total_title = f"Total Field (dB)\nbase for d/d{payload['label']}"
            ad_title = f"AD d|E|^2/d{payload['label']} (dB)"
            fd_title = f"FD d|E|^2/d{payload['label']} (dB)"
            diff_title = f"AD - FD for d/d{payload['label']} (dB)"
            if row_idx == 0:
                total_title = f"{total_title}\n{_panel_stats_text(payload['total_db'])}"
                ad_title = f"{ad_title}\n{_panel_stats_text(payload['ad_vis'])}"
                fd_title = f"{fd_title}\n{_panel_stats_text(payload['fd_vis'])}"
                diff_title = f"{diff_title}\n{_panel_stats_text(payload['diff_vis'])}"

            im_total = row_axes[0].imshow(
                payload["total_db"],
                origin="lower",
                extent=payload["extent"],
                cmap="jet",
                vmin=-60.0,
                vmax=-20.0,
                interpolation="nearest",
            )
            _decorate_axis(
                row_axes[0],
                payload["specs"],
                payload["tx_xy"],
                total_title,
            )

            im_ad = row_axes[1].imshow(
                payload["ad_vis"],
                origin="lower",
                extent=payload["extent"],
                cmap="RdBu_r",
                vmin=grad_vmin,
                vmax=payload["grad_vmax"],
                interpolation="nearest",
            )
            _decorate_axis(row_axes[1], payload["specs"], payload["tx_xy"], ad_title)

            im_fd = row_axes[2].imshow(
                payload["fd_vis"],
                origin="lower",
                extent=payload["extent"],
                cmap="RdBu_r",
                vmin=grad_vmin,
                vmax=payload["grad_vmax"],
                interpolation="nearest",
            )
            _decorate_axis(row_axes[2], payload["specs"], payload["tx_xy"], fd_title)

            im_diff = row_axes[3].imshow(
                payload["diff_vis"],
                origin="lower",
                extent=payload["extent"],
                cmap="RdBu_r",
                vmin=-payload["diff_vmax"],
                vmax=payload["diff_vmax"],
                interpolation="nearest",
            )
            _decorate_axis(row_axes[3], payload["specs"], payload["tx_xy"], diff_title)

            fig.colorbar(im_total, ax=row_axes[0], shrink=0.82)
            fig.colorbar(im_ad, ax=row_axes[1], shrink=0.82)
            fig.colorbar(im_fd, ax=row_axes[2], shrink=0.82)
            fig.colorbar(im_diff, ax=row_axes[3], shrink=0.82)

        fig.suptitle(
            "Multipath Total Field And Gradients\n"
            f"rows={', '.join(FIGURE_PARAMETERS)}, grid={grid_size}, n_rays={n_rays}, fd_step={fd_step}",
            fontsize=14,
        )
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        _log_stage(f"start save main figure -> {OUTPUT_PATH.name}")
        fig.savefig(OUTPUT_PATH, dpi=180)
        _log_stage(f"done save main figure -> {OUTPUT_PATH.name}")

        if component_figures_enabled:
            for parameter_name, component_output_path in COMPONENT_OUTPUT_PATHS.items():
                _run_timed_stage(
                    f"save_component_figure[{parameter_name}]",
                    _save_component_figure,
                    parameter_name=parameter_name,
                    output_path=component_output_path,
                    grid_size=grid_size,
                    n_rays=n_rays,
                    fd_step=fd_step,
                    expand_mixed=True,
                    quantity=_QUANTITY_COMPLEX_FIELD,
                    figure_data=parameter_artifacts[parameter_name]["component_figure_data"],
                )
        else:
            _log_stage(
                "skip component figures; set WITWIN_CHANNEL_MAIN_COMPONENTS=1 to enable"
            )

        assert OUTPUT_PATH.exists()
        assert OUTPUT_PATH.stat().st_size > 0
        if component_figures_enabled:
            for component_output_path in COMPONENT_OUTPUT_PATHS.values():
                assert component_output_path.exists()
                assert component_output_path.stat().st_size > 0
        _log_stage("test completed successfully")
    except Exception as exc:
        _log_stage(f"test failed with {type(exc).__name__}: {exc}")
        if fig is not None:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            fig.savefig(OUTPUT_PATH, dpi=180)
        else:
            _save_failure_figure(exc)
        raise
    finally:
        if fig is not None:
            plt.close(fig)

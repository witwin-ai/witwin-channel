"""Integrated visual test for position, rotation, and TX gradients under use_scene_materials."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import witwin as wt
from tests.main.test_position_rotation_tx import (
    _position_ad_gradient,
    _position_fd_gradient,
    _relative_l2_error,
    _render_row,
    _rotation_ad_gradient,
    _rotation_fd_gradient,
    _to_db,
    _tx_ad_gradient,
    _tx_fd_gradient,
)
from witwin.channel import DEFAULT_VARIANT, Material, draw_scene, to_numpy, to_power_db
if os.environ.get("WITWIN_CHANNEL_MAIN_SHOW", "0") != "1":
    os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt

pytestmark = pytest.mark.gpu

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_PATH = OUTPUT_DIR / "position_rotation_tx_use_scene_materials.png"
PEC_LIKE_COMPARE_OUTPUT_PATH = OUTPUT_DIR / "position_rotation_tx_use_scene_materials_pec_like_legacy_compare.png"
SCENE_MATERIAL_EPS_R = 1.0e4
PEC_LIKE_SCENE_MATERIAL_EPS_R = 1.0e4


def test_position_rotation_tx_use_scene_materials_visual_grid():
    params = {
        "grid_size": 512,
        "freq": 1e9,
        "range_x": (-8.0, 8.0),
        "range_y": (-8.0, 8.0),
        "center_vals": (0.0, 0.0, 2.0),
        "size": 4.0,
        "tx_vals": (-5.0, 5.0, 1.5),
        "rotation_val": float(np.deg2rad(15.0)),
        "n_rays": 20_000,
        "max_reflections": 1,
        "reflection_coef": 1.0,
    }
    scene_material_kwargs = {
        "material": Material(eps_r=SCENE_MATERIAL_EPS_R, sigma_e=0.0),
        "use_scene_materials_for_reflection": True,
        "use_scene_materials_for_diffraction": True,
    }
    pec_like_scene_material_kwargs = {
        "material": Material(eps_r=PEC_LIKE_SCENE_MATERIAL_EPS_R, sigma_e=0.0),
        "use_scene_materials_for_reflection": True,
        "use_scene_materials_for_diffraction": True,
    }

    pos_result, pos_scene, pos_ad = _position_ad_gradient(**scene_material_kwargs, **params)
    _, _, pos_fd = _position_fd_gradient(**scene_material_kwargs, **params)
    rot_result, rot_scene, rot_ad = _rotation_ad_gradient(**scene_material_kwargs, **params)
    _, _, rot_fd = _rotation_fd_gradient(**scene_material_kwargs, **params)
    tx_result, tx_scene, tx_ad = _tx_ad_gradient(**scene_material_kwargs, **params)
    _, _, tx_fd = _tx_fd_gradient(**scene_material_kwargs, **params)
    legacy_pos_result, legacy_pos_scene, legacy_pos_ad = _position_ad_gradient(**params)
    legacy_rot_result, legacy_rot_scene, legacy_rot_ad = _rotation_ad_gradient(**params)
    legacy_tx_result, legacy_tx_scene, legacy_tx_ad = _tx_ad_gradient(**params)
    pec_pos_result, pec_pos_scene, pec_pos_ad = _position_ad_gradient(**pec_like_scene_material_kwargs, **params)
    pec_rot_result, pec_rot_scene, pec_rot_ad = _rotation_ad_gradient(**pec_like_scene_material_kwargs, **params)
    pec_tx_result, pec_tx_scene, pec_tx_ad = _tx_ad_gradient(**pec_like_scene_material_kwargs, **params)

    grid_size = params["grid_size"]
    tx_vals = params["tx_vals"]
    range_x = params["range_x"]
    range_y = params["range_y"]

    fig, axes = plt.subplots(3, 4, figsize=(19, 14), constrained_layout=True)

    pos_field = to_numpy(to_power_db(pos_result.primary.field.total)).reshape(grid_size, grid_size)
    pos_edges = pos_scene.get_edge_data(pos_result.primary.plane_position)["edges_2d"]
    _render_row(
        axes[0],
        field_db=pos_field,
        grad_ad_db=_to_db(pos_ad.reshape(grid_size, grid_size)),
        grad_fd_db=_to_db(pos_fd.reshape(grid_size, grid_size)),
        edges=pos_edges,
        tx_pos=tx_vals,
        range_x=range_x,
        range_y=range_y,
        row_title="Position (Scene Materials)",
    )

    rot_field = to_numpy(to_power_db(rot_result.primary.field.total)).reshape(grid_size, grid_size)
    rot_edges = rot_scene.get_edge_data(rot_result.primary.plane_position)["edges_2d"]
    _render_row(
        axes[1],
        field_db=rot_field,
        grad_ad_db=_to_db(rot_ad.reshape(grid_size, grid_size)),
        grad_fd_db=_to_db(rot_fd.reshape(grid_size, grid_size)),
        edges=rot_edges,
        tx_pos=tx_vals,
        range_x=range_x,
        range_y=range_y,
        row_title="Rotation (Scene Materials)",
    )

    tx_field = to_numpy(to_power_db(tx_result.primary.field.total)).reshape(grid_size, grid_size)
    tx_edges = tx_scene.get_edge_data(tx_result.primary.plane_position)["edges_2d"]
    _render_row(
        axes[2],
        field_db=tx_field,
        grad_ad_db=_to_db(tx_ad.reshape(grid_size, grid_size)),
        grad_fd_db=_to_db(tx_fd.reshape(grid_size, grid_size)),
        edges=tx_edges,
        tx_pos=tx_vals,
        range_x=range_x,
        range_y=range_y,
        row_title="TX Position (Scene Materials)",
    )

    fig.suptitle(
        (
            "Integrated Gradient Visual Test: Position, Rotation, TX with "
            f"use_scene_materials=True (eps_r={SCENE_MATERIAL_EPS_R:.0f})"
        ),
        fontsize=14,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=180)
    plt.close(fig)

    compare_fig, compare_axes = plt.subplots(3, 3, figsize=(15, 14), constrained_layout=True)
    extent = [range_x[0], range_x[1], range_y[0], range_y[1]]
    compare_rows = (
        {
            "row_idx": 0,
            "title": "Position",
            "legacy_result": legacy_pos_result,
            "legacy_scene": legacy_pos_scene,
            "legacy_grad": legacy_pos_ad,
            "pec_result": pec_pos_result,
            "pec_scene": pec_pos_scene,
            "pec_grad": pec_pos_ad,
        },
        {
            "row_idx": 1,
            "title": "Rotation",
            "legacy_result": legacy_rot_result,
            "legacy_scene": legacy_rot_scene,
            "legacy_grad": legacy_rot_ad,
            "pec_result": pec_rot_result,
            "pec_scene": pec_rot_scene,
            "pec_grad": pec_rot_ad,
        },
        {
            "row_idx": 2,
            "title": "TX Position",
            "legacy_result": legacy_tx_result,
            "legacy_scene": legacy_tx_scene,
            "legacy_grad": legacy_tx_ad,
            "pec_result": pec_tx_result,
            "pec_scene": pec_tx_scene,
            "pec_grad": pec_tx_ad,
        },
    )
    pec_like_rels = []
    current_rels = (
        _relative_l2_error(pos_ad, legacy_pos_ad),
        _relative_l2_error(rot_ad, legacy_rot_ad),
        _relative_l2_error(tx_ad, legacy_tx_ad),
    )
    for row, current_rel in zip(compare_rows, current_rels):
        legacy_db = _to_db(row["legacy_grad"].reshape(grid_size, grid_size))
        pec_db = _to_db(row["pec_grad"].reshape(grid_size, grid_size))
        diff = row["pec_grad"].reshape(grid_size, grid_size) - row["legacy_grad"].reshape(grid_size, grid_size)
        rel = _relative_l2_error(row["pec_grad"], row["legacy_grad"])
        pec_like_rels.append(rel)
        vmax = max(float(np.percentile(legacy_db, 99.5)), float(np.percentile(pec_db, 99.5)))
        vmin = vmax - 60.0
        diff_vmax = max(float(np.percentile(np.abs(diff), 99.5)), 1e-12)
        edges = row["legacy_scene"].get_edge_data(row["legacy_result"].primary.plane_position)["edges_2d"]
        panels = (
            {
                "ax": compare_axes[row["row_idx"], 0],
                "image": legacy_db,
                "cmap": "inferno",
                "vmin": vmin,
                "vmax": vmax,
                "title": f"Legacy {row['title']} Gradient (dB)",
                "note": "legacy scalar reflection / PEC diffraction",
            },
            {
                "ax": compare_axes[row["row_idx"], 1],
                "image": pec_db,
                "cmap": "inferno",
                "vmin": vmin,
                "vmax": vmax,
                "title": f"Near-PEC {row['title']} Gradient (dB)",
                "note": (
                    f"use_scene_materials eps_r={PEC_LIKE_SCENE_MATERIAL_EPS_R:.0e}\n"
                    f"current eps_r={SCENE_MATERIAL_EPS_R:.0f} vs legacy rel-L2={current_rel:.2e}"
                ),
            },
            {
                "ax": compare_axes[row["row_idx"], 2],
                "image": diff,
                "cmap": "RdBu_r",
                "vmin": -diff_vmax,
                "vmax": diff_vmax,
                "title": f"Near-PEC Minus Legacy {row['title']} (Linear)",
                "note": f"rel-L2 vs legacy={rel:.2e}",
            },
        )
        for panel in panels:
            image = panel["ax"].imshow(
                panel["image"],
                extent=extent,
                origin="lower",
                cmap=panel["cmap"],
                vmin=panel["vmin"],
                vmax=panel["vmax"],
            )
            draw_scene(panel["ax"], edges, tx_vals, range_x, range_y)
            panel["ax"].set_title(panel["title"], fontsize=10)
            panel["ax"].text(
                0.02,
                0.02,
                panel["note"],
                transform=panel["ax"].transAxes,
                fontsize=8,
                color="white",
                va="bottom",
                ha="left",
                bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 3.0},
            )
            plt.colorbar(image, ax=panel["ax"], shrink=0.84)

    compare_fig.suptitle(
        (
            "Near-PEC Scene Materials vs Legacy Gradient Comparison "
            f"(eps_r={PEC_LIKE_SCENE_MATERIAL_EPS_R:.0e})"
        ),
        fontsize=14,
    )
    compare_fig.savefig(PEC_LIKE_COMPARE_OUTPUT_PATH, dpi=180)
    plt.close(compare_fig)

    assert OUTPUT_PATH.exists()
    assert OUTPUT_PATH.stat().st_size > 0
    assert PEC_LIKE_COMPARE_OUTPUT_PATH.exists()
    assert PEC_LIKE_COMPARE_OUTPUT_PATH.stat().st_size > 0
    assert float(np.sum(pos_ad)) > 0.0
    assert float(np.sum(pos_fd)) > 0.0
    assert float(np.sum(rot_ad)) > 0.0
    assert float(np.sum(rot_fd)) > 0.0
    assert float(np.sum(tx_ad)) > 0.0
    assert float(np.sum(tx_fd)) > 0.0
    assert pos_result.primary.metadata["reflection_model_source"] == "scene"
    assert pos_result.primary.metadata["diffraction_face_material_source"] == "scene"
    assert rot_result.primary.metadata["reflection_model_source"] == "scene"
    assert rot_result.primary.metadata["diffraction_face_material_source"] == "scene"
    assert tx_result.primary.metadata["reflection_model_source"] == "scene"
    assert tx_result.primary.metadata["diffraction_face_material_source"] == "scene"
    assert pec_pos_result.primary.metadata["reflection_model_source"] == "scene"
    assert pec_pos_result.primary.metadata["diffraction_face_material_source"] == "scene"
    assert all(np.isfinite(rel) for rel in pec_like_rels)

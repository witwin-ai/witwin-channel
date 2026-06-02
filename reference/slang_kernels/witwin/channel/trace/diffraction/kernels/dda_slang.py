"""Slang-backed DDA traversal helpers for diffraction suffix accumulation."""

from __future__ import annotations

from pathlib import Path

import drjit as dr
import torch
import witwin as wt

from ....utils import drjit_to_torch_view
from ..execution import experimental_slang_enabled
from .slang_runtime import launch_shape_1d, load_slang_module


_BLOCK_SIZE = 256


def _get_dda_module():
    return load_slang_module(Path(__file__).with_name("dda_traverse.slang"))


def slang_dda_available() -> bool:
    if not experimental_slang_enabled():
        return False
    try:
        return _get_dda_module() is not None
    except Exception:
        return False


def _float_tensor_view(value, *, detach=False) -> torch.Tensor:
    return drjit_to_torch_view(value, detach=detach, dtype=torch.float32).contiguous()


def _int_tensor_view(value, *, detach=False) -> torch.Tensor:
    return drjit_to_torch_view(value, detach=detach, dtype=torch.int32).contiguous()


def _scalar_float(value) -> float:
    if isinstance(value, (float, int)):
        return float(value)
    return float(dr.slice(value))


def _has_active_gradients(*values) -> bool:
    for value in values:
        if value is None:
            continue
        if dr.grad_enabled(value):
            return True
    return False


def run_suffix_dda_slang(
    *,
    grid,
    grid_data,
    rx_z,
    seg_origin,
    seg_dir,
    blocker_dist,
    seg_field,
    seg_vector,
    state_idx,
    wavelength,
    k,
    active,
    result_real,
    result_imag,
    result_count,
    result_vector_real_x,
    result_vector_imag_x,
    result_vector_real_y,
    result_vector_imag_y,
    result_vector_real_z,
    result_vector_imag_z,
    max_steps,
) -> bool:
    if not slang_dda_available():
        return False
    if dr.backend_v(type(seg_origin.x)) != dr.JitBackend.CUDA:
        return False
    if _has_active_gradients(
        seg_origin,
        seg_dir,
        blocker_dist,
        seg_field,
        seg_vector["x"],
        seg_vector["y"],
        seg_vector["z"],
    ):
        return False

    module = _get_dda_module()
    active_mask = _int_tensor_view(dr.select(active, wt.Int32(1), wt.Int32(0)), detach=True)
    if int(active_mask.numel()) == 0:
        return True

    (x_min, x_max), (y_min, y_max) = grid.bounds
    cell_size_x, cell_size_y = grid.cell_size
    nx, _ny = grid.size
    n_rx = grid.n_cells

    block_size = (_BLOCK_SIZE, 1, 1)
    module.suffixDdaTraverseForward(
        activeMask=active_mask,
        stateIndex=_int_tensor_view(state_idx, detach=True),
        segOriginX=_float_tensor_view(seg_origin.x, detach=True),
        segOriginY=_float_tensor_view(seg_origin.y, detach=True),
        segOriginZ=_float_tensor_view(seg_origin.z, detach=True),
        segDirX=_float_tensor_view(seg_dir.x, detach=True),
        segDirY=_float_tensor_view(seg_dir.y, detach=True),
        segDirZ=_float_tensor_view(seg_dir.z, detach=True),
        blockerDist=_float_tensor_view(blocker_dist, detach=True),
        segFieldReal=_float_tensor_view(seg_field.real, detach=True),
        segFieldImag=_float_tensor_view(seg_field.imag, detach=True),
        segVectorXReal=_float_tensor_view(seg_vector["x"].real, detach=True),
        segVectorXImag=_float_tensor_view(seg_vector["x"].imag, detach=True),
        segVectorYReal=_float_tensor_view(seg_vector["y"].real, detach=True),
        segVectorYImag=_float_tensor_view(seg_vector["y"].imag, detach=True),
        segVectorZReal=_float_tensor_view(seg_vector["z"].real, detach=True),
        segVectorZImag=_float_tensor_view(seg_vector["z"].imag, detach=True),
        xCoords=_float_tensor_view(grid_data["x_coords"], detach=True),
        yCoords=_float_tensor_view(grid_data["y_coords"], detach=True),
        resultReal=_float_tensor_view(result_real),
        resultImag=_float_tensor_view(result_imag),
        resultCount=_float_tensor_view(result_count),
        resultVectorRealX=_float_tensor_view(result_vector_real_x),
        resultVectorImagX=_float_tensor_view(result_vector_imag_x),
        resultVectorRealY=_float_tensor_view(result_vector_real_y),
        resultVectorImagY=_float_tensor_view(result_vector_imag_y),
        resultVectorRealZ=_float_tensor_view(result_vector_real_z),
        resultVectorImagZ=_float_tensor_view(result_vector_imag_z),
        xMin=float(x_min),
        xMax=float(x_max),
        yMin=float(y_min),
        yMax=float(y_max),
        cellSizeX=float(cell_size_x),
        cellSizeY=float(cell_size_y),
        nx=int(nx),
        nRx=int(n_rx),
        maxSteps=int(max_steps),
        rxZ=float(rx_z),
        wavelength=float(wavelength),
        k=_scalar_float(k),
    ).launchRaw(
        blockSize=block_size,
        gridSize=launch_shape_1d(int(active_mask.shape[0]), block_size[0]),
    )
    return True


__all__ = [
    "run_suffix_dda_slang",
    "slang_dda_available",
]

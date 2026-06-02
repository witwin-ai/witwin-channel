# Native UTD Receiver-Tiling Reference

Status: Superseded
Category: Archive
Last reviewed: 2026-04-09

## Purpose

This document preserves the removed UTD receiver-tiling rollout code that previously lived in
`witwin/channel/kernels/trace/utd/native_impl.py`.

The archived code below is historical reference only. It is not part of the active execution path
and must not be treated as the source of truth for current runtime behavior.

## Removal Reason

- The tiled UTD rollout was detached from the main execution path.
- Current public UTD execution stays on the validated Dr.Jit finite-wedge path until the native
  full-cartesian kernels are revalidated.
- The archived code is kept only so the previous planner-dependent rollout can be inspected without
  keeping it in the runtime module.

## Archived Imports And Constants

```python
from witwin.channel.kernels.monitors.common.receiver_tiles import resolve_receiver_tiles
from witwin.channel.kernels.monitors.common.receiver_tiles.native_impl import (
    receiver_index_for_tile_slot,
)
from witwin.channel.kernels.monitors.common.utd_state_tiles import build_utd_state_tile_plan

_UTD_TILED_MAX_PAIR_RATIO = 0.60
```

## Archived `_utd_tiled_replay_eligible`

```python
def _utd_tiled_replay_eligible(state_arrays: dict, rx_pos, receiver_tiles) -> bool:
    if receiver_tiles is None or int(receiver_tiles.n_tiles) <= 1:
        return False
    if _state_arrays_have_grad(state_arrays):
        return False
    if _point_has_grad(rx_pos):
        return False
    return True
```

## Archived `utd_accumulate_tiled_vector_power_pairs`

```python
def utd_accumulate_tiled_vector_power_pairs(
    state_arrays: dict,
    rx_pos,
    state_idx,
    receiver_idx,
    *,
    n_output_rx: int,
    k: float,
    wavelength: float | None = None,
    material_detail=None,
    rx_polarization=None,
    ownership_code=None,
    valid_mask=None,
):
    _require_finite_edge_bounds(
        state_arrays,
        context="Native tiled UTD vector-power accumulation",
    )
    ext = _extension()
    local_n_states = int(dr.width(state_idx))
    local_n_receivers = int(dr.width(receiver_idx))
    n_rx = int(n_output_rx)
    if local_n_states <= 0 or local_n_receivers <= 0 or n_rx <= 0:
        zero_scalar = complex_zero(n_rx)
        zero_vector = vector_zero(n_rx)
        zero_power = dr.zeros(wt.Float, n_rx)
        dr.eval(
            zero_scalar.real,
            zero_scalar.imag,
            zero_vector["x"].real,
            zero_vector["x"].imag,
            zero_vector["y"].real,
            zero_vector["y"].imag,
            zero_vector["z"].real,
            zero_vector["z"].imag,
            zero_power,
        )
        return (
            eval_complex(zero_scalar),
            eval_complex(zero_scalar),
            vector_eval(zero_vector),
            vector_eval(zero_vector),
            zero_power,
            0,
        )

    native_rx_pos = _materialize_receiver_positions(rx_pos)
    active_rx_pol = effective_rx_polarization(rx_polarization, (1.0, 0.0, 0.0))
    mat = _build_material_params(ext, material_detail, wavelength)
    out_buffers = _zero_pair_output_buffers(n_rx)
    matched_power = dr.zeros(wt.Float, n_rx)
    valid_pair_count = dr.zeros(wt.Float, 1)
    soa = _pack_state_soa(state_arrays)
    ownership = _ownership_codes(state_arrays) if ownership_code is None else ownership_code
    native_state_idx = wt.Int32(state_idx)
    native_receiver_idx = wt.Int32(receiver_idx)
    native_valid_mask = None if valid_mask is None else wt.Int32(valid_mask)
    native_ownership = wt.Int32(ownership)
    dr.eval(
        native_state_idx,
        native_receiver_idx,
        native_ownership,
        matched_power,
        valid_pair_count,
    )
    output_arrays = _pair_output_arrays(out_buffers) + (matched_power, valid_pair_count)
    ext.utd_accumulate_tiled_vector_power_into(
        _detach_native_index_array(native_state_idx),
        _detach_native_index_array(native_receiver_idx),
        False if native_valid_mask is None else _detach_native_index_array(native_valid_mask),
        _detach_native_index_array(native_ownership),
        soa,
        (native_rx_pos.x, native_rx_pos.y, native_rx_pos.z),
        output_arrays,
        mat,
        local_n_states,
        local_n_receivers,
        k,
        float(active_rx_pol[0]),
        float(active_rx_pol[1]),
        float(active_rx_pol[2]),
    )

    dr.eval(*out_buffers.values(), matched_power, valid_pair_count)
    direct_total = eval_complex(wt.Complex2f(out_buffers["direct_re"], out_buffers["direct_im"]))
    multi_total = eval_complex(wt.Complex2f(out_buffers["multi_re"], out_buffers["multi_im"]))
    direct_vector_total = vector_eval(
        {
            "x": wt.Complex2f(out_buffers["direct_vec_x_re"], out_buffers["direct_vec_x_im"]),
            "y": wt.Complex2f(out_buffers["direct_vec_y_re"], out_buffers["direct_vec_y_im"]),
            "z": wt.Complex2f(out_buffers["direct_vec_z_re"], out_buffers["direct_vec_z_im"]),
        }
    )
    multi_vector_total = vector_eval(
        {
            "x": wt.Complex2f(out_buffers["multi_vec_x_re"], out_buffers["multi_vec_x_im"]),
            "y": wt.Complex2f(out_buffers["multi_vec_y_re"], out_buffers["multi_vec_y_im"]),
            "z": wt.Complex2f(out_buffers["multi_vec_z_re"], out_buffers["multi_vec_z_im"]),
        }
    )
    return (
        direct_total,
        multi_total,
        direct_vector_total,
        multi_vector_total,
        matched_power,
        int(float(valid_pair_count[0])),
    )
```

## Archived `_utd_accumulate_forward_native_tiled_primal`

```python
def _utd_accumulate_forward_native_tiled_primal(
    state_arrays: dict,
    rx_pos,
    k: float,
    n_edges: int,
    return_per_edge: bool,
    *,
    scene=None,
    wavelength: float | None = None,
    material_detail=None,
    rx_polarization=None,
    receiver_axis: str = "z",
    execution=None,
    receiver_tiles=None,
):
    _require_finite_edge_bounds(
        state_arrays,
        context="Native tiled UTD accumulation",
    )
    from witwin.channel.trace.diffraction.geometry import (
        _edge_owner_structure_idx,
        _segment_visibility_mask,
    )

    del n_edges, return_per_edge, execution
    plan = build_utd_state_tile_plan(
        state_arrays=state_arrays,
        receiver_tiles=receiver_tiles,
    )
    if plan is None or plan.tile_task_count <= 0:
        return None

    ext = _extension()
    n_rx = dr.width(rx_pos.x)
    n_states = int(state_arrays["n_states"])
    full_pair_count = int(n_states * n_rx)
    estimated_pair_count = int(plan.estimated_pair_count)
    if estimated_pair_count >= full_pair_count:
        return None
    if estimated_pair_count > int(full_pair_count * _UTD_TILED_MAX_PAIR_RATIO):
        return None

    active_rx_pol = (1.0, 0.0, 0.0) if rx_polarization is None else rx_polarization
    native_rx_pos = _materialize_receiver_positions(rx_pos)
    mat = _build_material_params(ext, material_detail, wavelength)
    soa = _pack_state_soa(state_arrays)
    ownership = _ownership_codes(state_arrays)

    out_buffers = {
        "direct_re": dr.zeros(wt.Float, n_rx),
        "direct_im": dr.zeros(wt.Float, n_rx),
        "multi_re": dr.zeros(wt.Float, n_rx),
        "multi_im": dr.zeros(wt.Float, n_rx),
        "direct_vec_x_re": dr.zeros(wt.Float, n_rx),
        "direct_vec_x_im": dr.zeros(wt.Float, n_rx),
        "direct_vec_y_re": dr.zeros(wt.Float, n_rx),
        "direct_vec_y_im": dr.zeros(wt.Float, n_rx),
        "direct_vec_z_re": dr.zeros(wt.Float, n_rx),
        "direct_vec_z_im": dr.zeros(wt.Float, n_rx),
        "multi_vec_x_re": dr.zeros(wt.Float, n_rx),
        "multi_vec_x_im": dr.zeros(wt.Float, n_rx),
        "multi_vec_y_re": dr.zeros(wt.Float, n_rx),
        "multi_vec_y_im": dr.zeros(wt.Float, n_rx),
        "multi_vec_z_re": dr.zeros(wt.Float, n_rx),
        "multi_vec_z_im": dr.zeros(wt.Float, n_rx),
    }
    dr.eval(*out_buffers.values())
    output_arrays = (
        out_buffers["direct_re"],
        out_buffers["direct_im"],
        out_buffers["multi_re"],
        out_buffers["multi_im"],
        out_buffers["direct_vec_x_re"],
        out_buffers["direct_vec_x_im"],
        out_buffers["direct_vec_y_re"],
        out_buffers["direct_vec_y_im"],
        out_buffers["direct_vec_z_re"],
        out_buffers["direct_vec_z_im"],
        out_buffers["multi_vec_x_re"],
        out_buffers["multi_vec_x_im"],
        out_buffers["multi_vec_y_re"],
        out_buffers["multi_vec_y_im"],
        out_buffers["multi_vec_z_re"],
        out_buffers["multi_vec_z_im"],
    )

    for tile_idx in range(int(plan.n_tiles)):
        tile_keep_idx = dr.compress(plan.tile_task_tile_idx == wt.UInt32(tile_idx))
        local_n_states = dr.width(tile_keep_idx)
        if local_n_states <= 0:
            continue
        local_n_receivers = int(
            receiver_tiles.tile_extent_0[tile_idx] * receiver_tiles.tile_extent_1[tile_idx]
        )
        if local_n_states <= 0 or local_n_receivers <= 0:
            continue

        tile_state_idx = wt.Int32(dr.gather(wt.UInt32, plan.tile_task_state_idx, tile_keep_idx))
        tile_rx_idx = wt.Int32(
            receiver_index_for_tile_slot(
                receiver_tiles,
                wt.UInt32(tile_idx),
                dr.arange(wt.UInt32, local_n_receivers),
            )
        )
        state_chunk_size = _cartesian_chunk_size(local_n_states, local_n_receivers)

        for state_start in range(0, local_n_states, state_chunk_size):
            chunk_n_states = min(state_chunk_size, local_n_states - state_start)
            chunk_state_idx = dr.gather(
                wt.Int32,
                tile_state_idx,
                dr.arange(wt.UInt32, chunk_n_states) + wt.UInt32(state_start),
            )
            valid_mask = None
            if scene is not None:
                n_pairs = chunk_n_states * local_n_receivers
                pair_idx = dr.arange(wt.UInt32, n_pairs)
                local_state_slot = pair_idx // local_n_receivers
                local_rx_slot = pair_idx % local_n_receivers
                state_idx = dr.gather(wt.Int32, chunk_state_idx, local_state_slot)
                rx_idx = wt.Int32(dr.gather(wt.Int32, tile_rx_idx, local_rx_slot))
                edge_pos = dr.gather(wt.Point3f, state_arrays["edge_pos"], state_idx)
                adj0 = dr.gather(wt.Int32, state_arrays["adjacent_face0"], state_idx)
                adj1 = dr.gather(wt.Int32, state_arrays["adjacent_face1"], state_idx)
                owner_structure_idx = _edge_owner_structure_idx(scene, adj0, adj1)
                batch_rx = wt.Point3f(
                    dr.gather(wt.Float, native_rx_pos.x, rx_idx),
                    dr.gather(wt.Float, native_rx_pos.y, rx_idx),
                    dr.gather(wt.Float, native_rx_pos.z, rx_idx),
                )
                visible = _segment_visibility_mask(
                    edge_pos,
                    batch_rx,
                    scene,
                    ignore_prim_idx=(adj0, adj1),
                    ignore_structure_idx=owner_structure_idx,
                )
                valid_mask = dr.select(visible, wt.Int32(1), wt.Int32(0))

            ext.utd_accumulate_tiled_into(
                _detach_native_index_array(chunk_state_idx),
                _detach_native_index_array(tile_rx_idx),
                False if valid_mask is None else _detach_native_index_array(valid_mask),
                _detach_native_index_array(ownership),
                soa,
                (native_rx_pos.x, native_rx_pos.y, native_rx_pos.z),
                output_arrays,
                mat,
                chunk_n_states,
                local_n_receivers,
                k,
            )

    direct_vector_total = {
        "x": wt.Complex2f(out_buffers["direct_vec_x_re"], out_buffers["direct_vec_x_im"]),
        "y": wt.Complex2f(out_buffers["direct_vec_y_re"], out_buffers["direct_vec_y_im"]),
        "z": wt.Complex2f(out_buffers["direct_vec_z_re"], out_buffers["direct_vec_z_im"]),
    }
    multi_vector_total = {
        "x": wt.Complex2f(out_buffers["multi_vec_x_re"], out_buffers["multi_vec_x_im"]),
        "y": wt.Complex2f(out_buffers["multi_vec_y_re"], out_buffers["multi_vec_y_im"]),
        "z": wt.Complex2f(out_buffers["multi_vec_z_re"], out_buffers["multi_vec_z_im"]),
    }
    direct_total = scalarize_tangential_jones(
        tangential_jones(direct_vector_total, axis=receiver_axis),
        active_rx_pol,
        axis=receiver_axis,
    )
    multi_total = scalarize_tangential_jones(
        tangential_jones(multi_vector_total, axis=receiver_axis),
        active_rx_pol,
        axis=receiver_axis,
    )
    return direct_total, multi_total, direct_vector_total, multi_vector_total, []
```

"""Exact path calculation helpers for reflection chains."""

from dataclasses import dataclass

import drjit as dr
import witwin as wt

from ...utils.constants import EPS
from ...utils.drjit_ops import ArrayInit, Gather
from ...utils.polarization import (
    project_real_polarization_to_ray,
    reflect_field_vector,
    vector_from_scalar_and_real_direction,
    vector_select,
    vector_zero,
)
from ..diffraction.geometry import _segment_visibility_mask, _triangle_surface_group_id
from ..materials import coerce_reflection_trace_detail, reflection_material_omega, resolve_surface_material
from ...utils.geometry import point_in_triangle_3d, reflect_point_across_plane, surface_contains_point


def _zero_float(width: int):
    return dr.zeros(wt.Float, width)


def _zero_point(width: int):
    return ArrayInit.zeros_point3(width)


def _zero_vector(width: int):
    return ArrayInit.zeros_vector3(width)


def _zero_complex(width: int):
    return ArrayInit.complex_zero(width)


def _point_from_grad(value, width: int):
    if value is None:
        return _zero_point(width)
    return wt.Point3f(value.x, value.y, value.z)


def _vector_from_grad(value, width: int):
    if value is None:
        return _zero_vector(width)
    return wt.Vector3f(value.x, value.y, value.z)


def _detach_point(value):
    return wt.Point3f(dr.detach(value.x), dr.detach(value.y), dr.detach(value.z))


def _detach_vector(value):
    return wt.Vector3f(dr.detach(value.x), dr.detach(value.y), dr.detach(value.z))


def _set_point_grad(value, tangent):
    dr.enable_grad(value.x, value.y, value.z)
    dr.set_grad(value.x, tangent.x)
    dr.set_grad(value.y, tangent.y)
    dr.set_grad(value.z, tangent.z)


def _set_vector_grad(value, tangent):
    dr.enable_grad(value.x, value.y, value.z)
    dr.set_grad(value.x, tangent.x)
    dr.set_grad(value.y, tangent.y)
    dr.set_grad(value.z, tangent.z)


def _surface_group_ignore_tuple(tri_data, prim_candidates) -> tuple[wt.Int32, ...]:
    if not isinstance(prim_candidates, (tuple, list)):
        prim_candidates = (prim_candidates,)
    groups: list[wt.Int32] = []
    for prim_idx in prim_candidates:
        if prim_idx is None:
            continue
        groups.append(_triangle_surface_group_id(tri_data, wt.Int32(prim_idx)))
    return tuple(groups)


def _gather_epc_slot_arrays(paths, path_idx, scene, reflection_detail):
    detail = coerce_reflection_trace_detail(reflection_detail)
    chain_depth = int(paths.chain_depth)
    width = dr.width(path_idx)
    valid_mask = dr.full(wt.Bool, True, width)

    image_source = Gather.point3(paths.image_source, path_idx)
    slot_plane_points = []
    slot_plane_normals = []
    slot_eta_r = []
    slot_sigma = []
    slot_gain = []
    for slot in range(chain_depth):
        plane_point = Gather.point3(paths.plane_point(slot), path_idx)
        plane_normal = Gather.vector3(paths.plane_normal(slot), path_idx)
        prim_idx = dr.gather(wt.Int32, paths.prim_idx(slot), path_idx)
        material_inputs = resolve_surface_material(
            scene=scene,
            prim_idx=prim_idx,
            override_material=detail.reflection_material,
            reflection_coef=detail.reflection_gain,
            default_eta_r=5.0,
            default_sigma=0.0,
            valid_mask=valid_mask,
            use_scene_materials=detail.use_scene_materials,
        )
        slot_plane_points.append(plane_point)
        slot_plane_normals.append(plane_normal)
        slot_eta_r.append(material_inputs["eta_r"])
        slot_sigma.append(material_inputs["sigma"])
        slot_gain.append(material_inputs["gain"])
    return {
        "image_source": image_source,
        "slot_plane_points": slot_plane_points,
        "slot_plane_normals": slot_plane_normals,
        "slot_eta_r": slot_eta_r,
        "slot_sigma": slot_sigma,
        "slot_gain": slot_gain,
        "chain_depth": chain_depth,
    }


def _flatten_slot_point_arrays(slot_points):
    if not slot_points:
        return _zero_point(0)
    return wt.Point3f(
        dr.concat([slot.x for slot in slot_points]),
        dr.concat([slot.y for slot in slot_points]),
        dr.concat([slot.z for slot in slot_points]),
    )


def _flatten_slot_vector_arrays(slot_vectors):
    if not slot_vectors:
        return _zero_vector(0)
    return wt.Vector3f(
        dr.concat([slot.x for slot in slot_vectors]),
        dr.concat([slot.y for slot in slot_vectors]),
        dr.concat([slot.z for slot in slot_vectors]),
    )


def _flatten_slot_scalars(slot_values):
    if not slot_values:
        return _zero_float(0)
    return dr.concat(slot_values)


@dataclass(frozen=True)
class PreparedReflectionEpcDescriptor:
    source_path_idx: wt.UInt32
    image_source: wt.Point3f
    slot_plane_point: wt.Point3f
    slot_plane_normal: wt.Vector3f
    slot_eta_r: wt.Float
    slot_sigma: wt.Float
    slot_gain: wt.Float
    chain_depth: int
    n_paths: int


def build_reflection_epc_descriptor(
    *,
    paths,
    path_idx,
    scene,
    reflection_detail,
) -> PreparedReflectionEpcDescriptor:
    chain_depth = int(0 if paths is None else paths.chain_depth)
    n_paths = int(dr.width(path_idx))
    if chain_depth <= 0 or n_paths <= 0:
        return PreparedReflectionEpcDescriptor(
            source_path_idx=dr.zeros(wt.UInt32, 0),
            image_source=_zero_point(0),
            slot_plane_point=_zero_point(0),
            slot_plane_normal=_zero_vector(0),
            slot_eta_r=_zero_float(0),
            slot_sigma=_zero_float(0),
            slot_gain=_zero_float(0),
            chain_depth=max(0, chain_depth),
            n_paths=0,
        )

    slot_arrays = _gather_epc_slot_arrays(paths, path_idx, scene, reflection_detail)
    return PreparedReflectionEpcDescriptor(
        source_path_idx=wt.UInt32(path_idx),
        image_source=slot_arrays["image_source"],
        slot_plane_point=_flatten_slot_point_arrays(slot_arrays["slot_plane_points"]),
        slot_plane_normal=_flatten_slot_vector_arrays(slot_arrays["slot_plane_normals"]),
        slot_eta_r=_flatten_slot_scalars(slot_arrays["slot_eta_r"]),
        slot_sigma=_flatten_slot_scalars(slot_arrays["slot_sigma"]),
        slot_gain=_flatten_slot_scalars(slot_arrays["slot_gain"]),
        chain_depth=chain_depth,
        n_paths=n_paths,
    )


def _epc_reflection_chain_math_reference(
    *,
    path_idx,
    image_source,
    slot_plane_point,
    slot_plane_normal,
    slot_eta_r,
    slot_sigma,
    slot_gain,
    target_pos,
    tx_polarization=(1.0, 0.0, 0.0),
    chain_depth: int,
    n_paths: int,
    wavelength: float,
):
    width = dr.width(target_pos.x)
    if width == 0 or chain_depth <= 0 or int(n_paths) <= 0:
        return {
            "geom_valid": dr.zeros(wt.Float, width),
            "tx_pos": _zero_point(width),
            "chain_vector": vector_zero(width),
            "hit_points": [],
            "hit_x": _zero_float(0),
            "hit_y": _zero_float(0),
            "hit_z": _zero_float(0),
        }

    descriptor_path_idx = wt.UInt32(path_idx)
    current_source = wt.Point3f(
        dr.gather(wt.Float, image_source.x, descriptor_path_idx),
        dr.gather(wt.Float, image_source.y, descriptor_path_idx),
        dr.gather(wt.Float, image_source.z, descriptor_path_idx),
    )
    current_target = target_pos
    hit_points_rev = []
    normals_rev = []
    geom_valid = dr.full(wt.Bool, True, width)

    for slot in range(chain_depth - 1, -1, -1):
        base = descriptor_path_idx + wt.UInt32(slot * int(n_paths))
        plane_point = wt.Point3f(
            dr.gather(wt.Float, slot_plane_point.x, base),
            dr.gather(wt.Float, slot_plane_point.y, base),
            dr.gather(wt.Float, slot_plane_point.z, base),
        )
        plane_normal = wt.Vector3f(
            dr.gather(wt.Float, slot_plane_normal.x, base),
            dr.gather(wt.Float, slot_plane_normal.y, base),
            dr.gather(wt.Float, slot_plane_normal.z, base),
        )
        segment = current_target - current_source
        denom = dr.dot(segment, plane_normal)
        denom_safe = dr.select(
            dr.abs(denom) > EPS,
            denom,
            dr.select(denom >= 0.0, wt.Float(EPS), wt.Float(-EPS)),
        )
        t_hit = dr.dot(plane_point - current_source, plane_normal) / denom_safe
        geom_valid = geom_valid & (dr.abs(denom) > EPS) & (t_hit > EPS) & (t_hit < (1.0 - EPS))
        hit_p = current_source + t_hit * segment
        hit_points_rev.append(hit_p)
        normals_rev.append(plane_normal)
        current_target = hit_p
        current_source = reflect_point_across_plane(current_source, plane_point, plane_normal)

    tx_pos = current_source
    hit_points = list(reversed(hit_points_rev))
    normals = list(reversed(normals_rev))
    first_segment_dir = (hit_points[0] - tx_pos) / (dr.norm(hit_points[0] - tx_pos) + EPS)
    chain_vector = vector_from_scalar_and_real_direction(
        wt.Complex2f(dr.ones(wt.Float, width), dr.zeros(wt.Float, width)),
        project_real_polarization_to_ray(tx_polarization, first_segment_dir),
    )
    material_omega = reflection_material_omega(wavelength)
    prev_point = tx_pos
    for slot, (hit_p, geom_n) in enumerate(zip(hit_points, normals)):
        base = descriptor_path_idx + wt.UInt32(slot * int(n_paths))
        eta_r = dr.gather(wt.Float, slot_eta_r, base)
        sigma = dr.gather(wt.Float, slot_sigma, base)
        gain = dr.gather(wt.Float, slot_gain, base)
        incoming = hit_p - prev_point
        incoming = incoming / (dr.norm(incoming) + EPS)
        chain_vector = reflect_field_vector(
            chain_vector,
            incoming,
            geom_n,
            eta_r=eta_r,
            sigma=sigma,
            omega=material_omega,
            gain=gain,
        )
        prev_point = hit_p

    return {
        "geom_valid": wt.Float(geom_valid),
        "tx_pos": tx_pos,
        "chain_vector": chain_vector,
        "hit_points": hit_points,
        "hit_x": dr.concat([hit_p.x for hit_p in hit_points]),
        "hit_y": dr.concat([hit_p.y for hit_p in hit_points]),
        "hit_z": dr.concat([hit_p.z for hit_p in hit_points]),
    }


def _reference_epc_forward_grads(
    *,
    path_idx,
    image_source,
    slot_plane_point,
    slot_plane_normal,
    slot_eta_r,
    slot_sigma,
    slot_gain,
    target_pos,
    tx_polarization,
    chain_depth,
    n_paths,
    wavelength,
    tangent_image_source,
    tangent_slot_plane_point,
    tangent_slot_plane_normal,
    tangent_target_pos,
):
    image_source_ad = _detach_point(image_source)
    slot_plane_point_ad = _detach_point(slot_plane_point)
    slot_plane_normal_ad = _detach_vector(slot_plane_normal)
    target_pos_ad = _detach_point(target_pos)

    _set_point_grad(image_source_ad, tangent_image_source)
    _set_point_grad(slot_plane_point_ad, tangent_slot_plane_point)
    _set_vector_grad(slot_plane_normal_ad, tangent_slot_plane_normal)
    _set_point_grad(target_pos_ad, tangent_target_pos)

    outputs = _epc_reflection_chain_math_reference(
        path_idx=path_idx,
        image_source=image_source_ad,
        slot_plane_point=slot_plane_point_ad,
        slot_plane_normal=slot_plane_normal_ad,
        slot_eta_r=slot_eta_r,
        slot_sigma=slot_sigma,
        slot_gain=slot_gain,
        target_pos=target_pos_ad,
        tx_polarization=tx_polarization,
        chain_depth=chain_depth,
        n_paths=n_paths,
        wavelength=wavelength,
    )
    differentiable_outputs = (
        outputs["geom_valid"],
        outputs["tx_pos"].x,
        outputs["tx_pos"].y,
        outputs["tx_pos"].z,
        outputs["chain_vector"]["x"].real,
        outputs["chain_vector"]["x"].imag,
        outputs["chain_vector"]["y"].real,
        outputs["chain_vector"]["y"].imag,
        outputs["chain_vector"]["z"].real,
        outputs["chain_vector"]["z"].imag,
        outputs["hit_x"],
        outputs["hit_y"],
        outputs["hit_z"],
    )
    dr.forward_to(differentiable_outputs)
    return tuple(dr.grad(value) for value in differentiable_outputs)


def _reference_epc_backward_grads(
    *,
    path_idx,
    image_source,
    slot_plane_point,
    slot_plane_normal,
    slot_eta_r,
    slot_sigma,
    slot_gain,
    target_pos,
    tx_polarization,
    chain_depth,
    n_paths,
    wavelength,
    grad_outputs,
):
    image_source_ad = _detach_point(image_source)
    slot_plane_point_ad = _detach_point(slot_plane_point)
    slot_plane_normal_ad = _detach_vector(slot_plane_normal)
    target_pos_ad = _detach_point(target_pos)
    dr.enable_grad(
        image_source_ad.x, image_source_ad.y, image_source_ad.z,
        slot_plane_point_ad.x, slot_plane_point_ad.y, slot_plane_point_ad.z,
        slot_plane_normal_ad.x, slot_plane_normal_ad.y, slot_plane_normal_ad.z,
        target_pos_ad.x, target_pos_ad.y, target_pos_ad.z,
    )
    outputs = _epc_reflection_chain_math_reference(
        path_idx=path_idx,
        image_source=image_source_ad,
        slot_plane_point=slot_plane_point_ad,
        slot_plane_normal=slot_plane_normal_ad,
        slot_eta_r=slot_eta_r,
        slot_sigma=slot_sigma,
        slot_gain=slot_gain,
        target_pos=target_pos_ad,
        tx_polarization=tx_polarization,
        chain_depth=chain_depth,
        n_paths=n_paths,
        wavelength=wavelength,
    )
    loss = dr.zeros(wt.Float, 1)
    differentiable_outputs = (
        outputs["geom_valid"],
        outputs["tx_pos"].x,
        outputs["tx_pos"].y,
        outputs["tx_pos"].z,
        outputs["chain_vector"]["x"].real,
        outputs["chain_vector"]["x"].imag,
        outputs["chain_vector"]["y"].real,
        outputs["chain_vector"]["y"].imag,
        outputs["chain_vector"]["z"].real,
        outputs["chain_vector"]["z"].imag,
        outputs["hit_x"],
        outputs["hit_y"],
        outputs["hit_z"],
    )
    for output, grad_output in zip(differentiable_outputs, grad_outputs):
        if grad_output is None:
            continue
        loss += dr.sum(output * grad_output)
    dr.backward(loss)
    return {
        "image_source": wt.Point3f(
            dr.grad(image_source_ad.x),
            dr.grad(image_source_ad.y),
            dr.grad(image_source_ad.z),
        ),
        "slot_plane_point": wt.Point3f(
            dr.grad(slot_plane_point_ad.x),
            dr.grad(slot_plane_point_ad.y),
            dr.grad(slot_plane_point_ad.z),
        ),
        "slot_plane_normal": wt.Vector3f(
            dr.grad(slot_plane_normal_ad.x),
            dr.grad(slot_plane_normal_ad.y),
            dr.grad(slot_plane_normal_ad.z),
        ),
        "target_pos": wt.Point3f(
            dr.grad(target_pos_ad.x),
            dr.grad(target_pos_ad.y),
            dr.grad(target_pos_ad.z),
        ),
    }


def _launch_reflection_epc_targets_forward(
    *,
    path_idx,
    image_source,
    slot_plane_point,
    slot_plane_normal,
    slot_eta_r,
    slot_sigma,
    slot_gain,
    target_pos,
    tx_polarization,
    n_pairs,
    n_paths,
    chain_depth,
    wavelength,
):
    from ..._native import _extension

    ext = _extension()
    raw_outputs = ext.reflection_epc_targets_forward_arrays(
        wt.Int32(path_idx),
        image_source.x,
        image_source.y,
        image_source.z,
        slot_plane_point.x,
        slot_plane_point.y,
        slot_plane_point.z,
        slot_plane_normal.x,
        slot_plane_normal.y,
        slot_plane_normal.z,
        slot_eta_r,
        slot_sigma,
        slot_gain,
        target_pos.x,
        target_pos.y,
        target_pos.z,
        float(tx_polarization[0]),
        float(tx_polarization[1]),
        float(tx_polarization[2]),
        int(n_pairs),
        int(n_paths),
        int(chain_depth),
        float(2.0 * dr.pi / wavelength),
        float(reflection_material_omega(wavelength)[0]),
    )
    (
        geom_valid_i32,
        tx_pos_x,
        tx_pos_y,
        tx_pos_z,
        vec_x_re,
        vec_x_im,
        vec_y_re,
        vec_y_im,
        vec_z_re,
        vec_z_im,
        hit_x,
        hit_y,
        hit_z,
    ) = raw_outputs
    return (
        wt.Float(geom_valid_i32),
        tx_pos_x,
        tx_pos_y,
        tx_pos_z,
        vec_x_re,
        vec_x_im,
        vec_y_re,
        vec_y_im,
        vec_z_re,
        vec_z_im,
        hit_x,
        hit_y,
        hit_z,
    )


class _ReflectionEpcTargetsOp(dr.CustomOp):
    def eval(
        self,
        path_idx,
        image_source,
        slot_plane_point,
        slot_plane_normal,
        slot_eta_r,
        slot_sigma,
        slot_gain,
        target_pos,
        *,
        tx_polarization,
        n_pairs,
        n_paths,
        chain_depth,
        wavelength,
    ):
        self.path_idx = path_idx
        self.image_source = image_source
        self.slot_plane_point = slot_plane_point
        self.slot_plane_normal = slot_plane_normal
        self.slot_eta_r = slot_eta_r
        self.slot_sigma = slot_sigma
        self.slot_gain = slot_gain
        self.target_pos = target_pos
        self.tx_polarization = tx_polarization
        self.n_pairs = int(n_pairs)
        self.n_paths = int(n_paths)
        self.chain_depth = int(chain_depth)
        self.wavelength = float(wavelength)
        return _launch_reflection_epc_targets_forward(
            path_idx=path_idx,
            image_source=image_source,
            slot_plane_point=slot_plane_point,
            slot_plane_normal=slot_plane_normal,
            slot_eta_r=slot_eta_r,
            slot_sigma=slot_sigma,
            slot_gain=slot_gain,
            target_pos=target_pos,
            tx_polarization=tx_polarization,
            n_pairs=self.n_pairs,
            n_paths=self.n_paths,
            chain_depth=self.chain_depth,
            wavelength=self.wavelength,
        )

    def forward(self):
        width = dr.width(self.image_source.x)
        slot_width = dr.width(self.slot_plane_point.x)
        grads = _reference_epc_forward_grads(
            path_idx=self.path_idx,
            image_source=self.image_source,
            slot_plane_point=self.slot_plane_point,
            slot_plane_normal=self.slot_plane_normal,
            slot_eta_r=self.slot_eta_r,
            slot_sigma=self.slot_sigma,
            slot_gain=self.slot_gain,
            target_pos=self.target_pos,
            tx_polarization=self.tx_polarization,
            chain_depth=self.chain_depth,
            n_paths=self.n_paths,
            wavelength=self.wavelength,
            tangent_image_source=_point_from_grad(self.grad_in("image_source"), width),
            tangent_slot_plane_point=_point_from_grad(self.grad_in("slot_plane_point"), slot_width),
            tangent_slot_plane_normal=_vector_from_grad(self.grad_in("slot_plane_normal"), slot_width),
            tangent_target_pos=_point_from_grad(self.grad_in("target_pos"), width),
        )
        self.set_grad_out(grads)

    def backward(self):
        grads = _reference_epc_backward_grads(
            path_idx=self.path_idx,
            image_source=self.image_source,
            slot_plane_point=self.slot_plane_point,
            slot_plane_normal=self.slot_plane_normal,
            slot_eta_r=self.slot_eta_r,
            slot_sigma=self.slot_sigma,
            slot_gain=self.slot_gain,
            target_pos=self.target_pos,
            tx_polarization=self.tx_polarization,
            chain_depth=self.chain_depth,
            n_paths=self.n_paths,
            wavelength=self.wavelength,
            grad_outputs=self.grad_out(),
        )
        self.set_grad_in("image_source", grads["image_source"])
        self.set_grad_in("slot_plane_point", grads["slot_plane_point"])
        self.set_grad_in("slot_plane_normal", grads["slot_plane_normal"])
        self.set_grad_in("slot_eta_r", dr.zeros(wt.Float, dr.width(self.slot_eta_r)))
        self.set_grad_in("slot_sigma", dr.zeros(wt.Float, dr.width(self.slot_sigma)))
        self.set_grad_in("slot_gain", dr.zeros(wt.Float, dr.width(self.slot_gain)))
        self.set_grad_in("target_pos", grads["target_pos"])


def _epc_reflection_targets_custom_op(
    *,
    path_idx,
    image_source,
    slot_plane_point,
    slot_plane_normal,
    slot_eta_r,
    slot_sigma,
    slot_gain,
    target_pos,
    tx_polarization,
    n_pairs,
    n_paths,
    chain_depth,
    wavelength,
):
    return dr.custom(
        _ReflectionEpcTargetsOp,
        path_idx,
        image_source,
        slot_plane_point,
        slot_plane_normal,
        slot_eta_r,
        slot_sigma,
        slot_gain,
        target_pos,
        tx_polarization=tx_polarization,
        n_pairs=int(n_pairs),
        n_paths=int(n_paths),
        chain_depth=int(chain_depth),
        wavelength=float(wavelength),
    )


def _finalize_epc_outputs(
    *,
    raw_outputs,
    paths,
    path_idx,
    target_pos,
    scene,
    target_adjacent_faces,
    chain_depth,
    return_geometry,
    return_endpoints,
):
    width = dr.width(target_pos.x)
    (
        geom_valid_f,
        tx_pos_x,
        tx_pos_y,
        tx_pos_z,
        vec_x_re,
        vec_x_im,
        vec_y_re,
        vec_y_im,
        vec_z_re,
        vec_z_im,
        hit_x_flat,
        hit_y_flat,
        hit_z_flat,
    ) = raw_outputs

    tx_pos = wt.Point3f(tx_pos_x, tx_pos_y, tx_pos_z)
    chain_vector = {
        "x": wt.Complex2f(vec_x_re, vec_x_im),
        "y": wt.Complex2f(vec_y_re, vec_y_im),
        "z": wt.Complex2f(vec_z_re, vec_z_im),
    }
    geom_valid = geom_valid_f > 0.5

    tri_data = scene.tri_data_gpu
    tri_surface_data = {
        "group_size": tri_data["surface_group_size"],
        "group_members": tri_data["surface_group_members"],
        "max_group_size": int(tri_data["surface_max_group_size"]),
    }
    target_adjacent_surface_groups = _surface_group_ignore_tuple(
        tri_data,
        target_adjacent_faces,
    )
    slot_base = dr.arange(wt.UInt32, width)
    if return_endpoints and not return_geometry:
        valid = geom_valid
        prev_point = tx_pos
        prev_surface_group = None
        first_hit = _zero_point(width)
        last_hit = _zero_point(width)
        for slot in range(chain_depth):
            slot_idx = slot_base + wt.UInt32(slot * width)
            hit_p = wt.Point3f(
                dr.gather(wt.Float, hit_x_flat, slot_idx),
                dr.gather(wt.Float, hit_y_flat, slot_idx),
                dr.gather(wt.Float, hit_z_flat, slot_idx),
            )
            prim_idx = dr.gather(wt.Int32, paths.prim_idx(slot), path_idx)
            valid_prim = prim_idx >= 0
            if int(tri_surface_data["max_group_size"]) == 2:
                safe_idx = wt.UInt32(dr.select(valid_prim, prim_idx, wt.Int32(0)))
                group_members = tri_surface_data["group_members"]
                flat_idx0 = safe_idx * wt.UInt32(2)
                flat_idx1 = flat_idx0 + wt.UInt32(1)
                member_idx0_i32 = dr.gather(wt.Int32, group_members, flat_idx0)
                member_idx1_i32 = dr.gather(wt.Int32, group_members, flat_idx1)
                active0 = valid_prim & (member_idx0_i32 >= 0)
                active1 = valid_prim & (member_idx1_i32 >= 0)
                safe_member_idx0 = wt.UInt32(dr.select(active0, member_idx0_i32, wt.Int32(0)))
                safe_member_idx1 = wt.UInt32(dr.select(active1, member_idx1_i32, wt.Int32(0)))
                hit0 = active0 & point_in_triangle_3d(
                    hit_p,
                    dr.gather(wt.Point3f, tri_data["v0"], safe_member_idx0),
                    dr.gather(wt.Point3f, tri_data["v1"], safe_member_idx0),
                    dr.gather(wt.Point3f, tri_data["v2"], safe_member_idx0),
                )
                hit1 = active1 & point_in_triangle_3d(
                    hit_p,
                    dr.gather(wt.Point3f, tri_data["v0"], safe_member_idx1),
                    dr.gather(wt.Point3f, tri_data["v1"], safe_member_idx1),
                    dr.gather(wt.Point3f, tri_data["v2"], safe_member_idx1),
                )
                surface_hit = hit0 | hit1
            else:
                surface_hit = surface_contains_point(
                    hit_p,
                    prim_idx,
                    tri_data["v0"],
                    tri_data["v1"],
                    tri_data["v2"],
                    tri_surface_data,
                )
            valid = valid & (prim_idx >= 0) & surface_hit
            prim_surface_group = _triangle_surface_group_id(tri_data, prim_idx)
            ignore_surface_group_idx = (
                (prim_surface_group,)
                if prev_surface_group is None
                else (prim_surface_group, prev_surface_group)
            )
            valid = valid & _segment_visibility_mask(
                prev_point,
                hit_p,
                scene,
                ignore_surface_group_idx=ignore_surface_group_idx,
            )
            if slot == 0:
                first_hit = hit_p
            last_hit = hit_p
            prev_point = hit_p
            prev_surface_group = prim_surface_group

        last_ignore_groups = list(target_adjacent_surface_groups)
        if prev_surface_group is not None:
            last_ignore_groups.append(prev_surface_group)
        valid = valid & _segment_visibility_mask(
            prev_point,
            target_pos,
            scene,
            ignore_surface_group_idx=tuple(last_ignore_groups),
        )
        chain_vector = vector_select(valid, chain_vector, vector_zero(width))
        return valid, chain_vector, {
            "tx_pos": tx_pos,
            "first_hit": first_hit,
            "last_hit": last_hit,
        }

    hit_points = []
    normals = []
    prim_indices = []
    valid = geom_valid
    for slot in range(chain_depth):
        slot_idx = slot_base + wt.UInt32(slot * width)
        hit_p = wt.Point3f(
            dr.gather(wt.Float, hit_x_flat, slot_idx),
            dr.gather(wt.Float, hit_y_flat, slot_idx),
            dr.gather(wt.Float, hit_z_flat, slot_idx),
        )
        prim_idx = dr.gather(wt.Int32, paths.prim_idx(slot), path_idx)
        geom_n = Gather.vector3(paths.plane_normal(slot), path_idx)
        valid_prim = prim_idx >= 0
        if int(tri_surface_data["max_group_size"]) == 2:
            safe_idx = wt.UInt32(dr.select(valid_prim, prim_idx, wt.Int32(0)))
            group_members = tri_surface_data["group_members"]
            flat_idx0 = safe_idx * wt.UInt32(2)
            flat_idx1 = flat_idx0 + wt.UInt32(1)
            member_idx0_i32 = dr.gather(wt.Int32, group_members, flat_idx0)
            member_idx1_i32 = dr.gather(wt.Int32, group_members, flat_idx1)
            active0 = valid_prim & (member_idx0_i32 >= 0)
            active1 = valid_prim & (member_idx1_i32 >= 0)
            safe_member_idx0 = wt.UInt32(dr.select(active0, member_idx0_i32, wt.Int32(0)))
            safe_member_idx1 = wt.UInt32(dr.select(active1, member_idx1_i32, wt.Int32(0)))
            hit0 = active0 & point_in_triangle_3d(
                hit_p,
                dr.gather(wt.Point3f, tri_data["v0"], safe_member_idx0),
                dr.gather(wt.Point3f, tri_data["v1"], safe_member_idx0),
                dr.gather(wt.Point3f, tri_data["v2"], safe_member_idx0),
            )
            hit1 = active1 & point_in_triangle_3d(
                hit_p,
                dr.gather(wt.Point3f, tri_data["v0"], safe_member_idx1),
                dr.gather(wt.Point3f, tri_data["v1"], safe_member_idx1),
                dr.gather(wt.Point3f, tri_data["v2"], safe_member_idx1),
            )
            surface_hit = hit0 | hit1
        else:
            surface_hit = surface_contains_point(
                hit_p,
                prim_idx,
                tri_data["v0"],
                tri_data["v1"],
                tri_data["v2"],
                tri_surface_data,
            )
        valid = valid & (prim_idx >= 0) & surface_hit
        hit_points.append(hit_p)
        normals.append(geom_n)
        prim_indices.append(prim_idx)

    prev_point = tx_pos
    prev_surface_group = None
    for slot, hit_p in enumerate(hit_points):
        ignore_groups = [_triangle_surface_group_id(tri_data, prim_indices[slot])]
        if prev_surface_group is not None:
            ignore_groups.append(prev_surface_group)
        valid = valid & _segment_visibility_mask(
            prev_point,
            hit_p,
            scene,
            ignore_surface_group_idx=tuple(ignore_groups),
        )
        prev_point = hit_p
        prev_surface_group = ignore_groups[0]

    last_ignore = list(target_adjacent_surface_groups)
    if prev_surface_group is not None:
        last_ignore.append(prev_surface_group)
    valid = valid & _segment_visibility_mask(
        prev_point,
        target_pos,
        scene,
        ignore_surface_group_idx=tuple(last_ignore),
    )
    chain_vector = vector_select(valid, chain_vector, vector_zero(width))

    if return_geometry:
        return valid, chain_vector, {
            "tx_pos": tx_pos,
            "hit_points": hit_points,
            "normals": normals,
            "prim_indices": prim_indices,
        }
    return valid, chain_vector


def _epc_reflection_chain_to_target_native_common(
    *,
    paths,
    path_idx,
    target_pos,
    scene,
    target_adjacent_faces=(),
    reflection_detail,
    wavelength,
    tx_polarization=(1.0, 0.0, 0.0),
    return_geometry: bool = False,
    return_endpoints: bool = False,
    use_custom_op: bool = False,
    epc_descriptor: PreparedReflectionEpcDescriptor | None = None,
):
    width = dr.width(target_pos.x)
    chain_depth = int(paths.chain_depth)
    n_paths = int(paths.n_paths)
    if chain_depth <= 0 or n_paths <= 0:
        if return_geometry:
            return (*_empty_epc(width), _empty_epc_geometry(width, max(0, chain_depth)))
        if return_endpoints:
            return (*_empty_epc(width), _empty_epc_endpoints(width))
        return _empty_epc(width)

    descriptor = epc_descriptor
    resolved_path_idx = path_idx
    if descriptor is None:
        descriptor = build_reflection_epc_descriptor(
            paths=paths,
            path_idx=path_idx,
            scene=scene,
            reflection_detail=reflection_detail,
        )
        path_idx = dr.arange(wt.UInt32, width)
        resolved_path_idx = dr.gather(wt.UInt32, descriptor.source_path_idx, path_idx)
    else:
        resolved_path_idx = dr.gather(wt.UInt32, descriptor.source_path_idx, wt.UInt32(path_idx))
    descriptor_n_paths = int(descriptor.n_paths)
    if descriptor_n_paths <= 0:
        if return_geometry:
            return (*_empty_epc(width), _empty_epc_geometry(width, max(0, chain_depth)))
        if return_endpoints:
            return (*_empty_epc(width), _empty_epc_endpoints(width))
        return _empty_epc(width)

    n_pairs = dr.width(target_pos.x)

    if use_custom_op:
        raw_outputs = _epc_reflection_targets_custom_op(
            path_idx=path_idx,
            image_source=descriptor.image_source,
            slot_plane_point=descriptor.slot_plane_point,
            slot_plane_normal=descriptor.slot_plane_normal,
            slot_eta_r=descriptor.slot_eta_r,
            slot_sigma=descriptor.slot_sigma,
            slot_gain=descriptor.slot_gain,
            target_pos=target_pos,
            tx_polarization=tx_polarization,
            n_pairs=n_pairs,
            n_paths=descriptor_n_paths,
            chain_depth=chain_depth,
            wavelength=wavelength,
        )
    else:
        raw_outputs = _launch_reflection_epc_targets_forward(
            path_idx=path_idx,
            image_source=descriptor.image_source,
            slot_plane_point=descriptor.slot_plane_point,
            slot_plane_normal=descriptor.slot_plane_normal,
            slot_eta_r=descriptor.slot_eta_r,
            slot_sigma=descriptor.slot_sigma,
            slot_gain=descriptor.slot_gain,
            target_pos=target_pos,
            tx_polarization=tx_polarization,
            n_pairs=n_pairs,
            n_paths=descriptor_n_paths,
            chain_depth=chain_depth,
            wavelength=wavelength,
        )

    return _finalize_epc_outputs(
        raw_outputs=raw_outputs,
        paths=paths,
        path_idx=resolved_path_idx,
        target_pos=target_pos,
        scene=scene,
        target_adjacent_faces=target_adjacent_faces,
        chain_depth=chain_depth,
        return_geometry=return_geometry,
        return_endpoints=return_endpoints,
    )


def _empty_epc(width: int):
    return dr.full(wt.Bool, False, width), vector_zero(width)


def _empty_epc_geometry(width: int, chain_depth: int):
    zero_point = wt.Point3f(
        dr.zeros(wt.Float, width),
        dr.zeros(wt.Float, width),
        dr.zeros(wt.Float, width),
    )
    zero_normal = wt.Vector3f(
        dr.zeros(wt.Float, width),
        dr.zeros(wt.Float, width),
        dr.zeros(wt.Float, width),
    )
    return {
        "tx_pos": zero_point,
        "hit_points": [zero_point for _ in range(chain_depth)],
        "normals": [zero_normal for _ in range(chain_depth)],
        "prim_indices": [dr.full(wt.Int32, -1, width) for _ in range(chain_depth)],
    }


def _empty_epc_endpoints(width: int):
    zero_point = wt.Point3f(
        dr.zeros(wt.Float, width),
        dr.zeros(wt.Float, width),
        dr.zeros(wt.Float, width),
    )
    return {
        "tx_pos": zero_point,
        "first_hit": zero_point,
        "last_hit": zero_point,
    }


def _plane_segment_surface_intersection(
    current_source,
    current_target,
    plane_point,
    plane_normal,
    prim_idx_i32,
    scene,
):
    tri_data = None if scene is None else scene.tri_data_gpu
    if tri_data is None:
        width = dr.width(current_target.x)
        zero_point = wt.Point3f(
            dr.zeros(wt.Float, width),
            dr.zeros(wt.Float, width),
            dr.zeros(wt.Float, width),
        )
        return dr.full(wt.Bool, False, width), zero_point

    tri_surface_data = {
        "group_size": tri_data["surface_group_size"],
        "group_members": tri_data["surface_group_members"],
        "max_group_size": int(tri_data["surface_max_group_size"]),
    }
    valid_prim = prim_idx_i32 >= 0
    segment = current_target - current_source
    denom = dr.dot(segment, plane_normal)
    valid_denom = dr.abs(denom) > EPS
    t_hit = dr.dot(plane_point - current_source, plane_normal) / (denom + EPS)
    hit_p = current_source + t_hit * segment
    if int(tri_surface_data["max_group_size"]) == 2:
        safe_idx = wt.UInt32(dr.select(valid_prim, prim_idx_i32, wt.Int32(0)))
        group_members = tri_surface_data["group_members"]
        flat_idx0 = safe_idx * wt.UInt32(2)
        flat_idx1 = flat_idx0 + wt.UInt32(1)
        member_idx0_i32 = dr.gather(wt.Int32, group_members, flat_idx0)
        member_idx1_i32 = dr.gather(wt.Int32, group_members, flat_idx1)
        active0 = valid_prim & (member_idx0_i32 >= 0)
        active1 = valid_prim & (member_idx1_i32 >= 0)
        safe_member_idx0 = wt.UInt32(dr.select(active0, member_idx0_i32, wt.Int32(0)))
        safe_member_idx1 = wt.UInt32(dr.select(active1, member_idx1_i32, wt.Int32(0)))
        hit0 = active0 & point_in_triangle_3d(
            hit_p,
            dr.gather(wt.Point3f, tri_data["v0"], safe_member_idx0),
            dr.gather(wt.Point3f, tri_data["v1"], safe_member_idx0),
            dr.gather(wt.Point3f, tri_data["v2"], safe_member_idx0),
        )
        hit1 = active1 & point_in_triangle_3d(
            hit_p,
            dr.gather(wt.Point3f, tri_data["v0"], safe_member_idx1),
            dr.gather(wt.Point3f, tri_data["v1"], safe_member_idx1),
            dr.gather(wt.Point3f, tri_data["v2"], safe_member_idx1),
        )
        surface_hit = hit0 | hit1
    else:
        surface_hit = surface_contains_point(
            hit_p,
            prim_idx_i32,
            tri_data["v0"],
            tri_data["v1"],
            tri_data["v2"],
            tri_surface_data,
        )
    valid = valid_prim & valid_denom & (t_hit > EPS) & (t_hit < (1.0 - EPS)) & surface_hit
    return valid, hit_p


def _array_has_grad(value) -> bool:
    try:
        return bool(dr.grad_enabled(value))
    except TypeError:
        return False


def _point_has_grad(value) -> bool:
    return _array_has_grad(value.x) or _array_has_grad(value.y) or _array_has_grad(value.z)


def _native_epc_eligible(paths, target_pos, chain_depth: int) -> bool:
    if not _point_has_grad(target_pos):
        if _point_has_grad(paths.image_source):
            return False
        for slot in range(chain_depth):
            if _point_has_grad(paths.plane_point(slot)):
                return False
            if _point_has_grad(paths.plane_normal(slot)):
                return False
        return True
    return False


def _native_epc_requires_ad(paths, target_pos, chain_depth: int) -> bool:
    if _point_has_grad(target_pos) or _point_has_grad(paths.image_source):
        return True
    for slot in range(chain_depth):
        if _point_has_grad(paths.plane_point(slot)):
            return True
        if _point_has_grad(paths.plane_normal(slot)):
            return True
    return False


def _epc_reflection_chain_to_target_native(
    *,
    paths,
    path_idx,
    target_pos,
    scene,
    target_adjacent_faces=(),
    reflection_detail,
    wavelength,
    tx_polarization=(1.0, 0.0, 0.0),
    return_geometry: bool = False,
    return_endpoints: bool = False,
    epc_descriptor: PreparedReflectionEpcDescriptor | None = None,
):
    return _epc_reflection_chain_to_target_native_common(
        paths=paths,
        path_idx=path_idx,
        target_pos=target_pos,
        scene=scene,
        target_adjacent_faces=target_adjacent_faces,
        reflection_detail=reflection_detail,
        wavelength=wavelength,
        tx_polarization=tx_polarization,
        return_geometry=return_geometry,
        return_endpoints=return_endpoints,
        use_custom_op=False,
        epc_descriptor=epc_descriptor,
    )


def _epc_reflection_chain_to_target_native_ad(
    *,
    paths,
    path_idx,
    target_pos,
    scene,
    target_adjacent_faces=(),
    reflection_detail,
    wavelength,
    tx_polarization=(1.0, 0.0, 0.0),
    return_geometry: bool = False,
    return_endpoints: bool = False,
    epc_descriptor: PreparedReflectionEpcDescriptor | None = None,
):
    return _epc_reflection_chain_to_target_native_common(
        paths=paths,
        path_idx=path_idx,
        target_pos=target_pos,
        scene=scene,
        target_adjacent_faces=target_adjacent_faces,
        reflection_detail=reflection_detail,
        wavelength=wavelength,
        tx_polarization=tx_polarization,
        return_geometry=return_geometry,
        return_endpoints=return_endpoints,
        use_custom_op=True,
        epc_descriptor=epc_descriptor,
    )


def _epc_reflection_chain_to_target_python(
    *,
    paths,
    path_idx,
    target_pos,
    scene,
    target_adjacent_faces=(),
    reflection_detail,
    wavelength,
    tx_polarization=(1.0, 0.0, 0.0),
    return_geometry: bool = False,
    return_endpoints: bool = False,
    epc_descriptor: PreparedReflectionEpcDescriptor | None = None,
):
    width = dr.width(target_pos.x)
    if scene is None or scene.tri_data_gpu is None or paths is None:
        if return_geometry:
            return (*_empty_epc(width), _empty_epc_geometry(width, 0))
        if return_endpoints:
            return (*_empty_epc(width), _empty_epc_endpoints(width))
        return _empty_epc(width)
    chain_depth = int(paths.chain_depth)
    n_paths = int(paths.n_paths)
    if chain_depth <= 0 or n_paths <= 0:
        if return_geometry:
            return (*_empty_epc(width), _empty_epc_geometry(width, max(0, chain_depth)))
        if return_endpoints:
            return (*_empty_epc(width), _empty_epc_endpoints(width))
        return _empty_epc(width)
    detail = coerce_reflection_trace_detail(reflection_detail)
    descriptor = epc_descriptor
    descriptor_path_idx = wt.UInt32(path_idx)
    descriptor_n_paths = int(0 if descriptor is None else descriptor.n_paths)
    resolved_path_idx = path_idx
    if descriptor is None:
        image_source = Gather.point3(paths.image_source, path_idx)
    else:
        if descriptor_n_paths <= 0:
            if return_geometry:
                return (*_empty_epc(width), _empty_epc_geometry(width, max(0, chain_depth)))
            if return_endpoints:
                return (*_empty_epc(width), _empty_epc_endpoints(width))
            return _empty_epc(width)
        resolved_path_idx = dr.gather(wt.UInt32, descriptor.source_path_idx, descriptor_path_idx)
        image_source = wt.Point3f(
            dr.gather(wt.Float, descriptor.image_source.x, descriptor_path_idx),
            dr.gather(wt.Float, descriptor.image_source.y, descriptor_path_idx),
            dr.gather(wt.Float, descriptor.image_source.z, descriptor_path_idx),
        )
    current_source = image_source
    current_target = target_pos
    hit_points_rev = []
    normals_rev = []
    prim_idx_rev = []
    valid = dr.full(wt.Bool, True, width)

    for slot in range(chain_depth - 1, -1, -1):
        prim_idx = dr.gather(wt.Int32, paths.prim_idx(slot), resolved_path_idx)
        if descriptor is None:
            plane_point = Gather.point3(paths.plane_point(slot), path_idx)
            plane_normal = Gather.vector3(paths.plane_normal(slot), path_idx)
        else:
            slot_base = descriptor_path_idx + wt.UInt32(slot * descriptor_n_paths)
            plane_point = wt.Point3f(
                dr.gather(wt.Float, descriptor.slot_plane_point.x, slot_base),
                dr.gather(wt.Float, descriptor.slot_plane_point.y, slot_base),
                dr.gather(wt.Float, descriptor.slot_plane_point.z, slot_base),
            )
            plane_normal = wt.Vector3f(
                dr.gather(wt.Float, descriptor.slot_plane_normal.x, slot_base),
                dr.gather(wt.Float, descriptor.slot_plane_normal.y, slot_base),
                dr.gather(wt.Float, descriptor.slot_plane_normal.z, slot_base),
            )
        segment_valid, hit_p = _plane_segment_surface_intersection(
            current_source,
            current_target,
            plane_point,
            plane_normal,
            prim_idx,
            scene,
        )
        valid = valid & segment_valid
        hit_points_rev.append(hit_p)
        normals_rev.append(plane_normal)
        prim_idx_rev.append(prim_idx)
        current_target = hit_p
        current_source = reflect_point_across_plane(current_source, plane_point, plane_normal)

    tx_pos = current_source
    hit_points = list(reversed(hit_points_rev))
    normals = list(reversed(normals_rev))
    prim_indices = list(reversed(prim_idx_rev))

    prev_point = tx_pos
    for slot, hit_p in enumerate(hit_points):
        ignore_list = [prim_indices[slot]]
        if slot > 0:
            ignore_list.append(prim_indices[slot - 1])
        valid = valid & _segment_visibility_mask(prev_point, hit_p, scene, ignore_prim_idx=tuple(ignore_list))
        prev_point = hit_p

    last_ignore = list(target_adjacent_faces)
    if prim_indices:
        last_ignore.append(prim_indices[-1])
    valid = valid & _segment_visibility_mask(prev_point, target_pos, scene, ignore_prim_idx=tuple(last_ignore))

    first_segment_dir = (hit_points[0] - tx_pos) / (dr.norm(hit_points[0] - tx_pos) + EPS)
    chain_vector = vector_from_scalar_and_real_direction(
        wt.Complex2f(dr.ones(wt.Float, width), dr.zeros(wt.Float, width)),
        project_real_polarization_to_ray(tx_polarization, first_segment_dir),
    )

    prev_point = tx_pos
    reflection_gain = detail.reflection_gain
    material_omega = reflection_material_omega(wavelength)
    reflection_material = detail.reflection_material
    use_scene_materials = detail.use_scene_materials
    for slot, (hit_p, geom_n, prim_idx) in enumerate(zip(hit_points, normals, prim_indices)):
        incoming = hit_p - prev_point
        incoming = incoming / (dr.norm(incoming) + EPS)
        if descriptor is None:
            material_inputs = resolve_surface_material(
                scene=scene,
                prim_idx=prim_idx,
                override_material=reflection_material,
                reflection_coef=reflection_gain,
                default_eta_r=5.0,
                default_sigma=0.0,
                valid_mask=valid,
                use_scene_materials=use_scene_materials,
            )
        else:
            slot_base = descriptor_path_idx + wt.UInt32(slot * descriptor_n_paths)
            material_inputs = {
                "eta_r": dr.gather(wt.Float, descriptor.slot_eta_r, slot_base),
                "sigma": dr.gather(wt.Float, descriptor.slot_sigma, slot_base),
                "gain": dr.gather(wt.Float, descriptor.slot_gain, slot_base),
            }
        chain_vector = reflect_field_vector(
            chain_vector,
            incoming,
            geom_n,
            eta_r=material_inputs["eta_r"],
            sigma=material_inputs["sigma"],
            omega=material_omega,
            gain=material_inputs["gain"],
        )
        prev_point = hit_p

    chain_vector = vector_select(valid, chain_vector, vector_zero(width))
    if return_geometry:
        return valid, chain_vector, {
            "tx_pos": tx_pos,
            "hit_points": hit_points,
            "normals": normals,
            "prim_indices": prim_indices,
        }
    if return_endpoints:
        return valid, chain_vector, {
            "tx_pos": tx_pos,
            "first_hit": hit_points[0],
            "last_hit": hit_points[-1],
        }
    return valid, chain_vector


def epc_reflection_chain_to_target(
    *,
    paths,
    path_idx,
    target_pos,
    scene,
    target_adjacent_faces=(),
    reflection_detail,
    wavelength,
    tx_polarization=(1.0, 0.0, 0.0),
    return_geometry: bool = False,
    return_endpoints: bool = False,
    epc_descriptor: PreparedReflectionEpcDescriptor | None = None,
):
    width = dr.width(target_pos.x)
    if scene is None or scene.tri_data_gpu is None or paths is None:
        if return_geometry:
            return (*_empty_epc(width), _empty_epc_geometry(width, 0))
        if return_endpoints:
            return (*_empty_epc(width), _empty_epc_endpoints(width))
        return _empty_epc(width)

    chain_depth = int(paths.chain_depth)
    n_paths = int(paths.n_paths)
    if chain_depth <= 0 or n_paths <= 0:
        if return_geometry:
            return (*_empty_epc(width), _empty_epc_geometry(width, max(0, chain_depth)))
        if return_endpoints:
            return (*_empty_epc(width), _empty_epc_endpoints(width))
        return _empty_epc(width)

    use_native = False
    use_native_ad = False
    try:
        from ..._native import native_extension_available

        if native_extension_available():
            use_native = _native_epc_eligible(paths, target_pos, chain_depth)
            use_native_ad = (not use_native) and _native_epc_requires_ad(paths, target_pos, chain_depth)
    except Exception:
        use_native = False
        use_native_ad = False

    if use_native:
        return _epc_reflection_chain_to_target_native(
            paths=paths,
            path_idx=path_idx,
            target_pos=target_pos,
            scene=scene,
            target_adjacent_faces=target_adjacent_faces,
            reflection_detail=reflection_detail,
            wavelength=wavelength,
            tx_polarization=tx_polarization,
            return_geometry=return_geometry,
            return_endpoints=return_endpoints,
            epc_descriptor=epc_descriptor,
        )
    if use_native_ad:
        # The native AD replay path still diverges in multipath diagnostics.
        # Keep AD-sensitive replay on the Dr.Jit reference implementation.
        return _epc_reflection_chain_to_target_python(
            paths=paths,
            path_idx=path_idx,
            target_pos=target_pos,
            scene=scene,
            target_adjacent_faces=target_adjacent_faces,
            reflection_detail=reflection_detail,
            wavelength=wavelength,
            tx_polarization=tx_polarization,
            return_geometry=return_geometry,
            return_endpoints=return_endpoints,
            epc_descriptor=epc_descriptor,
        )
    return _epc_reflection_chain_to_target_python(
        paths=paths,
        path_idx=path_idx,
        target_pos=target_pos,
        scene=scene,
        target_adjacent_faces=target_adjacent_faces,
        reflection_detail=reflection_detail,
        wavelength=wavelength,
        tx_polarization=tx_polarization,
        return_geometry=return_geometry,
        return_endpoints=return_endpoints,
        epc_descriptor=epc_descriptor,
    )


__all__ = [
    "PreparedReflectionEpcDescriptor",
    "build_reflection_epc_descriptor",
    "epc_reflection_chain_to_target",
]

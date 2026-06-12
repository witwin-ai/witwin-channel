"""Exact path calculation (EPC) for reflection chains.

The EPC primitives here image-source-fold a chain of planar reflections so
that a ray from a virtual source reaches each receiver via the requested
sequence of triangles. The dispatcher requires the native CUDA EPC path; AD
inputs use the native forward CustomOp and its registered tangent rule.
"""

from __future__ import annotations

from dataclasses import dataclass

import drjit as dr
import rayd
from witwin.channel.deterministic import types as wt

from witwin.channel.core.runtime import (
    Tx,
    Wave,
    point_grad_enabled,
    scene_geometry_grad_enabled,
)
from witwin.channel.core.physics.materials import resolve_surface_material
from witwin.channel.core.physics.wave_math import material_angular_frequency, unit_phase_neg_kd
from witwin.channel.core.numerics.constants import EPS, RAY_ORIGIN_BIAS
from witwin.channel.core.numerics.arrays import (
    broadcast,
    gather,
    scalar,
    zeros_point3,
    zeros_vector3,
)
from witwin.channel.core.geometry import (
    point_in_triangle_3d,
    reflect_point_across_plane,
    surface_contains_point,
)
from witwin.channel.core.physics.polarization import (
    project_real_polarization_to_ray,
    reflect_field_vector,
    vector_add,
    vector_scale,
    vector_from_scalar,
    vector_select,
    vector_zero,
)
from .boundary import nearest_surface_boundary_edge
from .detail import coerce_trace_detail
from .f_weight import reflection_segment_attenuation, reflection_transition_weights
from .secondary_visibility import nearest_blocker_silhouette_edge


RAYD_REFLECTION_EPC_MAX_BOUNCES = 8
_RAYD_EPC_FINE_GEOMETRY_OPTIONS: bool | None = None
_RAYD_EPC_FIELD_PIPELINE_AVAILABLE: bool | None = None


# ---------- empty/zero return helpers --------------------------------------

def empty_geometry(width: int, chain_depth: int):
    zp = zeros_point3(width)
    zn = zeros_vector3(width)
    return {
        "tx_pos": zp,
        "hit_points": [zp for _ in range(chain_depth)],
        "normals": [zn for _ in range(chain_depth)],
        "prim_indices": [dr.full(wt.Int32, -1, width) for _ in range(chain_depth)],
        "primary_sides": [dr.full(wt.Bool, False, width) for _ in range(chain_depth)],
    }


def empty_endpoints(width: int):
    zp = zeros_point3(width)
    return {"tx_pos": zp, "first_hit": zp, "last_hit": zp}


def empty_return(width, chain_depth, return_geometry, return_endpoints):
    valid = dr.full(wt.Bool, False, width)
    chain_vector = vector_zero(width)
    if return_geometry:
        return valid, chain_vector, empty_geometry(width, max(0, chain_depth))
    if return_endpoints:
        return valid, chain_vector, empty_endpoints(width)
    return valid, chain_vector


# ---------- detach / grad plumbing -----------------------------------------

def detach_point(value):
    return wt.Point3f(dr.detach(value.x), dr.detach(value.y), dr.detach(value.z))


def detach_vector(value):
    return wt.Vector3f(dr.detach(value.x), dr.detach(value.y), dr.detach(value.z))


def set_point_grad(value, tangent):
    dr.enable_grad(value.x, value.y, value.z)
    dr.set_grad(value.x, tangent.x)
    dr.set_grad(value.y, tangent.y)
    dr.set_grad(value.z, tangent.z)


def point_from_grad(value, width: int):
    if value is None:
        return zeros_point3(width)
    return wt.Point3f(value.x, value.y, value.z)


def vector_from_grad(value, width: int):
    if value is None:
        return zeros_vector3(width)
    return wt.Vector3f(value.x, value.y, value.z)


def array_from_grad(value, width: int):
    return dr.zeros(wt.Float, width) if value is None else value


def point_has_grad(value) -> bool:
    for axis in (value.x, value.y, value.z):
        try:
            if bool(dr.grad_enabled(axis)):
                return True
        except TypeError:
            pass
    return False


def array_has_grad(value) -> bool:
    try:
        return bool(dr.grad_enabled(value))
    except TypeError:
        return False


def descriptor_material_has_grad(descriptor: Descriptor | None) -> bool:
    if descriptor is None:
        return False
    return any(
        array_has_grad(value)
        for value in (
            descriptor.slot_eta_r,
            descriptor.slot_mu_r,
            descriptor.slot_sigma,
            descriptor.slot_gain,
        )
    )


def descriptor_geometry_has_grad(descriptor: Descriptor | None) -> bool:
    if descriptor is None:
        return False
    return (
        point_has_grad(descriptor.image_source)
        or point_has_grad(descriptor.slot_plane_point)
        or point_has_grad(descriptor.slot_plane_normal)
    )


def scene_material_has_grad(scene) -> bool:
    if scene is None or not hasattr(scene, "_triangle_runtime"):
        return False
    tri_data = scene._triangle_runtime()
    if tri_data is None:
        return False
    return any(
        array_has_grad(tri_data.get(key))
        for key in (
            "material_eps_r",
            "material_mu_r",
            "material_sigma_e",
        )
    )


def native_epc_eligible(
    paths,
    target_pos,
    chain_depth: int,
    *,
    scene=None,
    descriptor: Descriptor | None = None,
) -> bool:
    if point_has_grad(target_pos) or point_has_grad(paths.image_source):
        return False
    for slot in range(chain_depth):
        if point_has_grad(paths.plane_point(slot)) or point_has_grad(paths.plane_normal(slot)):
            return False
    if descriptor_material_has_grad(descriptor) or scene_material_has_grad(scene):
        return False
    return True


# ---------- geometric helpers ----------------------------------------------

def surface_group_ignore_tuple(scene, prim_candidates) -> tuple[wt.Int32, ...]:
    if not isinstance(prim_candidates, (tuple, list)):
        prim_candidates = (prim_candidates,)
    return tuple(
        scene.triangle_group_id(wt.Int32(p))
        for p in prim_candidates if p is not None
    )


def _surface_contains_rayd(hit_p, prim_idx_i32, scene, plane_normal, valid_prim):
    normal_norm = dr.norm(plane_normal)
    active = valid_prim & (normal_norm > wt.Float(EPS))
    normal = plane_normal / (normal_norm + wt.Float(EPS))
    bias_value = 2.0 * RAY_ORIGIN_BIAS
    bias = wt.Float(bias_value)
    expected_group = scene.triangle_group_id(prim_idx_i32)

    def trace_side(origin, direction, side_active):
        hit, _, hit_prim_u32 = scene.intersect_rays_raw_with_prim(
            origin,
            direction,
            side_active,
            tmax=wt.Float(2.0 * bias_value),
        )
        hit_prim_i32 = wt.Int32(hit_prim_u32)
        return side_active & hit & (scene.triangle_group_id(hit_prim_i32) == expected_group)

    front_hit = trace_side(hit_p + normal * bias, -normal, active)
    back_hit = trace_side(hit_p - normal * bias, normal, active & ~front_hit)
    return front_hit | back_hit


def surface_contains(
    hit_p,
    prim_idx_i32,
    tri_data,
    tri_surface_data,
    valid_prim,
    *,
    scene=None,
    plane_normal=None,
):
    """Dispatch to the closed-surface 2-member fast path or the generic scan."""
    max_group_size = int(tri_surface_data["max_group_size"])
    if (
        max_group_size > 2
        and scene is not None
        and plane_normal is not None
        and hasattr(scene, "intersect_rays_raw_with_prim")
        and not point_has_grad(hit_p)
        and not point_has_grad(plane_normal)
        and not (hasattr(scene, "_triangle_runtime") and scene_geometry_grad_enabled(scene))
    ):
        return _surface_contains_rayd(hit_p, prim_idx_i32, scene, plane_normal, valid_prim)
    if max_group_size == 2:
        safe_idx = wt.UInt32(dr.select(valid_prim, prim_idx_i32, wt.Int32(0)))
        group_members = tri_surface_data["group_members"]
        member0 = dr.gather(wt.Int32, group_members, safe_idx * wt.UInt32(2))
        member1 = dr.gather(wt.Int32, group_members, safe_idx * wt.UInt32(2) + wt.UInt32(1))
        active0 = valid_prim & (member0 >= 0)
        active1 = valid_prim & (member1 >= 0)
        safe_m0 = wt.UInt32(dr.select(active0, member0, wt.Int32(0)))
        safe_m1 = wt.UInt32(dr.select(active1, member1, wt.Int32(0)))
        hit0 = active0 & point_in_triangle_3d(
            hit_p,
            dr.gather(wt.Point3f, tri_data["v0"], safe_m0),
            dr.gather(wt.Point3f, tri_data["v1"], safe_m0),
            dr.gather(wt.Point3f, tri_data["v2"], safe_m0),
        )
        hit1 = active1 & point_in_triangle_3d(
            hit_p,
            dr.gather(wt.Point3f, tri_data["v0"], safe_m1),
            dr.gather(wt.Point3f, tri_data["v1"], safe_m1),
            dr.gather(wt.Point3f, tri_data["v2"], safe_m1),
        )
        return hit0 | hit1
    return surface_contains_point(
        hit_p, prim_idx_i32,
        tri_data["v0"], tri_data["v1"], tri_data["v2"],
        tri_surface_data,
    )


# ---------- descriptor (cached per chain) ----------------------------------

@dataclass(frozen=True)
class Descriptor:
    source_path_idx: wt.UInt32
    image_source: wt.Point3f
    slot_plane_point: wt.Point3f
    slot_plane_normal: wt.Vector3f
    slot_eta_r: wt.Float
    slot_mu_r: wt.Float
    slot_sigma: wt.Float
    slot_gain: wt.Float
    chain_depth: int
    n_paths: int


def flatten_slot_points(slot_points):
    if not slot_points:
        return zeros_point3(0)
    return wt.Point3f(
        dr.concat([s.x for s in slot_points]),
        dr.concat([s.y for s in slot_points]),
        dr.concat([s.z for s in slot_points]),
    )


def flatten_slot_vectors(slot_vectors):
    if not slot_vectors:
        return zeros_vector3(0)
    return wt.Vector3f(
        dr.concat([s.x for s in slot_vectors]),
        dr.concat([s.y for s in slot_vectors]),
        dr.concat([s.z for s in slot_vectors]),
    )


def build_descriptor(
    *,
    paths,
    path_idx,
    scene,
    reflection_detail,
) -> Descriptor:
    chain_depth = int(0 if paths is None else paths.chain_depth)
    n_paths = int(dr.width(path_idx))
    if chain_depth <= 0 or n_paths <= 0:
        return Descriptor(
            source_path_idx=dr.zeros(wt.UInt32, 0),
            image_source=zeros_point3(0),
            slot_plane_point=zeros_point3(0),
            slot_plane_normal=zeros_vector3(0),
            slot_eta_r=dr.zeros(wt.Float, 0),
            slot_mu_r=dr.zeros(wt.Float, 0),
            slot_sigma=dr.zeros(wt.Float, 0),
            slot_gain=dr.zeros(wt.Float, 0),
            chain_depth=max(0, chain_depth),
            n_paths=0,
        )
    detail = coerce_trace_detail(reflection_detail)
    width = dr.width(path_idx)
    valid_mask = dr.full(wt.Bool, True, width)
    img = gather(paths.image_source, path_idx)
    image_source = wt.Point3f(img.x, img.y, img.z)
    slot_plane_points: list = []
    slot_plane_normals: list = []
    slot_eta_r: list = []
    slot_mu_r: list = []
    slot_sigma: list = []
    slot_gain: list = []
    for slot in range(chain_depth):
        pp = gather(paths.plane_point(slot), path_idx)
        pn = gather(paths.plane_normal(slot), path_idx)
        slot_plane_points.append(wt.Point3f(pp.x, pp.y, pp.z))
        slot_plane_normals.append(wt.Vector3f(pn.x, pn.y, pn.z))
        prim_idx = wt.Int32(dr.gather(type(paths.prim_idx(slot)), paths.prim_idx(slot), path_idx))
        material_inputs = resolve_surface_material(
            scene=scene,
            prim_idx=prim_idx,
            default_gain=detail.reflection_gain,
            valid_mask=valid_mask,
        )
        slot_eta_r.append(material_inputs.eta_r)
        slot_mu_r.append(material_inputs.mu_r)
        slot_sigma.append(material_inputs.sigma)
        slot_gain.append(material_inputs.gain)

    return Descriptor(
        source_path_idx=wt.UInt32(path_idx),
        image_source=image_source,
        slot_plane_point=flatten_slot_points(slot_plane_points),
        slot_plane_normal=flatten_slot_vectors(slot_plane_normals),
        slot_eta_r=dr.concat(slot_eta_r),
        slot_mu_r=dr.concat(slot_mu_r),
        slot_sigma=dr.concat(slot_sigma),
        slot_gain=dr.concat(slot_gain),
        chain_depth=chain_depth,
        n_paths=n_paths,
    )


# ---------- pure Dr.Jit reference math (also drives AD reference) ----------

def chain_math_reference(
    *,
    path_idx,
    image_source,
    slot_plane_point,
    slot_plane_normal,
    slot_eta_r,
    slot_mu_r,
    slot_sigma,
    slot_gain,
    target_pos,
    tx_polarization,
    chain_depth: int,
    n_paths: int,
    wavelength: float,
):
    width = dr.width(target_pos.x)
    if width == 0 or chain_depth <= 0 or int(n_paths) <= 0:
        return {
            "geom_valid": dr.zeros(wt.Float, width),
            "tx_pos": zeros_point3(width),
            "chain_vector": vector_zero(width),
            "hit_points": [],
            "hit_x": dr.zeros(wt.Float, 0),
            "hit_y": dr.zeros(wt.Float, 0),
            "hit_z": dr.zeros(wt.Float, 0),
        }

    descriptor_path_idx = wt.UInt32(path_idx)
    current_source = wt.Point3f(
        dr.gather(wt.Float, image_source.x, descriptor_path_idx),
        dr.gather(wt.Float, image_source.y, descriptor_path_idx),
        dr.gather(wt.Float, image_source.z, descriptor_path_idx),
    )
    current_target = target_pos
    hit_points_rev: list = []
    normals_rev: list = []
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
    chain_vector = vector_from_scalar(
        wt.Complex2f(dr.ones(wt.Float, width), dr.zeros(wt.Float, width)),
        project_real_polarization_to_ray(tx_polarization, first_segment_dir),
    )
    material_omega = material_angular_frequency(wavelength)
    prev_point = tx_pos
    for slot, (hit_p, geom_n) in enumerate(zip(hit_points, normals)):
        base = descriptor_path_idx + wt.UInt32(slot * int(n_paths))
        eta_r = dr.gather(wt.Float, slot_eta_r, base)
        mu_r = dr.gather(wt.Float, slot_mu_r, base)
        sigma = dr.gather(wt.Float, slot_sigma, base)
        gain = dr.gather(wt.Float, slot_gain, base)
        incoming = (hit_p - prev_point) / (dr.norm(hit_p - prev_point) + EPS)
        chain_vector = reflect_field_vector(
            chain_vector, incoming, geom_n,
            eta_r=eta_r, sigma=sigma, omega=material_omega, gain=gain, mu_r=mu_r,
        )
        prev_point = hit_p

    return {
        "geom_valid": wt.Float(geom_valid),
        "tx_pos": tx_pos,
        "chain_vector": chain_vector,
        "hit_points": hit_points,
        "hit_x": dr.concat([hp.x for hp in hit_points]),
        "hit_y": dr.concat([hp.y for hp in hit_points]),
        "hit_z": dr.concat([hp.z for hp in hit_points]),
    }


def reference_outputs_tuple(outputs):
    return (
        outputs["geom_valid"],
        outputs["tx_pos"].x, outputs["tx_pos"].y, outputs["tx_pos"].z,
        outputs["chain_vector"]["x"].real, outputs["chain_vector"]["x"].imag,
        outputs["chain_vector"]["y"].real, outputs["chain_vector"]["y"].imag,
        outputs["chain_vector"]["z"].real, outputs["chain_vector"]["z"].imag,
        outputs["hit_x"], outputs["hit_y"], outputs["hit_z"],
    )


def reference_forward_grads(
    *,
    path_idx, image_source, slot_plane_point, slot_plane_normal,
    slot_eta_r, slot_mu_r, slot_sigma, slot_gain, target_pos,
    tx_polarization, chain_depth, n_paths, wavelength,
    tangent_image_source, tangent_slot_plane_point,
    tangent_slot_plane_normal, tangent_slot_eta_r,
    tangent_slot_mu_r, tangent_slot_sigma, tangent_slot_gain, tangent_target_pos,
):
    image_source_ad = detach_point(image_source)
    slot_plane_point_ad = detach_point(slot_plane_point)
    slot_plane_normal_ad = detach_vector(slot_plane_normal)
    slot_eta_r_ad = dr.detach(slot_eta_r)
    slot_mu_r_ad = dr.detach(slot_mu_r)
    slot_sigma_ad = dr.detach(slot_sigma)
    slot_gain_ad = dr.detach(slot_gain)
    target_pos_ad = detach_point(target_pos)

    dr.enable_grad(slot_eta_r_ad, slot_mu_r_ad, slot_sigma_ad, slot_gain_ad)
    set_point_grad(image_source_ad, tangent_image_source)
    set_point_grad(slot_plane_point_ad, tangent_slot_plane_point)
    set_point_grad(slot_plane_normal_ad, tangent_slot_plane_normal)
    dr.set_grad(slot_eta_r_ad, tangent_slot_eta_r)
    dr.set_grad(slot_mu_r_ad, tangent_slot_mu_r)
    dr.set_grad(slot_sigma_ad, tangent_slot_sigma)
    dr.set_grad(slot_gain_ad, tangent_slot_gain)
    set_point_grad(target_pos_ad, tangent_target_pos)

    outputs = chain_math_reference(
        path_idx=path_idx,
        image_source=image_source_ad,
        slot_plane_point=slot_plane_point_ad,
        slot_plane_normal=slot_plane_normal_ad,
        slot_eta_r=slot_eta_r_ad,
        slot_mu_r=slot_mu_r_ad,
        slot_sigma=slot_sigma_ad,
        slot_gain=slot_gain_ad,
        target_pos=target_pos_ad,
        tx_polarization=tx_polarization,
        chain_depth=chain_depth,
        n_paths=n_paths,
        wavelength=wavelength,
    )
    differentiable = reference_outputs_tuple(outputs)
    dr.forward_to(differentiable)
    return tuple(dr.grad(v) for v in differentiable)


def reference_backward_grads(
    *,
    path_idx, image_source, slot_plane_point, slot_plane_normal,
    slot_eta_r, slot_mu_r, slot_sigma, slot_gain, target_pos,
    tx_polarization, chain_depth, n_paths, wavelength, grad_outputs,
):
    image_source_ad = detach_point(image_source)
    slot_plane_point_ad = detach_point(slot_plane_point)
    slot_plane_normal_ad = detach_vector(slot_plane_normal)
    slot_eta_r_ad = dr.detach(slot_eta_r)
    slot_mu_r_ad = dr.detach(slot_mu_r)
    slot_sigma_ad = dr.detach(slot_sigma)
    slot_gain_ad = dr.detach(slot_gain)
    target_pos_ad = detach_point(target_pos)
    dr.enable_grad(
        image_source_ad.x, image_source_ad.y, image_source_ad.z,
        slot_plane_point_ad.x, slot_plane_point_ad.y, slot_plane_point_ad.z,
        slot_plane_normal_ad.x, slot_plane_normal_ad.y, slot_plane_normal_ad.z,
        slot_eta_r_ad, slot_mu_r_ad, slot_sigma_ad, slot_gain_ad,
        target_pos_ad.x, target_pos_ad.y, target_pos_ad.z,
    )
    outputs = chain_math_reference(
        path_idx=path_idx,
        image_source=image_source_ad,
        slot_plane_point=slot_plane_point_ad,
        slot_plane_normal=slot_plane_normal_ad,
        slot_eta_r=slot_eta_r_ad,
        slot_mu_r=slot_mu_r_ad,
        slot_sigma=slot_sigma_ad,
        slot_gain=slot_gain_ad,
        target_pos=target_pos_ad,
        tx_polarization=tx_polarization,
        chain_depth=chain_depth,
        n_paths=n_paths,
        wavelength=wavelength,
    )
    loss = dr.zeros(wt.Float, 1)
    differentiable = reference_outputs_tuple(outputs)
    for output, grad_output in zip(differentiable, grad_outputs):
        if grad_output is None:
            continue
        loss += dr.sum(output * grad_output)
    dr.backward(loss)
    return {
        "image_source": wt.Point3f(
            dr.grad(image_source_ad.x), dr.grad(image_source_ad.y), dr.grad(image_source_ad.z),
        ),
        "slot_plane_point": wt.Point3f(
            dr.grad(slot_plane_point_ad.x), dr.grad(slot_plane_point_ad.y), dr.grad(slot_plane_point_ad.z),
        ),
        "slot_plane_normal": wt.Vector3f(
            dr.grad(slot_plane_normal_ad.x), dr.grad(slot_plane_normal_ad.y), dr.grad(slot_plane_normal_ad.z),
        ),
        "slot_eta_r": dr.grad(slot_eta_r_ad),
        "slot_mu_r": dr.grad(slot_mu_r_ad),
        "slot_sigma": dr.grad(slot_sigma_ad),
        "slot_gain": dr.grad(slot_gain_ad),
        "target_pos": wt.Point3f(
            dr.grad(target_pos_ad.x), dr.grad(target_pos_ad.y), dr.grad(target_pos_ad.z),
        ),
    }


# ---------- native CUDA forward + CustomOp wrapper -------------------------

def launch_native_forward(
    *,
    path_idx, image_source, slot_plane_point, slot_plane_normal,
    slot_eta_r, slot_mu_r, slot_sigma, slot_gain, target_pos, tx_polarization,
    n_pairs, n_paths, chain_depth, wavelength,
):
    from witwin.channel._native.deterministic import NativeExtension

    if hasattr(tx_polarization, "x"):
        pol_x, pol_y, pol_z = tx_polarization.x, tx_polarization.y, tx_polarization.z
    else:
        pol_x, pol_y, pol_z = tx_polarization[0], tx_polarization[1], tx_polarization[2]
    raw_outputs = NativeExtension.load().reflection_epc_targets_forward_arrays(
        wt.Int32(path_idx),
        image_source.x, image_source.y, image_source.z,
        slot_plane_point.x, slot_plane_point.y, slot_plane_point.z,
        slot_plane_normal.x, slot_plane_normal.y, slot_plane_normal.z,
        slot_eta_r, slot_mu_r, slot_sigma, slot_gain,
        target_pos.x, target_pos.y, target_pos.z,
        float(scalar(pol_x)), float(scalar(pol_y)), float(scalar(pol_z)),
        int(n_pairs), int(n_paths), int(chain_depth),
        float(2.0 * dr.pi / wavelength),
        float(material_angular_frequency(wavelength)[0]),
    )
    (geom_valid_i32, tx_pos_x, tx_pos_y, tx_pos_z,
     vec_x_re, vec_x_im, vec_y_re, vec_y_im, vec_z_re, vec_z_im,
     hit_x, hit_y, hit_z) = raw_outputs
    return (
        wt.Float(geom_valid_i32),
        tx_pos_x, tx_pos_y, tx_pos_z,
        vec_x_re, vec_x_im, vec_y_re, vec_y_im, vec_z_re, vec_z_im,
        hit_x, hit_y, hit_z,
    )


class EpcTargetsOp(dr.CustomOp):
    def eval(
        self,
        path_idx, image_source, slot_plane_point, slot_plane_normal,
        slot_eta_r, slot_mu_r, slot_sigma, slot_gain, target_pos,
        *,
        tx_polarization, n_pairs, n_paths, chain_depth, wavelength,
    ):
        self.path_idx = path_idx
        self.image_source = image_source
        self.slot_plane_point = slot_plane_point
        self.slot_plane_normal = slot_plane_normal
        self.slot_eta_r = slot_eta_r
        self.slot_mu_r = slot_mu_r
        self.slot_sigma = slot_sigma
        self.slot_gain = slot_gain
        self.target_pos = target_pos
        self.tx_polarization = tx_polarization
        self.n_pairs = int(n_pairs)
        self.n_paths = int(n_paths)
        self.chain_depth = int(chain_depth)
        self.wavelength = float(wavelength)
        return launch_native_forward(
            path_idx=path_idx,
            image_source=image_source,
            slot_plane_point=slot_plane_point,
            slot_plane_normal=slot_plane_normal,
            slot_eta_r=slot_eta_r,
            slot_mu_r=slot_mu_r,
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
        grads = reference_forward_grads(
            path_idx=self.path_idx,
            image_source=self.image_source,
            slot_plane_point=self.slot_plane_point,
            slot_plane_normal=self.slot_plane_normal,
            slot_eta_r=self.slot_eta_r,
            slot_mu_r=self.slot_mu_r,
            slot_sigma=self.slot_sigma,
            slot_gain=self.slot_gain,
            target_pos=self.target_pos,
            tx_polarization=self.tx_polarization,
            chain_depth=self.chain_depth,
            n_paths=self.n_paths,
            wavelength=self.wavelength,
            tangent_image_source=point_from_grad(self.grad_in("image_source"), width),
            tangent_slot_plane_point=point_from_grad(self.grad_in("slot_plane_point"), slot_width),
            tangent_slot_plane_normal=vector_from_grad(self.grad_in("slot_plane_normal"), slot_width),
            tangent_slot_eta_r=array_from_grad(self.grad_in("slot_eta_r"), dr.width(self.slot_eta_r)),
            tangent_slot_mu_r=array_from_grad(self.grad_in("slot_mu_r"), dr.width(self.slot_mu_r)),
            tangent_slot_sigma=array_from_grad(self.grad_in("slot_sigma"), dr.width(self.slot_sigma)),
            tangent_slot_gain=array_from_grad(self.grad_in("slot_gain"), dr.width(self.slot_gain)),
            tangent_target_pos=point_from_grad(self.grad_in("target_pos"), width),
        )
        self.set_grad_out(grads)

    def backward(self):
        grads = reference_backward_grads(
            path_idx=self.path_idx,
            image_source=self.image_source,
            slot_plane_point=self.slot_plane_point,
            slot_plane_normal=self.slot_plane_normal,
            slot_eta_r=self.slot_eta_r,
            slot_mu_r=self.slot_mu_r,
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
        self.set_grad_in("slot_eta_r", grads["slot_eta_r"])
        self.set_grad_in("slot_mu_r", grads["slot_mu_r"])
        self.set_grad_in("slot_sigma", grads["slot_sigma"])
        self.set_grad_in("slot_gain", grads["slot_gain"])
        self.set_grad_in("target_pos", grads["target_pos"])


# ---------- output finalization (visibility + endpoints) -------------------

def gather_hit_point(slot_idx, hit_x_flat, hit_y_flat, hit_z_flat):
    return wt.Point3f(
        dr.gather(wt.Float, hit_x_flat, slot_idx),
        dr.gather(wt.Float, hit_y_flat, slot_idx),
        dr.gather(wt.Float, hit_z_flat, slot_idx),
    )


def finalize_native_outputs(
    *,
    raw_outputs, paths, path_idx, target_pos, scene,
    target_adjacent_faces, chain_depth,
    return_geometry, return_endpoints,
):
    width = dr.width(target_pos.x)
    (geom_valid_f, tx_pos_x, tx_pos_y, tx_pos_z,
     vec_x_re, vec_x_im, vec_y_re, vec_y_im, vec_z_re, vec_z_im,
     hit_x_flat, hit_y_flat, hit_z_flat) = raw_outputs

    tx_pos = wt.Point3f(tx_pos_x, tx_pos_y, tx_pos_z)
    chain_vector = {
        "x": wt.Complex2f(vec_x_re, vec_x_im),
        "y": wt.Complex2f(vec_y_re, vec_y_im),
        "z": wt.Complex2f(vec_z_re, vec_z_im),
    }
    geom_valid = geom_valid_f > 0.5

    tri_data = scene._triangle_runtime()
    tri_surface_data = {
        "group_size": tri_data["surface_group_size"],
        "group_members": tri_data["surface_group_members"],
        "max_group_size": int(tri_data["surface_max_group_size"]),
    }
    target_adjacent_surface_groups = surface_group_ignore_tuple(scene, target_adjacent_faces)
    slot_base = dr.arange(wt.UInt32, width)

    if return_endpoints and not return_geometry:
        valid = geom_valid
        prev_point = tx_pos
        prev_surface_group = None
        first_hit = zeros_point3(width)
        last_hit = zeros_point3(width)
        for slot in range(chain_depth):
            slot_idx = slot_base + wt.UInt32(slot * width)
            hit_p = gather_hit_point(slot_idx, hit_x_flat, hit_y_flat, hit_z_flat)
            prim_idx = dr.gather(wt.Int32, paths.prim_idx(slot), path_idx)
            valid_prim = prim_idx >= 0
            geom_n = gather(paths.plane_normal(slot), path_idx)
            surface_hit = surface_contains(
                hit_p,
                prim_idx,
                tri_data,
                tri_surface_data,
                valid_prim,
                scene=scene,
                plane_normal=geom_n,
            )
            valid = valid & valid_prim & surface_hit
            prim_surface_group = scene.triangle_group_id(prim_idx)
            ignore = (prim_surface_group,) if prev_surface_group is None else (prim_surface_group, prev_surface_group)
            valid = valid & scene.segment_visible(prev_point, hit_p, ignore_surface_group_idx=ignore)
            if slot == 0:
                first_hit = hit_p
            last_hit = hit_p
            prev_point = hit_p
            prev_surface_group = prim_surface_group

        last_ignore_groups = list(target_adjacent_surface_groups)
        if prev_surface_group is not None:
            last_ignore_groups.append(prev_surface_group)
        valid = valid & scene.segment_visible(
            prev_point, target_pos,
            ignore_surface_group_idx=tuple(last_ignore_groups),
        )
        chain_vector = vector_select(valid, chain_vector, vector_zero(width))
        return valid, chain_vector, {"tx_pos": tx_pos, "first_hit": first_hit, "last_hit": last_hit}

    hit_points: list = []
    normals: list = []
    prim_indices: list = []
    primary_sides: list = []
    valid = geom_valid
    for slot in range(chain_depth):
        slot_idx = slot_base + wt.UInt32(slot * width)
        hit_p = gather_hit_point(slot_idx, hit_x_flat, hit_y_flat, hit_z_flat)
        prim_idx = dr.gather(wt.Int32, paths.prim_idx(slot), path_idx)
        geom_n = gather(paths.plane_normal(slot), path_idx)
        valid_prim = prim_idx >= 0
        surface_hit = surface_contains(
            hit_p,
            prim_idx,
            tri_data,
            tri_surface_data,
            valid_prim,
            scene=scene,
            plane_normal=geom_n,
        )
        valid = valid & valid_prim & surface_hit
        hit_points.append(hit_p)
        normals.append(geom_n)
        prim_indices.append(prim_idx)
        primary_sides.append(surface_hit)

    prev_point = tx_pos
    prev_surface_group = None
    for slot, hit_p in enumerate(hit_points):
        ignore_groups = [scene.triangle_group_id(prim_indices[slot])]
        if prev_surface_group is not None:
            ignore_groups.append(prev_surface_group)
        valid = valid & scene.segment_visible(
            prev_point, hit_p,
            ignore_surface_group_idx=tuple(ignore_groups),
        )
        prev_point = hit_p
        prev_surface_group = ignore_groups[0]

    last_ignore = list(target_adjacent_surface_groups)
    if prev_surface_group is not None:
        last_ignore.append(prev_surface_group)
    valid = valid & scene.segment_visible(
        prev_point, target_pos,
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


def _adjacent_face_for_support(support, prim_idx):
    face0 = support.adjacent_face0
    face1 = support.adjacent_face1
    return dr.select(
        face0 == prim_idx,
        face1,
        dr.select(face1 == prim_idx, face0, wt.Int32(-1)),
    )


def _reference_outputs_with_slot_override(
    *,
    paths,
    path_idx,
    scene,
    reflection_detail,
    target_pos,
    tx: Tx,
    override_slot: int,
    override_plane_point: wt.Point3f,
    override_plane_normal: wt.Vector3f,
    override_prim_idx,
    chain_depth: int,
    wavelength: float,
):
    detail = coerce_trace_detail(reflection_detail)
    width = dr.width(target_pos.x)
    valid_mask = dr.full(wt.Bool, True, width)
    slot_plane_points: list = []
    slot_plane_normals: list = []
    slot_eta_r: list = []
    slot_mu_r: list = []
    slot_sigma: list = []
    slot_gain: list = []

    for slot in range(chain_depth):
        if slot == int(override_slot):
            plane_point = override_plane_point
            plane_normal = override_plane_normal
            prim_idx = wt.Int32(override_prim_idx)
        else:
            plane_point = gather(paths.plane_point(slot), path_idx)
            plane_normal = gather(paths.plane_normal(slot), path_idx)
            prim_idx = dr.gather(wt.Int32, paths.prim_idx(slot), path_idx)
        slot_plane_points.append(plane_point)
        slot_plane_normals.append(plane_normal)
        material_inputs = resolve_surface_material(
            scene=scene,
            prim_idx=prim_idx,
            default_gain=detail.reflection_gain,
            valid_mask=valid_mask & (prim_idx >= wt.Int32(0)),
        )
        slot_eta_r.append(material_inputs.eta_r)
        slot_mu_r.append(material_inputs.mu_r)
        slot_sigma.append(material_inputs.sigma)
        slot_gain.append(material_inputs.gain)

    image_source = broadcast(tx.position, width)
    for plane_point, plane_normal in zip(slot_plane_points, slot_plane_normals):
        image_source = reflect_point_across_plane(image_source, plane_point, plane_normal)

    local_path_idx = dr.arange(wt.UInt32, width)
    outputs = chain_math_reference(
        path_idx=local_path_idx,
        image_source=image_source,
        slot_plane_point=flatten_slot_points(slot_plane_points),
        slot_plane_normal=flatten_slot_vectors(slot_plane_normals),
        slot_eta_r=dr.concat(slot_eta_r),
        slot_mu_r=dr.concat(slot_mu_r),
        slot_sigma=dr.concat(slot_sigma),
        slot_gain=dr.concat(slot_gain),
        target_pos=target_pos,
        tx_polarization=tx.polarization,
        chain_depth=chain_depth,
        n_paths=width,
        wavelength=wavelength,
    )
    outputs["image_source"] = image_source
    return outputs


def _one_complex(width: int) -> wt.Complex2f:
    return wt.Complex2f(dr.ones(wt.Float, width), dr.zeros(wt.Float, width))


def _reference_segment_visibility_weight(
    *,
    scene,
    start_pos,
    end_pos,
    ignore_surface_groups,
    detail,
    wave: Wave,
) -> tuple[wt.Bool, wt.Complex2f]:
    width = dr.width(end_pos.x)
    if detail.reflection_secondary_visibility_mode == "hard":
        return (
            scene.segment_visible(
                start_pos,
                end_pos,
                ignore_surface_group_idx=tuple(ignore_surface_groups),
            ),
            _one_complex(width),
        )

    support = nearest_blocker_silhouette_edge(
        scene=scene,
        hit_p=start_pos,
        rx_pos=end_pos,
        primary_surface_group=tuple(ignore_surface_groups),
        mode=detail.reflection_secondary_visibility_mode,
        wavelength=wave.wavelength_scalar,
        boundary_radius_wavelengths=detail.reflection_f_weight_boundary_radius_wavelengths,
    )
    weight = reflection_segment_attenuation(support=support, wave_k=wave.k)
    return (~support.is_occluded) | support.valid, weight


def _reference_branch_visibility(
    *,
    outputs,
    paths,
    path_idx,
    scene,
    target_pos,
    target_adjacent_surface_groups,
    tri_data,
    tri_surface_data,
    reflection_detail,
    wave: Wave,
    branch_active,
    override_slot: int,
    override_prim_idx,
    chain_depth: int,
):
    width = dr.width(target_pos.x)
    valid = (outputs["geom_valid"] > 0.5) & branch_active
    visibility_weight = _one_complex(width)
    prev_point = outputs["tx_pos"]
    prev_surface_group = None
    for slot, hit_p in enumerate(outputs["hit_points"]):
        prim_idx = (
            wt.Int32(override_prim_idx)
            if slot == int(override_slot)
            else dr.gather(wt.Int32, paths.prim_idx(slot), path_idx)
        )
        valid_prim = prim_idx >= wt.Int32(0)
        geom_n = gather(paths.plane_normal(slot), path_idx)
        surface_hit = surface_contains(
            hit_p,
            prim_idx,
            tri_data,
            tri_surface_data,
            valid_prim,
            scene=scene,
            plane_normal=geom_n,
        )
        valid = valid & valid_prim & surface_hit
        prim_surface_group = scene.triangle_group_id(prim_idx)
        ignore_groups = [prim_surface_group]
        if prev_surface_group is not None:
            ignore_groups.append(prev_surface_group)
        segment_valid, segment_weight = _reference_segment_visibility_weight(
            scene=scene,
            start_pos=prev_point,
            end_pos=hit_p,
            ignore_surface_groups=ignore_groups,
            detail=reflection_detail,
            wave=wave,
        )
        valid = valid & segment_valid
        visibility_weight = visibility_weight * segment_weight
        prev_point = hit_p
        prev_surface_group = prim_surface_group

    last_ignore = list(target_adjacent_surface_groups)
    if prev_surface_group is not None:
        last_ignore.append(prev_surface_group)
    segment_valid, segment_weight = _reference_segment_visibility_weight(
        scene=scene,
        start_pos=prev_point,
        end_pos=target_pos,
        ignore_surface_groups=last_ignore,
        detail=reflection_detail,
        wave=wave,
    )
    visibility_weight = visibility_weight * segment_weight
    return valid & segment_valid, visibility_weight


def finalize_reference_f_weight_outputs(
    *,
    outputs,
    paths,
    path_idx,
    target_pos,
    scene,
    target_adjacent_faces,
    chain_depth,
    reflection_detail,
    wave: Wave,
    tx: Tx,
    return_geometry,
    return_endpoints,
):
    width = dr.width(target_pos.x)
    tx_pos = outputs["tx_pos"]
    hit_points = list(outputs["hit_points"])
    chain_vector = outputs["chain_vector"]
    geom_valid = outputs["geom_valid"] > 0.5

    tri_data = scene._triangle_runtime()
    tri_surface_data = {
        "group_size": tri_data["surface_group_size"],
        "group_members": tri_data["surface_group_members"],
        "max_group_size": int(tri_data["surface_max_group_size"]),
    }
    detail = coerce_trace_detail(reflection_detail)
    target_adjacent_surface_groups = surface_group_ignore_tuple(scene, target_adjacent_faces)

    valid = geom_valid
    chain_weight = wt.Complex2f(dr.ones(wt.Float, width), dr.zeros(wt.Float, width))
    residual_vector = vector_zero(width)
    residual_valid = dr.full(wt.Bool, False, width)
    slot_weights = []
    slot_supports = []
    prev_point = tx_pos
    prev_surface_group = None
    first_hit = zeros_point3(width)
    last_hit = zeros_point3(width)
    normals: list = []
    prim_indices: list = []
    primary_sides: list = []

    for slot, hit_p in enumerate(hit_points):
        prim_idx = dr.gather(wt.Int32, paths.prim_idx(slot), path_idx)
        plane_point = gather(paths.plane_point(slot), path_idx)
        geom_n = gather(paths.plane_normal(slot), path_idx)
        valid_prim = prim_idx >= 0
        primary_side = surface_contains(
            hit_p,
            prim_idx,
            tri_data,
            tri_surface_data,
            valid_prim,
            scene=scene,
            plane_normal=geom_n,
        )
        support = nearest_surface_boundary_edge(
            scene=scene,
            prim_idx=prim_idx,
            hit_p=hit_p,
            mode=detail.reflection_transition_mode,
            wavelength=wave.wavelength_scalar,
            boundary_radius_wavelengths=detail.reflection_f_weight_boundary_radius_wavelengths,
            max_edges_per_slot=detail.reflection_f_weight_max_edges_per_slot,
        )
        next_point = hit_points[slot + 1] if slot + 1 < chain_depth else target_pos
        weights = reflection_transition_weights(
            hit_p=hit_p,
            previous_point=prev_point,
            next_point=next_point,
            primary_plane_point=plane_point,
            primary_plane_normal=geom_n,
            edge_support=support,
            wave_k=wave.k,
            primary_side_mask=primary_side,
        )
        chain_weight = chain_weight * weights.primary_weight
        slot_weights.append(weights)
        slot_supports.append(support)
        valid = valid & valid_prim & (primary_side | support.valid)

        prim_surface_group = scene.triangle_group_id(prim_idx)
        ignore_groups = [prim_surface_group]
        if prev_surface_group is not None:
            ignore_groups.append(prev_surface_group)
        segment_valid, segment_weight = _reference_segment_visibility_weight(
            scene=scene,
            start_pos=prev_point,
            end_pos=hit_p,
            ignore_surface_groups=ignore_groups,
            detail=detail,
            wave=wave,
        )
        valid = valid & segment_valid
        chain_weight = chain_weight * segment_weight

        if slot == 0:
            first_hit = hit_p
        last_hit = hit_p
        normals.append(geom_n)
        prim_indices.append(prim_idx)
        primary_sides.append(primary_side)
        prev_point = hit_p
        prev_surface_group = prim_surface_group

    last_ignore = list(target_adjacent_surface_groups)
    if prev_surface_group is not None:
        last_ignore.append(prev_surface_group)
    segment_valid, segment_weight = _reference_segment_visibility_weight(
        scene=scene,
        start_pos=prev_point,
        end_pos=target_pos,
        ignore_surface_groups=last_ignore,
        detail=detail,
        wave=wave,
    )
    valid = valid & segment_valid
    chain_weight = chain_weight * segment_weight

    one_weight = _one_complex(width)
    prefix_weights = []
    running_prefix = one_weight
    for weights in slot_weights:
        prefix_weights.append(running_prefix)
        running_prefix = running_prefix * weights.primary_weight

    suffix_weights = [one_weight for _ in slot_weights]
    running_suffix = one_weight
    for slot in range(len(slot_weights) - 1, -1, -1):
        suffix_weights[slot] = running_suffix
        running_suffix = slot_weights[slot].primary_weight * running_suffix

    for slot, weights in enumerate(slot_weights):
        prim_idx = prim_indices[slot]
        adjacent_prim_idx = _adjacent_face_for_support(slot_supports[slot], prim_idx)
        branch_active = weights.adjacent_valid & (adjacent_prim_idx >= wt.Int32(0))
        if bool(dr.any(branch_active)):
            branch_outputs = _reference_outputs_with_slot_override(
                paths=paths,
                path_idx=path_idx,
                scene=scene,
                reflection_detail=detail,
                target_pos=target_pos,
                tx=tx,
                override_slot=slot,
                override_plane_point=weights.adjacent_plane_point,
                override_plane_normal=weights.adjacent_plane_normal,
                override_prim_idx=adjacent_prim_idx,
                chain_depth=chain_depth,
                wavelength=wave.wavelength_scalar,
            )
            branch_valid, branch_visibility_weight = _reference_branch_visibility(
                outputs=branch_outputs,
                paths=paths,
                path_idx=path_idx,
                scene=scene,
                target_pos=target_pos,
                target_adjacent_surface_groups=target_adjacent_surface_groups,
                tri_data=tri_data,
                tri_surface_data=tri_surface_data,
                reflection_detail=detail,
                wave=wave,
                branch_active=branch_active,
                override_slot=slot,
                override_prim_idx=adjacent_prim_idx,
                chain_depth=chain_depth,
            )
            # The branch has its own image source; correct the spreading and
            # phase relative to the primary unit field applied downstream.
            primary_image = gather(paths.image_source, path_idx)
            primary_distance = dr.norm(target_pos - wt.Point3f(
                primary_image.x, primary_image.y, primary_image.z,
            )) + wt.Float(EPS)
            branch_distance = dr.norm(target_pos - branch_outputs["image_source"]) + wt.Float(EPS)
            branch_unit_ratio = wt.Complex2f(primary_distance / branch_distance, 0.0) * dr.exp(
                wt.Complex2f(0.0, -wave.k * (branch_distance - primary_distance))
            )
            residual_weight = (
                prefix_weights[slot]
                * weights.adjacent_weight
                * suffix_weights[slot]
                * branch_visibility_weight
                * branch_unit_ratio
            )
            residual_vector = vector_add(
                residual_vector,
                vector_select(
                    branch_valid,
                    vector_scale(branch_outputs["chain_vector"], residual_weight),
                    vector_zero(width),
                ),
            )
            residual_valid = residual_valid | branch_valid

    primary_vector = vector_select(valid, vector_scale(chain_vector, chain_weight), vector_zero(width))
    valid = valid | residual_valid
    chain_vector = vector_select(valid, vector_add(primary_vector, residual_vector), vector_zero(width))
    if return_endpoints and not return_geometry:
        return valid, chain_vector, {"tx_pos": tx_pos, "first_hit": first_hit, "last_hit": last_hit}
    if return_geometry:
        return valid, chain_vector, {
            "tx_pos": tx_pos,
            "hit_points": hit_points,
            "normals": normals,
            "prim_indices": prim_indices,
            "primary_sides": primary_sides,
        }
    return valid, chain_vector


def _detached_point(value):
    point_type = dr.detached_t(wt.Point3f)
    return point_type(dr.detach(value.x), dr.detach(value.y), dr.detach(value.z))


def _detached_vector(value):
    vector_type = dr.detached_t(wt.Vector3f)
    return vector_type(dr.detach(value.x), dr.detach(value.y), dr.detach(value.z))


def _detached_bool(value):
    bool_type = dr.detached_t(wt.Bool)
    return bool_type(dr.detach(value))


def _rayd_point_arg(value, *, ad: bool):
    if ad:
        return wt.Point3f(value.x, value.y, value.z)
    return _detached_point(value)


def _rayd_bool_arg(value, *, ad: bool):
    if ad:
        return wt.Bool(value)
    return _detached_bool(value)


def _rayd_vector_arg(value, *, ad: bool):
    if ad:
        return wt.Vector3f(value.x, value.y, value.z)
    return _detached_vector(value)


def _detached_int(value):
    int_type = dr.detached_t(wt.Int32)
    return int_type(dr.detach(wt.Int32(value)))


def _detached_float(value):
    float_type = dr.detached_t(wt.Float)
    return float_type(dr.detach(wt.Float(value)))


def _rayd_float_arg(value, *, ad: bool):
    if ad:
        return wt.Float(value)
    return _detached_float(value)


def _point_from_rayd(value):
    return wt.Point3f(wt.Float(value.x), wt.Float(value.y), wt.Float(value.z))


def _vector_from_rayd(value):
    return wt.Vector3f(wt.Float(value.x), wt.Float(value.y), wt.Float(value.z))


def _complex_inverse(value):
    denom = dr.maximum(value.real * value.real + value.imag * value.imag, wt.Float(1.0e-30))
    return wt.Complex2f(value.real / denom, -value.imag / denom)


def _rayd_path_unit_field(path_length, valid, wave: Wave):
    safe_length = dr.select(valid & (path_length > wt.Float(EPS)), path_length, wt.Float(1.0))
    phase = unit_phase_neg_kd(wave.k, safe_length)
    amplitude = wt.Float(wave.wavelength_scalar / (4.0 * dr.pi)) / dr.maximum(
        safe_length,
        wt.Float(1.0e-6),
    )
    return wt.Complex2f(amplitude, 0.0) * phase


def _rayd_expected_prim_ids(*, paths, resolved_path_idx, chain_depth: int, width: int):
    expected = dr.full(wt.Int32, -1, width * chain_depth)
    lane = dr.arange(wt.UInt32, width)
    for slot in range(chain_depth):
        expected_prim = dr.gather(wt.Int32, paths.prim_idx(slot), resolved_path_idx)
        dr.scatter(expected, expected_prim, lane * wt.UInt32(chain_depth) + wt.UInt32(slot))
    return expected


def _populate_rayd_epc_options(options, *, scene, paths, resolved_path_idx, chain_depth: int, width: int):
    tri_data = scene._triangle_runtime()
    options.expected_prim_ids = _detached_int(
        _rayd_expected_prim_ids(
            paths=paths,
            resolved_path_idx=resolved_path_idx,
            chain_depth=chain_depth,
            width=width,
        )
    )
    options.surface_group_id = _detached_int(tri_data["surface_group_id"])
    options.surface_group_size = _detached_int(tri_data["surface_group_size_by_group"])
    options.surface_group_members = _detached_int(tri_data["surface_group_members_by_group"])
    options.surface_max_group_size = int(tri_data["surface_max_group_size"])
    options.visibility_ignore_mode = "surface_group"
    return options


def _rayd_lane_major_slot_float(source, *, descriptor, descriptor_path_idx, chain_depth: int, width: int):
    out = dr.zeros(wt.Float, width * chain_depth)
    lane = dr.arange(wt.UInt32, width)
    for slot in range(chain_depth):
        source_idx = descriptor_path_idx + wt.UInt32(slot * int(descriptor.n_paths))
        target_idx = lane * wt.UInt32(chain_depth) + wt.UInt32(slot)
        dr.scatter(out, dr.gather(wt.Float, source, source_idx), target_idx)
    return out


def _rayd_lane_major_slot_vector(source, *, descriptor, descriptor_path_idx, chain_depth: int, width: int):
    return wt.Vector3f(
        _rayd_lane_major_slot_float(
            source.x,
            descriptor=descriptor,
            descriptor_path_idx=descriptor_path_idx,
            chain_depth=chain_depth,
            width=width,
        ),
        _rayd_lane_major_slot_float(
            source.y,
            descriptor=descriptor,
            descriptor_path_idx=descriptor_path_idx,
            chain_depth=chain_depth,
            width=width,
        ),
        _rayd_lane_major_slot_float(
            source.z,
            descriptor=descriptor,
            descriptor_path_idx=descriptor_path_idx,
            chain_depth=chain_depth,
            width=width,
        ),
    )


def _rayd_epc_field_options(
    *,
    scene,
    paths,
    resolved_path_idx,
    descriptor: Descriptor,
    descriptor_path_idx,
    chain_depth: int,
    width: int,
    wave: Wave,
    tx: Tx,
    return_geometry: bool,
    return_endpoints: bool,
    ad: bool,
):
    if ad and not hasattr(rayd, "ReflEpcFieldOptionsAD"):
        raise RuntimeError("RayD ReflEpcFieldOptionsAD is required for AD reflection EPC.")
    options_type = rayd.ReflEpcFieldOptionsAD if ad else rayd.ReflEpcFieldOptions
    options = _populate_rayd_epc_options(
        options_type(),
        scene=scene,
        paths=paths,
        resolved_path_idx=resolved_path_idx,
        chain_depth=chain_depth,
        width=width,
    )
    options.slot_plane_point = _rayd_vector_arg(
        _rayd_lane_major_slot_vector(
            descriptor.slot_plane_point,
            descriptor=descriptor,
            descriptor_path_idx=descriptor_path_idx,
            chain_depth=chain_depth,
            width=width,
        ),
        ad=ad,
    )
    options.slot_plane_normal = _rayd_vector_arg(
        _rayd_lane_major_slot_vector(
            descriptor.slot_plane_normal,
            descriptor=descriptor,
            descriptor_path_idx=descriptor_path_idx,
            chain_depth=chain_depth,
            width=width,
        ),
        ad=ad,
    )
    options.slot_eta_r = _rayd_float_arg(
        _rayd_lane_major_slot_float(
            descriptor.slot_eta_r,
            descriptor=descriptor,
            descriptor_path_idx=descriptor_path_idx,
            chain_depth=chain_depth,
            width=width,
        ),
        ad=ad,
    )
    options.slot_mu_r = _rayd_float_arg(
        _rayd_lane_major_slot_float(
            descriptor.slot_mu_r,
            descriptor=descriptor,
            descriptor_path_idx=descriptor_path_idx,
            chain_depth=chain_depth,
            width=width,
        ),
        ad=ad,
    )
    options.slot_sigma = _rayd_float_arg(
        _rayd_lane_major_slot_float(
            descriptor.slot_sigma,
            descriptor=descriptor,
            descriptor_path_idx=descriptor_path_idx,
            chain_depth=chain_depth,
            width=width,
        ),
        ad=ad,
    )
    options.slot_gain = _rayd_float_arg(
        _rayd_lane_major_slot_float(
            descriptor.slot_gain,
            descriptor=descriptor,
            descriptor_path_idx=descriptor_path_idx,
            chain_depth=chain_depth,
            width=width,
        ),
        ad=ad,
    )
    options.tx_polarization = _rayd_vector_arg(tx.polarization, ad=ad)
    options.omega = float(material_angular_frequency(wave.wavelength_scalar)[0])
    options.wavelength = float(wave.wavelength_scalar)
    options.return_geom = bool(return_geometry)
    options.return_endpoints = bool(return_endpoints)
    options.return_hit_points = bool(return_geometry)
    options.return_normals = bool(return_geometry)
    options.return_resolved_prim_ids = False
    options.return_surface_group_ids = False
    return options


def _target_adjacent_faces_present(target_adjacent_faces) -> bool:
    if not target_adjacent_faces:
        return False
    return any(face is not None for face in target_adjacent_faces)


def _rayd_epc_supports_fine_geometry_options() -> bool:
    global _RAYD_EPC_FINE_GEOMETRY_OPTIONS
    if _RAYD_EPC_FINE_GEOMETRY_OPTIONS is None:
        field_options = rayd.ReflEpcFieldOptions()
        _RAYD_EPC_FINE_GEOMETRY_OPTIONS = all(
            hasattr(field_options, attr)
            for attr in (
                "return_hit_points",
                "return_normals",
                "return_resolved_prim_ids",
                "return_surface_group_ids",
            )
        )
    return bool(_RAYD_EPC_FINE_GEOMETRY_OPTIONS)


def rayd_epc_eligible(
    *,
    chain_depth: int,
    scene,
    target_adjacent_faces,
) -> bool:
    if chain_depth <= 0 or chain_depth > RAYD_REFLECTION_EPC_MAX_BOUNCES:
        return False
    if _RAYD_EPC_FIELD_PIPELINE_AVAILABLE is False:
        return False
    if _target_adjacent_faces_present(target_adjacent_faces):
        return False
    if scene is None or getattr(scene, "_rayd_scene", None) is None:
        return False
    if not hasattr(scene._rayd_scene, "trace_refl_epc_field"):
        return False
    if not _rayd_epc_supports_fine_geometry_options():
        return False
    if hasattr(scene, "_symbolic_recording_active") and scene._symbolic_recording_active():
        return False
    return True


def chain_to_target_rayd(
    *,
    paths,
    path_idx,
    target_pos,
    scene,
    reflection_detail,
    wave: Wave,
    tx: Tx,
    return_geometry,
    return_endpoints,
    epc_descriptor: Descriptor | None,
):
    width = dr.width(target_pos.x)
    chain_depth = int(paths.chain_depth)
    n_paths = int(paths.n_paths)
    if chain_depth <= 0 or n_paths <= 0:
        return empty_return(width, chain_depth, return_geometry, return_endpoints)

    descriptor = epc_descriptor
    descriptor_path_idx = wt.UInt32(path_idx)
    resolved_path_idx = path_idx
    if descriptor is None:
        descriptor = build_descriptor(
            paths=paths,
            path_idx=path_idx,
            scene=scene,
            reflection_detail=reflection_detail,
        )
        descriptor_path_idx = dr.arange(wt.UInt32, width)
        resolved_path_idx = dr.gather(wt.UInt32, descriptor.source_path_idx, descriptor_path_idx)
    else:
        resolved_path_idx = dr.gather(wt.UInt32, descriptor.source_path_idx, descriptor_path_idx)

    if int(descriptor.n_paths) <= 0:
        return empty_return(width, chain_depth, return_geometry, return_endpoints)

    tx_pos = broadcast(tx.position, width)
    slot_base = dr.arange(wt.UInt32, width) * wt.UInt32(chain_depth)
    use_rayd_ad = (
        point_grad_enabled(tx_pos)
        or point_grad_enabled(target_pos)
        or descriptor_geometry_has_grad(descriptor)
        or descriptor_material_has_grad(descriptor)
        or point_has_grad(tx.polarization)
    )
    options = _rayd_epc_field_options(
        scene=scene,
        paths=paths,
        resolved_path_idx=resolved_path_idx,
        descriptor=descriptor,
        descriptor_path_idx=descriptor_path_idx,
        chain_depth=chain_depth,
        width=width,
        wave=wave,
        tx=tx,
        return_geometry=return_geometry,
        return_endpoints=return_endpoints and not return_geometry,
        ad=use_rayd_ad,
    )
    active = dr.full(wt.Bool, True, width)
    with dr.scoped_set_flag(dr.JitFlag.Recording, False):
        rayd_result = scene._rayd_scene.trace_refl_epc_field(
            _rayd_point_arg(tx_pos, ad=use_rayd_ad),
            _rayd_point_arg(target_pos, ad=use_rayd_ad),
            int(chain_depth),
            options=options,
            active=_rayd_bool_arg(active, ad=use_rayd_ad),
        )

    valid = wt.Bool(rayd_result.valid) & (wt.Int32(rayd_result.bounce_count) == wt.Int32(chain_depth))
    rayd_field = {
        "x": wt.Complex2f(wt.Float(rayd_result.field_x_re), wt.Float(rayd_result.field_x_im)),
        "y": wt.Complex2f(wt.Float(rayd_result.field_y_re), wt.Float(rayd_result.field_y_im)),
        "z": wt.Complex2f(wt.Float(rayd_result.field_z_re), wt.Float(rayd_result.field_z_im)),
    }
    unit_field_inv = _complex_inverse(
        _rayd_path_unit_field(wt.Float(rayd_result.path_length), valid, wave)
    )
    chain_vector = {axis: rayd_field[axis] * unit_field_inv for axis in ("x", "y", "z")}
    chain_vector = vector_select(valid, chain_vector, vector_zero(width))

    if return_endpoints and not return_geometry:
        return valid, chain_vector, {
            "tx_pos": tx_pos,
            "first_hit": _point_from_rayd(rayd_result.first_hit),
            "last_hit": _point_from_rayd(rayd_result.last_hit),
        }
    if return_geometry:
        hit_points: list = []
        normals: list = []
        prim_indices: list = []
        for slot in range(chain_depth):
            rayd_slot_idx = slot_base + wt.UInt32(slot)
            hit_points.append(_point_from_rayd(gather(rayd_result.hit_points, rayd_slot_idx)))
            normals.append(_vector_from_rayd(gather(rayd_result.normals, rayd_slot_idx)))
            prim_indices.append(dr.gather(wt.Int32, paths.prim_idx(slot), resolved_path_idx))
        return valid, chain_vector, {
            "tx_pos": tx_pos,
            "hit_points": hit_points,
            "normals": normals,
            "prim_indices": prim_indices,
        }
    return valid, chain_vector


# ---------- backend dispatch ------------------------------------------------

def chain_to_target_native(
    *,
    paths, path_idx, target_pos, scene, target_adjacent_faces,
    reflection_detail, wave: Wave, tx: Tx,
    return_geometry, return_endpoints, use_custom_op,
    epc_descriptor: Descriptor | None,
):
    width = dr.width(target_pos.x)
    chain_depth = int(paths.chain_depth)
    n_paths = int(paths.n_paths)
    if chain_depth <= 0 or n_paths <= 0:
        return empty_return(width, chain_depth, return_geometry, return_endpoints)

    descriptor = epc_descriptor
    resolved_path_idx = path_idx
    if descriptor is None:
        descriptor = build_descriptor(
            paths=paths, path_idx=path_idx, scene=scene, reflection_detail=reflection_detail,
        )
        path_idx = dr.arange(wt.UInt32, width)
        resolved_path_idx = dr.gather(wt.UInt32, descriptor.source_path_idx, path_idx)
    else:
        resolved_path_idx = dr.gather(wt.UInt32, descriptor.source_path_idx, wt.UInt32(path_idx))
    descriptor_n_paths = int(descriptor.n_paths)
    if descriptor_n_paths <= 0:
        return empty_return(width, chain_depth, return_geometry, return_endpoints)

    n_pairs = dr.width(target_pos.x)
    wavelength = wave.wavelength_scalar
    tx_polarization = tx.polarization

    if use_custom_op:
        raw_outputs = dr.custom(
            EpcTargetsOp,
            path_idx,
            descriptor.image_source,
            descriptor.slot_plane_point,
            descriptor.slot_plane_normal,
            descriptor.slot_eta_r,
            descriptor.slot_mu_r,
            descriptor.slot_sigma,
            descriptor.slot_gain,
            target_pos,
            tx_polarization=tx_polarization,
            n_pairs=int(n_pairs),
            n_paths=descriptor_n_paths,
            chain_depth=chain_depth,
            wavelength=float(wavelength),
        )
    else:
        raw_outputs = launch_native_forward(
            path_idx=path_idx,
            image_source=descriptor.image_source,
            slot_plane_point=descriptor.slot_plane_point,
            slot_plane_normal=descriptor.slot_plane_normal,
            slot_eta_r=descriptor.slot_eta_r,
            slot_mu_r=descriptor.slot_mu_r,
            slot_sigma=descriptor.slot_sigma,
            slot_gain=descriptor.slot_gain,
            target_pos=target_pos,
            tx_polarization=tx_polarization,
            n_pairs=n_pairs,
            n_paths=descriptor_n_paths,
            chain_depth=chain_depth,
            wavelength=wavelength,
        )

    return finalize_native_outputs(
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


def chain_to_target_reference_f_weight(
    *,
    paths,
    path_idx,
    target_pos,
    scene,
    target_adjacent_faces,
    reflection_detail,
    wave: Wave,
    tx: Tx,
    return_geometry,
    return_endpoints,
    epc_descriptor: Descriptor | None,
):
    width = dr.width(target_pos.x)
    chain_depth = int(paths.chain_depth)
    n_paths = int(paths.n_paths)
    if chain_depth <= 0 or n_paths <= 0:
        return empty_return(width, chain_depth, return_geometry, return_endpoints)

    descriptor = epc_descriptor
    resolved_path_idx = path_idx
    if descriptor is None:
        descriptor = build_descriptor(
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
        return empty_return(width, chain_depth, return_geometry, return_endpoints)

    outputs = chain_math_reference(
        path_idx=path_idx,
        image_source=descriptor.image_source,
        slot_plane_point=descriptor.slot_plane_point,
        slot_plane_normal=descriptor.slot_plane_normal,
        slot_eta_r=descriptor.slot_eta_r,
        slot_mu_r=descriptor.slot_mu_r,
        slot_sigma=descriptor.slot_sigma,
        slot_gain=descriptor.slot_gain,
        target_pos=target_pos,
        tx_polarization=tx.polarization,
        chain_depth=chain_depth,
        n_paths=descriptor_n_paths,
        wavelength=wave.wavelength_scalar,
    )
    return finalize_reference_f_weight_outputs(
        outputs=outputs,
        paths=paths,
        path_idx=resolved_path_idx,
        target_pos=target_pos,
        scene=scene,
        target_adjacent_faces=target_adjacent_faces,
        chain_depth=chain_depth,
        reflection_detail=reflection_detail,
        wave=wave,
        tx=tx,
        return_geometry=return_geometry,
        return_endpoints=return_endpoints,
    )


def chain_to_target(
    *,
    paths,
    path_idx,
    target_pos,
    scene,
    target_adjacent_faces=(),
    reflection_detail,
    wave: Wave,
    tx: Tx,
    return_geometry: bool = False,
    return_endpoints: bool = False,
    epc_descriptor: Descriptor | None = None,
    prefer_rayd_epc: bool = True,
    require_rayd_epc: bool = False,
):
    """Image-source EPC of a reflection chain to ``target_pos``."""
    width = dr.width(target_pos.x)
    if scene is None or scene._triangle_runtime() is None or paths is None:
        return empty_return(width, 0, return_geometry, return_endpoints)

    chain_depth = int(paths.chain_depth)
    n_paths = int(paths.n_paths)
    if chain_depth <= 0 or n_paths <= 0:
        return empty_return(width, chain_depth, return_geometry, return_endpoints)
    detail = coerce_trace_detail(reflection_detail)
    if (
        detail.reflection_transition_mode in {"f_weight_reference", "f_weight_native"}
        or detail.reflection_secondary_visibility_mode != "hard"
    ):
        return chain_to_target_reference_f_weight(
            paths=paths,
            path_idx=path_idx,
            target_pos=target_pos,
            scene=scene,
            target_adjacent_faces=target_adjacent_faces,
            reflection_detail=detail,
            wave=wave,
            tx=tx,
            return_geometry=return_geometry,
            return_endpoints=return_endpoints,
            epc_descriptor=epc_descriptor,
        )

    rayd_eligible = prefer_rayd_epc and rayd_epc_eligible(
        chain_depth=chain_depth,
        scene=scene,
        target_adjacent_faces=target_adjacent_faces,
    )
    if rayd_eligible:
        try:
            return chain_to_target_rayd(
                paths=paths,
                path_idx=path_idx,
                target_pos=target_pos,
                scene=scene,
                reflection_detail=detail,
                wave=wave,
                tx=tx,
                return_geometry=return_geometry,
                return_endpoints=return_endpoints,
                epc_descriptor=epc_descriptor,
            )
        except RuntimeError as exc:
            if require_rayd_epc or "optixPipelineCreate" not in str(exc):
                raise
            global _RAYD_EPC_FIELD_PIPELINE_AVAILABLE
            _RAYD_EPC_FIELD_PIPELINE_AVAILABLE = False
    if require_rayd_epc:
        raise RuntimeError(
            "RayD reflection EPC was required for this path solve, but the workload "
            "is not eligible for scene._rayd_scene.trace_refl_epc_field()."
        )
    native_eligible = native_epc_eligible(
        paths,
        target_pos,
        chain_depth,
        scene=scene,
        descriptor=epc_descriptor,
    )
    return chain_to_target_native(
        paths=paths,
        path_idx=path_idx,
        target_pos=target_pos,
        scene=scene,
        target_adjacent_faces=target_adjacent_faces,
        reflection_detail=reflection_detail,
        wave=wave,
        tx=tx,
        return_geometry=return_geometry,
        return_endpoints=return_endpoints,
        use_custom_op=not native_eligible,
        epc_descriptor=epc_descriptor,
    )


__all__ = [
    "Descriptor",
    "build_descriptor",
    "chain_to_target",
]

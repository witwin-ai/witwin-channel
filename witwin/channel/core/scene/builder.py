"""Scene building: material binding, surface groups, edge runtime, and orchestration."""

from __future__ import annotations

import drjit as dr
import rayd.drjit as rayd
import torch
from witwin.channel import types as wt

from witwin.core import Mesh as CoreMesh
from witwin.channel.core.numerics.arrays import concat_arrays, scalar
from witwin.channel.core.numerics.constants import EPS, SMALL_EPS
from witwin.channel.core.geometry import triangles_are_coplanar_mask
from witwin.channel.core.geometry.mesh_buffers import mesh_buffer_count, to_point3f, to_vector3u

from .edge_policy import DEFAULT_EDGE_POLICY, EdgePolicy
from .material_presets import material_sample_for_frequency
from .wedge import WedgeConfig, WedgeOps

SURFACE_GROUP_NORMAL_COS_TOL = 1.0 - 1e-5
SURFACE_GROUP_PLANE_TOL = 1e-5
MATERIAL_EPS_R_DEFAULT = 1.0
MATERIAL_MU_R_DEFAULT = 1.0
MATERIAL_SIGMA_E_DEFAULT = 0.0


class SceneBuilder:
    """Static helpers that build and update Scene runtime state."""

    _SURFACE_LABEL_PROPAGATION_STEPS = 8

    # ------------------------------------------------------------------
    # RayD runtime backend
    # ------------------------------------------------------------------

    @staticmethod
    def build_rayd_scene(meshes: list[dict]) -> rayd.Scene:
        """Build a RayD scene from per-structure mesh dicts (must contain 'vertices' and 'faces')."""
        if not meshes:
            raise ValueError("Cannot build a RayD scene with no meshes.")
        rayd_scene = rayd.Scene()
        for m in meshes:
            rayd_scene.add_mesh(rayd.Mesh(to_point3f(m["vertices"]), to_vector3u(m["faces"])), dynamic=True)
        rayd_scene.build()
        rayd_scene.sync()
        return rayd_scene

    @staticmethod
    def configure_runtime_backends(scene, *, update_mesh_id: int | None = None, vertices=None) -> None:
        """Build (full rebuild) or update (in-place vertex update) the RayD runtime scene."""
        try:
            if update_mesh_id is not None and scene._rayd_scene is not None and vertices is not None:
                scene._rayd_scene.update_mesh_vertices(update_mesh_id, to_point3f(vertices))
                scene._rayd_scene.sync()
            else:
                scene._rayd_scene = SceneBuilder.build_rayd_scene(scene._structure_meshes)
            if hasattr(scene, "_rayd_visibility_ignore_pipeline_warmed"):
                scene._rayd_visibility_ignore_pipeline_warmed = False
        except Exception:
            scene._rayd_scene = None
            raise

    # ------------------------------------------------------------------
    # Top-level orchestration
    # ------------------------------------------------------------------

    @staticmethod
    def rebuild(scene) -> None:
        """Full rebuild: structures -> RayD -> triangle data -> surface groups -> edges."""
        structure_meshes, material_entries = SceneBuilder._build_structure_meshes(scene)
        scene._structure_meshes = structure_meshes

        if structure_meshes:
            SceneBuilder.configure_runtime_backends(scene)
            n_triangles = SceneBuilder._total_triangles(structure_meshes)
            scene._triangle_material_data = SceneBuilder._build_triangle_material_data(material_entries, n_triangles)
        else:
            scene._rayd_scene = None
            scene._triangle_material_data = None

        SceneBuilder._link_meshes(scene)
        scene._mesh_version += 1
        scene._edge_cache.clear()
        SceneBuilder._preload_triangle_data(scene)
        SceneBuilder._refresh_surface_groups(scene)
        SceneBuilder._clear_edge_runtime(scene)
        scene._edge_runtime_dirty = True

    @staticmethod
    def sync(scene, recompute_edges: bool = True) -> None:
        """Sync RayD BVH and recompute derived scene data after vertex updates."""
        if scene._rayd_scene is not None:
            scene._rayd_scene.sync()
        scene._mesh_version += 1
        scene._edge_cache.clear()
        SceneBuilder._preload_triangle_data(scene)
        SceneBuilder._refresh_surface_groups(scene)
        if recompute_edges:
            SceneBuilder._clear_edge_runtime(scene)
            scene._edge_runtime_dirty = True
        else:
            SceneBuilder._clear_edge_runtime(scene)
            scene._edge_runtime_dirty = True

    @staticmethod
    def _link_meshes(scene) -> None:
        """Attach scene + mesh_id to Mesh geometries so they can push vertex updates."""
        from .mesh import Mesh
        mesh_id = 0
        for structure in scene.structures:
            if not structure.enabled:
                continue
            if isinstance(structure.geometry, Mesh):
                structure.geometry._scene = scene
                structure.geometry._mesh_id = mesh_id
            mesh_id += 1

    # ------------------------------------------------------------------
    # Mesh construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_structure_meshes(scene) -> tuple[list[dict], list[dict]]:
        structure_meshes, material_entries = [], []
        for idx, structure in enumerate(scene.structures):
            if not structure.enabled:
                continue
            geometry = structure.geometry
            to_mesh_kwargs = {"device": scene.device} if (isinstance(geometry, CoreMesh) or geometry.kind == "mesh") else {}
            vertices, faces = geometry.to_mesh(**to_mesh_kwargs)

            vertices = vertices.to(dtype=torch.float32).contiguous() if isinstance(vertices, torch.Tensor) else to_point3f(vertices)
            faces = faces.to(dtype=torch.int32).contiguous() if isinstance(faces, torch.Tensor) else to_vector3u(faces)

            n_tri = mesh_buffer_count(faces)
            sample = material_sample_for_frequency(structure.material, getattr(scene, "frequency", None))
            specified = SceneBuilder._material_is_specified(sample)

            structure_meshes.append({"vertices": vertices, "faces": faces})
            material_entries.append({
                "n_triangles": n_tri,
                "structure_idx": idx,
                "eps_r": SceneBuilder._material_array(sample.eps_r, n_tri, name=f"{structure.name}.material.eps_r"),
                "mu_r": SceneBuilder._material_array(sample.mu_r, n_tri, name=f"{structure.name}.material.mu_r"),
                "sigma_e": SceneBuilder._material_array(sample.sigma_e, n_tri, name=f"{structure.name}.material.sigma_e"),
                "specified": dr.full(wt.Bool, specified, n_tri),
                "structure_idx_array": dr.full(wt.Int32, idx, n_tri),
                "specified_value": bool(specified),
            })
        return structure_meshes, material_entries

    @staticmethod
    def _total_triangles(structure_meshes: list[dict]) -> int:
        return sum(mesh_buffer_count(m["faces"]) for m in structure_meshes)

    # ------------------------------------------------------------------
    # Material helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _real_material_value(value, *, name: str):
        if isinstance(value, complex):
            if abs(value.imag) > 1.0e-12:
                raise TypeError(f"{name} must be real-valued; got {value!r}.")
            return value.real
        return value

    @staticmethod
    def _material_array(value, n: int, *, name: str):
        if n <= 0:
            return dr.zeros(wt.Float, 0)
        value = SceneBuilder._real_material_value(value, name=name)
        if isinstance(value, wt.Float):
            w = int(dr.width(value))
            if w == n:
                return value
            if w == 1:
                return dr.repeat(value, n)
            raise ValueError(f"{name} must be scalar or length {n}, got DrJit width {w}.")
        if isinstance(value, torch.Tensor):
            flat = value.reshape(-1).contiguous()
            if flat.is_complex():
                if bool((flat.imag.abs() > 1.0e-12).any()):
                    raise TypeError(f"{name} must be real-valued.")
                flat = flat.real.contiguous()
            if flat.numel() == n:
                return wt.Float(flat)
            if flat.numel() == 1:
                return dr.full(wt.Float, float(flat.detach().cpu()[0]), n)
            raise ValueError(f"{name} must be scalar or length {n}, got tensor shape {tuple(value.shape)}.")
        try:
            return dr.full(wt.Float, float(value), n)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be a float-compatible scalar or DrJit Float.") from exc

    @staticmethod
    def _material_is_specified(sample) -> bool:
        try:
            eps_r = float(SceneBuilder._real_material_value(sample.eps_r, name="material.eps_r"))
            mu_r = float(SceneBuilder._real_material_value(getattr(sample, "mu_r", MATERIAL_MU_R_DEFAULT), name="material.mu_r"))
            sigma_e = float(SceneBuilder._real_material_value(sample.sigma_e, name="material.sigma_e"))
        except (TypeError, ValueError):
            return True
        return not (
            abs(eps_r - MATERIAL_EPS_R_DEFAULT) <= 1e-9
            and abs(mu_r - MATERIAL_MU_R_DEFAULT) <= 1e-9
            and abs(sigma_e - MATERIAL_SIGMA_E_DEFAULT) <= 1e-12
        )

    @staticmethod
    def _build_triangle_material_data(material_entries: list[dict], n_triangles: int) -> dict | None:
        if not material_entries or n_triangles <= 0:
            return None
        n_specified = sum(e["n_triangles"] for e in material_entries if e["specified_value"])
        return {
            "eps_r": concat_arrays(wt.Float, [e["eps_r"] for e in material_entries]),
            "mu_r": concat_arrays(wt.Float, [e["mu_r"] for e in material_entries]),
            "sigma_e": concat_arrays(wt.Float, [e["sigma_e"] for e in material_entries]),
            "specified": concat_arrays(wt.Bool, [e["specified"] for e in material_entries]),
            "structure_idx": concat_arrays(wt.Int32, [e["structure_idx_array"] for e in material_entries]),
            "has_specified_materials": bool(n_specified > 0),
            "n_specified_triangles": int(n_specified),
            "n_default_material_triangles": int(n_triangles - n_specified),
        }

    @staticmethod
    def _attach_material_data(scene) -> None:
        if scene.tri_data is None:
            return
        n = int(scene.tri_data["n_triangles"])
        mat = scene._triangle_material_data
        if mat is None:
            scene.tri_data.update({
                "material_eps_r": dr.full(wt.Float, MATERIAL_EPS_R_DEFAULT, n),
                "material_mu_r": dr.full(wt.Float, MATERIAL_MU_R_DEFAULT, n),
                "material_sigma_e": dr.full(wt.Float, MATERIAL_SIGMA_E_DEFAULT, n),
                "material_specified": dr.full(wt.Bool, False, n),
                "material_structure_idx": dr.full(wt.Int32, -1, n),
                "material_has_specified_materials": False,
                "material_n_specified_triangles": 0,
                "material_n_default_material_triangles": int(n),
            })
            return
        scene.tri_data.update({
            "material_eps_r": mat["eps_r"],
            "material_mu_r": mat["mu_r"],
            "material_sigma_e": mat["sigma_e"],
            "material_specified": mat["specified"],
            "material_structure_idx": mat["structure_idx"],
            "material_has_specified_materials": bool(mat["has_specified_materials"]),
            "material_n_specified_triangles": int(mat["n_specified_triangles"]),
            "material_n_default_material_triangles": int(mat["n_default_material_triangles"]),
        })

    # ------------------------------------------------------------------
    # Surface groups (coplanar triangle grouping via union-find)
    # ------------------------------------------------------------------

    @staticmethod
    def _attach_surface_data(scene) -> None:
        if scene.tri_data is None:
            return
        sd = scene._triangle_surface_data
        if sd is None:
            scene.tri_data.update({
                "surface_group_id": dr.zeros(wt.Int32, 0),
                "surface_canonical_prim": dr.zeros(wt.Int32, 0),
                "surface_group_size": dr.zeros(wt.UInt32, 0),
                "surface_group_members": dr.zeros(wt.Int32, 0),
                "surface_group_size_by_group": dr.zeros(wt.UInt32, 0),
                "surface_group_members_by_group": dr.zeros(wt.Int32, 0),
                "surface_group_count": 0,
                "surface_max_group_size": 0,
                "surface_edge_size": dr.zeros(wt.UInt32, 0),
                "surface_edge_indices": dr.zeros(wt.Int32, 0),
                "surface_max_edge_count": 0,
            })
            return
        scene.tri_data.update({
            "surface_group_id": sd["group_id"],
            "surface_canonical_prim": sd["canonical_prim"],
            "surface_group_size": sd["group_size"],
            "surface_group_members": sd["group_members"],
            "surface_group_size_by_group": sd["group_size_by_group"],
            "surface_group_members_by_group": sd["group_members_by_group"],
            "surface_group_count": int(sd["group_count"]),
            "surface_max_group_size": int(sd["max_group_size"]),
            "surface_edge_size": sd["surface_edge_size"],
            "surface_edge_indices": sd["surface_edge_indices"],
            "surface_max_edge_count": int(sd["max_surface_edge_count"]),
        })

    @staticmethod
    def _refresh_surface_groups(scene) -> None:
        rayd_scene = scene._rayd_scene
        global_geometry = None if rayd_scene is None else rayd_scene.global_geometry()
        n_triangles = 0 if global_geometry is None else int(global_geometry.face_count())
        if n_triangles == 0:
            scene._face_normals = None
            scene._triangle_surface_groups = ()
            scene._triangle_surface_group_by_triangle = dr.zeros(wt.Int32, 0)
            scene._triangle_surface_edge_groups = ()
            scene._triangle_surface_data = None
            SceneBuilder._attach_surface_data(scene)
            return

        vertices, faces = SceneBuilder._global_geometry_buffers(global_geometry)
        face_normals = wt.Vector3f(global_geometry.face_normal)
        scene._face_normals = face_normals

        edge_topology = rayd_scene.edge_topology()
        face0_i32 = wt.Int32(edge_topology.face0_global)
        face1_i32 = wt.Int32(edge_topology.face1_global)
        valid_adjacent = (face0_i32 >= 0) & (face1_i32 >= 0)
        coplanar = triangles_are_coplanar_mask(
            face0_i32, face1_i32, face_normals, vertices, faces,
            shared_v0=wt.Int32(edge_topology.v0_global),
            shared_v1=wt.Int32(edge_topology.v1_global),
            normal_cos_tol=SURFACE_GROUP_NORMAL_COS_TOL,
            plane_tol=SURFACE_GROUP_PLANE_TOL,
        )
        link_mask = valid_adjacent & coplanar

        labels = SceneBuilder._surface_component_labels(face0_i32, face1_i32, link_mask, n_triangles)
        tri_idx = dr.arange(wt.UInt32, n_triangles)
        root_mask = labels == tri_idx
        root_counts = dr.select(root_mask, wt.UInt32(1), wt.UInt32(0))
        root_prefix = dr.prefix_reduce(dr.ReduceOp.Add, root_counts, exclusive=False)
        root_group_id = root_prefix - wt.UInt32(1)
        group_id = dr.gather(wt.UInt32, root_group_id, labels)

        group_sizes = dr.zeros(wt.UInt32, n_triangles)
        dr.scatter_reduce(dr.ReduceOp.Add, group_sizes, wt.UInt32(1), labels)
        group_size = dr.gather(wt.UInt32, group_sizes, labels)
        max_size = int(scalar(dr.max(group_sizes))) if n_triangles > 0 else 0
        n_groups = int(scalar(dr.sum(root_counts))) if n_triangles > 0 else 0

        if n_groups > 0 and max_size > 0:
            group_member_slots = dr.zeros(wt.UInt32, n_groups)
            compact_members = dr.full(wt.Int32, -1, n_groups * max_size)
            compact_index = group_id * wt.UInt32(max_size) + dr.scatter_inc(group_member_slots, group_id)
            dr.scatter(compact_members, wt.Int32(tri_idx), compact_index)

            row_base = dr.repeat(group_id * wt.UInt32(max_size), max_size)
            row_slot = dr.tile(dr.arange(wt.UInt32, max_size), n_triangles)
            flat_members = dr.gather(wt.Int32, compact_members, row_base + row_slot)
        else:
            flat_members = dr.zeros(wt.Int32, 0)

        scene._triangle_surface_groups = ()
        scene._triangle_surface_group_by_triangle = wt.Int32(group_id)
        scene._triangle_surface_edge_groups = ()
        scene._triangle_surface_data = {
            "group_id": wt.Int32(group_id),
            "canonical_prim": wt.Int32(labels),
            "group_size": group_size,
            "group_members": flat_members,
            "group_size_by_group": group_member_slots if n_groups > 0 else dr.zeros(wt.UInt32, 0),
            "group_members_by_group": compact_members if n_groups > 0 and max_size > 0 else dr.zeros(wt.Int32, 0),
            "group_count": int(n_groups),
            "max_group_size": int(max_size),
            "surface_edge_size": dr.zeros(wt.UInt32, 0),
            "surface_edge_indices": dr.zeros(wt.Int32, 0),
            "max_surface_edge_count": 0,
        }
        SceneBuilder._attach_surface_data(scene)

    @staticmethod
    def _surface_component_labels(face0_i32: wt.Int32, face1_i32: wt.Int32, active: wt.Bool, n_triangles: int) -> wt.UInt32:
        """Compute minimum-index component labels for the coplanar surface graph."""
        labels = dr.arange(wt.UInt32, n_triangles)
        if n_triangles <= 1 or not bool(dr.any(active)):
            return labels

        face0 = wt.UInt32(dr.select(active, face0_i32, wt.Int32(0)))
        face1 = wt.UInt32(dr.select(active, face1_i32, wt.Int32(0)))
        max_steps = max(SceneBuilder._SURFACE_LABEL_PROPAGATION_STEPS, n_triangles.bit_length() + 2)

        for _ in range(max_steps):
            prev = labels
            mins = wt.UInt32(labels)
            label_min = dr.minimum(dr.gather(wt.UInt32, labels, face0), dr.gather(wt.UInt32, labels, face1))
            dr.scatter_reduce(dr.ReduceOp.Min, mins, label_min, face0, active)
            dr.scatter_reduce(dr.ReduceOp.Min, mins, label_min, face1, active)
            labels = dr.gather(wt.UInt32, mins, mins)
            labels = dr.gather(wt.UInt32, labels, labels)
            if not bool(dr.any(labels != prev)):
                break

        return labels

    # ------------------------------------------------------------------
    # Triangle data preloading
    # ------------------------------------------------------------------

    @staticmethod
    def _global_geometry_buffers(global_geometry):
        if global_geometry is None:
            return wt.Point3f(), wt.Vector3u()
        return wt.Point3f(global_geometry.vertices), wt.Vector3u(global_geometry.faces)

    @staticmethod
    def _preload_triangle_data(scene) -> None:
        n_triangles = SceneBuilder._total_triangles(scene._structure_meshes) if scene._structure_meshes else 0
        if n_triangles == 0:
            scene.tri_data = None
            return
        scene.tri_data = {"n_triangles": n_triangles}
        SceneBuilder._attach_material_data(scene)
        SceneBuilder._attach_surface_data(scene)

    # ------------------------------------------------------------------
    # Edge runtime (wedge pipeline)
    # ------------------------------------------------------------------

    @staticmethod
    def _clear_edge_runtime(scene) -> None:
        scene._wedge_geometry = None
        scene._wedge_selection = None
        scene._wedge_pack = None
        scene._wedge_tri_map = None
        scene._edge_runtime_dirty = False
        scene._triangle_surface_edge_groups = ()
        sd = scene._triangle_surface_data
        if sd is not None:
            sd["surface_edge_size"] = dr.zeros(wt.UInt32, 0)
            sd["surface_edge_indices"] = dr.zeros(wt.Int32, 0)
            sd["max_surface_edge_count"] = 0

    @staticmethod
    def _wedge_config(edge_policy: EdgePolicy) -> WedgeConfig:
        return WedgeConfig(
            boundary_policy=edge_policy.boundary_edge_policy,
            vertical_only=edge_policy.vertical_only,
            vertical_ratio=edge_policy.vertical_ratio,
            min_wedge_n=1.0 + SMALL_EPS,
            epsilon=EPS,
        )

    @staticmethod
    def _preload_edge_runtime(scene, edge_policy: EdgePolicy | None = None) -> None:
        edge_policy = DEFAULT_EDGE_POLICY if edge_policy is None else edge_policy
        scene._edge_policy = edge_policy
        scene._edge_policy_cache_key = edge_policy.cache_key
        scene._edge_cache.clear()
        scene._edge_data_cache.clear()
        rayd_scene = scene._rayd_scene
        if rayd_scene is None:
            SceneBuilder._clear_edge_runtime(scene)
            return

        edge_info = rayd_scene.edge_info()
        edge_topology = rayd_scene.edge_topology()
        if int(dr.width(edge_info.global_edge_id)) == 0:
            SceneBuilder._clear_edge_runtime(scene)
            return

        config = SceneBuilder._wedge_config(edge_policy)
        geometry = WedgeOps.build_geometry(edge_info, edge_topology, config)
        selection = WedgeOps.select(geometry, config)
        scene._wedge_geometry = geometry
        scene._wedge_selection = selection

        if selection.size() == 0:
            scene._wedge_pack = scene._wedge_tri_map = None
            scene._edge_runtime_dirty = False
            return

        scene._wedge_pack = WedgeOps.pack(selection, WedgeOps.build_midpoint_anchors(selection))
        SceneBuilder._build_triangle_edge_data(scene, selection)
        scene._edge_runtime_dirty = False

    @staticmethod
    def _build_triangle_edge_data(scene, selection) -> None:
        n_triangles = SceneBuilder._total_triangles(scene._structure_meshes) if scene._structure_meshes else 0
        if n_triangles == 0:
            scene._wedge_tri_map = None
            return

        tri_map = WedgeOps.build_triangle_map(selection, n_triangles)
        scene._wedge_tri_map = tri_map

        if scene._triangle_surface_data is None:
            return
        SceneBuilder._compute_surface_edge_groups(scene, tri_map, n_triangles)

    @staticmethod
    def _compute_surface_edge_groups(scene, tri_map, n_triangles: int) -> None:
        sd = scene._triangle_surface_data
        if sd is None:
            scene._triangle_surface_edge_groups = ()
            return

        group_id = wt.UInt32(sd["group_id"]) if tri_map.n_wedges > 0 else None
        n_groups = (int(scalar(dr.max(group_id))) + 1) if (tri_map.n_wedges > 0 and n_triangles > 0) else 0
        if n_groups <= 0:
            scene._triangle_surface_edge_groups = ()
            sd["surface_edge_size"] = dr.zeros(wt.UInt32, 0)
            sd["surface_edge_indices"] = dr.zeros(wt.Int32, 0)
            sd["max_surface_edge_count"] = 0
            SceneBuilder._attach_surface_data(scene)
            return

        selection = scene._wedge_selection
        selected_edge_slot = dr.arange(wt.Int32, tri_map.n_wedges)
        face0 = wt.Int32(dr.gather(type(selection.geometry.face0), selection.geometry.face0, selection.selected_idx))
        face1 = wt.Int32(dr.gather(type(selection.geometry.face1), selection.geometry.face1, selection.selected_idx))

        valid0 = face0 >= 0
        valid1 = face1 >= 0
        safe_face0 = wt.UInt32(dr.select(valid0, face0, wt.Int32(0)))
        safe_face1 = wt.UInt32(dr.select(valid1, face1, wt.Int32(0)))
        group0 = dr.gather(wt.UInt32, group_id, safe_face0)
        group1 = dr.gather(wt.UInt32, group_id, safe_face1)
        valid1 = valid1 & (~valid0 | (group1 != group0))

        per_group_size = dr.zeros(wt.UInt32, n_groups)
        one = wt.UInt32(1)
        dr.scatter_reduce(dr.ReduceOp.Add, per_group_size, one, group0, valid0)
        dr.scatter_reduce(dr.ReduceOp.Add, per_group_size, one, group1, valid1)
        max_edge_count = int(scalar(dr.max(per_group_size))) if n_groups > 0 else 0

        if max_edge_count > 0:
            group_slots = dr.zeros(wt.UInt32, n_groups)
            compact_edges = dr.full(wt.Int32, -1, n_groups * max_edge_count)
            dr.scatter(compact_edges, selected_edge_slot, group0 * wt.UInt32(max_edge_count) + dr.scatter_inc(group_slots, group0, valid0), valid0)
            dr.scatter(compact_edges, selected_edge_slot, group1 * wt.UInt32(max_edge_count) + dr.scatter_inc(group_slots, group1, valid1), valid1)

            row_base = dr.repeat(group_id * wt.UInt32(max_edge_count), max_edge_count)
            row_slot = dr.tile(dr.arange(wt.UInt32, max_edge_count), n_triangles)
            surface_edge_indices = dr.gather(wt.Int32, compact_edges, row_base + row_slot)
        else:
            surface_edge_indices = dr.zeros(wt.Int32, 0)

        scene._triangle_surface_edge_groups = ()
        sd["surface_edge_size"] = dr.gather(wt.UInt32, per_group_size, group_id)
        sd["surface_edge_indices"] = surface_edge_indices
        sd["max_surface_edge_count"] = int(max_edge_count)
        SceneBuilder._attach_surface_data(scene)

    @staticmethod
    def ensure_edge_runtime(scene, edge_policy: EdgePolicy | None = None) -> None:
        edge_policy = getattr(scene, "_edge_policy", DEFAULT_EDGE_POLICY) if edge_policy is None else edge_policy
        if scene._edge_runtime_dirty or scene._edge_policy_cache_key != edge_policy.cache_key:
            SceneBuilder._preload_edge_runtime(scene, edge_policy=edge_policy)

import os
import sys
import unittest
import importlib
import math
from pathlib import Path

import torch


RAYDI_ROOT = Path(r"E:\Code\RayDi")


def _load_backends():
    sys.path.insert(0, str(RAYDI_ROOT))
    import rayd as dr_backend
    import raydtorch as rt

    cuda = importlib.import_module("dr" + "jit.cuda")
    return dr_backend, rt, cuda


def _torch_scene(rt):
    verts = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        device="cuda",
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
    scene = rt.Scene()
    scene.add_mesh(rt.Mesh(verts, faces))
    scene.build()
    return scene


def _rayd_scene(dr_backend, cuda):
    verts = cuda.Array3f([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0])
    faces = cuda.Array3i([0], [1], [2])
    scene = dr_backend.Scene()
    scene.add_mesh(dr_backend.Mesh(verts, faces))
    scene.build()
    return scene


def _torch_dfr_scene(rt):
    verts = torch.tensor(
        [[-1.0, -1.0, 10.0], [1.0, -1.0, 10.0], [-1.0, 1.0, 10.0]],
        device="cuda",
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
    scene = rt.Scene()
    scene.add_mesh(rt.Mesh(verts, faces))
    scene.build()
    return scene


def _rayd_dfr_scene(dr_backend, cuda):
    verts = cuda.Array3f([-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0], [10.0, 10.0, 10.0])
    faces = cuda.Array3i([0], [1], [2])
    scene = dr_backend.Scene()
    scene.add_mesh(dr_backend.Mesh(verts, faces))
    scene.build()
    return scene


def _torch_dfr_states(rt, src_power: float):
    device = torch.device("cuda")
    return rt.DfrStates(
        edge_index=torch.tensor([0], device=device, dtype=torch.int32),
        edge_pos=torch.tensor([[0.0, 0.0, 0.0]], device=device, dtype=torch.float32),
        edge_dir=torch.tensor([[1.0, 0.0, 0.0]], device=device, dtype=torch.float32),
        edge_t_min=torch.tensor([-0.5], device=device, dtype=torch.float32),
        edge_t_max=torch.tensor([0.5], device=device, dtype=torch.float32),
        n0=torch.tensor([[0.0, 1.0, 0.0]], device=device, dtype=torch.float32),
        n1=torch.tensor([[0.0, -1.0, 0.0]], device=device, dtype=torch.float32),
        prim0=torch.tensor([-1], device=device, dtype=torch.int32),
        prim1=torch.tensor([-1], device=device, dtype=torch.int32),
        exterior_angle=torch.tensor([1.5 * math.pi], device=device, dtype=torch.float32),
        src=torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=torch.float32),
        src_power=torch.tensor([src_power], device=device, dtype=torch.float32),
        wi=torch.tensor([[0.0, 0.0, -1.0]], device=device, dtype=torch.float32),
        d0=torch.tensor([[0.0, 0.0, -1.0]], device=device, dtype=torch.float32),
        count=1,
    )


def _rayd_dfr_states(dr_backend, cuda, src_power: float):
    states = dr_backend.DfrStates()
    states.count = 1
    states.edge_index = cuda.Int([0])
    states.edge_pos = cuda.Array3f([0.0], [0.0], [0.0])
    states.edge_dir = cuda.Array3f([1.0], [0.0], [0.0])
    states.edge_t_min = cuda.Float([-0.5])
    states.edge_t_max = cuda.Float([0.5])
    states.n0 = cuda.Array3f([0.0], [1.0], [0.0])
    states.n1 = cuda.Array3f([0.0], [-1.0], [0.0])
    states.prim0 = cuda.Int([-1])
    states.prim1 = cuda.Int([-1])
    states.exterior_angle = cuda.Float([1.5 * math.pi])
    states.src = cuda.Array3f([0.0], [0.0], [1.0])
    states.src_power = cuda.Float([src_power])
    states.wi = cuda.Array3f([0.0], [0.0], [-1.0])
    states.d0 = cuda.Array3f([0.0], [0.0], [-1.0])
    states.prefix_depth = cuda.Int([0])
    return states


def _torch_recursive_dfr_states(rt, count: int = 1):
    device = torch.device("cuda")
    y = [0.5] if count == 1 else [0.5, 1.0]
    return rt.DfrStates(
        edge_index=torch.tensor([1, 2][:count], device=device, dtype=torch.int32),
        edge_pos=torch.tensor([[0.0, yy, 0.0] for yy in y], device=device, dtype=torch.float32),
        edge_dir=torch.tensor([[1.0, 0.0, 0.0]] * count, device=device, dtype=torch.float32),
        edge_t_min=torch.tensor([-0.5] * count, device=device, dtype=torch.float32),
        edge_t_max=torch.tensor([0.5] * count, device=device, dtype=torch.float32),
        n0=torch.tensor([[0.0, 1.0, 0.0]] * count, device=device, dtype=torch.float32),
        n1=torch.tensor([[0.0, -1.0, 0.0]] * count, device=device, dtype=torch.float32),
        prim0=torch.tensor([-1] * count, device=device, dtype=torch.int32),
        prim1=torch.tensor([-1] * count, device=device, dtype=torch.int32),
        exterior_angle=torch.tensor([1.5 * math.pi] * count, device=device, dtype=torch.float32),
        src=torch.tensor([[0.0, 0.0, 1.0]] * count, device=device, dtype=torch.float32),
        src_power=torch.tensor([1.0] * count, device=device, dtype=torch.float32),
        wi=torch.tensor([[0.0, 1.0, 0.0]] * count, device=device, dtype=torch.float32),
        d0=torch.tensor([[0.0, 0.0, -1.0]] * count, device=device, dtype=torch.float32),
        count=count,
    )


def _rayd_recursive_dfr_states(dr_backend, cuda, count: int = 1):
    states = dr_backend.DfrStates()
    states.count = count
    states.edge_index = cuda.Int([1, 2][:count])
    states.edge_pos = cuda.Array3f([0.0] * count, ([0.5] if count == 1 else [0.5, 1.0]), [0.0] * count)
    states.edge_dir = cuda.Array3f([1.0] * count, [0.0] * count, [0.0] * count)
    states.edge_t_min = cuda.Float([-0.5] * count)
    states.edge_t_max = cuda.Float([0.5] * count)
    states.n0 = cuda.Array3f([0.0] * count, [1.0] * count, [0.0] * count)
    states.n1 = cuda.Array3f([0.0] * count, [-1.0] * count, [0.0] * count)
    states.prim0 = cuda.Int([-1] * count)
    states.prim1 = cuda.Int([-1] * count)
    states.exterior_angle = cuda.Float([1.5 * math.pi] * count)
    states.src = cuda.Array3f([0.0] * count, [0.0] * count, [1.0] * count)
    states.src_power = cuda.Float([1.0] * count)
    states.wi = cuda.Array3f([0.0] * count, [1.0] * count, [0.0] * count)
    states.d0 = cuda.Array3f([0.0] * count, [0.0] * count, [-1.0] * count)
    states.prefix_depth = cuda.Int([0] * count)
    return states


def _torch_dfr_material(rt):
    device = torch.device("cuda")
    return rt.DfrMaterial(
        eta_r=torch.tensor([4.0], device=device, dtype=torch.float32),
        sigma=torch.tensor([0.0], device=device, dtype=torch.float32),
        mu_r=torch.tensor([1.0], device=device, dtype=torch.float32),
        gain=torch.tensor([1.0], device=device, dtype=torch.float32),
        valid=torch.tensor([True], device=device, dtype=torch.bool),
    )


def _rayd_dfr_material(dr_backend, cuda):
    material = dr_backend.DfrMaterial()
    material.eta_r = cuda.Float([4.0])
    material.sigma = cuda.Float([0.0])
    material.mu_r = cuda.Float([1.0])
    material.gain = cuda.Float([1.0])
    material.valid = cuda.Bool([True])
    return material


def _torch_dfr_grid(rt, axis: int = 2, position: float = -1.0):
    return rt.DfrGrid(
        axis=axis,
        position=position,
        coord0_min=-1.0,
        coord0_max=1.0,
        coord1_min=-1.0,
        coord1_max=1.0,
        resolution0=1,
        resolution1=1,
        cell_area=4.0,
    )


def _rayd_dfr_grid(dr_backend, axis: int = 2, position: float = -1.0):
    grid = dr_backend.DfrGrid()
    grid.axis = axis
    grid.position = position
    grid.coord0_min = -1.0
    grid.coord0_max = 1.0
    grid.coord1_min = -1.0
    grid.coord1_max = 1.0
    grid.resolution0 = 1
    grid.resolution1 = 1
    grid.cell_area = 4.0
    return grid


def _torch_suffix_scene(rt):
    verts = torch.tensor(
        [[-2.0, 0.0, -2.0], [2.0, 0.0, -2.0], [-2.0, 0.0, 2.0]],
        device="cuda",
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
    scene = rt.Scene()
    scene.add_mesh(rt.Mesh(verts, faces))
    scene.build()
    return scene


def _rayd_suffix_scene(dr_backend, cuda):
    verts = cuda.Array3f([-2.0, 2.0, -2.0], [0.0, 0.0, 0.0], [-2.0, -2.0, 2.0])
    scene = dr_backend.Scene()
    scene.add_mesh(dr_backend.Mesh(verts, cuda.Array3i([0], [1], [2])))
    scene.build()
    return scene


def _torch_suffix_states(rt):
    device = torch.device("cuda")
    return rt.DfrStates(
        edge_index=torch.tensor([0], device=device, dtype=torch.int32),
        edge_pos=torch.tensor([[0.0, -1.0, 0.0]], device=device, dtype=torch.float32),
        edge_dir=torch.tensor([[1.0, 0.0, 0.0]], device=device, dtype=torch.float32),
        edge_t_min=torch.tensor([-0.25], device=device, dtype=torch.float32),
        edge_t_max=torch.tensor([0.25], device=device, dtype=torch.float32),
        n0=torch.tensor([[0.0, 1.0, 0.0]], device=device, dtype=torch.float32),
        n1=torch.tensor([[0.0, -1.0, 0.0]], device=device, dtype=torch.float32),
        prim0=torch.tensor([0], device=device, dtype=torch.int32),
        prim1=torch.tensor([0], device=device, dtype=torch.int32),
        exterior_angle=torch.tensor([1.5 * math.pi], device=device, dtype=torch.float32),
        src=torch.tensor([[0.0, -1.0, 1.0]], device=device, dtype=torch.float32),
        src_power=torch.tensor([1.0], device=device, dtype=torch.float32),
        wi=torch.tensor([[0.0, 0.0, -1.0]], device=device, dtype=torch.float32),
        d0=torch.tensor([[0.0, 0.0, -1.0]], device=device, dtype=torch.float32),
        count=1,
    )


def _rayd_suffix_states(dr_backend, cuda):
    states = dr_backend.DfrStates()
    states.count = 1
    states.edge_index = cuda.Int([0])
    states.edge_pos = cuda.Array3f([0.0], [-1.0], [0.0])
    states.edge_dir = cuda.Array3f([1.0], [0.0], [0.0])
    states.edge_t_min = cuda.Float([-0.25])
    states.edge_t_max = cuda.Float([0.25])
    states.n0 = cuda.Array3f([0.0], [1.0], [0.0])
    states.n1 = cuda.Array3f([0.0], [-1.0], [0.0])
    states.prim0 = cuda.Int([0])
    states.prim1 = cuda.Int([0])
    states.exterior_angle = cuda.Float([1.5 * math.pi])
    states.src = cuda.Array3f([0.0], [-1.0], [1.0])
    states.src_power = cuda.Float([1.0])
    states.wi = cuda.Array3f([0.0], [0.0], [-1.0])
    states.d0 = cuda.Array3f([0.0], [0.0], [-1.0])
    states.prefix_depth = cuda.Int([0])
    return states


@unittest.skipUnless(torch.cuda.is_available(), "CUDA torch is required")
@unittest.skipUnless(os.environ.get("RAYDTORCH_RUN_DR_JIT_PARITY") == "1", "external RayDi parity is opt-in")
class DrJitParityTests(unittest.TestCase):
    def test_intersect_forward_matches_external_baseline_case(self):
        dr_backend, rt, cuda = _load_backends()
        scene_t = _torch_scene(rt)
        ray_t = rt.Ray(
            torch.tensor([[0.25, 0.25, -1.0]], device="cuda", dtype=torch.float32),
            torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32),
        )
        out_t = scene_t.intersect(ray_t)

        scene_d = _rayd_scene(dr_backend, cuda)
        ray_d = dr_backend.Ray(cuda.Array3f([0.25], [0.25], [-1.0]), cuda.Array3f([0.0], [0.0], [1.0]))
        out_d = scene_d.intersect(ray_d)
        self.assertAlmostEqual(float(out_t.t[0].item()), float(out_d.t[0]), places=5)

    def test_nearest_edge_point_forward_matches_external_baseline_case(self):
        dr_backend, rt, cuda = _load_backends()
        scene_t = _torch_scene(rt)
        out_t = scene_t.nearest_edge(torch.tensor([[0.25, -0.2, 0.0]], device="cuda", dtype=torch.float32))

        scene_d = _rayd_scene(dr_backend, cuda)
        out_d = scene_d.nearest_edge(cuda.Array3f([0.25], [-0.2], [0.0]))
        self.assertAlmostEqual(float(out_t.distance[0].item()), float(out_d.distance[0]), places=5)
        self.assertAlmostEqual(float(out_t.edge_t[0].item()), float(out_d.edge_t[0]), places=5)
        self.assertEqual(int(out_t.edge_id[0].item()), int(out_d.edge_id[0]))

    def test_visibility_forward_matches_external_baseline_case(self):
        dr_backend, rt, cuda = _load_backends()
        scene_t = _torch_scene(rt)
        start_t = torch.tensor([[0.25, 0.25, -1.0]], device="cuda", dtype=torch.float32)
        end_t = torch.tensor([[0.25, 0.25, 1.0]], device="cuda", dtype=torch.float32)
        out_t = scene_t.visible(start_t, end_t)

        scene_d = _rayd_scene(dr_backend, cuda)
        out_d = scene_d.visible(cuda.Array3f([0.25], [0.25], [-1.0]), cuda.Array3f([0.25], [0.25], [1.0]))
        self.assertEqual(bool(out_t[0].item()), bool(out_d.visible[0]))

    def test_intersect_forward_matches_external_multi_mesh_global_ids(self):
        dr_backend, rt, cuda = _load_backends()
        verts0 = torch.tensor([[0.0, 0.0, 10.0], [1.0, 0.0, 10.0], [0.0, 1.0, 10.0]], device="cuda", dtype=torch.float32)
        verts1 = torch.tensor([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [0.0, 1.0, 2.0]], device="cuda", dtype=torch.float32)
        faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
        scene_t = rt.Scene()
        scene_t.add_mesh(rt.Mesh(verts0, faces))
        scene_t.add_mesh(rt.Mesh(verts1, faces))
        scene_t.build()
        ray_t = rt.Ray(torch.tensor([[0.25, 0.25, 0.0]], device="cuda", dtype=torch.float32), torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32))
        out_t = scene_t.intersect(ray_t)

        scene_d = dr_backend.Scene()
        scene_d.add_mesh(dr_backend.Mesh(cuda.Array3f([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [10.0, 10.0, 10.0]), cuda.Array3i([0], [1], [2])))
        scene_d.add_mesh(dr_backend.Mesh(cuda.Array3f([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [2.0, 2.0, 2.0]), cuda.Array3i([0], [1], [2])))
        scene_d.build()
        out_d = scene_d.intersect(dr_backend.Ray(cuda.Array3f([0.25], [0.25], [0.0]), cuda.Array3f([0.0], [0.0], [1.0])))

        self.assertAlmostEqual(float(out_t.t[0].item()), float(out_d.t[0]), places=5)
        self.assertEqual(int(out_t.shape_id[0].item()), int(out_d.shape_id[0]))
        self.assertEqual(int(out_t.local_prim_id[0].item()), int(out_d.local_prim_id[0]))
        self.assertEqual(int(out_t.global_prim_id[0].item()), int(out_d.global_prim_id[0]))

    def test_reflection_trace_forward_matches_external_baseline_case(self):
        dr_backend, rt, cuda = _load_backends()
        scene_t = _torch_scene(rt)
        ray_t = rt.Ray(
            torch.tensor([[0.25, 0.25, -1.0]], device="cuda", dtype=torch.float32),
            torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32),
        )
        out_t = scene_t.trace_reflections(ray_t, max_bounces=1)

        scene_d = _rayd_scene(dr_backend, cuda)
        ray_d = dr_backend.Ray(cuda.Array3f([0.25], [0.25], [-1.0]), cuda.Array3f([0.0], [0.0], [1.0]))
        out_d = scene_d.trace_reflections(ray_d, max_bounces=1, symbolic=False)
        self.assertEqual(bool(out_t.valid[0, 0].item()), bool(out_d.is_valid()[0]))
        self.assertAlmostEqual(float(out_t.t[0, 0].item()), float(out_d.t[0]), places=5)
        self.assertEqual(int(out_t.prim_ids[0, 0].item()), int(out_d.prim_ids[0]))

    def test_diffraction_paths_order1_matches_external_baseline_case(self):
        dr_backend, rt, cuda = _load_backends()
        dr = importlib.import_module("dr" + "jit")
        scene_t = _torch_dfr_scene(rt)
        states_t = _torch_dfr_states(rt, src_power=1.0)
        material_t = _torch_dfr_material(rt)
        tx_t = torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32)
        rx_t = torch.tensor([[0.0, 0.0, -1.0]], device="cuda", dtype=torch.float32)
        active_t = torch.tensor([True], device="cuda", dtype=torch.bool)
        out_t = scene_t.trace_dfr_paths(
            tx_positions=tx_t,
            rx_positions=rx_t,
            states=states_t,
            material=material_t,
            active=active_t,
            max_paths=4,
            wavelength=0.125,
        )

        scene_d = _rayd_dfr_scene(dr_backend, cuda)
        options = dr_backend.DfrPathOptions()
        options.wavelength = 0.125
        options.k = 50.26548245743669
        options.seed = 17
        options.max_order = 1
        options.max_paths = 4
        options.max_rx = 1
        options.strategy_mask = dr_backend.RAYD_DFR_DIRECT
        options.sample_count = 1
        options.return_geom = 1
        options.receiver_model = dr_backend.RAYD_DFR_MATCHED_ISO
        out_d = scene_d.trace_dfr_paths(
            cuda.Array3f([0.0], [0.0], [1.0]),
            cuda.Array3f([0.0], [0.0], [-1.0]),
            _rayd_dfr_states(dr_backend, cuda, src_power=1.0),
            _rayd_dfr_material(dr_backend, cuda),
            options,
            cuda.Bool([True]),
        )
        dr.eval(out_d.count, out_d.valid, out_d.rx_id, out_d.edge0, out_d.delay, out_d.field_x.real, out_d.field_x.imag)

        self.assertEqual(out_t.capacity, int(out_d.capacity))
        self.assertEqual(int(out_t.count[0].item()), int(out_d.count[0]))
        self.assertEqual(bool(out_t.valid[0].item()), bool(out_d.valid[0]))
        self.assertEqual(int(out_t.rx_id[0].item()), int(out_d.rx_id[0]))
        self.assertEqual(int(out_t.edge0[0].item()), int(out_d.edge0[0]))
        self.assertAlmostEqual(float(out_t.delay[0].item()), float(out_d.delay[0]), places=5)
        self.assertAlmostEqual(float(out_t.field_x_re[0].item()), float(out_d.field_x.real[0]), places=5)
        self.assertAlmostEqual(float(out_t.field_x_im[0].item()), float(out_d.field_x.imag[0]), places=5)

    def test_diffraction_accum_direct_matches_external_baseline_case(self):
        dr_backend, rt, cuda = _load_backends()
        dr = importlib.import_module("dr" + "jit")
        scene_t = _torch_dfr_scene(rt)
        states_t = _torch_dfr_states(rt, src_power=2.0)
        material_t = _torch_dfr_material(rt)
        grid_t = rt.DfrGrid(
            axis=2,
            position=-1.0,
            coord0_min=-1.0,
            coord0_max=1.0,
            coord1_min=-1.0,
            coord1_max=1.0,
            resolution0=1,
            resolution1=1,
            cell_area=4.0,
        )
        out_t = scene_t.accum_dfr_direct(
            states=states_t,
            grid=grid_t,
            material=material_t,
            wavelength=0.125,
            seed=17,
            direct_samples=64,
        )

        scene_d = _rayd_dfr_scene(dr_backend, cuda)
        grid_d = dr_backend.DfrGrid()
        grid_d.axis = 2
        grid_d.position = -1.0
        grid_d.coord0_min = -1.0
        grid_d.coord0_max = 1.0
        grid_d.coord1_min = -1.0
        grid_d.coord1_max = 1.0
        grid_d.resolution0 = 1
        grid_d.resolution1 = 1
        grid_d.cell_area = 4.0
        options = dr_backend.DfrOptions()
        options.wavelength = 0.125
        options.k = 50.26548245743669
        options.seed = 17
        options.samples = 64
        options.max_order = 1
        options.direct_samples = 64
        options.keller_samples = 0
        options.strategy_mask = dr_backend.RAYD_DFR_DIRECT
        options.sample_sequence = dr_backend.RAYD_DFR_HASH
        options.receiver_model = dr_backend.RAYD_DFR_MATCHED_ISO
        options.collect_edge_use = True
        options.collect_debug_counts = True
        out_d = scene_d.accum_dfr_direct(
            _rayd_dfr_states(dr_backend, cuda, src_power=2.0),
            grid_d,
            _rayd_dfr_material(dr_backend, cuda),
            options,
            True,
        )
        dr.eval(out_d.power, out_d.field_x.real, out_d.field_x.imag, out_d.direct_count, out_d.keller_count)

        self.assertEqual(out_t.grid_cell_count, int(out_d.grid_cell_count))
        self.assertAlmostEqual(float(out_t.power.flatten()[0].item()), float(out_d.power[0]), places=5)
        self.assertAlmostEqual(float(out_t.field_x_re.flatten()[0].item()), float(out_d.field_x.real[0]), places=5)
        self.assertAlmostEqual(float(out_t.field_x_im.flatten()[0].item()), float(out_d.field_x.imag[0]), places=5)
        self.assertEqual(int(out_t.direct_count.flatten()[0].item()), int(out_d.direct_count[0]))
        self.assertEqual(int(out_t.keller_count.flatten()[0].item()), int(out_d.keller_count[0]))

    def test_diffraction_accum_keller_matches_external_baseline_case(self):
        dr_backend, rt, cuda = _load_backends()
        dr = importlib.import_module("dr" + "jit")
        scene_t = _torch_dfr_scene(rt)
        grid_t = rt.DfrGrid(
            axis=2,
            position=-1.0,
            coord0_min=-1.0,
            coord0_max=1.0,
            coord1_min=-1.0,
            coord1_max=1.0,
            resolution0=1,
            resolution1=1,
            cell_area=4.0,
        )
        out_t = scene_t.accum_dfr_direct(
            states=_torch_dfr_states(rt, src_power=2.0),
            grid=grid_t,
            material=_torch_dfr_material(rt),
            wavelength=0.125,
            seed=23,
            direct_samples=0,
            keller_samples=64,
        )

        scene_d = _rayd_dfr_scene(dr_backend, cuda)
        grid_d = dr_backend.DfrGrid()
        grid_d.axis = 2
        grid_d.position = -1.0
        grid_d.coord0_min = -1.0
        grid_d.coord0_max = 1.0
        grid_d.coord1_min = -1.0
        grid_d.coord1_max = 1.0
        grid_d.resolution0 = 1
        grid_d.resolution1 = 1
        grid_d.cell_area = 4.0
        options = dr_backend.DfrOptions()
        options.wavelength = 0.125
        options.k = 50.26548245743669
        options.seed = 23
        options.samples = 64
        options.max_order = 1
        options.direct_samples = 0
        options.keller_samples = 64
        options.strategy_mask = dr_backend.RAYD_DFR_KELLER
        options.sample_sequence = dr_backend.RAYD_DFR_HASH
        options.receiver_model = dr_backend.RAYD_DFR_MATCHED_ISO
        options.collect_edge_use = True
        options.collect_debug_counts = True
        out_d = scene_d.accum_dfr_direct(
            _rayd_dfr_states(dr_backend, cuda, src_power=2.0),
            grid_d,
            _rayd_dfr_material(dr_backend, cuda),
            options,
            True,
        )
        dr.eval(out_d.power, out_d.field_x.real, out_d.field_x.imag, out_d.direct_count, out_d.keller_count)

        self.assertEqual(out_t.grid_cell_count, int(out_d.grid_cell_count))
        self.assertAlmostEqual(float(out_t.power.flatten()[0].item()), float(out_d.power[0]), places=5)
        self.assertAlmostEqual(float(out_t.field_x_re.flatten()[0].item()), float(out_d.field_x.real[0]), places=5)
        self.assertAlmostEqual(float(out_t.field_x_im.flatten()[0].item()), float(out_d.field_x.imag[0]), places=5)
        self.assertEqual(int(out_t.direct_count.flatten()[0].item()), int(out_d.direct_count[0]))
        self.assertEqual(int(out_t.keller_count.flatten()[0].item()), int(out_d.keller_count[0]))

    def test_diffraction_accum_suffix_matches_external_baseline_case(self):
        dr_backend, rt, cuda = _load_backends()
        dr = importlib.import_module("dr" + "jit")
        scene_t = _torch_suffix_scene(rt)
        out_t = scene_t.accum_dfr_direct(
            states=_torch_suffix_states(rt),
            grid=_torch_dfr_grid(rt, axis=1, position=-2.0),
            material=_torch_dfr_material(rt),
            wavelength=0.125,
            seed=41,
            direct_samples=0,
            keller_samples=0,
            suffix_samples=16,
        )

        scene_d = _rayd_suffix_scene(dr_backend, cuda)
        options = dr_backend.DfrOptions()
        options.wavelength = 0.125
        options.k = 50.26548245743669
        options.seed = 41
        options.samples = 16
        options.max_order = 1
        options.direct_samples = 0
        options.keller_samples = 0
        options.suffix_samples = 16
        options.strategy_mask = dr_backend.RAYD_DFR_SUFFIX_REFL
        options.sample_sequence = dr_backend.RAYD_DFR_HASH
        options.receiver_model = dr_backend.RAYD_DFR_MATCHED_ISO
        options.collect_debug_counts = True
        out_d = scene_d.accum_dfr_direct(
            _rayd_suffix_states(dr_backend, cuda),
            _rayd_dfr_grid(dr_backend, axis=1, position=-2.0),
            _rayd_dfr_material(dr_backend, cuda),
            options,
            cuda.Bool([True]),
        )
        dr.eval(out_d.power, out_d.direct_count, out_d.keller_count, out_d.suffix_count)

        self.assertAlmostEqual(float(out_t.power.flatten()[0].item()), float(out_d.power[0]), places=5)
        self.assertEqual(int(out_t.direct_count.flatten()[0].item()), int(out_d.direct_count[0]))
        self.assertEqual(int(out_t.keller_count.flatten()[0].item()), int(out_d.keller_count[0]))
        self.assertEqual(int(out_t.suffix_count.flatten()[0].item()), int(out_d.suffix_count[0]))

    def test_diffraction_accum_order2_chain_matches_external_baseline_case(self):
        dr_backend, rt, cuda = _load_backends()
        dr = importlib.import_module("dr" + "jit")
        scene_t = _torch_dfr_scene(rt)
        out_t = scene_t.accum_dfr(
            initial_states=_torch_dfr_states(rt, src_power=2.0),
            recursive_states=_torch_recursive_dfr_states(rt, count=1),
            grid=_torch_dfr_grid(rt),
            material=_torch_dfr_material(rt),
            wavelength=0.125,
            seed=41,
            direct_samples=32,
            keller_samples=256,
            max_order=2,
        )

        scene_d = _rayd_dfr_scene(dr_backend, cuda)
        options = dr_backend.DfrOptions()
        options.wavelength = 0.125
        options.k = 50.26548245743669
        options.seed = 41
        options.samples = 288
        options.max_order = 2
        options.direct_samples = 32
        options.keller_samples = 256
        options.strategy_mask = dr_backend.RAYD_DFR_DIRECT | dr_backend.RAYD_DFR_KELLER
        options.sample_sequence = dr_backend.RAYD_DFR_HASH
        options.receiver_model = dr_backend.RAYD_DFR_MATCHED_ISO
        options.collect_edge_use = True
        options.collect_debug_counts = True
        out_d = scene_d.accum_dfr(
            _rayd_dfr_states(dr_backend, cuda, src_power=2.0),
            _rayd_recursive_dfr_states(dr_backend, cuda, count=1),
            _rayd_dfr_grid(dr_backend),
            _rayd_dfr_material(dr_backend, cuda),
            options,
            True,
        )
        dr.eval(out_d.power, out_d.direct_count, out_d.keller_count, out_d.edge_uses)

        self.assertAlmostEqual(float(out_t.power.flatten()[0].item()), float(out_d.power[0]), places=5)
        self.assertEqual(int(out_t.direct_count.flatten()[0].item()), int(out_d.direct_count[0]))
        self.assertEqual(int(out_t.keller_count.flatten()[0].item()), int(out_d.keller_count[0]))
        self.assertEqual(int(out_t.edge_uses.flatten()[0].item()), int(out_d.edge_uses[0]))

    def test_diffraction_accum_order3_chain_matches_external_baseline_case(self):
        dr_backend, rt, cuda = _load_backends()
        dr = importlib.import_module("dr" + "jit")
        scene_t = _torch_dfr_scene(rt)
        out_t = scene_t.accum_dfr(
            initial_states=_torch_dfr_states(rt, src_power=2.0),
            recursive_states=_torch_recursive_dfr_states(rt, count=2),
            grid=_torch_dfr_grid(rt),
            material=_torch_dfr_material(rt),
            wavelength=0.125,
            seed=43,
            direct_samples=64,
            keller_samples=256,
            max_order=3,
        )

        scene_d = _rayd_dfr_scene(dr_backend, cuda)
        options = dr_backend.DfrOptions()
        options.wavelength = 0.125
        options.k = 50.26548245743669
        options.seed = 43
        options.samples = 320
        options.max_order = 3
        options.direct_samples = 64
        options.keller_samples = 256
        options.strategy_mask = dr_backend.RAYD_DFR_DIRECT | dr_backend.RAYD_DFR_KELLER
        options.sample_sequence = dr_backend.RAYD_DFR_HASH
        options.receiver_model = dr_backend.RAYD_DFR_MATCHED_ISO
        options.collect_edge_use = True
        options.collect_debug_counts = True
        out_d = scene_d.accum_dfr(
            _rayd_dfr_states(dr_backend, cuda, src_power=2.0),
            _rayd_recursive_dfr_states(dr_backend, cuda, count=2),
            _rayd_dfr_grid(dr_backend),
            _rayd_dfr_material(dr_backend, cuda),
            options,
            True,
        )
        dr.eval(out_d.power, out_d.direct_count, out_d.keller_count, out_d.edge_uses)

        self.assertAlmostEqual(float(out_t.power.flatten()[0].item()), float(out_d.power[0]), places=5)
        self.assertEqual(int(out_t.direct_count.flatten()[0].item()), int(out_d.direct_count[0]))
        self.assertEqual(int(out_t.keller_count.flatten()[0].item()), int(out_d.keller_count[0]))
        self.assertEqual(int(out_t.edge_uses.flatten()[0].item()), int(out_d.edge_uses[0]))

    def test_diffraction_coherent_direct_matches_external_baseline_case(self):
        dr_backend, rt, cuda = _load_backends()
        dr = importlib.import_module("dr" + "jit")
        scene_t = _torch_dfr_scene(rt)
        out_t = scene_t.accum_dfr_coherent_direct(
            states=_torch_dfr_states(rt, src_power=2.0),
            grid=_torch_dfr_grid(rt),
            material=_torch_dfr_material(rt),
            wavelength=0.125,
            select_diffraction_point=True,
            prefilter_visibility=True,
        )

        scene_d = _rayd_dfr_scene(dr_backend, cuda)
        options = dr_backend.DfrCoherentOptions()
        options.wavelength = 0.125
        options.k = 50.26548245743669
        options.max_order = 1
        options.receiver_model = dr_backend.RAYD_DFR_MATCHED_ISO
        options.select_diffraction_point = True
        options.prefilter_visibility = True
        options.collect_debug_counts = True
        out_d = scene_d.accum_dfr_coherent_direct(
            _rayd_dfr_states(dr_backend, cuda, src_power=2.0),
            _rayd_dfr_grid(dr_backend),
            _rayd_dfr_material(dr_backend, cuda),
            options,
            cuda.Bool([True]),
        )
        dr.eval(out_d.direct_field_x.real, out_d.direct_field_x.imag, out_d.direct_count)

        self.assertAlmostEqual(float(out_t.direct_field_x_re.flatten()[0].item()), float(out_d.direct_field_x.real[0]), delta=3.0e-5)
        self.assertAlmostEqual(float(out_t.direct_field_x_im.flatten()[0].item()), float(out_d.direct_field_x.imag[0]), delta=3.0e-5)
        self.assertEqual(int(out_t.direct_count.flatten()[0].item()), int(out_d.direct_count[0]))

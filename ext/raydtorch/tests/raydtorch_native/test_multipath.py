import unittest

import torch
import raydtorch as rt


@unittest.skipUnless(torch.cuda.is_available(), "CUDA torch is required")
class MultipathTests(unittest.TestCase):
    def test_visibility_returns_bool_tensor(self):
        verts = torch.tensor(
            [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
        )
        faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(verts, faces))
        scene.build()
        start = torch.tensor([[0.0, 0.0, -1.0]], device="cuda", dtype=torch.float32)
        end = torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32)
        visible = scene.visible(start, end)
        self.assertEqual(visible.dtype, torch.bool)
        self.assertFalse(bool(visible[0].item()))

    def test_single_reflection_t_has_gradient(self):
        verts = torch.tensor(
            [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(verts, faces))
        scene.build()
        ray = rt.Ray(
            torch.tensor([[0.0, 0.0, -1.0]], device="cuda", dtype=torch.float32),
            torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32),
        )
        chain = scene.trace_reflections(ray, max_bounces=1)
        chain.t.sum().backward()
        self.assertIsNotNone(verts.grad)
        self.assertGreater(float(verts.grad.abs().sum().item()), 0.0)

    def test_multi_mesh_reflection_gradient_routes_to_hit_mesh(self):
        verts0 = torch.tensor(
            [[10.0, -1.0, 0.0], [12.0, -1.0, 0.0], [10.0, 1.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        verts1 = torch.tensor(
            [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(verts0, faces))
        scene.add_mesh(rt.Mesh(verts1, faces))
        scene.build()
        ray = rt.Ray(
            torch.tensor([[0.0, 0.0, -1.0]], device="cuda", dtype=torch.float32),
            torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32),
        )
        scene.trace_reflections(ray, max_bounces=1).t.sum().backward()
        self.assertIsNotNone(verts0.grad)
        self.assertIsNotNone(verts1.grad)
        torch.testing.assert_close(verts0.grad, torch.zeros_like(verts0), atol=1e-5, rtol=1e-5)
        self.assertTrue(bool(torch.isfinite(verts1.grad).all().item()))

    def test_two_bounce_reflection_trace_fills_subsequent_bounces(self):
        verts = torch.tensor(
            [
                [0.0, -1.0, 0.0],
                [2.0, -1.0, 0.0],
                [0.0, 1.0, 0.0],
                [2.0, -1.0, 0.0],
                [2.0, 1.0, 0.0],
                [2.0, -1.0, 4.0],
            ],
            device="cuda",
            dtype=torch.float32,
        )
        faces = torch.tensor([[0, 1, 2], [3, 4, 5]], device="cuda", dtype=torch.int32)
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(verts, faces))
        scene.build()
        ray = rt.Ray(
            torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32),
            torch.tensor([[1.0, 0.0, -1.0]], device="cuda", dtype=torch.float32),
        )
        chain = scene.trace_reflections(ray, max_bounces=2)
        self.assertTrue(bool(chain.valid[0, 0].item()))
        self.assertTrue(bool(chain.valid[0, 1].item()))
        self.assertEqual([int(v) for v in chain.prim_ids[0].tolist()], [0, 1])
        torch.testing.assert_close(chain.t[0], torch.tensor([1.0, 1.0], device="cuda"), atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(
            chain.image_sources[0],
            torch.tensor([[0.0, 0.0, -1.0], [4.0, 0.0, -1.0]], device="cuda"),
            atol=1e-3,
            rtol=1e-3,
        )

    def test_two_bounce_reflection_second_t_has_gradient(self):
        verts = torch.tensor(
            [
                [0.0, -1.0, 0.0],
                [2.0, -1.0, 0.0],
                [0.0, 1.0, 0.0],
                [2.0, -1.0, 0.0],
                [2.0, 1.0, 0.0],
                [2.0, -1.0, 4.0],
            ],
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        faces = torch.tensor([[0, 1, 2], [3, 4, 5]], device="cuda", dtype=torch.int32)
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(verts, faces))
        scene.build()
        ray_o = torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32, requires_grad=True)
        ray_d = torch.tensor([[1.0, 0.0, -1.0]], device="cuda", dtype=torch.float32, requires_grad=True)
        chain = scene.trace_reflections(rt.Ray(ray_o, ray_d), max_bounces=2)
        chain.t[:, 1].sum().backward()
        self.assertIsNotNone(verts.grad)
        self.assertIsNotNone(ray_o.grad)
        self.assertIsNotNone(ray_d.grad)
        self.assertGreater(float(verts.grad.abs().sum().item()), 0.0)
        self.assertGreater(float(ray_o.grad.abs().sum().item()), 0.0)
        self.assertGreater(float(ray_d.grad.abs().sum().item()), 0.0)

    def test_two_bounce_reflection_second_t_vjp_matches_finite_difference(self):
        base_verts = torch.tensor(
            [
                [0.0, -1.0, 0.0],
                [2.0, -1.0, 0.0],
                [0.0, 1.0, 0.0],
                [2.0, -1.0, 0.0],
                [2.0, 1.0, 0.0],
                [2.0, -1.0, 4.0],
            ],
            device="cuda",
            dtype=torch.float32,
        )
        faces = torch.tensor([[0, 1, 2], [3, 4, 5]], device="cuda", dtype=torch.int32)
        ray_o = torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32)
        ray_d = torch.tensor([[1.0, 0.0, -1.0]], device="cuda", dtype=torch.float32)
        tangent = torch.zeros_like(base_verts)
        tangent[3:, 0] = 0.1

        def second_t(verts_value):
            scene = rt.Scene()
            scene.add_mesh(rt.Mesh(verts_value.contiguous(), faces))
            scene.build()
            return scene.trace_reflections(rt.Ray(ray_o, ray_d), max_bounces=2).t[0, 1]

        verts = base_verts.clone().detach().requires_grad_(True)
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(verts, faces))
        scene.build()
        scene.trace_reflections(rt.Ray(ray_o, ray_d), max_bounces=2).t[0, 1].backward()
        vjp_dot = (verts.grad * tangent).sum()
        eps = 1e-3
        fd = (second_t(base_verts + eps * tangent) - second_t(base_verts - eps * tangent)) / (2.0 * eps)
        torch.testing.assert_close(vjp_dot, fd, atol=2e-2, rtol=2e-2)

    def test_two_bounce_reflection_second_t_jvp_matches_finite_difference(self):
        base_verts = torch.tensor(
            [
                [0.0, -1.0, 0.0],
                [2.0, -1.0, 0.0],
                [0.0, 1.0, 0.0],
                [2.0, -1.0, 0.0],
                [2.0, 1.0, 0.0],
                [2.0, -1.0, 4.0],
            ],
            device="cuda",
            dtype=torch.float32,
        )
        faces = torch.tensor([[0, 1, 2], [3, 4, 5]], device="cuda", dtype=torch.int32)
        ray_o = torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32)
        ray_d = torch.tensor([[1.0, 0.0, -1.0]], device="cuda", dtype=torch.float32)
        tangent = torch.zeros_like(base_verts)
        tangent[3:, 0] = 0.1

        def fn(verts_value):
            scene = rt.Scene()
            scene.add_mesh(rt.Mesh(verts_value.contiguous(), faces))
            scene.build()
            return scene.trace_reflections(rt.Ray(ray_o, ray_d), max_bounces=2).t[:, 1]

        _primal, jvp = torch.func.jvp(fn, (base_verts,), (tangent,))
        eps = 1e-3
        fd = (fn(base_verts + eps * tangent) - fn(base_verts - eps * tangent)) / (2.0 * eps)
        torch.testing.assert_close(jvp, fd, atol=2e-2, rtol=2e-2)

    def test_reflection_epc_field_backward_reaches_vertices(self):
        verts = torch.tensor(
            [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(verts, faces))
        scene.build()
        source = torch.tensor([[0.0, 0.0, -1.0]], device="cuda", dtype=torch.float32)
        receiver = torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32)
        out = scene.trace_refl_epc_field(source, receiver, max_bounces=1)
        loss = out.field_real.sum() + out.field_imag.sum()
        loss.backward()
        self.assertIsNotNone(verts.grad)

    def test_reflection_dedup_native_binding_smoke(self):
        ray_count = 2
        max_bounces = 1
        slot_count = ray_count * max_bounces
        device = "cuda"
        bounce_count = torch.ones((ray_count,), device=device, dtype=torch.int32)
        shape_ids = torch.zeros((slot_count,), device=device, dtype=torch.int32)
        prim_ids = torch.zeros((slot_count,), device=device, dtype=torch.int32)
        t = torch.ones((slot_count,), device=device, dtype=torch.float32)
        zeros = torch.zeros((slot_count,), device=device, dtype=torch.float32)
        norm_z = torch.ones((slot_count,), device=device, dtype=torch.float32)
        out = rt._C.reflection_dedup_forward(
            bounce_count,
            shape_ids,
            prim_ids,
            t,
            zeros,
            zeros,
            zeros,
            zeros,
            zeros,
            zeros,
            zeros,
            norm_z,
            zeros,
            zeros,
            zeros,
            max_bounces,
            1e-5,
        )
        unique_count = int(out[0])
        discovery_count = out[-2]
        self.assertEqual(unique_count, 1)
        self.assertEqual(int(discovery_count[0].item()), 2)

    def test_reflection_accumulation_native_binding_smoke(self):
        verts = torch.tensor(
            [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
        )
        faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(verts, faces))
        scene.build()
        ray_o = torch.tensor([[0.0, 0.0, -1.0]], device="cuda", dtype=torch.float32)
        ray_d = torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32)
        ray_tmax = torch.tensor([2.0], device="cuda", dtype=torch.float32)
        active = torch.ones((1,), device="cuda", dtype=torch.bool)
        tx_pol = torch.tensor([[1.0, 0.0, 0.0]], device="cuda", dtype=torch.float32)
        out = rt._C.reflection_accumulation_forward(
            scene._native_handle,
            ray_o,
            ray_d,
            ray_tmax,
            active,
            ray_o,
            tx_pol,
            1,
            2,
            -1.0,
            -1.0,
            1.0,
            -1.0,
            1.0,
            4,
            4,
            1.0,
        )
        self.assertEqual(out[0].shape, (4, 4))
        self.assertEqual(out[-1].dtype, torch.int32)

    def test_legacy_dfr_direct_entrypoints_are_removed(self):
        scene = rt.Scene()
        verts = torch.tensor(
            [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
        )
        faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
        scene.add_mesh(rt.Mesh(verts, faces))
        scene.build()
        edge_pos = torch.tensor([[0.0, 0.0, 0.0]], device="cuda", dtype=torch.float32, requires_grad=True)
        edge_dir = torch.tensor([[1.0, 0.0, 0.0]], device="cuda", dtype=torch.float32, requires_grad=True)
        src = torch.tensor([[0.0, -1.0, 0.2]], device="cuda", dtype=torch.float32, requires_grad=True)
        self.assertFalse(hasattr(scene, "accum_dfr_legacy_direct"))
        self.assertFalse(hasattr(rt._C, "accum_dfr_direct_forward"))
        with self.assertRaises(TypeError):
            scene.accum_dfr_direct(edge_pos=edge_pos, edge_dir=edge_dir, src=src)

    def test_diffraction_paths_order1_native_binding_smoke(self):
        verts = torch.tensor(
            [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
        )
        faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(verts, faces))
        scene.build()

        tx_pos = torch.tensor([[0.0, -1.0, 0.25]], device="cuda", dtype=torch.float32)
        rx_pos = torch.tensor([[0.0, 1.0, 0.25]], device="cuda", dtype=torch.float32)
        state_edge_index = torch.tensor([0], device="cuda", dtype=torch.int32)
        state_edge_pos = torch.tensor([[0.0, 0.0, 0.0]], device="cuda", dtype=torch.float32)
        state_edge_dir = torch.tensor([[1.0, 0.0, 0.0]], device="cuda", dtype=torch.float32)
        state_edge_t_min = torch.tensor([-1.0], device="cuda", dtype=torch.float32)
        state_edge_t_max = torch.tensor([1.0], device="cuda", dtype=torch.float32)
        state_n0 = torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32)
        state_n1 = torch.tensor([[0.0, 0.0, -1.0]], device="cuda", dtype=torch.float32)
        state_prim0 = torch.tensor([0], device="cuda", dtype=torch.int32)
        state_prim1 = torch.tensor([0], device="cuda", dtype=torch.int32)
        state_exterior_angle = torch.tensor([torch.pi], device="cuda", dtype=torch.float32)
        state_src = tx_pos.clone()
        state_src_power = torch.ones((1,), device="cuda", dtype=torch.float32)
        active = torch.ones((1,), device="cuda", dtype=torch.bool)
        material_gain = torch.ones((1,), device="cuda", dtype=torch.float32)
        material_valid = torch.ones((1,), device="cuda", dtype=torch.bool)
        empty_i = torch.empty((0,), device="cuda", dtype=torch.int32)
        empty_f = torch.empty((0,), device="cuda", dtype=torch.float32)
        empty_v = torch.empty((0, 3), device="cuda", dtype=torch.float32)
        empty_b = torch.empty((0,), device="cuda", dtype=torch.bool)

        out = rt._C.diffraction_paths_order1_forward(
            scene._native_handle,
            tx_pos,
            rx_pos,
            active,
            state_edge_index,
            state_edge_pos,
            state_edge_dir,
            state_edge_t_min,
            state_edge_t_max,
            state_n0,
            state_n1,
            state_prim0,
            state_prim1,
            state_exterior_angle,
            state_src,
            state_src_power,
            material_gain,
            material_valid,
            8,
            1.0,
        )
        count = int(out[0].item())
        self.assertGreaterEqual(count, 0)
        self.assertEqual(out[1].shape, (8,))
        self.assertEqual(out[8].dtype, torch.float32)

    def test_diffraction_paths_order1_accepts_multi_mesh_scene(self):
        verts0 = torch.tensor(
            [[10.0, -1.0, 0.0], [12.0, -1.0, 0.0], [10.0, 1.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
        )
        verts1 = torch.tensor(
            [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
        )
        faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(verts0, faces))
        scene.add_mesh(rt.Mesh(verts1, faces))
        scene.build()

        tx_pos = torch.tensor([[0.0, -1.0, 0.25]], device="cuda", dtype=torch.float32)
        rx_pos = torch.tensor([[0.0, 1.0, 0.25]], device="cuda", dtype=torch.float32)
        state_edge_index = torch.tensor([0], device="cuda", dtype=torch.int32)
        state_edge_pos = torch.tensor([[0.0, 0.0, 0.0]], device="cuda", dtype=torch.float32)
        state_edge_dir = torch.tensor([[1.0, 0.0, 0.0]], device="cuda", dtype=torch.float32)
        state_edge_t_min = torch.tensor([-1.0], device="cuda", dtype=torch.float32)
        state_edge_t_max = torch.tensor([1.0], device="cuda", dtype=torch.float32)
        state_n0 = torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32)
        state_n1 = torch.tensor([[0.0, 0.0, -1.0]], device="cuda", dtype=torch.float32)
        state_prim0 = torch.tensor([1], device="cuda", dtype=torch.int32)
        state_prim1 = torch.tensor([1], device="cuda", dtype=torch.int32)
        state_exterior_angle = torch.tensor([torch.pi], device="cuda", dtype=torch.float32)
        state_src = tx_pos.clone()
        state_src_power = torch.ones((1,), device="cuda", dtype=torch.float32)
        active = torch.ones((1,), device="cuda", dtype=torch.bool)
        material_gain = torch.ones((2,), device="cuda", dtype=torch.float32)
        material_valid = torch.ones((2,), device="cuda", dtype=torch.bool)

        out = rt._C.diffraction_paths_order1_forward(
            scene._native_handle,
            tx_pos,
            rx_pos,
            active,
            state_edge_index,
            state_edge_pos,
            state_edge_dir,
            state_edge_t_min,
            state_edge_t_max,
            state_n0,
            state_n1,
            state_prim0,
            state_prim1,
            state_exterior_angle,
            state_src,
            state_src_power,
            material_gain,
            material_valid,
            8,
            1.0,
        )
        self.assertEqual(out[1].shape, (8,))
        self.assertEqual(out[8].dtype, torch.float32)

    def test_default_diffraction_material_covers_all_mesh_faces(self):
        verts0 = torch.tensor(
            [[10.0, -1.0, 0.0], [12.0, -1.0, 0.0], [10.0, 1.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
        )
        verts1 = torch.tensor(
            [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
        )
        faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(verts0, faces))
        scene.add_mesh(rt.Mesh(verts1, faces))
        scene.build()
        material = scene._default_dfr_material(device=torch.device("cuda"), dtype=torch.float32)
        self.assertEqual(tuple(material.gain.shape), (2,))
        self.assertEqual(tuple(material.valid.shape), (2,))

    def test_diffraction_accumulation_native_binding_smoke(self):
        verts = torch.tensor(
            [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
        )
        faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(verts, faces))
        scene.build()

        active = torch.ones((1,), device="cuda", dtype=torch.bool)
        state_edge_index = torch.tensor([0], device="cuda", dtype=torch.int32)
        state_edge_pos = torch.tensor([[0.0, 0.0, 0.0]], device="cuda", dtype=torch.float32)
        state_edge_dir = torch.tensor([[1.0, 0.0, 0.0]], device="cuda", dtype=torch.float32)
        state_edge_t_min = torch.tensor([-1.0], device="cuda", dtype=torch.float32)
        state_edge_t_max = torch.tensor([1.0], device="cuda", dtype=torch.float32)
        state_n0 = torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32)
        state_n1 = torch.tensor([[0.0, 0.0, -1.0]], device="cuda", dtype=torch.float32)
        state_prim0 = torch.tensor([0], device="cuda", dtype=torch.int32)
        state_prim1 = torch.tensor([0], device="cuda", dtype=torch.int32)
        state_exterior_angle = torch.tensor([torch.pi], device="cuda", dtype=torch.float32)
        state_src = torch.tensor([[0.0, -1.0, 0.25]], device="cuda", dtype=torch.float32)
        state_src_power = torch.ones((1,), device="cuda", dtype=torch.float32)
        zeros_vec = torch.zeros((1, 3), device="cuda", dtype=torch.float32)
        material_eta_r = torch.ones((1,), device="cuda", dtype=torch.float32)
        material_sigma = torch.zeros((1,), device="cuda", dtype=torch.float32)
        material_mu_r = torch.ones((1,), device="cuda", dtype=torch.float32)
        material_gain = torch.ones((1,), device="cuda", dtype=torch.float32)
        material_valid = torch.ones((1,), device="cuda", dtype=torch.bool)
        empty_i = torch.empty((0,), device="cuda", dtype=torch.int32)
        empty_f = torch.empty((0,), device="cuda", dtype=torch.float32)
        empty_v = torch.empty((0, 3), device="cuda", dtype=torch.float32)
        empty_b = torch.empty((0,), device="cuda", dtype=torch.bool)

        out = rt._C.diffraction_accumulation_forward(
            scene._native_handle,
            active,
            state_edge_index,
            state_edge_pos,
            state_edge_dir,
            state_edge_t_min,
            state_edge_t_max,
            state_n0,
            state_n1,
            state_prim0,
            state_prim1,
            state_exterior_angle,
            state_src,
            state_src_power,
            zeros_vec,
            zeros_vec,
            material_eta_r,
            material_sigma,
            material_mu_r,
            material_gain,
            material_valid,
            2,
            0.0,
            -1.0,
            1.0,
            -1.0,
            1.0,
            4,
            4,
            0.25,
            1.0,
            4,
            0,
            0,
            0,
            1,
            empty_b,
            empty_i,
            empty_v,
            empty_v,
            empty_f,
            empty_f,
            empty_v,
            empty_v,
            empty_i,
            empty_i,
            empty_f,
            1,
        )
        self.assertEqual(out[0].shape, (4, 4))
        self.assertEqual(out[1].dtype, torch.float32)

    def test_scene_accum_dfr_direct_native_api(self):
        verts = torch.tensor(
            [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
        )
        faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(verts, faces))
        scene.build()

        states = rt.DfrStates(
            edge_index=torch.tensor([0], device="cuda", dtype=torch.int32),
            edge_pos=torch.tensor([[0.0, 0.0, 0.0]], device="cuda", dtype=torch.float32),
            edge_dir=torch.tensor([[1.0, 0.0, 0.0]], device="cuda", dtype=torch.float32),
            edge_t_min=torch.tensor([-1.0], device="cuda", dtype=torch.float32),
            edge_t_max=torch.tensor([1.0], device="cuda", dtype=torch.float32),
            n0=torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32),
            n1=torch.tensor([[0.0, 0.0, -1.0]], device="cuda", dtype=torch.float32),
            prim0=torch.tensor([0], device="cuda", dtype=torch.int32),
            prim1=torch.tensor([0], device="cuda", dtype=torch.int32),
            exterior_angle=torch.tensor([torch.pi], device="cuda", dtype=torch.float32),
            src=torch.tensor([[0.0, -1.0, 0.25]], device="cuda", dtype=torch.float32),
            src_power=torch.ones((1,), device="cuda", dtype=torch.float32),
        )
        grid = rt.DfrGrid(
            axis=2,
            position=0.0,
            coord0_min=-1.0,
            coord0_max=1.0,
            coord1_min=-1.0,
            coord1_max=1.0,
            resolution0=4,
            resolution1=4,
        )
        out = scene.accum_dfr_direct(states=states, grid=grid, wavelength=1.0, direct_samples=4)
        self.assertEqual(out.power.shape, (4, 4))
        self.assertEqual(out.field_x_re.dtype, torch.float32)

    def test_scene_accum_dfr_direct_backward_reaches_state_and_material(self):
        verts = torch.tensor(
            [[-1.0, -1.0, 10.0], [1.0, -1.0, 10.0], [-1.0, 1.0, 10.0]],
            device="cuda",
            dtype=torch.float32,
        )
        faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(verts, faces))
        scene.build()

        edge_pos = torch.tensor([[0.0, 0.0, 0.0]], device="cuda", dtype=torch.float32, requires_grad=True)
        edge_dir = torch.tensor([[1.0, 0.0, 0.0]], device="cuda", dtype=torch.float32, requires_grad=True)
        edge_t_min = torch.tensor([-0.5], device="cuda", dtype=torch.float32, requires_grad=True)
        edge_t_max = torch.tensor([0.5], device="cuda", dtype=torch.float32, requires_grad=True)
        exterior_angle = torch.tensor([1.5 * torch.pi], device="cuda", dtype=torch.float32, requires_grad=True)
        src = torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32, requires_grad=True)
        src_power = torch.tensor([2.0], device="cuda", dtype=torch.float32, requires_grad=True)
        wi = torch.tensor([[0.0, 0.0, -1.0]], device="cuda", dtype=torch.float32, requires_grad=True)
        states = rt.DfrStates(
            edge_index=torch.tensor([0], device="cuda", dtype=torch.int32),
            edge_pos=edge_pos,
            edge_dir=edge_dir,
            edge_t_min=edge_t_min,
            edge_t_max=edge_t_max,
            n0=torch.tensor([[0.0, 1.0, 0.0]], device="cuda", dtype=torch.float32),
            n1=torch.tensor([[0.0, -1.0, 0.0]], device="cuda", dtype=torch.float32),
            prim0=torch.tensor([-1], device="cuda", dtype=torch.int32),
            prim1=torch.tensor([-1], device="cuda", dtype=torch.int32),
            exterior_angle=exterior_angle,
            src=src,
            src_power=src_power,
            wi=wi,
            d0=torch.tensor([[0.0, 0.0, -1.0]], device="cuda", dtype=torch.float32),
            count=1,
        )
        gain = torch.tensor([1.0], device="cuda", dtype=torch.float32, requires_grad=True)
        material = rt.DfrMaterial(
            eta_r=torch.tensor([4.0], device="cuda", dtype=torch.float32),
            sigma=torch.tensor([0.0], device="cuda", dtype=torch.float32),
            mu_r=torch.tensor([1.0], device="cuda", dtype=torch.float32),
            gain=gain,
            valid=torch.tensor([True], device="cuda", dtype=torch.bool),
        )
        grid = rt.DfrGrid(axis=2, position=-1.0, resolution0=1, resolution1=1, cell_area=4.0)
        out = scene.accum_dfr_direct(
            states=states,
            grid=grid,
            material=material,
            wavelength=0.125,
            seed=17,
            direct_samples=64,
        )
        (out.power.sum() + out.field_x_re.sum()).backward()
        for tensor in (edge_pos, edge_dir, edge_t_min, edge_t_max, exterior_angle, src, src_power, wi, gain):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(bool(torch.isfinite(tensor.grad).all().item()))

    def test_scene_accum_dfr_direct_jvp_reaches_power_and_field_x_re(self):
        verts = torch.tensor(
            [[-1.0, -1.0, 10.0], [1.0, -1.0, 10.0], [-1.0, 1.0, 10.0]],
            device="cuda",
            dtype=torch.float32,
        )
        faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(verts, faces))
        scene.build()

        edge_pos = torch.tensor([[0.0, 0.0, 0.0]], device="cuda", dtype=torch.float32)
        states = rt.DfrStates(
            edge_index=torch.tensor([0], device="cuda", dtype=torch.int32),
            edge_pos=edge_pos,
            edge_dir=torch.tensor([[1.0, 0.0, 0.0]], device="cuda", dtype=torch.float32),
            edge_t_min=torch.tensor([-0.5], device="cuda", dtype=torch.float32),
            edge_t_max=torch.tensor([0.5], device="cuda", dtype=torch.float32),
            n0=torch.tensor([[0.0, 1.0, 0.0]], device="cuda", dtype=torch.float32),
            n1=torch.tensor([[0.0, -1.0, 0.0]], device="cuda", dtype=torch.float32),
            prim0=torch.tensor([-1], device="cuda", dtype=torch.int32),
            prim1=torch.tensor([-1], device="cuda", dtype=torch.int32),
            exterior_angle=torch.tensor([1.5 * torch.pi], device="cuda", dtype=torch.float32),
            src=torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32),
            src_power=torch.tensor([2.0], device="cuda", dtype=torch.float32),
            wi=torch.tensor([[0.0, 0.0, -1.0]], device="cuda", dtype=torch.float32),
            d0=torch.tensor([[0.0, 0.0, -1.0]], device="cuda", dtype=torch.float32),
            count=1,
        )
        material = rt.DfrMaterial(
            eta_r=torch.tensor([4.0], device="cuda", dtype=torch.float32),
            sigma=torch.tensor([0.0], device="cuda", dtype=torch.float32),
            mu_r=torch.tensor([1.0], device="cuda", dtype=torch.float32),
            gain=torch.tensor([1.0], device="cuda", dtype=torch.float32),
            valid=torch.tensor([True], device="cuda", dtype=torch.bool),
        )
        grid = rt.DfrGrid(axis=2, position=-1.0, resolution0=1, resolution1=1, cell_area=4.0)
        with torch.autograd.forward_ad.dual_level():
            dual_edge_pos = torch.autograd.forward_ad.make_dual(edge_pos, torch.ones_like(edge_pos) * 0.01)
            dual_states = rt.DfrStates(
                edge_index=states.edge_index,
                edge_pos=dual_edge_pos,
                edge_dir=states.edge_dir,
                edge_t_min=states.edge_t_min,
                edge_t_max=states.edge_t_max,
                n0=states.n0,
                n1=states.n1,
                prim0=states.prim0,
                prim1=states.prim1,
                exterior_angle=states.exterior_angle,
                src=states.src,
                src_power=states.src_power,
                wi=states.wi,
                d0=states.d0,
                count=states.count,
            )
            out = scene.accum_dfr_direct(
                states=dual_states,
                grid=grid,
                material=material,
                wavelength=0.125,
                seed=17,
                direct_samples=64,
            )
            _power, tangent_power = torch.autograd.forward_ad.unpack_dual(out.power)
            _field_x_re, tangent_field_x_re = torch.autograd.forward_ad.unpack_dual(out.field_x_re)
        self.assertIsNotNone(tangent_power)
        self.assertIsNotNone(tangent_field_x_re)
        self.assertTrue(bool(torch.isfinite(tangent_power).all().item()))
        self.assertTrue(bool(torch.isfinite(tangent_field_x_re).all().item()))

    def test_scene_accum_dfr_chain_backward_reaches_initial_recursive_and_material(self):
        verts = torch.tensor(
            [[-1.0, -1.0, 10.0], [1.0, -1.0, 10.0], [-1.0, 1.0, 10.0]],
            device="cuda",
            dtype=torch.float32,
        )
        faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(verts, faces))
        scene.build()

        edge_pos = torch.tensor([[0.0, 0.0, 0.0]], device="cuda", dtype=torch.float32, requires_grad=True)
        edge_dir = torch.tensor([[1.0, 0.0, 0.0]], device="cuda", dtype=torch.float32, requires_grad=True)
        edge_t_min = torch.tensor([-0.5], device="cuda", dtype=torch.float32, requires_grad=True)
        edge_t_max = torch.tensor([0.5], device="cuda", dtype=torch.float32, requires_grad=True)
        exterior_angle = torch.tensor([1.5 * torch.pi], device="cuda", dtype=torch.float32, requires_grad=True)
        src = torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32, requires_grad=True)
        src_power = torch.tensor([2.0], device="cuda", dtype=torch.float32, requires_grad=True)
        initial = rt.DfrStates(
            edge_index=torch.tensor([0], device="cuda", dtype=torch.int32),
            edge_pos=edge_pos,
            edge_dir=edge_dir,
            edge_t_min=edge_t_min,
            edge_t_max=edge_t_max,
            n0=torch.tensor([[0.0, 1.0, 0.0]], device="cuda", dtype=torch.float32),
            n1=torch.tensor([[0.0, -1.0, 0.0]], device="cuda", dtype=torch.float32),
            prim0=torch.tensor([-1], device="cuda", dtype=torch.int32),
            prim1=torch.tensor([-1], device="cuda", dtype=torch.int32),
            exterior_angle=exterior_angle,
            src=src,
            src_power=src_power,
            wi=torch.tensor([[0.0, 0.0, -1.0]], device="cuda", dtype=torch.float32),
            d0=torch.tensor([[0.0, 0.0, -1.0]], device="cuda", dtype=torch.float32),
            count=1,
        )
        rec_edge_pos = torch.tensor([[0.0, 0.5, 0.0]], device="cuda", dtype=torch.float32, requires_grad=True)
        rec_edge_dir = torch.tensor([[1.0, 0.0, 0.0]], device="cuda", dtype=torch.float32, requires_grad=True)
        rec_edge_t_min = torch.tensor([-0.5], device="cuda", dtype=torch.float32, requires_grad=True)
        rec_edge_t_max = torch.tensor([0.5], device="cuda", dtype=torch.float32, requires_grad=True)
        rec_exterior_angle = torch.tensor([1.5 * torch.pi], device="cuda", dtype=torch.float32, requires_grad=True)
        recursive = rt.DfrStates(
            edge_index=torch.tensor([1], device="cuda", dtype=torch.int32),
            edge_pos=rec_edge_pos,
            edge_dir=rec_edge_dir,
            edge_t_min=rec_edge_t_min,
            edge_t_max=rec_edge_t_max,
            n0=torch.tensor([[0.0, 1.0, 0.0]], device="cuda", dtype=torch.float32),
            n1=torch.tensor([[0.0, -1.0, 0.0]], device="cuda", dtype=torch.float32),
            prim0=torch.tensor([-1], device="cuda", dtype=torch.int32),
            prim1=torch.tensor([-1], device="cuda", dtype=torch.int32),
            exterior_angle=rec_exterior_angle,
            src=torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32),
            src_power=torch.tensor([1.0], device="cuda", dtype=torch.float32),
            wi=torch.tensor([[0.0, 1.0, 0.0]], device="cuda", dtype=torch.float32),
            d0=torch.tensor([[0.0, 0.0, -1.0]], device="cuda", dtype=torch.float32),
            count=1,
        )
        gain = torch.tensor([1.0], device="cuda", dtype=torch.float32, requires_grad=True)
        material = rt.DfrMaterial(
            eta_r=torch.tensor([4.0], device="cuda", dtype=torch.float32),
            sigma=torch.tensor([0.0], device="cuda", dtype=torch.float32),
            mu_r=torch.tensor([1.0], device="cuda", dtype=torch.float32),
            gain=gain,
            valid=torch.tensor([True], device="cuda", dtype=torch.bool),
        )
        grid = rt.DfrGrid(axis=2, position=-1.0, resolution0=1, resolution1=1, cell_area=4.0)
        out = scene.accum_dfr(
            initial_states=initial,
            recursive_states=recursive,
            grid=grid,
            material=material,
            wavelength=0.125,
            seed=17,
            direct_samples=64,
            keller_samples=64,
            max_order=2,
        )
        (out.power.sum() + out.field_x_re.sum()).backward()
        for tensor in (
            edge_pos,
            edge_dir,
            edge_t_min,
            edge_t_max,
            exterior_angle,
            src,
            src_power,
            rec_edge_pos,
            rec_edge_dir,
            rec_edge_t_min,
            rec_edge_t_max,
            rec_exterior_angle,
            gain,
        ):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(bool(torch.isfinite(tensor.grad).all().item()))

    def test_scene_accum_dfr_chain_jvp_reaches_power_and_field_x_re(self):
        verts = torch.tensor(
            [[-1.0, -1.0, 10.0], [1.0, -1.0, 10.0], [-1.0, 1.0, 10.0]],
            device="cuda",
            dtype=torch.float32,
        )
        faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(verts, faces))
        scene.build()

        edge_pos = torch.tensor([[0.0, 0.0, 0.0]], device="cuda", dtype=torch.float32)
        initial = rt.DfrStates(
            edge_index=torch.tensor([0], device="cuda", dtype=torch.int32),
            edge_pos=edge_pos,
            edge_dir=torch.tensor([[1.0, 0.0, 0.0]], device="cuda", dtype=torch.float32),
            edge_t_min=torch.tensor([-0.5], device="cuda", dtype=torch.float32),
            edge_t_max=torch.tensor([0.5], device="cuda", dtype=torch.float32),
            n0=torch.tensor([[0.0, 1.0, 0.0]], device="cuda", dtype=torch.float32),
            n1=torch.tensor([[0.0, -1.0, 0.0]], device="cuda", dtype=torch.float32),
            prim0=torch.tensor([-1], device="cuda", dtype=torch.int32),
            prim1=torch.tensor([-1], device="cuda", dtype=torch.int32),
            exterior_angle=torch.tensor([1.5 * torch.pi], device="cuda", dtype=torch.float32),
            src=torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32),
            src_power=torch.tensor([2.0], device="cuda", dtype=torch.float32),
            wi=torch.tensor([[0.0, 0.0, -1.0]], device="cuda", dtype=torch.float32),
            d0=torch.tensor([[0.0, 0.0, -1.0]], device="cuda", dtype=torch.float32),
            count=1,
        )
        recursive = rt.DfrStates(
            edge_index=torch.tensor([1], device="cuda", dtype=torch.int32),
            edge_pos=torch.tensor([[0.0, 0.5, 0.0]], device="cuda", dtype=torch.float32),
            edge_dir=torch.tensor([[1.0, 0.0, 0.0]], device="cuda", dtype=torch.float32),
            edge_t_min=torch.tensor([-0.5], device="cuda", dtype=torch.float32),
            edge_t_max=torch.tensor([0.5], device="cuda", dtype=torch.float32),
            n0=torch.tensor([[0.0, 1.0, 0.0]], device="cuda", dtype=torch.float32),
            n1=torch.tensor([[0.0, -1.0, 0.0]], device="cuda", dtype=torch.float32),
            prim0=torch.tensor([-1], device="cuda", dtype=torch.int32),
            prim1=torch.tensor([-1], device="cuda", dtype=torch.int32),
            exterior_angle=torch.tensor([1.5 * torch.pi], device="cuda", dtype=torch.float32),
            src=torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32),
            src_power=torch.tensor([1.0], device="cuda", dtype=torch.float32),
            wi=torch.tensor([[0.0, 1.0, 0.0]], device="cuda", dtype=torch.float32),
            d0=torch.tensor([[0.0, 0.0, -1.0]], device="cuda", dtype=torch.float32),
            count=1,
        )
        material = rt.DfrMaterial(
            eta_r=torch.tensor([4.0], device="cuda", dtype=torch.float32),
            sigma=torch.tensor([0.0], device="cuda", dtype=torch.float32),
            mu_r=torch.tensor([1.0], device="cuda", dtype=torch.float32),
            gain=torch.tensor([1.0], device="cuda", dtype=torch.float32),
            valid=torch.tensor([True], device="cuda", dtype=torch.bool),
        )
        grid = rt.DfrGrid(axis=2, position=-1.0, resolution0=1, resolution1=1, cell_area=4.0)
        with torch.autograd.forward_ad.dual_level():
            dual_edge_pos = torch.autograd.forward_ad.make_dual(edge_pos, torch.ones_like(edge_pos) * 0.01)
            dual_initial = rt.DfrStates(
                edge_index=initial.edge_index,
                edge_pos=dual_edge_pos,
                edge_dir=initial.edge_dir,
                edge_t_min=initial.edge_t_min,
                edge_t_max=initial.edge_t_max,
                n0=initial.n0,
                n1=initial.n1,
                prim0=initial.prim0,
                prim1=initial.prim1,
                exterior_angle=initial.exterior_angle,
                src=initial.src,
                src_power=initial.src_power,
                wi=initial.wi,
                d0=initial.d0,
                count=initial.count,
            )
            out = scene.accum_dfr(
                initial_states=dual_initial,
                recursive_states=recursive,
                grid=grid,
                material=material,
                wavelength=0.125,
                seed=17,
                direct_samples=64,
                keller_samples=64,
                max_order=2,
            )
            _power, tangent_power = torch.autograd.forward_ad.unpack_dual(out.power)
            _field_x_re, tangent_field_x_re = torch.autograd.forward_ad.unpack_dual(out.field_x_re)
        self.assertIsNotNone(tangent_power)
        self.assertIsNotNone(tangent_field_x_re)
        self.assertTrue(bool(torch.isfinite(tangent_power).all().item()))
        self.assertTrue(bool(torch.isfinite(tangent_field_x_re).all().item()))

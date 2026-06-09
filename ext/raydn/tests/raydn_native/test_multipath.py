import unittest
from unittest import mock

import torch
import raydn as rt


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
        with mock.patch.object(
            torch.Tensor,
            "contiguous",
            side_effect=AssertionError("Scene.visible() must not copy endpoints in Python."),
        ):
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
        upstream = torch.tensor([[1.75]], device="cuda", dtype=torch.float32)
        chain = scene.trace_reflections(ray, max_bounces=1)
        with (
            mock.patch("torch.cat", side_effect=AssertionError("Scene.trace_reflections() must not use torch.cat.")),
            mock.patch(
                "torch.zeros_like",
                side_effect=AssertionError("Scene.trace_reflections() backward must not fill grads in Python."),
            ),
            mock.patch(
                "torch.empty",
                side_effect=AssertionError("Scene.trace_reflections() backward must not create empty grad sentinels in Python."),
            ),
        ):
            chain.t.backward(upstream)
        self.assertIsNotNone(verts0.grad)
        self.assertIsNotNone(verts1.grad)
        torch.testing.assert_close(verts0.grad, torch.zeros_like(verts0), atol=1e-5, rtol=1e-5)
        self.assertTrue(bool(torch.isfinite(verts1.grad).all().item()))

    def test_single_bounce_reflection_t_backward_matches_intersect_t_vjp_nonuniform(self):
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
            torch.tensor(
                [[0.0, 0.0, -1.0], [0.25, 0.25, -1.0], [-0.25, 0.25, -1.0]],
                device="cuda",
                dtype=torch.float32,
            ),
            torch.tensor(
                [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
                device="cuda",
                dtype=torch.float32,
            ),
        )
        upstream = torch.tensor([[0.25], [-1.5], [2.0]], device="cuda", dtype=torch.float32)

        scene.trace_reflections(ray, max_bounces=1).t.backward(upstream)
        public_grad = verts.grad.detach().clone()

        active = torch.empty((0,), device="cuda", dtype=torch.bool)
        values = torch.ops.raydn.trace_reflections_forward(scene._require_native_scene(), ray.o, ray.d, ray.tmax, active, 1)
        expected = torch.ops.raydn.intersect_backward_t(
            scene._require_native_scene(),
            ray.o,
            ray.d,
            active,
            values[4][:, 0],
            values[5][:, 0],
            upstream[:, 0],
            True,
            False,
            False,
            False,
        )[0]
        torch.testing.assert_close(public_grad, expected, atol=1e-5, rtol=1e-5)

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

    def test_reflection_trace_reduced_fields_match_full_image_source_trace(self):
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

        reduced = scene.trace_reflections(ray, max_bounces=2)
        valid = reduced.valid
        t = reduced.t
        prim_ids = reduced.prim_ids
        image_sources = reduced.image_sources

        torch.testing.assert_close(valid, torch.tensor([[True, True]], device="cuda"))
        torch.testing.assert_close(t, torch.tensor([[1.0, 1.0]], device="cuda"), atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(prim_ids, torch.tensor([[0, 1]], device="cuda", dtype=torch.int32))
        torch.testing.assert_close(
            image_sources[0],
            torch.tensor([[0.0, 0.0, -1.0], [4.0, 0.0, -1.0]], device="cuda"),
            atol=1e-3,
            rtol=1e-3,
        )

    def test_native_path_stats_match_reference(self):
        valid = torch.tensor(
            [[True, True, False], [True, False, False], [True, True, True]],
            device="cuda",
            dtype=torch.bool,
        )
        t = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            device="cuda",
            dtype=torch.float32,
        )
        counts, checksum = torch.ops.raydn.reflection_trace_stats(valid.contiguous(), t.contiguous())
        self.assertEqual(int(counts[0].item()), 6)
        self.assertEqual(int(counts[1].item()), 1)
        self.assertAlmostEqual(float(checksum[0].item()), 31.0, places=5)

        path_count = torch.tensor([2], device="cuda", dtype=torch.int32)
        path_valid = torch.tensor([True, False, True], device="cuda", dtype=torch.bool)
        delay = torch.tensor([0.25, 10.0, 0.5], device="cuda", dtype=torch.float32)
        valid_count, path_checksum = torch.ops.raydn.diffraction_path_stats(path_count, path_valid, delay)
        self.assertEqual(int(valid_count[0].item()), 2)
        self.assertAlmostEqual(float(path_checksum[0].item()), 0.75, places=5)

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
        empty_uv = torch.empty((0, 2), device="cuda", dtype=torch.float32)
        empty_face_uv = torch.empty((0, 3), device="cuda", dtype=torch.int32)
        empty_transform = torch.empty((0, 4), device="cuda", dtype=torch.float32)
        ray = rt.Ray(ray_o, ray_d)

        def fn(verts_value):
            scene = rt.Scene()
            scene.add_mesh(
                rt.Mesh(
                    verts_value.contiguous(),
                    faces,
                    uv=empty_uv,
                    face_uv=empty_face_uv,
                    to_world_left=empty_transform,
                    to_world_right=empty_transform,
                )
            )
            scene.build()
            return scene.trace_reflections(ray, max_bounces=2).t[:, 1]

        with (
            mock.patch(
                "torch.zeros_like",
                side_effect=AssertionError("Scene.trace_reflections() jvp must not fill tangents in Python."),
            ),
            mock.patch(
                "torch.empty",
                side_effect=AssertionError("Scene.trace_reflections() jvp must not create active mask sentinels in Python."),
            ),
        ):
            _primal, jvp = torch.func.jvp(fn, (base_verts,), (tangent,))
        eps = 1e-3
        fd = (fn(base_verts + eps * tangent) - fn(base_verts - eps * tangent)) / (2.0 * eps)
        torch.testing.assert_close(jvp, fd, atol=2e-2, rtol=2e-2)

    def test_two_bounce_reflection_image_sources_vjp_uses_strided_upstream(self):
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
        base_ray_o = torch.tensor(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            device="cuda",
            dtype=torch.float32,
        )
        base_ray_d = torch.tensor(
            [[0.8, 0.0, -1.0], [0.8, 0.0, -1.0]],
            device="cuda",
            dtype=torch.float32,
        )
        upstream = torch.tensor(
            [
                [[0.25, -0.5, 0.75], [1.0, -0.25, 0.5]],
                [[-0.75, 0.4, 0.2], [0.1, 0.8, -0.3]],
            ],
            device="cuda",
            dtype=torch.float32,
        ).permute(1, 0, 2)
        self.assertFalse(upstream.is_contiguous())

        def run_backward(upstream_value):
            verts = base_verts.clone().detach().requires_grad_(True)
            ray_o = base_ray_o.clone().detach().requires_grad_(True)
            ray_d = base_ray_d.clone().detach().requires_grad_(True)
            scene = rt.Scene()
            scene.add_mesh(rt.Mesh(verts, faces))
            scene.build()
            chain = scene.trace_reflections(rt.Ray(ray_o, ray_d), max_bounces=2)
            with mock.patch(
                "torch.zeros_like",
                side_effect=AssertionError("Scene.trace_reflections() backward must not fill grads in Python."),
            ):
                chain.image_sources.backward(upstream_value)
            return verts.grad.detach(), ray_o.grad.detach(), ray_d.grad.detach()

        strided_grads = run_backward(upstream)
        contiguous_grads = run_backward(upstream.contiguous())
        for strided, contiguous in zip(strided_grads, contiguous_grads):
            torch.testing.assert_close(strided, contiguous, atol=2e-5, rtol=2e-5)

        tangent_verts = torch.zeros_like(base_verts)
        tangent_verts[:3, 2] = 0.03
        tangent_verts[3:, 0] = -0.04
        tangent_ray_o = torch.tensor(
            [[0.02, -0.01, 0.03], [-0.03, 0.04, -0.02]],
            device="cuda",
            dtype=torch.float32,
        )
        tangent_ray_d = torch.zeros_like(base_ray_d)
        vjp_dot = (
            (strided_grads[0] * tangent_verts).sum()
            + (strided_grads[1] * tangent_ray_o).sum()
            + (strided_grads[2] * tangent_ray_d).sum()
        )

        def weighted_image_sources(verts_value, ray_o_value, ray_d_value):
            scene = rt.Scene()
            scene.add_mesh(rt.Mesh(verts_value.contiguous(), faces))
            scene.build()
            chain = scene.trace_reflections(rt.Ray(ray_o_value.contiguous(), ray_d_value.contiguous()), max_bounces=2)
            return (chain.image_sources * upstream.contiguous()).sum()

        eps = 1e-3
        fd = (
            weighted_image_sources(
                base_verts + eps * tangent_verts,
                base_ray_o + eps * tangent_ray_o,
                base_ray_d + eps * tangent_ray_d,
            )
            - weighted_image_sources(
                base_verts - eps * tangent_verts,
                base_ray_o - eps * tangent_ray_o,
                base_ray_d - eps * tangent_ray_d,
            )
        ) / (2.0 * eps)
        torch.testing.assert_close(vjp_dot, fd, atol=3e-2, rtol=3e-2)

    def test_reflection_epc_field_backward_matches_intersect_t_vjp_nonuniform(self):
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
        source = torch.tensor(
            [[0.0, 0.0, -1.0], [0.2, 0.1, -1.0], [-0.2, 0.2, -1.0]],
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        receiver = torch.tensor(
            [[0.0, 0.0, 1.0], [0.2, 0.1, 1.0], [-0.2, 0.2, 1.0]],
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        out = scene.trace_refl_epc_field(source, receiver, max_bounces=1)
        grad_real = torch.tensor(
            [[0.25, 9.0], [-1.5, 9.0], [0.75, 9.0]], device="cuda", dtype=torch.float32
        )[:, 0]
        grad_imag = torch.tensor(
            [[1.25, 9.0], [-0.5, 9.0], [0.0, 9.0]], device="cuda", dtype=torch.float32
        )[:, 0]
        grad_path = torch.tensor(
            [[-0.25, 9.0], [0.6, 9.0], [2.0, 9.0]], device="cuda", dtype=torch.float32
        )[:, 0]
        self.assertFalse(grad_real.is_contiguous())
        self.assertFalse(grad_imag.is_contiguous())
        self.assertFalse(grad_path.is_contiguous())
        with mock.patch(
            "torch.zeros_like",
            side_effect=AssertionError("Scene.trace_refl_epc_field() backward must not fill grads in Python."),
        ):
            torch.autograd.backward((out.field_real, out.field_imag, out.path_length), (grad_real, grad_imag, grad_path))
        self.assertIsNotNone(verts.grad)
        self.assertIsNotNone(source.grad)
        self.assertIsNotNone(receiver.grad)

        active = torch.empty((0,), device="cuda", dtype=torch.bool)
        values = torch.ops.raydn.trace_refl_epc_field_forward(
            scene._require_native_scene(),
            source.detach(),
            receiver.detach(),
            active,
            1,
        )
        tape_prim_id, tape_barycentric, tape_t = values[5], values[6], values[2]
        inv_denom = 1.0 / (1.0 + tape_t)
        real_dt = -torch.sin(tape_t) * inv_denom - torch.cos(tape_t) * inv_denom * inv_denom
        imag_dt = torch.cos(tape_t) * inv_denom - torch.sin(tape_t) * inv_denom * inv_denom
        grad_t = grad_path + grad_real * real_dt + grad_imag * imag_dt
        ray_d = (receiver.detach() - source.detach()).contiguous()
        expected_vertices, expected_source_ray, expected_ray_d, _ = torch.ops.raydn.intersect_backward_t(
            scene._require_native_scene(),
            source.detach(),
            ray_d,
            active,
            tape_prim_id,
            tape_barycentric,
            grad_t.contiguous(),
            True,
            True,
            True,
            True,
        )
        torch.testing.assert_close(verts.grad, expected_vertices, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(source.grad, expected_source_ray - expected_ray_d, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(receiver.grad, expected_ray_d, atol=2e-5, rtol=2e-5)

    def test_reflection_epc_field_jvp_avoids_python_zero_tangents(self):
        verts = torch.tensor(
            [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
        )
        faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(verts, faces))
        scene.build()
        source = torch.tensor(
            [[0.0, 0.0, -1.0], [0.2, 0.1, -1.0], [-0.2, 0.2, -1.0]],
            device="cuda",
            dtype=torch.float32,
        )
        receiver = torch.tensor(
            [[0.0, 0.0, 1.0], [0.2, 0.1, 1.0], [-0.2, 0.2, 1.0]],
            device="cuda",
            dtype=torch.float32,
        )
        tangent_receiver = torch.tensor(
            [[0.05, -0.03, 0.02], [-0.02, 0.04, 0.01], [0.1, 0.2, -0.1]],
            device="cuda",
            dtype=torch.float32,
        ).t()
        self.assertFalse(tangent_receiver.is_contiguous())

        def fn(receiver_value):
            return scene.trace_refl_epc_field(source, receiver_value, max_bounces=1).path_length

        with (
            mock.patch(
                "torch.zeros_like",
                side_effect=AssertionError("Scene.trace_refl_epc_field() jvp must not fill tangents in Python."),
            ),
            mock.patch(
                "torch.empty",
                side_effect=AssertionError("Scene.trace_refl_epc_field() jvp must not create active mask sentinels in Python."),
            ),
        ):
            _primal, jvp = torch.func.jvp(fn, (receiver,), (tangent_receiver,))
        torch.testing.assert_close(jvp, torch.zeros_like(jvp), atol=0.0, rtol=0.0)

    def test_multi_mesh_reflection_epc_field_backward_avoids_python_cat(self):
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
        source = torch.tensor([[0.0, 0.0, -1.0]], device="cuda", dtype=torch.float32)
        receiver = torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32)
        upstream_real = torch.ones_like(source[:, 0])
        upstream_imag = torch.full_like(source[:, 0], -0.5)
        with mock.patch("torch.cat", side_effect=AssertionError("Scene.trace_refl_epc_field() must not use torch.cat.")), \
             mock.patch.object(
                 torch.Tensor,
                 "contiguous",
                 side_effect=AssertionError("Scene.trace_refl_epc_field() must not copy source/receiver in Python."),
             ), \
             mock.patch(
                 "torch.zeros_like",
                 side_effect=AssertionError("Scene.trace_refl_epc_field() backward must not fill grads in Python."),
             ):
            out = scene.trace_refl_epc_field(source, receiver, max_bounces=1)
            torch.autograd.backward(
                (out.field_real, out.field_imag),
                (
                    upstream_real,
                    upstream_imag,
                ),
            )
        self.assertIsNotNone(verts0.grad)
        self.assertIsNotNone(verts1.grad)
        torch.testing.assert_close(verts0.grad, torch.zeros_like(verts0), atol=1e-5, rtol=1e-5)

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
        out = torch.ops.raydn.reflection_dedup_forward(
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
        out = torch.ops.raydn.reflection_accumulation_forward(
            scene._native_scene,
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

        out = torch.ops.raydn.diffraction_paths_order1_forward(
            scene._native_scene,
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
            1,
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

        out = torch.ops.raydn.diffraction_paths_order1_forward(
            scene._native_scene,
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
            1,
            8,
            1.0,
        )
        self.assertEqual(out[1].shape, (8,))
        self.assertEqual(out[8].dtype, torch.float32)

    def test_diffraction_path_and_coherent_public_calls_do_not_create_empty_active_sentinel(self):
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
            src=tx_pos,
            src_power=torch.ones((1,), device="cuda", dtype=torch.float32),
        )
        material = rt.DfrMaterial(
            eta_r=torch.ones((1,), device="cuda", dtype=torch.float32),
            sigma=torch.zeros((1,), device="cuda", dtype=torch.float32),
            mu_r=torch.ones((1,), device="cuda", dtype=torch.float32),
            gain=torch.ones((1,), device="cuda", dtype=torch.float32),
            valid=torch.ones((1,), device="cuda", dtype=torch.bool),
        )
        grid = rt.DfrGrid(axis=2, position=0.0, resolution0=2, resolution1=2)
        with mock.patch(
            "torch.empty",
            side_effect=AssertionError("Multipath public calls must not create Python empty active sentinels."),
        ):
            paths = scene.trace_dfr_paths(
                tx_positions=tx_pos,
                rx_positions=rx_pos,
                states=states,
                material=material,
                active=None,
                max_paths=1,
                wavelength=1.0,
            )
            coherent = scene.accum_dfr_coherent_direct(
                states=states,
                grid=grid,
                material=material,
                active=None,
                wavelength=1.0,
            )
        self.assertEqual(tuple(paths.valid.shape), (1,))
        self.assertEqual(tuple(coherent.direct_field_x_re.shape), (2, 2))

    def test_diffraction_public_calls_accept_strided_active_masks(self):
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
        states = rt.DfrStates(
            edge_index=torch.tensor([0, 1], device="cuda", dtype=torch.int32),
            edge_pos=torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], device="cuda", dtype=torch.float32),
            edge_dir=torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], device="cuda", dtype=torch.float32),
            edge_t_min=torch.tensor([-1.0, -1.0], device="cuda", dtype=torch.float32),
            edge_t_max=torch.tensor([1.0, 1.0], device="cuda", dtype=torch.float32),
            n0=torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32),
            n1=torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]], device="cuda", dtype=torch.float32),
            prim0=torch.tensor([0, 0], device="cuda", dtype=torch.int32),
            prim1=torch.tensor([0, 0], device="cuda", dtype=torch.int32),
            exterior_angle=torch.tensor([torch.pi, torch.pi], device="cuda", dtype=torch.float32),
            src=torch.tensor([[0.0, -1.0, 0.25], [0.0, -1.0, 0.25]], device="cuda", dtype=torch.float32),
            src_power=torch.ones((2,), device="cuda", dtype=torch.float32),
        )
        material = rt.DfrMaterial(
            eta_r=torch.ones((1,), device="cuda", dtype=torch.float32),
            sigma=torch.zeros((1,), device="cuda", dtype=torch.float32),
            mu_r=torch.ones((1,), device="cuda", dtype=torch.float32),
            gain=torch.ones((1,), device="cuda", dtype=torch.float32),
            valid=torch.ones((1,), device="cuda", dtype=torch.bool),
        )
        grid = rt.DfrGrid(axis=2, position=0.0, resolution0=2, resolution1=2)
        active_strided = torch.tensor([True, True, False, False], device="cuda", dtype=torch.bool)[::2]
        self.assertFalse(active_strided.is_contiguous())
        active_contig = active_strided.contiguous()

        paths_strided = scene.trace_dfr_paths(
            tx_positions=tx_pos,
            rx_positions=rx_pos,
            states=states,
            material=material,
            active=active_strided,
            max_paths=2,
            wavelength=1.0,
        )
        paths_contig = scene.trace_dfr_paths(
            tx_positions=tx_pos,
            rx_positions=rx_pos,
            states=states,
            material=material,
            active=active_contig,
            max_paths=2,
            wavelength=1.0,
        )
        torch.testing.assert_close(paths_strided.count, paths_contig.count)
        torch.testing.assert_close(paths_strided.valid, paths_contig.valid)
        torch.testing.assert_close(paths_strided.delay, paths_contig.delay)

        accum_strided = scene.accum_dfr_direct(
            states=states,
            grid=grid,
            material=material,
            active=active_strided,
            wavelength=1.0,
            direct_samples=4,
            seed=13,
        )
        accum_contig = scene.accum_dfr_direct(
            states=states,
            grid=grid,
            material=material,
            active=active_contig,
            wavelength=1.0,
            direct_samples=4,
            seed=13,
        )
        torch.testing.assert_close(accum_strided.power, accum_contig.power)
        torch.testing.assert_close(accum_strided.field_x_re, accum_contig.field_x_re)

    def test_trace_dfr_paths_accepts_strided_state_material_and_endpoints(self):
        def strided_vec3(rows):
            values = torch.tensor(rows, device="cuda", dtype=torch.float32)
            base = torch.full((values.shape[0], 5), 123.0, device="cuda", dtype=torch.float32)
            base[:, ::2] = values
            view = base[:, ::2]
            self.assertFalse(view.is_contiguous())
            return view

        def strided_f32(values):
            values_t = torch.tensor(values, device="cuda", dtype=torch.float32)
            base = torch.full((values_t.shape[0] * 2,), 123.0, device="cuda", dtype=torch.float32)
            base[::2] = values_t
            view = base[::2]
            self.assertFalse(view.is_contiguous())
            return view

        def strided_i32(values):
            values_t = torch.tensor(values, device="cuda", dtype=torch.int32)
            base = torch.full((values_t.shape[0] * 2,), 123, device="cuda", dtype=torch.int32)
            base[::2] = values_t
            view = base[::2]
            self.assertFalse(view.is_contiguous())
            return view

        def strided_bool(values):
            values_t = torch.tensor(values, device="cuda", dtype=torch.bool)
            base = torch.zeros((values_t.shape[0] * 2,), device="cuda", dtype=torch.bool)
            base[::2] = values_t
            view = base[::2]
            self.assertFalse(view.is_contiguous())
            return view

        verts = torch.tensor(
            [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
        )
        faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(verts, faces))
        scene.build()

        tx_pos = strided_vec3([[0.0, -1.0, 0.25]])
        rx_pos = strided_vec3([[0.0, 1.0, 0.25]])
        states = rt.DfrStates(
            edge_index=strided_i32([0, 0, 0]),
            edge_pos=strided_vec3([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [99.0, 99.0, 99.0]]),
            edge_dir=strided_vec3([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            edge_t_min=strided_f32([-1.0, -1.0, 99.0]),
            edge_t_max=strided_f32([1.0, 1.0, 100.0]),
            n0=strided_vec3([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
            n1=strided_vec3([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]]),
            prim0=strided_i32([0, 0, 0]),
            prim1=strided_i32([0, 0, 0]),
            exterior_angle=strided_f32([float(torch.pi), float(torch.pi), 0.5]),
            src=strided_vec3([[0.0, -1.0, 0.25], [0.0, -1.0, 0.25], [99.0, 99.0, 99.0]]),
            src_power=strided_f32([1.0, 2.0, 99.0]),
            count=2,
        )
        material = rt.DfrMaterial(
            eta_r=strided_f32([1.0, 1.0]),
            sigma=strided_f32([0.0, 0.0]),
            mu_r=strided_f32([1.0, 1.0]),
            gain=strided_f32([1.0, 0.25]),
            valid=strided_bool([True, False]),
        )
        active = strided_bool([True, False, True])

        expected_states = rt.DfrStates(
            edge_index=states.edge_index[:2].contiguous(),
            edge_pos=states.edge_pos[:2].contiguous(),
            edge_dir=states.edge_dir[:2].contiguous(),
            edge_t_min=states.edge_t_min[:2].contiguous(),
            edge_t_max=states.edge_t_max[:2].contiguous(),
            n0=states.n0[:2].contiguous(),
            n1=states.n1[:2].contiguous(),
            prim0=states.prim0[:2].contiguous(),
            prim1=states.prim1[:2].contiguous(),
            exterior_angle=states.exterior_angle[:2].contiguous(),
            src=states.src[:2].contiguous(),
            src_power=states.src_power[:2].contiguous(),
            count=2,
        )
        expected_material = rt.DfrMaterial(
            eta_r=material.eta_r.contiguous(),
            sigma=material.sigma.contiguous(),
            mu_r=material.mu_r.contiguous(),
            gain=material.gain.contiguous(),
            valid=material.valid.contiguous(),
        )

        paths_strided = scene.trace_dfr_paths(
            tx_positions=tx_pos,
            rx_positions=rx_pos,
            states=states,
            material=material,
            active=active,
            max_paths=2,
            wavelength=1.0,
        )
        paths_contig = scene.trace_dfr_paths(
            tx_positions=tx_pos.contiguous(),
            rx_positions=rx_pos.contiguous(),
            states=expected_states,
            material=expected_material,
            active=active[:2].contiguous(),
            max_paths=2,
            wavelength=1.0,
        )

        for name in (
            "count",
            "valid",
            "tx_id",
            "rx_id",
            "order",
            "edge0",
            "delay",
            "field_x_re",
            "field_x_im",
            "p0",
        ):
            torch.testing.assert_close(getattr(paths_strided, name), getattr(paths_contig, name))

    def test_coherent_diffraction_accepts_strided_state_material_and_logical_count(self):
        def strided_vec3(rows):
            values = torch.tensor(rows, device="cuda", dtype=torch.float32)
            base = torch.full((values.shape[0], 5), 123.0, device="cuda", dtype=torch.float32)
            base[:, ::2] = values
            view = base[:, ::2]
            self.assertFalse(view.is_contiguous())
            return view

        def strided_f32(values):
            values_t = torch.tensor(values, device="cuda", dtype=torch.float32)
            base = torch.full((values_t.shape[0] * 2,), 123.0, device="cuda", dtype=torch.float32)
            base[::2] = values_t
            view = base[::2]
            self.assertFalse(view.is_contiguous())
            return view

        def strided_i32(values):
            values_t = torch.tensor(values, device="cuda", dtype=torch.int32)
            base = torch.full((values_t.shape[0] * 2,), 123, device="cuda", dtype=torch.int32)
            base[::2] = values_t
            view = base[::2]
            self.assertFalse(view.is_contiguous())
            return view

        def strided_bool(values):
            values_t = torch.tensor(values, device="cuda", dtype=torch.bool)
            base = torch.zeros((values_t.shape[0] * 2,), device="cuda", dtype=torch.bool)
            base[::2] = values_t
            view = base[::2]
            self.assertFalse(view.is_contiguous())
            return view

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
            edge_index=strided_i32([0, 0, 0]),
            edge_pos=strided_vec3([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [99.0, 99.0, 99.0]]),
            edge_dir=strided_vec3([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            edge_t_min=strided_f32([-1.0, -1.0, 99.0]),
            edge_t_max=strided_f32([1.0, 1.0, 100.0]),
            n0=strided_vec3([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
            n1=strided_vec3([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]]),
            prim0=strided_i32([0, 0, 0]),
            prim1=strided_i32([0, 0, 0]),
            exterior_angle=strided_f32([float(torch.pi), float(torch.pi), 0.5]),
            src=strided_vec3([[0.0, -1.0, 0.25], [0.0, -1.0, 0.25], [99.0, 99.0, 99.0]]),
            src_power=strided_f32([1.0, 2.0, 99.0]),
            count=2,
        )
        material = rt.DfrMaterial(
            eta_r=strided_f32([1.0, 1.0]),
            sigma=strided_f32([0.0, 0.0]),
            mu_r=strided_f32([1.0, 1.0]),
            gain=strided_f32([1.0, 0.25]),
            valid=strided_bool([True, False]),
        )
        active = strided_bool([True, False])
        grid = rt.DfrGrid(axis=2, position=0.0, resolution0=2, resolution1=2)
        expected_states = rt.DfrStates(
            edge_index=states.edge_index[:2].contiguous(),
            edge_pos=states.edge_pos[:2].contiguous(),
            edge_dir=states.edge_dir[:2].contiguous(),
            edge_t_min=states.edge_t_min[:2].contiguous(),
            edge_t_max=states.edge_t_max[:2].contiguous(),
            n0=states.n0[:2].contiguous(),
            n1=states.n1[:2].contiguous(),
            prim0=states.prim0[:2].contiguous(),
            prim1=states.prim1[:2].contiguous(),
            exterior_angle=states.exterior_angle[:2].contiguous(),
            src=states.src[:2].contiguous(),
            src_power=states.src_power[:2].contiguous(),
            count=2,
        )
        expected_material = rt.DfrMaterial(
            eta_r=material.eta_r.contiguous(),
            sigma=material.sigma.contiguous(),
            mu_r=material.mu_r.contiguous(),
            gain=material.gain.contiguous(),
            valid=material.valid.contiguous(),
        )

        direct_strided = scene.accum_dfr_direct(
            states=states,
            grid=grid,
            material=material,
            active=active,
            wavelength=1.0,
            direct_samples=4,
            seed=17,
        )
        direct_expected = scene.accum_dfr_direct(
            states=expected_states,
            grid=grid,
            material=expected_material,
            active=active.contiguous(),
            wavelength=1.0,
            direct_samples=4,
            seed=17,
        )
        for name in (
            "power",
            "field_x_re",
            "field_x_im",
            "field_y_re",
            "field_y_im",
            "field_z_re",
            "field_z_im",
            "direct_count",
            "vis_rejects",
            "utd_rejects",
        ):
            torch.testing.assert_close(getattr(direct_strided, name), getattr(direct_expected, name))

        strided = scene.accum_dfr_coherent_direct(
            states=states,
            grid=grid,
            material=material,
            active=active,
            wavelength=1.0,
        )
        expected = scene.accum_dfr_coherent_direct(
            states=expected_states,
            grid=grid,
            material=expected_material,
            active=active.contiguous(),
            wavelength=1.0,
        )

        for name in (
            "direct_field_x_re",
            "direct_field_x_im",
            "multi_field_x_re",
            "multi_field_x_im",
            "direct_count",
            "multi_count",
            "visibility_reject_count",
            "utd_reject_count",
        ):
            torch.testing.assert_close(getattr(strided, name), getattr(expected, name))

    def test_chain_diffraction_noad_accepts_strided_states_material_and_logical_count(self):
        def strided_vec3(rows):
            values = torch.tensor(rows, device="cuda", dtype=torch.float32)
            base = torch.full((values.shape[0], 5), 123.0, device="cuda", dtype=torch.float32)
            base[:, ::2] = values
            view = base[:, ::2]
            self.assertFalse(view.is_contiguous())
            return view

        def strided_f32(values):
            values_t = torch.tensor(values, device="cuda", dtype=torch.float32)
            base = torch.full((values_t.shape[0] * 2,), 123.0, device="cuda", dtype=torch.float32)
            base[::2] = values_t
            view = base[::2]
            self.assertFalse(view.is_contiguous())
            return view

        def strided_i32(values):
            values_t = torch.tensor(values, device="cuda", dtype=torch.int32)
            base = torch.full((values_t.shape[0] * 2,), 123, device="cuda", dtype=torch.int32)
            base[::2] = values_t
            view = base[::2]
            self.assertFalse(view.is_contiguous())
            return view

        def strided_bool(values):
            values_t = torch.tensor(values, device="cuda", dtype=torch.bool)
            base = torch.zeros((values_t.shape[0] * 2,), device="cuda", dtype=torch.bool)
            base[::2] = values_t
            view = base[::2]
            self.assertFalse(view.is_contiguous())
            return view

        def make_states(offset: float) -> rt.DfrStates:
            return rt.DfrStates(
                edge_index=strided_i32([0, 0, 0]),
                edge_pos=strided_vec3([[0.0 + offset, 0.0, 0.0], [0.25 + offset, 0.0, 0.0], [99.0, 99.0, 99.0]]),
                edge_dir=strided_vec3([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
                edge_t_min=strided_f32([-1.0, -1.0, 99.0]),
                edge_t_max=strided_f32([1.0, 1.0, 100.0]),
                n0=strided_vec3([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
                n1=strided_vec3([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]]),
                prim0=strided_i32([0, 0, 0]),
                prim1=strided_i32([0, 0, 0]),
                exterior_angle=strided_f32([float(torch.pi), float(torch.pi), 0.5]),
                src=strided_vec3([[0.0, -1.0, 0.25], [0.0, -1.0, 0.25], [99.0, 99.0, 99.0]]),
                src_power=strided_f32([1.0, 2.0, 99.0]),
                count=2,
            )

        def contig_states(states: rt.DfrStates) -> rt.DfrStates:
            return rt.DfrStates(
                edge_index=states.edge_index[:2].contiguous(),
                edge_pos=states.edge_pos[:2].contiguous(),
                edge_dir=states.edge_dir[:2].contiguous(),
                edge_t_min=states.edge_t_min[:2].contiguous(),
                edge_t_max=states.edge_t_max[:2].contiguous(),
                n0=states.n0[:2].contiguous(),
                n1=states.n1[:2].contiguous(),
                prim0=states.prim0[:2].contiguous(),
                prim1=states.prim1[:2].contiguous(),
                exterior_angle=states.exterior_angle[:2].contiguous(),
                src=states.src[:2].contiguous(),
                src_power=states.src_power[:2].contiguous(),
                count=2,
            )

        verts = torch.tensor(
            [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
        )
        faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(verts, faces))
        scene.build()
        initial_states = make_states(0.0)
        recursive_states = make_states(0.1)
        material = rt.DfrMaterial(
            eta_r=strided_f32([1.0, 1.0]),
            sigma=strided_f32([0.0, 0.0]),
            mu_r=strided_f32([1.0, 1.0]),
            gain=strided_f32([1.0, 0.25]),
            valid=strided_bool([True, False]),
        )
        active = strided_bool([True, True])
        recursive_active = strided_bool([True, True])
        grid = rt.DfrGrid(axis=2, position=0.0, resolution0=2, resolution1=2)
        expected_material = rt.DfrMaterial(
            eta_r=material.eta_r.contiguous(),
            sigma=material.sigma.contiguous(),
            mu_r=material.mu_r.contiguous(),
            gain=material.gain.contiguous(),
            valid=material.valid.contiguous(),
        )
        expected = scene.accum_dfr(
            initial_states=contig_states(initial_states),
            recursive_states=contig_states(recursive_states),
            grid=grid,
            material=expected_material,
            active=active.contiguous(),
            recursive_active=recursive_active.contiguous(),
            wavelength=1.0,
            direct_samples=4,
            seed=23,
            max_order=2,
        )
        with mock.patch.object(
            torch.Tensor,
            "contiguous",
            side_effect=AssertionError("Scene.accum_dfr() no-AD chain path must not stage states in Python."),
        ):
            strided = scene.accum_dfr(
                initial_states=initial_states,
                recursive_states=recursive_states,
                grid=grid,
                material=material,
                active=active,
                recursive_active=recursive_active,
                wavelength=1.0,
                direct_samples=4,
                seed=23,
                max_order=2,
            )
        for name in (
            "power",
            "field_x_re",
            "field_x_im",
            "field_y_re",
            "field_y_im",
            "field_z_re",
            "field_z_im",
            "direct_count",
            "keller_count",
            "suffix_count",
            "vis_rejects",
            "edge_vis_rejects",
            "utd_rejects",
            "edge_uses",
        ):
            torch.testing.assert_close(getattr(strided, name), getattr(expected, name))

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
        material = scene._default_dfr_material(like=verts0)
        self.assertEqual(tuple(material.gain.shape), (2,))
        self.assertEqual(tuple(material.valid.shape), (2,))
        torch.testing.assert_close(material.eta_r, torch.ones((2,), device="cuda"))
        torch.testing.assert_close(material.sigma, torch.zeros((2,), device="cuda"))
        torch.testing.assert_close(material.mu_r, torch.ones((2,), device="cuda"))

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

        out = torch.ops.raydn.diffraction_accumulation_forward(
            scene._native_scene,
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
            1,
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
            0,
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
        material = rt.DfrMaterial(
            eta_r=torch.ones((1,), device="cuda", dtype=torch.float32),
            sigma=torch.zeros((1,), device="cuda", dtype=torch.float32),
            mu_r=torch.ones((1,), device="cuda", dtype=torch.float32),
            gain=torch.ones((1,), device="cuda", dtype=torch.float32),
            valid=torch.ones((1,), device="cuda", dtype=torch.bool),
        )
        out = scene.accum_dfr_direct(states=states, grid=grid, material=material, wavelength=1.0, direct_samples=4)
        self.assertEqual(out.power.shape, (4, 4))
        self.assertEqual(out.field_x_re.dtype, torch.float32)
        zero_vec = torch.zeros_like(states.edge_pos)
        explicit_zero_states = rt.DfrStates(
            edge_index=states.edge_index,
            edge_pos=states.edge_pos,
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
            wi=zero_vec,
            d0=zero_vec,
            count=states.count,
        )
        explicit_out = scene.accum_dfr_direct(
            states=explicit_zero_states,
            grid=grid,
            material=material,
            wavelength=1.0,
            direct_samples=4,
            keller_samples=4,
            seed=11,
        )
        with (
            mock.patch(
                "torch.empty",
                side_effect=AssertionError("Scene.accum_dfr_direct() no-AD path must not create Python empty sentinels."),
            ),
            mock.patch(
                "torch.zeros_like",
                side_effect=AssertionError("Scene.accum_dfr_direct() no-AD path must not fill missing state vectors in Python."),
            ),
        ):
            missing_out = scene.accum_dfr_direct(
                states=states,
                grid=grid,
                material=material,
                wavelength=1.0,
                direct_samples=4,
                keller_samples=4,
                seed=11,
            )
        torch.testing.assert_close(missing_out.power, explicit_out.power)
        torch.testing.assert_close(missing_out.field_x_re, explicit_out.field_x_re)

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
        grid = rt.DfrGrid(axis=2, position=-1.0, resolution0=2, resolution1=2)
        out = scene.accum_dfr_direct(
            states=states,
            grid=grid,
            material=material,
            wavelength=0.125,
            seed=17,
            direct_samples=64,
        )
        upstream_field = torch.arange(
            1,
            out.field_x_re.numel() + 1,
            device="cuda",
            dtype=torch.float32,
        ).reshape_as(out.field_x_re)
        with (
            mock.patch(
                "torch.zeros",
                side_effect=AssertionError("Scene.accum_dfr_direct() backward must not fill missing upstreams in Python."),
            ),
            mock.patch(
                "torch.zeros_like",
                side_effect=AssertionError("Scene.accum_dfr_direct() backward must not fill missing upstreams in Python."),
            ),
        ):
            field_grads = torch.autograd.grad(
                out.field_x_re,
                (edge_pos, gain),
                grad_outputs=upstream_field,
                retain_graph=True,
            )
        for grad in field_grads:
            self.assertIsNotNone(grad)
            self.assertTrue(bool(torch.isfinite(grad).all().item()))
        (out.power.sum() + out.field_x_re.sum()).backward()
        for tensor in (edge_pos, edge_dir, edge_t_min, edge_t_max, exterior_angle, src, src_power, wi, gain):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(bool(torch.isfinite(tensor.grad).all().item()))

    def test_scene_accum_dfr_direct_backward_accepts_strided_states_material_and_upstream_grads(self):
        def strided_vec3_leaf(rows):
            values = torch.tensor(rows, device="cuda", dtype=torch.float32)
            base = torch.zeros((values.shape[0], 6), device="cuda", dtype=torch.float32)
            base[:, 0] = values[:, 0]
            base[:, 2] = values[:, 1]
            base[:, 4] = values[:, 2]
            base.requires_grad_()
            view = base[:, ::2]
            self.assertFalse(view.is_contiguous())
            return view

        def strided_f32_leaf(values):
            dense = torch.tensor(values, device="cuda", dtype=torch.float32)
            base = torch.zeros((dense.shape[0] * 2,), device="cuda", dtype=torch.float32)
            base[::2] = dense
            base.requires_grad_()
            view = base[::2]
            self.assertFalse(view.is_contiguous())
            return view

        def strided_f32(values):
            dense = torch.tensor(values, device="cuda", dtype=torch.float32)
            base = torch.zeros((dense.shape[0] * 2,), device="cuda", dtype=torch.float32)
            base[::2] = dense
            view = base[::2]
            self.assertFalse(view.is_contiguous())
            return view

        def strided_i32(values):
            dense = torch.tensor(values, device="cuda", dtype=torch.int32)
            base = torch.zeros((dense.shape[0] * 2,), device="cuda", dtype=torch.int32)
            base[::2] = dense
            view = base[::2]
            self.assertFalse(view.is_contiguous())
            return view

        def strided_bool(values):
            dense = torch.tensor(values, device="cuda", dtype=torch.bool)
            base = torch.zeros((dense.shape[0] * 2,), device="cuda", dtype=torch.bool)
            base[::2] = dense
            view = base[::2]
            self.assertFalse(view.is_contiguous())
            return view

        def strided_grid(rows):
            dense = torch.tensor(rows, device="cuda", dtype=torch.float32)
            base = torch.zeros((dense.shape[0], dense.shape[1] * 2), device="cuda", dtype=torch.float32)
            base[:, ::2] = dense
            view = base[:, ::2]
            self.assertFalse(view.is_contiguous())
            return view

        verts = torch.tensor(
            [[-1.0, -1.0, 10.0], [1.0, -1.0, 10.0], [-1.0, 1.0, 10.0]],
            device="cuda",
            dtype=torch.float32,
        )
        faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(verts, faces))
        scene.build()

        edge_pos = strided_vec3_leaf([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [99.0, 99.0, 99.0]])
        edge_dir = strided_vec3_leaf([[1.0, 0.0, 0.0], [1.0, 0.1, 0.0], [0.0, 1.0, 0.0]])
        edge_t_min = strided_f32_leaf([-0.5, -0.4, 99.0])
        edge_t_max = strided_f32_leaf([0.5, 0.6, 100.0])
        exterior_angle = strided_f32_leaf([1.5 * torch.pi, 1.25 * torch.pi, 0.5])
        src = strided_vec3_leaf([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [99.0, 99.0, 99.0]])
        src_power = strided_f32_leaf([2.0, 1.5, 99.0])
        wi = strided_vec3_leaf([[0.0, 0.0, -1.0], [0.0, -0.1, -1.0], [1.0, 0.0, 0.0]])
        gain = strided_f32_leaf([1.0, 0.25])
        states = rt.DfrStates(
            edge_index=strided_i32([0, 0, 0]),
            edge_pos=edge_pos,
            edge_dir=edge_dir,
            edge_t_min=edge_t_min,
            edge_t_max=edge_t_max,
            n0=strided_vec3_leaf([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]).detach(),
            n1=strided_vec3_leaf([[0.0, -1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]).detach(),
            prim0=strided_i32([0, 0, 0]),
            prim1=strided_i32([0, 0, 0]),
            exterior_angle=exterior_angle,
            src=src,
            src_power=src_power,
            wi=wi,
            d0=strided_vec3_leaf([[0.0, 0.0, -1.0], [0.0, -0.1, -1.0], [1.0, 0.0, 0.0]]).detach(),
            count=2,
        )
        material = rt.DfrMaterial(
            eta_r=strided_f32([4.0, 4.0]),
            sigma=strided_f32([0.0, 0.0]),
            mu_r=strided_f32([1.0, 1.0]),
            gain=gain,
            valid=strided_bool([True, False]),
        )
        grid = rt.DfrGrid(axis=2, position=-1.0, resolution0=2, resolution1=2)
        upstream_power = strided_grid([[0.5, -1.0], [1.25, 0.75]])
        upstream_field = strided_grid([[1.5, -0.25], [0.5, 2.0]])
        inputs = (edge_pos, edge_dir, edge_t_min, edge_t_max, exterior_angle, src, src_power, wi, gain)
        with mock.patch.object(
            torch.Tensor,
            "contiguous",
            side_effect=AssertionError("Scene.accum_dfr_direct() AD path must not stage strided tensors in Python."),
        ):
            out = scene.accum_dfr_direct(
                states=states,
                grid=grid,
                material=material,
                active=strided_bool([True, True]),
                wavelength=0.125,
                seed=17,
                direct_samples=8,
            )
            strided_grads = torch.autograd.grad(
                (out.power, out.field_x_re),
                inputs,
                grad_outputs=(upstream_power, upstream_field),
            )

        expected_states = rt.DfrStates(
            edge_index=states.edge_index[:2].contiguous(),
            edge_pos=edge_pos[:2].detach().contiguous().requires_grad_(),
            edge_dir=edge_dir[:2].detach().contiguous().requires_grad_(),
            edge_t_min=edge_t_min[:2].detach().contiguous().requires_grad_(),
            edge_t_max=edge_t_max[:2].detach().contiguous().requires_grad_(),
            n0=states.n0[:2].contiguous(),
            n1=states.n1[:2].contiguous(),
            prim0=states.prim0[:2].contiguous(),
            prim1=states.prim1[:2].contiguous(),
            exterior_angle=exterior_angle[:2].detach().contiguous().requires_grad_(),
            src=src[:2].detach().contiguous().requires_grad_(),
            src_power=src_power[:2].detach().contiguous().requires_grad_(),
            wi=wi[:2].detach().contiguous().requires_grad_(),
            d0=states.d0[:2].contiguous(),
            count=2,
        )
        expected_gain = gain.detach().contiguous().requires_grad_()
        expected_material = rt.DfrMaterial(
            eta_r=material.eta_r.contiguous(),
            sigma=material.sigma.contiguous(),
            mu_r=material.mu_r.contiguous(),
            gain=expected_gain,
            valid=material.valid.contiguous(),
        )
        expected_out = scene.accum_dfr_direct(
            states=expected_states,
            grid=grid,
            material=expected_material,
            active=torch.tensor([True, True], device="cuda", dtype=torch.bool),
            wavelength=0.125,
            seed=17,
            direct_samples=8,
        )
        expected_inputs = (
            expected_states.edge_pos,
            expected_states.edge_dir,
            expected_states.edge_t_min,
            expected_states.edge_t_max,
            expected_states.exterior_angle,
            expected_states.src,
            expected_states.src_power,
            expected_states.wi,
            expected_gain,
        )
        expected_grads = torch.autograd.grad(
            (expected_out.power, expected_out.field_x_re),
            expected_inputs,
            grad_outputs=(upstream_power.contiguous(), upstream_field.contiguous()),
        )
        for strided_grad, expected_grad in zip(strided_grads[:-1], expected_grads[:-1]):
            torch.testing.assert_close(strided_grad[:2], expected_grad, atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(strided_grad[2], torch.zeros_like(strided_grad[2]))
        torch.testing.assert_close(strided_grads[-1], expected_grads[-1], atol=2e-5, rtol=2e-5)

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
            with mock.patch(
                "torch.zeros_like",
                side_effect=AssertionError("Scene.accum_dfr_direct() jvp must not fill tangents in Python."),
            ):
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
        grid = rt.DfrGrid(axis=2, position=-1.0, resolution0=2, resolution1=2)
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
        upstream_field = torch.arange(
            1,
            out.field_x_re.numel() + 1,
            device="cuda",
            dtype=torch.float32,
        ).reshape_as(out.field_x_re)
        with (
            mock.patch(
                "torch.zeros",
                side_effect=AssertionError("Scene.accum_dfr() backward must not fill missing upstreams in Python."),
            ),
            mock.patch(
                "torch.zeros_like",
                side_effect=AssertionError("Scene.accum_dfr() backward must not fill missing upstreams in Python."),
            ),
        ):
            field_grads = torch.autograd.grad(
                out.field_x_re,
                (edge_pos, rec_edge_pos, gain),
                grad_outputs=upstream_field,
                retain_graph=True,
            )
        for grad in field_grads:
            self.assertIsNotNone(grad)
            self.assertTrue(bool(torch.isfinite(grad).all().item()))
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
            with mock.patch(
                "torch.zeros_like",
                side_effect=AssertionError("Scene.accum_dfr() jvp must not fill tangents in Python."),
            ):
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

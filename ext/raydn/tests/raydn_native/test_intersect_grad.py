import unittest
from unittest import mock

import torch
import raydn as rt


@unittest.skipUnless(torch.cuda.is_available(), "CUDA torch is required")
class IntersectGradientTests(unittest.TestCase):
    def test_vertex_gradient_exact_values_through_t(self):
        verts = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(verts, faces))
        scene.build()
        ray = rt.Ray(
            torch.tensor([[0.25, 0.25, -1.0]], device="cuda", dtype=torch.float32),
            torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32),
        )
        its = scene.intersect(ray)
        its.t.sum().backward()
        torch.testing.assert_close(
            verts.grad[:, 2],
            torch.tensor([0.5, 0.25, 0.25], device="cuda"),
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(verts.grad[:, 0], torch.zeros(3, device="cuda"))
        torch.testing.assert_close(verts.grad[:, 1], torch.zeros(3, device="cuda"))

    def test_rayflags_none_backward_through_t_uses_hidden_tape(self):
        verts = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(verts, faces))
        scene.build()
        ray = rt.Ray(
            torch.tensor([[0.25, 0.25, -1.0]], device="cuda", dtype=torch.float32),
            torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32),
        )
        flags_none = getattr(rt.RayFlags, "None")

        its = scene.intersect(ray, flags=flags_none)
        self.assertEqual(tuple(its.p.shape), (0, 3))
        self.assertEqual(tuple(its.shape_id.shape), (0,))
        its.t.sum().backward()

        torch.testing.assert_close(
            verts.grad[:, 2],
            torch.tensor([0.5, 0.25, 0.25], device="cuda"),
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(verts.grad[:, 0], torch.zeros(3, device="cuda"))
        torch.testing.assert_close(verts.grad[:, 1], torch.zeros(3, device="cuda"))

    def test_public_t_backward_accepts_nonuniform_upstream_gradient(self):
        faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
        ray = rt.Ray(
            torch.tensor(
                [[0.25, 0.25, -1.0], [0.50, 0.25, -1.0], [0.25, 0.50, -1.0]],
                device="cuda",
                dtype=torch.float32,
            ),
            torch.tensor(
                [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
                device="cuda",
                dtype=torch.float32,
            ),
        )
        upstream = torch.tensor([0.25, -1.5, 2.0], device="cuda", dtype=torch.float32)
        base = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
        )
        public_verts = base.clone().detach().requires_grad_(True)
        public_scene = rt.Scene()
        public_scene.add_mesh(rt.Mesh(public_verts, faces))
        public_scene.build()
        public_t = public_scene.intersect(ray).t
        public_t.backward(upstream)

        native_scene = rt.Scene()
        native_scene.add_mesh(rt.Mesh(base.clone(), faces))
        native_scene.build()
        active = torch.ones((ray.o.shape[0],), device="cuda", dtype=torch.bool)
        flags_none = getattr(rt.RayFlags, "None")
        values = torch.ops.raydn.intersect_forward_ad_flags(
            native_scene._require_native_scene(),
            ray.o,
            ray.d,
            ray.tmax,
            active,
            int(flags_none),
        )
        expected = torch.ops.raydn.intersect_backward_t(
            native_scene._require_native_scene(),
            ray.o,
            ray.d,
            active,
            values[10],
            values[11],
            upstream,
            True,
            False,
            False,
            False,
        )[0]

        torch.testing.assert_close(public_verts.grad, expected, atol=1e-5, rtol=1e-5)

    def test_native_t_backward_accepts_expanded_grad_t(self):
        verts = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
        )
        faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(verts, faces))
        scene.build()
        ray = rt.Ray(
            torch.tensor(
                [[0.25, 0.25, -1.0], [0.50, 0.25, -1.0], [0.25, 0.50, -1.0]],
                device="cuda",
                dtype=torch.float32,
            ),
            torch.tensor(
                [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
                device="cuda",
                dtype=torch.float32,
            ),
        )
        active = torch.ones((ray.o.shape[0],), device="cuda", dtype=torch.bool)
        flags_none = getattr(rt.RayFlags, "None")
        values = torch.ops.raydn.intersect_forward_ad_flags(
            scene._require_native_scene(),
            ray.o,
            ray.d,
            ray.tmax,
            active,
            int(flags_none),
        )
        tape_prim_id = values[10]
        tape_barycentric = values[11]

        contiguous_grad = torch.ones((ray.o.shape[0],), device="cuda", dtype=torch.float32)
        expanded_grad = torch.ones((), device="cuda", dtype=torch.float32).expand(ray.o.shape[0])
        self.assertFalse(expanded_grad.is_contiguous())
        expected = torch.ops.raydn.intersect_backward_t(
            scene._require_native_scene(),
            ray.o,
            ray.d,
            active,
            tape_prim_id,
            tape_barycentric,
            contiguous_grad,
            True,
            False,
            False,
            False,
        )[0]
        actual = torch.ops.raydn.intersect_backward_t(
            scene._require_native_scene(),
            ray.o,
            ray.d,
            active,
            tape_prim_id,
            tape_barycentric,
            expanded_grad,
            True,
            False,
            False,
            False,
        )[0]
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_ray_origin_gradient_through_t(self):
        verts = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
        )
        faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(verts, faces))
        scene.build()
        origin = torch.tensor([[0.25, 0.25, -1.0]], device="cuda", dtype=torch.float32, requires_grad=True)
        direction = torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32)
        its = scene.intersect(rt.Ray(origin, direction))
        its.t.sum().backward()
        torch.testing.assert_close(origin.grad, torch.tensor([[0.0, 0.0, -1.0]], device="cuda"), atol=1e-5, rtol=1e-5)

    def test_arbitrary_triangle_vertex_grad_matches_finite_difference(self):
        base = torch.tensor(
            [[-0.2, 0.1, 0.3], [1.2, -0.1, 0.4], [0.1, 0.9, -0.2]],
            device="cuda",
            dtype=torch.float32,
        )
        faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
        ray = rt.Ray(
            torch.tensor([[0.25, 0.20, -2.0]], device="cuda", dtype=torch.float32),
            torch.tensor([[0.02, -0.01, 1.0]], device="cuda", dtype=torch.float32),
        )

        verts = base.clone().detach().requires_grad_(True)
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(verts, faces))
        scene.build()
        loss = scene.intersect(ray).t.sum()
        loss.backward()
        analytic = verts.grad[0, 2].detach().clone()

        eps = 1e-3
        plus = base.clone()
        plus[0, 2] += eps
        minus = base.clone()
        minus[0, 2] -= eps
        scene_p = rt.Scene()
        scene_p.add_mesh(rt.Mesh(plus, faces))
        scene_p.build()
        scene_m = rt.Scene()
        scene_m.add_mesh(rt.Mesh(minus, faces))
        scene_m.build()
        fd = (scene_p.intersect(ray).t - scene_m.intersect(ray).t) / (2 * eps)
        torch.testing.assert_close(analytic, fd[0], atol=5e-3, rtol=5e-3)

    def test_intersect_autograd_func_jvp(self):
        faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
        ray = rt.Ray(
            torch.tensor([[0.25, 0.25, -1.0]], device="cuda", dtype=torch.float32),
            torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32),
        )
        empty_uv = torch.empty((0, 2), device="cuda", dtype=torch.float32)
        empty_face_uv = torch.empty((0, 3), device="cuda", dtype=torch.int32)
        empty_transform = torch.empty((0, 4), device="cuda", dtype=torch.float32)

        def fn(verts):
            scene = rt.Scene()
            scene.add_mesh(
                rt.Mesh(
                    verts,
                    faces,
                    uv=empty_uv,
                    face_uv=empty_face_uv,
                    to_world_left=empty_transform,
                    to_world_right=empty_transform,
                )
            )
            scene.build()
            return scene.intersect(ray).t

        verts = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
        )
        tangent = torch.tensor(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
        )
        with mock.patch.object(
            torch.Tensor,
            "new_empty",
            side_effect=AssertionError("Scene.intersect() JVP must not create inactive output sentinels in Python."),
        ), mock.patch(
            "torch.empty",
            side_effect=AssertionError("Scene.intersect() JVP must not create active mask sentinels in Python."),
        ):
            primal, jvp = torch.func.jvp(fn, (verts,), (tangent,))
        torch.testing.assert_close(primal, torch.tensor([1.0], device="cuda"))
        torch.testing.assert_close(jvp, torch.tensor([0.5], device="cuda"), atol=1e-5, rtol=1e-5)

    def test_rayflags_none_intersect_jvp(self):
        faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
        ray = rt.Ray(
            torch.tensor([[0.25, 0.25, -1.0]], device="cuda", dtype=torch.float32),
            torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32),
        )
        flags_none = getattr(rt.RayFlags, "None")

        def fn(verts):
            scene = rt.Scene()
            scene.add_mesh(rt.Mesh(verts, faces))
            scene.build()
            return scene.intersect(ray, flags=flags_none).t

        verts = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
        )
        tangent = torch.tensor(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
        )
        primal, jvp = torch.func.jvp(fn, (verts,), (tangent,))
        torch.testing.assert_close(primal, torch.tensor([1.0], device="cuda"))
        torch.testing.assert_close(jvp, torch.tensor([0.5], device="cuda"), atol=1e-5, rtol=1e-5)

    def test_intersect_jvp_accepts_noncontiguous_tangent_vertices(self):
        faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
        ray = rt.Ray(
            torch.tensor([[0.25, 0.25, -1.0]], device="cuda", dtype=torch.float32),
            torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32),
        )

        def fn(verts):
            scene = rt.Scene()
            scene.add_mesh(rt.Mesh(verts, faces))
            scene.build()
            return scene.intersect(ray).t

        verts = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
        )
        tangent = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
        ).t()
        self.assertFalse(tangent.is_contiguous())
        primal, jvp = torch.func.jvp(fn, (verts,), (tangent,))
        torch.testing.assert_close(primal, torch.tensor([1.0], device="cuda"))
        torch.testing.assert_close(jvp, torch.tensor([0.5], device="cuda"), atol=1e-5, rtol=1e-5)

    def test_multi_mesh_vertex_gradient_routes_to_hit_mesh(self):
        verts0 = torch.tensor(
            [[10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [10.0, 1.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        verts1 = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
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
            torch.tensor([[0.25, 0.25, -1.0]], device="cuda", dtype=torch.float32),
            torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32),
        )
        upstream = torch.tensor([2.0], device="cuda", dtype=torch.float32)
        with (
            mock.patch("torch.cat", side_effect=AssertionError("Scene.intersect() must not use torch.cat.")),
            mock.patch("torch.zeros_like", side_effect=AssertionError("Scene.intersect() must not use torch.zeros_like.")),
        ):
            scene.intersect(ray).t.backward(upstream)
        self.assertIsNotNone(verts0.grad)
        self.assertIsNotNone(verts1.grad)
        torch.testing.assert_close(verts0.grad, torch.zeros_like(verts0), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            verts1.grad[:, 2],
            torch.tensor([1.0, 0.5, 0.5], device="cuda"),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_multi_mesh_full_intersect_backward_accepts_non_sum_upstream_without_python_zero_ops(self):
        verts0 = torch.tensor(
            [[10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [10.0, 1.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        verts1 = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
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
            torch.tensor([[0.25, 0.25, -1.0]], device="cuda", dtype=torch.float32),
            torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32),
        )
        upstream_p = torch.tensor([[0.25, -0.5, 1.5]], device="cuda", dtype=torch.float32)
        with (
            mock.patch("torch.cat", side_effect=AssertionError("Scene.intersect() must not use torch.cat.")),
            mock.patch("torch.zeros_like", side_effect=AssertionError("Scene.intersect() must not use torch.zeros_like.")),
        ):
            scene.intersect(ray).p.backward(upstream_p)

        active = torch.empty((0,), device="cuda", dtype=torch.bool)
        values = torch.ops.raydn.intersect_forward_ad_flags(
            scene._require_native_scene(),
            ray.o,
            ray.d,
            ray.tmax,
            active,
            int(rt.RayFlags.All),
        )
        expected_global = torch.ops.raydn.intersect_backward_optional(
            scene._require_native_scene(),
            ray.o,
            ray.d,
            ray.tmax,
            active,
            values[10],
            values[11],
            None,
            upstream_p,
            None,
            None,
            None,
            None,
            True,
            False,
            False,
            False,
        )[0]
        expected_mesh = torch.ops.raydn.split_scene_vertex_grad(scene._require_native_scene(), expected_global)
        torch.testing.assert_close(verts0.grad, expected_mesh[0], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(verts1.grad, expected_mesh[1], atol=1e-5, rtol=1e-5)

    def test_full_intersect_backward_accepts_noncontiguous_upstream_p(self):
        faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
        ray = rt.Ray(
            torch.tensor(
                [[0.20, 0.20, -1.0], [0.55, 0.20, -1.0], [0.20, 0.55, -1.0]],
                device="cuda",
                dtype=torch.float32,
            ),
            torch.tensor(
                [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
                device="cuda",
                dtype=torch.float32,
            ),
        )
        upstream_p = torch.tensor(
            [[0.25, -0.75, 1.25], [0.50, 0.25, -1.50], [-0.20, 1.10, 0.80]],
            device="cuda",
            dtype=torch.float32,
        ).t()
        self.assertEqual(tuple(upstream_p.shape), (3, 3))
        self.assertFalse(upstream_p.is_contiguous())

        base = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
        )
        actual_verts = base.clone().detach().requires_grad_(True)
        actual_scene = rt.Scene()
        actual_scene.add_mesh(rt.Mesh(actual_verts, faces))
        actual_scene.build()
        actual_scene.intersect(ray).p.backward(upstream_p)

        expected_verts = base.clone().detach().requires_grad_(True)
        expected_scene = rt.Scene()
        expected_scene.add_mesh(rt.Mesh(expected_verts, faces))
        expected_scene.build()
        expected_scene.intersect(ray).p.backward(upstream_p.contiguous())

        torch.testing.assert_close(actual_verts.grad, expected_verts.grad, atol=1e-5, rtol=1e-5)

    def test_multi_mesh_intersect_jvp_uses_global_vertices(self):
        faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
        ray = rt.Ray(
            torch.tensor([[0.25, 0.25, -1.0]], device="cuda", dtype=torch.float32),
            torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32),
        )
        empty_uv = torch.empty((0, 2), device="cuda", dtype=torch.float32)
        empty_face_uv = torch.empty((0, 3), device="cuda", dtype=torch.int32)
        empty_transform = torch.empty((0, 4), device="cuda", dtype=torch.float32)

        def fn(verts0, verts1):
            scene = rt.Scene()
            scene.add_mesh(
                rt.Mesh(
                    verts0,
                    faces,
                    uv=empty_uv,
                    face_uv=empty_face_uv,
                    to_world_left=empty_transform,
                    to_world_right=empty_transform,
                )
            )
            scene.add_mesh(
                rt.Mesh(
                    verts1,
                    faces,
                    uv=empty_uv,
                    face_uv=empty_face_uv,
                    to_world_left=empty_transform,
                    to_world_right=empty_transform,
                )
            )
            scene.build()
            return scene.intersect(ray).t

        verts0 = torch.tensor(
            [[10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [10.0, 1.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
        )
        verts1 = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
        )
        tangent0 = torch.zeros_like(verts0)
        tangent1 = torch.tensor(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
        )
        with mock.patch.object(
            torch.Tensor,
            "new_empty",
            side_effect=AssertionError("Scene.intersect() multi-mesh JVP must not create output sentinels in Python."),
        ), mock.patch(
            "torch.empty",
            side_effect=AssertionError("Scene.intersect() multi-mesh JVP must not create active mask sentinels in Python."),
        ):
            primal, jvp = torch.func.jvp(fn, (verts0, verts1), (tangent0, tangent1))
        torch.testing.assert_close(primal, torch.tensor([1.0], device="cuda"))
        torch.testing.assert_close(jvp, torch.tensor([0.5], device="cuda"), atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()

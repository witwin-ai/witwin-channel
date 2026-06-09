import unittest

import torch
import raydtorch as rt


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
            [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
        )
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
        scene.intersect(ray).t.sum().backward()
        self.assertIsNotNone(verts0.grad)
        self.assertIsNotNone(verts1.grad)
        torch.testing.assert_close(verts0.grad, torch.zeros_like(verts0), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            verts1.grad[:, 2],
            torch.tensor([0.5, 0.25, 0.25], device="cuda"),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_multi_mesh_intersect_jvp_uses_global_vertices(self):
        faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
        ray = rt.Ray(
            torch.tensor([[0.25, 0.25, -1.0]], device="cuda", dtype=torch.float32),
            torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32),
        )

        def fn(verts0, verts1):
            scene = rt.Scene()
            scene.add_mesh(rt.Mesh(verts0, faces))
            scene.add_mesh(rt.Mesh(verts1, faces))
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
        primal, jvp = torch.func.jvp(fn, (verts0, verts1), (tangent0, tangent1))
        torch.testing.assert_close(primal, torch.tensor([1.0], device="cuda"))
        torch.testing.assert_close(jvp, torch.tensor([0.5], device="cuda"), atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()

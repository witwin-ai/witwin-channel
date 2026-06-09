import unittest

import torch
import raydtorch as rt


@unittest.skipUnless(torch.cuda.is_available(), "CUDA torch is required")
class EdgeQueryTests(unittest.TestCase):
    def test_nearest_edge_point_forward_and_grad(self):
        verts = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
        point = torch.tensor([[0.5, -0.25, 0.0]], device="cuda", dtype=torch.float32, requires_grad=True)
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(verts, faces))
        scene.build()
        result = scene.nearest_edge(point)
        torch.testing.assert_close(result.distance, torch.tensor([0.25], device="cuda"), atol=1e-5, rtol=1e-5)
        result.distance.sum().backward()
        self.assertIsNotNone(point.grad)
        self.assertIsNotNone(verts.grad)

    def test_multi_mesh_nearest_edge_gradient_routes_to_hit_mesh(self):
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
        point = torch.tensor([[0.5, -0.25, 0.0]], device="cuda", dtype=torch.float32)
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(verts0, faces))
        scene.add_mesh(rt.Mesh(verts1, faces))
        scene.build()
        scene.nearest_edge(point).distance.sum().backward()
        self.assertIsNotNone(verts0.grad)
        self.assertIsNotNone(verts1.grad)
        torch.testing.assert_close(verts0.grad, torch.zeros_like(verts0), atol=1e-5, rtol=1e-5)
        self.assertGreater(float(verts1.grad.abs().sum().item()), 0.0)

    def test_multi_mesh_nearest_edge_jvp_uses_global_vertices(self):
        faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
        point = torch.tensor([[0.5, -0.25, 0.0]], device="cuda", dtype=torch.float32)

        def fn(verts0, verts1):
            scene = rt.Scene()
            scene.add_mesh(rt.Mesh(verts0, faces))
            scene.add_mesh(rt.Mesh(verts1, faces))
            scene.build()
            return scene.nearest_edge(point).edge_point

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
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
        )
        primal, jvp = torch.func.jvp(fn, (verts0, verts1), (tangent0, tangent1))
        torch.testing.assert_close(primal, torch.tensor([[0.5, 0.0, 0.0]], device="cuda"), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(jvp, torch.tensor([[0.0, 0.0, 1.0]], device="cuda"), atol=1e-5, rtol=1e-5)

    def test_nearest_edge_point_edge_t_vjp_matches_interior_edge(self):
        verts = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
        point = torch.tensor([[0.25, 0.2, 0.0]], device="cuda", dtype=torch.float32, requires_grad=True)
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(verts, faces))
        scene.build()
        result = scene.nearest_edge(point)
        self.assertEqual(int(result.edge_id[0].item()), 0)
        result.edge_t.sum().backward()
        torch.testing.assert_close(point.grad, torch.tensor([[1.0, 0.0, 0.0]], device="cuda"))

    def test_nearest_edge_point_edge_point_vjp_reaches_query_point(self):
        verts = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
        point = torch.tensor([[0.25, 0.2, 0.0]], device="cuda", dtype=torch.float32, requires_grad=True)
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(verts, faces))
        scene.build()
        result = scene.nearest_edge(point)
        self.assertEqual(int(result.edge_id[0].item()), 0)
        result.edge_point[:, 0].sum().backward()
        torch.testing.assert_close(point.grad, torch.tensor([[1.0, 0.0, 0.0]], device="cuda"))

    def test_nearest_edge_ray_forward(self):
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
            torch.tensor([[0.5, -0.25, 1.0]], device="cuda", dtype=torch.float32),
            torch.tensor([[0.0, 0.0, -1.0]], device="cuda", dtype=torch.float32),
        )
        result = scene.nearest_edge(ray)
        self.assertIsInstance(result, rt.NearestRayEdge)
        torch.testing.assert_close(result.distance, torch.tensor([0.25], device="cuda"), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(result.ray_t, torch.tensor([1.0], device="cuda"), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            result.point,
            torch.tensor([[0.5, -0.25, 0.0]], device="cuda"),
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(result.edge_t, torch.tensor([0.5], device="cuda"), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            result.edge_point,
            torch.tensor([[0.5, 0.0, 0.0]], device="cuda"),
            atol=1e-5,
            rtol=1e-5,
        )
        self.assertEqual(int(result.edge_id[0].item()), 0)

    def test_large_grid_edge_query_returns_finite_distances(self):
        n = 64
        xs, ys = torch.meshgrid(
            torch.linspace(0, 1, n, device="cuda"),
            torch.linspace(0, 1, n, device="cuda"),
            indexing="ij",
        )
        verts = torch.stack([xs.reshape(-1), ys.reshape(-1), torch.zeros(n * n, device="cuda")], dim=1).contiguous()
        faces = []
        for i in range(n - 1):
            for j in range(n - 1):
                a = i * n + j
                b = a + 1
                c = a + n
                d = c + 1
                faces.append([a, b, c])
                faces.append([b, d, c])
        faces_t = torch.tensor(faces, device="cuda", dtype=torch.int32)
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(verts, faces_t))
        scene.build()
        q = torch.rand((4096, 3), device="cuda", dtype=torch.float32)
        out = scene.nearest_edge(q)
        self.assertTrue(torch.isfinite(out.distance).all().item())

    def test_edges_disabled_mesh_has_no_nearest_edge_hits(self):
        verts = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
        )
        faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
        point = torch.tensor([[0.5, -0.25, 0.0]], device="cuda", dtype=torch.float32)
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(verts, faces, edges_enabled=False))
        scene.build()
        out = scene.nearest_edge(point)
        self.assertTrue(torch.isinf(out.distance).all().item())
        self.assertEqual(int(out.global_edge_id[0].item()), -1)

    def test_nonmanifold_edge_uses_rayd_wedge_count(self):
        verts = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            device="cuda",
            dtype=torch.float32,
        )
        faces = torch.tensor(
            [
                [0, 1, 2],
                [1, 0, 3],
                [0, 1, 4],
            ],
            device="cuda",
            dtype=torch.int32,
        )
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(verts, faces))
        scene.build()
        self.assertEqual(int(rt._C.scene_edge_count(scene._native_handle)), 9)


if __name__ == "__main__":
    unittest.main()

import unittest

import torch
import raydn as rt


@unittest.skipUnless(torch.cuda.is_available(), "CUDA torch is required")
class IntersectForwardTests(unittest.TestCase):
    def test_single_triangle_hit_and_miss(self):
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
            torch.tensor([[0.25, 0.25, -1.0], [2.0, 2.0, -1.0]], device="cuda", dtype=torch.float32),
            torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32),
        )
        its = scene.intersect(ray)
        torch.testing.assert_close(its.t[0], torch.tensor(1.0, device="cuda"))
        torch.testing.assert_close(its.p[0], torch.tensor([0.25, 0.25, 0.0], device="cuda"))
        torch.testing.assert_close(its.barycentric[0], torch.tensor([0.5, 0.25, 0.25], device="cuda"))
        self.assertEqual(int(its.shape_id[0].item()), 0)
        self.assertEqual(int(its.shape_id[1].item()), -1)
        self.assertTrue(torch.isinf(its.t[1]))

    def test_default_tmax_sentinel_matches_unbounded_tmax(self):
        verts = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
        )
        faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(verts, faces))
        scene.build()
        ray_o = torch.tensor([[0.25, 0.25, -1.0]], device="cuda", dtype=torch.float32)
        ray_d = torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32)

        default_hit = scene.intersect(rt.Ray(ray_o, ray_d))
        explicit_hit = scene.intersect(rt.Ray(ray_o, ray_d, torch.tensor([10.0], device="cuda")))
        clipped_hit = scene.intersect(rt.Ray(ray_o, ray_d, torch.tensor([0.5], device="cuda")))

        torch.testing.assert_close(default_hit.t, explicit_hit.t)
        torch.testing.assert_close(default_hit.p, explicit_hit.p)
        self.assertTrue(torch.isinf(clipped_hit.t[0]))

    def test_two_triangles_returns_nearest_hit(self):
        verts = torch.tensor(
            [
                [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                [0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [0.0, 1.0, 2.0],
            ],
            device="cuda",
            dtype=torch.float32,
        )
        faces = torch.tensor([[0, 1, 2], [3, 4, 5]], device="cuda", dtype=torch.int32)
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(verts, faces))
        scene.build()
        ray = rt.Ray(
            torch.tensor([[0.25, 0.25, -1.0]], device="cuda", dtype=torch.float32),
            torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32),
        )
        its = scene.intersect(ray)
        torch.testing.assert_close(its.t[0], torch.tensor(1.0, device="cuda"))
        self.assertEqual(int(its.global_prim_id[0].item()), 0)

    def test_intersect_rayflags_none_uses_t_only_result(self):
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
            torch.tensor([[0.25, 0.25, -1.0], [2.0, 2.0, -1.0]], device="cuda", dtype=torch.float32),
            torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32),
        )
        flags_none = getattr(rt.RayFlags, "None")

        reduced = scene.intersect(ray, flags=flags_none)
        full = scene.intersect(ray, flags=rt.RayFlags.Geometric)

        torch.testing.assert_close(reduced.t, full.t)
        self.assertEqual(reduced.p.numel(), 0)
        self.assertEqual(reduced.shape_id.numel(), 0)
        torch.testing.assert_close(
            reduced.is_valid(),
            torch.tensor([True, False], device="cuda"),
        )
        self.assertEqual(full.p.shape, (2, 3))
        self.assertEqual(full.geo_n.shape, (2, 3))
        self.assertEqual(full.n.numel(), 0)
        self.assertEqual(full.uv.numel(), 0)

        shading = scene.intersect(ray, flags=rt.RayFlags.ShadingN)
        self.assertEqual(shading.n.shape, (2, 3))
        self.assertEqual(shading.p.numel(), 0)
        self.assertEqual(shading.shape_id.numel(), 0)

        uv = scene.intersect(ray, flags=rt.RayFlags.UV)
        self.assertEqual(uv.uv.shape, (2, 2))
        self.assertEqual(uv.p.numel(), 0)
        self.assertEqual(uv.shape_id.numel(), 0)


if __name__ == "__main__":
    unittest.main()

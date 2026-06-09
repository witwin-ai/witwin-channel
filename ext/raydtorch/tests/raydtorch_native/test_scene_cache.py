import unittest

import torch
import raydtorch as rt


@unittest.skipUnless(torch.cuda.is_available(), "CUDA torch is required")
class SceneCacheTests(unittest.TestCase):
    def _mesh(self):
        verts = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
        )
        faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
        return rt.Mesh(verts, faces)

    def test_scene_build_creates_native_handle_and_version(self):
        scene = rt.Scene()
        mesh_id = scene.add_mesh(self._mesh())
        self.assertEqual(mesh_id, 0)
        scene.build()
        self.assertTrue(scene.is_ready())
        self.assertEqual(scene.num_meshes, 1)
        self.assertGreaterEqual(scene.version, 1)

    def test_query_before_build_fails(self):
        scene = rt.Scene()
        scene.add_mesh(self._mesh())
        with self.assertRaisesRegex(RuntimeError, "Call build"):
            scene.intersect(
                rt.Ray(
                    torch.zeros((1, 3), device="cuda", dtype=torch.float32),
                    torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32),
                )
            )

    def test_build_uses_current_torch_stream(self):
        scene = rt.Scene()
        scene.add_mesh(self._mesh())
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            scene.build()
        stream.synchronize()
        self.assertTrue(scene.is_ready())

    def test_dynamic_vertex_update_changes_intersection(self):
        verts = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
        )
        faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
        scene = rt.Scene()
        mesh_id = scene.add_mesh(rt.Mesh(verts, faces), dynamic=True)
        scene.build()
        ray = rt.Ray(
            torch.tensor([[0.25, 0.25, -1.0]], device="cuda", dtype=torch.float32),
            torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32),
        )
        first = scene.intersect(ray).t.detach()
        shifted = verts.clone()
        shifted[:, 2] += 1.0
        scene.update_mesh_vertices(mesh_id, shifted)
        self.assertTrue(scene.has_pending_updates())
        scene.sync()
        second = scene.intersect(ray).t.detach()
        torch.testing.assert_close(second - first, torch.tensor([1.0], device="cuda"))


if __name__ == "__main__":
    unittest.main()

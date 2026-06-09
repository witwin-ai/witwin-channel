import unittest

import torch
import raydn as rt


@unittest.skipUnless(torch.cuda.is_available(), "CUDA torch is required")
class TensorContractTests(unittest.TestCase):
    def test_mesh_requires_cuda_float32_vertices_and_int32_faces(self):
        verts = torch.zeros((3, 3), device="cuda", dtype=torch.float64)
        faces = torch.zeros((1, 3), device="cuda", dtype=torch.int64)
        with self.assertRaisesRegex(TypeError, "vertices must be torch.float32"):
            rt.Mesh(verts, faces)

        verts = torch.zeros((3, 3), device="cuda", dtype=torch.float32)
        with self.assertRaisesRegex(TypeError, "faces must be torch.int32"):
            rt.Mesh(verts, faces)

    def test_mesh_default_transforms_are_empty_identity_sentinels(self):
        verts = torch.zeros((3, 3), device="cuda", dtype=torch.float32)
        faces = torch.zeros((1, 3), device="cuda", dtype=torch.int32)
        mesh = rt.Mesh(verts, faces)
        self.assertEqual(mesh.to_world_left.shape, (0, 4))
        self.assertEqual(mesh.to_world_right.shape, (0, 4))
        self.assertEqual(mesh.to_world_left.dtype, torch.float32)
        self.assertEqual(mesh.to_world_left.device.type, "cuda")

    def test_mesh_rejects_cpu_tensors(self):
        verts = torch.zeros((3, 3), dtype=torch.float32)
        faces = torch.zeros((1, 3), dtype=torch.int32)
        with self.assertRaisesRegex(TypeError, "vertices must be CUDA"):
            rt.Mesh(verts, faces)

    def test_ray_contract(self):
        o = torch.zeros((2, 3), device="cuda", dtype=torch.float32)
        d = torch.zeros((2, 3), device="cuda", dtype=torch.float32)
        ray = rt.Ray(o, d)
        self.assertEqual(ray.tmax.shape, (0,))
        self.assertEqual(ray.tmax.dtype, torch.float32)
        self.assertEqual(ray.tmax.device.type, "cuda")


if __name__ == "__main__":
    unittest.main()

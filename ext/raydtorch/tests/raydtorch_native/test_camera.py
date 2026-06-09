import unittest

import torch
import raydtorch as rt


@unittest.skipUnless(torch.cuda.is_available(), "CUDA torch is required")
class CameraTests(unittest.TestCase):
    def test_camera_sample_ray_backward(self):
        camera = rt.Camera(width=16, height=12, fov_x=45.0)
        sample = torch.tensor([[0.5, 0.5]], device="cuda", dtype=torch.float32, requires_grad=True)
        ray = camera.sample_ray(sample)
        ray.o.sum().backward()
        self.assertIsNotNone(sample.grad)

    def test_camera_sample_ray_shapes(self):
        camera = rt.Camera(width=16, height=12, fov_x=45.0)
        sample = torch.tensor([[0.0, 0.0], [1.0, 1.0]], device="cuda", dtype=torch.float32)
        ray = camera.sample_ray(sample)
        self.assertEqual(ray.o.shape, (2, 3))
        self.assertEqual(ray.d.shape, (2, 3))
        self.assertEqual(ray.tmax.shape, (2,))

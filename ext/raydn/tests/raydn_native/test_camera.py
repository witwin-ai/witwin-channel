import unittest
import math

import torch
import raydn as rt


@unittest.skipUnless(torch.cuda.is_available(), "CUDA torch is required")
class CameraTests(unittest.TestCase):
    def test_camera_sample_ray_backward(self):
        camera = rt.Camera(width=16, height=12, fov_x=45.0)
        sample = torch.tensor([[0.5, 0.5]], device="cuda", dtype=torch.float32, requires_grad=True)
        ray = camera.sample_ray(sample)
        ray.o.sum().backward()
        self.assertIsNone(sample.grad)

    def test_camera_sample_ray_shapes(self):
        camera = rt.Camera(width=16, height=12, fov_x=45.0)
        sample = torch.tensor([[0.0, 0.0], [1.0, 1.0]], device="cuda", dtype=torch.float32)
        ray = camera.sample_ray(sample)
        self.assertEqual(ray.o.shape, (2, 3))
        self.assertEqual(ray.d.shape, (2, 3))
        self.assertEqual(ray.tmax.shape, (0,))

    def test_camera_native_vjp_uses_upstream_gradients(self):
        camera = rt.Camera(width=16, height=12, fov_x=45.0)
        tan_x = math.tan(math.radians(camera.fov_x) * 0.5)
        tan_y = tan_x / camera.aspect

        sample = torch.tensor(
            [[0.25, 0.75], [0.8, 0.1]],
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        world = camera.sample_to_world(sample, depth=2.0)
        grad_world = torch.tensor(
            [[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]],
            device="cuda",
            dtype=torch.float32,
        )
        world.backward(grad_world)
        expected_sample_grad = torch.tensor(
            [
                [1.0 * 2.0 * tan_x * 2.0, 2.0 * -2.0 * tan_y * 2.0],
                [-4.0 * 2.0 * tan_x * 2.0, 5.0 * -2.0 * tan_y * 2.0],
            ],
            device="cuda",
            dtype=torch.float32,
        )
        torch.testing.assert_close(sample.grad, expected_sample_grad, atol=1e-6, rtol=1e-6)

        point = torch.tensor(
            [[0.1, -0.2, 2.0], [0.3, 0.4, 4.0]],
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        sample_out = camera.world_to_sample(point)
        grad_sample = torch.tensor(
            [[1.5, -2.0], [-3.0, 4.0]],
            device="cuda",
            dtype=torch.float32,
        )
        sample_out.backward(grad_sample)
        expected_point_grad = torch.empty_like(point)
        expected_point_grad[:, 0] = grad_sample[:, 0] * (0.5 / (point.detach()[:, 2] * tan_x))
        expected_point_grad[:, 1] = grad_sample[:, 1] * (-0.5 / (point.detach()[:, 2] * tan_y))
        expected_point_grad[:, 2] = (
            grad_sample[:, 0] * (-0.5 * point.detach()[:, 0] / (tan_x * point.detach()[:, 2] ** 2))
            + grad_sample[:, 1] * (0.5 * point.detach()[:, 1] / (tan_y * point.detach()[:, 2] ** 2))
        )
        torch.testing.assert_close(point.grad, expected_point_grad, atol=1e-6, rtol=1e-6)

        ray_sample = torch.tensor(
            [[0.2, 0.3], [0.9, 0.7]],
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        ray = camera.sample_ray(ray_sample)
        grad_direction = torch.tensor(
            [[0.7, -0.2, 0.5], [-0.3, 0.4, -0.1]],
            device="cuda",
            dtype=torch.float32,
        )
        ray.d.backward(grad_direction)
        target = torch.stack(
            (
                (ray_sample.detach()[:, 0] * 2.0 - 1.0) * tan_x,
                (1.0 - ray_sample.detach()[:, 1] * 2.0) * tan_y,
                torch.ones((ray_sample.shape[0],), device="cuda", dtype=torch.float32),
            ),
            dim=1,
        )
        direction = target / target.norm(dim=1, keepdim=True)
        inv_norm = 1.0 / target.norm(dim=1)
        dot = (grad_direction * direction).sum(dim=1)
        grad_target = (grad_direction - direction * dot[:, None]) * inv_norm[:, None]
        expected_ray_grad = torch.stack(
            (grad_target[:, 0] * 2.0 * tan_x, grad_target[:, 1] * -2.0 * tan_y),
            dim=1,
        )
        torch.testing.assert_close(ray_sample.grad, expected_ray_grad, atol=1e-6, rtol=1e-6)

    def test_camera_accepts_strided_inputs_and_upstream_gradients(self):
        camera = rt.Camera(width=16, height=12, fov_x=45.0)

        sample_base = torch.tensor(
            [[0.25, -9.0, 0.75, -8.0], [0.8, -7.0, 0.1, -6.0]],
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        sample = sample_base[:, ::2]
        self.assertFalse(sample.is_contiguous())
        world = camera.sample_to_world(sample, depth=2.0)
        expected_world = camera.sample_to_world(sample.detach().contiguous(), depth=2.0)
        torch.testing.assert_close(world, expected_world, atol=1e-6, rtol=1e-6)
        grad_world = torch.tensor(
            [[1.0, 99.0, 2.0, 98.0, 3.0], [-4.0, 97.0, 5.0, 96.0, -6.0]],
            device="cuda",
            dtype=torch.float32,
        )[:, ::2]
        self.assertFalse(grad_world.is_contiguous())
        world.backward(grad_world)
        self.assertIsNotNone(sample_base.grad)

        point_base = torch.tensor(
            [[0.1, 9.0, -0.2, 8.0, 2.0], [0.3, 7.0, 0.4, 6.0, 4.0]],
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        point = point_base[:, ::2]
        self.assertFalse(point.is_contiguous())
        sample_out = camera.world_to_sample(point)
        expected_sample = camera.world_to_sample(point.detach().contiguous())
        torch.testing.assert_close(sample_out, expected_sample, atol=1e-6, rtol=1e-6)
        grad_sample = torch.tensor(
            [[1.5, 42.0, -2.0], [-3.0, 41.0, 4.0]],
            device="cuda",
            dtype=torch.float32,
        )[:, ::2]
        self.assertFalse(grad_sample.is_contiguous())
        sample_out.backward(grad_sample)
        self.assertIsNotNone(point_base.grad)

        ray_base = torch.tensor(
            [[0.2, 5.0, 0.3], [0.9, 4.0, 0.7]],
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        ray_sample = ray_base[:, ::2]
        ray = camera.sample_ray(ray_sample)
        expected_ray = camera.sample_ray(ray_sample.detach().contiguous())
        torch.testing.assert_close(ray.d, expected_ray.d, atol=1e-6, rtol=1e-6)
        grad_direction = torch.tensor(
            [[0.7, 31.0, -0.2, 30.0, 0.5], [-0.3, 29.0, 0.4, 28.0, -0.1]],
            device="cuda",
            dtype=torch.float32,
        )[:, ::2]
        self.assertFalse(grad_direction.is_contiguous())
        ray.d.backward(grad_direction)
        self.assertIsNotNone(ray_base.grad)

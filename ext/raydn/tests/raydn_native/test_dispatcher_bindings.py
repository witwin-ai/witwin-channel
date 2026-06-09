import unittest

import torch
import raydn as rt


@unittest.skipUnless(torch.cuda.is_available(), "CUDA torch is required")
class DispatcherBindingTests(unittest.TestCase):
    def _scene(self):
        verts = torch.tensor(
            [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
        )
        faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
        scene = rt.Scene()
        scene.add_mesh(rt.Mesh(verts, faces))
        scene.build()
        return scene

    def _dfr_inputs(self):
        return {
            "active": torch.ones((1,), device="cuda", dtype=torch.bool),
            "state_edge_index": torch.tensor([0], device="cuda", dtype=torch.int32),
            "state_edge_pos": torch.tensor([[0.0, 0.0, 0.0]], device="cuda", dtype=torch.float32),
            "state_edge_dir": torch.tensor([[1.0, 0.0, 0.0]], device="cuda", dtype=torch.float32),
            "state_edge_t_min": torch.tensor([-1.0], device="cuda", dtype=torch.float32),
            "state_edge_t_max": torch.tensor([1.0], device="cuda", dtype=torch.float32),
            "state_n0": torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32),
            "state_n1": torch.tensor([[0.0, 0.0, -1.0]], device="cuda", dtype=torch.float32),
            "state_prim0": torch.tensor([0], device="cuda", dtype=torch.int32),
            "state_prim1": torch.tensor([0], device="cuda", dtype=torch.int32),
            "state_exterior_angle": torch.tensor([torch.pi], device="cuda", dtype=torch.float32),
            "state_src": torch.tensor([[0.0, -1.0, 0.25]], device="cuda", dtype=torch.float32),
            "state_src_power": torch.ones((1,), device="cuda", dtype=torch.float32),
            "zeros_vec": torch.zeros((1, 3), device="cuda", dtype=torch.float32),
            "material_eta_r": torch.ones((1,), device="cuda", dtype=torch.float32),
            "material_sigma": torch.zeros((1,), device="cuda", dtype=torch.float32),
            "material_mu_r": torch.ones((1,), device="cuda", dtype=torch.float32),
            "material_gain": torch.ones((1,), device="cuda", dtype=torch.float32),
            "material_valid": torch.ones((1,), device="cuda", dtype=torch.bool),
            "empty_i": torch.empty((0,), device="cuda", dtype=torch.int32),
            "empty_v": torch.empty((0, 3), device="cuda", dtype=torch.float32),
            "empty_b": torch.empty((0,), device="cuda", dtype=torch.bool),
        }

    def test_torchbind_scene_intersect_dispatcher(self):
        scene = self._scene()
        ray_o = torch.tensor([[0.0, -0.25, -1.0]], device="cuda", dtype=torch.float32)
        ray_d = torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32)
        ray_tmax = torch.empty((0,), device="cuda", dtype=torch.float32)

        out = torch.ops.raydn.intersect_forward_flags(scene._native_scene, ray_o, ray_d, ray_tmax, None, 7)

        torch.testing.assert_close(out[0], torch.tensor([1.0], device="cuda"))
        self.assertEqual(int(out[6][0].item()), 0)

    def test_reflection_accumulation_dispatcher_uses_typed_scene(self):
        scene = self._scene()
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

    def test_diffraction_accumulation_dispatcher_uses_typed_scene(self):
        scene = self._scene()
        x = self._dfr_inputs()

        out = torch.ops.raydn.diffraction_accumulation_forward(
            scene._native_scene,
            x["active"],
            x["state_edge_index"],
            x["state_edge_pos"],
            x["state_edge_dir"],
            x["state_edge_t_min"],
            x["state_edge_t_max"],
            x["state_n0"],
            x["state_n1"],
            x["state_prim0"],
            x["state_prim1"],
            x["state_exterior_angle"],
            x["state_src"],
            x["state_src_power"],
            x["zeros_vec"],
            x["zeros_vec"],
            x["material_eta_r"],
            x["material_sigma"],
            x["material_mu_r"],
            x["material_gain"],
            x["material_valid"],
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
            x["empty_b"],
            x["empty_i"],
            x["empty_v"],
            x["empty_v"],
            torch.empty((0,), device="cuda", dtype=torch.float32),
            torch.empty((0,), device="cuda", dtype=torch.float32),
            x["empty_v"],
            x["empty_v"],
            x["empty_i"],
            x["empty_i"],
            torch.empty((0,), device="cuda", dtype=torch.float32),
            0,
        )

        self.assertEqual(out[0].shape, (4, 4))
        self.assertEqual(out[14].dtype, torch.bool)


if __name__ == "__main__":
    unittest.main()

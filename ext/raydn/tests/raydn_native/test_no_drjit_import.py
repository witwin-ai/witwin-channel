import subprocess
import sys
import textwrap
import unittest


class TorchNativeImportTests(unittest.TestCase):
    def test_raydn_import_does_not_import_dr_jit(self):
        code = textwrap.dedent(
            """
            import sys
            import raydn as rt
            print(("dr" + "jit") in sys.modules)
            print(hasattr(rt, "Scene"))
            """
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        lines = proc.stdout.strip().splitlines()
        self.assertEqual(lines[0], "False")
        self.assertEqual(lines[1], "True")

    def test_core_torch_workflow_does_not_import_dr_jit(self):
        code = textwrap.dedent(
            """
            import sys
            import torch
            import raydn as rt
            v = torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]], device='cuda', dtype=torch.float32)
            f = torch.tensor([[0,1,2]], device='cuda', dtype=torch.int32)
            s = rt.Scene(); s.add_mesh(rt.Mesh(v, f)); s.build()
            r = rt.Ray(torch.tensor([[0.25,0.25,-1.]], device='cuda'), torch.tensor([[0.,0.,1.]], device='cuda'))
            print(float(s.intersect(r).t[0]))
            print(("dr" + "jit") in sys.modules)
            """
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        lines = proc.stdout.strip().splitlines()
        self.assertEqual(lines[-1], "False")

    def test_native_extension_loads(self):
        import raydn as rt
        self.assertTrue(hasattr(rt, "_C"))
        self.assertTrue(hasattr(rt._C, "build_info"))
        info = rt._C.build_info()
        self.assertEqual(info["backend"], "rayd-native")


if __name__ == "__main__":
    unittest.main()

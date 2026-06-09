import tomllib
import unittest
from pathlib import Path


class ProjectMetadataTests(unittest.TestCase):
    def test_project_name_is_rayd_native(self):
        data = tomllib.loads(Path("pyproject.toml").read_text())
        self.assertEqual(data["project"]["name"], "rayd-native")

    def test_default_dependencies_require_torch_not_dr_jit(self):
        data = tomllib.loads(Path("pyproject.toml").read_text())
        deps = [dep.lower() for dep in data["project"].get("dependencies", [])]
        self.assertTrue(any(dep.startswith("torch") for dep in deps))
        self.assertFalse(any(dep.startswith("dr" + "jit") for dep in deps))

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from witwin.channel.core.scene import Mesh, Scene
from witwin.channel.core.scene.sionna_adaptor import SionnaAdaptor, _source_roots
from witwin.core import Material, Structure


class FakeParams(dict):
    def update(self) -> None:
        return None


class FakeMiMesh:
    def __init__(self, name: str, vertex_count: int, face_count: int, **kwargs) -> None:
        del kwargs
        self.name = name
        self.vertex_count = vertex_count
        self.face_count = face_count
        self.params = FakeParams(
            {
                "vertex_positions": np.zeros(vertex_count * 3, dtype=np.float32),
                "faces": np.zeros(face_count * 3, dtype=np.int32),
            }
        )


class FakeRadioMaterial:
    def __init__(
        self,
        *,
        name: str,
        thickness: float,
        relative_permittivity: float,
        conductivity: float,
        scattering_coefficient: float,
        xpd_coefficient: float,
    ) -> None:
        self.name = name
        self.thickness = np.array([thickness], dtype=np.float32)
        self.relative_permittivity = np.array([relative_permittivity], dtype=np.float32)
        self.conductivity = np.array([conductivity], dtype=np.float32)
        self.scattering_coefficient = np.array([scattering_coefficient], dtype=np.float32)
        self.xpd_coefficient = np.array([xpd_coefficient], dtype=np.float32)


class FakeSceneObject:
    def __init__(self, *, mi_mesh, name: str, radio_material, remove_duplicate_vertices: bool) -> None:
        self.mi_mesh = mi_mesh
        self.name = name
        self.radio_material = radio_material
        self.remove_duplicate_vertices = remove_duplicate_vertices


class FakeScene:
    def __init__(self) -> None:
        self.objects = {}

    def edit(self, *, add) -> None:
        for obj in add:
            self.objects[obj.name] = obj


def _install_fake_sionna(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, package_root: Path | None = None):
    source_root = tmp_path / "local_sionna" / "src"
    (source_root / "sionna" / "rt").mkdir(parents=True)
    package_root = source_root if package_root is None else package_root

    rt_module = types.ModuleType("sionna.rt")
    rt_module.Scene = FakeScene
    rt_module.RadioMaterial = FakeRadioMaterial
    rt_module.SceneObject = FakeSceneObject
    rt_module.loaded_scene_paths = []

    def fake_load_scene(path):
        rt_module.loaded_scene_paths.append(str(path))
        material = rt_module.RadioMaterial(
            name="loaded-radio",
            thickness=0.1,
            relative_permittivity=4.0,
            conductivity=0.05,
            scattering_coefficient=0.0,
            xpd_coefficient=0.0,
        )
        scene = rt_module.Scene()
        scene.edit(
            add=[
                _fake_object(
                    rt_module,
                    "loaded-wall",
                    np.array(
                        [
                            [0.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0],
                        ],
                        dtype=np.float32,
                    ),
                    np.array([[0, 1, 2]], dtype=np.int32),
                    material,
                )
            ]
        )
        return scene

    rt_module.load_scene = fake_load_scene

    sionna_module = types.ModuleType("sionna")
    sionna_module.__file__ = str(package_root / "sionna" / "__init__.py")

    mitsuba_module = types.ModuleType("mitsuba")
    mitsuba_module.Mesh = FakeMiMesh
    mitsuba_module.traverse = lambda mesh: mesh.params

    original_import_module = SionnaAdaptor.load_rt.__globals__["importlib"].import_module

    def fake_import_module(name: str):
        if name == "sionna.rt":
            return rt_module
        if name == "sionna":
            return sionna_module
        if name == "mitsuba":
            return mitsuba_module
        return original_import_module(name)

    monkeypatch.setattr(SionnaAdaptor.load_rt.__globals__["importlib"], "import_module", fake_import_module)
    sys.path[:] = [entry for entry in sys.path if entry != str(source_root)]
    return source_root, rt_module


def _fake_object(rt_module, name: str, verts: np.ndarray, faces: np.ndarray, radio_material) -> object:
    mesh = FakeMiMesh(name, verts.shape[0], faces.shape[0])
    params = mesh.params
    params["vertex_positions"] = verts.reshape(-1).astype(np.float32)
    params["faces"] = faces.reshape(-1).astype(np.int32)
    params.update()
    return rt_module.SceneObject(
        mi_mesh=mesh,
        name=name,
        radio_material=radio_material,
        remove_duplicate_vertices=False,
    )


def test_sionna_adaptor_load_rt_prefers_explicit_local_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source_root, rt_module = _install_fake_sionna(monkeypatch, tmp_path)

    loaded = SionnaAdaptor.load_rt(source_root=source_root, prefer_local=True)

    assert loaded is rt_module
    assert str(source_root) in sys.path


def test_sionna_adaptor_default_roots_include_bundled_reference() -> None:
    roots = _source_roots(None)

    assert (
        Path(__file__).resolve().parents[2]
        / "reference"
        / "sionna-rt-reference-2.0.1"
        / "src"
    ) in roots


def test_sionna_adaptor_load_rt_rejects_wrong_resolved_package(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    wrong_root = tmp_path / "other_root"
    (wrong_root / "sionna").mkdir(parents=True)
    source_root, _ = _install_fake_sionna(monkeypatch, tmp_path, package_root=wrong_root)

    with pytest.raises(RuntimeError, match="resolved a different"):
        SionnaAdaptor.load_rt(source_root=source_root, prefer_local=True)


def test_sionna_adaptor_export_reuses_materials_and_skips_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_root, _ = _install_fake_sionna(monkeypatch, tmp_path)
    shared_material = Material(name="shared-material", eps_r=4.0, sigma_e=0.2)
    scene = Scene(
        structures=[
            Structure(
                geometry=Mesh(
                    torch.tensor(
                        [
                            [0.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                        ],
                        dtype=torch.float32,
                    ),
                    torch.tensor([[0, 1, 2]], dtype=torch.int32),
                ),
                material=shared_material,
                name="left",
            ),
            Structure(
                geometry=Mesh(
                    torch.tensor(
                        [
                            [2.0, 0.0, 0.0],
                            [3.0, 0.0, 0.0],
                            [2.0, 1.0, 0.0],
                        ],
                        dtype=torch.float32,
                    ),
                    torch.tensor([[0, 1, 2]], dtype=torch.int32),
                ),
                material=shared_material,
                name="right",
            ),
            Structure(
                geometry=Mesh(
                    torch.tensor(
                        [
                            [4.0, 0.0, 0.0],
                            [5.0, 0.0, 0.0],
                            [4.0, 1.0, 0.0],
                        ],
                        dtype=torch.float32,
                    ),
                    torch.tensor([[0, 1, 2]], dtype=torch.int32),
                ),
                material=Material(name="disabled-material", eps_r=5.0),
                name="disabled",
                enabled=False,
            ),
        ],
        device="cpu",
    )

    exported = SionnaAdaptor.export(scene, source_root=source_root, prefer_local=True)

    assert sorted(exported.objects) == ["left", "right"]
    assert exported.objects["left"].radio_material is exported.objects["right"].radio_material
    assert exported.objects["left"].mi_mesh.params["vertex_positions"].shape == (9,)


def test_sionna_adaptor_export_validates_mu_r(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source_root, _ = _install_fake_sionna(monkeypatch, tmp_path)
    scene = Scene(
        structures=[
            Structure(
                geometry=Mesh(
                    torch.tensor(
                        [
                            [0.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                        ],
                        dtype=torch.float32,
                    ),
                    torch.tensor([[0, 1, 2]], dtype=torch.int32),
                ),
                material=Material(name="bad-material", eps_r=3.0, mu_r=1.5, sigma_e=0.1),
                name="bad",
            )
        ],
        device="cpu",
    )

    with pytest.raises(ValueError, match="mu_r"):
        SionnaAdaptor.export(scene, source_root=source_root, prefer_local=True)


def test_scene_to_sionna_validates_mu_r(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source_root, _ = _install_fake_sionna(monkeypatch, tmp_path)
    scene = Scene(
        structures=[
            Structure(
                geometry=Mesh(
                    torch.tensor(
                        [
                            [0.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                        ],
                        dtype=torch.float32,
                    ),
                    torch.tensor([[0, 1, 2]], dtype=torch.int32),
                ),
                material=Material(name="bad-material", eps_r=3.0, mu_r=2.0, sigma_e=0.1),
                name="bad",
            )
        ],
        device="cpu",
    )

    with pytest.raises(ValueError, match="mu_r"):
        scene.to_sionna(source_root=source_root, prefer_local=True)


def test_sionna_adaptor_import_scene_reuses_materials_and_scene_classmethod(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_root, rt_module = _install_fake_sionna(monkeypatch, tmp_path)
    shared_material = rt_module.RadioMaterial(
        name="shared-radio",
        thickness=0.1,
        relative_permittivity=3.5,
        conductivity=0.2,
        scattering_coefficient=0.0,
        xpd_coefficient=0.0,
    )
    sionna_scene = rt_module.Scene()
    sionna_scene.edit(
        add=[
            _fake_object(
                rt_module,
                "left",
                np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                    ],
                    dtype=np.float32,
                ),
                np.array([[0, 1, 2]], dtype=np.int32),
                shared_material,
            ),
            _fake_object(
                rt_module,
                "right",
                np.array(
                    [
                        [2.0, 0.0, 0.0],
                        [3.0, 0.0, 0.0],
                        [2.0, 1.0, 0.0],
                    ],
                    dtype=np.float32,
                ),
                np.array([[0, 1, 2]], dtype=np.int32),
                shared_material,
            ),
        ]
    )

    imported = Scene.from_sionna(sionna_scene, device="cpu")

    assert len(imported.structures) == 2
    assert imported.structures[0].material is imported.structures[1].material
    assert imported.structures[0].material.eps_r == pytest.approx(3.5)
    assert imported.structures[0].material.sigma_e == pytest.approx(0.2)
    assert imported.structures[0].metadata["sionna"]["radio_material_name"] == "shared-radio"
    assert imported.structures[0].geometry.to_mesh()[0].shape == (3, 3)


def test_scene_load_mitsuba_loads_xml_path_through_sionna_rt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_root, rt_module = _install_fake_sionna(monkeypatch, tmp_path)
    scene_path = tmp_path / "munich.xml"
    scene_path.write_text("<scene version=\"2.1.0\" />", encoding="utf-8")

    imported = Scene.load_mitsuba(
        scene_path,
        source_root=source_root,
        prefer_local=True,
        device="cpu",
        metadata={"case": "munich"},
    )

    assert rt_module.loaded_scene_paths == [str(scene_path)]
    assert len(imported.structures) == 1
    assert imported.structures[0].name == "loaded-wall"
    assert imported.structures[0].material.eps_r == pytest.approx(4.0)
    assert imported.metadata["case"] == "munich"

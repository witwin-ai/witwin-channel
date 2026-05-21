from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _python_and_doc_files_under(*roots: str) -> list[Path]:
    files: list[Path] = []
    for root_name in roots:
        root = ROOT / root_name
        for pattern in ("*.py", "*.md", "*.ipynb"):
            files.extend(
                path
                for path in root.rglob(pattern)
                if "__pycache__" not in path.parts
            )
    files.extend(ROOT.glob("*.md"))
    return files


def test_phase4_channel_core_replaces_top_level_channel_helpers() -> None:
    core_root = ROOT / "witwin" / "channel" / "core"
    scene_root = core_root / "scene"

    assert (core_root / "__init__.py").is_file()
    assert (core_root / "numerics" / "arrays.py").is_file()
    assert (core_root / "numerics" / "constants.py").is_file()
    assert (core_root / "numerics" / "tensors.py").is_file()
    assert (core_root / "geometry" / "primitives.py").is_file()
    assert (core_root / "geometry" / "diffraction.py").is_file()
    assert (core_root / "geometry" / "mesh_buffers.py").is_file()
    assert (core_root / "geometry" / "raygen.py").is_file()
    assert (core_root / "physics" / "materials.py").is_file()
    assert (core_root / "physics" / "polarization.py").is_file()
    assert (core_root / "physics" / "wave_math.py").is_file()
    assert (core_root / "runtime" / "context.py").is_file()
    assert (core_root / "grid.py").is_file()
    assert (core_root / "results" / "__init__.py").is_file()
    assert (core_root / "results" / "radiomap_result.py").is_file()
    assert (core_root / "results" / "ray_mode.py").is_file()
    assert (core_root / "kernels" / "shadow_boundary" / "native_impl.py").is_file()
    assert (scene_root / "__init__.py").is_file()
    assert (scene_root / "scene.py").is_file()
    assert (scene_root / "mesh.py").is_file()
    assert (scene_root / "sionna_adaptor.py").is_file()

    assert not (ROOT / "witwin" / "channel_utils").exists()
    assert not (ROOT / "witwin" / "channel_scene").exists()


def test_phase4_public_channel_exports_scene_and_core_types() -> None:
    import witwin.channel as wc
    from witwin.channel.core import Grid, RadioMapResult
    from witwin.channel.core.scene import Scene

    assert wc.Scene is Scene
    assert wc.Grid is Grid
    assert wc.RadioMapResult is RadioMapResult


def test_phase4_imports_do_not_reference_removed_public_packages() -> None:
    forbidden = (
        b"witwin.channel_utils",
        b"witwin.channel_scene",
        b"from witwin import channel_utils",
        b"from witwin import channel_scene",
    )

    offenders: list[str] = []
    for path in _python_and_doc_files_under("witwin", "tests", "examples", "docs"):
        if path.name == Path(__file__).name:
            continue
        text = path.read_bytes()
        for needle in forbidden:
            if needle in text:
                offenders.append(
                    f"{path.relative_to(ROOT)} contains {needle.decode('ascii')}"
                )

    assert offenders == []

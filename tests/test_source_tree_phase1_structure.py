from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _python_files_under(*parts: str) -> list[Path]:
    root = ROOT.joinpath(*parts)
    return [
        path
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
    ]


def test_channel_umbrella_package_remains_public() -> None:
    channel_root = ROOT / "witwin" / "channel"

    assert (channel_root / "__init__.py").is_file()
    assert (channel_root / "deterministic" / "__init__.py").is_file()
    assert (channel_root / "montecarlo" / "__init__.py").is_file()
    assert (channel_root / "path" / "__init__.py").is_file()


def test_native_specs_are_consolidated_under_top_level_native_package() -> None:
    native_root = ROOT / "witwin" / "channel" / "_native"

    assert (native_root / "__init__.py").is_file()
    assert (native_root / "loader.py").is_file()
    assert (native_root / "channel_utils.py").is_file()
    assert (native_root / "deterministic.py").is_file()
    assert (native_root / "montecarlo.py").is_file()

    assert not (ROOT / "witwin" / "channel_utils" / "_native").exists()
    assert not (ROOT / "witwin" / "deterministic" / "_native").exists()
    assert not (ROOT / "witwin" / "montecarlo" / "_native").exists()


def test_deterministic_impl_modules_are_flattened() -> None:
    deterministic_root = ROOT / "witwin" / "channel" / "deterministic"

    for filename in [
        "__init__.py",
        "accumulation.py",
        "builders.py",
        "forward.py",
        "math.py",
        "postprocessing.py",
        "state.py",
    ]:
        assert (deterministic_root / "diffraction" / filename).is_file()

    for filename in [
        "__init__.py",
        "accumulation.py",
        "common.py",
        "detail.py",
        "epc.py",
        "paths.py",
    ]:
        assert (deterministic_root / "reflection" / filename).is_file()

    assert not (deterministic_root / "path" / "diffraction_impl").exists()
    assert not (deterministic_root / "path" / "reflection_impl").exists()


def test_phase1_imports_do_not_reference_removed_private_paths() -> None:
    forbidden = (
        b"witwin.channel.core._native",
        b"witwin.channel.deterministic._native",
        b"witwin.channel.montecarlo._native",
        b"witwin.channel.deterministic.trace.diffraction_impl",
        b"witwin.channel.deterministic.trace.reflection_impl",
        b".diffraction_impl",
        b".reflection_impl",
        b"..diffraction_impl",
        b"..reflection_impl",
    )

    offenders: list[str] = []
    for path in _python_files_under("witwin"):
        text = path.read_bytes()
        for needle in forbidden:
            if needle in text:
                offenders.append(
                    f"{path.relative_to(ROOT)} contains {needle.decode('ascii')}"
                )

    assert offenders == []


def test_phase2_internal_path_packages_are_renamed() -> None:
    deterministic_root = ROOT / "witwin" / "channel" / "deterministic"
    montecarlo_root = ROOT / "witwin" / "channel" / "montecarlo"

    for filename in [
        "__init__.py",
        "diffraction.py",
        "los.py",
        "path_export.py",
        "path_export_assembly.py",
        "reflection.py",
    ]:
        assert (deterministic_root / "trace" / filename).is_file()

    for filename in [
        "__init__.py",
        "ad_support.py",
        "diffraction.py",
        "diffraction_utd.py",
        "los.py",
        "postprocessing.py",
        "reflection.py",
    ]:
        assert (montecarlo_root / "trace" / filename).is_file()

    assert not (deterministic_root / "path").exists()
    assert not (montecarlo_root / "path").exists()
    assert not (deterministic_root / "traversal").exists()
    assert not (montecarlo_root / "transport").exists()


def test_phase2_ad_variants_are_flattened() -> None:
    montecarlo_root = ROOT / "witwin" / "channel" / "montecarlo"

    assert (montecarlo_root / "integrators" / "basic_ad.py").is_file()
    assert (montecarlo_root / "integrators" / "bdpt_ad.py").is_file()
    assert (montecarlo_root / "trace" / "diffraction_ad.py").is_file()

    assert not (montecarlo_root / "integrators" / "ad").exists()
    assert not (montecarlo_root / "trace" / "ad").exists()


def test_phase2_imports_do_not_reference_removed_path_packages() -> None:
    forbidden = (
        b"witwin.channel.deterministic.path",
        b"witwin.channel.montecarlo.path",
        b"witwin.channel.deterministic.traversal",
        b"witwin.channel.montecarlo.transport",
        b"from .path import",
        b"from .path.",
        b"from ..path import",
        b"from ..path.",
        b"witwin.channel.montecarlo.integrators.ad",
        b"witwin.channel.montecarlo.trace.ad",
        b"from .ad import",
        b"from .ad.",
        b"from ..ad import",
        b"from ..ad.",
    )

    offenders: list[str] = []
    for path in _python_files_under("witwin"):
        text = path.read_bytes()
        for needle in forbidden:
            if needle in text:
                offenders.append(
                    f"{path.relative_to(ROOT)} contains {needle.decode('ascii')}"
                )

    assert offenders == []

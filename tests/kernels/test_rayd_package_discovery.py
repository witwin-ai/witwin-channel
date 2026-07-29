# Copyright Xingyu Chen.
# Tests rayd package discovery.

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = ROOT / "dependencies" / "rayd.lock.json"
RESOLVER_PATH = ROOT / "cmake" / "resolve_rayd_source.py"


def _load_resolver():
    spec = importlib.util.spec_from_file_location("resolve_rayd_source", RESOLVER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _rayd_source() -> Path:
    configured = os.environ.get("RAYD_SOURCE_DIR")
    platform_root = ROOT.parent.parent if ROOT.parent.name == ".worktrees" else ROOT.parent
    source = Path(configured) if configured else platform_root / "RayD"
    assert (source / ".git").exists()
    return source


class _FakeDistribution:
    def __init__(self, root: Path, *, version: str) -> None:
        self.root = root
        self.version = version
        self._path = root / f"rayd_torch-{version}.dist-info"
        self.files = [
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
        ]

    def locate_file(self, path: str):
        return self.root / path


def _package(tmp_path: Path) -> tuple[_FakeDistribution, Path]:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    version = str(lock["source_bundle"]["distribution_version"])
    workspace = tmp_path / "rayd"
    cloned = subprocess.run(
        (
            "git",
            "clone",
            "--shared",
            "--no-checkout",
            os.fspath(_rayd_source()),
            os.fspath(workspace),
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    assert cloned.returncode == 0, cloned.stderr
    checked_out = subprocess.run(
        ("git", "checkout", "--detach", str(lock["commit"])),
        cwd=workspace,
        check=False,
        capture_output=True,
        text=True,
    )
    assert checked_out.returncode == 0, checked_out.stderr
    bundle = tmp_path / "generated"
    completed = subprocess.run(
        (
            sys.executable,
            os.fspath(
                workspace
                / "backends"
                / "torch"
                / "scripts"
                / "generate_source_bundle.py"
            ),
            "--workspace",
            os.fspath(workspace),
            "--output",
            os.fspath(bundle),
            "--distribution-version",
            version,
            "--commit",
            lock["commit"],
            "--repository-url",
            lock["repository_url"],
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    package_root = tmp_path / "site packages"
    resource = package_root / "rayd" / "torch" / "_source"
    shutil.copytree(bundle, resource)
    dist_info = package_root / f"rayd_torch-{version}.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: rayd-torch\nVersion: {version}\n",
        encoding="utf-8",
    )
    distribution = _FakeDistribution(package_root, version=version)
    return distribution, resource


def test_resolver_accepts_unique_lock_valid_package(tmp_path: Path):
    resolver = _load_resolver()
    distribution, resource = _package(tmp_path)
    with patch.object(
        resolver.importlib.metadata,
        "distributions",
        return_value=[distribution],
    ):
        result = resolver.resolve(LOCK_PATH)

    assert result["source_kind"] == "python-package"
    assert Path(result["source_dir"]) == (resource / "source").resolve()
    assert result["source_manifest_sha256"] == json.loads(
        LOCK_PATH.read_text(encoding="utf-8")
    )["source_bundle"]["manifest_sha256"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("commit", "commit does not match"),
        ("repository", "repository URL does not match"),
        ("dirty", "dirty=false"),
        ("source", "source file changed"),
        ("extra", "unmanifested RayD source file"),
        ("record", "RECORD does not own"),
        ("escape", "safe POSIX relative path"),
    ],
)
def test_resolver_fails_loudly_on_package_mutation(
    tmp_path: Path, mutation: str, message: str
):
    resolver = _load_resolver()
    distribution, resource = _package(tmp_path)
    metadata_path = resource / "rayd-source.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if mutation == "commit":
        metadata["commit"] = "2" * 40
    elif mutation == "repository":
        metadata["repository_url"] = "https://example.invalid/RayD.git"
    elif mutation == "dirty":
        metadata["dirty"] = True
    elif mutation == "source":
        target = resource / "source" / "backends" / "torch" / "CMakeLists.txt"
        target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    elif mutation == "extra":
        (resource / "source" / "extra.cu").write_text("", encoding="utf-8")
    elif mutation == "record":
        distribution.files.remove("rayd/torch/_source/rayd-source.json")
    else:
        metadata["source_root"] = "../escape"
    if mutation in {"commit", "repository", "dirty", "escape"}:
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with patch.object(
        resolver.importlib.metadata,
        "distributions",
        return_value=[distribution],
    ):
        with pytest.raises(resolver.RayDDiscoveryError, match=message):
            resolver.resolve(LOCK_PATH)


def test_resolver_rejects_missing_or_ambiguous_distribution(tmp_path: Path):
    resolver = _load_resolver()
    distribution, _ = _package(tmp_path)
    for candidates in ([], [distribution, distribution]):
        with patch.object(
            resolver.importlib.metadata,
            "distributions",
            return_value=candidates,
        ):
            with pytest.raises(
                resolver.RayDDiscoveryError,
                match="expected exactly one active",
            ):
                resolver.resolve(LOCK_PATH)


def test_cmake_explicit_source_is_strictly_higher_priority():
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    explicit = cmake.index('if(NOT RAYD_SOURCE_DIR STREQUAL "")')
    discovery = cmake.index("cmake/resolve_rayd_source.py")
    assert explicit < discovery
    assert "package discovery is not a fallback for an invalid explicit path" in cmake
    assert "$ENV{CONDA_PREFIX}" not in cmake
    assert "CMAKE_PREFIX_PATH" not in cmake[explicit:discovery]
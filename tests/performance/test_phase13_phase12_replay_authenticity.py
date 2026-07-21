from __future__ import annotations

import copy
from pathlib import Path

import pytest

from benchmarks.phase13_phase12.contracts import EvidenceError, VariantConfig
from benchmarks.phase13_phase12.workers import (
    RUNNER_REPO_PATHS,
    _bind_runner_hash_maps,
    _clean_git_environment,
    validate_retained_worker_capture,
    worker_argv,
)


def _runner_hashes() -> dict[str, str]:
    return {
        relative.as_posix(): f"{index:064x}"
        for index, relative in enumerate(RUNNER_REPO_PATHS, start=1)
    }


def test_runner_hash_binding_requires_the_complete_exact_frozen_map() -> None:
    frozen = _runner_hashes()
    assert _bind_runner_hash_maps(copy.deepcopy(frozen), frozen) == frozen

    subset = dict(frozen)
    subset.pop(next(iter(subset)))
    with pytest.raises(EvidenceError, match="complete canonical runner source hash map"):
        _bind_runner_hash_maps(subset, frozen)

    changed = dict(frozen)
    changed[next(iter(changed))] = "f" * 64
    with pytest.raises(EvidenceError, match="differ from the complete frozen hash map"):
        _bind_runner_hash_maps(changed, frozen)


def test_git_environment_drops_repository_redirecting_ambient_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    redirected_names = (
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_NAMESPACE",
        "GIT_CONFIG_SYSTEM",
        "GIT_CONFIG_GLOBAL",
    )
    for name in redirected_names:
        monkeypatch.setenv(name, str(tmp_path / name.lower()))

    git = tmp_path / "git-bin" / "git.exe"
    environment = _clean_git_environment(git)

    assert not set(redirected_names).intersection(environment)
    assert environment["PATH"] == str(git.parent.resolve())
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert "HOME" not in environment


def _canonical_captures(tmp_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    checkout = (tmp_path / "checkout").resolve()
    checkout.mkdir()
    python = (tmp_path / "runtime" / "python.exe").resolve()
    site_packages = (tmp_path / "install" / "site-packages").resolve()
    variant = VariantConfig(
        checkout=checkout,
        python_executable=python,
        runner_site_packages=site_packages,
        runner_extension=(site_packages / "witwin" / "channel_native" / "_channel_native.pyd"),
    )
    argv = worker_argv(
        variant,
        group="diffraction",
        name="candidate",
        process_index=3,
        order="BA",
        munich_scene_xml=(tmp_path / "munich.xml").resolve(),
        sionna_source_root=(tmp_path / "sionna").resolve(),
    )
    capture: dict[str, object] = {
        "argv": argv,
        "cwd": str(checkout),
        "returncode": 0,
        "timed_out": False,
        "started_time_ns": 10,
        "completed_time_ns": 20,
        "stdout_artifact": {"path": "stdout", "sha256": "0" * 64, "bytes": 1},
        "stderr_artifact": {"path": "stderr", "sha256": "1" * 64, "bytes": 0},
    }
    identity_capture = {
        "argv": [
            str(python),
            "-I",
            str(checkout / "benchmarks" / "phase13_phase12_bootstrap.py"),
            "--site-packages",
            str(site_packages),
            "--script",
            str(checkout / "benchmarks" / "phase13_phase12_identity_probe.py"),
        ],
        "cwd": str(checkout),
    }
    return capture, identity_capture


def _validate_capture(capture: object, identity_capture: object) -> dict[str, object]:
    return validate_retained_worker_capture(
        capture,
        identity_capture=identity_capture,
        expected_group="diffraction",
        expected_variant="candidate",
        process_index=3,
        order="BA",
    )


def test_retained_timed_worker_capture_binds_canonical_argv_cwd_and_success(
    tmp_path: Path,
) -> None:
    capture, identity_capture = _canonical_captures(tmp_path)
    assert _validate_capture(capture, identity_capture) == capture

    for field, value, message in (
        ("returncode", 1, "did not return success"),
        ("timed_out", True, "must not be timed out"),
        ("cwd", str((tmp_path / "other").resolve()), "invocation differ"),
    ):
        tampered = copy.deepcopy(capture)
        tampered[field] = value
        with pytest.raises(EvidenceError, match=message):
            _validate_capture(tampered, identity_capture)

    tampered_argv = copy.deepcopy(capture)
    tampered_argv["argv"][16] = "2"  # type: ignore[index]
    with pytest.raises(EvidenceError, match="argv is not canonical"):
        _validate_capture(tampered_argv, identity_capture)

    mismatched_identity = copy.deepcopy(identity_capture)
    mismatched_identity["argv"][4] = str((tmp_path / "other-site").resolve())  # type: ignore[index]
    with pytest.raises(EvidenceError, match="argv is not canonical"):
        _validate_capture(capture, mismatched_identity)

# Copyright Xingyu Chen.
# Resolve and verify the lock-pinned RayD source bundle in a Python package.

"""Resolve and verify the lock-pinned RayD source bundle in a Python package."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any


SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class RayDDiscoveryError(RuntimeError):
    """Raised when the active Python environment has no trusted RayD source."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_text_sha256(path: Path) -> str:
    content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(content).hexdigest()


def _mapping(value: object, *, label: str, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise RayDDiscoveryError(f"{label} has unexpected schema: {actual}")
    return value


def _string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RayDDiscoveryError(f"{label} must be a non-empty string")
    return value


def _safe_relative(value: object, *, label: str) -> PurePosixPath:
    text = _string(value, label=label)
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or "\\" in text
        or ";" in text
        or any(ord(character) < 32 for character in text)
    ):
        raise RayDDiscoveryError(f"{label} is not a safe POSIX relative path: {text!r}")
    return path


def _integration_abi(value: object, *, label: str) -> dict[str, Any]:
    integration = _mapping(
        value,
        label=label,
        keys={"kind", "entrypoint", "headers", "sha256", "api_version", "identity"},
    )
    if integration["kind"] != "source-header-set-sha256":
        raise RayDDiscoveryError(f"{label} kind must be source-header-set-sha256")
    entrypoint = _safe_relative(integration["entrypoint"], label=f"{label} entrypoint")
    headers = integration["headers"]
    if not isinstance(headers, list) or not headers:
        raise RayDDiscoveryError(f"{label} headers must be a non-empty list")
    described: list[tuple[str, str]] = []
    for index, raw_header in enumerate(headers):
        header = _mapping(
            raw_header,
            label=f"{label} header {index}",
            keys={"path", "sha256"},
        )
        path = _safe_relative(header["path"], label=f"{label} header path {index}")
        digest = _string(header["sha256"], label=f"{label} header SHA {index}")
        if SHA256_PATTERN.fullmatch(digest) is None:
            raise RayDDiscoveryError(f"{label} header SHA {index} must be a lowercase SHA-256")
        described.append((path.as_posix(), digest))
    if described != sorted(described) or len({path for path, _ in described}) != len(described):
        raise RayDDiscoveryError(f"{label} headers must have unique paths in sorted order")
    if entrypoint.as_posix() not in {path for path, _ in described}:
        raise RayDDiscoveryError(f"{label} entrypoint is not present in headers")
    aggregate = hashlib.sha256()
    for path, digest in described:
        aggregate.update(path.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
    expected_aggregate = _string(integration["sha256"], label=f"{label} SHA")
    if aggregate.hexdigest() != expected_aggregate:
        raise RayDDiscoveryError(f"{label} aggregate SHA does not match its headers")
    if not isinstance(integration["api_version"], int) or integration["api_version"] < 1:
        raise RayDDiscoveryError(f"{label} api_version must be a positive integer")
    _string(integration["identity"], label=f"{label} identity")
    return integration


def _inside(path: Path, root: Path, *, label: str) -> Path:
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RayDDiscoveryError(f"{label} escapes its distribution root") from exc
    return resolved


def _read_json(path: Path, *, label: str) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RayDDiscoveryError(f"could not read {label} at '{path}': {exc}") from exc


def _validate_lock(path: Path) -> dict[str, Any]:
    lock = _mapping(
        _read_json(path, label="RayD lock"),
        label="RayD lock",
        keys={
            "schema_version",
            "repository_url",
            "commit",
            "integration_abi",
            "source_bundle",
        },
    )
    if lock["schema_version"] != 2:
        raise RayDDiscoveryError("RayD lock schema_version must be 2")
    commit = _string(lock["commit"], label="RayD lock commit")
    if SHA_PATTERN.fullmatch(commit) is None:
        raise RayDDiscoveryError("RayD lock commit must be a lowercase 40-digit SHA")
    integration = _integration_abi(lock["integration_abi"], label="RayD lock integration_abi")
    bundle = _mapping(
        lock["source_bundle"],
        label="RayD lock source_bundle",
        keys={"distribution", "distribution_version", "metadata_path", "manifest_sha256"},
    )
    for label, value in (
        ("integration ABI SHA", integration["sha256"]),
        ("source manifest SHA", bundle["manifest_sha256"]),
    ):
        if SHA256_PATTERN.fullmatch(_string(value, label=label)) is None:
            raise RayDDiscoveryError(f"{label} must be a lowercase SHA-256")
    _safe_relative(integration["entrypoint"], label="RayD lock integration ABI entrypoint")
    _safe_relative(bundle["metadata_path"], label="RayD lock metadata path")
    return lock


def _validate_source_tree(source_root: Path, integration: dict[str, Any]) -> None:
    source_root = source_root.resolve(strict=True)
    for index, header in enumerate(integration["headers"]):
        relative = _safe_relative(header["path"], label=f"RayD integration header path {index}")
        path = _inside(
            source_root / Path(*relative.parts),
            source_root,
            label=f"RayD integration header {index}",
        )
        if _normalized_text_sha256(path) != header["sha256"]:
            raise RayDDiscoveryError(f"RayD integration header changed: {relative.as_posix()!r}")
    if not (source_root / "torch" / "CMakeLists.txt").is_file():
        raise RayDDiscoveryError("RayD source lacks torch/CMakeLists.txt")
    if not (source_root / "include" / "rayd" / "integration.h").is_file():
        raise RayDDiscoveryError("RayD source lacks include/rayd/integration.h")
    if not (source_root / "src" / "field_transport_ad.cuh").is_file():
        raise RayDDiscoveryError("RayD source lacks src/field_transport_ad.cuh")
    if not (source_root / "src" / "transmission_device.cuh").is_file():
        raise RayDDiscoveryError("RayD source lacks src/transmission_device.cuh")


def _active_distribution(name: str) -> importlib.metadata.Distribution:
    distributions = list(importlib.metadata.distributions(name=name))
    if len(distributions) != 1:
        locations = sorted(
            str(getattr(distribution, "_path", "unknown")) for distribution in distributions
        )
        raise RayDDiscoveryError(
            f"expected exactly one active {name!r} distribution, found "
            f"{len(distributions)}: {locations}"
        )
    return distributions[0]


def _record_paths(distribution: importlib.metadata.Distribution) -> set[str]:
    files = distribution.files
    if files is None:
        raise RayDDiscoveryError("rayd-torch distribution has no RECORD file list")
    return {PurePosixPath(str(path).replace("\\", "/")).as_posix() for path in files}


def resolve(lock_path: Path) -> dict[str, object]:
    lock_path = lock_path.resolve(strict=True)
    lock = _validate_lock(lock_path)
    bundle_lock = lock["source_bundle"]
    distribution_name = _string(bundle_lock["distribution"], label="RayD source distribution")
    distribution = _active_distribution(distribution_name)
    expected_version = _string(
        bundle_lock["distribution_version"], label="RayD source distribution version"
    )
    if distribution.version != expected_version:
        raise RayDDiscoveryError(
            f"rayd-torch version mismatch: active package is {distribution.version!r}, "
            f"lock requires {expected_version!r}"
        )

    distribution_root = Path(distribution.locate_file("")).resolve(strict=True)
    metadata_relative = _safe_relative(
        bundle_lock["metadata_path"], label="RayD package metadata path"
    )
    metadata_record_path = metadata_relative.as_posix()
    record_paths = _record_paths(distribution)
    if metadata_record_path not in record_paths:
        raise RayDDiscoveryError(f"rayd-torch RECORD does not own {metadata_record_path!r}")
    metadata_path = _inside(
        distribution.locate_file(metadata_record_path),
        distribution_root,
        label="RayD package metadata",
    )
    metadata = _mapping(
        _read_json(metadata_path, label="RayD package metadata"),
        label="RayD package metadata",
        keys={
            "schema_version",
            "distribution",
            "repository_url",
            "commit",
            "dirty",
            "source_root",
            "source_manifest",
            "integration_abi",
        },
    )
    if metadata["schema_version"] != 2:
        raise RayDDiscoveryError("RayD package metadata schema_version must be 2")
    package_distribution = _mapping(
        metadata["distribution"],
        label="RayD package distribution",
        keys={"name", "version"},
    )
    if package_distribution != {
        "name": distribution_name,
        "version": expected_version,
    }:
        raise RayDDiscoveryError("RayD package distribution identity does not match the lock")
    if metadata["repository_url"] != lock["repository_url"]:
        raise RayDDiscoveryError("RayD package repository URL does not match the lock")
    if metadata["commit"] != lock["commit"]:
        raise RayDDiscoveryError("RayD package commit does not match the lock")
    if metadata["dirty"] is not False:
        raise RayDDiscoveryError("RayD package source metadata must report dirty=false")

    integration = _integration_abi(
        metadata["integration_abi"], label="RayD package integration_abi"
    )
    if integration != lock["integration_abi"]:
        raise RayDDiscoveryError("RayD package integration ABI does not match the lock")
    manifest_metadata = _mapping(
        metadata["source_manifest"],
        label="RayD package source_manifest",
        keys={"path", "sha256"},
    )
    if manifest_metadata["sha256"] != bundle_lock["manifest_sha256"]:
        raise RayDDiscoveryError("RayD package source manifest SHA does not match the lock")
    source_relative = _safe_relative(metadata["source_root"], label="RayD package source_root")
    manifest_relative = _safe_relative(
        manifest_metadata["path"], label="RayD package manifest path"
    )
    metadata_root = metadata_path.parent.resolve(strict=True)
    source_root = _inside(
        metadata_root / Path(*source_relative.parts),
        distribution_root,
        label="RayD package source_root",
    )
    manifest_path = _inside(
        metadata_root / Path(*manifest_relative.parts),
        distribution_root,
        label="RayD package manifest",
    )
    if _sha256(manifest_path) != bundle_lock["manifest_sha256"]:
        raise RayDDiscoveryError("RayD package source manifest bytes changed")
    for resource in (manifest_path, source_root):
        resource_record = resource.relative_to(distribution_root).as_posix()
        if resource.is_file() and resource_record not in record_paths:
            raise RayDDiscoveryError(f"rayd-torch RECORD does not own {resource_record!r}")

    manifest = _mapping(
        _read_json(manifest_path, label="RayD package source manifest"),
        label="RayD package source manifest",
        keys={"schema_version", "files"},
    )
    if manifest["schema_version"] != 1 or not isinstance(manifest["files"], list):
        raise RayDDiscoveryError("RayD package source manifest has invalid contents")
    described: dict[str, str] = {}
    for index, raw_entry in enumerate(manifest["files"]):
        entry = _mapping(
            raw_entry,
            label=f"RayD package source manifest entry {index}",
            keys={"path", "sha256"},
        )
        relative = _safe_relative(entry["path"], label=f"RayD package source manifest path {index}")
        relative_text = relative.as_posix()
        digest = _string(entry["sha256"], label=f"RayD source SHA {relative_text}")
        if SHA256_PATTERN.fullmatch(digest) is None:
            raise RayDDiscoveryError(f"RayD source SHA is invalid for {relative_text!r}")
        if relative_text in described:
            raise RayDDiscoveryError(f"duplicate RayD source manifest path {relative_text!r}")
        described[relative_text] = digest

    actual: set[str] = set()
    for candidate in source_root.rglob("*"):
        if candidate.is_symlink():
            raise RayDDiscoveryError(f"RayD package source contains a symlink: {candidate}")
        if not candidate.is_file():
            continue
        resolved = _inside(candidate, source_root, label="RayD package source file")
        relative_text = resolved.relative_to(source_root).as_posix()
        actual.add(relative_text)
        expected_digest = described.get(relative_text)
        if expected_digest is None:
            raise RayDDiscoveryError(f"unmanifested RayD source file {relative_text!r}")
        if _sha256(resolved) != expected_digest:
            raise RayDDiscoveryError(f"RayD source file changed: {relative_text!r}")
        record_path = resolved.relative_to(distribution_root).as_posix()
        if record_path not in record_paths:
            raise RayDDiscoveryError(f"rayd-torch RECORD does not own {record_path!r}")
    if actual != set(described):
        missing = sorted(set(described) - actual)
        raise RayDDiscoveryError(f"RayD package source files are missing: {missing}")

    _validate_source_tree(source_root, integration)

    return {
        "source_kind": "python-package",
        "source_dir": os.fspath(source_root),
        "metadata_path": os.fspath(metadata_path),
        "commit": lock["commit"],
        "repository_url": lock["repository_url"],
        "dirty": False,
        "source_manifest_sha256": bundle_lock["manifest_sha256"],
    }


def validate_source(lock_path: Path, source_root: Path) -> None:
    lock = _validate_lock(lock_path.resolve(strict=True))
    integration = _integration_abi(lock["integration_abi"], label="RayD lock integration_abi")
    _validate_source_tree(source_root, integration)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-source", type=Path)
    parser.add_argument("--validate-source", type=Path)
    arguments = parser.parse_args()
    try:
        if arguments.validate_source is not None:
            if arguments.output is not None or arguments.expected_source is not None:
                raise RayDDiscoveryError(
                    "--validate-source cannot be combined with --output or --expected-source"
                )
            validate_source(arguments.lock, arguments.validate_source)
        else:
            result = resolve(arguments.lock)
            if arguments.expected_source is not None:
                expected = arguments.expected_source.resolve(strict=True)
                actual = Path(str(result["source_dir"])).resolve(strict=True)
                if actual != expected:
                    raise RayDDiscoveryError(
                        f"RayD package source changed after configure: {actual} != {expected}"
                    )
            if arguments.output is not None:
                arguments.output.write_text(
                    json.dumps(result, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                    newline="\n",
                )
    except RayDDiscoveryError as exc:
        parser.exit(2, f"RayD package discovery failed: {exc}\n")


if __name__ == "__main__":
    main()

# Copyright Xingyu Chen.
# Immutable, root-relative raw evidence artifacts.

"""Immutable, root-relative raw evidence artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile

from .contracts import EvidenceError


_REPARSE_FLAG = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
RAW_ARG_PREFIX = "{RAW}/"


def _is_reparse_point(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError as exc:
        raise EvidenceError(f"cannot inspect artifact path {path}: {exc}") from exc
    return path.is_symlink() or bool(
        getattr(info, "st_file_attributes", 0) & _REPARSE_FLAG
    )


def reject_reparse_chain(path: Path, *, stop: Path | None = None) -> None:
    current = path
    stop_resolved = stop.resolve() if stop is not None else None
    while current.exists():
        if _is_reparse_point(current):
            raise EvidenceError(f"artifact path contains a symlink/junction: {current}")
        if stop_resolved is not None and current.resolve() == stop_resolved:
            return
        if current.parent == current:
            return
        current = current.parent


def safe_relative_path(value: object) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise EvidenceError("artifact path must be a non-empty relative POSIX path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise EvidenceError(f"unsafe artifact path: {value}")
    if ":" in relative.parts[0] or "\\" in value:
        raise EvidenceError(f"unsafe artifact path: {value}")
    return relative


def _validate_open_file(stream: object, *, label: str) -> os.stat_result:
    try:
        info = os.fstat(stream.fileno())  # type: ignore[attr-defined]
    except OSError as exc:
        raise EvidenceError(f"cannot inspect opened {label}: {exc}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise EvidenceError(f"opened {label} is not a regular file")
    if getattr(info, "st_file_attributes", 0) & _REPARSE_FLAG:
        raise EvidenceError(f"opened {label} is a reparse point")
    return info


def _stream_open_file(
    stream: object,
    *,
    label: str,
    sink: object | None = None,
    collect_limit: int | None = None,
) -> tuple[bytes | None, str, int]:
    before = _validate_open_file(stream, label=label)
    digest = hashlib.sha256()
    payload = bytearray() if collect_limit is not None else None
    size = 0
    while True:
        chunk = stream.read(1024 * 1024)  # type: ignore[attr-defined]
        if not chunk:
            break
        if sink is not None:
            sink.write(chunk)  # type: ignore[attr-defined]
        if payload is not None:
            if size + len(chunk) > collect_limit:
                raise EvidenceError(
                    f"{label} exceeds the fixed read limit of {collect_limit} bytes"
                )
            payload.extend(chunk)
        digest.update(chunk)
        size += len(chunk)
    after = _validate_open_file(stream, label=label)
    stable = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) == (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if not stable or size != after.st_size:
        raise EvidenceError(f"opened {label} changed while being read")
    return (bytes(payload) if payload is not None else None), digest.hexdigest(), size


def read_external_stable(
    path: Path, *, label: str, allow_empty: bool = False,
    max_bytes: int = 16 * 1024 * 1024,
) -> tuple[bytes, str, int]:
    reject_reparse_chain(path)
    try:
        with path.open("rb") as stream:
            payload, digest, size = _stream_open_file(
                stream, label=label, collect_limit=max_bytes
            )
    except OSError as exc:
        raise EvidenceError(f"cannot read {label} {path}: {exc}") from exc
    if size == 0 and not allow_empty:
        raise EvidenceError(f"{label} must not be empty: {path}")
    assert payload is not None
    return payload, digest, size


def hash_external_stable(
    path: Path, *, label: str, allow_empty: bool = False
) -> tuple[str, int]:
    reject_reparse_chain(path)
    try:
        with path.open("rb") as stream:
            _, digest, size = _stream_open_file(stream, label=label)
    except OSError as exc:
        raise EvidenceError(f"cannot hash {label} {path}: {exc}") from exc
    if size == 0 and not allow_empty:
        raise EvidenceError(f"{label} must not be empty: {path}")
    return digest, size


@dataclass(frozen=True, slots=True)
class ArtifactStore:
    root: Path

    @classmethod
    def create(cls, parent: Path) -> "ArtifactStore":
        parent.mkdir(parents=True, exist_ok=True)
        reject_reparse_chain(parent)
        run_root = Path(tempfile.mkdtemp(prefix="phase13-phase12-", dir=parent))
        reject_reparse_chain(run_root, stop=parent)
        return cls(run_root.resolve())

    @classmethod
    def open_existing(cls, root: Path) -> "ArtifactStore":
        if not root.is_dir():
            raise EvidenceError(f"raw artifact root is not a directory: {root}")
        reject_reparse_chain(root)
        return cls(root.resolve())

    def _path(self, relative_value: object) -> Path:
        relative = safe_relative_path(relative_value)
        path = self.root.joinpath(*relative.parts)
        parent = path.parent
        if parent.exists():
            reject_reparse_chain(parent, stop=self.root)
        resolved_parent = parent.resolve()
        if resolved_parent != self.root and not resolved_parent.is_relative_to(self.root):
            raise EvidenceError(f"artifact escapes raw root: {relative}")
        return path

    def make_directory(self, relative_value: object) -> Path:
        path = self._path(relative_value)
        path.mkdir(parents=True, exist_ok=False)
        reject_reparse_chain(path, stop=self.root)
        return path

    def write_bytes(
        self, relative_value: object, payload: bytes, *, allow_empty: bool = True
    ) -> dict[str, object]:
        if not payload and not allow_empty:
            raise EvidenceError(f"artifact must not be empty: {relative_value}")
        path = self._path(relative_value)
        path.parent.mkdir(parents=True, exist_ok=True)
        reject_reparse_chain(path.parent, stop=self.root)
        try:
            with path.open("x+b") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
                stream.seek(0)
                _, digest, size = _stream_open_file(
                    stream, label=f"raw artifact {path}"
                )
        except OSError as exc:
            raise EvidenceError(f"cannot create raw artifact {path}: {exc}") from exc
        if size != len(payload) or digest != hashlib.sha256(payload).hexdigest():
            raise EvidenceError(f"short write for raw artifact {path}")
        return {"path": safe_relative_path(relative_value).as_posix(), "sha256": digest, "bytes": size}

    def retain_external(
        self,
        source: Path,
        relative_value: object,
        *,
        label: str,
        allow_empty: bool = False,
        minimum_mtime_ns: int | None = None,
    ) -> dict[str, object]:
        """Copy one external file while reading/hash-checking one source handle."""
        target = self._path(relative_value)
        target.parent.mkdir(parents=True, exist_ok=True)
        reject_reparse_chain(target.parent, stop=self.root)
        reject_reparse_chain(source)
        try:
            with source.open("rb") as source_stream, target.open("x+b") as target_stream:
                opened = _validate_open_file(source_stream, label=label)
                if minimum_mtime_ns is not None and opened.st_mtime_ns < minimum_mtime_ns:
                    raise EvidenceError(f"{label} predates the required fresh invocation")
                _, source_digest, source_size = _stream_open_file(
                    source_stream, label=label, sink=target_stream
                )
                target_stream.flush()
                os.fsync(target_stream.fileno())
                target_stream.seek(0)
                _, target_digest, target_size = _stream_open_file(
                    target_stream, label=f"retained {label}"
                )
        except OSError as exc:
            raise EvidenceError(f"cannot retain {label}: {exc}") from exc
        if source_size == 0 and not allow_empty:
            raise EvidenceError(f"{label} must not be empty")
        if (source_digest, source_size) != (target_digest, target_size):
            raise EvidenceError(f"retained {label} differs from opened source")
        return {
            "path": safe_relative_path(relative_value).as_posix(),
            "sha256": target_digest,
            "bytes": target_size,
        }

    def inspect(
        self,
        relative_value: object,
        *,
        label: str,
        allow_empty: bool = False,
        minimum_mtime_ns: int | None = None,
    ) -> dict[str, object]:
        relative = safe_relative_path(relative_value)
        path = self._path(relative.as_posix())
        if not path.is_file() or _is_reparse_point(path):
            raise EvidenceError(f"{label} is missing or is a reparse point: {relative}")
        try:
            with path.open("rb") as stream:
                opened = _validate_open_file(stream, label=label)
                if minimum_mtime_ns is not None and opened.st_mtime_ns < minimum_mtime_ns:
                    raise EvidenceError(f"{label} predates its capture invocation")
                _, digest, size = _stream_open_file(stream, label=label)
        except OSError as exc:
            raise EvidenceError(f"cannot read {label} {relative}: {exc}") from exc
        if size == 0 and not allow_empty:
            raise EvidenceError(f"{label} must not be empty: {relative}")
        return {"path": relative.as_posix(), "sha256": digest, "bytes": size}

    def verify_reference(
        self, reference: object, *, label: str, allow_empty: bool = False,
        max_bytes: int = 64 * 1024 * 1024,
    ) -> dict[str, object]:
        if not isinstance(reference, dict) or set(reference) != {"path", "sha256", "bytes"}:
            raise EvidenceError(f"{label} artifact reference is malformed")
        actual = self.inspect(reference["path"], label=label, allow_empty=allow_empty)
        if actual != reference:
            raise EvidenceError(f"{label} artifact hash/size differs from the report")
        return actual

    def read_verified(
        self, reference: object, *, label: str, allow_empty: bool = False,
        max_bytes: int = 64 * 1024 * 1024,
    ) -> bytes:
        if not isinstance(reference, dict) or set(reference) != {"path", "sha256", "bytes"}:
            raise EvidenceError(f"{label} artifact reference is malformed")
        relative = safe_relative_path(reference["path"])
        path = self._path(relative.as_posix())
        if _is_reparse_point(path):
            raise EvidenceError(f"{label} is a reparse point")
        try:
            with path.open("rb") as stream:
                payload, digest, size = _stream_open_file(
                    stream, label=label, collect_limit=max_bytes
                )
        except OSError as exc:
            raise EvidenceError(f"cannot read verified {label}: {exc}") from exc
        actual = {"path": relative.as_posix(), "sha256": digest, "bytes": size}
        if actual != reference:
            raise EvidenceError(f"{label} artifact hash/size differs from the report")
        if size == 0 and not allow_empty:
            raise EvidenceError(f"{label} must not be empty")
        assert payload is not None
        return payload

    def inventory(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for path in sorted(self.root.rglob("*")):
            if path.is_dir():
                if _is_reparse_point(path):
                    raise EvidenceError(f"raw artifact tree contains a junction: {path}")
                continue
            relative = path.relative_to(self.root).as_posix()
            rows.append(self.inspect(relative, label="raw artifact", allow_empty=True))
        return rows

    def verify_inventory(self, expected: object) -> list[dict[str, object]]:
        if not isinstance(expected, list):
            raise EvidenceError("raw artifact inventory must be an array")
        actual = self.inventory()
        if actual != expected:
            raise EvidenceError("raw artifact inventory differs from the report")
        return actual

    def relative_for_created_file(self, path: Path) -> str:
        resolved = path.resolve()
        if resolved == self.root or not resolved.is_relative_to(self.root):
            raise EvidenceError(f"artifact is outside raw root: {path}")
        return resolved.relative_to(self.root).as_posix()

    def normalize_argv(self, argv: list[str]) -> list[str]:
        """Replace raw-root absolute argv items with relocatable tokens."""
        normalized: list[str] = []
        for item in argv:
            try:
                path = Path(item)
                resolved = path.resolve() if path.is_absolute() else None
            except (OSError, ValueError):
                resolved = None
            if (
                resolved is not None
                and resolved != self.root
                and resolved.is_relative_to(self.root)
            ):
                normalized.append(
                    RAW_ARG_PREFIX + resolved.relative_to(self.root).as_posix()
                )
            else:
                normalized.append(item)
        return normalized


__all__ = [
    "ArtifactStore", "RAW_ARG_PREFIX", "hash_external_stable",
    "read_external_stable",
    "reject_reparse_chain", "safe_relative_path",
]
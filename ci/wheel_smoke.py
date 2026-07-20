from __future__ import annotations

import argparse
import base64
import csv
from functools import lru_cache
import hashlib
from email.parser import BytesParser
from email import policy
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unicodedata
import zipfile

if __package__:
    from . import audit_windows_pe
else:
    import audit_windows_pe


_DISTRIBUTION = "witwin-channel-native"
_DIST_INFO_FILES = frozenset({"METADATA", "RECORD", "WHEEL"})
_NATIVE_MEMBER = "witwin/channel_native/_channel_native.cp311-win_amd64.pyd"
_REQUIRED_RUNTIME_MEMBERS = {
    "witwin/channel_native/runtime/_channel_native.build-fingerprint",
    "witwin/channel_native/runtime/rayd.lock.json",
}
_SPECIAL_PACKAGE_MEMBERS = frozenset({_NATIVE_MEMBER, *_REQUIRED_RUNTIME_MEMBERS})
_SMOKE_KEYS = frozenset(
    {
        "build_info",
        "distribution",
        "native_origin",
        "package_origin",
        "wheel_sha256",
        "wheel_smoke",
    }
)
_BUILD_INFO_KEYS = frozenset(
    {
        "backend",
        "build_fingerprint",
        "build_type",
        "channel_native_abi_version",
        "channel_native_git_dirty",
        "channel_native_git_sha",
        "compiler",
        "cuda_architectures",
        "cuda_available",
        "cuda_compiler_version",
        "cuda_version",
        "cxx_abi",
        "material_abi_version",
        "optix_available",
        "rayd_commit",
        "rayd_dirty",
        "rayd_integration",
        "rayd_integration_abi_kind",
        "rayd_integration_abi_path",
        "rayd_integration_abi_sha256",
        "rayd_repository_url",
        "torch_version",
        "uses_dr_jit",
        "uses_path_native",
        "uses_rayd_native",
    }
)
_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_BUILD_SUFFIXES = {
    ".exp",
    ".ilk",
    ".lib",
    ".lock",
    ".o",
    ".obj",
    ".pdb",
}
_ABSOLUTE_PATH_PATTERN = re.compile(
    rb"(?:(?<![A-Za-z0-9+.-])[A-Za-z]:[\\/]"
    rb"|(?<![A-Za-z0-9])/(?:home|Users|private/tmp|tmp|workspace)/)"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_wheel(path: Path) -> Path:
    """Resolve a wheel path or require exactly one wheel in a directory."""

    path = path.resolve()
    if path.is_dir():
        wheels = sorted(path.glob("*.whl"))
        if len(wheels) != 1:
            raise ValueError(
                f"wheel directory must contain exactly one .whl file; found {len(wheels)}"
            )
        return wheels[0]
    if path.suffix != ".whl" or not path.is_file():
        raise ValueError(f"wheel does not exist: {path}")
    return path


def _canonical_member(name: str) -> str:
    if (
        not name
        or "\\" in name
        or name.startswith("/")
        or re.match(r"^[A-Za-z]:", name)
    ):
        raise ValueError(f"wheel member is not a canonical relative path: {name!r}")
    if unicodedata.normalize("NFC", name) != name or any(
        ord(char) < 32 for char in name
    ):
        raise ValueError(f"wheel member is not canonical Unicode/text: {name!r}")
    parts = name.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"wheel member has an empty/dot/traversal segment: {name!r}")
    return "/".join(parts)


def _canonical_members(archive: zipfile.ZipFile) -> list[str]:
    members: list[str] = []
    identities: set[str] = set()
    for info in archive.infolist():
        if info.is_dir():
            raise ValueError(
                f"wheel must not contain directory entries: {info.filename!r}"
            )
        member = _canonical_member(info.filename)
        identity = unicodedata.normalize("NFC", member).casefold()
        if identity in identities:
            raise ValueError(
                f"wheel contains duplicate normalized/casefold member: {member!r}"
            )
        identities.add(identity)
        members.append(member)
    if not members:
        raise ValueError("wheel contains no members")
    return members


def _metadata_identity(
    archive: zipfile.ZipFile, members: list[str]
) -> tuple[str, str, str]:
    metadata_files = [
        member
        for member in members
        if len(member.split("/")) == 2
        and member.split("/")[0].endswith(".dist-info")
        and member.split("/")[1] == "METADATA"
    ]
    if len(metadata_files) != 1:
        raise ValueError(
            "wheel must contain exactly one canonical .dist-info/METADATA file; "
            f"found {metadata_files}"
        )
    metadata = BytesParser(policy=policy.default).parsebytes(
        archive.read(metadata_files[0])
    )
    names = metadata.get_all("Name", [])
    versions = metadata.get_all("Version", [])
    if len(names) != 1 or len(versions) != 1:
        raise ValueError("wheel METADATA must contain exactly one Name and Version")
    name, version = names[0], versions[0]
    if name != _DISTRIBUTION or not version or any(char.isspace() for char in version):
        raise ValueError(
            f"wheel identity must be {_DISTRIBUTION!r} with a non-empty version; "
            f"found name={name!r}, version={version!r}"
        )
    dist_info = f"witwin_channel_native-{version}.dist-info"
    if metadata_files[0] != f"{dist_info}/METADATA":
        raise ValueError(
            "wheel .dist-info directory does not exactly match METADATA version: "
            f"{metadata_files[0]!r}"
        )
    return name, version, dist_info


@lru_cache(maxsize=1)
def _checked_in_package_members() -> frozenset[str]:
    repository_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", "src/witwin"],
        cwd=repository_root,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"cannot read checked-in src/witwin member list: {detail}")
    paths = [path for path in result.stdout.decode("utf-8").split("\0") if path]
    prefix = "src/"
    if not paths or any(not path.startswith(prefix) for path in paths):
        raise ValueError("checked-in src/witwin member list is malformed or empty")
    return frozenset(path[len(prefix) :] for path in paths)


def _source_member_payload(member: str) -> bytes:
    repository_root = Path(__file__).resolve().parents[1]
    path = repository_root / "src" / member
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ValueError(f"checked-in source member is unreadable: {member!r}") from exc


def _rayd_lock_identity(payload: bytes, *, label: str) -> dict[str, str]:
    try:
        lock = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from exc
    lock = _exact_keys(
        lock,
        frozenset({"commit", "integration_abi", "repository_url", "schema_version"}),
        label=label,
    )
    integration = _exact_keys(
        lock["integration_abi"],
        frozenset({"kind", "path", "sha256"}),
        label=f"{label} integration ABI",
    )
    if type(lock["schema_version"]) is not int or lock["schema_version"] != 1:
        raise ValueError(f"{label} schema_version must be integer 1")
    strings = {
        "rayd_commit": lock["commit"],
        "rayd_repository_url": lock["repository_url"],
        "rayd_integration_abi_kind": integration["kind"],
        "rayd_integration_abi_path": integration["path"],
        "rayd_integration_abi_sha256": integration["sha256"],
    }
    if any(type(value) is not str or not value for value in strings.values()):
        raise ValueError(f"{label} identity fields must be non-empty strings")
    if _SHA_PATTERN.fullmatch(str(strings["rayd_commit"])) is None:
        raise ValueError(f"{label} commit must be a Git SHA")
    if _SHA256_PATTERN.fullmatch(str(strings["rayd_integration_abi_sha256"])) is None:
        raise ValueError(f"{label} integration ABI must be a SHA-256")
    return {name: str(value) for name, value in strings.items()}


def _runtime_identity_from_archive(archive: zipfile.ZipFile) -> dict[str, object]:
    fingerprint_member = (
        "witwin/channel_native/runtime/_channel_native.build-fingerprint"
    )
    fingerprint = archive.read(fingerprint_member)
    match = re.fullmatch(rb"([0-9a-f]{64})(?:\r?\n)?", fingerprint)
    if match is None:
        raise ValueError("wheel build fingerprint must contain exactly one SHA-256")

    lock_member = "witwin/channel_native/runtime/rayd.lock.json"
    wheel_lock_payload = archive.read(lock_member)
    repository_root = Path(__file__).resolve().parents[1]
    try:
        authoritative_payload = (
            repository_root / "dependencies" / "rayd.lock.json"
        ).read_bytes()
    except OSError as exc:
        raise ValueError(
            "authoritative dependencies/rayd.lock.json is unreadable"
        ) from exc
    wheel_identity = _rayd_lock_identity(wheel_lock_payload, label="wheel RayD lock")
    authoritative_identity = _rayd_lock_identity(
        authoritative_payload, label="authoritative RayD lock"
    )
    if (
        wheel_identity != authoritative_identity
        or wheel_lock_payload != authoritative_payload
    ):
        raise ValueError(
            "wheel RayD lock does not exactly match authoritative "
            "dependencies/rayd.lock.json"
        )
    return {
        "build_fingerprint": match.group(1).decode("ascii"),
        "build_type": "Release",
        "channel_native_abi_version": 1,
        "channel_native_git_dirty": False,
        "rayd_dirty": False,
        **wheel_identity,
    }


def _wheel_runtime_identity(path: Path) -> dict[str, object]:
    with zipfile.ZipFile(path) as archive:
        return _runtime_identity_from_archive(archive)


def _record_hash(payload: bytes) -> str:
    digest = hashlib.sha256(payload).digest()
    encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return f"sha256={encoded}"


def _audit_record(
    archive: zipfile.ZipFile, members: list[str], *, dist_info: str
) -> None:
    record_member = f"{dist_info}/RECORD"
    try:
        record_text = archive.read(record_member).decode("utf-8")
    except (KeyError, UnicodeDecodeError) as exc:
        raise ValueError("wheel RECORD is missing or is not UTF-8") from exc
    try:
        rows = list(csv.reader(io.StringIO(record_text, newline=""), strict=True))
    except csv.Error as exc:
        raise ValueError(f"wheel RECORD is invalid CSV: {exc}") from exc
    if not rows or any(len(row) != 3 for row in rows):
        raise ValueError("wheel RECORD must contain non-empty three-column rows")

    recorded: dict[str, tuple[str, str]] = {}
    for raw_member, digest, size in rows:
        member = _canonical_member(raw_member)
        identity = member.casefold()
        if identity in recorded:
            raise ValueError(f"wheel RECORD contains duplicate member: {member!r}")
        recorded[identity] = (digest, size)
    expected_identities = {member.casefold(): member for member in members}
    if set(recorded) != set(expected_identities):
        missing = sorted(set(expected_identities) - set(recorded))
        extra = sorted(set(recorded) - set(expected_identities))
        raise ValueError(
            f"wheel RECORD member coverage mismatch: missing={missing}, extra={extra}"
        )
    mismatched_case = sorted(
        member
        for member in expected_identities.values()
        if member not in {row[0] for row in rows}
    )
    if mismatched_case:
        raise ValueError(
            f"wheel RECORD member spelling/case mismatch: {mismatched_case}"
        )

    for identity, member in expected_identities.items():
        digest, size = recorded[identity]
        if member == record_member:
            if digest or size:
                raise ValueError("wheel RECORD self row must have empty hash and size")
            continue
        payload = archive.read(member)
        if digest != _record_hash(payload) or size != str(len(payload)):
            raise ValueError(f"wheel RECORD hash/size mismatch for {member!r}")


def _wheel_identity(path: Path) -> tuple[str, str]:
    with zipfile.ZipFile(path) as archive:
        name, version, _ = _metadata_identity(archive, _canonical_members(archive))
    return name, version


def _audit_wheel_contents(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        names = _canonical_members(archive)
        _, _, dist_info = _metadata_identity(archive, names)
        native_extensions = [name for name in names if name == _NATIVE_MEMBER]
        if native_extensions != [_NATIVE_MEMBER]:
            raise ValueError(
                "wheel must contain exactly one packaged _channel_native extension; "
                f"found {native_extensions}"
            )
        native_extension = _NATIVE_MEMBER
        shared_libraries = [
            name
            for name in names
            if Path(name).suffix.lower() in {".dll", ".dylib", ".pyd", ".so"}
        ]
        if shared_libraries != [native_extension]:
            raise ValueError(
                "wheel must contain no DSO except its single _channel_native "
                f"extension; found {shared_libraries}"
            )
        missing = sorted(_REQUIRED_RUNTIME_MEMBERS.difference(names))
        if missing:
            raise ValueError(f"wheel is missing runtime identity files: {missing}")

        checked_in = _checked_in_package_members()
        package_members = {name for name in names if name.startswith("witwin/")}
        expected_package_members = set(checked_in) | set(_SPECIAL_PACKAGE_MEMBERS)
        if package_members != expected_package_members:
            missing_package = sorted(expected_package_members - package_members)
            extra_package = sorted(package_members - expected_package_members)
            raise ValueError(
                "wheel package source closure mismatch: "
                f"missing={missing_package}, extra={extra_package}"
            )
        mismatched_sources = sorted(
            member
            for member in checked_in
            if archive.read(member) != _source_member_payload(member)
        )
        if mismatched_sources:
            raise ValueError(
                f"wheel checked-in source bytes differ: {mismatched_sources}"
            )

        allowed_dist_info = {f"{dist_info}/{name}" for name in _DIST_INFO_FILES}
        unexpected_members = sorted(
            name
            for name in names
            if name not in checked_in
            and name not in _SPECIAL_PACKAGE_MEMBERS
            and name not in allowed_dist_info
        )
        if unexpected_members:
            raise ValueError(
                "wheel member is outside the checked-in package/dist-info allowlist: "
                f"{unexpected_members}"
            )
        roots = {name.split("/", 1)[0] for name in names}
        if roots != {"witwin", dist_info}:
            raise ValueError(f"wheel has unexpected top-level roots: {sorted(roots)}")
        dist_info_members = {name for name in names if name.startswith(f"{dist_info}/")}
        if dist_info_members != allowed_dist_info:
            raise ValueError(
                "wheel dist-info must contain exactly METADATA, WHEEL, and RECORD"
            )
        _audit_record(archive, names, dist_info=dist_info)
        _runtime_identity_from_archive(archive)

        forbidden: list[str] = []
        for name in names:
            parts = tuple(part.lower() for part in name.split("/"))
            suffix = Path(name).suffix.lower()
            if (
                (parts and parts[0] == "rayd")
                or any(part in {"cmakefiles", "_skbuild", "build"} for part in parts)
                or suffix in _FORBIDDEN_BUILD_SUFFIXES
                or name.endswith(".pyd.pyd")
            ):
                forbidden.append(name)
                continue
            if suffix in {".json", ".md", ".py", ".toml", ".txt"}:
                if _ABSOLUTE_PATH_PATTERN.search(archive.read(name)):
                    forbidden.append(f"{name} (contains an absolute local path)")
        if forbidden:
            raise ValueError(f"wheel contains forbidden build content: {forbidden}")
    return native_extension


def _audit_wheel_pe(
    path: Path, native_member: str, *, dumpbin: str = "dumpbin"
) -> dict[str, object]:
    with zipfile.ZipFile(path) as archive:
        payload = archive.read(native_member)
    with tempfile.TemporaryDirectory(prefix="channel-native-pe-audit-") as raw:
        extracted = Path(raw) / Path(native_member).name
        extracted.write_bytes(payload)
        evidence = audit_windows_pe.audit_pe(extracted, dumpbin=dumpbin)
    evidence["wheel_member"] = native_member
    evidence.pop("path", None)
    return evidence


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"isolated wheel smoke JSON has duplicate key {key!r}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"isolated wheel smoke JSON contains non-finite value {value}")


def _exact_keys(
    value: object, expected: frozenset[str], *, label: str
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != expected:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(f"isolated wheel smoke {label} schema mismatch: {actual}")
    return value


def _isolated_path(value: object, *, label: str, target: Path) -> Path:
    if type(value) is not str or not value:
        raise ValueError(f"isolated wheel smoke {label} must be a non-empty string")
    try:
        resolved = Path(value).resolve(strict=True)
    except OSError as exc:
        raise ValueError(
            f"isolated wheel smoke {label} is not an existing path"
        ) from exc
    if not resolved.is_relative_to(target):
        raise ValueError(f"isolated wheel smoke {label} resolved outside target")
    return resolved


def _validate_build_info(value: object) -> dict[str, object]:
    info = _exact_keys(value, _BUILD_INFO_KEYS, label="build_info")
    booleans = {
        "channel_native_git_dirty",
        "cuda_available",
        "optix_available",
        "rayd_dirty",
        "uses_dr_jit",
        "uses_path_native",
        "uses_rayd_native",
    }
    integers = {"channel_native_abi_version", "material_abi_version"}
    strings = _BUILD_INFO_KEYS - booleans - integers - {"cuda_architectures"}
    if any(type(info[name]) is not bool for name in booleans):
        raise ValueError("isolated wheel smoke build_info boolean field has wrong type")
    if any(type(info[name]) is not int for name in integers):
        raise ValueError("isolated wheel smoke build_info integer field has wrong type")
    if any(type(info[name]) is not str or not info[name] for name in strings):
        raise ValueError(
            "isolated wheel smoke build_info string field is empty or invalid"
        )
    architectures = info["cuda_architectures"]
    if (
        not isinstance(architectures, list)
        or not architectures
        or not all(
            type(item) is str and re.fullmatch(r"\d+-(?:real|virtual)", item)
            for item in architectures
        )
    ):
        raise ValueError(
            "isolated wheel smoke build_info CUDA architectures are invalid"
        )
    expected_values = {
        "backend": "channel-native",
        "build_type": "Release",
        "channel_native_abi_version": 1,
        "material_abi_version": 3,
        "rayd_integration": "source-linked",
        "rayd_integration_abi_kind": "source-header-sha256",
        "rayd_integration_abi_path": "backends/torch/include/rayd/torch/integration_v2.h",
        "rayd_repository_url": "https://github.com/Asixa/RayD.git",
        "uses_dr_jit": False,
        "uses_path_native": True,
        "uses_rayd_native": True,
    }
    mismatched = [
        name for name, expected in expected_values.items() if info[name] != expected
    ]
    if mismatched:
        raise ValueError(
            "isolated wheel smoke build_info has unexpected release identity: "
            + ", ".join(mismatched)
        )
    if info["channel_native_git_dirty"] is not False or info["rayd_dirty"] is not False:
        raise ValueError(
            "isolated wheel smoke build_info must report clean repositories"
        )
    if _SHA_PATTERN.fullmatch(str(info["channel_native_git_sha"])) is None:
        raise ValueError("isolated wheel smoke Channel Git identity is invalid")
    if _SHA_PATTERN.fullmatch(str(info["rayd_commit"])) is None:
        raise ValueError("isolated wheel smoke RayD Git identity is invalid")
    if _SHA256_PATTERN.fullmatch(str(info["rayd_integration_abi_sha256"])) is None:
        raise ValueError("isolated wheel smoke RayD ABI identity is invalid")
    if _SHA256_PATTERN.fullmatch(str(info["build_fingerprint"])) is None:
        raise ValueError("isolated wheel smoke build fingerprint is invalid")
    return info


def _parse_smoke_evidence(
    stdout: str,
    *,
    expected_wheel_sha256: str,
    expected_name: str,
    expected_version: str,
    target: Path,
    native_member: str,
    expected_build_identity: dict[str, object],
) -> dict[str, object]:
    try:
        evidence = json.loads(
            stdout,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("isolated wheel smoke did not emit one JSON object") from exc
    evidence = _exact_keys(evidence, _SMOKE_KEYS, label="top-level")
    if evidence["wheel_smoke"] is not True:
        raise ValueError("isolated wheel smoke success flag must be true boolean")
    if (
        type(evidence["wheel_sha256"]) is not str
        or evidence["wheel_sha256"] != expected_wheel_sha256
    ):
        raise ValueError("isolated wheel smoke wheel SHA-256 disagrees with parent")
    distribution = _exact_keys(
        evidence["distribution"],
        frozenset({"name", "root", "version"}),
        label="distribution",
    )
    if (
        distribution["name"] != expected_name
        or distribution["version"] != expected_version
    ):
        raise ValueError(
            "isolated wheel smoke distribution identity disagrees with parent"
        )

    target = target.resolve(strict=True)
    distribution_root = _isolated_path(
        distribution["root"], label="distribution root", target=target
    )
    if not distribution_root.is_dir() or distribution_root != target:
        raise ValueError("isolated wheel smoke distribution root is not a directory")
    package_origin = _isolated_path(
        evidence["package_origin"], label="package origin", target=target
    )
    native_origin = _isolated_path(
        evidence["native_origin"], label="native origin", target=target
    )
    expected_package_origin = (target / "witwin/channel_native/__init__.py").resolve()
    if package_origin != expected_package_origin or not package_origin.is_file():
        raise ValueError("isolated wheel smoke package origin is invalid")
    expected_native_origin = (target / Path(native_member)).resolve()
    if native_origin != expected_native_origin or not native_origin.is_file():
        raise ValueError("isolated wheel smoke native origin basename is invalid")
    if native_origin.relative_to(target).as_posix() != native_member:
        raise ValueError("isolated wheel smoke native origin member is invalid")
    build_info = _validate_build_info(evidence["build_info"])
    mismatched_build_identity = [
        name
        for name, expected in expected_build_identity.items()
        if build_info.get(name) != expected
    ]
    if mismatched_build_identity:
        raise ValueError(
            "isolated wheel smoke build_info disagrees with wheel identity: "
            + ", ".join(mismatched_build_identity)
        )
    return evidence


def _smoke_code(
    *,
    target: Path,
    wheel: Path,
    wheel_sha256: str,
    expected_name: str,
    expected_version: str,
) -> str:
    return f"""
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys

target = Path({str(target)!r}).resolve()
sys.path.insert(0, str(target))
wheel = Path({str(wheel)!r}).resolve()
wheel_digest = hashlib.sha256()
with wheel.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        wheel_digest.update(chunk)
wheel_sha256 = wheel_digest.hexdigest()
if wheel_sha256 != {wheel_sha256!r}:
    raise RuntimeError(f"wheel SHA-256 changed during smoke: {{wheel_sha256}}")

distribution = importlib.metadata.distribution({expected_name!r})
distribution_root = Path(distribution.locate_file(\"\")).resolve()
if not distribution_root.is_relative_to(target):
    raise RuntimeError(f\"distribution resolved outside isolated target: {{distribution_root}}\")
if distribution.metadata[\"Name\"] != {expected_name!r}:
    raise RuntimeError(f\"unexpected distribution name: {{distribution.metadata['Name']!r}}\")
if distribution.version != {expected_version!r}:
    raise RuntimeError(f\"unexpected distribution version: {{distribution.version!r}}\")

package_spec = importlib.util.find_spec(\"witwin.channel_native\")
if package_spec is None or package_spec.origin is None:
    raise RuntimeError(\"witwin.channel_native has no import origin\")
package_origin = Path(package_spec.origin).resolve()
if not package_origin.is_relative_to(target):
    raise RuntimeError(f\"package resolved outside isolated target: {{package_origin}}\")

native_spec = importlib.util.find_spec(\"witwin.channel_native._channel_native\")
if native_spec is None or native_spec.origin is None:
    raise RuntimeError(\"packaged native extension has no import origin\")
native_origin = Path(native_spec.origin).resolve()
if not native_origin.is_relative_to(target):
    raise RuntimeError(f\"native extension resolved outside isolated target: {{native_origin}}\")

import witwin.channel_native as channel_native

build_info = channel_native.build_info()
if build_info.get(\"backend\") != \"channel-native\":
    raise RuntimeError(f\"unexpected native backend: {{build_info.get('backend')!r}}\")
if build_info.get(\"uses_dr_jit\") is not False:
    raise RuntimeError(\"wheel native extension must report uses_dr_jit=false\")
if build_info.get(\"uses_rayd_native\") is not True:
    raise RuntimeError(\"wheel native extension must report uses_rayd_native=true\")

print(json.dumps({{
    \"wheel_smoke\": True,
    \"wheel_sha256\": wheel_sha256,
    \"distribution\": {{
        \"name\": distribution.metadata[\"Name\"],
        \"version\": distribution.version,
        \"root\": str(distribution_root),
    }},
    \"package_origin\": str(package_origin),
    \"native_origin\": str(native_origin),
    \"build_info\": build_info,
}}, sort_keys=True))
"""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Install a built wheel into an isolated target and smoke its native ABI."
    )
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--dumpbin", default="dumpbin")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        wheel = _resolve_wheel(args.wheel)
        expected_name, expected_version = _wheel_identity(wheel)
        native_member = _audit_wheel_contents(wheel)
        expected_build_identity = _wheel_runtime_identity(wheel)
        pe_evidence = _audit_wheel_pe(wheel, native_member, dumpbin=args.dumpbin)
    except (
        OSError,
        ValueError,
        audit_windows_pe.PEAuditError,
        zipfile.BadZipFile,
    ) as exc:
        parser.error(str(exc))
    wheel_sha256 = _sha256(wheel)

    with tempfile.TemporaryDirectory(prefix="channel-native-wheel-smoke-") as raw:
        target = Path(raw) / "site-packages"
        install = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-deps",
                "--target",
                str(target),
                str(wheel),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if install.returncode != 0:
            print(install.stderr, file=sys.stderr)
            return install.returncode

        code = _smoke_code(
            target=target,
            wheel=wheel,
            wheel_sha256=wheel_sha256,
            expected_name=expected_name,
            expected_version=expected_version,
        )
        smoke = subprocess.run(
            [sys.executable, "-I", "-c", code],
            cwd=target,
            capture_output=True,
            text=True,
            check=False,
        )
        if smoke.stderr:
            print(smoke.stderr.strip(), file=sys.stderr)
        if smoke.returncode != 0:
            if smoke.stdout:
                print(smoke.stdout.strip())
            return smoke.returncode
        try:
            evidence = _parse_smoke_evidence(
                smoke.stdout,
                expected_wheel_sha256=wheel_sha256,
                expected_name=expected_name,
                expected_version=expected_version,
                target=target,
                native_member=native_member,
                expected_build_identity=expected_build_identity,
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        evidence["pe_audit"] = pe_evidence
        encoded = json.dumps(evidence, indent=2, sort_keys=True) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded, encoding="utf-8")
        print(encoded, end="")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

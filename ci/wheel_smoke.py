from __future__ import annotations

import argparse
import hashlib
from email.parser import BytesParser
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import zipfile


_DISTRIBUTION = "witwin-channel-native"
_REQUIRED_RUNTIME_MEMBERS = {
    "witwin/channel_native/runtime/_channel_native.build-fingerprint",
    "witwin/channel_native/runtime/rayd.lock.json",
}
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


def _wheel_identity(path: Path) -> tuple[str, str]:
    with zipfile.ZipFile(path) as archive:
        metadata_files = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_files) != 1:
            raise ValueError(
                f"wheel must contain exactly one .dist-info/METADATA file; found "
                f"{len(metadata_files)}"
            )
        metadata = BytesParser().parsebytes(archive.read(metadata_files[0]))
    name = metadata.get("Name")
    version = metadata.get("Version")
    if name != _DISTRIBUTION or not version:
        raise ValueError(
            f"wheel identity must be {_DISTRIBUTION!r} with a non-empty version; "
            f"found name={name!r}, version={version!r}"
        )
    return name, version


def _audit_wheel_contents(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        native_extensions = [
            name
            for name in names
            if name.startswith("witwin/channel_native/_channel_native")
            and name.endswith((".pyd", ".so"))
        ]
        if len(native_extensions) != 1:
            raise ValueError(
                "wheel must contain exactly one packaged _channel_native extension; "
                f"found {native_extensions}"
            )
        missing = sorted(_REQUIRED_RUNTIME_MEMBERS.difference(names))
        if missing:
            raise ValueError(f"wheel is missing runtime identity files: {missing}")

        forbidden: list[str] = []
        for name in names:
            normalized = name.replace("\\", "/")
            parts = tuple(part.lower() for part in normalized.split("/") if part)
            suffix = Path(normalized).suffix.lower()
            if (
                normalized.startswith("/")
                or re.match(r"^[A-Za-z]:/", normalized)
                or (parts and parts[0] == "rayd")
                or any(part in {"cmakefiles", "_skbuild", "build"} for part in parts)
                or suffix in _FORBIDDEN_BUILD_SUFFIXES
                or normalized.endswith(".pyd.pyd")
            ):
                forbidden.append(name)
                continue
            if suffix in {".json", ".md", ".py", ".toml", ".txt"}:
                if _ABSOLUTE_PATH_PATTERN.search(archive.read(name)):
                    forbidden.append(f"{name} (contains an absolute local path)")
        if forbidden:
            raise ValueError(f"wheel contains forbidden build content: {forbidden}")


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
if build_info.get(\"uses_raydn_native\") is not True:
    raise RuntimeError(\"wheel native extension must report uses_raydn_native=true\")

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
    args = parser.parse_args()
    wheel = args.wheel.resolve()
    if wheel.suffix != ".whl" or not wheel.is_file():
        parser.error(f"wheel does not exist: {wheel}")
    try:
        expected_name, expected_version = _wheel_identity(wheel)
        _audit_wheel_contents(wheel)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
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
        if smoke.stdout:
            print(smoke.stdout.strip())
        if smoke.stderr:
            print(smoke.stderr.strip(), file=sys.stderr)
        return smoke.returncode


if __name__ == "__main__":
    raise SystemExit(main())

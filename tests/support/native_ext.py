# Copyright Xingyu Chen.
# Select an explicit development extension for tests and benchmarks.

"""Select an explicit development extension for tests and benchmarks."""

from __future__ import annotations

from datetime import datetime
import importlib.machinery
import importlib.metadata
import importlib.util
import os
from pathlib import Path, PurePosixPath

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENABLE_ENV = "WITWIN_CHANNEL_DEVELOPER_OVERRIDE"
_PATH_ENV = "WITWIN_CHANNEL_EXTENSION_PATH"
_FINGERPRINT_ENV = "WITWIN_CHANNEL_EXPECTED_FINGERPRINT"
_FINGERPRINT_FILE = "_channel.build-fingerprint"
_LISTED_BUILDS = 5


def _importable(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def _is_fingerprint(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _override_problem() -> str | None:
    """Return why the explicit override is unusable, or ``None`` when it is complete."""

    if os.environ.get(_ENABLE_ENV) != "1":
        return f"{_ENABLE_ENV} is not set to 1"
    configured = os.environ.get(_PATH_ENV)
    if configured is None:
        return f"{_PATH_ENV} is not set"
    if not Path(configured).is_absolute():
        return f"{_PATH_ENV} is not an absolute path: {configured}"
    if not Path(configured).is_file():
        return f"{_PATH_ENV} does not name a file: {configured}"
    fingerprint = os.environ.get(_FINGERPRINT_ENV)
    if fingerprint is None:
        return f"{_FINGERPRINT_ENV} is not set"
    if not _is_fingerprint(fingerprint):
        return f"{_FINGERPRINT_ENV} is not a 64-character lowercase hex digest"
    return None


def inject_native_paths() -> bool:
    """Return ``True`` when a packaged or explicitly configured extension is available.

    Local build directories are never selected automatically: picking the newest one by
    modification time silently tested native code older than the checkout, so the developer
    has to name the build through the three override variables.
    """

    return _importable("witwin.channel._channel") or _override_problem() is None


def _installed_extension() -> Path | None:
    """Return the ``_channel`` file of an installed ``witwin-channel`` distribution, if any."""

    try:
        distribution = importlib.metadata.distribution("witwin-channel")
    except importlib.metadata.PackageNotFoundError:
        return None
    for entry in distribution.files or ():
        member = PurePosixPath(str(entry).replace("\\", "/"))
        if member.parent.as_posix() != "witwin/channel" or not member.name.startswith("_channel"):
            continue
        if not any(member.name.endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES):
            continue
        located = Path(str(distribution.locate_file(entry))).resolve()
        if located.is_file():
            return located
    return None


def _local_builds() -> list[Path]:
    build_dirs = [*(_REPO_ROOT / "build").glob("dev-*"), *(_REPO_ROOT / "artifacts").glob("cmake-*")]
    builds: list[Path] = []
    for build_dir in build_dirs:
        for suffix in importlib.machinery.EXTENSION_SUFFIXES:
            extension = build_dir / f"_channel{suffix}"
            if extension.is_file():
                builds.append(extension.resolve())
                break
    return sorted(builds, key=lambda path: path.stat().st_mtime, reverse=True)


def build_guidance() -> str:
    """Explain how to name an extension, listing local builds without loading any."""

    configured = any(
        os.environ.get(name) is not None for name in (_ENABLE_ENV, _PATH_ENV, _FINGERPRINT_ENV)
    )
    problem = _override_problem()
    installed = _installed_extension()
    if configured and problem is not None:
        reason = f"Reason: {problem}."
    elif installed is not None:
        reason = (
            f"witwin-channel is installed with an extension at {installed}, but this "
            f"checkout's witwin/channel package ({_REPO_ROOT / 'witwin' / 'channel'}) shadows "
            "the installed package, so the tests import the checkout, which has no extension, "
            "and no developer override is set."
        )
    else:
        reason = "No packaged extension is installed and no developer override is set."
    lines = [
        "No validated _channel extension is configured for the test suite.",
        reason,
        "On Linux, build and select this checkout's extension with "
        "`scripts/dev_build.sh && source scripts/dev_env.sh`.",
        "Otherwise name the build to test explicitly (see "
        "docs/dev/standards/adr-006-extension-developer-override.md):",
        f"  {_ENABLE_ENV}=1",
        f"  {_PATH_ENV}=<absolute path to the _channel extension>",
        f"  {_FINGERPRINT_ENV}=<contents of {_FINGERPRINT_FILE} beside it>",
    ]
    if installed is not None and not configured:
        lines.append(
            f"  (for the installed build: {_PATH_ENV}={installed} and the fingerprint in "
            f"{installed.with_name(_FINGERPRINT_FILE)})"
        )
    builds = _local_builds()
    if builds:
        lines.append(
            "Local builds under build/ and artifacts/ (newest first; check that the one "
            "you pick was built from the current source):"
        )
        for extension in builds[:_LISTED_BUILDS]:
            built = datetime.fromtimestamp(extension.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            marker = "" if extension.with_name(_FINGERPRINT_FILE).is_file() else " (no fingerprint)"
            lines.append(f"  {built}  {extension}{marker}")
    return "\n".join(lines)

# Copyright Xingyu Chen.
# Re-lock Channel to a RayD commit and update every file that pins the lock in lockstep.

# Usage::
#
#     python tools/relock_rayd.py --rayd /path/to/RayD [--rev HEAD] [--check]
#
# The tool reads the committed tree of ``--rev`` (default ``HEAD``) of the RayD
# checkout, never its working tree, so a dirty checkout gives the same lock as a
# clean one. It exports that tree with ``git archive`` and runs RayD's own
# ``torch/scripts/generate_source_bundle.py`` on it, which yields exactly the
# metadata a ``rayd-torch`` wheel built from that commit ships: the eight
# integration-header SHA-256 values, their aggregate digest, the integration API
# version and the source-bundle manifest SHA-256. The distribution version comes
# from ``torch/pyproject.toml`` at that commit.
#
# The manifest digest covers raw file bytes, so it depends on the line endings of
# the checkout the wheel was built from. The published rayd-torch 0.8.0 wheel was
# built from a Windows checkout with ``core.autocrlf=true`` (only
# ``include/rayd/integration.h`` is pinned to LF by RayD's ``.gitattributes``);
# ``--eol crlf``, the default, reproduces its digest exactly. Use ``--eol lf`` for
# a wheel built from a Linux checkout. If the published wheel of the locked
# commit reports another digest, copy ``source_manifest.sha256`` from its
# ``rayd/torch/_source/rayd-source.json`` into the lock.
#
# It then rewrites ``dependencies/rayd.lock.json`` and replaces every old pinned
# value (commit, header digests, aggregate digest, manifest digest) by its new
# value in each tracked file outside the historical records listed in
# ``HISTORICAL_PREFIXES`` (governance tests, the publish workflow, the feature
# list). The packaged ``witwin/channel/rayd.lock.json`` is not a tracked file; the
# build installs a copy of ``dependencies/rayd.lock.json``.
#
# A change of the distribution version or the API version cannot be replaced
# textually; the tool updates the lock and lists the files that still quote the
# old value for manual review. Re-running the tool on a locked tree changes
# nothing. ``--check`` reports what would change and exits 1 if anything would.

"""Re-lock Channel to a RayD commit and update the files that pin the lock."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from io import BytesIO
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCK_RELATIVE = "dependencies/rayd.lock.json"
# Dated design records, audits, plans and baselines describe the lock as it was
# when they were written and are never rewritten.
HISTORICAL_PREFIXES = (
    "CHANGELOG.md",
    "docs/dev/audit/",
    "docs/dev/baselines/",
    "docs/dev/plans/",
    "docs/dev/standards/",
)
GENERATOR = "torch/scripts/generate_source_bundle.py"


class RelockError(RuntimeError):
    """Raised when the RayD checkout cannot produce a lock."""


def _git_bytes(cwd: Path, *arguments: str) -> bytes:
    completed = subprocess.run(("git", *arguments), cwd=cwd, check=False, capture_output=True)
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", "replace").strip()
        raise RelockError(f"git {' '.join(arguments)} failed in {cwd}: {message}")
    return completed.stdout


def _git(cwd: Path, *arguments: str) -> str:
    return _git_bytes(cwd, *arguments).decode("utf-8").strip()


def _normalize_repository_url(url: str) -> str:
    url = url.strip()
    match = re.fullmatch(r"git@([^:]+):(.+)", url)
    if match:
        url = f"https://{match.group(1)}/{match.group(2)}"
    return url.removesuffix(".git").rstrip("/")


def compute_lock(rayd: Path, rev: str, old_lock: dict, *, eol: str = "crlf") -> dict:
    """Return the lock for ``rev`` of the RayD checkout ``rayd``."""
    commit = _git(rayd, "rev-parse", "--verify", f"{rev}^{{commit}}")
    origin = _normalize_repository_url(_git(rayd, "remote", "get-url", "origin"))
    if origin != _normalize_repository_url(old_lock["repository_url"]):
        raise RelockError(
            f"RayD origin is {origin!r}, the lock pins {old_lock['repository_url']!r}"
        )
    autocrlf = "true" if eol == "crlf" else "false"
    archive = _git_bytes(
        rayd, "-c", f"core.autocrlf={autocrlf}", "archive", "--format=tar", commit
    )
    with tempfile.TemporaryDirectory(prefix="channel-relock-") as scratch:
        workspace = Path(scratch) / "workspace"
        output = Path(scratch) / "bundle"
        workspace.mkdir()
        with tarfile.open(fileobj=BytesIO(archive)) as tar:
            tar.extractall(workspace, filter="data")
        generator = workspace / GENERATOR
        if not generator.is_file():
            raise RelockError(
                f"RayD {commit} has no {GENERATOR}; fill integration_abi and "
                "source_bundle.manifest_sha256 from the rayd-source.json of a "
                "rayd-torch wheel built from that commit"
            )
        completed = subprocess.run(
            (
                sys.executable,
                "-I",
                str(generator),
                "--workspace",
                str(workspace),
                "--output",
                str(output),
                "--distribution-version",
                "0",
                "--commit",
                commit,
                "--repository-url",
                old_lock["repository_url"],
            ),
            cwd=scratch,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RelockError(f"RayD source-bundle generator failed:\n{completed.stderr}")
        metadata = json.loads((output / "rayd-source.json").read_text(encoding="utf-8"))
        pyproject = tomllib.loads(
            (workspace / "torch" / "pyproject.toml").read_text(encoding="utf-8")
        )
    if metadata["integration_abi"]["kind"] != old_lock["integration_abi"]["kind"]:
        raise RelockError("RayD integration ABI kind changed; the lock schema needs an update")
    bundle = dict(old_lock["source_bundle"])
    if pyproject["project"]["name"] != bundle["distribution"]:
        raise RelockError(
            f"RayD torch distribution is {pyproject['project']['name']!r}, "
            f"the lock pins {bundle['distribution']!r}"
        )
    bundle["distribution_version"] = pyproject["project"]["version"]
    bundle["manifest_sha256"] = metadata["source_manifest"]["sha256"]
    return {
        "schema_version": old_lock["schema_version"],
        "repository_url": old_lock["repository_url"],
        "commit": commit,
        "integration_abi": metadata["integration_abi"],
        "source_bundle": bundle,
    }


def render_lock(lock: dict) -> str:
    """Serialize the lock in its committed layout (one line per header)."""
    abi = lock["integration_abi"]
    headers = ",\n".join(
        f"      {json.dumps({'path': header['path'], 'sha256': header['sha256']})}"
        for header in abi["headers"]
    )
    bundle = lock["source_bundle"]
    lines = [
        "{",
        f'  "schema_version": {json.dumps(lock["schema_version"])},',
        f'  "repository_url": {json.dumps(lock["repository_url"])},',
        f'  "commit": {json.dumps(lock["commit"])},',
        '  "integration_abi": {',
        f'    "kind": {json.dumps(abi["kind"])},',
        f'    "entrypoint": {json.dumps(abi["entrypoint"])},',
        '    "headers": [',
        headers,
        "    ],",
        f'    "sha256": {json.dumps(abi["sha256"])},',
        f'    "api_version": {json.dumps(abi["api_version"])},',
        f'    "identity": {json.dumps(abi["identity"])}',
        "  },",
        '  "source_bundle": {',
        f'    "distribution": {json.dumps(bundle["distribution"])},',
        f'    "distribution_version": {json.dumps(bundle["distribution_version"])},',
        f'    "metadata_path": {json.dumps(bundle["metadata_path"])},',
        f'    "manifest_sha256": {json.dumps(bundle["manifest_sha256"])}',
        "  }",
        "}",
    ]
    rendered = "\n".join(lines) + "\n"
    if json.loads(rendered) != lock:
        raise RelockError("lock rendering lost information")
    return rendered


def replacements(old: dict, new: dict) -> dict[str, str]:
    """Return the textual old -> new value map for pinned digests and the commit."""
    mapping = {
        old["commit"]: new["commit"],
        old["integration_abi"]["sha256"]: new["integration_abi"]["sha256"],
        old["source_bundle"]["manifest_sha256"]: new["source_bundle"]["manifest_sha256"],
    }
    new_headers = {header["path"]: header["sha256"] for header in new["integration_abi"]["headers"]}
    for header in old["integration_abi"]["headers"]:
        if header["path"] in new_headers:
            mapping[header["sha256"]] = new_headers[header["path"]]
    old_paths = [header["path"] for header in old["integration_abi"]["headers"]]
    if sorted(old_paths) != sorted(new_headers):
        raise RelockError(
            "the RayD integration header set changed "
            f"({sorted(old_paths)} -> {sorted(new_headers)}); update the governance "
            "tests that list the headers by hand, then re-run"
        )
    return {before: after for before, after in mapping.items() if before != after}


def _lockstep_files(root: Path) -> list[str]:
    tracked = _git(root, "ls-files", "-z").split("\0")
    return sorted(
        path
        for path in tracked
        if path
        and path != LOCK_RELATIVE
        and not path.startswith(HISTORICAL_PREFIXES)
        and (root / path).is_file()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rayd", required=True, type=Path, help="RayD git checkout")
    parser.add_argument("--rev", default="HEAD", help="RayD revision to lock (default HEAD)")
    parser.add_argument("--root", type=Path, default=ROOT, help="Channel checkout to update")
    parser.add_argument(
        "--eol", choices=("crlf", "lf"), default="crlf",
        help="line endings of the checkout the rayd-torch wheel is built from (default crlf)",
    )
    parser.add_argument("--check", action="store_true", help="report changes without writing")
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    lock_path = root / LOCK_RELATIVE
    try:
        old_text = lock_path.read_text(encoding="utf-8")
        old = json.loads(old_text)
        new = compute_lock(arguments.rayd.resolve(), arguments.rev, old, eol=arguments.eol)
        new_text = render_lock(new)
        mapping = replacements(old, new)
    except (RelockError, OSError, KeyError, json.JSONDecodeError) as exc:
        print(f"relock_rayd: {exc}", file=sys.stderr)
        return 2

    changed: list[str] = []
    if new_text != old_text:
        changed.append(LOCK_RELATIVE)
        for label, before, after in (
            ("commit", old["commit"], new["commit"]),
            ("integration_abi.sha256", old["integration_abi"]["sha256"], new["integration_abi"]["sha256"]),
            ("integration_abi.api_version", old["integration_abi"]["api_version"], new["integration_abi"]["api_version"]),
            ("source_bundle.distribution_version", old["source_bundle"]["distribution_version"], new["source_bundle"]["distribution_version"]),
            ("source_bundle.manifest_sha256", old["source_bundle"]["manifest_sha256"], new["source_bundle"]["manifest_sha256"]),
        ):
            if before != after:
                print(f"  {label}: {before} -> {after}")
        if not arguments.check:
            lock_path.write_text(new_text, encoding="utf-8", newline="\n")

    pattern = re.compile("|".join(re.escape(value) for value in mapping)) if mapping else None
    for relative in _lockstep_files(root):
        path = root / relative
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if pattern is None or pattern.search(text) is None:
            continue
        updated, count = pattern.subn(lambda match: mapping[match.group(0)], text)
        changed.append(relative)
        print(f"{relative}: {count} pinned value(s) replaced")
        if not arguments.check:
            path.write_bytes(updated.encode("utf-8"))

    for label, before, after in (
        ("distribution_version", old["source_bundle"]["distribution_version"], new["source_bundle"]["distribution_version"]),
        ("api_version", str(old["integration_abi"]["api_version"]), str(new["integration_abi"]["api_version"])),
    ):
        if before != after:
            quoted = [
                relative
                for relative in _lockstep_files(root)
                if relative.endswith((".py", ".yml", ".md", ".txt", ".cmake"))
                and before in (root / relative).read_text(encoding="utf-8", errors="replace")
            ]
            print(f"review: {label} changed {before} -> {after}; files quoting {before!r}: {quoted}")

    if not changed:
        print(f"{LOCK_RELATIVE} already locks RayD {new['commit']}; nothing changed")
        return 0
    verb = "would change" if arguments.check else "changed"
    print(f"relock_rayd: {verb} {len(changed)} file(s) for RayD {new['commit']}")
    return 1 if arguments.check else 0


if __name__ == "__main__":
    sys.exit(main())

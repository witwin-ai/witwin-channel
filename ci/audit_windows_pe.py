from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any


_ALLOWED_DEPENDENCIES = frozenset(
    {
        "advapi32.dll",
        "api-ms-win-crt-environment-l1-1-0.dll",
        "api-ms-win-crt-heap-l1-1-0.dll",
        "api-ms-win-crt-math-l1-1-0.dll",
        "api-ms-win-crt-runtime-l1-1-0.dll",
        "api-ms-win-crt-stdio-l1-1-0.dll",
        "api-ms-win-crt-string-l1-1-0.dll",
        "c10.dll",
        "c10_cuda.dll",
        "cfgmgr32.dll",
        "cudart64_12.dll",
        "cusolver64_11.dll",
        "kernel32.dll",
        "msvcp140.dll",
        "nvcuda.dll",
        "python311.dll",
        "torch_cpu.dll",
        "torch_cuda.dll",
        "torch_python.dll",
        "vcruntime140.dll",
        "vcruntime140_1.dll",
    }
)
_REQUIRED_DEPENDENCIES = frozenset(
    {
        "c10.dll",
        "c10_cuda.dll",
        "cudart64_12.dll",
        "nvcuda.dll",
        "python311.dll",
        "torch_cuda.dll",
        "torch_python.dll",
    }
)
_DEPENDENCY_PATTERN = re.compile(r"^[A-Za-z0-9_.+-]+\.dll$", re.IGNORECASE)
_EXPORT_PATTERN = re.compile(
    r"^\s*(?P<ordinal>\d+)\s+(?P<hint>[0-9A-Fa-f]+)\s+"
    r"(?P<rva>[0-9A-Fa-f]+)\s+(?P<name>\S+)\s*$"
)
_EXPORT_COUNT_PATTERN = re.compile(r"^\s*(\d+) number of names\s*$")
_PYTHON_INIT_EXPORT = "PyInit__channel_native"


class PEAuditError(ValueError):
    """The Windows extension does not satisfy its packaged PE contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _section(lines: list[str], start: str) -> list[str]:
    starts = [index for index, line in enumerate(lines) if line.strip() == start]
    if len(starts) != 1:
        raise PEAuditError(f"dumpbin output must contain exactly one {start!r} section")
    begin = starts[0] + 1
    summaries = [
        index for index in range(begin, len(lines)) if lines[index].strip() == "Summary"
    ]
    if len(summaries) != 1:
        raise PEAuditError("dumpbin output must contain one Summary after the section")
    return lines[begin : summaries[0]]


def parse_dependents(output: str) -> list[str]:
    section = _section(output.splitlines(), "Image has the following dependencies:")
    dependencies: list[str] = []
    for line in section:
        value = line.strip()
        if not value:
            continue
        if _DEPENDENCY_PATTERN.fullmatch(value) is None:
            raise PEAuditError(f"unparsed dumpbin dependency line: {line!r}")
        dependencies.append(value)
    if not dependencies or len(dependencies) != len(
        {item.lower() for item in dependencies}
    ):
        raise PEAuditError("dumpbin dependency list must be non-empty and unique")

    normalized = {item.lower() for item in dependencies}
    unexpected = sorted(normalized - _ALLOWED_DEPENDENCIES)
    missing = sorted(_REQUIRED_DEPENDENCIES - normalized)
    if unexpected:
        raise PEAuditError(
            f"PE has dependencies outside the release allowlist: {unexpected}"
        )
    if missing:
        raise PEAuditError(f"PE is missing required runtime dependencies: {missing}")
    return dependencies


def _allowed_export(name: str) -> bool:
    if name == _PYTHON_INIT_EXPORT:
        return True
    owner_name = name.casefold()
    if any(
        forbidden in owner_name
        for forbidden in ("rayd", "raydn", "channel_native", "channelnative", "legacy")
    ):
        return False
    return name.startswith("?") and any(
        namespace in name for namespace in ("@c10@@", "@caffe2@@", "@torch@@")
    )


def parse_exports(output: str) -> list[str]:
    lines = output.splitlines()
    expected_counts = [
        int(match.group(1))
        for line in lines
        if (match := _EXPORT_COUNT_PATTERN.fullmatch(line)) is not None
    ]
    if len(expected_counts) != 1:
        raise PEAuditError("dumpbin exports must declare exactly one number of names")

    section = _section(lines, "ordinal hint RVA      name")
    exports: list[str] = []
    for line in section:
        if not line.strip():
            continue
        match = _EXPORT_PATTERN.fullmatch(line)
        if match is None:
            raise PEAuditError(f"unparsed dumpbin export line: {line!r}")
        exports.append(match.group("name"))
    if len(exports) != expected_counts[0]:
        raise PEAuditError(
            "dumpbin export row count does not match its declared number of names"
        )
    if len(exports) != len(set(exports)):
        raise PEAuditError("dumpbin export names must be unique")
    if exports.count(_PYTHON_INIT_EXPORT) != 1:
        raise PEAuditError(f"PE must export exactly one {_PYTHON_INIT_EXPORT}")
    unexpected = [name for name in exports if not _allowed_export(name)]
    if unexpected:
        raise PEAuditError(
            f"PE has exports outside the release allowlist: {unexpected}"
        )
    return exports


def _dumpbin(path: Path, option: str, *, dumpbin: str) -> str:
    try:
        result = subprocess.run(
            [dumpbin, option, str(path)],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PEAuditError(f"failed to execute dumpbin {option}: {exc}") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise PEAuditError(
            f"dumpbin {option} failed with exit code {result.returncode}: {detail}"
        )
    if not result.stdout.strip():
        raise PEAuditError(f"dumpbin {option} produced no output")
    return result.stdout


def audit_pe(path: Path, *, dumpbin: str = "dumpbin") -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file() or path.suffix.lower() not in {".pyd", ".dll"}:
        raise PEAuditError(
            f"Windows PE does not exist or has an invalid suffix: {path}"
        )
    dependencies = parse_dependents(_dumpbin(path, "/DEPENDENTS", dumpbin=dumpbin))
    exports = parse_exports(_dumpbin(path, "/EXPORTS", dumpbin=dumpbin))
    exports_digest = hashlib.sha256("\n".join(exports).encode("utf-8")).hexdigest()
    return {
        "schema_version": 1,
        "path": str(path),
        "sha256": _sha256(path),
        "dependencies": dependencies,
        "export_count": len(exports),
        "exports_sha256": exports_digest,
        "python_init_export": _PYTHON_INIT_EXPORT,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit a packaged Windows Python extension with dumpbin."
    )
    parser.add_argument("pe", type=Path)
    parser.add_argument("--dumpbin", default="dumpbin")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        evidence = audit_pe(args.pe, dumpbin=args.dumpbin)
    except (OSError, PEAuditError) as exc:
        parser.error(str(exc))
    encoded = json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

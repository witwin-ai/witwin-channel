from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from ci import audit_windows_pe


_DEPENDENTS = """
Dump of file _channel.cp311-win_amd64.pyd

  Image has the following dependencies:

    c10.dll
    c10_cuda.dll
    torch_python.dll
    python311.dll
    torch_cuda.dll
    cudart64_12.dll
    nvcuda.dll

  Summary
"""
_EXPORTS = """
Dump of file _channel.cp311-win_amd64.pyd

           2 number of names

    ordinal hint RVA      name

          1    0 00015360 ?type@Future@ivalue@c10@@QEBA_NXZ
          2    1 00013580 PyInit__channel

  Summary
"""


def test_dumpbin_parsers_accept_only_declared_dependencies_and_exports():
    assert audit_windows_pe.parse_dependents(_DEPENDENTS) == [
        "c10.dll",
        "c10_cuda.dll",
        "torch_python.dll",
        "python311.dll",
        "torch_cuda.dll",
        "cudart64_12.dll",
        "nvcuda.dll",
    ]
    assert audit_windows_pe.parse_exports(_EXPORTS) == [
        "?type@Future@ivalue@c10@@QEBA_NXZ",
        "PyInit__channel",
    ]


@pytest.mark.parametrize(
    ("output", "match"),
    [
        (
            _DEPENDENTS.replace("nvcuda.dll", "rayd.dll"),
            "outside the release allowlist",
        ),
        (_DEPENDENTS.replace("    nvcuda.dll\n", ""), "missing required"),
        (_DEPENDENTS.replace("    nvcuda.dll", "    not a dependency"), "unparsed"),
    ],
)
def test_dependency_parser_fails_loudly(output: str, match: str):
    with pytest.raises(audit_windows_pe.PEAuditError, match=match):
        audit_windows_pe.parse_dependents(output)


@pytest.mark.parametrize(
    ("output", "match"),
    [
        (_EXPORTS.replace("PyInit__channel", "rayd_legacy"), "exactly one"),
        (_EXPORTS.replace("2 number of names", "3 number of names"), "row count"),
        (
            _EXPORTS.replace("?type@Future@ivalue@c10@@QEBA_NXZ", "channel_rf"),
            "allowlist",
        ),
        (
            _EXPORTS.replace(
                "?type@Future@ivalue@c10@@QEBA_NXZ",
                "?rayd_legacy_dispatch@Future@ivalue@c10@@QEBA_NXZ",
            ),
            "allowlist",
        ),
        (_EXPORTS.replace("00015360", "not-rva"), "unparsed"),
    ],
)
def test_export_parser_fails_loudly(output: str, match: str):
    with pytest.raises(audit_windows_pe.PEAuditError, match=match):
        audit_windows_pe.parse_exports(output)


def test_audit_pe_invokes_both_dumpbin_modes_and_records_identity(
    tmp_path: Path, monkeypatch
):
    pe = tmp_path / "_channel.pyd"
    pe.write_bytes(b"MZ-test")
    calls: list[str] = []

    def fake_run(command, **kwargs):
        calls.append(command[1])
        output = _DEPENDENTS if command[1] == "/DEPENDENTS" else _EXPORTS
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    monkeypatch.setattr(audit_windows_pe.subprocess, "run", fake_run)

    evidence = audit_windows_pe.audit_pe(pe)

    assert calls == ["/DEPENDENTS", "/EXPORTS"]
    assert evidence["schema_version"] == 1
    assert evidence["sha256"] == __import__("hashlib").sha256(b"MZ-test").hexdigest()
    assert evidence["export_count"] == 2
    assert evidence["python_init_export"] == "PyInit__channel"


def test_audit_pe_rejects_dumpbin_failure(tmp_path: Path, monkeypatch):
    pe = tmp_path / "_channel.pyd"
    pe.write_bytes(b"MZ-test")
    monkeypatch.setattr(
        audit_windows_pe.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1, stdout="", stderr="corrupt image"
        ),
    )

    with pytest.raises(audit_windows_pe.PEAuditError, match="corrupt image"):
        audit_windows_pe.audit_pe(pe)

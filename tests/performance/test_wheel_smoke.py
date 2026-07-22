from __future__ import annotations

import base64
import csv
from email.message import Message
import hashlib
import io
import json
from pathlib import Path
import zipfile

import pytest

from ci import wheel_smoke


def _write_wheel(path: Path, *, name: str, version: str) -> bytes:
    metadata = Message()
    metadata["Metadata-Version"] = "2.1"
    metadata["Name"] = name
    metadata["Version"] = version
    dist_info = f"witwin_channel-{version}.dist-info"
    repository_root = Path(wheel_smoke.__file__).resolve().parents[1]
    members = {
        member: (repository_root / "src" / member).read_bytes()
        for member in wheel_smoke._checked_in_package_members()
    }
    members.update(
        {
            f"{dist_info}/METADATA": metadata.as_bytes(),
            f"{dist_info}/WHEEL": b"Wheel-Version: 1.0\nGenerator: tests\nRoot-Is-Purelib: false\nTag: cp311-cp311-win_amd64\n",
            "witwin/channel/_channel.cp311-win_amd64.pyd": b"native",
            "witwin/channel/runtime/_channel.build-fingerprint": b"f" * 64
            + b"\n",
            "witwin/channel/runtime/rayd.lock.json": (
                repository_root / "dependencies" / "rayd.lock.json"
            ).read_bytes(),
        }
    )
    record = io.StringIO(newline="")
    writer = csv.writer(record, lineterminator="\n")
    for member, payload in sorted(members.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=")
        writer.writerow((member, f"sha256={digest.decode('ascii')}", len(payload)))
    writer.writerow((f"{dist_info}/RECORD", "", ""))
    members[f"{dist_info}/RECORD"] = record.getvalue().encode("utf-8")
    with zipfile.ZipFile(path, "w") as archive:
        for member, payload in members.items():
            archive.writestr(member, payload)
    return path.read_bytes()


def _replace_member(path: Path, member: str, payload: bytes) -> None:
    with zipfile.ZipFile(path) as archive:
        members = {
            info.filename: archive.read(info.filename)
            for info in archive.infolist()
            if info.filename != member
        }
    members[member] = payload
    with zipfile.ZipFile(path, "w") as archive:
        for name, value in members.items():
            archive.writestr(name, value)


def _refresh_record(path: Path) -> None:
    record_member = "witwin_channel-0.1.0.dist-info/RECORD"
    with zipfile.ZipFile(path) as archive:
        members = {
            info.filename: archive.read(info.filename)
            for info in archive.infolist()
            if info.filename != record_member
        }
    record = io.StringIO(newline="")
    writer = csv.writer(record, lineterminator="\n")
    for member, payload in sorted(members.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=")
        writer.writerow((member, f"sha256={digest.decode('ascii')}", len(payload)))
    writer.writerow((record_member, "", ""))
    members[record_member] = record.getvalue().encode("utf-8")
    with zipfile.ZipFile(path, "w") as archive:
        for member, payload in members.items():
            archive.writestr(member, payload)


def test_wheel_identity_and_sha256_are_read_from_wheel(tmp_path: Path):
    wheel = tmp_path / "witwin_channel-0.1.0-py3-none-any.whl"
    payload = _write_wheel(wheel, name="witwin-channel", version="0.1.0")

    assert wheel_smoke._wheel_identity(wheel) == (
        "witwin-channel",
        "0.1.0",
    )
    assert (
        wheel_smoke._sha256(wheel) == __import__("hashlib").sha256(payload).hexdigest()
    )


def test_wheel_identity_rejects_unexpected_distribution(tmp_path: Path):
    wheel = tmp_path / "other-0.1.0-py3-none-any.whl"
    _write_wheel(wheel, name="other", version="0.1.0")

    with pytest.raises(ValueError, match="wheel identity"):
        wheel_smoke._wheel_identity(wheel)


def test_wheel_identity_rejects_dist_info_version_mismatch(tmp_path: Path):
    wheel = tmp_path / "witwin_channel-0.1.0-py3-none-any.whl"
    _write_wheel(wheel, name="witwin-channel", version="0.1.0")
    metadata_member = "witwin_channel-0.1.0.dist-info/METADATA"
    with zipfile.ZipFile(wheel) as archive:
        metadata = archive.read(metadata_member).replace(
            b"Version: 0.1.0", b"Version: 0.2.0"
        )
    _replace_member(wheel, metadata_member, metadata)

    with pytest.raises(ValueError, match="does not exactly match"):
        wheel_smoke._wheel_identity(wheel)


def test_wheel_content_audit_accepts_owned_runtime_files(tmp_path: Path):
    wheel = tmp_path / "witwin_channel-0.1.0-py3-none-any.whl"
    _write_wheel(wheel, name="witwin-channel", version="0.1.0")

    assert wheel_smoke._audit_wheel_contents(wheel).endswith("win_amd64.pyd")


def test_wheel_content_audit_rejects_missing_checked_in_init(tmp_path: Path):
    wheel = tmp_path / "witwin_channel-0.1.0-py3-none-any.whl"
    _write_wheel(wheel, name="witwin-channel", version="0.1.0")
    member = "witwin/channel/__init__.py"
    with zipfile.ZipFile(wheel) as archive:
        members = {
            info.filename: archive.read(info.filename)
            for info in archive.infolist()
            if info.filename != member
        }
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)

    with pytest.raises(ValueError, match="source closure mismatch"):
        wheel_smoke._audit_wheel_contents(wheel)


def test_wheel_content_audit_rejects_tampered_allowlisted_source(tmp_path: Path):
    wheel = tmp_path / "witwin_channel-0.1.0-py3-none-any.whl"
    _write_wheel(wheel, name="witwin-channel", version="0.1.0")
    _replace_member(wheel, "witwin/channel/__init__.py", b"tampered")

    with pytest.raises(ValueError, match="source bytes differ"):
        wheel_smoke._audit_wheel_contents(wheel)


def test_wheel_content_audit_does_not_treat_https_url_as_drive_path(tmp_path: Path):
    assert (
        wheel_smoke._ABSOLUTE_PATH_PATTERN.search(
            b'{"repository_url":"https://github.com/Asixa/RayD.git"}'
        )
        is None
    )


@pytest.mark.parametrize(
    "member",
    [
        "../escape.py",
        "/absolute.py",
        "C:/drive.py",
        "witwin//channel/empty.py",
        "witwin/./channel/dot.py",
        "witwin/channel/../traversal.py",
    ],
)
def test_wheel_content_audit_rejects_noncanonical_members(tmp_path: Path, member: str):
    wheel = tmp_path / "witwin_channel-0.1.0-py3-none-any.whl"
    _write_wheel(wheel, name="witwin-channel", version="0.1.0")
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr(member, b"injected")

    with pytest.raises(ValueError, match="canonical|segment"):
        wheel_smoke._audit_wheel_contents(wheel)


def test_canonical_member_rejects_backslash():
    with pytest.raises(ValueError, match="canonical"):
        wheel_smoke._canonical_member("witwin\\channel\\injected.py")


@pytest.mark.parametrize("member", ["witwin/cafe\u0301.py", "witwin/control\x00.py"])
def test_canonical_member_rejects_noncanonical_unicode_or_control(member: str):
    with pytest.raises(ValueError, match="canonical"):
        wheel_smoke._canonical_member(member)


def test_wheel_content_audit_rejects_casefold_duplicate(tmp_path: Path):
    wheel = tmp_path / "witwin_channel-0.1.0-py3-none-any.whl"
    _write_wheel(wheel, name="witwin-channel", version="0.1.0")
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr(
            "WITWIN/channel/runtime/rayd.lock.json", b"casefold duplicate"
        )

    with pytest.raises(ValueError, match="duplicate normalized/casefold"):
        wheel_smoke._audit_wheel_contents(wheel)


@pytest.mark.parametrize(
    "member",
    [
        "bootstrap.pth",
        "witwin_channel-0.1.0.data/scripts/run.py",
        "other_root/file.py",
        "witwin/channel/untracked_injection.py",
        "witwin_channel-0.1.0.dist-info/entry_points.txt",
    ],
)
def test_wheel_content_audit_rejects_nonallowlisted_members(
    tmp_path: Path, member: str
):
    wheel = tmp_path / "witwin_channel-0.1.0-py3-none-any.whl"
    _write_wheel(wheel, name="witwin-channel", version="0.1.0")
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr(member, b"injected")

    with pytest.raises(ValueError, match="allowlist|source closure"):
        wheel_smoke._audit_wheel_contents(wheel)


def test_wheel_content_audit_rejects_record_coverage_and_digest_mismatch(
    tmp_path: Path,
):
    wheel = tmp_path / "witwin_channel-0.1.0-py3-none-any.whl"
    _write_wheel(wheel, name="witwin-channel", version="0.1.0")
    record = "witwin_channel-0.1.0.dist-info/RECORD"
    with zipfile.ZipFile(wheel) as archive:
        original = archive.read(record).decode("utf-8")

    _replace_member(
        wheel,
        record,
        original.replace("sha256=", "sha256=corrupt", 1).encode("utf-8"),
    )
    with pytest.raises(ValueError, match="hash/size mismatch"):
        wheel_smoke._audit_wheel_contents(wheel)

    _write_wheel(wheel, name="witwin-channel", version="0.1.0")
    with zipfile.ZipFile(wheel) as archive:
        rows = archive.read(record).decode("utf-8").splitlines()
    _replace_member(wheel, record, ("\n".join(rows[:-1]) + "\n").encode("utf-8"))
    with pytest.raises(ValueError, match="coverage mismatch"):
        wheel_smoke._audit_wheel_contents(wheel)


@pytest.mark.parametrize(
    "member",
    [
        "witwin/channel/vendor/helper.dll",
        "other/package/extension.pyd",
        "lib/unrelated.so",
        "lib/unrelated.dylib",
    ],
)
def test_wheel_content_audit_rejects_any_extra_dso(tmp_path: Path, member: str):
    wheel = tmp_path / "witwin_channel-0.1.0-py3-none-any.whl"
    _write_wheel(wheel, name="witwin-channel", version="0.1.0")
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr(member, b"extra shared library")

    with pytest.raises(ValueError, match="no DSO except"):
        wheel_smoke._audit_wheel_contents(wheel)


@pytest.mark.parametrize(
    ("member", "payload", "match"),
    [
        ("rayd/torch/_stable_ops.dll", b"leaked dependency", "no DSO except"),
        (
            "witwin/channel/CMakeFiles/kernel.obj",
            b"build output",
            "source closure",
        ),
        (
            "witwin/channel/config.txt",
            b"E:\\private\\checkout",
            "source closure",
        ),
    ],
)
def test_wheel_content_audit_rejects_build_leaks(
    tmp_path: Path, member: str, payload: bytes, match: str
):
    wheel = tmp_path / "witwin_channel-0.1.0-py3-none-any.whl"
    _write_wheel(wheel, name="witwin-channel", version="0.1.0")
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr(member, payload)

    with pytest.raises(ValueError, match=match):
        wheel_smoke._audit_wheel_contents(wheel)


def test_smoke_program_contains_strict_isolation_and_native_checks(tmp_path: Path):
    code = wheel_smoke._smoke_code(
        target=tmp_path,
        wheel=tmp_path / "package.whl",
        wheel_sha256="a" * 64,
        expected_name="witwin-channel",
        expected_version="0.1.0",
    )

    compile(code, "<wheel-smoke>", "exec")
    assert "is_relative_to(target)" in code
    assert 'finder.__class__.__module__ != "_witwin_channel_editable"' in code
    assert 'find_spec("witwin.channel._channel")' in code
    assert 'build_info.get("uses_dr_jit") is not False' in code
    assert 'build_info.get("uses_rayd_native") is not True' in code


def test_wheel_pe_audit_extracts_only_owned_extension(tmp_path: Path, monkeypatch):
    wheel = tmp_path / "witwin_channel-0.1.0-py3-none-any.whl"
    _write_wheel(wheel, name="witwin-channel", version="0.1.0")
    member = wheel_smoke._audit_wheel_contents(wheel)

    def fake_audit(path: Path, *, dumpbin: str):
        assert path.name == "_channel.cp311-win_amd64.pyd"
        assert path.read_bytes() == b"native"
        assert dumpbin == "custom-dumpbin"
        return {"schema_version": 1, "path": str(path), "dependencies": []}

    monkeypatch.setattr(wheel_smoke.audit_windows_pe, "audit_pe", fake_audit)

    evidence = wheel_smoke._audit_wheel_pe(wheel, member, dumpbin="custom-dumpbin")

    assert "path" not in evidence
    assert evidence["wheel_member"] == member


def _build_info() -> dict[str, object]:
    repository_root = Path(wheel_smoke.__file__).resolve().parents[1]
    lock = json.loads(
        (repository_root / "dependencies" / "rayd.lock.json").read_text(
            encoding="utf-8"
        )
    )
    integration = lock["integration_abi"]
    return {
        "backend": "channel",
        "build_fingerprint": "f" * 64,
        "build_type": "Release",
        "channel_abi_version": 1,
        "channel_git_dirty": False,
        "channel_git_sha": "c" * 40,
        "compiler": "MSVC",
        "cuda_architectures": ["120-real", "120-virtual"],
        "cuda_available": True,
        "cuda_compiler_version": "12.9",
        "cuda_version": "12.8",
        "cxx_abi": "msvc",
        "material_abi_version": 3,
        "optix_available": True,
        "rayd_commit": lock["commit"],
        "rayd_dirty": False,
        "rayd_integration": "source-linked",
        "rayd_integration_abi_kind": integration["kind"],
        "rayd_integration_abi_path": integration["path"],
        "rayd_integration_abi_sha256": integration["sha256"],
        "rayd_repository_url": lock["repository_url"],
        "rayd_source_kind": "git-checkout",
        "rayd_source_manifest_sha256": lock["source_bundle"]["manifest_sha256"],
        "torch_version": "2.10.0",
        "uses_dr_jit": False,
        "uses_path_native": True,
        "uses_rayd_native": True,
    }


def _smoke_evidence(target: Path) -> dict[str, object]:
    package = target / "witwin" / "channel"
    package.mkdir(parents=True)
    package_origin = package / "__init__.py"
    native_origin = package / "_channel.cp311-win_amd64.pyd"
    package_origin.write_text("", encoding="utf-8")
    native_origin.write_bytes(b"native")
    return {
        "build_info": _build_info(),
        "distribution": {
            "name": "witwin-channel",
            "root": str(target),
            "version": "0.1.0",
        },
        "native_origin": str(native_origin),
        "package_origin": str(package_origin),
        "wheel_sha256": "a" * 64,
        "wheel_smoke": True,
    }


def _expected_build_identity(build_info: dict[str, object]) -> dict[str, object]:
    return {
        "build_fingerprint": "f" * 64,
        "build_type": "Release",
        "channel_abi_version": 1,
        "channel_git_dirty": False,
        "rayd_commit": build_info["rayd_commit"],
        "rayd_dirty": False,
        "rayd_integration_abi_kind": build_info["rayd_integration_abi_kind"],
        "rayd_integration_abi_path": build_info["rayd_integration_abi_path"],
        "rayd_integration_abi_sha256": build_info["rayd_integration_abi_sha256"],
        "rayd_repository_url": build_info["rayd_repository_url"],
        "rayd_source_manifest_sha256": build_info[
            "rayd_source_manifest_sha256"
        ],
    }


def _parse_evidence(payload: str, target: Path) -> dict[str, object]:
    return wheel_smoke._parse_smoke_evidence(
        payload,
        expected_wheel_sha256="a" * 64,
        expected_name="witwin-channel",
        expected_version="0.1.0",
        target=target,
        native_member="witwin/channel/_channel.cp311-win_amd64.pyd",
        expected_build_identity=_expected_build_identity(_build_info()),
    )


def test_smoke_evidence_parser_requires_exact_independently_verified_schema(
    tmp_path: Path,
):
    evidence = _smoke_evidence(tmp_path)
    assert _parse_evidence(__import__("json").dumps(evidence), tmp_path) == evidence
    with pytest.raises(ValueError, match="one JSON object"):
        _parse_evidence("log\n{}", tmp_path)
    with pytest.raises(ValueError, match="schema mismatch"):
        _parse_evidence('{"wheel_smoke": true}', tmp_path)


@pytest.mark.parametrize(
    "payload",
    [
        '{"wheel_smoke":true,"wheel_smoke":true}',
        '{"wheel_smoke":NaN}',
        '{"wheel_smoke":Infinity}',
    ],
)
def test_smoke_evidence_parser_rejects_duplicate_keys_and_nonfinite_json(
    tmp_path: Path, payload: str
):
    with pytest.raises(ValueError, match="one JSON object"):
        _parse_evidence(payload, tmp_path)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda data: data.update(wheel_sha256="b" * 64), "disagrees with parent"),
        (
            lambda data: data["distribution"].update(version="0.2.0"),
            "distribution identity",
        ),
        (lambda data: data["build_info"].update(uses_rayd_native=False), "identity"),
        (
            lambda data: data["build_info"].update(build_fingerprint="e" * 64),
            "disagrees with wheel identity",
        ),
        (lambda data: data["build_info"].update(rayd_commit="missing"), "RayD Git"),
        (lambda data: data.update(extra=True), "schema mismatch"),
    ],
)
def test_smoke_evidence_parser_rejects_identity_or_schema_mismatch(
    tmp_path: Path, mutation, match: str
):
    evidence = _smoke_evidence(tmp_path)
    mutation(evidence)
    with pytest.raises(ValueError, match=match):
        _parse_evidence(__import__("json").dumps(evidence), tmp_path)


def test_smoke_evidence_parser_rejects_origins_outside_target(tmp_path: Path):
    evidence = _smoke_evidence(tmp_path / "target")
    outside = tmp_path / "outside.py"
    outside.write_text("", encoding="utf-8")
    evidence["package_origin"] = str(outside)

    with pytest.raises(ValueError, match="outside target"):
        _parse_evidence(__import__("json").dumps(evidence), tmp_path / "target")


def test_smoke_evidence_parser_rejects_elsewhere_origins_inside_target(tmp_path: Path):
    target = tmp_path / "target"
    evidence = _smoke_evidence(target)
    elsewhere = target / "elsewhere" / "__init__.py"
    elsewhere.parent.mkdir()
    elsewhere.write_text("", encoding="utf-8")
    evidence["package_origin"] = str(elsewhere)

    with pytest.raises(ValueError, match="package origin is invalid"):
        _parse_evidence(json.dumps(evidence), target)


def test_smoke_evidence_parser_requires_distribution_root_exactly_target(
    tmp_path: Path,
):
    target = tmp_path / "target"
    evidence = _smoke_evidence(target)
    nested = target / "nested"
    nested.mkdir()
    evidence["distribution"]["root"] = str(nested)

    with pytest.raises(ValueError, match="distribution root"):
        _parse_evidence(json.dumps(evidence), target)


@pytest.mark.parametrize(
    "payload",
    [
        b'{"schema_version":1,"schema_version":1}',
        b'{"schema_version":1,"repository_url":NaN}',
    ],
)
def test_wheel_runtime_identity_rejects_non_strict_lock_json(
    tmp_path: Path, payload: bytes
):
    wheel = tmp_path / "witwin_channel-0.1.0-py3-none-any.whl"
    _write_wheel(wheel, name="witwin-channel", version="0.1.0")
    _replace_member(wheel, "witwin/channel/runtime/rayd.lock.json", payload)

    with pytest.raises(ValueError, match="strict UTF-8 JSON"):
        wheel_smoke._wheel_runtime_identity(wheel)


def test_wheel_runtime_identity_rejects_malformed_fingerprint(tmp_path: Path):
    wheel = tmp_path / "witwin_channel-0.1.0-py3-none-any.whl"
    _write_wheel(wheel, name="witwin-channel", version="0.1.0")
    _replace_member(
        wheel,
        "witwin/channel/runtime/_channel.build-fingerprint",
        b"not-a-sha256\n",
    )

    with pytest.raises(ValueError, match="exactly one SHA-256"):
        wheel_smoke._wheel_runtime_identity(wheel)


def test_wheel_rejects_alternate_valid_rayd_identity_with_synced_record_and_build_info(
    tmp_path: Path,
):
    wheel = tmp_path / "witwin_channel-0.1.0-py3-none-any.whl"
    _write_wheel(wheel, name="witwin-channel", version="0.1.0")
    repository_root = Path(wheel_smoke.__file__).resolve().parents[1]
    alternate = json.loads(
        (repository_root / "dependencies" / "rayd.lock.json").read_text(
            encoding="utf-8"
        )
    )
    alternate["commit"] = "e" * 40
    alternate["integration_abi"]["sha256"] = "b" * 64
    _replace_member(
        wheel,
        "witwin/channel/runtime/rayd.lock.json",
        (json.dumps(alternate, indent=2) + "\n").encode("utf-8"),
    )
    _refresh_record(wheel)

    synced_build_info = _build_info()
    synced_build_info["rayd_commit"] = alternate["commit"]
    synced_build_info["rayd_integration_abi_sha256"] = alternate["integration_abi"][
        "sha256"
    ]
    assert synced_build_info["rayd_commit"] == "e" * 40
    assert synced_build_info["rayd_integration_abi_sha256"] == "b" * 64
    target = tmp_path / "target"
    synced_evidence = _smoke_evidence(target)
    synced_evidence["build_info"] = synced_build_info
    wheel_smoke._parse_smoke_evidence(
        json.dumps(synced_evidence),
        expected_wheel_sha256="a" * 64,
        expected_name="witwin-channel",
        expected_version="0.1.0",
        target=target,
        native_member="witwin/channel/_channel.cp311-win_amd64.pyd",
        expected_build_identity=_expected_build_identity(synced_build_info),
    )
    with pytest.raises(ValueError, match="does not exactly match authoritative"):
        wheel_smoke._audit_wheel_contents(wheel)


def test_resolve_wheel_accepts_one_file_or_one_wheel_directory(tmp_path: Path):
    wheel = tmp_path / "package.whl"
    wheel.write_bytes(b"wheel")

    assert wheel_smoke._resolve_wheel(wheel) == wheel.resolve()
    assert wheel_smoke._resolve_wheel(tmp_path) == wheel.resolve()


def test_resolve_wheel_rejects_empty_or_ambiguous_directory(tmp_path: Path):
    with pytest.raises(ValueError, match="exactly one"):
        wheel_smoke._resolve_wheel(tmp_path)
    (tmp_path / "one.whl").write_bytes(b"one")
    (tmp_path / "two.whl").write_bytes(b"two")
    with pytest.raises(ValueError, match="found 2"):
        wheel_smoke._resolve_wheel(tmp_path)

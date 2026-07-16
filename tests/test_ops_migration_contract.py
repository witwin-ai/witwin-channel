from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import pytest

from ci import check_ops_migration as migration


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPOSITORY_ROOT / "ci" / "ops_migration_manifest.json"


def _write_package(repo: Path, files: dict[str, str]) -> None:
    package_root = repo / "src" / "witwin" / "channel_native"
    for relative_path, source in files.items():
        path = package_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")


def _initial_manifest(repo: Path) -> dict[str, object]:
    prefix = f"{migration.OPS_MODULE}."
    definitions = [
        item
        for item in migration.scan_definitions(repo)
        if item.qualified_name.startswith(prefix)
    ]
    contracts = [
        {
            "id": item.terminal_name,
            "kind": item.kind,
            "signature": item.signature,
            "body_sha256": item.body_sha256,
            "normalized_ast_sha256": item.normalized_ast_sha256,
        }
        for item in definitions
    ]
    return {
        "schema_version": 1,
        "source_module": migration.OPS_MODULE,
        "approved_owner_roots": list(migration.APPROVED_OWNER_ROOTS),
        "contracts": contracts,
        "active_ops": [str(item["id"]) for item in contracts],
        "canonical_owners": {},
        "retired_ops": [],
        "approved_body_projections": [],
    }


def _synthetic_manifest(tmp_path: Path) -> dict[str, object]:
    _write_package(
        tmp_path,
        {
            "core/kernels/ops.py": """
def stay(value):
    return value * 2

def move(value=1):
    return value + 1

class Worker:
    @staticmethod
    def work(value):
        return value - 1
""",
        },
    )
    return _initial_manifest(tmp_path)


def _register_move(manifest: dict[str, object]) -> None:
    active = manifest["active_ops"]
    owners = manifest["canonical_owners"]
    assert isinstance(active, list)
    assert isinstance(owners, dict)
    active.remove("move")
    owners["move"] = "witwin.channel_native.scene.kernels.move"


def _retire_move(manifest: dict[str, object]) -> None:
    owners = manifest["canonical_owners"]
    retired = manifest["retired_ops"]
    assert isinstance(owners, dict)
    assert isinstance(retired, list)
    owners.pop("move")
    retired.append("move")


def _register_move_projection(manifest: dict[str, object]) -> None:
    projections = manifest["approved_body_projections"]
    assert isinstance(projections, list)
    projections.append(
        {
            "id": "move",
            "owner": "witwin.channel_native.scene.kernels.move",
            "native_symbol": "move",
            "kind": "remove_trailing_raydn_module_handle",
        }
    )


def _projected_synthetic_manifest(tmp_path: Path) -> dict[str, object]:
    _write_package(
        tmp_path,
        {
            "core/kernels/ops.py": """
def move(value=1):
    return _required_native_op("move")(value, _raydn_module_handle())
""",
        },
    )
    manifest = _initial_manifest(tmp_path)
    _register_move(manifest)
    _register_move_projection(manifest)
    return manifest


def _check_synthetic(tmp_path: Path, manifest: dict[str, object]) -> list[str]:
    return migration.check_manifest(
        tmp_path,
        manifest,
        expected_digest=migration.contract_digest(manifest),
    )


def test_current_manifest_covers_every_movable_ops_body():
    manifest = migration.load_manifest(MANIFEST_PATH)

    assert migration.check_manifest(REPOSITORY_ROOT, manifest) == []
    assert len(manifest["contracts"]) == 282
    assert len(manifest["active_ops"]) == 0
    assert len(manifest["canonical_owners"]) == 281
    assert manifest["retired_ops"] == ["_raydn_module_handle"]
    assert len(manifest["approved_body_projections"]) == 25
    assert migration.contract_digest(manifest) == migration.FROZEN_CONTRACT_DIGEST
    assert migration.BOOTSTRAP_CANONICAL_OWNERS.items() <= (
        manifest["canonical_owners"].items()
    )
    assert {
        owner
        for entry_id, owner in manifest["canonical_owners"].items()
        if entry_id.startswith("_ad_")
    } == {
        f"witwin.channel_native.runtime.autograd_contracts.{entry_id}"
        for entry_id in manifest["canonical_owners"]
        if entry_id.startswith("_ad_")
    }
    contract_ids = {entry["id"] for entry in manifest["contracts"]}
    assert "_RaydnIntersectAdFunction.forward" in contract_ids
    assert "_FieldCoupledRdAdFunction.backward.material_column" not in contract_ids
    assert "_FieldCoupledRdAdFunction.jvp.material_pack" not in contract_ids


def test_ops_facade_is_a_bounded_pure_reexport_module():
    path = REPOSITORY_ROOT / "src/witwin/channel_native/core/kernels/ops.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    assert len(source.splitlines()) <= 300
    assert all(isinstance(node, ast.ImportFrom) for node in tree.body)


def test_pure_move_with_compatibility_reexport_has_one_body(tmp_path: Path):
    manifest = _synthetic_manifest(tmp_path)
    _register_move(manifest)
    _write_package(
        tmp_path,
        {
            "core/kernels/ops.py": """
from witwin.channel_native.scene.kernels import move

def stay(value):
    return value * 2

class Worker:
    @staticmethod
    def work(value):
        return value - 1
""",
            "scene/kernels.py": """
def move(value=1):
    return value + 1
""",
        },
    )

    assert _check_synthetic(tmp_path, manifest) == []


def test_copied_body_is_rejected_even_when_new_owner_is_registered(tmp_path: Path):
    manifest = _synthetic_manifest(tmp_path)
    _register_move(manifest)
    _write_package(
        tmp_path,
        {
            "scene/kernels.py": """
def move(value=1):
    return value + 1
""",
        },
    )

    issues = _check_synthetic(tmp_path, manifest)

    assert any("migrated body remains in ops" in issue for issue in issues)
    assert any("duplicate frozen body for move" in issue for issue in issues)


@pytest.mark.parametrize(
    ("owner_source", "expected_issue"),
    [
        ("def move(value=1):\n    return value + 2\n", "body changed for move"),
        ("def move(value=2):\n    return value + 1\n", "signature changed for move"),
        (
            "def marker(function):\n    return function\n\n"
            "@marker\ndef move(value=1):\n    return value + 1\n",
            "normalized AST changed for move",
        ),
    ],
)
def test_body_normalized_ast_and_signature_are_independent_gates(
    tmp_path: Path, owner_source: str, expected_issue: str
):
    manifest = _synthetic_manifest(tmp_path)
    _register_move(manifest)
    _write_package(
        tmp_path,
        {
            "core/kernels/ops.py": """
from witwin.channel_native.scene.kernels import move

def stay(value):
    return value * 2

class Worker:
    @staticmethod
    def work(value):
        return value - 1
""",
            "scene/kernels.py": owner_source,
        },
    )

    assert any(
        expected_issue in issue for issue in _check_synthetic(tmp_path, manifest)
    )


def test_missing_body_and_unregistered_owner_are_rejected(tmp_path: Path):
    manifest = _synthetic_manifest(tmp_path)
    _register_move(manifest)
    _write_package(
        tmp_path,
        {
            "core/kernels/ops.py": """
from witwin.channel_native.scene.kernels import move

def stay(value):
    return value * 2

class Worker:
    @staticmethod
    def work(value):
        return value - 1
""",
            "scene/kernels.py": "",
        },
    )

    issues = _check_synthetic(tmp_path, manifest)

    assert any("body lost from canonical owner for move" in issue for issue in issues)


def test_retired_contract_requires_the_definition_to_be_absent(tmp_path: Path):
    manifest = _synthetic_manifest(tmp_path)
    _register_move(manifest)
    _retire_move(manifest)
    _write_package(
        tmp_path,
        {
            "core/kernels/ops.py": """
def stay(value):
    return value * 2

class Worker:
    @staticmethod
    def work(value):
        return value - 1
""",
            "scene/kernels.py": "",
        },
    )

    assert _check_synthetic(tmp_path, manifest) == []

    _write_package(
        tmp_path,
        {"scene/kernels.py": "def move(value=1):\n    return value + 1\n"},
    )
    issues = _check_synthetic(tmp_path, manifest)
    assert any("retired definition remains for move" in issue for issue in issues)


@pytest.mark.parametrize("ledger", ["active_ops", "canonical_owners"])
def test_retired_contract_cannot_remain_in_another_ledger(tmp_path: Path, ledger: str):
    manifest = _synthetic_manifest(tmp_path)
    retired = manifest["retired_ops"]
    assert isinstance(retired, list)
    retired.append("move")

    issues = _check_synthetic(tmp_path, manifest)

    expected = (
        "active and retired" if ledger == "active_ops" else "migrated and retired"
    )
    if ledger == "canonical_owners":
        active = manifest["active_ops"]
        owners = manifest["canonical_owners"]
        assert isinstance(active, list)
        assert isinstance(owners, dict)
        active.remove("move")
        owners["move"] = "witwin.channel_native.scene.kernels.move"
        issues = _check_synthetic(tmp_path, manifest)
    assert any(expected in issue for issue in issues)


def test_retired_ledger_cannot_name_an_unknown_contract(tmp_path: Path):
    manifest = _synthetic_manifest(tmp_path)
    retired = manifest["retired_ops"]
    assert isinstance(retired, list)
    retired.append("unknown")

    issues = _check_synthetic(tmp_path, manifest)

    assert any("unknown migration IDs: unknown" in issue for issue in issues)


def test_approved_body_projection_restores_only_the_removed_trailing_handle(
    tmp_path: Path,
):
    manifest = _projected_synthetic_manifest(tmp_path)
    _write_package(
        tmp_path,
        {
            "core/kernels/ops.py": "",
            "scene/kernels.py": """
def move(value=1):
    return _required_native_op("move")(value)
""",
        },
    )

    assert _check_synthetic(tmp_path, manifest) == []


@pytest.mark.parametrize(
    "source",
    [
        """
def move(value=1):
    return _required_native_op("move")(_raydn_module_handle())
""",
        """
def move(value=1):
    return _required_native_op("move")(value + 1)
""",
        """
def move(value=1):
    marker = 1
    return _required_native_op("move")(value)
""",
        """
def move(value=1):
    return _required_native_op("other")(value)
""",
    ],
    ids=(
        "wrong_parameter",
        "non_trailing_change",
        "extra_body",
        "other_native_call",
    ),
)
def test_approved_body_projection_rejects_other_body_changes(
    tmp_path: Path, source: str
):
    manifest = _projected_synthetic_manifest(tmp_path)
    _write_package(
        tmp_path,
        {"core/kernels/ops.py": "", "scene/kernels.py": source},
    )

    issues = _check_synthetic(tmp_path, manifest)

    assert any(
        "approved trailing RayDN handle projection mismatch" in issue
        for issue in issues
    )


def test_approved_body_projection_rejects_unknown_and_duplicate_ids(tmp_path: Path):
    manifest = _synthetic_manifest(tmp_path)
    projections = manifest["approved_body_projections"]
    assert isinstance(projections, list)
    entry = {
        "id": "unknown",
        "owner": "witwin.channel_native.scene.kernels.unknown",
        "native_symbol": "unknown",
        "kind": "remove_trailing_raydn_module_handle",
    }
    projections.extend([entry, copy.deepcopy(entry)])

    issues = _check_synthetic(tmp_path, manifest)

    assert any(
        "unknown approved body projection ID: unknown" in issue for issue in issues
    )
    assert any(
        "duplicate approved body projection ID: unknown" in issue for issue in issues
    )


def test_frozen_universe_and_ledger_cannot_grow():
    manifest = migration.load_manifest(MANIFEST_PATH)
    changed_contract = copy.deepcopy(manifest)
    changed_contract["contracts"][0]["signature"] += " "
    changed_ledger = copy.deepcopy(manifest)
    changed_ledger["active_ops"].append("new_debt")

    contract_issues = migration.check_manifest(REPOSITORY_ROOT, changed_contract)
    ledger_issues = migration.check_manifest(REPOSITORY_ROOT, changed_ledger)

    assert any(
        "frozen ops contract universe changed" in issue for issue in contract_issues
    )
    assert any("unknown migration IDs: new_debt" in issue for issue in ledger_issues)


def test_cli_passes_with_repository_defaults(capsys: pytest.CaptureFixture[str]):
    assert migration.main(["--repository-root", str(REPOSITORY_ROOT)]) == 0
    assert "ops migration contract passed" in capsys.readouterr().out


def test_manifest_is_stable_canonical_json():
    manifest = migration.load_manifest(MANIFEST_PATH)
    rendered = json.dumps(manifest, indent=2) + "\n"

    assert MANIFEST_PATH.read_text(encoding="utf-8") == rendered

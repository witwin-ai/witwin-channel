"""Enforce body-preserving migration out of ``core.kernels.ops``.

The frozen contracts are produced with ``tools.refactor_baseline.python_body_hashes``.
Only top-level functions and direct class methods participate; compatibility
re-exports therefore do not look like a second implementation.
"""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.refactor_baseline import python_body_hashes  # noqa: E402


PACKAGE = "witwin.channel_native"
OPS_MODULE = f"{PACKAGE}.core.kernels.ops"
DEFAULT_MANIFEST_PATH = Path("ci/ops_migration_manifest.json")
APPROVED_OWNER_ROOTS = (
    f"{PACKAGE}.runtime",
    f"{PACKAGE}.scene",
    f"{PACKAGE}.materials",
    f"{PACKAGE}.propagation",
    f"{PACKAGE}.scattering",
    f"{PACKAGE}.path",
    f"{PACKAGE}.deterministic",
    f"{PACKAGE}.montecarlo",
)
BOOTSTRAP_CANONICAL_OWNERS = {
    "_raydn_scene_handle_id": (
        f"{PACKAGE}.runtime.native_handles._raydn_scene_handle_id"
    ),
    "raydn_scene_create": f"{PACKAGE}.scene.kernels.rayd_scene.raydn_scene_create",
    "raydn_scene_edge_records": (
        f"{PACKAGE}.scene.kernels.rayd_scene.raydn_scene_edge_records"
    ),
}

# The digest covers immutable contracts and approved destinations, but excludes
# the migration ledgers. A migration may move an ID from active_ops to
# canonical_owners; Phase 10 may then move a proven-dead canonical ID to
# retired_ops without rewriting the frozen implementation contract.
FROZEN_CONTRACT_DIGEST = (
    "ff9c4cd45b2f1091c9ba05e1a311e6e569945e18badc7b7a67a3f8f56ccda3a9"
)


@dataclass(frozen=True, order=True, slots=True)
class Definition:
    terminal_name: str
    qualified_name: str
    path: str
    line: int
    kind: str
    signature: str
    body_sha256: str
    normalized_ast_sha256: str
    projected_native_symbol: str | None
    projected_body_sha256: str | None
    projected_normalized_ast_sha256: str | None


class _RestoreRaydnModuleHandle(ast.NodeTransformer):
    def __init__(self, native_symbol: str) -> None:
        self.native_symbol = native_symbol
        self.restored = 0

    def visit_Call(self, node: ast.Call) -> ast.AST:  # noqa: N802
        self.generic_visit(node)
        factory = node.func
        if not (
            isinstance(factory, ast.Call)
            and isinstance(factory.func, ast.Name)
            and factory.func.id == "_required_native_op"
            and len(factory.args) == 1
            and isinstance(factory.args[0], ast.Constant)
            and factory.args[0].value == self.native_symbol
        ):
            return node
        if node.args and isinstance(node.args[-1], ast.Call):
            trailing = node.args[-1]
            if (
                isinstance(trailing.func, ast.Name)
                and trailing.func.id == "_raydn_module_handle"
                and not trailing.args
                and not trailing.keywords
            ):
                return node
        node.args.append(
            ast.Call(
                func=ast.Name(id="_raydn_module_handle", ctx=ast.Load()),
                args=[],
                keywords=[],
            )
        )
        self.restored += 1
        return node


def _project_removed_raydn_module_handle(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str | None, str | None, str | None]:
    symbols = {
        str(factory.args[0].value)
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance((factory := call.func), ast.Call)
        and isinstance(factory.func, ast.Name)
        and factory.func.id == "_required_native_op"
        and len(factory.args) == 1
        and isinstance(factory.args[0], ast.Constant)
        and isinstance(factory.args[0].value, str)
    }
    projected: list[tuple[str, str, str]] = []
    for native_symbol in sorted(symbols):
        candidate = copy.deepcopy(node)
        transformer = _RestoreRaydnModuleHandle(native_symbol)
        candidate = transformer.visit(candidate)
        if transformer.restored != 1:
            continue
        ast.fix_missing_locations(candidate)
        body_ast = ast.dump(
            ast.Module(body=candidate.body, type_ignores=[]),
            annotate_fields=True,
            include_attributes=False,
        )
        normalized_ast = ast.dump(
            candidate,
            annotate_fields=True,
            include_attributes=False,
        )
        projected.append(
            (
                native_symbol,
                hashlib.sha256(body_ast.encode()).hexdigest(),
                hashlib.sha256(normalized_ast.encode()).hexdigest(),
            )
        )
    if len(projected) != 1:
        return None, None, None
    return projected[0]


def _module_name(repo: Path, path: Path) -> str:
    parts = list(path.relative_to(repo / "src").with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _direct_functions(
    tree: ast.Module,
) -> list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    functions: list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append((node.name, node))
        elif isinstance(node, ast.ClassDef):
            functions.extend(
                (f"{node.name}.{child.name}", child)
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            )
    return functions


def scan_definitions(repo: Path) -> list[Definition]:
    """Return movable definitions using the Phase 0 hash implementation."""
    hash_entries = {
        (
            str(entry["path"]),
            int(entry["line"]),
            str(entry["qualified_name"]),
        ): entry
        for entry in python_body_hashes(repo)
    }
    package_root = repo / "src" / "witwin" / "channel_native"
    definitions: list[Definition] = []
    for path in sorted(package_root.rglob("*.py")):
        relative_path = path.relative_to(repo).as_posix()
        module = _module_name(repo, path)
        source = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(source, filename=str(path))
        for terminal_name, node in _direct_functions(tree):
            qualified_name = f"{module}.{terminal_name}"
            entry = hash_entries[(relative_path, node.lineno, qualified_name)]
            projected_symbol, projected_body, projected_ast = (
                _project_removed_raydn_module_handle(node)
            )
            definitions.append(
                Definition(
                    terminal_name=terminal_name,
                    qualified_name=qualified_name,
                    path=relative_path,
                    line=node.lineno,
                    kind=str(entry["kind"]),
                    signature=f"({ast.unparse(node.args)})",
                    body_sha256=str(entry["body_sha256"]),
                    normalized_ast_sha256=str(entry["normalized_ast_sha256"]),
                    projected_native_symbol=projected_symbol,
                    projected_body_sha256=projected_body,
                    projected_normalized_ast_sha256=projected_ast,
                )
            )
    return sorted(definitions)


def build_manifest(repo: Path) -> dict[str, object]:
    """Build the initial ledger; intended for review, never silent updating."""
    prefix = f"{OPS_MODULE}."
    current_definitions = scan_definitions(repo)
    definitions_by_name = {item.qualified_name: item for item in current_definitions}
    definitions = [
        item for item in current_definitions if item.qualified_name.startswith(prefix)
    ]
    definitions.extend(
        definitions_by_name[owner] for owner in BOOTSTRAP_CANONICAL_OWNERS.values()
    )
    definitions.sort(key=lambda item: item.terminal_name)
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
        "source_module": OPS_MODULE,
        "approved_owner_roots": list(APPROVED_OWNER_ROOTS),
        "contracts": contracts,
        "active_ops": [
            str(item["id"])
            for item in contracts
            if item["id"] not in BOOTSTRAP_CANONICAL_OWNERS
        ],
        "canonical_owners": BOOTSTRAP_CANONICAL_OWNERS,
        "retired_ops": [],
        "approved_body_projections": [],
    }


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()


def contract_digest(manifest: dict[str, Any]) -> str:
    frozen = {
        key: manifest[key]
        for key in (
            "schema_version",
            "source_module",
            "approved_owner_roots",
            "contracts",
        )
    }
    return hashlib.sha256(_canonical_bytes(frozen)).hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("ops migration manifest must be an object")
    return value


def _matches_root(module: str, root: str) -> bool:
    return module == root or module.startswith(f"{root}.")


def _contract_index(
    manifest: dict[str, Any],
) -> tuple[dict[str, dict[str, str]], list[str]]:
    entries = manifest.get("contracts")
    if not isinstance(entries, list) or not all(
        isinstance(entry, dict) for entry in entries
    ):
        return {}, ["contracts must be a list of objects"]
    issues: list[str] = []
    contracts: dict[str, dict[str, str]] = {}
    required = {
        "id",
        "kind",
        "signature",
        "body_sha256",
        "normalized_ast_sha256",
    }
    for entry in entries:
        if set(entry) != required or not all(
            isinstance(entry.get(key), str) for key in required
        ):
            issues.append("each contract must contain exactly the frozen string fields")
            continue
        entry_id = str(entry["id"])
        if entry_id in contracts:
            issues.append(f"duplicate contract ID: {entry_id}")
        contracts[entry_id] = {key: str(value) for key, value in entry.items()}
    return contracts, issues


def _ledger(
    manifest: dict[str, Any], contracts: dict[str, dict[str, str]]
) -> tuple[set[str], dict[str, str], set[str], list[str]]:
    active_value = manifest.get("active_ops")
    owners_value = manifest.get("canonical_owners")
    retired_value = manifest.get("retired_ops")
    if not isinstance(active_value, list) or not all(
        isinstance(item, str) for item in active_value
    ):
        return set(), {}, set(), ["active_ops must be a list of contract IDs"]
    if not isinstance(owners_value, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in owners_value.items()
    ):
        return set(), {}, set(), ["canonical_owners must map contract IDs to names"]
    if not isinstance(retired_value, list) or not all(
        isinstance(item, str) for item in retired_value
    ):
        return set(), {}, set(), ["retired_ops must be a list of contract IDs"]
    active = set(active_value)
    owners = {str(key): str(value) for key, value in owners_value.items()}
    retired = set(retired_value)
    contract_ids = set(contracts)
    issues: list[str] = []
    if len(active) != len(active_value):
        issues.append("active_ops contains duplicate IDs")
    if len(retired) != len(retired_value):
        issues.append("retired_ops contains duplicate IDs")
    unknown = (active | set(owners) | retired) - contract_ids
    if unknown:
        issues.append("unknown migration IDs: " + ", ".join(sorted(unknown)))
    overlaps = {
        "active and migrated": active & set(owners),
        "active and retired": active & retired,
        "migrated and retired": set(owners) & retired,
    }
    for label, overlap in overlaps.items():
        if overlap:
            issues.append(f"IDs cannot be both {label}: " + ", ".join(sorted(overlap)))
    missing = contract_ids - active - set(owners) - retired
    if missing:
        issues.append("contracts without an owner: " + ", ".join(sorted(missing)))
    return active, owners, retired, issues


def _body_projection_ledger(
    manifest: dict[str, Any],
    contracts: dict[str, dict[str, str]],
    owners: dict[str, str],
) -> tuple[dict[str, dict[str, str]], list[str]]:
    value = manifest.get("approved_body_projections")
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        return {}, ["approved_body_projections must be a list of objects"]
    projections: dict[str, dict[str, str]] = {}
    issues: list[str] = []
    required = {"id", "owner", "native_symbol", "kind"}
    for entry in value:
        if set(entry) != required or not all(
            isinstance(entry.get(field), str) for field in required
        ):
            issues.append(
                "approved body projection entries must contain string fields: "
                "id, owner, native_symbol, kind"
            )
            continue
        entry_id = str(entry["id"])
        if entry_id in projections:
            issues.append(f"duplicate approved body projection ID: {entry_id}")
            continue
        projection = {field: str(entry[field]) for field in required}
        projections[entry_id] = projection
        if entry_id not in contracts:
            issues.append(f"unknown approved body projection ID: {entry_id}")
        if projection["kind"] != "remove_trailing_raydn_module_handle":
            issues.append(f"unsupported body projection kind for {entry_id}")
        if owners.get(entry_id) != projection["owner"]:
            issues.append(
                f"approved body projection owner mismatch for {entry_id}: "
                f"{projection['owner']}"
            )
    return projections, issues


def check_manifest(
    repo: Path,
    manifest: dict[str, Any],
    *,
    expected_digest: str = FROZEN_CONTRACT_DIGEST,
) -> list[str]:
    """Validate the frozen universe, migration ledger, and current definitions."""
    issues: list[str] = []
    try:
        digest = contract_digest(manifest)
    except (KeyError, TypeError, ValueError) as exc:
        return [f"invalid frozen contract payload: {exc}"]
    if digest != expected_digest:
        issues.append(
            "frozen ops contract universe changed: "
            f"expected {expected_digest}, got {digest}"
        )

    if manifest.get("source_module") != OPS_MODULE:
        issues.append(f"source_module must remain {OPS_MODULE}")
    roots_value = manifest.get("approved_owner_roots")
    if roots_value != list(APPROVED_OWNER_ROOTS):
        issues.append("approved canonical owner roots changed")

    contracts, contract_issues = _contract_index(manifest)
    issues.extend(contract_issues)
    active, owners, retired, ledger_issues = _ledger(manifest, contracts)
    issues.extend(ledger_issues)
    body_projections, projection_issues = _body_projection_ledger(
        manifest, contracts, owners
    )
    issues.extend(projection_issues)
    if contract_issues:
        return issues

    for entry_id, owner in sorted(owners.items()):
        suffix = f".{entry_id}"
        if not owner.endswith(suffix):
            issues.append(
                f"canonical owner must preserve terminal name {entry_id}: {owner}"
            )
            continue
        module = owner[: -len(suffix)]
        if module == OPS_MODULE or not any(
            _matches_root(module, root) for root in APPROVED_OWNER_ROOTS
        ):
            issues.append(f"unapproved canonical owner for {entry_id}: {owner}")

    definitions = scan_definitions(repo)
    by_qualified: dict[str, list[Definition]] = defaultdict(list)
    by_terminal_body: dict[tuple[str, str], list[Definition]] = defaultdict(list)
    by_terminal: dict[str, list[Definition]] = defaultdict(list)
    for item in definitions:
        by_qualified[item.qualified_name].append(item)
        by_terminal_body[(item.terminal_name, item.body_sha256)].append(item)
        by_terminal[item.terminal_name].append(item)

    for entry_id in sorted(retired):
        remaining = by_terminal.get(entry_id, [])
        if remaining:
            locations = ", ".join(item.qualified_name for item in remaining)
            issues.append(f"retired definition remains for {entry_id}: {locations}")

    ops_prefix = f"{OPS_MODULE}."
    for item in definitions:
        if not item.qualified_name.startswith(ops_prefix):
            continue
        if item.terminal_name not in contracts:
            issues.append(
                f"new implementation in ops is forbidden: {item.qualified_name}"
            )
        elif item.terminal_name not in active:
            issues.append(f"migrated body remains in ops: {item.qualified_name}")

    for entry_id, contract in sorted(contracts.items()):
        expected_owner = (
            f"{OPS_MODULE}.{entry_id}" if entry_id in active else owners.get(entry_id)
        )
        if expected_owner is None:
            continue
        candidates = by_qualified.get(expected_owner, [])
        if not candidates:
            matching = by_terminal_body.get((entry_id, contract["body_sha256"]), [])
            locations = ", ".join(item.qualified_name for item in matching)
            detail = f"; frozen body found at {locations}" if locations else ""
            issues.append(
                f"body lost from canonical owner for {entry_id}: {expected_owner}{detail}"
            )
            continue
        if len(candidates) > 1:
            issues.append(
                f"duplicate canonical definition for {entry_id}: {expected_owner}"
            )
        item = candidates[0]
        projection = body_projections.get(entry_id)
        if projection is None:
            for field, label in (
                ("body_sha256", "body"),
                ("normalized_ast_sha256", "normalized AST"),
            ):
                if getattr(item, field) != contract[field]:
                    issues.append(f"{label} changed for {entry_id} at {expected_owner}")
        elif (
            item.projected_native_symbol != projection["native_symbol"]
            or item.projected_body_sha256 != contract["body_sha256"]
            or item.projected_normalized_ast_sha256 != contract["normalized_ast_sha256"]
        ):
            issues.append(
                f"approved trailing RayDN handle projection mismatch for {entry_id} "
                f"at {expected_owner}"
            )
        for field, label in (
            ("signature", "signature"),
            ("kind", "function kind"),
        ):
            if getattr(item, field) != contract[field]:
                issues.append(f"{label} changed for {entry_id} at {expected_owner}")
        body_matches = by_terminal_body[(entry_id, contract["body_sha256"])]
        if len(body_matches) > 1:
            locations = ", ".join(item.qualified_name for item in body_matches)
            issues.append(f"duplicate frozen body for {entry_id}: {locations}")
    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--print-initial",
        action="store_true",
        help="print a reviewed initial ledger; never updates the manifest",
    )
    args = parser.parse_args(argv)
    repository_root = args.repository_root.resolve()
    if args.print_initial:
        print(json.dumps(build_manifest(repository_root), indent=2) + "\n")
        return 0
    manifest_path = (args.manifest or repository_root / DEFAULT_MANIFEST_PATH).resolve()
    try:
        manifest = load_manifest(manifest_path)
        issues = check_manifest(repository_root, manifest)
    except (KeyError, OSError, SyntaxError, TypeError, ValueError) as exc:
        print(f"ops migration configuration error: {exc}")
        return 2
    if issues:
        for issue in issues:
            print(issue)
        return 1
    print("ops migration contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

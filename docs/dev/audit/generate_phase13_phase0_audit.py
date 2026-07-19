"""Generate Plan 13 Phase 0 ownership and integration audit artifacts.

This is a read-only source scanner.  It intentionally does not import the
production package or load ``_channel_native``: Phase 0 must remain usable on a
clean host before CUDA configuration.  Run from the repository root with the
project-mandated environment::

    conda run -n witwin2 python docs/dev/audit/generate_phase13_phase0_audit.py
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


AUDIT_DIR = Path(__file__).resolve().parent
REPO = AUDIT_DIR.parents[2]
RAYD = REPO.parents[1] / "RayDi"
BASELINE_CHANNEL = "a741f8d2a0ff5ba353be60584f21ee7f910f03ad"
BASELINE_RAYD = "346416f8f35250cd50c7d320d877307d55a8fc9f"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path: Path, root: Path = REPO) -> str:
    return path.relative_to(root).as_posix()


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True
    ).strip()


def source_files(root: Path, prefixes: tuple[str, ...]) -> list[Path]:
    suffixes = {".py", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".json", ".cmake", ".txt", ".md"}
    excluded_dirs = {".git", ".pytest_cache", "__pycache__", "artifacts", "dist"}

    def is_source(candidate: Path) -> bool:
        directory_names = {part.lower() for part in candidate.relative_to(root).parts[:-1]}
        return (
            candidate.is_file()
            and candidate.suffix.lower() in suffixes
            and not (directory_names & excluded_dirs)
            and not any(name == "build" or name.startswith("build-") for name in directory_names)
        )

    found: list[Path] = []
    for prefix in prefixes:
        path = root / prefix
        if path.is_file():
            found.append(path)
        elif path.is_dir():
            found.extend(
                candidate
                for candidate in path.rglob("*")
                if is_source(candidate)
            )
    return sorted(set(found))


CHANNEL_PRODUCTION = source_files(REPO, ("src", "native", "CMakeLists.txt", "cmake", "ci"))
CHANNEL_SRC = source_files(REPO, ("src",))
CHANNEL_NATIVE = source_files(REPO, ("native",))
CHANNEL_TESTS = source_files(REPO, ("tests",))
RAYD_PRODUCTION = source_files(RAYD, ("backends/torch", "include", "src"))
_TEXT_CACHE: dict[Path, str] = {}


def source_text(path: Path) -> str:
    text = _TEXT_CACHE.get(path)
    if text is None:
        text = path.read_text(encoding="utf-8", errors="replace")
        _TEXT_CACHE[path] = text
    return text


def occurrences(token: str, files: list[Path], root: Path = REPO) -> list[dict[str, Any]]:
    pattern = re.compile(rf"\b{re.escape(token)}\b")
    hits: list[dict[str, Any]] = []
    for path in files:
        try:
            text = source_text(path)
        except OSError:
            continue
        lines = [index for index, line in enumerate(text.splitlines(), 1) if pattern.search(line)]
        if lines:
            hits.append({"path": rel(path, root), "lines": lines, "count": len(lines)})
    return hits


def coverage_maps() -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    coverage = read_json(REPO / "ci/contract-coverage-manifest.json")
    native = {row[0]: row for row in coverage["native_bindings"]}
    tests = coverage["contract_tests"]
    scenarios = coverage["e2e_scenarios"]
    return native, tests, scenarios


MANIFEST = read_json(REPO / "ci/native-binding-manifest.json")
PUBLIC_SNAPSHOT_TEXT = (REPO / "ci/public-api-snapshot.json").read_text(encoding="utf-8")
COVERAGE, CONTRACT_TESTS, E2E_SCENARIOS = coverage_maps()
SYMBOLS = {item["name"]: item for item in MANIFEST["symbols"]}


def identifier_index(
    tokens: set[str], files: list[Path], root: Path = REPO, *, require_call: bool = False
) -> dict[str, list[dict[str, Any]]]:
    by_token: dict[str, dict[str, list[int]]] = {token: {} for token in tokens}
    for path in files:
        path_text = source_text(path)
        path_label = rel(path, root)
        for line_number, line in enumerate(path_text.splitlines(), 1):
            present = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", line)) & tokens
            for token in present:
                if require_call and not re.search(rf"\b{re.escape(token)}\s*\(", line):
                    continue
                by_token[token].setdefault(path_label, []).append(line_number)
    return {
        token: [
            {"path": path, "lines": lines, "count": len(lines)}
            for path, lines in sorted(paths.items())
        ]
        for token, paths in by_token.items()
    }


_SYMBOL_NAMES = set(SYMBOLS)
_TARGET_NAMES = {item["target"] for item in MANIFEST["symbols"]}
_SRC_SYMBOL_INDEX = identifier_index(_SYMBOL_NAMES, CHANNEL_SRC)
_TEST_SYMBOL_INDEX = identifier_index(_SYMBOL_NAMES, CHANNEL_TESTS)
_NATIVE_TARGET_INDEX = identifier_index(_TARGET_NAMES, CHANNEL_NATIVE, require_call=True)


def python_call_indexes() -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
]:
    calls: dict[str, list[dict[str, Any]]] = {name: [] for name in _SYMBOL_NAMES}
    dispatches: dict[str, list[dict[str, Any]]] = {name: [] for name in _SYMBOL_NAMES}
    definitions: dict[str, list[dict[str, Any]]] = {name: [] for name in _SYMBOL_NAMES}
    parsed: list[tuple[Path, ast.AST]] = []
    aliases: dict[str, str] = {}
    for path in CHANNEL_SRC:
        if path.suffix != ".py":
            continue
        try:
            tree = ast.parse(source_text(path), filename=str(path))
        except SyntaxError:
            continue
        parsed.append((path, tree))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Name):
                continue
            if node.value.id not in _SYMBOL_NAMES:
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    aliases[target.id] = node.value.id
    for path, tree in parsed:
        path_label = rel(path)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in definitions:
                definitions[node.name].append({"path": path_label, "line": node.lineno})
            if not isinstance(node, ast.Call):
                continue
            called_name = None
            if isinstance(node.func, ast.Name):
                called_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                called_name = node.func.attr
            if called_name in calls:
                calls[called_name].append({"path": path_label, "line": node.lineno})
            elif called_name in aliases:
                calls[aliases[called_name]].append(
                    {"path": path_label, "line": node.lineno, "via_alias": called_name}
                )
            if called_name == "_required_native_op" and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str) and first.value in dispatches:
                    dispatches[first.value].append({"path": path_label, "line": node.lineno})
    return calls, dispatches, definitions


_PYTHON_CALL_INDEX, _NATIVE_DISPATCH_INDEX, _PYTHON_DEFINITION_INDEX = python_call_indexes()


def target_definitions(target: str) -> list[dict[str, Any]]:
    return _NATIVE_TARGET_INDEX.get(target, [])


RAYD_NUMERICAL_SYMBOLS = {
    "bdpt_diffraction_accumulation_forward",
    "bdpt_intersect_forward",
    "bdpt_reflection_accumulation_forward",
    "bdpt_visibility_forward",
    "raydn_diffraction_paths_order1_forward",
    "raydn_intersect_backward",
    "raydn_intersect_jvp",
    "raydn_reflection_epc_paths_backward",
    "raydn_reflection_epc_paths_forward",
    "raydn_reflection_epc_paths_jvp",
    "raydn_scene_create",
    "raydn_scene_edge_records",
    "raydn_scene_face_normals_backward",
    "raydn_scene_face_normals_jvp",
    "raydn_trace_reflections_backward",
    "raydn_trace_reflections_forward",
    "raydn_trace_reflections_forward_tape",
    "raydn_trace_reflections_jvp",
}
LAYERED_SYMBOLS = {
    "raydn_coupled_dd_geometry_forward",
    "raydn_coupled_rd_geometry_forward",
}
CHANNEL_MISPLACED_SYMBOLS = {
    "bdpt_diffraction_discover_edges",
    "bdpt_diffraction_discover_edges_counted",
}

TRANSMISSION = [
    "em_layer_stack_eval",
    "em_layer_stack_backward",
    "em_layer_stack_jvp",
    "field_transmission_sequence",
    "field_transmission_sequence_backward",
    "field_transmission_sequence_jvp",
]
PURE_WEDGE = [
    "field_diffraction_wedge",
    "field_diffraction_wedge_backward",
    "field_diffraction_wedge_jvp",
]
SCATTERING_MOVE = [
    "scattering_table_eval",
    "scattering_table_eval_backward",
    "scattering_table_eval_jvp",
    "scattering_table_sample",
    "scattering_table_pdf",
    "scattering_ensemble_eval",
    "scattering_ensemble_eval_backward",
    "scattering_ensemble_eval_jvp",
    "scattering_patch_integral_eval",
    "scattering_patch_integral_eval_backward",
    "scattering_patch_integral_eval_jvp",
    "scattering_chain_ensemble_eval",
    "scattering_chain_ensemble_eval_backward",
    "scattering_chain_ensemble_eval_jvp",
    "scattering_chain_realization_eval",
    "scattering_chain_realization_eval_backward",
    "scattering_chain_realization_eval_jvp",
]
SCATTERING_RETAIN = ["scattering_event_probabilities"]


def family_for(symbol: str) -> str:
    if symbol.startswith("em_layer_stack_"):
        return "resident CSR layer-stack"
    if symbol.startswith("field_transmission_sequence"):
        return "complete-row transmission field"
    if symbol.startswith("scattering_table_eval"):
        return "table evaluation AD"
    if symbol in {"scattering_table_sample", "scattering_table_pdf"}:
        return "table sampling"
    if symbol.startswith("scattering_ensemble_eval"):
        return "single-bounce ensemble"
    if symbol.startswith("scattering_patch_integral_eval"):
        return "phase-screen patch integral"
    if symbol.startswith("scattering_chain_ensemble_eval"):
        return "v2 chain ensemble"
    if symbol.startswith("scattering_chain_realization_eval"):
        return "v2 chain realization"
    return "unaffected"


def owner_for(symbol: str) -> tuple[str, str]:
    if symbol in RAYD_NUMERICAL_SYMBOLS:
        return "RayD", "RayD native implementation through legacy Channel bridge"
    if symbol in LAYERED_SYMBOLS:
        return "Channel operation / RayD primitives", "composed owner"
    if symbol in CHANNEL_MISPLACED_SYMBOLS:
        return "Channel Native", "Channel CUDA implementation registered through rayd bridge"
    return "Channel Native", "Channel native implementation"


def production_callers(symbol: str) -> list[dict[str, Any]]:
    return _PYTHON_CALL_INDEX.get(symbol, [])


def test_callers(symbol: str) -> list[dict[str, Any]]:
    return _TEST_SYMBOL_INDEX.get(symbol, [])


def public_status(symbol: str) -> str:
    return "public-snapshot-reference" if re.search(rf'"{re.escape(symbol)}"', PUBLIC_SNAPSHOT_TEXT) else "internal-native-ABI"


def current_owner_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in MANIFEST["symbols"]:
        symbol = item["name"]
        coverage = COVERAGE.get(symbol)
        callers = production_callers(symbol)
        dispatches = _NATIVE_DISPATCH_INDEX.get(symbol, [])
        e2e = coverage[4] if coverage else []
        numerical_owner, owner_kind = owner_for(symbol)
        records.append(
            {
                "symbol": symbol,
                "abi_owner": "Channel Native _channel_native",
                "binding_target": item["target"],
                "binding_registration": {"path": item["path"], "line": item["line"]},
                "target_source_evidence": target_definitions(item["target"]),
                "numerical_owner": numerical_owner,
                "owner_kind": owner_kind,
                "python_owner": coverage[1] if coverage else None,
                "python_definitions": _PYTHON_DEFINITION_INDEX.get(symbol, []),
                "production_callers": callers,
                "native_dispatch_evidence": dispatches,
                "contract_test": CONTRACT_TESTS.get(coverage[3]) if coverage else None,
                "e2e_callers": [E2E_SCENARIOS.get(name, name) for name in e2e],
                "surface": public_status(symbol),
                "liveness": (
                    "live-static-production-consumer"
                    if callers
                    else "declared-e2e-without-static-production-consumer"
                    if e2e
                    else "binding-wrapper-only; four-part audit required before deletion"
                    if dispatches
                    else "no-static-or-declared-e2e-caller; four-part audit required before deletion"
                ),
                "plan13_disposition": (
                    "move numerical owner to RayD as a complete family"
                    if symbol in TRANSMISSION + PURE_WEDGE + SCATTERING_MOVE
                    else "retain Channel numerical owner"
                    if symbol in SCATTERING_RETAIN
                    else "rename/replace legacy integration identity"
                    if symbol in RAYD_NUMERICAL_SYMBOLS | LAYERED_SYMBOLS | CHANNEL_MISPLACED_SYMBOLS
                    else "unaffected by Plan 13"
                ),
            }
        )
    return records


CURRENT_RECORDS = current_owner_records()


def write_json(name: str, payload: Any) -> None:
    (AUDIT_DIR / name).write_text(
        json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


BASELINE = {
    "channel_commit_required": BASELINE_CHANNEL,
    "channel_commit_scanned": git_head(REPO),
    "rayd_commit_required": BASELINE_RAYD,
    "rayd_commit_scanned": git_head(RAYD),
    "binding_count": len(MANIFEST["symbols"]),
    "binding_manifest_sha256": sha256(REPO / "ci/native-binding-manifest.json"),
    "immutable_owner_inventory_sha256": sha256(AUDIT_DIR / "phase9-native-owner-inventory.json"),
    "rayd_lock": read_json(REPO / "dependencies/rayd.lock.json"),
}


write_json(
    "phase13-current-native-owner-inventory.json",
    {
        "schema_version": 1,
        "plan": "Plan 13 Phase 0",
        "baseline": BASELINE,
        "immutability_note": "phase9-native-owner-inventory.json is an ADR-009 baseline and was not modified.",
        "owner_model": {
            "abi_owner": "the project exporting the _channel_native symbol",
            "numerical_owner": "the project owning the authoritative implementation",
            "layered_owner": "a Channel-composed operation may consume RayD device/trace primitives without transferring the complete operation",
        },
        "counts": {
            "bindings": len(CURRENT_RECORDS),
            "rayd_numerical": sum(row["numerical_owner"] == "RayD" for row in CURRENT_RECORDS),
            "layered": sum(" / " in row["numerical_owner"] for row in CURRENT_RECORDS),
            "channel_numerical": sum(row["numerical_owner"] == "Channel Native" for row in CURRENT_RECORDS),
        },
        "symbols": CURRENT_RECORDS,
    },
)


write_json(
    "phase13-migration-delta.json",
    {
        "schema_version": 1,
        "baseline": BASELINE,
        "current_phase": 0,
        "transfers_applied": [],
        "renames_applied": [],
        "deletions_applied": [],
        "binding_count_delta": 0,
        "implementation_owner_delta": {"Channel Native": 0, "RayD": 0},
        "next_update_rule": "Append one record after each accepted owner transfer; never rewrite the immutable Phase 9 inventory.",
    },
)


RAYDN_RENAMES = {
    "raydn_scene_create": "rayd_scene_create",
    "raydn_scene_edge_records": "rayd_scene_edge_records",
    "raydn_intersect_backward": "rayd_intersect_backward",
    "raydn_intersect_jvp": "rayd_intersect_jvp",
    "raydn_trace_reflections_forward": "rayd_trace_reflections_forward",
    "raydn_trace_reflections_forward_tape": "rayd_trace_reflections_forward_tape",
    "raydn_trace_reflections_backward": "rayd_trace_reflections_backward",
    "raydn_trace_reflections_jvp": "rayd_trace_reflections_jvp",
    "raydn_reflection_epc_paths_forward": "rayd_reflection_epc_paths_forward",
    "raydn_reflection_epc_paths_backward": "rayd_reflection_epc_paths_backward",
    "raydn_reflection_epc_paths_jvp": "rayd_reflection_epc_paths_jvp",
    "raydn_scene_face_normals_backward": "rayd_scene_face_normals_backward",
    "raydn_scene_face_normals_jvp": "rayd_scene_face_normals_jvp",
    "raydn_diffraction_paths_order1_forward": "rayd_diffraction_paths_order1_forward",
    "raydn_coupled_rd_geometry_forward": "rayd_coupled_rd_geometry_forward",
    "raydn_coupled_dd_geometry_forward": "rayd_coupled_dd_geometry_forward",
    "bdpt_intersect_forward": "rayd_intersect_forward",
    "bdpt_visibility_forward": "rayd_visibility_forward",
    "bdpt_diffraction_accumulation_forward": "rayd_diffraction_sample_tape_forward",
    "bdpt_diffraction_discover_edges": "mc_diffraction_discover_edges",
    "bdpt_diffraction_discover_edges_counted": "mc_diffraction_discover_edges_counted",
}
LEGACY_DELETE_CANDIDATES = [
    "bdpt_diffraction_connection_samples_from_tape",
    "bdpt_diffraction_point_connection_samples",
    "bdpt_diffraction_state_pack",
    "bdpt_diffraction_state_wi",
    "bdpt_diffraction_edge_geometry",
]


delta_actions: list[dict[str, Any]] = []
for old, new in RAYDN_RENAMES.items():
    if old in SYMBOLS:
        delta_actions.append(
            {
                "symbol": old,
                "action": "rename",
                "replacement": new,
                "count_delta": 0,
                "status": "planned",
                "reason": "retire RayDN/BDPT-misattributed identity without compatibility alias",
            }
        )
for symbol in TRANSMISSION + PURE_WEDGE + SCATTERING_MOVE:
    delta_actions.append(
        {
            "symbol": symbol,
            "action": "implementation-owner-transfer",
            "replacement": symbol,
            "count_delta": 0,
            "status": "conditional on accepted ADR and cross-repository switch",
            "reason": "move complete operation family while preserving Channel-facing ABI",
        }
    )
for symbol in LEGACY_DELETE_CANDIDATES:
    if symbol in SYMBOLS:
        delta_actions.append(
            {
                "symbol": symbol,
                "action": "four-part-reachability-audit",
                "replacement": None,
                "count_delta": None,
                "status": "unresolved; deletion is not authorized by static scan alone",
                "reason": "Plan 13 section 6.4 requires static, dynamic binding, public import and BDPT E2E evidence",
            }
        )
write_json(
    "phase13-symbol-delta-ledger.json",
    {
        "schema_version": 1,
        "baseline_binding_count": len(SYMBOLS),
        "applied_count_delta": 0,
        "projected_final_count": None,
        "projected_count_formula": "211 minus only those legacy candidates later proven dead by all four reachability checks",
        "actions": delta_actions,
    },
)


def legacy_term_inventory() -> list[dict[str, Any]]:
    pattern = re.compile(r"RayDN|raydn|uses_raydn_native")
    records: list[dict[str, Any]] = []
    for path in CHANNEL_PRODUCTION:
        text = source_text(path)
        lines = [index for index, line in enumerate(text.splitlines(), 1) if pattern.search(line)]
        if lines:
            records.append({"path": rel(path), "lines": lines, "count": len(lines)})
    return records


integration_header = RAYD / "backends/torch/include/rayd/torch/integration.h"
integration_text = integration_header.read_text(encoding="utf-8")
rayd_entries = []
for match in re.finditer(r'^extern "C"\s+[^\n]*?\b(rayd_torch_native_[A-Za-z0-9_]+)\s*\(', integration_text, re.MULTILINE):
    name = match.group(1)
    channel_evidence = occurrences(name, CHANNEL_NATIVE)
    rayd_entries.append(
        {
            "entry": name,
            "header_line": integration_text.count("\n", 0, match.start()) + 1,
            "implementation_evidence": occurrences(name, RAYD_PRODUCTION, RAYD),
            "channel_evidence": channel_evidence,
            "channel_consumer_status": (
                "referenced by Channel native integration"
                if channel_evidence
                else "no Channel reference at Phase 0; do not delete until all RayD consumers are audited"
            ),
        }
    )


common_text = (REPO / "native/channel_native/rayd/common.cpp").read_text(encoding="utf-8")
getter_mappings = []
getter_pattern = re.compile(
    r"(?P<type>\w+)\s+(?P<getter>raydn_\w+_fn)\(\)\s*\{\s*return\s+&(?P<entry>\w+);",
    re.MULTILINE,
)
for match in getter_pattern.finditer(common_text):
    getter_mappings.append(match.groupdict())


rayd_binding_symbols = [
    row
    for row in CURRENT_RECORDS
    if row["binding_registration"]["path"] == "native/channel_native/binding/rayd.cpp"
]
term_inventory = legacy_term_inventory()
write_json(
    "phase13-rayd-integration-inventory.json",
    {
        "schema_version": 1,
        "baseline": BASELINE,
        "single_extension_model": {
            "python_extension": "_channel_native",
            "linked_target": "rayd_torch_native_core",
            "rayd_python_module_built": False,
            "cmake_evidence": ["CMakeLists.txt:305", "CMakeLists.txt:306", "CMakeLists.txt:307", "CMakeLists.txt:573"],
        },
        "legacy_term_summary": {
            "files": len(term_inventory),
            "occurrences": sum(row["count"] for row in term_inventory),
            "scope": ["src", "native", "CMakeLists.txt", "cmake", "ci"],
            "files_detail": term_inventory,
        },
        "rayd_source_integration": {
            "header": rel(integration_header, RAYD),
            "header_sha256": sha256(integration_header),
            "extern_c_entry_count": len(rayd_entries),
            "extern_c_entries": rayd_entries,
            "typed_cpp_v2_detection": {
                "namespace_rayd_torch": bool(re.search(r"namespace\s+rayd::torch", integration_text)),
                "named_result_structs": re.findall(r"\bstruct\s+(\w+Result)\b", integration_text),
                "raii_scene_types": re.findall(r"\b(?:class|struct)\s+(Scene(?:Handle|Resource))\b", integration_text),
                "status": "absent at Phase 0 baseline",
            },
        },
        "channel_legacy_bridge": {
            "signature_copy": "native/channel_native/rayd/bridge.h",
            "getter_implementation": "native/channel_native/rayd/common.cpp",
            "getter_count": len(getter_mappings),
            "getter_mappings": getter_mappings,
            "raw_handle_evidence": occurrences("scene_handle", source_files(REPO, ("native/channel_native/rayd",))),
        },
        "channel_rayd_binding_symbols": rayd_binding_symbols,
        "public_internal_conclusion": "The _channel_native RayD/RayDN bindings are internal native ABI; public Python entry points remain package/solver facades unless explicitly present in the public API snapshot.",
    },
)


write_json(
    "phase13-live-dead-public-internal-inventory.json",
    {
        "schema_version": 1,
        "method": {
            "static_live": "Python AST call to the façade name under src/; definitions and __all__ strings do not count",
            "native_dispatch": "AST call _required_native_op(<literal symbol>) proves the façade reaches the binding, not that a solver consumes the façade",
            "declared_e2e": "ci/contract-coverage-manifest.json e2e_scenarios",
            "public": "exact symbol in ci/public-api-snapshot.json",
            "deletion_limit": "no-static result is only a candidate; dynamic binding and real E2E audit remain mandatory",
        },
        "counts": dict(Counter(row["liveness"] for row in CURRENT_RECORDS)),
        "symbols": [
            {
                "symbol": row["symbol"],
                "surface": row["surface"],
                "liveness": row["liveness"],
                "python_definitions": row["python_definitions"],
                "production_callers": row["production_callers"],
                "native_dispatch_evidence": row["native_dispatch_evidence"],
                "e2e_callers": row["e2e_callers"],
                "test_references": test_callers(row["symbol"]),
            }
            for row in CURRENT_RECORDS
        ],
    },
)


def selected_contract(symbol: str, target_owner: str, compile_contract: str, fusion: str, tape: str) -> dict[str, Any]:
    row = next(item for item in CURRENT_RECORDS if item["symbol"] == symbol)
    return {
        "symbol": symbol,
        "family": family_for(symbol),
        "current_numerical_owner": row["numerical_owner"],
        "target_numerical_owner": target_owner,
        "channel_abi_owner_after_move": "_channel_native binding and domain façade",
        "binding": row["binding_registration"],
        "implementation_evidence": row["target_source_evidence"],
        "python_callers": row["production_callers"],
        "contract_test": row["contract_test"],
        "e2e_callers": row["e2e_callers"],
        "compile_contract": compile_contract,
        "fusion_contract": fusion,
        "tape_contract": tape,
        "acceptance": ["exact/ULP parity", "primal/JVP/VJP lockstep", "launch and stream parity", "no fallback"],
    }


transmission_contracts = [
    selected_contract(
        symbol,
        "RayD after ADR-024",
        "precise math; must not inherit --use_fast_math or scattering --fmad=false",
        "complete family move; CSR layers stay resident and no per-layer cross-launch tensor is introduced",
        "backward/JVP recompute the current layer chain; no persistent tape; shared layer gradients retain atomic accumulation/order",
    )
    for symbol in TRANSMISSION
]
write_json(
    "phase13-transmission-contracts.json",
    {
        "schema_version": 1,
        "contract_count": len(transmission_contracts),
        "families": ["resident CSR layer-stack", "complete-row transmission field"],
        "contracts": transmission_contracts,
        "retained_channel_owners": [
            "material models, CSR encoding, cache and validation façade",
            "topology winner/eligibility/component-5 packing",
            "bdpt_transmitted_light_subpath_state primal/backward/JVP complete 19-field family",
            "BDPT event probability, MIS and RNG",
            "MC Basic incident-polarization power estimator semantics",
            "component accumulation, metadata and results",
        ],
        "deferred_adr027": ["batched straight-segment penetration trace", "MC wall-product/active-state native estimator if profiling proves hot"],
    },
)


diffraction_families = [
    {
        "family": "RayD order-1 path exporter / visibility",
        "symbols": ["raydn_diffraction_paths_order1_forward", "bdpt_diffraction_accumulation_forward"],
        "current_owner": "RayD",
        "target_owner": "RayD",
        "action": "typed direct API and semantic rename; discrete winner remains frozen",
        "compile_contract": "RayD OptiX --use_fast_math",
        "fusion_tape": "path discovery/export or MC sampling tape producer; not the Channel estimator",
    },
    {
        "family": "pure wedge fixed-winner field",
        "symbols": PURE_WEDGE,
        "current_owner": "Channel Native",
        "target_owner": "RayD after ADR-025",
        "action": "move primal/backward/JVP together",
        "compile_contract": "--use_fast_math restricted to field_wedge_ad_diffraction.cu equivalent",
        "fusion_tape": "preserve three launches, wedge_row_eval order, optional winner vertices and fixed-winner AD",
    },
    {
        "family": "MC Sionna fixed-tape estimator",
        "symbols": ["mc_sionna_diffraction_tape_accumulate", "mc_sionna_diffraction_tape_accumulate_backward", "mc_sionna_diffraction_tape_accumulate_jvp"],
        "current_owner": "Channel Native",
        "target_owner": "Channel Native",
        "action": "retain complete family",
        "compile_contract": "precise math",
        "fusion_tape": "proposal/Jacobian, slab, cell atomics and RNG remain one estimator contract",
    },
    {
        "family": "coupled RD field",
        "symbols": ["field_coupled_rd", "field_coupled_rd_backward", "field_coupled_rd_jvp"],
        "current_owner": "Channel Native",
        "target_owner": "Channel Native",
        "action": "retain complete family; include RayD shared device primitives",
        "compile_contract": "precise math",
        "fusion_tape": "reflection slab plus UTD row fusion; no UTD sub-launch",
    },
    {
        "family": "coupled DD field",
        "symbols": ["field_coupled_dd", "field_coupled_dd_backward", "field_coupled_dd_jvp"],
        "current_owner": "Channel Native",
        "target_owner": "Channel Native",
        "action": "retain complete family; include RayD shared device primitives",
        "compile_contract": "precise math",
        "fusion_tape": "two-wedge one-launch row fusion",
    },
    {
        "family": "coupled RD stationary geometry",
        "symbols": ["coupled_rd_prepare", "coupled_rd_prepare_backward", "coupled_rd_prepare_jvp"],
        "current_owner": "Channel Native",
        "target_owner": "Channel Native",
        "action": "retain complete family",
        "compile_contract": "precise math",
        "fusion_tape": "continuous stationary geometry family",
    },
    {
        "family": "composed RD/DD geometry ABI",
        "symbols": ["raydn_coupled_rd_geometry_forward", "raydn_coupled_dd_geometry_forward"],
        "current_owner": "Channel operation / RayD EPC and visibility primitives",
        "target_owner": "Channel operation / RayD typed primitives",
        "action": "rename and record layered ownership",
        "compile_contract": "no numerical or fusion change",
        "fusion_tape": "Channel prepare/finalize around RayD primitive calls",
    },
    {
        "family": "MC edge discovery",
        "symbols": ["bdpt_diffraction_discover_edges", "bdpt_diffraction_discover_edges_counted"],
        "current_owner": "Channel Native CUDA",
        "target_owner": "Channel Native CUDA",
        "action": "rename mc_diffraction_discover_edges* and move out of RayD bridge owner",
        "compile_contract": "no numerical change",
        "fusion_tape": "MC/Sionna sampling policy, not a RayD geometry primitive",
    },
    {
        "family": "solver packing, MIS and accumulation",
        "symbols": LEGACY_DELETE_CANDIDATES,
        "current_owner": "Channel Native",
        "target_owner": "Channel Native only when real caller exists",
        "action": "retain only after four-part reachability audit; otherwise governance-complete deletion",
        "compile_contract": "unchanged until decision",
        "fusion_tape": "solver policy and storage are not eligible for RayD runtime migration",
    },
]
for family in diffraction_families:
    family["binding_evidence"] = [
        {
            "symbol": symbol,
            "present": symbol in SYMBOLS,
            "binding": (
                {"path": SYMBOLS[symbol]["path"], "line": SYMBOLS[symbol]["line"]}
                if symbol in SYMBOLS
                else None
            ),
            "production_callers": production_callers(symbol),
        }
        for symbol in family["symbols"]
    ]
write_json(
    "phase13-diffraction-family-matrix.json",
    {"schema_version": 1, "family_count": len(diffraction_families), "families": diffraction_families},
)


scattering_contracts = [
    selected_contract(
        symbol,
        "RayD after ADR-026" if symbol in SCATTERING_MOVE else "Channel Native",
        "--fmad=false for migrated runtime TUs; event-probability policy remains on its existing Channel TU contract",
        (
            "complete Dmax=8 C1/scatter/C2 row-fused operation; no split or materialized intermediate"
            if symbol.startswith("scattering_chain_")
            else "preserve current launch geometry, numerical order and resident tensors"
        ),
        (
            "fixed topology/sample/visibility/PDF/MIS; chain geometry reverse mode remains fail-loud"
            if symbol.startswith("scattering_chain_")
            else "preserve current primal/AD companion and resource lifetime"
        ),
    )
    for symbol in SCATTERING_MOVE + SCATTERING_RETAIN
]
write_json(
    "phase13-scattering-bindings.json",
    {
        "schema_version": 1,
        "binding_count": len(scattering_contracts),
        "move_count": len(SCATTERING_MOVE),
        "retain_count": len(SCATTERING_RETAIN),
        "contracts": scattering_contracts,
        "retained_channel_owners": [
            "scattering_event_probabilities",
            "Kirchhoff table builder/cache/version/validation",
            "ScatteringTableRuntime and PhaseScreenRuntime resource lifecycles",
            "rough C_r composition",
            "chain discovery/join/row budget/C1-C2 packing",
            "coherent combine and deterministic accumulation",
            "BDPT continuation/NEE/MIS/event glue/AD companions",
            "MC Basic estimator and result/metadata assembly",
        ],
    },
)


SHARED_HEADERS = [
    "native/channel_native/em/complex.cuh",
    "native/channel_native/em/medium.cuh",
    "native/channel_native/em/fresnel.cuh",
    "native/channel_native/em/layer_stack.cuh",
    "native/channel_native/field_transport.cuh",
    "native/channel_native/field_transport_ad.cuh",
    "native/channel_native/kernels/field_transport_ad_common.cuh",
    "native/channel_native/kernels/scattering_table.cuh",
]


def resolve_local_include(source: Path, include: str) -> Path | None:
    candidates = [source.parent / include, REPO / "native/channel_native" / include]
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved.exists() and (REPO in resolved.parents or resolved == REPO):
            return resolved
    return None


include_edges: list[dict[str, str]] = []
direct_consumers: dict[str, list[str]] = {header: [] for header in SHARED_HEADERS}
for path in CHANNEL_NATIVE:
    text = source_text(path)
    for line_number, line in enumerate(text.splitlines(), 1):
        match = re.match(r'\s*#include\s+["<]([^">]+)[">]', line)
        if not match:
            continue
        include = match.group(1)
        resolved = resolve_local_include(path, include)
        target: str | None = rel(resolved) if resolved else None
        if include == "rayd/shared/utd/utd_math.h":
            target = "RayD:include/rayd/shared/utd/utd_math.h"
        if target in SHARED_HEADERS or target == "RayD:include/rayd/shared/utd/utd_math.h":
            source = rel(path)
            include_edges.append({"from": source, "to": target, "line": line_number})
            if target in direct_consumers:
                direct_consumers[target].append(source)


graph_nodes = [
    {
        "id": header,
        "current_source_owner": "Channel Native",
        "target_source_owner": (
            "RayD after ADR-024"
            if header != "native/channel_native/kernels/field_transport_ad_common.cuh"
            else "split: RayD numerical helpers / Channel Torch validation wrapper"
        ),
        "direct_consumers": sorted(set(direct_consumers[header])),
        "sha256": sha256(REPO / header),
    }
    for header in SHARED_HEADERS
]
graph_nodes.append(
    {
        "id": "RayD:include/rayd/shared/utd/utd_math.h",
        "current_source_owner": "RayD",
        "target_source_owner": "RayD",
        "direct_consumers": sorted({edge["from"] for edge in include_edges if edge["to"].startswith("RayD:")}),
        "sha256": None,
    }
)
write_json(
    "phase13-shared-rf-dependency-graph.json",
    {
        "schema_version": 1,
        "roots": [
            "native/channel_native/kernels/scattering_chain_ensemble.cu",
            "native/channel_native/kernels/scattering_chain_ensemble_ad.cu",
            "native/channel_native/kernels/scattering_chain_realization.cu",
            "native/channel_native/kernels/scattering_chain_realization_ad.cu",
            "native/channel_native/kernels/field_transport.cu",
            "native/channel_native/kernels/field_transport_transmission.cu",
        ],
        "nodes": graph_nodes,
        "edges": include_edges,
        "forbidden_future_edge": "RayD -> native/channel_native private header",
    },
)


def helper_records(header: str) -> list[dict[str, Any]]:
    path = REPO / header
    text = source_text(path)
    consumers = sorted(set(direct_consumers[header]))
    pattern = re.compile(
        r"(?m)^(?P<prefix>(?:template\s*<[^;{]+>\s*)?(?:(?:__host__|__device__|__forceinline__|inline|static)\s+)*)"
        r"(?P<return>[A-Za-z_:][A-Za-z0-9_:<>\s,.*&]*?)\s+"
        r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\("
    )
    records = []
    for match in pattern.finditer(text):
        name = match.group("name")
        lower = name.lower()
        if header.endswith("field_transport_ad_common.cuh"):
            target = "Channel Native wrapper; move referenced numerical helper only"
        elif header.endswith("field_transport.cuh") or header.endswith("field_transport_ad.cuh"):
            shared_tokens = ("slab", "fresnel", "medium", "stack", "jones", "frame", "complex", "dc_", "dual", "adj_")
            target = "RayD after ADR-024" if any(token in lower for token in shared_tokens) else "Channel Native pending helper-by-helper closure decision"
        else:
            target = "RayD after ADR-024" if not header.endswith("scattering_table.cuh") else "RayD after ADR-026"
        records.append(
            {
                "name": name,
                "source": header,
                "line": text.count("\n", 0, match.start()) + 1,
                "current_unique_source_owner": "Channel Native",
                "target_unique_source_owner": target,
                "compiler_attributes": match.group("prefix").split(),
                "mirror_role": (
                    "dual/adjoint"
                    if any(token in lower for token in ("dual", "adj", "grad", "jvp", "backward", "dc_", "dlc_", "dv3", "df3"))
                    else "primal"
                ),
                "direct_consumers": consumers,
                "acceptance": ["normalized-body comparison", "compile-attribute parity", "PTX/SASS or equivalent codegen evidence", "primal/dual lockstep"],
            }
        )
    return records


helpers = [record for header in SHARED_HEADERS for record in helper_records(header)]
write_json(
    "phase13-shared-rf-helper-ledger.json",
    {
        "schema_version": 1,
        "helper_count": len(helpers),
        "extraction": "source-level inline/device function declarations from the Phase 0 shared-header closure",
        "helpers": helpers,
        "decision_rule": "Move only generic RF numerical helpers; Channel tensor checks, pybind validation and solver schema remain Channel-owned.",
    },
)


summary = f"""# Plan 13 Phase 0 audit summary

This directory now contains a reproducible, source-scanned Phase 0 baseline for
Channel Native `{BASELINE_CHANNEL[:8]}` and locked RayD `{BASELINE_RAYD[:8]}`.
The immutable ADR-009 file `phase9-native-owner-inventory.json` was hashed and
read, but not modified.

## Frozen counts

- `_channel_native` bindings: **{len(SYMBOLS)}**.
- Current numerical ownership: **{sum(row['numerical_owner'] == 'RayD' for row in CURRENT_RECORDS)} RayD**, **{sum(row['numerical_owner'] == 'Channel Native' for row in CURRENT_RECORDS)} Channel**, and **{sum(' / ' in row['numerical_owner'] for row in CURRENT_RECORDS)} layered Channel-operation/RayD-primitive** records.
- Legacy `RayDN/raydn/uses_raydn_native` scan: **{len(term_inventory)} files**, **{sum(row['count'] for row in term_inventory)} matching lines** across production source/build/CI scope.
- RayD source integration header: **{len(rayd_entries)} extern-C entries**, with no detected typed C++ v2 namespace, named result structs, or RAII scene type at the frozen baseline.
- Channel legacy indirection: **{len(getter_mappings)} function-pointer getters** in `native/channel_native/rayd/common.cpp`; two point to Channel diffraction-discovery CUDA entries and the rest point to RayD extern-C entries.
- Candidate owner moves: **6 transmission**, **3 pure-wedge diffraction**, and **17 scattering runtime** bindings. `scattering_event_probabilities` remains Channel-owned.
- Shared RF helper scan: **{len(helpers)} inline/device helper declarations** across **{len(SHARED_HEADERS)}** headers.
- Runtime capture: **38 frozen cells** across all four solvers plus Path/
  Deterministic JVP/VJP, measured in two independent processes with one warmup
  and seven steady repeats. One deterministic rough-scattering-ensemble cell
  is explicitly excluded because its exact hash differed across processes;
  both fingerprints remain recorded instead of being hidden by a tolerance.

## Artifact map

- `phase13-current-native-owner-inventory.json`: all 211 symbols, separate ABI and numerical ownership, source definitions, callers, tests and disposition.
- `phase13-migration-delta.json`: empty Phase 0 transfer baseline.
- `phase13-symbol-delta-ledger.json`: planned renames/transfers plus unresolved four-part deletion audits.
- `phase13-rayd-integration-inventory.json`: legacy terms, bridge/getter map, RayD extern-C entries and typed-v2 absence evidence.
- `phase13-live-dead-public-internal-inventory.json`: conservative static/E2E/public classification; no-static is never treated as deletion authorization.
- `phase13-transmission-contracts.json`: the six complete-family contracts and frozen fusion/tape/compile constraints.
- `phase13-diffraction-family-matrix.json`: operation-family ownership, including layered geometry and MC tape producer/consumer distinctions.
- `phase13-scattering-bindings.json`: 17 conditional moves plus the one retained event-policy binding.
- `phase13-shared-rf-dependency-graph.json` and `phase13-shared-rf-helper-ledger.json`: include closure and helper-level owner/mirror/compiler evidence.
- `phase13-baseline-evidence.json`: toolchain/build identity and digest index
  for the immutable runtime baseline under
  `docs/dev/baselines/{BASELINE_CHANNEL}/runtime/`.

## Known Phase 0 limits

Static references establish positive liveness evidence but cannot prove a symbol
dead. Every unresolved diffraction legacy binding remains marked for the four
checks required by Plan 13: static caller, dynamic binding, public import and
real BDPT E2E. Runtime exact outputs, Nsight launch/sync/memcpy/peak-memory and
PTX/SASS evidence are separate executable baselines; this source-only generator
does not fabricate them.
"""
(AUDIT_DIR / "phase13-phase0-audit.md").write_text(summary, encoding="utf-8")


if BASELINE["channel_commit_scanned"] != BASELINE_CHANNEL:
    raise SystemExit("Channel checkout moved from the required Phase 0 baseline")
if BASELINE["rayd_commit_scanned"] != BASELINE_RAYD:
    raise SystemExit("RayD checkout moved from the required Phase 0 baseline")
if len(SYMBOLS) != 211:
    raise SystemExit(f"Expected 211 native bindings, found {len(SYMBOLS)}")
if len(TRANSMISSION) != 6 or len(SCATTERING_MOVE) != 17 or len(SCATTERING_RETAIN) != 1:
    raise SystemExit("Plan 13 family cardinality invariant failed")

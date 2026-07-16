"""Validate complete public-API and native-binding contract/E2E coverage."""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

# Direct script execution starts with only ci/ on sys.path.
from tools.refactor_baseline import binding_manifest  # noqa: E402


DEFAULT_MANIFEST_PATH = Path("ci/contract-coverage-manifest.json")
PUBLIC_SNAPSHOT_PATH = Path("ci/public-api-snapshot.json")
BINDING_BASELINE_PATH = Path(
    "docs/dev/baselines/0892d855b27ee851521a181f5158b0bf41091eda/"
    "static/bindings.json"
)
PHASE10_AUDIT_PATH = Path("docs/dev/audit/phase10-legacy-dead-binding.json")
PYTHON_PACKAGE_PATH = Path("src/witwin/channel_native")
EXPECTED_PUBLIC_EXPORT_COUNT = 37
EXPECTED_NATIVE_BINDING_COUNT = 174
PUBLIC_COLUMNS = ("export", "contract_test", "e2e_callers")
NATIVE_COLUMNS = (
    "symbol",
    "python_owner",
    "owner_kind",
    "contract_test",
    "e2e_callers",
)
OWNER_KINDS = frozenset({"named_wrapper", "native_call_site"})
BOOTSTRAP_CALL_SITE_OWNERS = {
    "coupled_rd_prepare": (
        "witwin.channel_native.propagation.fields.kernels.autograd."
        "_CoupledRdPrepareAdFunction.forward"
    ),
    "coupled_rd_prepare_backward": (
        "witwin.channel_native.propagation.fields.kernels.autograd."
        "_CoupledRdPrepareAdFunction.backward"
    ),
    "coupled_rd_prepare_jvp": (
        "witwin.channel_native.propagation.fields.kernels.autograd."
        "_CoupledRdPrepareAdFunction.jvp"
    ),
    "deterministic_accumulate_flat_fwd64": (
        "witwin.channel_native.deterministic.kernels.accumulation."
        "_DeterministicAccumulateFlatAdFunction.forward"
    ),
    "deterministic_los_topology_block_all_visible": (
        "witwin.channel_native.propagation.topology.kernels.construction."
        "deterministic_los_topology_block"
    ),
    "field_coupled_rd_backward": (
        "witwin.channel_native.propagation.fields.kernels.autograd."
        "_FieldCoupledRdAdFunction.backward"
    ),
    "field_coupled_rd_jvp": (
        "witwin.channel_native.propagation.fields.kernels.autograd."
        "_FieldCoupledRdAdFunction.jvp"
    ),
    "field_diffraction_wedge_backward": (
        "witwin.channel_native.propagation.fields.kernels.autograd."
        "_FieldDiffractionWedgeAdFunction.backward"
    ),
    "field_diffraction_wedge_jvp": (
        "witwin.channel_native.propagation.fields.kernels.autograd."
        "_FieldDiffractionWedgeAdFunction.jvp"
    ),
    "field_free_space_fwd64": (
        "witwin.channel_native.propagation.fields.kernels.autograd."
        "_FieldFreeSpaceAdFunction.forward"
    ),
    "field_project_complex3_backward": (
        "witwin.channel_native.propagation.fields.kernels.autograd."
        "_FieldProjectComplex3AdFunction.backward"
    ),
    "field_project_complex3_jvp": (
        "witwin.channel_native.propagation.fields.kernels.autograd."
        "_FieldProjectComplex3AdFunction.jvp"
    ),
}
BOOTSTRAP_E2E_SCENARIOS = {
    "bdpt-diffraction": (
        "tests/montecarlo/bdpt/test_diffraction_single_wedge.py::"
        "test_bdpt_single_wedge_diffraction_returns_finite_native_component_when_available"
    ),
    "bdpt-los": (
        "tests/montecarlo/bdpt/test_los_empty_space.py::"
        "test_bdpt_los_empty_space_matches_analytic_reference"
    ),
    "bdpt-reflection": (
        "tests/montecarlo/bdpt/test_reflection_single_plane.py::"
        "test_bdpt_single_plane_reflection_returns_nonzero_native_component_when_available"
    ),
    "bdpt-scattering": (
        "tests/montecarlo/bdpt/test_scattering.py::"
        "test_bdpt_scattering_matches_area_quadrature_reference"
    ),
    "bdpt-transmission": (
        "tests/montecarlo/bdpt/test_transmission.py::"
        "test_bdpt_lossy_wall_transmission_power_ratio_matches_stack"
    ),
    "build-info": "tests/kernels/test_build_info.py::test_build_info_contract",
    "deterministic-ad": (
        "tests/ad/test_deterministic_accum_ad.py::"
        "test_accumulate_flat_jvp_vjp_duality"
    ),
    "deterministic-diffraction": (
        "tests/deterministic/test_diffraction_single_wedge.py::"
        "test_single_wedge_diffraction_matches_path_reference"
    ),
    "deterministic-los": (
        "tests/deterministic/test_los_empty_space.py::"
        "test_empty_space_los_matches_analytic_reference"
    ),
    "deterministic-reflection": (
        "tests/deterministic/test_reflection_single_plane.py::"
        "test_single_plane_reflection_matches_path_reference"
    ),
    "field-coupled": (
        "tests/ad/test_solver_diffraction_coupled_ad.py::"
        "test_coupled_material_grad_matches_fd"
    ),
    "field-diffraction": (
        "tests/ad/test_solver_diffraction_coupled_ad.py::"
        "test_wedge_material_grad_matches_fd"
    ),
    "field-free-space": (
        "tests/ad/test_field_em_ad.py::"
        "test_free_space_frequency_vjp_matches_reference_oracle"
    ),
    "field-reflection": (
        "tests/ad/test_field_em_ad.py::"
        "test_reflection_material_vjp_matches_reference_oracle"
    ),
    "field-transmission": (
        "tests/ad/test_field_em_ad.py::"
        "test_transmission_layer_vjp_matches_reference_oracle"
    ),
    "mc-basic-ad": "tests/ad/test_mc_basic_ad.py::test_frequency_grad_matches_fd",
    "mc-basic-diffraction": (
        "tests/montecarlo/basic/test_basic_component_maps.py::"
        "test_basic_solver_returns_native_diffraction_component_map_when_available"
    ),
    "mc-basic-los": (
        "tests/montecarlo/basic/test_basic_solver_smoke.py::"
        "test_basic_solver_los_smoke_returns_cuda_result"
    ),
    "mc-basic-reflection": (
        "tests/montecarlo/basic/test_basic_component_maps.py::"
        "test_basic_solver_returns_native_reflection_component_map_when_available"
    ),
    "mc-basic-scattering": (
        "tests/montecarlo/basic/test_basic_scattering.py::"
        "test_basic_scattering_map_matches_area_quadrature_reference"
    ),
    "mc-basic-transmission": (
        "tests/montecarlo/basic/test_basic_transmission.py::"
        "test_lossy_wall_attenuates_by_stack_power_transmittance"
    ),
    "path-coupled": (
        "tests/path/test_path_reflection_diffraction_sequences.py::"
        "test_solve_exports_bounded_reflection_diffraction_sequences"
    ),
    "path-diffraction": (
        "tests/path/test_path_solver_smoke.py::"
        "test_path_solver_exports_native_diffraction_paths_when_available"
    ),
    "path-los": (
        "tests/path/test_path_solver_smoke.py::"
        "test_path_solver_empty_space_los_returns_one_path_per_pair"
    ),
    "path-reflection": (
        "tests/path/test_path_solver_smoke.py::"
        "test_path_solver_exports_native_reflection_paths_when_available"
    ),
    "public-capabilities": (
        "tests/test_capabilities.py::"
        "test_capability_manifest_is_versioned_serializable_and_defensive"
    ),
    "public-core": (
        "tests/core/test_public_scene.py::"
        "test_public_scene_objects_capture_structured_inputs"
    ),
    "public-deployment": (
        "tests/performance/test_deployment_contract.py::"
        "test_pipeline_cache_key_is_stable_and_invalidates_all_contract_inputs"
    ),
    "public-materials": (
        "tests/core/test_public_scene.py::test_materials_compile_scalar_parameters"
    ),
    "scene-native": (
        "tests/scene/test_rayd_scene_kernels.py::"
        "test_raydn_scene_builder_preserves_native_order_flags_uv_and_keepalive"
    ),
}


def load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("contract coverage manifest must be an object")
    return value


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain an object")
    return value


def _public_snapshot_exports(snapshot: dict[str, Any]) -> list[str]:
    exports: list[str] = []
    for module in snapshot.get("modules", []):
        module_name = module["module"]
        exports.extend(
            f"{module_name}.{entry['name']}" for entry in module["exports"]
        )
    return exports


def _nodeid_issue(repo: Path, nodeid: object, *, label: str) -> str | None:
    if not isinstance(nodeid, str) or "::" not in nodeid:
        return f"{label} must be a pytest nodeid"
    path_text, *parts = nodeid.split("::")
    path = repo / path_text
    if not path.is_file():
        return f"{label} test path does not exist: {path_text}"
    try:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    except (OSError, SyntaxError) as exc:
        return f"{label} test cannot be parsed: {exc}"
    test_name = parts[-1].split("[", maxsplit=1)[0]
    definitions = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if test_name not in definitions:
        return f"{label} nodeid does not exist: {nodeid}"
    return None


class _DefinitionVisitor(ast.NodeVisitor):
    def __init__(self, module: str) -> None:
        self.module = module
        self.stack: list[str] = []
        self.definitions: dict[str, ast.AST] = {}

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._visit_function(node)

    def visit_AsyncFunctionDef(  # noqa: N802
        self, node: ast.AsyncFunctionDef
    ) -> None:
        self._visit_function(node)

    def _visit_function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        self.stack.append(node.name)
        qualified_name = ".".join((self.module, *self.stack))
        self.definitions[qualified_name] = node
        self.generic_visit(node)
        self.stack.pop()


def _python_definitions(repo: Path) -> dict[str, ast.AST]:
    root = repo / PYTHON_PACKAGE_PATH
    definitions: dict[str, ast.AST] = {}
    for path in sorted(root.rglob("*.py")):
        suffix = list(path.relative_to(root).with_suffix("").parts)
        if suffix[-1] == "__init__":
            suffix.pop()
        module = ".".join(("witwin", "channel_native", *suffix))
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        visitor = _DefinitionVisitor(module)
        visitor.visit(tree)
        overlap = set(definitions) & set(visitor.definitions)
        if overlap:
            raise ValueError(
                "duplicate Python qualified definitions: " + ", ".join(sorted(overlap))
            )
        definitions.update(visitor.definitions)
    return definitions


def _native_reference_count(node: ast.AST, symbol: str) -> int:
    return sum(
        1
        for child in ast.walk(node)
        if (
            isinstance(child, ast.Constant)
            and child.value == symbol
            or isinstance(child, ast.Attribute)
            and child.attr == symbol
        )
    )


def _validate_registry(
    repo: Path, value: object, *, label: str
) -> tuple[dict[str, str], list[str]]:
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(nodeid, str)
        for key, nodeid in value.items()
    ):
        return {}, [f"{label} must map IDs to pytest nodeids"]
    registry = {str(key): str(nodeid) for key, nodeid in value.items()}
    issues = [
        issue
        for key, nodeid in sorted(registry.items())
        if (issue := _nodeid_issue(repo, nodeid, label=f"{label}.{key}")) is not None
    ]
    return registry, issues


def _validate_public_rows(
    value: object,
) -> tuple[dict[str, tuple[str, tuple[str, ...]]], list[str]]:
    if not isinstance(value, list):
        return {}, ["public_exports must be a list"]
    entries: dict[str, tuple[str, tuple[str, ...]]] = {}
    issues: list[str] = []
    for row in value:
        if not (
            isinstance(row, list)
            and len(row) == len(PUBLIC_COLUMNS)
            and isinstance(row[0], str)
            and isinstance(row[1], str)
            and isinstance(row[2], list)
            and all(isinstance(item, str) for item in row[2])
        ):
            issues.append("each public export row must match public_export_columns")
            continue
        export, contract_test, callers = row
        if export in entries:
            issues.append(f"duplicate public export coverage: {export}")
        if not callers:
            issues.append(f"public export has no E2E caller: {export}")
        entries[export] = (contract_test, tuple(callers))
    return entries, issues


def _validate_native_rows(
    value: object,
) -> tuple[dict[str, tuple[str, str, str, tuple[str, ...]]], list[str]]:
    if not isinstance(value, list):
        return {}, ["native_bindings must be a list"]
    entries: dict[str, tuple[str, str, str, tuple[str, ...]]] = {}
    issues: list[str] = []
    for row in value:
        if not (
            isinstance(row, list)
            and len(row) == len(NATIVE_COLUMNS)
            and all(isinstance(item, str) for item in row[:4])
            and isinstance(row[4], list)
            and all(isinstance(item, str) for item in row[4])
        ):
            issues.append("each native binding row must match native_binding_columns")
            continue
        symbol, owner, owner_kind, contract_test, callers = row
        if symbol in entries:
            issues.append(f"duplicate native binding coverage: {symbol}")
        if owner_kind not in OWNER_KINDS:
            issues.append(f"unsupported owner kind for {symbol}: {owner_kind}")
        if not callers:
            issues.append(f"native binding has no E2E caller: {symbol}")
        entries[symbol] = (owner, owner_kind, contract_test, tuple(callers))
    return entries, issues


def _set_mismatch(label: str, actual: set[str], expected: set[str]) -> list[str]:
    issues: list[str] = []
    missing = expected - actual
    extra = actual - expected
    if missing:
        issues.append(f"{label} missing: " + ", ".join(sorted(missing)))
    if extra:
        issues.append(f"{label} unknown: " + ", ".join(sorted(extra)))
    return issues


def _initial_public_scenario(module: str, name: str) -> str:
    if module == "witwin.channel_native.path":
        return "path-los" if name == "solve" else "path-reflection"
    if module == "witwin.channel_native.deterministic":
        return "deterministic-los"
    if module == "witwin.channel_native.montecarlo.basic":
        return "mc-basic-los"
    if module == "witwin.channel_native.montecarlo.bdpt":
        return "bdpt-los"
    if name == "build_info":
        return "build-info"
    if name in {"pipeline_cache_key", "runtime_diagnostics"}:
        return "public-deployment"
    if name == "capabilities":
        return "public-capabilities"
    if name in {
        "Dielectric",
        "DispersiveMaterial",
        "ITUMaterial",
        "LossyDielectric",
        "PerfectConductor",
    }:
        return "public-materials"
    return "public-core"


def _initial_native_scenario(name: str) -> str:
    if name == "build_info":
        return "build-info"
    if name.startswith("em_layer_stack"):
        return "field-transmission"
    if name.startswith("coupled_rd_prepare") or name.startswith("field_coupled_rd"):
        return "field-coupled"
    if name.startswith("field_"):
        if "diffraction" in name:
            return "field-diffraction"
        if "transmission" in name:
            return "field-transmission"
        if "free_space" in name:
            return "field-free-space"
        return "field-reflection"
    if name.startswith("deterministic_"):
        if any(token in name for token in ("backward", "jvp", "fwd64")):
            return "deterministic-ad"
        if "diffraction" in name or "edge" in name:
            return "deterministic-diffraction"
        if "los" in name:
            return "deterministic-los"
        return "deterministic-reflection"
    if name.startswith("mc_"):
        if any(token in name for token in ("backward", "jvp", "adjoint")):
            return "mc-basic-ad"
        if "diffraction" in name or "edge" in name:
            return "mc-basic-diffraction"
        if "reflection" in name:
            return "mc-basic-reflection"
        if "los" in name:
            return "mc-basic-los"
        return "mc-basic-reflection"
    if name.startswith("bdpt_"):
        if "diffraction" in name or "edge" in name:
            return "bdpt-diffraction"
        if "transmission" in name or "transmitted" in name:
            return "bdpt-transmission"
        if "scatter" in name:
            return "bdpt-scattering"
        if "los" in name:
            return "bdpt-los"
        return "bdpt-reflection"
    if name.startswith("path_"):
        if "diffraction" in name or "edge" in name:
            return "path-diffraction"
        if "los" in name:
            return "path-los"
        return "path-reflection"
    if name.startswith("raydn_scene_"):
        return "scene-native"
    if name.startswith("raydn_"):
        if "coupled" in name:
            return "path-coupled"
        if "diffraction" in name or "edge" in name:
            return "path-diffraction"
        return "path-reflection"
    if name.startswith("scattering_"):
        return "bdpt-scattering"
    if name.startswith("core_"):
        return "path-diffraction"
    raise ValueError(f"native binding has no E2E scenario rule: {name}")


def _initial_native_contract(name: str) -> str:
    if name == "build_info":
        return "native-bootstrap"
    if name.startswith("em_layer_stack"):
        return "native-material-layer-stack"
    if name.startswith("scattering_"):
        return "native-scattering"
    if name.startswith("coupled_rd_prepare") or name.startswith("field_coupled_rd"):
        return "native-field-coupled"
    if name.startswith("field_"):
        if "diffraction" in name:
            return "native-field-diffraction"
        if "transmission" in name:
            return "native-field-transmission"
        if "free_space" in name:
            return "native-field-free-space"
        return "native-field-reflection"
    if name.startswith("deterministic_"):
        if "accumulate" in name or name == "deterministic_component_counts":
            return "native-deterministic-accumulation"
        if any(
            token in name
            for token in (
                "field",
                "phase",
                "delay_to_path_length",
                "pack_complex",
            )
        ):
            return "native-deterministic-fields"
        return "native-deterministic-topology"
    if name.startswith("mc_"):
        if any(
            token in name
            for token in (
                "sample",
                "selected_edge",
                "diffraction_state",
                "pack_vec3",
            )
        ):
            return "native-mc-sampling"
        return "native-mc-maps"
    if name.startswith("bdpt_"):
        if any(
            token in name
            for token in (
                "component_map",
                "finalize",
                "los_",
                "point_component",
                "receiver_grid",
                "store_",
                "zero_matrix",
                "face_material",
            )
        ):
            return "native-bdpt-maps"
        if any(
            token in name
            for token in (
                "launch_state",
                "pack_vec3",
                "reflection_launch",
                "sample_directions",
                "selected_edge",
                "diffraction_state",
            )
        ):
            return "native-bdpt-sampling"
        return "native-bdpt-paths"
    if name.startswith("path_") or name.startswith("core_"):
        return "native-path-topology"
    if name.startswith("raydn_scene_"):
        return "native-raydn-scene"
    if name.startswith("raydn_"):
        return "native-raydn-geometry"
    raise ValueError(f"native binding has no runtime contract rule: {name}")


def build_initial_manifest(repo: Path) -> dict[str, object]:
    """Build a reviewable initial matrix; never update the checked-in file."""
    repo = repo.resolve()
    snapshot = _load_json_object(repo / PUBLIC_SNAPSHOT_PATH)
    baseline = _load_json_object(repo / BINDING_BASELINE_PATH)
    definitions = _python_definitions(repo)
    by_terminal: dict[str, list[str]] = defaultdict(list)
    for qualified_name in definitions:
        by_terminal[qualified_name.rsplit(".", maxsplit=1)[-1]].append(qualified_name)

    public_rows: list[list[object]] = []
    for module in snapshot["modules"]:
        module_name = str(module["module"])
        for entry in module["exports"]:
            name = str(entry["name"])
            public_rows.append(
                [
                    f"{module_name}.{name}",
                    "public-api-snapshot",
                    [_initial_public_scenario(module_name, name)],
                ]
            )

    native_rows: list[list[object]] = []
    for entry in baseline["symbols"]:
        symbol = str(entry["name"])
        if symbol in BOOTSTRAP_CALL_SITE_OWNERS:
            owner = BOOTSTRAP_CALL_SITE_OWNERS[symbol]
            owner_kind = "native_call_site"
        else:
            candidates = by_terminal.get(symbol, [])
            if len(candidates) != 1:
                raise ValueError(
                    f"native binding needs one named wrapper: {symbol}: {candidates}"
                )
            owner = candidates[0]
            owner_kind = "named_wrapper"
        native_rows.append(
            [
                symbol,
                owner,
                owner_kind,
                _initial_native_contract(symbol),
                [_initial_native_scenario(symbol)],
            ]
        )

    used_scenarios = {
        str(row[-1][0]) for row in [*public_rows, *native_rows]  # type: ignore[index]
    }
    return {
        "schema_version": 1,
        "sources": {
            "public_api_snapshot": PUBLIC_SNAPSHOT_PATH.as_posix(),
            "native_binding_baseline": BINDING_BASELINE_PATH.as_posix(),
            "phase10_binding_audit": PHASE10_AUDIT_PATH.as_posix(),
        },
        "public_export_columns": list(PUBLIC_COLUMNS),
        "native_binding_columns": list(NATIVE_COLUMNS),
        "contract_tests": {
            "public-api-snapshot": (
                "tests/test_public_api_snapshot.py::"
                "test_curated_public_api_matches_frozen_snapshot"
            ),
            "native-bdpt-maps": (
                "tests/kernels/test_bdpt_ops_facade.py::"
                "test_bdpt_los_component_maps_from_matrix_uses_native_grid_layout"
            ),
            "native-bdpt-paths": (
                "tests/kernels/test_bdpt_ops_facade.py::"
                "test_bdpt_endpoint_connection_samples_emit_native_connection_schema"
            ),
            "native-bdpt-sampling": (
                "tests/kernels/test_bdpt_ops_facade.py::"
                "test_bdpt_launch_state_returns_stable_cuda_tensors"
            ),
            "native-bootstrap": (
                "tests/kernels/test_extension_loading.py::"
                "test_missing_bootstrap_symbol_fails_before_any_computation"
            ),
            "native-deterministic-accumulation": (
                "tests/kernels/test_ops_facade.py::"
                "test_deterministic_accumulate_flat_matches_torch_reference"
            ),
            "native-deterministic-fields": (
                "tests/kernels/test_ops_facade.py::"
                "test_deterministic_reflection_field_returns_native_complex_field"
            ),
            "native-deterministic-topology": (
                "tests/kernels/test_ops_facade.py::"
                "test_deterministic_los_topology_block_compacts_and_fills_extended_fields"
            ),
            "native-field-coupled": (
                "tests/ad/test_solver_diffraction_coupled_ad.py::"
                "test_coupled_forward_stays_primal_without_geometry_leaves"
            ),
            "native-field-diffraction": (
                "tests/ad/test_solver_diffraction_coupled_ad.py::"
                "test_wedge_reevaluation_forward_parity"
            ),
            "native-field-free-space": (
                "tests/kernels/test_field_transport.py::"
                "test_free_space_complex3_matches_analytic_phase_and_power_contract"
            ),
            "native-field-reflection": (
                "tests/kernels/test_field_transport.py::"
                "test_lossy_slab_reflection_matches_complex_reference"
            ),
            "native-field-transmission": (
                "tests/kernels/test_field_transmission_sequence.py::"
                "test_lossy_wall_attenuates_consistently_with_layer_stack_eval"
            ),
            "native-material-layer-stack": (
                "tests/kernels/test_em_layer_stack.py::"
                "test_single_layer_matches_complex128_oracle_at_oblique_lossy_conditions"
            ),
            "native-mc-maps": (
                "tests/kernels/test_ops_facade.py::"
                "test_mc_component_map_buffer_and_store_kernels_write_tx_slots"
            ),
            "native-mc-sampling": (
                "tests/kernels/test_ops_facade.py::"
                "test_mc_diffraction_state_pack_gathers_edge_state_tensors"
            ),
            "native-path-topology": (
                "tests/kernels/test_ops_facade.py::"
                "test_path_diffraction_block_and_merge_use_native_compaction"
            ),
            "native-raydn-geometry": (
                "tests/propagation/geometry/test_kernel_bridge.py::"
                "test_intersection_returns_the_named_tensor_contract"
            ),
            "native-raydn-scene": (
                "tests/scene/test_rayd_scene_kernels.py::"
                "test_scene_create_preserves_native_argument_order"
            ),
            "native-scattering": (
                "tests/kernels/test_scattering_kernels.py::"
                "test_scattering_event_probabilities_require_native_kernel"
            ),
        },
        "e2e_scenarios": {
            key: BOOTSTRAP_E2E_SCENARIOS[key] for key in sorted(used_scenarios)
        },
        "public_exports": public_rows,
        "native_bindings": native_rows,
    }


def check_contract_coverage(repo: Path, manifest: dict[str, Any]) -> list[str]:
    repo = repo.resolve()
    issues: list[str] = []
    expected_keys = {
        "schema_version",
        "sources",
        "public_export_columns",
        "native_binding_columns",
        "contract_tests",
        "e2e_scenarios",
        "public_exports",
        "native_bindings",
    }
    if set(manifest) != expected_keys:
        issues.append("contract coverage manifest fields are not exact")
    if manifest.get("schema_version") != 1:
        issues.append("contract coverage manifest must use schema_version 1")
    if manifest.get("public_export_columns") != list(PUBLIC_COLUMNS):
        issues.append("public_export_columns changed")
    if manifest.get("native_binding_columns") != list(NATIVE_COLUMNS):
        issues.append("native_binding_columns changed")

    sources = manifest.get("sources")
    expected_sources = {
        "public_api_snapshot": PUBLIC_SNAPSHOT_PATH.as_posix(),
        "native_binding_baseline": BINDING_BASELINE_PATH.as_posix(),
        "phase10_binding_audit": PHASE10_AUDIT_PATH.as_posix(),
    }
    if sources != expected_sources:
        issues.append("contract coverage source paths changed")

    contract_tests, registry_issues = _validate_registry(
        repo, manifest.get("contract_tests"), label="contract_tests"
    )
    issues.extend(registry_issues)
    scenarios, scenario_issues = _validate_registry(
        repo, manifest.get("e2e_scenarios"), label="e2e_scenarios"
    )
    issues.extend(scenario_issues)
    public_entries, public_issues = _validate_public_rows(
        manifest.get("public_exports")
    )
    issues.extend(public_issues)
    native_entries, native_issues = _validate_native_rows(
        manifest.get("native_bindings")
    )
    issues.extend(native_issues)

    snapshot = _load_json_object(repo / PUBLIC_SNAPSHOT_PATH)
    snapshot_exports = _public_snapshot_exports(snapshot)
    if len(snapshot_exports) != len(set(snapshot_exports)):
        issues.append("public API snapshot contains duplicate exports")
    if len(snapshot_exports) != EXPECTED_PUBLIC_EXPORT_COUNT:
        issues.append(
            "public API snapshot count changed: "
            f"expected {EXPECTED_PUBLIC_EXPORT_COUNT}, got {len(snapshot_exports)}"
        )
    issues.extend(
        _set_mismatch(
            "public export coverage", set(public_entries), set(snapshot_exports)
        )
    )

    baseline = _load_json_object(repo / BINDING_BASELINE_PATH)
    baseline_names = [str(entry["name"]) for entry in baseline["symbols"]]
    current = binding_manifest(repo)
    current_names = [str(entry["name"]) for entry in current["symbols"]]
    if current.get("duplicate_symbols") != []:
        issues.append("current native binding manifest contains duplicate symbols")
    if len(current_names) != len(set(current_names)):
        issues.append("current native binding names are not unique")
    if len(current_names) != EXPECTED_NATIVE_BINDING_COUNT:
        issues.append(
            "current native binding count changed: "
            f"expected {EXPECTED_NATIVE_BINDING_COUNT}, got {len(current_names)}"
        )
    if baseline_names != current_names:
        issues.append("current native binding universe differs from frozen baseline")
    issues.extend(
        _set_mismatch(
            "native binding coverage", set(native_entries), set(current_names)
        )
    )

    phase10 = _load_json_object(repo / PHASE10_AUDIT_PATH)
    phase10_ownership = phase10["binding_ownership"]
    if (
        phase10_ownership.get("expected_count") != EXPECTED_NATIVE_BINDING_COUNT
        or phase10_ownership.get("symbols") != current_names
    ):
        issues.append("Phase10 binding ownership audit differs from current bindings")

    used_contracts: set[str] = set()
    used_scenarios: set[str] = set()
    for export, (contract_test, callers) in public_entries.items():
        used_contracts.add(contract_test)
        used_scenarios.update(callers)
        if contract_test not in contract_tests:
            issues.append(f"unknown contract test for public export {export}: {contract_test}")
        unknown_callers = set(callers) - set(scenarios)
        if unknown_callers:
            issues.append(
                f"unknown E2E callers for public export {export}: "
                + ", ".join(sorted(unknown_callers))
            )

    definitions = _python_definitions(repo)
    by_terminal: dict[str, list[str]] = defaultdict(list)
    for qualified_name in definitions:
        by_terminal[qualified_name.rsplit(".", maxsplit=1)[-1]].append(qualified_name)

    for symbol, (owner, owner_kind, contract_test, callers) in native_entries.items():
        used_contracts.add(contract_test)
        used_scenarios.update(callers)
        if contract_test not in contract_tests:
            issues.append(
                f"unknown contract test for native binding {symbol}: {contract_test}"
            )
        unknown_callers = set(callers) - set(scenarios)
        if unknown_callers:
            issues.append(
                f"unknown E2E callers for native binding {symbol}: "
                + ", ".join(sorted(unknown_callers))
            )
        owner_node = definitions.get(owner)
        if owner_node is None:
            issues.append(f"native binding Python owner does not exist: {symbol}: {owner}")
            continue
        if owner_kind == "named_wrapper":
            candidates = sorted(by_terminal.get(symbol, []))
            if candidates != [owner]:
                issues.append(
                    f"native binding named wrapper is not unique: {symbol}: "
                    + ", ".join(candidates)
                )
        elif owner_kind == "native_call_site":
            reference_owners = sorted(
                qualified_name
                for qualified_name, node in definitions.items()
                if _native_reference_count(node, symbol)
            )
            if reference_owners != [owner]:
                issues.append(
                    f"native binding call-site owner is not unique: {symbol}: "
                    + ", ".join(reference_owners)
                )
            elif _native_reference_count(owner_node, symbol) != 1:
                issues.append(
                    f"native binding owner must reference symbol once: {symbol}: {owner}"
                )

    unused_contracts = set(contract_tests) - used_contracts
    unused_scenarios = set(scenarios) - used_scenarios
    if unused_contracts:
        issues.append("unused contract tests: " + ", ".join(sorted(unused_contracts)))
    if unused_scenarios:
        issues.append("unused E2E scenarios: " + ", ".join(sorted(unused_scenarios)))
    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--print-initial",
        action="store_true",
        help="print a reviewable initial matrix; never updates the manifest",
    )
    args = parser.parse_args(argv)
    repository_root = args.repository_root.resolve()
    if args.print_initial:
        print(json.dumps(build_initial_manifest(repository_root), indent=2) + "\n")
        return 0
    manifest_path = (
        args.manifest or repository_root / DEFAULT_MANIFEST_PATH
    ).resolve()
    try:
        manifest = load_manifest(manifest_path)
        issues = check_contract_coverage(repository_root, manifest)
    except (KeyError, OSError, SyntaxError, TypeError, ValueError) as exc:
        print(f"contract coverage configuration error: {exc}")
        return 2
    if issues:
        for issue in issues:
            print(issue)
        return 1
    print(
        "contract coverage passed "
        f"({EXPECTED_PUBLIC_EXPORT_COUNT} public exports, "
        f"{EXPECTED_NATIVE_BINDING_COUNT} native bindings)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

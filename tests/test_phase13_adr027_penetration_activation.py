from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BINDING_MANIFEST = ROOT / "ci/native-binding-manifest.json"
COVERAGE_MANIFEST = ROOT / "ci/contract-coverage-manifest.json"
OWNER_INVENTORY = ROOT / "docs/dev/audit/phase13-current-native-owner-inventory.json"
SYMBOL_LEDGER = ROOT / "docs/dev/audit/phase13-symbol-delta-ledger.json"

_PHASE_E_SYMBOLS = {
    "enumerated_transmission_topology_pack_backward",
    "enumerated_transmission_topology_pack_jvp",
    "enumerated_capacity_failure_sanitize",
    "enumerated_capacity_failure_vector_sanitize",
}
_RETIRED_NAMES = {
    "TransmissionClosestHitQuery",
    "query_transmission_closest_hit",
    "iter_transmission_active_rows",
}


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_phase_e_symbols_have_one_live_owner_contract_and_e2e_caller() -> None:
    binding = _json(BINDING_MANIFEST)
    coverage = _json(COVERAGE_MANIFEST)
    inventory = _json(OWNER_INVENTORY)
    ledger = _json(SYMBOL_LEDGER)
    binding_names = {entry["name"] for entry in binding["symbols"]}
    coverage_rows = {entry[0]: entry for entry in coverage["native_bindings"]}
    owner_rows = {entry["symbol"]: entry for entry in inventory["symbols"]}
    actions = {entry["symbol"]: entry for entry in ledger["actions"]}

    assert len(binding_names) == 234
    assert _PHASE_E_SYMBOLS <= binding_names
    assert _PHASE_E_SYMBOLS <= set(coverage_rows) == binding_names
    assert _PHASE_E_SYMBOLS <= set(owner_rows) == binding_names
    assert _PHASE_E_SYMBOLS <= set(actions)
    assert ledger["phase6c_phase_e_delta"] == {
        "before": 234,
        "added": 4,
        "after": 238,
    }
    for symbol in _PHASE_E_SYMBOLS:
        row = coverage_rows[symbol]
        assert row[1]
        assert row[3]
        assert row[4]
        assert owner_rows[symbol]["production_callers"]
        assert actions[symbol]["count_delta"] == 1
        assert "v2" not in symbol.casefold()


def test_phase_e_deletes_old_depth_march_without_a_compatibility_alias() -> None:
    package = ROOT / "witwin/channel"
    assert not (package / "propagation/geometry/transmission.py").exists()
    assert not (package / "propagation/topology/discovery/transmission.py").exists()
    for path in package.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for name in _RETIRED_NAMES:
            assert name not in source, f"{path} retains retired {name}"


def test_phase_e_sanitizers_are_async_current_stream_and_have_no_trap() -> None:
    source = (
        ROOT / "native/channel/kernels/enumerated_capacity_failure_sanitize.cu"
    ).read_text(encoding="utf-8")
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")

    assert "enumerated_capacity_failure_sanitize.cu" in cmake
    assert "getCurrentCUDAStream" in source
    assert "failure_state[0]" in source
    for forbidden in (
        "trap;",
        "cudaMemcpy",
        "cudaStreamSynchronize",
        "cudaDeviceSynchronize",
    ):
        assert forbidden not in source


def test_phase_e_docs_keep_the_adr029_compaction_blocker_explicit() -> None:
    plan = (
        ROOT
        / "docs/dev/plans/13-direct-rayd-integration-and-rf-runtime-ownership-plan.md"
    ).read_text(encoding="utf-8")
    propagation = (ROOT / "docs/dev/propagation/README.md").read_text(
        encoding="utf-8"
    )

    assert "enumerated atomic" in plan
    assert "ADR-029 capacity-pack activation 已由 ADR-032 取消" in plan
    assert "不得再宣称或追求 public capacity result 与" in plan
    assert "The canonical selector still compacts valid candidate rows" in propagation
    assert "ADR-029's downstream capacity closure is superseded" in propagation

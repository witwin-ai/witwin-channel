from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BINDING_MANIFEST = ROOT / "ci/native-binding-manifest.json"
COVERAGE_MANIFEST = ROOT / "ci/contract-coverage-manifest.json"
OWNER_INVENTORY = ROOT / "docs/dev/audit/phase13-current-native-owner-inventory.json"
SYMBOL_LEDGER = ROOT / "docs/dev/audit/phase13-symbol-delta-ledger.json"
RESOURCE_LEDGER = (
    ROOT / "docs/dev/audit/adr-027-mc-transmission-wall-product-resource-ledger.json"
)

_LIVE_MC_SYMBOLS = {
    "mc_capacity_failure_component_maps_sanitize",
    "mc_capacity_failure_component_maps_sanitize_backward",
    "mc_capacity_failure_component_maps_sanitize_jvp",
    "mc_transmission_wall_product",
    "mc_transmission_wall_product_backward",
    "mc_transmission_wall_product_jvp",
    "rayd_segment_penetration_forward",
    "rayd_segment_penetration_forward_tape",
    "rayd_segment_penetration_backward",
    "rayd_segment_penetration_jvp",
}


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _basic_section(title: str) -> str:
    """One section of the collapsed Monte Carlo basic solver module.

    ``montecarlo/basic.py`` concatenates the former package's modules behind
    ``# --- <title> ---`` rules, so an assertion that used to bound one file
    bounds the corresponding section instead.
    """

    source = (ROOT / "witwin/channel/montecarlo/basic.py").read_text(
        encoding="utf-8"
    )
    start = source.index(f"\n# --- {title} ")
    end = source.find("\n# --- ", start + 1)
    return source[start : end if end != -1 else len(source)]


def test_phase_m_native_symbols_have_live_owners_and_complete_coverage() -> None:
    binding = _json(BINDING_MANIFEST)
    coverage = _json(COVERAGE_MANIFEST)
    inventory = _json(OWNER_INVENTORY)
    ledger = _json(SYMBOL_LEDGER)
    binding_names = {entry["name"] for entry in binding["symbols"]}
    coverage_rows = {entry[0]: entry for entry in coverage["native_bindings"]}
    owner_rows = {entry["symbol"]: entry for entry in inventory["symbols"]}
    actions = {entry["symbol"]: entry for entry in ledger["actions"]}

    assert len(binding_names) == 234
    assert _LIVE_MC_SYMBOLS <= binding_names
    assert set(coverage_rows) == binding_names
    assert set(owner_rows) == binding_names
    assert _LIVE_MC_SYMBOLS <= set(actions)
    assert ledger["phase6c_phase_m_delta"] == {
        "before": 238,
        "added": 3,
        "after": 241,
    }
    for symbol in _LIVE_MC_SYMBOLS:
        coverage_row = coverage_rows[symbol]
        owner_row = owner_rows[symbol]
        assert coverage_row[1]
        assert coverage_row[3]
        assert coverage_row[4]
        assert owner_row["production_callers"]
        assert owner_row["liveness"] == "live-static-production-consumer"
        assert "v2" not in symbol.casefold()
        assert "wip" not in symbol.casefold()


def _transmission_events_section() -> str:
    """The shared Monte Carlo event section of the merged transmission owner.

    The concept axis merged ``montecarlo/events/transmission.py`` into
    ``interactions/transmission.py``, so the assertions that used to bound that
    whole file bound the section it became. Slicing keeps the enumerated
    discovery owner above it out of scope, exactly as a separate file did.
    """

    source = (ROOT / "witwin/channel/interactions/transmission.py").read_text(
        encoding="utf-8"
    )
    return source[
        source.index("# Shared Monte Carlo specular-transmission events (was") :
    ]


def test_phase_m_removes_python_torch_penetration_and_wall_product_route() -> None:
    events = _transmission_events_section()
    component = _basic_section("RayD component maps")

    assert events.count("rayd_segment_penetration_forward(") == 1
    assert events.count("rayd_segment_penetration_ad(") == 1
    assert "SegmentPenetrationPolicy.MonteCarloTargetInset" in events
    transmission_component = component[
        component.index("def transmission_component_map(") : component.index(
            "\ndef scattering_component_map("
        )
    ]
    assert transmission_component.count("straight_transmission_chains(") == 1
    assert "mc_transmission_wall_product" in transmission_component
    for forbidden in (
        "for depth in range(",
        "rayd_intersect_forward(",
        "torch.nonzero(",
        "bool(active.any())",
        "incident_te_tm_fractions",
        "em_layer_stack_eval(",
        "em_layer_stack_ad(",
    ):
        assert forbidden not in events
    assert "for tx_index in range(" not in transmission_component


def test_phase_m_pipeline_has_one_transaction_sanitizer_and_terminal_order() -> None:
    source = _basic_section("Shared solve pipeline")

    create = source.index("create_solve_capacity_transaction(")
    transmission = source.index("transmission_component_map(")
    sanitize = source.index("mc_capacity_failure_component_maps_sanitize(")
    finalize = source.index("finalized = finalize(")
    result = source.index("result = Result(")
    terminal = source.index("capacity_transaction.terminal_check()")
    returned = source.index("return result")

    assert create < transmission < sanitize < finalize < result < terminal < returned
    assert source.count("create_solve_capacity_transaction(") == 1
    assert source.count("capacity_transaction.terminal_check()") == 1
    assert "failure_state=capacity_transaction.failure_state" in source
    for forbidden in (
        "torch.count_nonzero(",
        ".item(",
        ".cpu(",
        ".numpy(",
        ".tolist(",
    ):
        assert forbidden not in source

    metadata = _basic_section("Solver metadata")
    assert "per transmitter" not in metadata
    assert (
        "per layer-stack evaluation inside the transmission chain march" not in metadata
    )


def test_phase_m_active_surface_has_no_generation_or_wip_names() -> None:
    manifest = _json(BINDING_MANIFEST)
    for entry in manifest["symbols"]:
        assert "v2" not in entry["name"].casefold()
        assert "wip" not in entry["name"].casefold()
        assert "v2" not in entry["target"].casefold()
        assert "wip" not in entry["target"].casefold()

    for root in (ROOT / "witwin", ROOT / "native"):
        for path in root.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                name = path.name.casefold()
                assert "wip" not in name
                assert "_v2" not in name
                assert "v2_" not in name


def test_phase_m_docs_do_not_claim_adr029_or_phase12_completion() -> None:
    plan = (
        ROOT
        / "docs/dev/plans/13-direct-rayd-integration-and-rf-runtime-ownership-plan.md"
    ).read_text(encoding="utf-8")
    migration = (ROOT / "docs/dev/replacement/channel-migration.md").read_text(
        encoding="utf-8"
    )

    assert "Phase M MC Basic atomic" in plan
    assert "switch/delete 均已进入 production" in plan
    assert "ADR-029 capacity-pack activation 已由 ADR-032 取消" in plan
    assert "Phase M does not activate ADR-029" in migration
    assert "Final Phase 12 acceptance uses compact E2E" in " ".join(migration.split())

    activation = _json(RESOURCE_LEDGER)["phase_m_activation"]
    assert activation["binding_delta"] == {
        "before": 238,
        "added": 3,
        "after": 241,
    }
    assert activation["adr029_complete"] is False
    assert "pending" in activation["final_performance_evidence"]

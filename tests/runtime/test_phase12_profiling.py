from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import pytest

from witwin.channel_native.runtime import profiling


_ROOT = Path(__file__).resolve().parents[2]
_CONTRACT = _ROOT / "benchmarks" / "phase13_phase12_profile_contract.json"


def _function(path: str, name: str) -> ast.FunctionDef:
    tree = ast.parse((_ROOT / path).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing function {name} in {path}")


def _profile_enum_members(node: ast.AST, helper: str) -> set[str]:
    members: set[str] = set()
    for call in (item for item in ast.walk(node) if isinstance(item, ast.Call)):
        if not isinstance(call.func, ast.Name) or call.func.id != helper:
            continue
        if len(call.args) != 1 or not isinstance(call.args[0], ast.Attribute):
            raise AssertionError(f"{helper} must receive one semantic enum member")
        members.add(call.args[0].attr)
    return members


def test_cuda_profile_range_is_balanced_on_failure(monkeypatch) -> None:
    calls: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        profiling.torch.cuda.nvtx,
        "range_push",
        lambda name: calls.append(("push", name)),
    )
    monkeypatch.setattr(
        profiling.torch.cuda.nvtx,
        "range_pop",
        lambda: calls.append(("pop", None)),
    )

    with pytest.raises(RuntimeError, match="sentinel"):
        with profiling.cuda_profile_range(
            profiling.CudaProfileRange.DIFFRACTION_TOTAL_STAGE
        ):
            raise RuntimeError("sentinel")

    assert calls == [
        ("push", "witwin.channel_native:diffraction_total_stage"),
        ("pop", None),
    ]


def test_profiled_cuda_range_preserves_callable_contract(monkeypatch) -> None:
    monkeypatch.setattr(profiling.torch.cuda.nvtx, "range_push", lambda _name: None)
    monkeypatch.setattr(profiling.torch.cuda.nvtx, "range_pop", lambda: None)

    def operation(value: int, *, increment: int = 1) -> int:
        """Example operation."""

        return value + increment

    wrapped = profiling.profiled_cuda_range(
        profiling.CudaProfileRange.ENUMERATED_PENETRATION_DISCOVERY
    )(operation)

    assert wrapped(2, increment=3) == 5
    assert wrapped.__name__ == operation.__name__
    assert wrapped.__doc__ == operation.__doc__
    assert inspect.signature(wrapped) == inspect.signature(operation)


def test_cuda_profile_mark_emits_only_the_semantic_payload(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(profiling.torch.cuda.nvtx, "mark", calls.append)

    profiling.cuda_profile_mark(
        profiling.CudaProfileMark.DIFFRACTION_EXPORTER_REQUEST
    )

    assert calls == ["witwin.channel_native:diffraction_exporter_request"]


def test_profiling_owner_has_no_tensor_or_cuda_execution_calls() -> None:
    source = (
        _ROOT / "src" / "witwin" / "channel_native" / "runtime" / "profiling.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    torch_calls = {
        ast.unparse(call.func)
        for call in ast.walk(tree)
        if isinstance(call, ast.Call) and ast.unparse(call.func).startswith("torch.")
    }

    assert torch_calls == {
        "torch.cuda.nvtx.mark",
        "torch.cuda.nvtx.range_pop",
        "torch.cuda.nvtx.range_push",
    }
    assert all(
        forbidden not in source
        for forbidden in (
            ".cpu(",
            ".item(",
            ".numpy(",
            ".tolist(",
            "synchronize(",
        )
    )


def test_profile_contract_matches_closed_semantic_name_sets() -> None:
    contract = json.loads(_CONTRACT.read_text(encoding="utf-8"))
    assert contract["schema"] == {
        "name": "witwin.channel_native.phase13-phase12-profile-contract",
        "version": 1,
    }
    assert contract["steady_repeats"] == 7
    assert tuple(contract["groups"]) == (
        "enumerated_penetration",
        "montecarlo_penetration",
        "diffraction",
    )

    range_names = {item.value for item in profiling.CudaProfileRange}
    mark_names = {item.value for item in profiling.CudaProfileMark}
    observed_ranges: set[str] = set()
    observed_marks: set[str] = set()
    expected_ncu_families = {
        "enumerated_penetration": ["segment_penetration"],
        "montecarlo_penetration": ["segment_penetration"],
        "diffraction": ["deterministic_diffraction_pair_reduce"],
    }
    accepted_owner_records = "\n".join(
        (_ROOT / "docs" / "dev" / "standards" / filename).read_text(
            encoding="utf-8"
        )
        for filename in (
            "adr-027-batched-segment-penetration.md",
            "adr-030-deterministic-diffraction-pair-reduction.md",
        )
    )
    for group_name, group in contract["groups"].items():
        assert group["target_timing_range"] in range_names
        assert group["solver_entrypoint"].startswith("witwin.channel_native.")
        assert group["ncu_kernel_family_match"] == "case_sensitive_substring"
        assert set(group["variants"]) == {"baseline", "candidate"}
        for variant_name, variant in group["variants"].items():
            ncu_families = variant["ncu_kernel_families"]
            assert ncu_families == sorted(set(ncu_families))
            assert ncu_families == (
                []
                if variant_name == "baseline"
                else expected_ncu_families[group_name]
            )
            assert all(
                family.isascii()
                and family == family.casefold()
                and family.replace("_", "").isalnum()
                and family in accepted_owner_records
                for family in ncu_families
            )
            required = set(variant["required_ranges"])
            forbidden = set(variant["forbidden_ranges"])
            known = set(variant["known_range_multiplicity_per_solve"])
            assert required <= range_names
            assert forbidden <= range_names
            assert not required & forbidden
            assert known <= required
            assert all(
                isinstance(value, int) and value > 0
                for value in variant["known_range_multiplicity_per_solve"].values()
            )
            observed_ranges.update(required)
            observed_ranges.update(forbidden)
            observed_marks.update(variant["required_markers"])

    assert observed_ranges == range_names
    assert observed_marks == mark_names
    diffraction = contract["groups"]["diffraction"]["variants"]
    assert diffraction["baseline"]["known_range_multiplicity_per_solve"] == {
        "witwin.channel_native:diffraction_exporter": 13,
        "witwin.channel_native:diffraction_topology_packing": 13,
        "witwin.channel_native:diffraction_total_stage": 1,
    }
    assert diffraction["candidate"]["known_range_multiplicity_per_solve"] == {
        "witwin.channel_native:diffraction_exporter": 2,
        "witwin.channel_native:diffraction_total_stage": 1,
    }
    assert all(
        token not in name.casefold()
        for name in range_names | mark_names
        for token in (
            "phase12",
            ":v2",
            "_v2",
            "wip",
            "legacy",
            "temporary",
            ":candidate",
        )
    )


def test_baseline_owners_emit_only_real_profile_annotations() -> None:
    enumerated = _function(
        "src/witwin/channel_native/propagation/enumerated/transmission.py",
        "_transmission_topology",
    )
    assert _profile_enum_members(enumerated, "profiled_cuda_range") == {
        "ENUMERATED_PENETRATION_DISCOVERY"
    }

    montecarlo = _function(
        "src/witwin/channel_native/montecarlo/events/transmission.py",
        "straight_transmission_chains",
    )
    assert _profile_enum_members(montecarlo, "profiled_cuda_range") == {
        "MONTECARLO_BASIC_PENETRATION_DISCOVERY"
    }
    assert _profile_enum_members(montecarlo, "cuda_profile_mark") == {
        "OPTIX_TRAVERSAL"
    }

    diffraction = _function(
        "src/witwin/channel_native/propagation/enumerated/diffraction.py",
        "_diffraction_topology_order1",
    )
    assert _profile_enum_members(diffraction, "profiled_cuda_range") == {
        "DIFFRACTION_TOTAL_STAGE"
    }
    assert _profile_enum_members(diffraction, "cuda_profile_range") == {
        "DIFFRACTION_EXPORTER",
        "DIFFRACTION_TOPOLOGY_PACKING",
    }
    assert "CAPACITY_STATUS" not in ast.dump(diffraction)
    assert "DIFFRACTION_PAIR_REDUCER" not in ast.dump(diffraction)

    transmission_query = _function(
        "src/witwin/channel_native/propagation/geometry/transmission.py",
        "query_transmission_closest_hit",
    )
    assert _profile_enum_members(transmission_query, "cuda_profile_mark") == {
        "OPTIX_TRAVERSAL"
    }

    diffraction_query = _function(
        "src/witwin/channel_native/propagation/geometry/diffraction.py",
        "query_diffraction_order1",
    )
    assert _profile_enum_members(diffraction_query, "cuda_profile_mark") == {
        "OPTIX_TRAVERSAL",
        "DIFFRACTION_EXPORTER_REQUEST",
    }

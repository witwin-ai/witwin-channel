from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

from witwin.channel.propagation.topology.kernels import compaction
from witwin.channel.propagation.topology.concatenate import (
    _canonical_selection_order,
)
from witwin.channel.propagation.topology.kernels.compaction import (
    enumerated_canonical_capacity_select,
)
from witwin.channel.runtime.capacity import (
    CapacityFailureBit,
    create_capacity_failure_state,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _block() -> dict[str, torch.Tensor]:
    device = torch.device("cuda")
    component = torch.tensor(
        [3, 1, 2, 4, 3, 3, 1, 2], device=device, dtype=torch.int32
    )
    depth = torch.tensor([2, 1, 1, 2, 2, 2, 1, 1], device=device, dtype=torch.int32)
    rx_id = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1], device=device, dtype=torch.int32)
    sequence = torch.tensor(
        [[5, 7], [5, -1], [-1, -1], [7, 5], [5, 7], [5, 8], [5, -1], [-1, -1]],
        device=device,
        dtype=torch.int32,
    )
    edge_id = torch.tensor([7, -1, 7, 7, 7, 8, -1, 7], device=device, dtype=torch.int32)
    count = int(component.numel())
    length = torch.arange(1, count + 1, device=device, dtype=torch.float32)
    return {
        "valid": torch.ones(count, device=device, dtype=torch.bool),
        "tx_id": torch.zeros(count, device=device, dtype=torch.int32),
        "rx_id": rx_id,
        "depth": depth,
        "component_id": component,
        "primitive_id": torch.where(
            component == 2, torch.full_like(component, -1), sequence[:, 0]
        ),
        "edge_id": edge_id,
        "primitive_sequence": sequence,
        "path_length_m": length,
    }


def _select(
    block: dict[str, torch.Tensor],
    *,
    max_paths: int | None = None,
    scope: str = "per_pair",
):
    return _select_layout(
        block,
        num_tx=1,
        num_rx=2,
        max_paths=max_paths,
        scope=scope,
    )


def _select_layout(
    block: dict[str, torch.Tensor],
    *,
    num_tx: int,
    num_rx: int,
    max_paths: int | None,
    scope: str,
):
    state = create_capacity_failure_state(block["valid"])
    selection = enumerated_canonical_capacity_select(
        failure_state=state,
        valid=block["valid"],
        tx_id=block["tx_id"],
        rx_id=block["rx_id"],
        depth=block["depth"],
        component_id=block["component_id"],
        primitive_id=block["primitive_id"],
        edge_id=block["edge_id"],
        primitive_sequence=block["primitive_sequence"],
        path_length_m=block["path_length_m"],
        pair_count=num_tx * num_rx,
        num_tx=num_tx,
        num_rx=num_rx,
        max_paths=max_paths,
        max_paths_scope=scope,
    )
    return state, selection


def _live_source_order(
    block: dict[str, torch.Tensor],
    *,
    max_paths: int | None,
    scope: str,
    tx_count: int = 1,
) -> torch.Tensor:
    source = torch.nonzero(block["valid"], as_tuple=False).reshape(-1)
    compact = {
        name: value[source]
        for name, value in block.items()
    }
    compact["valid"] = torch.ones_like(compact["valid"])
    order = _canonical_selection_order(
        compact,
        tx_count=tx_count,
        max_depth=int(compact["primitive_sequence"].shape[1]),
        max_paths=max_paths,
        max_paths_scope=scope,
    )
    return source[order]


@pytest.mark.parametrize("scope", ["global", "per_pair"])
@pytest.mark.parametrize("max_paths", [None, 2, 5])
def test_selector_is_exact_live_order(scope: str, max_paths: int | None) -> None:
    block = _block()
    _state, selected = _select(block, max_paths=max_paths, scope=scope)
    expected = _live_source_order(block, max_paths=max_paths, scope=scope)
    count = int(expected.numel())

    torch.testing.assert_close(selected.selected_row_index[:count], expected)
    assert selected.selected_row_index[count:].tolist() == [-1] * (8 - count)
    assert selected.valid.tolist() == [True] * count + [False] * (8 - count)
    assert selected.num_selected.tolist() == [count]
    expected_counts = torch.bincount(
        block["rx_id"][expected].to(torch.int64), minlength=2
    ).to(torch.int32)
    torch.testing.assert_close(selected.num_paths, expected_counts)


def test_multi_endpoint_pair_order_caps_and_source_stability_are_frozen() -> None:
    pair = torch.tensor(
        [5, 0, 4, 2, 1, 3, 0, 5, 1, 3, 2, 4],
        device="cuda",
        dtype=torch.int32,
    )
    object_id = torch.tensor(
        [1, 4, 6, 8, 3, 5, 2, 1, 9, 0, 4, 2],
        device="cuda",
        dtype=torch.int32,
    )
    count = int(pair.numel())
    block = {
        "valid": torch.ones(count, device="cuda", dtype=torch.bool),
        "tx_id": pair.remainder(3),
        "rx_id": torch.div(pair, 3, rounding_mode="floor"),
        "depth": torch.ones(count, device="cuda", dtype=torch.int32),
        "component_id": torch.ones(count, device="cuda", dtype=torch.int32),
        "primitive_id": object_id.clone(),
        "edge_id": torch.full((count,), -1, device="cuda", dtype=torch.int32),
        "primitive_sequence": torch.stack(
            (object_id, torch.full_like(object_id, -1)), dim=1
        ),
        "path_length_m": torch.ones(count, device="cuda", dtype=torch.float32),
    }

    _state, per_pair = _select_layout(
        block,
        num_tx=3,
        num_rx=2,
        max_paths=1,
        scope="per_pair",
    )
    assert per_pair.selected_row_index[:6].tolist() == [6, 4, 10, 9, 11, 0]
    assert per_pair.num_paths.tolist() == [1, 1, 1, 1, 1, 1]

    _state, global_selection = _select_layout(
        block,
        num_tx=3,
        num_rx=2,
        max_paths=4,
        scope="global",
    )
    assert global_selection.selected_row_index[:4].tolist() == [6, 1, 4, 8]
    assert global_selection.num_paths.tolist() == [2, 2, 0, 0, 0, 0]


@pytest.mark.parametrize("seed", [0, 1, 7])
@pytest.mark.parametrize(("scope", "max_paths"), [("global", 11), ("per_pair", 4)])
def test_randomized_multi_pair_selector_is_exact_live_oracle(
    seed: int, scope: str, max_paths: int
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    count = 96
    components = torch.tensor([1, 2, 3, 4, 5, 7], device="cuda", dtype=torch.int32)
    component = components[
        torch.randint(0, 6, (count,), device="cuda", generator=generator)
    ]
    depth = torch.where(
        (component == 3) | (component == 4) | (component == 7),
        torch.full_like(component, 2),
        torch.ones_like(component),
    )
    sequence = torch.randint(
        0, 9, (count, 2), device="cuda", dtype=torch.int32, generator=generator
    )
    one_slot = (component == 1) | (component == 2) | (component == 5)
    sequence[one_slot, 1] = -1
    sequence[component == 2, 0] = -1
    edge_id = torch.randint(
        0, 7, (count,), device="cuda", dtype=torch.int32, generator=generator
    )
    edge_id[component != 2] = -1
    primitive_id = sequence[:, 0].clone()
    primitive_id[component == 2] = -1
    valid = torch.rand(count, device="cuda", generator=generator) > 0.2
    valid[0] = True
    block = {
        "valid": valid,
        "tx_id": torch.randint(
            0, 3, (count,), device="cuda", dtype=torch.int32, generator=generator
        ),
        "rx_id": torch.randint(
            0, 2, (count,), device="cuda", dtype=torch.int32, generator=generator
        ),
        "depth": depth,
        "component_id": component,
        "primitive_id": primitive_id,
        "edge_id": edge_id,
        "primitive_sequence": sequence,
        "path_length_m": torch.randint(
            1, 6, (count,), device="cuda", dtype=torch.int32, generator=generator
        ).to(torch.float32),
    }
    for name in (
        "tx_id",
        "rx_id",
        "depth",
        "component_id",
        "primitive_id",
        "edge_id",
        "primitive_sequence",
        "path_length_m",
    ):
        block[name][-12:] = block[name][:12]
    block["valid"][-12:] = block["valid"][:12]
    invalid = ~block["valid"]
    poison = torch.iinfo(torch.int32).min
    for name in (
        "tx_id",
        "rx_id",
        "depth",
        "component_id",
        "primitive_id",
        "edge_id",
        "primitive_sequence",
    ):
        block[name][invalid] = poison
    block["path_length_m"][invalid] = float("nan")

    _state, selected = _select_layout(
        block,
        num_tx=3,
        num_rx=2,
        max_paths=max_paths,
        scope=scope,
    )
    expected = _live_source_order(
        block, max_paths=max_paths, scope=scope, tx_count=3
    )
    selected_count = int(expected.numel())
    torch.testing.assert_close(
        selected.selected_row_index[:selected_count], expected
    )
    assert selected.selected_row_index[selected_count:].tolist() == [-1] * (
        count - selected_count
    )
    pair = block["rx_id"][expected].to(torch.int64) * 3 + block["tx_id"][
        expected
    ].to(torch.int64)
    torch.testing.assert_close(
        selected.num_paths, torch.bincount(pair, minlength=6).to(torch.int32)
    )


def test_selector_omits_invalid_poison_before_every_payload_read() -> None:
    baseline = _block()
    poison_rows = torch.tensor([1, 6], device="cuda")
    baseline["valid"][poison_rows] = False
    poison = torch.iinfo(torch.int32).min
    for name in (
        "tx_id",
        "rx_id",
        "depth",
        "component_id",
        "primitive_id",
        "edge_id",
        "primitive_sequence",
    ):
        baseline[name][poison_rows] = poison
    baseline["path_length_m"][poison_rows] = float("nan")

    state, selected = _select(baseline)
    expected = _live_source_order(baseline, max_paths=None, scope="per_pair")
    count = int(expected.numel())
    torch.testing.assert_close(selected.selected_row_index[:count], expected)
    assert selected.num_selected.tolist() == [count]
    assert state.bits.tolist() == [0]


def test_shortest_and_full_stable_tie_break_match_live() -> None:
    block = _block()
    block["path_length_m"][4] = block["path_length_m"][0]
    block["primitive_id"][4] = block["primitive_id"][0]
    block["edge_id"][4] = block["edge_id"][0]
    state, selected = _select(block)
    expected = _live_source_order(block, max_paths=None, scope="per_pair")
    torch.testing.assert_close(
        selected.selected_row_index[: expected.numel()], expected
    )
    assert state.bits.tolist() == [0]


def test_single_valid_nan_preserves_live_k_one_special_case() -> None:
    block = _block()
    block["valid"].fill_(False)
    block["valid"][3] = True
    block["path_length_m"][3] = float("nan")
    _state, selected = _select(block)
    assert selected.selected_row_index.tolist() == [3, -1, -1, -1, -1, -1, -1, -1]
    assert selected.num_selected.tolist() == [1]


def test_global_and_per_pair_caps_are_applied_after_dedup() -> None:
    block = _block()
    _state, per_pair = _select(block, max_paths=2, scope="per_pair")
    assert per_pair.num_paths.tolist() == [2, 2]
    _state, global_selection = _select(block, max_paths=3, scope="global")
    assert global_selection.num_selected.tolist() == [3]
    assert global_selection.num_paths.tolist() == [3, 0]


def test_selector_does_not_apply_public_pair_capacity_overflow() -> None:
    block = _block()
    state, selected = _select(block, max_paths=None, scope="global")
    expected = _live_source_order(block, max_paths=None, scope="global")
    count = int(expected.numel())

    assert state.bits.tolist() == [0]
    torch.testing.assert_close(selected.selected_row_index[:count], expected)
    assert selected.valid.tolist() == [True] * count + [False] * (8 - count)
    assert selected.num_selected.tolist() == [count]
    assert max(selected.num_paths.tolist()) > 2


def test_selector_contract_needs_no_public_result_capacity() -> None:
    parameters = inspect.signature(enumerated_canonical_capacity_select).parameters
    assert "path_capacity_per_pair" not in parameters

    for scope in ("global", "per_pair"):
        block = _block()
        state, selected = _select(block, max_paths=100, scope=scope)
        expected = _live_source_order(block, max_paths=100, scope=scope)
        count = int(expected.numel())
        assert state.bits.tolist() == [0]
        torch.testing.assert_close(selected.selected_row_index[:count], expected)
        assert selected.candidate_capacity == 8
        assert selected.num_tx == 1
        assert selected.num_rx == 2
        assert selected.pair_count == selected.num_tx * selected.num_rx
        assert not hasattr(selected, "path_capacity_per_pair")
        assert not hasattr(selected, "overflow")


def test_prior_failure_and_valid_bad_endpoint_are_inert() -> None:
    block = _block()
    state = create_capacity_failure_state(block["valid"])
    state.bits.fill_(int(CapacityFailureBit.DIFFRACTION_STATE_OVERFLOW))
    selected = enumerated_canonical_capacity_select(
        failure_state=state,
        valid=block["valid"],
        tx_id=block["tx_id"],
        rx_id=block["rx_id"],
        depth=block["depth"],
        component_id=block["component_id"],
        primitive_id=block["primitive_id"],
        edge_id=block["edge_id"],
        primitive_sequence=block["primitive_sequence"],
        path_length_m=block["path_length_m"],
        pair_count=2,
        num_tx=1,
        num_rx=2,
        max_paths=None,
        max_paths_scope="per_pair",
    )
    assert selected.valid.tolist() == [False] * 8
    assert state.bits.tolist() == [int(CapacityFailureBit.DIFFRACTION_STATE_OVERFLOW)]

    block = _block()
    block["tx_id"][0] = 4
    state, selected = _select(block)
    assert state.bits.tolist() == [int(CapacityFailureBit.PAIR_CONTRACT_ERROR)]
    assert selected.valid.tolist() == [False] * 8


def test_zero_candidate_capacity_and_current_stream() -> None:
    block = _block()
    block = {name: value[:0] for name, value in block.items()}
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        state, selected = _select(block)
    stream.synchronize()
    assert state.bits.tolist() == [0]
    assert selected.selected_row_index.shape == (0,)
    assert selected.valid.shape == (0,)
    assert selected.num_selected.tolist() == [0]
    assert selected.num_paths.tolist() == [0, 0]


def test_selector_outputs_are_discrete_under_forward_and_reverse_ad() -> None:
    block = _block()
    block["path_length_m"].requires_grad_(True)
    _state, selected = _select(block)
    assert not selected.selected_row_index.requires_grad
    assert not selected.valid.requires_grad
    assert not selected.num_selected.requires_grad
    with torch.autograd.forward_ad.dual_level():
        block["path_length_m"] = torch.autograd.forward_ad.make_dual(
            block["path_length_m"], torch.ones_like(block["path_length_m"])
        )
        _state, dual_selected = _select(block)
        assert torch.autograd.forward_ad.unpack_dual(
            dual_selected.selected_row_index
        ).tangent is None


def test_missing_native_symbol_fails_without_fallback(monkeypatch) -> None:
    block = _block()
    state = create_capacity_failure_state(block["valid"])

    def missing(name: str):
        raise RuntimeError(f"missing required native symbol: {name}")

    monkeypatch.setattr(compaction, "_required_native_op", missing)
    with pytest.raises(RuntimeError, match="enumerated_canonical_capacity_select"):
        enumerated_canonical_capacity_select(
            failure_state=state,
            valid=block["valid"],
            tx_id=block["tx_id"],
            rx_id=block["rx_id"],
            depth=block["depth"],
            component_id=block["component_id"],
            primitive_id=block["primitive_id"],
            edge_id=block["edge_id"],
            primitive_sequence=block["primitive_sequence"],
            path_length_m=block["path_length_m"],
            pair_count=2,
            num_tx=1,
            num_rx=2,
            max_paths=None,
            max_paths_scope="per_pair",
        )


def test_selector_has_no_fallback_sync_or_intermediate_trap() -> None:
    root = Path(__file__).resolve().parents[3]
    native = (
        root
        / "native/channel/kernels/enumerated_canonical_capacity_select.cu"
    ).read_text(encoding="utf-8")
    facade = (
        root
        / "src/witwin/channel/propagation/topology/kernels/compaction.py"
    ).read_text(encoding="utf-8")
    live_engine = (
        root / "src/witwin/channel/propagation/enumerated/engine.py"
    ).read_text(encoding="utf-8")
    for forbidden in (
        "cudaMemcpy",
        "cudaStreamSynchronize",
        ".item(",
        ".cpu(",
        ".numpy(",
        "thrust::",
        'asm volatile("trap;")',
    ):
        assert forbidden not in native
    assert 'required_symbol as _required_native_op' in facade
    assert "enumerated_canonical_capacity_select" in facade
    assert "path_capacity_per_pair" not in native
    assert "kPairCapacityOverflow" not in native
    assert 'result["overflow"]' not in native
    assert "enumerated_canonical_capacity_select" not in live_engine

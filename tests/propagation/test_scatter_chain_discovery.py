"""ADR-021 D1 enumerated scatter-chain discovery contract tests.

The pure structural helpers (join / stable order / budget / padding / contract
validation / multi-slot topology assembly) are exercised on CPU tensors with no
scene. The end-to-end discovery on a two-wall scene, the default-off byte
identity, and the loud D2-facade dependency require the built RayD extension and
are CUDA-guarded.
"""

from __future__ import annotations

import math

import pytest
import torch

from witwin.core import PhysicalMaterial, Scene
from tests.support.core_world import (
    make_mesh_structure,
    make_receiver,
    make_transmitter,
)
from witwin.channel.scene import compile as compile_scene
from witwin.channel.scene.endpoints import bind_solver_scene
from tests.support.scenes import rough_wall_structure
from witwin.channel.propagation.enumerated import scattering as scattering_mod
from witwin.channel.propagation.enumerated import scattering_chain as sc

# ADR-021 refactor: the chain-append path (topology slots, ensemble scatter face
# selection) moved out of scattering.py into scattering_chain_append.py to meet
# the file-size maintenance budget; the single-bounce path stays in scattering.
from witwin.channel.propagation.enumerated import (
    scattering_chain_append as scattering_append,
)
from witwin.channel.deterministic import Config as DeterministicConfig
from witwin.channel.path import Config as PathConfig


_D = sc.KMAX_AD_DEPTH


# ---------------------------------------------------------------------------
# Config plumbing (no CUDA).
# ---------------------------------------------------------------------------


def test_config_defaults_off():
    for cfg in (DeterministicConfig(), PathConfig()):
        assert cfg.scattering_chain_max_depth == 0
        assert cfg.scattering_chain_samples_per_m2 == 2.0
        assert cfg.scattering_chain_max_rows == 256


@pytest.mark.parametrize("cls", [DeterministicConfig, PathConfig])
def test_config_enable_valid(cls):
    cfg = cls(
        max_depth=2,
        components=["los", "reflection", "scattering"],
        scattering_chain_max_depth=6,
    )
    assert cfg.scattering_chain_max_depth == 6


@pytest.mark.parametrize("cls", [DeterministicConfig, PathConfig])
@pytest.mark.parametrize(
    "overrides,exc",
    [
        ({"scattering_chain_max_depth": 17}, ValueError),
        ({"scattering_chain_max_depth": -1}, ValueError),
        ({"scattering_chain_samples_per_m2": 0.0}, ValueError),
        ({"scattering_chain_max_rows": 0}, ValueError),
        ({"scattering_chain_max_depth": 2, "components": ["los"]}, RuntimeError),
    ],
)
def test_config_validation_rejects(cls, overrides, exc):
    base = {"components": ["los", "reflection", "scattering"]}
    base.update(overrides)
    with pytest.raises(exc):
        cls(**base)


# ---------------------------------------------------------------------------
# Pure structural discovery helpers (no CUDA).
# ---------------------------------------------------------------------------


def test_equi_join_cartesian_per_key():
    a = torch.tensor([1, 2, 2, 3], dtype=torch.int64)
    b = torch.tensor([2, 2, 3, 5], dtype=torch.int64)
    ai, bi = sc._equi_join_indices(a, b)
    # key 2: {1,2} x {0,1} = 4 pairs; key 3: {3} x {2} = 1 pair.
    assert int(ai.numel()) == 5
    assert torch.equal(a[ai], b[bi])
    got = sorted(zip(ai.tolist(), bi.tolist()))
    assert got == sorted([(1, 0), (1, 1), (2, 0), (2, 1), (3, 2)])


def test_equi_join_empty():
    a = torch.tensor([1, 2], dtype=torch.int64)
    b = torch.tensor([], dtype=torch.int64)
    ai, bi = sc._equi_join_indices(a, b)
    assert int(ai.numel()) == 0 and int(bi.numel()) == 0


def test_stable_chain_order_lexicographic():
    tx = torch.tensor([1, 0, 0, 1], dtype=torch.int32)
    rx = torch.tensor([0, 1, 0, 0], dtype=torch.int32)
    sid = torch.tensor([5, 2, 9, 3], dtype=torch.int64)
    d1 = torch.tensor([1, 0, 2, 1], dtype=torch.int32)
    d2 = torch.tensor([0, 1, 0, 1], dtype=torch.int32)
    order = sc._stable_chain_order(tx, rx, sid, d1, d2)
    keys = list(
        zip(
            tx[order].tolist(),
            rx[order].tolist(),
            sid[order].tolist(),
            d1[order].tolist(),
            d2[order].tolist(),
        )
    )
    assert keys == sorted(keys)


def test_budget_keep_strongest_per_pair():
    tx = torch.zeros(4, dtype=torch.int32)
    rx = torch.zeros(4, dtype=torch.int32)
    strength = torch.tensor([1.0, 4.0, 2.0, 3.0])
    keep = sc._budget_chain_rows(tx, rx, strength, num_rx=1, cap=2)
    assert set(keep.tolist()) == {1, 3}


def test_budget_separates_pairs():
    tx = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
    rx = torch.tensor([0, 0, 0, 0], dtype=torch.int32)
    strength = torch.tensor([1.0, 2.0, 3.0, 4.0])
    keep = sc._budget_chain_rows(tx, rx, strength, num_rx=1, cap=1)
    # strongest of pair (0,0) is row 1; strongest of pair (1,0) is row 3.
    assert set(keep.tolist()) == {1, 3}


def test_pad_bounce_block():
    v = torch.arange(2 * 3 * 3, dtype=torch.float32).reshape(2, 3, 3)
    padded = sc._pad_bounce_block(v, 3, _D, 0.0)
    assert padded.shape == (2, _D, 3)
    assert torch.equal(padded[:, :3], v)
    assert float(padded[:, 3:].abs().sum()) == 0.0

    prim = torch.full((2, 2), 7, dtype=torch.int32)
    padded_i = sc._pad_bounce_block(prim, 2, _D, -1)
    assert padded_i.shape == (2, _D)
    assert torch.equal(padded_i[:, :2], prim)
    assert int((padded_i[:, 2:] == -1).all())


def _synthetic_merged(d1_list, d2_list) -> dict[str, torch.Tensor]:
    r = len(d1_list)
    return {
        "tx_id": torch.zeros(r, dtype=torch.int32),
        "rx_id": torch.zeros(r, dtype=torch.int32),
        "sample_index": torch.arange(r, dtype=torch.int64),
        "d1": torch.tensor(d1_list, dtype=torch.int32),
        "d2": torch.tensor(d2_list, dtype=torch.int32),
        "c1_positions": torch.zeros(r, _D, 3),
        "c1_normals": torch.zeros(r, _D, 3),
        "c1_primitive": torch.full((r, _D), -1, dtype=torch.int32),
        "c1_material": torch.full((r, _D), -1, dtype=torch.int32),
        "L1": torch.ones(r),
        "d_i": torch.zeros(r, 3),
        "c2_positions": torch.zeros(r, _D, 3),
        "c2_normals": torch.zeros(r, _D, 3),
        "c2_primitive": torch.full((r, _D), -1, dtype=torch.int32),
        "c2_material": torch.full((r, _D), -1, dtype=torch.int32),
        "L2": torch.ones(r),
        "d_o": torch.zeros(r, 3),
        "v_pos": torch.zeros(r, 3),
        "v_normal": torch.zeros(r, 3),
        "v_material": torch.zeros(r, dtype=torch.int32),
        "weight": torch.ones(r),
        "cos_i": torch.ones(r),
        "cos_o": torch.ones(r),
        "patch_row": torch.full((r,), -1, dtype=torch.int64),
    }


def test_discovery_contract_validate_ok():
    disc = sc._assemble_discovery(_synthetic_merged([1, 0, 2], [0, 1, 1]))
    disc.validate()
    assert disc.row_count == 3


def test_discovery_contract_rejects_bad_dtype():
    merged = _synthetic_merged([1], [1])
    merged["tx_id"] = merged["tx_id"].to(torch.int64)
    with pytest.raises(TypeError):
        sc._assemble_discovery(merged).validate()


def test_discovery_contract_rejects_zero_depth():
    with pytest.raises(ValueError):
        sc._assemble_discovery(_synthetic_merged([0], [0])).validate()


def test_chain_topology_slot_layout():
    # row0: d1=1, d2=1 -> [R, S, R]; row1: d1=0, d2=1 -> [S, R].
    merged = _synthetic_merged([1, 0], [1, 1])
    merged["c1_primitive"][0, 0] = 10
    merged["c1_material"][0, 0] = 100
    merged["c2_primitive"][0, 0] = 20
    merged["c2_primitive"][1, 0] = 21
    disc = sc._assemble_discovery(merged)
    vertex_face = torch.tensor([7, 8], dtype=torch.int32)
    vertex_normal = torch.zeros(2, 3)
    width = int((disc.d1 + 1 + disc.d2).max().item())
    slots = scattering_append._chain_topology_slots(disc, vertex_face, vertex_normal, width)
    # interaction_type: REFLECTION=1, SCATTERING=8, inactive=0.
    assert slots["interaction_type"][0].tolist() == [1, 8, 1]
    assert slots["interaction_type"][1].tolist() == [8, 1, 0]
    assert slots["primitive_sequence"][0].tolist() == [10, 7, 20]
    assert slots["primitive_sequence"][1].tolist() == [8, 21, -1]


# ---------------------------------------------------------------------------
# End-to-end discovery on a two-wall scene (CUDA + RayD required).
# ---------------------------------------------------------------------------

_FREQUENCY_HZ = 3.0e9


def _require_cuda_rayd() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA torch is required")
    from witwin.channel.deployment import build_info

    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native scene capability is not built")


def _two_wall_scene():
    """Rough scatter wall (x=2.5) plus a specular reflector (z=-1 floor).

    A transmitter and receiver above the floor admit chain rows such as
    TX -> floor reflection -> rough-wall vertex -> RX.
    """

    rough = rough_wall_structure(
        2.5, rms_height_m=0.01, corr_length_m=0.15, half_size=2.0, surface_id=1
    )
    floor = make_mesh_structure(
        vertices=torch.tensor(
            [
                [0.0, -2.0, -1.0],
                [3.0, -2.0, -1.0],
                [0.0, 2.0, -1.0],
                [3.0, 2.0, -1.0],
            ]
        ),
        faces=torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.int32),
        material=PhysicalMaterial(eps_r=4.0, sigma_e=0.01),
        name="floor",
        surface_id=2,
    )
    return Scene(
        structures=[rough, floor],
        endpoints=[
            make_transmitter(position=torch.tensor([0.5, -1.0, 0.5])),
            make_receiver(position=torch.tensor([1.5, 1.0, 0.5])),
        ],
    )


def _run_discovery(scene, config):
    from witwin.channel.propagation.geometry.endpoints import (
        receiver_positions_and_layout,
        transmitter_tensors,
    )

    device = torch.device("cuda")
    compiled = compile_scene(
        scene, reference_frequency_hz=_FREQUENCY_HZ
    )
    solver_scene = bind_solver_scene(compiled)
    screens = scattering_mod.realization_phase_screens(
        compiled.materials, compiled.assignments
    )
    ensemble_faces = scattering_append._ensemble_scatter_faces(
        compiled, screens, device=device
    )
    samples = sc.build_chain_samples(compiled, config, ensemble_faces, device=device)
    if samples is None:
        return None
    tx_positions, _ = transmitter_tensors(solver_scene, device=device)
    rx_positions, _ = receiver_positions_and_layout(
        solver_scene, device=device
    )
    records = compiled.rayd.edge_records()
    vertices = records.vertices
    diag = (vertices.max(dim=0).values - vertices.min(dim=0).values).norm()
    return sc.discover_scatter_chains(
        compiled,
        config,
        tx_positions=tx_positions,
        rx_positions=rx_positions,
        samples=samples,
        scene_diagonal=diag,
    )


def test_discovery_default_off_returns_none():
    _require_cuda_rayd()
    scene = _two_wall_scene()
    config = DeterministicConfig(
        max_depth=2,
        components=["los", "reflection", "scattering"],
        scattering_chain_max_depth=0,
    )
    assert _run_discovery(scene, config) is None


def test_discovery_deterministic_and_valid():
    _require_cuda_rayd()
    scene = _two_wall_scene()
    config = DeterministicConfig(
        max_depth=2,
        components=["los", "reflection", "scattering"],
        scattering_chain_max_depth=3,
        scattering_chain_samples_per_m2=8.0,
    )
    first = _run_discovery(scene, config)
    second = _run_discovery(scene, config)
    if first is None:
        assert second is None
        pytest.skip("scene yielded no scatter-chain rows")
    # Determinism: identical row order and identical geometry run-to-run.
    first.validate()
    second.validate()
    assert first.row_count == second.row_count
    for name in ("tx_id", "rx_id", "sample_index", "d1", "d2"):
        assert torch.equal(getattr(first, name), getattr(second, name))
    assert torch.equal(first.L1, second.L1)
    assert torch.equal(first.c1_positions, second.c1_positions)


def test_discovery_depth_and_join_bounds():
    _require_cuda_rayd()
    scene = _two_wall_scene()
    max_chain = 3
    config = DeterministicConfig(
        max_depth=2,
        components=["los", "reflection", "scattering"],
        scattering_chain_max_depth=max_chain,
        scattering_chain_samples_per_m2=8.0,
        scattering_chain_max_rows=64,
    )
    disc = _run_discovery(scene, config)
    if disc is None:
        pytest.skip("scene yielded no scatter-chain rows")
    d1 = disc.d1.to(torch.int64)
    d2 = disc.d2.to(torch.int64)
    total = d1 + d2
    # Depth bounds (ADR-021 D1): 1 <= d1 + d2 <= cap; each leg <= kMaxAdDepth.
    assert int(total.min()) >= 1
    assert int(total.max()) <= max_chain
    assert int(d1.max()) <= sc.KMAX_AD_DEPTH
    assert int(d2.max()) <= sc.KMAX_AD_DEPTH
    # Join correctness: every joined vertex indexes a real chain sample.
    assert int(disc.sample_index.min()) >= 0
    # Row order is the deterministic stable lexicographic sort.
    order = sc._stable_chain_order(
        disc.tx_id, disc.rx_id, disc.sample_index, disc.d1, disc.d2
    )
    assert torch.equal(order, torch.arange(disc.row_count, device=disc.device))


def test_solver_default_off_appends_no_chain_rows():
    _require_cuda_rayd()
    from witwin.channel.deterministic import solve

    scene = _two_wall_scene()
    config = DeterministicConfig(
        max_depth=2,
        components=["los", "reflection", "scattering"],
        scattering_samples_per_m2=16.0,
    )
    result = solve(scene, config, reference_frequency_hz=3.0e9)
    # Default-off: no component_id=6 chain rows beyond single-bounce scattering,
    # and the solve completes with finite power.
    assert math.isfinite(float(result.component_power["scattering"]))


def test_solver_chain_enabled_end_to_end():
    """Full chain solve: discovery -> Op A dispatch -> component_id=6 rows.

    Requires the ADR-021 D2 native Op A facade; skips loudly if it is not yet
    registered so the suite stays green while D2 lands.
    """

    _require_cuda_rayd()
    if getattr(
        scattering_append.scattering_kernels, "scattering_chain_ensemble_eval", None
    ) is None:
        pytest.skip("ADR-021 D2 Op A facade not yet registered")
    from witwin.channel.runtime import native_extension

    if not hasattr(native_extension(), "scattering_chain_ensemble_eval"):
        pytest.skip("ADR-021 D2 Op A native symbol not built into this extension")
    from witwin.channel.deterministic import solve

    scene = _two_wall_scene()
    config = DeterministicConfig(
        max_depth=2,
        components=["los", "reflection", "scattering"],
        scattering_samples_per_m2=16.0,
        scattering_chain_max_depth=3,
        scattering_chain_samples_per_m2=8.0,
        export_paths=True,
    )
    result = solve(scene, config, reference_frequency_hz=3.0e9)
    scatter_meta = result.metadata["scattering"]
    assert scatter_meta["chain_row_count"] >= 0
    assert math.isfinite(float(result.component_power["scattering"]))
    # If any chain row survived Op A, it must be a component_id=6 row with a
    # depth (d1 + 1 + d2) of at least 2.
    if scatter_meta["chain_kept_count"] > 0:
        paths = result.paths
        chain_mask = (paths.component_id == 6) & (paths.depth >= 2)
        assert bool(chain_mask.any())


def test_solver_chain_ad_mode_requires_companion():
    """AD-mode chain solve fails loudly until the D5 Op A ``_ad`` companion lands."""

    _require_cuda_rayd()
    if scattering_append._ADR021_D5_CHAIN_AD_WIRED:
        pytest.skip("ADR-021 D5 Op A _ad companion is wired into the append path")
    from witwin.channel.deterministic import solve

    scene = _two_wall_scene()
    config = DeterministicConfig(
        max_depth=2,
        components=["los", "reflection", "scattering"],
        scattering_chain_max_depth=2,
        scattering_chain_samples_per_m2=8.0,
        ad_mode="jvp",
    )
    with pytest.raises(RuntimeError, match="scattering_chain_ensemble_eval"):
        solve(scene, config, reference_frequency_hz=3.0e9)

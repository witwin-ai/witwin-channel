# Copyright Xingyu Chen.
# Legal-but-hostile inputs, each run in its own process, followed by a CUDA context health check.

"""Legal-but-hostile inputs, each run in its own process, followed by a CUDA context health check."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass, fields, replace
import sys
from typing import Any

import torch

FREQ = 3.5e9
HEALTH_OK = "SURVIVAL_HEALTH ok"
OUTCOME = "SURVIVAL_OUTCOME"


@dataclass(frozen=True, slots=True)
class SurvivalCase:
    """``expect`` is ``survive`` (complete or raise a Python exception) or ``reject`` (must raise)."""

    name: str
    run: Callable[[], Any]
    finding: str
    expect: str = "survive"
    timeout_s: float = 300.0
    gpus: int = 1
    long_gpu_work: bool = False


CASES: dict[str, SurvivalCase] = {}


def case(
    name: str, finding: str, *, expect: str = "survive", timeout_s: float = 300.0, gpus: int = 1,
    long_gpu_work: bool = False,
) -> Callable[[Callable[[], Any]], Callable[[], Any]]:
    def register(fn: Callable[[], Any]) -> Callable[[], Any]:
        CASES[name] = SurvivalCase(name, fn, finding, expect, timeout_s, gpus, long_gpu_work)
        return fn

    return register


# --------------------------------------------------------------------------- scene helpers


def _antennas() -> tuple[Any, Any]:
    from witwin.core import AntennaState
    from witwin.core.identity import new_antenna_id

    def tx(pos: Any, **kw: Any) -> Any:
        return AntennaState(new_antenna_id(), "tx", torch.as_tensor(pos, dtype=torch.float32),
                            power_w=1.0, **kw)

    def rx(pos: Any, **kw: Any) -> Any:
        return AntennaState(new_antenna_id(), "rx", torch.as_tensor(pos, dtype=torch.float32), **kw)

    return tx, rx


def _structure(
    vertices: Any, faces: Any, *, diagnostics: bool = False, material: Any = None,
    **structure_kw: Any,
) -> Any:
    from witwin.core import Mesh, PhysicalMaterial, Structure

    mesh = Mesh(vertices, faces, recenter=False, fill_mode="surface", topology_diagnostics=diagnostics)
    return Structure(mesh, PhysicalMaterial(eps_r=4.0, sigma_e=0.01) if material is None else material,
                     **structure_kw)


def _floor(half: float = 10.0, material: Any = None) -> Any:
    v = torch.tensor([[-half, -half, 0.0], [half, -half, 0.0], [-half, half, 0.0], [half, half, 0.0]])
    return _structure(v, torch.tensor([[0, 1, 2], [1, 3, 2]]), material=material)


def _wall(x: float = 2.5, half: float = 2.0, material: Any = None) -> Any:
    v = torch.tensor([[x, -half, -1.0], [x, half, -1.0], [x, -half, 3.0], [x, half, 3.0]])
    return _structure(v, torch.tensor([[0, 1, 2], [1, 3, 2]]), material=material)


def _closed_box(
    center: tuple[float, float, float] = (3.0, 0.0, 1.0),
    half: tuple[float, float, float] = (0.5, 0.5, 1.0),
) -> Any:
    cx, cy, cz = center
    hx, hy, hz = half
    v = torch.tensor([[cx + sx * hx, cy + sy * hy, cz + sz * hz]
                      for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    f = torch.tensor([[0, 1, 3], [0, 3, 2], [4, 7, 5], [4, 6, 7], [0, 5, 1], [0, 4, 5], [2, 3, 7],
                      [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return _structure(v, f)


def _box_between_scene(*, rx_pos: Any = (6.0, 0.3, 1.2), tx_pos: Any = (0.0, 0.0, 1.5)) -> Any:
    from witwin.core import Scene

    tx, rx = _antennas()
    return Scene(structures=[_closed_box()], endpoints=[tx(tx_pos), rx(rx_pos)])


def _solver(name: str) -> Any:
    import importlib

    return importlib.import_module({
        "det": "witwin.channel.deterministic", "path": "witwin.channel.path",
        "mc": "witwin.channel.montecarlo.basic", "bdpt": "witwin.channel.montecarlo.bdpt",
    }[name])


def _solve(solver: str, scene: Any, **config: Any) -> Any:
    module = _solver(solver)
    if solver in ("mc", "bdpt"):
        config.setdefault("samples", 1024)
    result = module.solve(scene, module.Config(**config), reference_frequency_hz=FREQ)
    _touch(result)
    return result


def _touch(result: Any) -> None:
    """Force every lazily-launched kernel to finish so asynchronous device faults surface here."""
    torch.cuda.synchronize()
    for name in ("path_gain", "a", "tau", "valid", "field"):
        value = getattr(result, name, None)
        if isinstance(value, torch.Tensor):
            value.detach().abs().float().sum().item()
    torch.cuda.synchronize()


def _city_scene(n_side: int, receivers: list[Any]) -> Any:
    from witwin.core import Scene

    from benchmarks import synthetic_city as sc

    city = sc.city(n_side)
    return Scene(structures=city.structures, endpoints=[sc.transmitter(city.extent_m), *receivers])


def _uniform_points(extent: float, n: int, seed: int = 1) -> list[Any]:
    """Seeded uniform points (some inside buildings), as in the original S-2 reproduction."""
    _, rx = _antennas()
    generator = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        p = torch.rand(3, generator=generator) * extent
        p[2] = 1.5
        out.append(rx(p))
    return out


def _grid(extent: float, n: int) -> Any:
    from benchmarks import synthetic_city as sc

    return sc.receiver_grid(extent, n)


# --------------------------------------------------------------------------- controls


@case("control.det_los_reflection", "control")
def _() -> None:
    from witwin.core import Scene

    tx, rx = _antennas()
    _solve("det", Scene(structures=[_floor(), _wall()], endpoints=[tx((0, 0, 1.5)), rx((1, 1, 1.5))]),
           components={"los", "reflection"})


@case("control.transmission_box_depth2", "control")
def _() -> None:
    _solve("det", _box_between_scene(), components={"los", "transmission"}, max_depth=2)


# --------------------------------------------------------------------------- S-1


for _solver_name in ("det", "path", "bdpt"):
    @case(f"s1.box_transmission_depth1.{_solver_name}", "S-1")
    def _(solver: str = _solver_name) -> None:
        _solve(solver, _box_between_scene(), components={"los", "transmission"}, max_depth=1)


@case("s1.city4_transmission_depth2.mc", "S-1")
def _() -> None:
    from benchmarks import synthetic_city as sc

    extent = 4 * sc.PITCH_M
    _solve("mc", _city_scene(4, [_grid(extent, 16)]), components={"los", "transmission"},
           max_depth=2, samples=4096)


@case("s1.city4_transmission_depth2.det", "S-1")
def _() -> None:
    from benchmarks import synthetic_city as sc

    extent = 4 * sc.PITCH_M
    _solve("det", _city_scene(4, [_grid(extent, 16)]), components={"los", "transmission"},
           max_depth=2)


@case("s1.city8_transmission_depth5.det", "S-1")
def _() -> None:
    from benchmarks import synthetic_city as sc

    extent = 8 * sc.PITCH_M
    _solve("det", _city_scene(8, [_grid(extent, 32)]), components={"los", "transmission"},
           max_depth=5)


def _consumer() -> Any:
    from witwin.channel.propagation import consumer

    return consumer


def _batch(
    positions: list[list[float]], *, ids: list[int] | None = None, power: bool = False,
    device: str = "cuda",
) -> Any:
    c = _consumer()
    pos = torch.tensor(positions, dtype=torch.float32, device=device).reshape(-1, 3)
    n = pos.shape[0]
    return c.EndpointBatch(
        stable_ids=torch.tensor(list(range(100, 100 + n)) if ids is None else ids, dtype=torch.int64,
                                device=device),
        positions_m=pos,
        polarizations=torch.tensor([[0.0, 0.0, 1.0]], device=device).repeat(n, 1).contiguous(),
        powers_w=torch.ones(n, device=device) if power else None,
        polarization_basis=None,
    )


def _request(
    source: Any, sink: Any, *, components: frozenset[str] = frozenset({"los", "reflection"}),
    max_depth: int = 1,
) -> Any:
    return _consumer().PropagationRequest(
        sources=source, sinks=sink, reference_frequency_hz=FREQ, components=components,
        max_depth=max_depth, response="scalar_transport", topology_mode="discover", ad_mode="none")


def _compiled(structures: list[Any]) -> Any:
    from witwin.core import Scene
    from witwin.channel.scene import compile as compile_scene

    return compile_scene(Scene(structures=structures, endpoints=[]), reference_frequency_hz=FREQ)


def _evaluate(compiled: Any, request: Any) -> Any:
    out = _consumer().evaluate(compiled, request)
    torch.cuda.synchronize()
    out.paths.transport.coefficient.abs().sum().item()
    return out


@case("s1.consumer_two_walls_transmission_depth1", "S-1")
def _() -> None:
    compiled = _compiled([_wall(2.0, 4.0), _wall(2.5, 4.0)])
    _evaluate(compiled, _request(_batch([[0.0, 0.0, 1.5]], power=True),
                                 _batch([[5.0, 0.3, 1.2]], ids=[201]),
                                 components=frozenset({"los", "transmission"})))


# --------------------------------------------------------------------------- S-2


@case("s2.path_one_building_4rx_coupled", "S-2")
def _() -> None:
    from benchmarks import synthetic_city as sc

    _solve("path", _city_scene(1, _uniform_points(sc.PITCH_M, 4)), max_depth=2,
           components={"los", "reflection", "diffraction"}, coupled_paths=True)


@case("s2.det_city4_grid16_coupled", "S-2")
def _() -> None:
    from benchmarks import synthetic_city as sc

    _solve("det", _city_scene(4, [_grid(4 * sc.PITCH_M, 16)]), max_depth=2,
           components={"los", "reflection", "diffraction"}, coupled_paths=True)


# --------------------------------------------------------------------------- S-3

_NAN, _INF = float("nan"), float("inf")
for _label, _tx, _rx in (("nan_rx", (0.0, 0.0, 1.5), (_NAN, 0.3, 1.2)),
                         ("inf_tx", (_INF, 0.0, 1.5), (6.0, 0.3, 1.2)),
                         ("far_rx_1e30", (0.0, 0.0, 1.5), (1e30, 0.3, 1.2))):
    @case(f"s3.{_label}_transmission.det", "S-3")
    def _(tx_pos: Any = _tx, rx_pos: Any = _rx) -> None:
        _solve("det", _box_between_scene(tx_pos=tx_pos, rx_pos=rx_pos),
               components={"los", "transmission"}, max_depth=2)


@case("s3.consumer_nan_source_transmission", "S-3")
def _() -> None:
    compiled = _compiled([_floor(), _wall()])
    _evaluate(compiled, _request(_batch([[_NAN, 0.0, 1.5]], power=True),
                                 _batch([[1.0, 1.0, 1.5]], ids=[201]),
                                 components=frozenset({"los", "transmission"})))


# --------------------------------------------------------------------------- S-4

_TRI = [[2.5, -2.0, -1.0], [2.5, 2.0, -1.0], [2.5, 0.0, 3.0]]


def _face_case(faces: Any, *, device: str = "cpu") -> None:
    from witwin.core import Scene

    tx, rx = _antennas()
    v = torch.tensor(_TRI, device=device)
    f = torch.as_tensor(faces, device=device)
    scene = Scene(structures=[_floor(), _structure(v, f)], endpoints=[tx((0, 0, 1.5)), rx((1, 1, 1.5))])
    _solve("det", scene, components={"los", "reflection"})


for _label, _faces in (("negative", [[0, 1, -1]]), ("equals_vertex_count", [[0, 1, 3]]),
                       ("int32_max", [[0, 1, 2**31 - 1]]), ("int64_wraps_to_valid", [[0, 1, 2**32 + 2]])):
    @case(f"s4.face_index_{_label}", "S-4", expect="reject")
    def _(faces: Any = _faces) -> None:
        _face_case(torch.tensor(faces, dtype=torch.int64))


@case("s4.face_index_oob_cuda_tensor", "S-4", expect="reject")
def _() -> None:
    _face_case(torch.tensor([[0, 1, 7]]), device="cuda")


@case("s4.face_uv_index_oob", "S-4", expect="reject")
def _() -> None:
    from witwin.core import Scene

    tx, rx = _antennas()
    v = torch.tensor([[2.5, -2.0, -1.0], [2.5, 2.0, -1.0], [2.5, -2.0, 3.0], [2.5, 2.0, 3.0]])
    uv = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    wall = _structure(v, torch.tensor([[0, 1, 2], [1, 3, 2]]), uv=uv, face_uv=torch.tensor([[0, 1, 2], [1, 3, 99]]))
    scene = Scene(structures=[_floor(), wall], endpoints=[tx((0, 0, 1.5)), rx((1, 1, 1.5))])
    _solve("bdpt", scene, components={"los", "reflection"})


@case("s4.face_tensor_mutated_after_compile", "S-4", expect="reject")
def _() -> None:
    from witwin.core import Scene

    tx, rx = _antennas()
    faces = torch.tensor([[0, 1, 2]])
    scene = Scene(structures=[_floor(), _structure(torch.tensor(_TRI), faces)],
                  endpoints=[tx((0, 0, 1.5)), rx((1, 1, 1.5))])
    _solve("det", scene, components={"los", "reflection"})
    faces[0, 2] = 1000
    _solve("det", scene, components={"los", "reflection"})


# --------------------------------------------------------------------------- S-5


@case("s5.consumer_endpoints_on_other_gpu", "S-5", gpus=2)
def _() -> None:
    compiled = _compiled([_floor(), _wall()])
    _evaluate(compiled, _request(_batch([[0.0, 0.0, 1.5]], power=True, device="cuda:1"),
                                 _batch([[1.0, 1.0, 1.5]], ids=[201], device="cuda:1")))


# --------------------------------------------------------------------------- S-6


def _tamper(topology: Any, **changes: Callable[[torch.Tensor], torch.Tensor]) -> Any:
    values = {f.name: getattr(topology, f.name) for f in fields(topology)}
    for key, fn in changes.items():
        values[key] = fn(values[key].clone())
    return type(topology)(**values)


def _reevaluate_tampered(**changes: Callable[[torch.Tensor], torch.Tensor]) -> None:
    c = _consumer()
    compiled = _compiled([_floor(), _wall()])
    source, sink = _batch([[0.0, 0.0, 1.5]], power=True), _batch([[1.0, 1.0, 1.5]], ids=[201])
    evaluated = c.evaluate(compiled, _request(source, sink))
    topology = _tamper(evaluated.paths.topology, **changes)
    request = c.FixedTopologyRequest(sources=source, sinks=sink, reference_frequency_hz=FREQ,
                                     topology=c.prepare_fixed_topology(topology),
                                     response="scalar_transport", ad_mode="none")
    out = c.reevaluate(compiled, request)
    torch.cuda.synchronize()
    out.paths.transport.coefficient.abs().nan_to_num(0).sum().item()


def _big(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x >= 0, torch.full_like(x, 10**6), x)


@case("s6.reevaluate_primitive_sequence_oob", "S-6")
def _() -> None:
    _reevaluate_tampered(primitive_sequence=_big, primitive_id=_big)


@case("s6.reevaluate_material_sequence_oob", "S-6")
def _() -> None:
    _reevaluate_tampered(material_sequence=_big, material_id=_big)


@case("s6.reevaluate_source_index_oob", "S-6")
def _() -> None:
    _reevaluate_tampered(source_index=lambda x: x + 10**6)


@case("s6.reevaluate_provenance_none_other_scene", "S-6")
def _() -> None:
    c = _consumer()
    compiled = _compiled([_floor(), _wall()])
    source, sink = _batch([[0.0, 0.0, 1.5]], power=True), _batch([[1.0, 1.0, 1.5]], ids=[201])
    topology = replace(c.evaluate(compiled, _request(source, sink)).paths.topology, provenance=None)
    other = _compiled([_structure(torch.tensor(_TRI), torch.tensor([[0, 1, 2]]))])
    request = c.FixedTopologyRequest(sources=source, sinks=sink, reference_frequency_hz=FREQ,
                                     topology=c.prepare_fixed_topology(topology),
                                     response="scalar_transport", ad_mode="none")
    out = c.reevaluate(other, request)
    torch.cuda.synchronize()
    out.paths.transport.coefficient.abs().nan_to_num(0).sum().item()


# --------------------------------------------------------------------------- S-7


def _wall_with_faces(faces: torch.Tensor) -> None:
    from witwin.core import Scene

    tx, rx = _antennas()
    v = torch.tensor([[2.5, -2.0, -1.0], [2.5, 2.0, -1.0], [2.5, -2.0, 3.0], [2.5, 2.0, 3.0]])
    scene = Scene(structures=[_floor(), _structure(v, faces)], endpoints=[tx((0, 0, 1.5)), rx((1, 1, 1.5))])
    _solve("det", scene, components={"los", "reflection"})


@case("s7.duplicate_faces_1000", "S-7", timeout_s=60.0)
def _() -> None:
    _wall_with_faces(torch.tensor([[0, 1, 2], [1, 3, 2]]).repeat(500, 1))


@case("s7.zero_area_faces_120", "S-7", timeout_s=60.0)
def _() -> None:
    _wall_with_faces(torch.cat([torch.tensor([[0, 1, 2], [1, 3, 2]]),
                                torch.tensor([[0, 0, 0]]).repeat(120, 1)]))


@case("s7.duplicate_faces_200_below_edge_limit", "S-7", timeout_s=60.0)
def _() -> None:
    _wall_with_faces(torch.tensor([[0, 1, 2], [1, 3, 2]]).repeat(100, 1))


@case("s7.zero_area_faces_1000", "S-7", timeout_s=60.0)
def _() -> None:
    _wall_with_faces(torch.cat([torch.tensor([[0, 1, 2], [1, 3, 2]]),
                                torch.tensor([[0, 0, 1]]).repeat(1000, 1)]))


# --------------------------------------------------------------------------- S-8


def _rough_scene(frequency_hz: float) -> None:
    from witwin.core import MaterialLayer, PhysicalMaterial, Scene, SurfaceRoughness

    tx, rx = _antennas()
    material = PhysicalMaterial(
        layers=(MaterialLayer(thickness_m=0.1, eps_r=4.0, sigma_e=0.01),),
        roughness_front=SurfaceRoughness(rms_height_m=0.01, correlation_length_x_m=0.2,
                                         correlation_length_y_m=0.2))
    scene = Scene(structures=[_floor(3.0, material), _wall(material=material)],
                  endpoints=[tx((0, 0, 1.5)), rx((1, 1, 1.5))])
    module = _solver("det")
    _touch(module.solve(scene, module.Config(components={"los", "scattering"}),
                        reference_frequency_hz=frequency_hz))


@case("s8.kirchhoff_300ghz", "S-8", timeout_s=30.0, long_gpu_work=True)
def _() -> None:
    _rough_scene(300e9)


@case("s8.kirchhoff_1thz", "S-8", timeout_s=30.0, long_gpu_work=True)
def _() -> None:
    _rough_scene(1e12)


# --------------------------------------------------------------------------- runner


def _health() -> bool:
    try:
        x = torch.zeros(1, device="cuda") + 1.0
        torch.cuda.synchronize()
        return bool(x.item() == 1.0)
    except Exception as error:  # noqa: BLE001 - reported, not hidden
        print(f"SURVIVAL_HEALTH failed: {type(error).__name__}: {str(error).splitlines()[0][:200]}",
              flush=True)
        return False


def main(argv: list[str] | None = None) -> int:
    """Run one case; exit 0 when the expectation holds and the CUDA context still works."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", nargs="?")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)
    if args.list or args.case is None:
        for item in CASES.values():
            print(f"{item.name:48s} {item.finding:8s} {item.expect}")
        return 0
    item = CASES[args.case]
    # Shared-GPU rule: at most 8 GiB per process.
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, (8 << 30) / total))
    raised = None
    try:
        item.run()
    except Exception as error:  # noqa: BLE001 - a Python exception is a legal outcome
        raised = error
        first = str(error).splitlines()[0][:300] if str(error) else ""
        print(f"{OUTCOME} raised {type(error).__name__}: {first}", flush=True)
    else:
        print(f"{OUTCOME} completed", flush=True)
    healthy = _health()
    if healthy:
        print(HEALTH_OK, flush=True)
    if not healthy:
        return 3
    if item.expect == "reject" and raised is None:
        print("SURVIVAL_EXPECTATION input was accepted but must be rejected", flush=True)
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""ADR-022 per-op lockstep: BDPT AD companions vs the float64 Torch oracles.

Each of the six BDPT-owned forward operations (plan 10a section 6) gains a native
``_backward`` / ``_jvp`` companion; this module pins each one against the
differentiable float64 oracle in ``tests.reference.bdpt_ad_oracles``:

* forward parity (native float32 forward vs the oracle),
* native VJP vs ``torch.autograd`` through the oracle,
* native JVP vs the oracle forward-mode dual,
* native JVP-vs-VJP inner-product duality,
* need-flag gating (off groups return ``None``),
* fixed-input loud rejection through the plan-07 autograd wrapper,
* missing-symbol loud failure.

These were introduced when ADR-022 raised ``EXPECTED_NATIVE_BINDING_COUNT`` to
211; Plan 13 Phase 4 later removes nine audited-dead bindings, so the current
count is 202. Every convention below is frozen by
plan 10a section 6 and the ADR-022 derivative specs; the pair convention
(``d/dF = 2 conj(F) rest`` for ``|F|^2``, pairwise real adjoints elsewhere) matches
``fold_output_cotangents`` (ADR-014).

Interface assumptions (documented deviations where plan 10a section 6 is
under-specified for the Python facade spelling; the native ABI symbol names are
normative):

* The ``_backward`` / ``_jvp`` FACADES live beside their forwards in
  ``montecarlo.bdpt.paths_ad`` / ``.maps`` (the ``scattering`` chain
  precedent: ``functional.*_backward``), named ``<forward>_backward`` /
  ``<forward>_jvp``, taking the forward's positional/keyword args plus
  ``grad_*`` / ``need_grad_*`` (backward) or ``tangent_*`` (jvp) keywords.
* The plan-07 ``torch.autograd.Function`` wrappers live in
  ``montecarlo.bdpt.kernels.autograd`` as ``<forward>_ad`` (fields/materials
  ``autograd.py`` precedent).
* The reflected-subpath differentiable material set is
  ``{eps_r, sigma_e, gain, thickness}`` with ``mu_r`` frozen (the
  ``field_reflection_sequence`` precedent; resolves the section-6.1
  ``material_mu_r`` vs ``grad_gain`` text inconsistency in favour of the
  concrete need-flag output list).
"""

from __future__ import annotations

import pytest
import torch

from tests.ad._fd import relative_error
from tests.ad._tolerances import ABS_TOL
from tests.reference import bdpt_ad_oracles as O
from witwin.channel_native.montecarlo.bdpt.kernels import maps as M
from witwin.channel_native.montecarlo.bdpt.kernels import paths as P
from witwin.channel_native.montecarlo.bdpt import paths_ad as PA
from witwin.channel_native.runtime import symbols

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for BDPT companion AD"
)

_FREQ = 3.0e9
_REL_TOL_LOCK = 1.0e-2
_REL_TOL_FWD = 5.0e-3
_REL_TOL_FD = 5.0e-2

_BDPT_COMPANION_SYMBOLS = (
    "bdpt_reflected_light_subpath_state_backward",
    "bdpt_reflected_light_subpath_state_jvp",
    "bdpt_transmitted_light_subpath_state_backward",
    "bdpt_transmitted_light_subpath_state_jvp",
    "bdpt_endpoint_connection_samples_backward",
    "bdpt_endpoint_connection_samples_jvp",
    "bdpt_accumulate_connection_samples_backward",
    "bdpt_accumulate_connection_samples_jvp",
    "bdpt_finalize_point_components_backward",
    "bdpt_finalize_point_components_jvp",
    "bdpt_finalize_component_maps_backward",
    "bdpt_finalize_component_maps_jvp",
)


# ---------------------------------------------------------------------------
# Shared builders.
# ---------------------------------------------------------------------------


def _unit(v: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(v, dim=-1)


def _f32(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float32).contiguous()


def _subpath_state(
    origin: torch.Tensor,
    direction: torch.Tensor,
    field: torch.Tensor,
    throughput: torch.Tensor,
    *,
    component_mask: int = 3,
    depth: int = 1,
    tx_id: int = 0,
    rx_id: int = -1,
    source_power: torch.Tensor | None = None,
    path_length: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    n = origin.shape[0]
    dev = origin.device
    src_power = (
        torch.ones(n, device=dev, dtype=torch.float32)
        if source_power is None
        else _f32(source_power)
    )
    plen = (
        torch.zeros(n, device=dev, dtype=torch.float32)
        if path_length is None
        else _f32(path_length)
    )
    return {
        "origin": _f32(origin),
        "direction": _f32(direction),
        "throughput_real": _f32(throughput.real),
        "throughput_imag": _f32(throughput.imag),
        "pdf_forward": torch.full((n,), 0.25, device=dev, dtype=torch.float32),
        "pdf_reverse": torch.full((n,), 0.25, device=dev, dtype=torch.float32),
        "depth": torch.full((n,), depth, device=dev, dtype=torch.int32),
        "component_mask": torch.full((n,), component_mask, device=dev, dtype=torch.int32),
        "primitive_id": torch.full((n,), -1, device=dev, dtype=torch.int32),
        "edge_id": torch.full((n,), -1, device=dev, dtype=torch.int32),
        "tx_id": torch.full((n,), tx_id, device=dev, dtype=torch.int32),
        "rx_id": torch.full((n,), rx_id, device=dev, dtype=torch.int32),
        "grid_linear_id": torch.full((n,), -1, device=dev, dtype=torch.int32),
        "valid": torch.ones(n, device=dev, dtype=torch.bool),
        "path_length": plen,
        "field_real": _f32(field.real),
        "field_imag": _f32(field.imag),
        "source_power": src_power,
        "event_type": torch.zeros(n, device=dev, dtype=torch.int32),
    }


def _intersection(
    t: torch.Tensor, p: torch.Tensor, n: torch.Tensor
) -> dict[str, torch.Tensor]:
    rows = t.shape[0]
    dev = t.device
    return {
        "t": _f32(t),
        "p": _f32(p),
        "n": _f32(n),
        "geo_n": _f32(n),
        "uv": torch.zeros((rows, 2), device=dev, dtype=torch.float32),
        "barycentric": torch.zeros((rows, 3), device=dev, dtype=torch.float32),
        "shape_id": torch.zeros(rows, device=dev, dtype=torch.int32),
        "prim_id": torch.arange(rows, device=dev, dtype=torch.int32),
        "local_prim_id": torch.arange(rows, device=dev, dtype=torch.int32),
        "global_prim_id": torch.arange(rows, device=dev, dtype=torch.int32),
    }


def _reflect_fixture(seed: int, rows: int = 8):
    """Shared reflected-subpath geometry + per-row (== per-face) materials."""

    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(seed)

    def randn(*s):
        return torch.randn(*s, generator=g, device=dev, dtype=torch.float64)

    def rand(*s):
        return torch.rand(*s, generator=g, device=dev, dtype=torch.float64)

    direction = _unit(randn(rows, 3))
    normal = _unit(
        randn(rows, 3) * 0.2 + torch.tensor([0.0, 0.0, 1.0], device=dev)
    )
    # Keep the ray hitting the front face (broadly downward into the +z wall).
    direction[:, 2] = -direction[:, 2].abs()
    field = randn(rows, 3) + 1.0j * randn(rows, 3)
    throughput = (0.5 + rand(rows)).to(torch.complex128)
    eps = 1.5 + 3.0 * rand(rows)
    sigma = 0.01 + 0.05 * rand(rows)
    mu = torch.ones(rows, device=dev, dtype=torch.float64)
    gain = 0.7 + 0.5 * rand(rows)
    thickness = 0.05 + 0.15 * rand(rows)
    hit_t = 1.0 + rand(rows)
    hit_p = randn(rows, 3)
    return {
        "direction": direction, "normal": normal, "field": field,
        "throughput": throughput, "eps": eps, "sigma": sigma, "mu": mu,
        "gain": gain, "thickness": thickness, "hit_t": hit_t, "hit_p": hit_p,
        "rows": rows, "device": dev,
    }


def _reflect_native(fx, *, origin=None):
    dev = fx["device"]
    rows = fx["rows"]
    orig = fx["hit_p"] if origin is None else origin
    light = _subpath_state(orig, fx["direction"], fx["field"], fx["throughput"])
    inter = _intersection(fx["hit_t"], fx["hit_p"], fx["normal"])
    material = {
        "material_gain": _f32(fx["gain"]),
        "material_valid": torch.ones(rows, device=dev, dtype=torch.bool),
        "material_eps_r": _f32(fx["eps"]),
        "material_sigma_e": _f32(fx["sigma"]),
        "material_mu_r": _f32(fx["mu"]),
        "material_thickness": _f32(fx["thickness"]),
    }
    return light, inter, material


# ---------------------------------------------------------------------------
# 6.1 reflected light subpath advance.
# ---------------------------------------------------------------------------


def test_reflected_forward_matches_oracle():
    fx = _reflect_fixture(11)
    light, inter, material = _reflect_native(fx)
    out = P.bdpt_reflected_light_subpath_state(
        light, inter, **material, frequency_hz=_FREQ
    )
    freq = torch.tensor(_FREQ, dtype=torch.float64, device=fx["device"])
    ref = O.reflected_subpath_advance_reference(
        fx["field"], fx["throughput"], fx["direction"], fx["normal"],
        fx["eps"], fx["sigma"], fx["mu"], fx["gain"], fx["thickness"], freq,
    )
    native_field = torch.complex(out["field_real"], out["field_imag"])
    native_thr = torch.complex(out["throughput_real"], out["throughput_imag"])
    assert relative_error(native_field, ref["field"], abs_floor=ABS_TOL) <= _REL_TOL_FWD
    assert relative_error(native_thr, ref["throughput"], abs_floor=ABS_TOL) <= _REL_TOL_FWD


def _reflect_oracle_leaves(fx):
    """Return real leaves + a zero-grad oracle forward closure (pair convention)."""

    dev = fx["device"]
    leaves = {
        "field_real": fx["field"].real.clone().requires_grad_(True),
        "field_imag": fx["field"].imag.clone().requires_grad_(True),
        "throughput_real": fx["throughput"].real.clone().requires_grad_(True),
        "throughput_imag": fx["throughput"].imag.clone().requires_grad_(True),
        "eps": fx["eps"].clone().requires_grad_(True),
        "sigma": fx["sigma"].clone().requires_grad_(True),
        "gain": fx["gain"].clone().requires_grad_(True),
        "thickness": fx["thickness"].clone().requires_grad_(True),
        "frequency": torch.tensor(_FREQ, dtype=torch.float64, device=dev, requires_grad=True),
    }

    def forward(values):
        field = torch.complex(values["field_real"], values["field_imag"])
        thr = torch.complex(values["throughput_real"], values["throughput_imag"])
        return O.reflected_subpath_advance_reference(
            field, thr, fx["direction"], fx["normal"], values["eps"], values["sigma"],
            fx["mu"], values["gain"], values["thickness"], values["frequency"],
        )

    return leaves, forward


def test_reflected_backward_matches_oracle():
    fx = _reflect_fixture(23)
    light, inter, material = _reflect_native(fx)
    g = torch.Generator(device="cuda").manual_seed(51)

    def rc(*s):
        return torch.randn(*s, generator=g, device="cuda", dtype=torch.float32)

    grads = {
        "grad_field_real": rc(fx["rows"], 3),
        "grad_field_imag": rc(fx["rows"], 3),
        "grad_throughput_real": rc(fx["rows"]),
        "grad_throughput_imag": rc(fx["rows"]),
    }
    native = PA.bdpt_reflected_light_subpath_state_backward(
        light, inter, **material, frequency_hz=_FREQ, **grads,
        need_grad_material=True, need_grad_field_in=True, need_grad_frequency=True,
    )

    leaves, forward = _reflect_oracle_leaves(fx)
    ref = forward(leaves)
    loss = (
        (grads["grad_field_real"].double() * ref["field"].real).sum()
        + (grads["grad_field_imag"].double() * ref["field"].imag).sum()
        + (grads["grad_throughput_real"].double() * ref["throughput"].real).sum()
        + (grads["grad_throughput_imag"].double() * ref["throughput"].imag).sum()
    )
    loss.backward()

    for native_key, leaf_key in (
        ("grad_eps_r", "eps"), ("grad_sigma_e", "sigma"),
        ("grad_gain", "gain"), ("grad_thickness", "thickness"),
        ("grad_light_field_real", "field_real"),
        ("grad_light_field_imag", "field_imag"),
        ("grad_light_throughput_real", "throughput_real"),
        ("grad_light_throughput_imag", "throughput_imag"),
    ):
        assert relative_error(
            native[native_key], leaves[leaf_key].grad, abs_floor=ABS_TOL
        ) <= _REL_TOL_LOCK, native_key
    assert relative_error(
        native["grad_frequency"].reshape(()), leaves["frequency"].grad,
        abs_floor=ABS_TOL,
    ) <= _REL_TOL_LOCK


def test_reflected_jvp_matches_oracle():
    fx = _reflect_fixture(31)
    light, inter, material = _reflect_native(fx)
    g = torch.Generator(device="cuda").manual_seed(72)

    def rc(*s):
        return torch.randn(*s, generator=g, device="cuda", dtype=torch.float32)

    tangents = {
        "tangent_eps_r": rc(fx["rows"]),
        "tangent_sigma_e": rc(fx["rows"]),
        "tangent_gain": rc(fx["rows"]),
        "tangent_thickness": rc(fx["rows"]),
        "tangent_light_field_real": rc(fx["rows"], 3),
        "tangent_light_field_imag": rc(fx["rows"], 3),
        "tangent_light_throughput_real": rc(fx["rows"]),
        "tangent_light_throughput_imag": rc(fx["rows"]),
    }
    t_freq = float(rc(1))
    native = PA.bdpt_reflected_light_subpath_state_jvp(
        light, inter, **material, frequency_hz=_FREQ, tangent_frequency=t_freq,
        **tangents,
    )

    # Oracle forward-mode dual (double precision) with the same tangents.
    dev = fx["device"]
    with torch.autograd.forward_ad.dual_level():
        def dual(primal, tangent):
            return torch.autograd.forward_ad.make_dual(
                primal, tangent.double() if isinstance(tangent, torch.Tensor) else tangent
            )
        field = torch.complex(
            dual(fx["field"].real, tangents["tangent_light_field_real"]),
            dual(fx["field"].imag, tangents["tangent_light_field_imag"]),
        )
        thr = torch.complex(
            dual(fx["throughput"].real, tangents["tangent_light_throughput_real"]),
            dual(fx["throughput"].imag, tangents["tangent_light_throughput_imag"]),
        )
        out = O.reflected_subpath_advance_reference(
            field, thr, fx["direction"], fx["normal"],
            dual(fx["eps"], tangents["tangent_eps_r"]),
            dual(fx["sigma"], tangents["tangent_sigma_e"]),
            fx["mu"],
            dual(fx["gain"], tangents["tangent_gain"]),
            dual(fx["thickness"], tangents["tangent_thickness"]),
            dual(torch.tensor(_FREQ, dtype=torch.float64, device=dev),
                 torch.tensor(t_freq, dtype=torch.float64, device=dev)),
        )
        tf = torch.autograd.forward_ad.unpack_dual(out["field"]).tangent
        tt = torch.autograd.forward_ad.unpack_dual(out["throughput"]).tangent

    native_tf = torch.complex(native["tangent_field_real"], native["tangent_field_imag"])
    native_tt = torch.complex(
        native["tangent_throughput_real"], native["tangent_throughput_imag"]
    )
    assert relative_error(native_tf, tf, abs_floor=ABS_TOL) <= _REL_TOL_LOCK
    assert relative_error(native_tt, tt, abs_floor=ABS_TOL) <= _REL_TOL_LOCK


def test_reflected_jvp_vjp_duality():
    fx = _reflect_fixture(42)
    light, inter, material = _reflect_native(fx)
    g = torch.Generator(device="cuda").manual_seed(84)

    def rc(*s):
        return torch.randn(*s, generator=g, device="cuda", dtype=torch.float32)

    tangents = {
        "tangent_eps_r": rc(fx["rows"]), "tangent_sigma_e": rc(fx["rows"]),
        "tangent_gain": rc(fx["rows"]), "tangent_thickness": rc(fx["rows"]),
        "tangent_light_field_real": rc(fx["rows"], 3),
        "tangent_light_field_imag": rc(fx["rows"], 3),
        "tangent_light_throughput_real": rc(fx["rows"]),
        "tangent_light_throughput_imag": rc(fx["rows"]),
    }
    jvp = PA.bdpt_reflected_light_subpath_state_jvp(
        light, inter, **material, frequency_hz=_FREQ, tangent_frequency=0.0, **tangents
    )
    cot = {k: rc(*jvp[k].shape) for k in (
        "tangent_field_real", "tangent_field_imag",
        "tangent_throughput_real", "tangent_throughput_imag",
    )}
    lhs = sum((cot[k].double() * jvp[k].double()).sum() for k in cot)

    vjp = PA.bdpt_reflected_light_subpath_state_backward(
        light, inter, **material, frequency_hz=_FREQ,
        grad_field_real=cot["tangent_field_real"],
        grad_field_imag=cot["tangent_field_imag"],
        grad_throughput_real=cot["tangent_throughput_real"],
        grad_throughput_imag=cot["tangent_throughput_imag"],
        need_grad_material=True, need_grad_field_in=True, need_grad_frequency=False,
    )
    rhs = (
        (vjp["grad_eps_r"].double() * tangents["tangent_eps_r"].double()).sum()
        + (vjp["grad_sigma_e"].double() * tangents["tangent_sigma_e"].double()).sum()
        + (vjp["grad_gain"].double() * tangents["tangent_gain"].double()).sum()
        + (vjp["grad_thickness"].double() * tangents["tangent_thickness"].double()).sum()
        + (vjp["grad_light_field_real"].double() * tangents["tangent_light_field_real"].double()).sum()
        + (vjp["grad_light_field_imag"].double() * tangents["tangent_light_field_imag"].double()).sum()
        + (vjp["grad_light_throughput_real"].double() * tangents["tangent_light_throughput_real"].double()).sum()
        + (vjp["grad_light_throughput_imag"].double() * tangents["tangent_light_throughput_imag"].double()).sum()
    )
    assert relative_error(lhs, rhs, abs_floor=ABS_TOL) <= _REL_TOL_LOCK


def test_reflected_backward_need_flag_gating():
    fx = _reflect_fixture(55)
    light, inter, material = _reflect_native(fx)
    zeros3 = torch.zeros(fx["rows"], 3, device="cuda")
    zeros1 = torch.zeros(fx["rows"], device="cuda")
    out = PA.bdpt_reflected_light_subpath_state_backward(
        light, inter, **material, frequency_hz=_FREQ,
        grad_field_real=zeros3.clone(), grad_field_imag=zeros3.clone(),
        grad_throughput_real=zeros1.clone(), grad_throughput_imag=zeros1.clone(),
        need_grad_material=True, need_grad_field_in=False, need_grad_frequency=False,
    )
    for key in ("grad_eps_r", "grad_sigma_e", "grad_gain", "grad_thickness"):
        assert out[key] is not None
    for key in (
        "grad_light_field_real", "grad_light_field_imag",
        "grad_light_throughput_real", "grad_light_throughput_imag",
        "grad_frequency",
    ):
        assert out[key] is None


# ---------------------------------------------------------------------------
# 6.2 transmitted light subpath advance.
# ---------------------------------------------------------------------------


def _transmit_fixture(seed: int, rows: int = 6, layers: int = 2):
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(seed)

    def randn(*s):
        return torch.randn(*s, generator=g, device=dev, dtype=torch.float64)

    def rand(*s):
        return torch.rand(*s, generator=g, device=dev, dtype=torch.float64)

    normal = _unit(randn(rows, 3) * 0.2 + torch.tensor([1.0, 0.0, 0.0], device=dev))
    direction = _unit(randn(rows, 3) * 0.2 + torch.tensor([1.0, 0.0, 0.0], device=dev))
    field = randn(rows, 3) + 1.0j * randn(rows, 3)
    throughput = (0.5 + rand(rows)).to(torch.complex128)
    lt = 0.05 + 0.15 * rand(rows, layers)
    le = 1.5 + 3.0 * rand(rows, layers)
    ls = 0.01 + 0.05 * rand(rows, layers)
    lm = torch.ones(rows, layers, device=dev, dtype=torch.float64)
    hit_t = 1.0 + rand(rows)
    hit_p = randn(rows, 3)
    return {
        "direction": direction, "normal": normal, "field": field,
        "throughput": throughput, "lt": lt, "le": le, "ls": ls, "lm": lm,
        "hit_t": hit_t, "hit_p": hit_p, "rows": rows, "layers": layers, "device": dev,
    }


def _transmit_native(fx):
    """Native inputs: each row -> its own material with its own layer block (CSR)."""

    dev, rows, layers = fx["device"], fx["rows"], fx["layers"]
    light = _subpath_state(fx["hit_p"], fx["direction"], fx["field"], fx["throughput"])
    inter = _intersection(fx["hit_t"], fx["hit_p"], fx["normal"])
    csr = {
        "face_material_id": torch.arange(rows, device=dev, dtype=torch.int32),
        "layer_offset": (torch.arange(rows, device=dev, dtype=torch.int32) * layers),
        "layer_count": torch.full((rows,), layers, device=dev, dtype=torch.int32),
        "layer_thickness_m": _f32(fx["lt"].reshape(-1)),
        "layer_eps_r": _f32(fx["le"].reshape(-1)),
        "layer_sigma_e": _f32(fx["ls"].reshape(-1)),
        "layer_mu_r": _f32(fx["lm"].reshape(-1)),
    }
    return light, inter, csr


def test_transmitted_forward_matches_oracle():
    fx = _transmit_fixture(13)
    light, inter, csr = _transmit_native(fx)
    out = P.bdpt_transmitted_light_subpath_state(
        light, inter, **csr, frequency_hz=_FREQ
    )
    freq = torch.tensor(_FREQ, dtype=torch.float64, device=fx["device"])
    ref = O.transmitted_subpath_advance_reference(
        fx["field"], fx["throughput"], fx["direction"], fx["normal"],
        fx["lt"], fx["le"], fx["ls"], fx["lm"], freq,
    )
    native_field = torch.complex(out["field_real"], out["field_imag"])
    native_thr = torch.complex(out["throughput_real"], out["throughput_imag"])
    assert relative_error(native_field, ref["field"], abs_floor=ABS_TOL) <= _REL_TOL_FWD
    assert relative_error(native_thr, ref["throughput"], abs_floor=ABS_TOL) <= _REL_TOL_FWD


def _transmit_oracle_leaves(fx):
    dev = fx["device"]
    leaves = {
        "field_real": fx["field"].real.clone().requires_grad_(True),
        "field_imag": fx["field"].imag.clone().requires_grad_(True),
        "throughput_real": fx["throughput"].real.clone().requires_grad_(True),
        "throughput_imag": fx["throughput"].imag.clone().requires_grad_(True),
        "lt": fx["lt"].clone().requires_grad_(True),
        "le": fx["le"].clone().requires_grad_(True),
        "ls": fx["ls"].clone().requires_grad_(True),
        "frequency": torch.tensor(_FREQ, dtype=torch.float64, device=dev, requires_grad=True),
    }

    def forward(values):
        field = torch.complex(values["field_real"], values["field_imag"])
        thr = torch.complex(values["throughput_real"], values["throughput_imag"])
        return O.transmitted_subpath_advance_reference(
            field, thr, fx["direction"], fx["normal"], values["lt"], values["le"],
            values["ls"], fx["lm"], values["frequency"],
        )

    return leaves, forward


def test_transmitted_backward_matches_oracle():
    fx = _transmit_fixture(17)
    light, inter, csr = _transmit_native(fx)
    g = torch.Generator(device="cuda").manual_seed(90)

    def rc(*s):
        return torch.randn(*s, generator=g, device="cuda", dtype=torch.float32)

    grads = {
        "grad_field_real": rc(fx["rows"], 3), "grad_field_imag": rc(fx["rows"], 3),
        "grad_throughput_real": rc(fx["rows"]), "grad_throughput_imag": rc(fx["rows"]),
    }
    native = PA.bdpt_transmitted_light_subpath_state_backward(
        light, inter, **csr, frequency_hz=_FREQ, **grads,
        need_grad_layers=True, need_grad_field_in=True, need_grad_frequency=True,
    )

    leaves, forward = _transmit_oracle_leaves(fx)
    ref = forward(leaves)
    loss = (
        (grads["grad_field_real"].double() * ref["field"].real).sum()
        + (grads["grad_field_imag"].double() * ref["field"].imag).sum()
        + (grads["grad_throughput_real"].double() * ref["throughput"].real).sum()
        + (grads["grad_throughput_imag"].double() * ref["throughput"].imag).sum()
    )
    loss.backward()

    # CSR layer grads are flat [rows*layers]; the oracle leaves are [rows, layers].
    for native_key, leaf_key in (
        ("grad_layer_thickness", "lt"),
        ("grad_layer_eps_r", "le"),
        ("grad_layer_sigma_e", "ls"),
    ):
        assert relative_error(
            native[native_key], leaves[leaf_key].grad.reshape(-1), abs_floor=ABS_TOL
        ) <= _REL_TOL_LOCK, native_key
    assert relative_error(
        native["grad_light_field_real"], leaves["field_real"].grad, abs_floor=ABS_TOL
    ) <= _REL_TOL_LOCK
    assert relative_error(
        native["grad_light_field_imag"], leaves["field_imag"].grad, abs_floor=ABS_TOL
    ) <= _REL_TOL_LOCK
    assert relative_error(
        native["grad_frequency"].reshape(()), leaves["frequency"].grad, abs_floor=ABS_TOL
    ) <= _REL_TOL_LOCK


def test_transmitted_jvp_vjp_duality():
    fx = _transmit_fixture(29)
    light, inter, csr = _transmit_native(fx)
    g = torch.Generator(device="cuda").manual_seed(93)

    def rc(*s):
        return torch.randn(*s, generator=g, device="cuda", dtype=torch.float32)

    tangents = {
        "tangent_layer_thickness": rc(fx["rows"] * fx["layers"]),
        "tangent_layer_eps_r": rc(fx["rows"] * fx["layers"]),
        "tangent_layer_sigma_e": rc(fx["rows"] * fx["layers"]),
        "tangent_light_field_real": rc(fx["rows"], 3),
        "tangent_light_field_imag": rc(fx["rows"], 3),
        "tangent_light_throughput_real": rc(fx["rows"]),
        "tangent_light_throughput_imag": rc(fx["rows"]),
    }
    jvp = PA.bdpt_transmitted_light_subpath_state_jvp(
        light, inter, **csr, frequency_hz=_FREQ, tangent_frequency=0.0, **tangents
    )
    cot = {k: rc(*jvp[k].shape) for k in (
        "tangent_field_real", "tangent_field_imag",
        "tangent_throughput_real", "tangent_throughput_imag",
    )}
    lhs = sum((cot[k].double() * jvp[k].double()).sum() for k in cot)

    vjp = PA.bdpt_transmitted_light_subpath_state_backward(
        light, inter, **csr, frequency_hz=_FREQ,
        grad_field_real=cot["tangent_field_real"],
        grad_field_imag=cot["tangent_field_imag"],
        grad_throughput_real=cot["tangent_throughput_real"],
        grad_throughput_imag=cot["tangent_throughput_imag"],
        need_grad_layers=True, need_grad_field_in=True, need_grad_frequency=False,
    )
    rhs = (
        (vjp["grad_layer_thickness"].double() * tangents["tangent_layer_thickness"].double()).sum()
        + (vjp["grad_layer_eps_r"].double() * tangents["tangent_layer_eps_r"].double()).sum()
        + (vjp["grad_layer_sigma_e"].double() * tangents["tangent_layer_sigma_e"].double()).sum()
        + (vjp["grad_light_field_real"].double() * tangents["tangent_light_field_real"].double()).sum()
        + (vjp["grad_light_field_imag"].double() * tangents["tangent_light_field_imag"].double()).sum()
        + (vjp["grad_light_throughput_real"].double() * tangents["tangent_light_throughput_real"].double()).sum()
        + (vjp["grad_light_throughput_imag"].double() * tangents["tangent_light_throughput_imag"].double()).sum()
    )
    assert relative_error(lhs, rhs, abs_floor=ABS_TOL) <= _REL_TOL_LOCK


def test_transmitted_backward_need_flag_gating():
    fx = _transmit_fixture(61)
    light, inter, csr = _transmit_native(fx)
    zeros3 = torch.zeros(fx["rows"], 3, device="cuda")
    zeros1 = torch.zeros(fx["rows"], device="cuda")
    out = PA.bdpt_transmitted_light_subpath_state_backward(
        light, inter, **csr, frequency_hz=_FREQ,
        grad_field_real=zeros3.clone(), grad_field_imag=zeros3.clone(),
        grad_throughput_real=zeros1.clone(), grad_throughput_imag=zeros1.clone(),
        need_grad_layers=False, need_grad_field_in=True, need_grad_frequency=False,
    )
    for key in ("grad_layer_thickness", "grad_layer_eps_r", "grad_layer_sigma_e"):
        assert out[key] is None
    for key in ("grad_light_field_real", "grad_light_field_imag"):
        assert out[key] is not None
    assert out["grad_frequency"] is None


# ---------------------------------------------------------------------------
# 6.3 endpoint (LoS / NEE) connection contribution.
# ---------------------------------------------------------------------------


def _endpoint_fixture(seed: int, tx: int = 3, rx: int = 4):
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(seed)

    def randn(*s):
        return torch.randn(*s, generator=g, device=dev, dtype=torch.float64)

    def rand(*s):
        return torch.rand(*s, generator=g, device=dev, dtype=torch.float64)

    light_origin = randn(tx, 3)
    sensor_origin = randn(rx, 3) + torch.tensor([4.0, 0.0, 0.0], device=dev)
    light_field = randn(tx, 3) + 1.0j * randn(tx, 3)
    source_power = 0.5 + rand(tx)
    light_path_length = rand(tx)
    rx_pol = _unit(randn(rx, 3))
    return {
        "tx": tx, "rx": rx, "device": dev,
        "light_origin": light_origin, "sensor_origin": sensor_origin,
        "light_field": light_field, "source_power": source_power,
        "light_path_length": light_path_length, "rx_pol": rx_pol,
    }


def _endpoint_native(fx, samples_per_tx=4):
    dev = fx["device"]
    txn, rxn = fx["tx"], fx["rx"]
    # Sensor direction is one -z unit vector per sensor row (rxn), not txn.
    ez = torch.zeros(rxn, 3, device=dev, dtype=torch.float64)
    ez[:, 2] = -1.0
    light = _subpath_state(
        fx["light_origin"], _unit(torch.randn_like(fx["light_origin"])),
        fx["light_field"], torch.ones(txn, device=dev, dtype=torch.complex128),
        depth=0, tx_id=0,
        source_power=fx["source_power"], path_length=fx["light_path_length"],
    )
    # Distinct tx ids per light row and rx ids per sensor row (frozen topology).
    light["tx_id"] = torch.arange(txn, device=dev, dtype=torch.int32)
    sensor_field = torch.complex(
        fx["rx_pol"], torch.zeros_like(fx["rx_pol"])
    )
    sensor = _subpath_state(
        fx["sensor_origin"], ez, sensor_field,
        torch.ones(rxn, device=fx["device"], dtype=torch.complex128),
        depth=0, tx_id=-1, rx_id=0,
    )
    sensor["rx_id"] = torch.arange(rxn, device=fx["device"], dtype=torch.int32)
    sensor["grid_linear_id"] = torch.arange(rxn, device=fx["device"], dtype=torch.int32)
    return light, sensor


def _endpoint_pairwise_reference(fx, samples_per_tx, frequency):
    """Row-major (light, sensor) contribution stack matching the native kernel order."""

    txn, rxn = fx["tx"], fx["rx"]
    rows = []
    for li in range(txn):
        for si in range(rxn):
            rows.append(
                O.endpoint_connection_contribution_reference(
                    fx["light_field"][li : li + 1],
                    fx["source_power"][li : li + 1],
                    fx["light_origin"][li : li + 1],
                    fx["sensor_origin"][si : si + 1],
                    fx["rx_pol"][si : si + 1],
                    fx["light_path_length"][li : li + 1],
                    frequency,
                    samples_per_tx,
                )
            )
    return torch.cat(rows, dim=0)


def test_endpoint_forward_matches_oracle():
    fx = _endpoint_fixture(37)
    light, sensor = _endpoint_native(fx)
    spt = 4
    out = P.bdpt_endpoint_connection_samples(
        light, sensor, frequency_hz=_FREQ, samples_per_tx=spt, mis="none",
    )
    freq = torch.tensor(_FREQ, dtype=torch.float64, device=fx["device"])
    ref = _endpoint_pairwise_reference(fx, spt, freq)
    valid = out["valid"]
    assert relative_error(
        out["contribution"][valid], ref[valid], abs_floor=ABS_TOL
    ) <= _REL_TOL_FWD


def test_endpoint_backward_matches_oracle():
    fx = _endpoint_fixture(43)
    light, sensor = _endpoint_native(fx)
    spt = 4
    g = torch.Generator(device="cuda").manual_seed(94)
    grad_contribution = torch.randn(
        fx["tx"] * fx["rx"], generator=g, device="cuda", dtype=torch.float32
    )
    native = PA.bdpt_endpoint_connection_samples_backward(
        light, sensor, frequency_hz=_FREQ, samples_per_tx=spt, mis="none",
        beta=2.0, strategy_count=1, max_paths=None,
        grad_contribution=grad_contribution,
        need_grad_field=True, need_grad_frequency=True, need_grad_tx_power=True,
    )

    dev = fx["device"]
    field_real = fx["light_field"].real.clone().requires_grad_(True)
    field_imag = fx["light_field"].imag.clone().requires_grad_(True)
    source_power = fx["source_power"].clone().requires_grad_(True)
    frequency = torch.tensor(_FREQ, dtype=torch.float64, device=dev, requires_grad=True)
    field = torch.complex(field_real, field_imag)
    fx_leaf = dict(fx)
    fx_leaf["light_field"] = field
    fx_leaf["source_power"] = source_power
    ref = _endpoint_pairwise_reference(fx_leaf, spt, frequency)
    (grad_contribution.double() * ref).sum().backward()

    # Sensor fields (rx polarization) are frozen; only light field carries grad.
    assert relative_error(
        native["grad_light_field_real"], field_real.grad, abs_floor=ABS_TOL
    ) <= _REL_TOL_LOCK
    assert relative_error(
        native["grad_light_field_imag"], field_imag.grad, abs_floor=ABS_TOL
    ) <= _REL_TOL_LOCK
    # tx_power grad is scattered per tx (many rows share a tx).
    assert relative_error(
        native["grad_tx_power"], source_power.grad, abs_floor=ABS_TOL
    ) <= _REL_TOL_LOCK
    assert relative_error(
        native["grad_frequency"].reshape(()), frequency.grad, abs_floor=ABS_TOL
    ) <= _REL_TOL_LOCK


def test_endpoint_jvp_vjp_duality():
    fx = _endpoint_fixture(47)
    light, sensor = _endpoint_native(fx)
    spt = 4
    g = torch.Generator(device="cuda").manual_seed(95)

    def rc(*s):
        return torch.randn(*s, generator=g, device="cuda", dtype=torch.float32)

    tangents = {
        "tangent_light_field_real": rc(fx["tx"], 3),
        "tangent_light_field_imag": rc(fx["tx"], 3),
        "tangent_tx_power": rc(fx["tx"]),
    }
    jvp = PA.bdpt_endpoint_connection_samples_jvp(
        light, sensor, frequency_hz=_FREQ, samples_per_tx=spt, mis="none",
        beta=2.0, strategy_count=1, max_paths=None,
        tangent_frequency=0.0, **tangents,
    )
    cot = rc(*jvp["tangent_contribution"].shape)
    lhs = (cot.double() * jvp["tangent_contribution"].double()).sum()

    vjp = PA.bdpt_endpoint_connection_samples_backward(
        light, sensor, frequency_hz=_FREQ, samples_per_tx=spt, mis="none",
        beta=2.0, strategy_count=1, max_paths=None,
        grad_contribution=cot,
        need_grad_field=True, need_grad_frequency=False, need_grad_tx_power=True,
    )
    rhs = (
        (vjp["grad_light_field_real"].double() * tangents["tangent_light_field_real"].double()).sum()
        + (vjp["grad_light_field_imag"].double() * tangents["tangent_light_field_imag"].double()).sum()
        + (vjp["grad_tx_power"].double() * tangents["tangent_tx_power"].double()).sum()
    )
    assert relative_error(lhs, rhs, abs_floor=ABS_TOL) <= _REL_TOL_LOCK


def test_endpoint_backward_need_flag_gating():
    fx = _endpoint_fixture(59)
    light, sensor = _endpoint_native(fx)
    grad_contribution = torch.zeros(fx["tx"] * fx["rx"], device="cuda")
    out = PA.bdpt_endpoint_connection_samples_backward(
        light, sensor, frequency_hz=_FREQ, samples_per_tx=4, mis="none",
        beta=2.0, strategy_count=1, max_paths=None,
        grad_contribution=grad_contribution,
        need_grad_field=True, need_grad_frequency=False, need_grad_tx_power=False,
    )
    assert out["grad_light_field_real"] is not None
    assert out["grad_frequency"] is None
    assert out["grad_tx_power"] is None


# ---------------------------------------------------------------------------
# 6.4 accumulate connection samples (power + coherent).
# ---------------------------------------------------------------------------


def _accum_fixture(seed: int, rows: int = 24, tx: int = 3, rx: int = 4):
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(seed)

    def randn(*s):
        return torch.randn(*s, generator=g, device=dev, dtype=torch.float32)

    def randint(high, *s):
        return torch.randint(0, high, s, generator=g, device=dev, dtype=torch.int32)

    contribution = randn(rows).abs()
    mis = 0.5 + torch.rand(rows, generator=g, device=dev)
    coeff_real = randn(rows)
    coeff_imag = randn(rows)
    tx_id = randint(tx, rows)
    rx_id = randint(rx, rows)
    # Sample from the native component-id encoding {0,1,2,5,6} (los, reflection,
    # diffraction, transmission, scattering); ids 3 and 4 are unused, so a
    # dense randint(5) would fabricate non-existent components the accumulate
    # kernels drop to zero.
    _component_ids = torch.tensor(
        [O.COMPONENT_LOS, O.COMPONENT_REFLECTION, O.COMPONENT_DIFFRACTION,
         O.COMPONENT_TRANSMISSION, O.COMPONENT_SCATTERING],
        device=dev, dtype=torch.int32,
    )
    component_id = _component_ids[randint(_component_ids.numel(), rows).long()]
    valid = torch.rand(rows, generator=g, device=dev) > 0.15
    return {
        "rows": rows, "tx": tx, "rx": rx, "device": dev,
        "contribution": contribution, "mis": mis,
        "coeff_real": coeff_real, "coeff_imag": coeff_imag,
        "tx_id": tx_id, "rx_id": rx_id, "component_id": component_id, "valid": valid,
    }


def _accum_samples(fx):
    rows = fx["rows"]
    dev = fx["device"]
    return {
        "topology": torch.stack(
            [fx["tx_id"], fx["rx_id"], fx["component_id"],
             torch.zeros(rows, device=dev, dtype=torch.int32)], dim=1
        ).contiguous(),
        "contribution": fx["contribution"],
        "pdf": torch.ones(rows, device=dev, dtype=torch.float32),
        "mis_weight": fx["mis"].to(torch.float32),
        "component_id": fx["component_id"],
        "valid": fx["valid"],
        "tx_id": fx["tx_id"],
        "rx_id": fx["rx_id"],
        "grid_linear_id": fx["rx_id"],
        "light_depth": torch.ones(rows, device=dev, dtype=torch.int32),
        "sensor_depth": torch.zeros(rows, device=dev, dtype=torch.int32),
        "path_length_m": torch.ones(rows, device=dev, dtype=torch.float32),
    }


def test_accumulate_power_backward_matches_oracle():
    fx = _accum_fixture(71)
    samples = _accum_samples(fx)
    g = torch.Generator(device="cuda").manual_seed(96)
    cot = {
        name: torch.randn(fx["tx"], fx["rx"], generator=g, device="cuda")
        for name in ("grad_path_gain", "grad_los", "grad_reflection",
                     "grad_diffraction", "grad_transmission", "grad_scattering")
    }
    native = PA.bdpt_accumulate_connection_samples_backward(
        samples, tx_count=fx["tx"], rx_count=fx["rx"], combine_domain="power",
        **cot, need_grad_contribution=True, need_grad_coeff=False,
    )

    contribution = fx["contribution"].double().clone().requires_grad_(True)
    ref = O.accumulate_power_reference(
        contribution, fx["mis"], fx["tx_id"], fx["rx_id"], fx["component_id"],
        fx["valid"], fx["tx"], fx["rx"],
    )
    loss = sum(
        (cot[f"grad_{name}"].double() * ref[name]).sum()
        for name in ("path_gain", "los", "reflection", "diffraction",
                     "transmission", "scattering")
    )
    loss.backward()
    assert relative_error(
        native["grad_contribution"], contribution.grad, abs_floor=ABS_TOL
    ) <= _REL_TOL_LOCK
    assert native["grad_coeff_real"] is None and native["grad_coeff_imag"] is None


def test_accumulate_coherent_backward_matches_oracle():
    fx = _accum_fixture(73)
    samples = _accum_samples(fx)
    g = torch.Generator(device="cuda").manual_seed(97)
    cot = {
        name: torch.randn(fx["tx"], fx["rx"], generator=g, device="cuda")
        for name in ("grad_path_gain", "grad_los", "grad_reflection",
                     "grad_diffraction", "grad_transmission", "grad_scattering")
    }
    # ADR-022 spec 6.4: the coherent forward retains the ten phasor bin sums
    # S_b; the backward reads them (no in-backward re-reduction, no sample
    # coefficients) to form grad_c_r = 2 grad_P[b] S_b.
    _matrices, bin_sums = PA.bdpt_accumulate_connection_samples_forward_ad(
        samples, tx_count=fx["tx"], rx_count=fx["rx"],
        accumulation_strategy="atomic", combine_domain="coherent",
        coeff_real=fx["coeff_real"], coeff_imag=fx["coeff_imag"],
    )
    native = PA.bdpt_accumulate_connection_samples_backward(
        samples, tx_count=fx["tx"], rx_count=fx["rx"], combine_domain="coherent",
        bin_sums=bin_sums,
        **cot, need_grad_contribution=False, need_grad_coeff=True,
    )

    coeff_real = fx["coeff_real"].double().clone().requires_grad_(True)
    coeff_imag = fx["coeff_imag"].double().clone().requires_grad_(True)
    ref = O.accumulate_coherent_reference(
        coeff_real, coeff_imag, fx["tx_id"], fx["rx_id"], fx["component_id"],
        fx["valid"], fx["tx"], fx["rx"],
    )
    loss = sum(
        (cot[f"grad_{name}"].double() * ref[name]).sum()
        for name in ("path_gain", "los", "reflection", "diffraction",
                     "transmission", "scattering")
    )
    loss.backward()
    assert relative_error(
        native["grad_coeff_real"], coeff_real.grad, abs_floor=ABS_TOL
    ) <= _REL_TOL_LOCK
    assert relative_error(
        native["grad_coeff_imag"], coeff_imag.grad, abs_floor=ABS_TOL
    ) <= _REL_TOL_LOCK
    assert native["grad_contribution"] is None


def test_accumulate_power_jvp_vjp_duality():
    fx = _accum_fixture(75)
    samples = _accum_samples(fx)
    g = torch.Generator(device="cuda").manual_seed(98)
    tangent_contribution = torch.randn(fx["rows"], generator=g, device="cuda")
    jvp = PA.bdpt_accumulate_connection_samples_jvp(
        samples, tx_count=fx["tx"], rx_count=fx["rx"], combine_domain="power",
        tangent_contribution=tangent_contribution,
    )
    cot = {
        name: torch.randn(fx["tx"], fx["rx"], generator=g, device="cuda")
        for name in ("tangent_path_gain", "tangent_los", "tangent_reflection",
                     "tangent_diffraction", "tangent_transmission", "tangent_scattering")
    }
    lhs = sum((cot[k].double() * jvp[k].double()).sum() for k in cot)

    vjp = PA.bdpt_accumulate_connection_samples_backward(
        samples, tx_count=fx["tx"], rx_count=fx["rx"], combine_domain="power",
        grad_path_gain=cot["tangent_path_gain"], grad_los=cot["tangent_los"],
        grad_reflection=cot["tangent_reflection"], grad_diffraction=cot["tangent_diffraction"],
        grad_transmission=cot["tangent_transmission"], grad_scattering=cot["tangent_scattering"],
        need_grad_contribution=True, need_grad_coeff=False,
    )
    rhs = (vjp["grad_contribution"].double() * tangent_contribution.double()).sum()
    assert relative_error(lhs, rhs, abs_floor=ABS_TOL) <= _REL_TOL_LOCK


def test_accumulate_coherent_jvp_vjp_duality():
    fx = _accum_fixture(77)
    samples = _accum_samples(fx)
    g = torch.Generator(device="cuda").manual_seed(99)
    t_cr = torch.randn(fx["rows"], generator=g, device="cuda")
    t_ci = torch.randn(fx["rows"], generator=g, device="cuda")
    # Retain the coherent forward bin sums S_b; both the JVP and VJP read them
    # (ADR-022 spec 6.4), so neither companion re-derives them from coeff.
    _matrices, bin_sums = PA.bdpt_accumulate_connection_samples_forward_ad(
        samples, tx_count=fx["tx"], rx_count=fx["rx"],
        accumulation_strategy="atomic", combine_domain="coherent",
        coeff_real=fx["coeff_real"], coeff_imag=fx["coeff_imag"],
    )
    jvp = PA.bdpt_accumulate_connection_samples_jvp(
        samples, tx_count=fx["tx"], rx_count=fx["rx"], combine_domain="coherent",
        bin_sums=bin_sums,
        tangent_coeff_real=t_cr, tangent_coeff_imag=t_ci,
    )
    cot = {
        name: torch.randn(fx["tx"], fx["rx"], generator=g, device="cuda")
        for name in ("tangent_path_gain", "tangent_los", "tangent_reflection",
                     "tangent_diffraction", "tangent_transmission", "tangent_scattering")
    }
    lhs = sum((cot[k].double() * jvp[k].double()).sum() for k in cot)

    vjp = PA.bdpt_accumulate_connection_samples_backward(
        samples, tx_count=fx["tx"], rx_count=fx["rx"], combine_domain="coherent",
        bin_sums=bin_sums,
        grad_path_gain=cot["tangent_path_gain"], grad_los=cot["tangent_los"],
        grad_reflection=cot["tangent_reflection"], grad_diffraction=cot["tangent_diffraction"],
        grad_transmission=cot["tangent_transmission"], grad_scattering=cot["tangent_scattering"],
        need_grad_contribution=False, need_grad_coeff=True,
    )
    rhs = (
        (vjp["grad_coeff_real"].double() * t_cr.double()).sum()
        + (vjp["grad_coeff_imag"].double() * t_ci.double()).sum()
    )
    assert relative_error(lhs, rhs, abs_floor=ABS_TOL) <= _REL_TOL_LOCK


# ---------------------------------------------------------------------------
# 6.5 / 6.6 finalize point components / component maps.
# ---------------------------------------------------------------------------


def _finalize_fixture(seed: int, shape):
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(seed)
    return [
        torch.randn(*shape, generator=g, device=dev, dtype=torch.float32).abs()
        for _ in range(5)
    ]


_COMPONENT_ORDER = ("los", "reflection", "diffraction", "transmission", "scattering")


def _finalize_backward_lockstep(backward_fn, forward_fn, shape, seed):
    components = _finalize_fixture(seed, shape)
    g = torch.Generator(device="cuda").manual_seed(seed + 1)
    grad_path_gain = torch.randn(*shape, generator=g, device="cuda")
    grad_powers = {
        f"grad_{name}_power": torch.randn((), generator=g, device="cuda")
        for name in _COMPONENT_ORDER
    }
    native = backward_fn(
        *components, grad_path_gain=grad_path_gain, **grad_powers,
        need_grad_components=True,
    )

    leaves = [c.double().clone().requires_grad_(True) for c in components]
    ref = forward_fn(*leaves)
    loss = (grad_path_gain.double() * ref["path_gain"]).sum()
    for name in _COMPONENT_ORDER:
        loss = loss + grad_powers[f"grad_{name}_power"].double() * ref[f"{name}_power"]
    loss.backward()
    for i, name in enumerate(_COMPONENT_ORDER):
        assert relative_error(
            native[f"grad_{name}"], leaves[i].grad, abs_floor=ABS_TOL
        ) <= _REL_TOL_LOCK, name


def test_finalize_point_components_backward_matches_oracle():
    _finalize_backward_lockstep(
        M.bdpt_finalize_point_components_backward,
        O.finalize_point_components_reference,
        (3, 4), 81,
    )


def test_finalize_component_maps_backward_matches_oracle():
    _finalize_backward_lockstep(
        M.bdpt_finalize_component_maps_backward,
        O.finalize_component_maps_reference,
        (2, 5, 6), 83,
    )


def test_finalize_point_components_jvp_vjp_duality():
    shape = (3, 4)
    components = _finalize_fixture(85, shape)
    g = torch.Generator(device="cuda").manual_seed(86)
    tangents = {
        f"tangent_{name}": torch.randn(*shape, generator=g, device="cuda")
        for name in _COMPONENT_ORDER
    }
    jvp = M.bdpt_finalize_point_components_jvp(*components, **tangents)
    cot_path = torch.randn(*shape, generator=g, device="cuda")
    cot_powers = {
        f"grad_{name}_power": torch.randn((), generator=g, device="cuda")
        for name in _COMPONENT_ORDER
    }
    lhs = (cot_path.double() * jvp["tangent_path_gain"].double()).sum()
    for name in _COMPONENT_ORDER:
        lhs = lhs + cot_powers[f"grad_{name}_power"].double() * jvp[f"tangent_{name}_power"].double()

    vjp = M.bdpt_finalize_point_components_backward(
        *components, grad_path_gain=cot_path, **cot_powers, need_grad_components=True,
    )
    rhs = sum(
        (vjp[f"grad_{name}"].double() * tangents[f"tangent_{name}"].double()).sum()
        for name in _COMPONENT_ORDER
    )
    assert relative_error(lhs, rhs, abs_floor=ABS_TOL) <= _REL_TOL_LOCK


# ---------------------------------------------------------------------------
# Fixed-input loud rejection + missing-symbol loud failure.
# ---------------------------------------------------------------------------


def test_fixed_inputs_reject_gradients_loudly():
    """Frozen inputs fed through the plan-07 autograd wrappers fail loudly.

    The wrappers ``_ad_reject_fixed_inputs`` any frozen slot instead of silently
    detaching (ADR-014 / ADR-022). Reflected subpath: mu_r and the intersection
    geometry are frozen; requesting their gradient raises.
    """

    from witwin.channel_native.montecarlo.bdpt import autograd as bdpt_autograd

    fx = _reflect_fixture(111)
    light, inter, material = _reflect_native(fx)
    material = dict(material)
    material["material_mu_r"] = material["material_mu_r"].clone().requires_grad_(True)
    out = bdpt_autograd.bdpt_reflected_light_subpath_state_ad(
        light, inter, **material, frequency=torch.tensor(_FREQ, device="cuda")
    )
    with pytest.raises((NotImplementedError, RuntimeError)):
        (out["field_real"].sum() + out["field_imag"].sum()).backward()


def test_ad_mode_none_reflected_has_no_tape():
    fx = _reflect_fixture(113)
    light, inter, material = _reflect_native(fx)
    out = P.bdpt_reflected_light_subpath_state(
        light, inter, **material, frequency_hz=_FREQ
    )
    assert not out["field_real"].requires_grad


def test_companion_symbols_are_registered():
    for name in _BDPT_COMPANION_SYMBOLS:
        assert symbols.has_symbol(name), name


def test_missing_symbol_fails_loudly():
    with pytest.raises(symbols.NativeSymbolError):
        symbols.required_symbol("bdpt_reflected_light_subpath_state_backward_absent")

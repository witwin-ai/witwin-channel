"""Differentiable Kirchhoff table construction (ADR-015 Part C).

The float64 numpy build (``scattering/tables.py::build_kirchhoff_table``) stays
the sanctioned compile-time island and is unchanged bit-for-bit. This module
adds the native derivative: a ``torch.autograd.Function`` whose forward passes
the numpy-built ``f_te``/``f_tm`` through unchanged and whose backward / jvp
dispatch the registered native companions
(``kirchhoff_table_build_backward`` / ``kirchhoff_table_build_jvp``) against the
f32 downcast structural intermediates the build exports (pre-balance
symmetrized lobe ``S``, balance factors ``a``, diffuse budgets ``r_diff``).

Differentiable inputs: ``rough_sigma_h_m`` / ``rough_corr_x_m`` /
``rough_corr_y_m`` and the material CSR ``layer_thickness_m`` / ``layer_eps_r``
/ ``layer_sigma_e`` slices, plus the carrier frequency. Fixed (reject loudly):
``layer_mu_r``, the directional grids (``cos_i`` / ``phi_i`` / ``cos_o`` /
``phi_o``) and ``principal_axis_rad`` (not a table input). This module is
self-contained; it does not touch ``functional.py`` or ``autograd.py``.
"""

from __future__ import annotations

import torch

from witwin.channel.runtime import torch_compat
from witwin.channel.runtime.symbols import required_symbol as _required_native_op
from witwin.channel.runtime.autograd_contracts import (
    _ad_frequency_grad,
    _ad_frequency_value,
    _ad_native_tangent_or_none,
    _ad_native_tensor,
    _ad_reject_fixed_inputs,
    _ad_reject_fixed_tangents,
)

__all__ = [
    "_KirchhoffTableBuildAdFunction",
    "kirchhoff_table_build_ad",
    "kirchhoff_table_build_backward",
    "kirchhoff_table_build_jvp",
]

# Output field contracts of the two native companions.
_BACKWARD_FIELDS = (
    "grad_sigma_h",
    "grad_corr_x",
    "grad_corr_y",
    "grad_layer_thickness_m",
    "grad_layer_eps_r",
    "grad_layer_sigma_e",
    "grad_frequency",
)
_JVP_FIELDS = ("tangent_f_te", "tangent_f_tm")


def _check_table_tensor(name: str, tensor: torch.Tensor, ndim: int) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dtype != torch.float32:
        raise TypeError(f"{name} must have dtype torch.float32")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tensor.dim() != ndim:
        raise ValueError(f"{name} must have rank {ndim}")
    return tensor.contiguous()


def kirchhoff_table_build_backward(
    s_te: torch.Tensor,
    s_tm: torch.Tensor,
    a_te: torch.Tensor,
    a_tm: torch.Tensor,
    r_diff_te: torch.Tensor,
    r_diff_tm: torch.Tensor,
    cos_i: torch.Tensor,
    phi_i: torch.Tensor,
    cos_o: torch.Tensor,
    phi_o: torch.Tensor,
    layer_thickness_m: torch.Tensor,
    layer_eps_r: torch.Tensor,
    layer_sigma_e: torch.Tensor,
    layer_mu_r: torch.Tensor,
    *,
    sigma_h: float,
    corr_x: float,
    corr_y: float,
    frequency_hz: float,
    grad_f_te: torch.Tensor,
    grad_f_tm: torch.Tensor,
    need_grad_rough: bool,
    need_grad_layers: bool,
    need_grad_frequency: bool,
) -> dict[str, torch.Tensor | None]:
    """Native table-build VJP facade (ADR-015 op 3)."""

    s_te = _check_table_tensor("s_te", s_te, 4)
    s_tm = _check_table_tensor("s_tm", s_tm, 4)
    grad_f_te = _check_table_tensor("grad_f_te", grad_f_te, 4)
    grad_f_tm = _check_table_tensor("grad_f_tm", grad_f_tm, 4)
    out = _required_native_op("kirchhoff_table_build_backward")(
        s_te,
        s_tm,
        a_te.contiguous(),
        a_tm.contiguous(),
        r_diff_te.contiguous(),
        r_diff_tm.contiguous(),
        cos_i.contiguous(),
        phi_i.contiguous(),
        cos_o.contiguous(),
        phi_o.contiguous(),
        layer_thickness_m.contiguous(),
        layer_eps_r.contiguous(),
        layer_sigma_e.contiguous(),
        layer_mu_r.contiguous(),
        float(sigma_h),
        float(corr_x),
        float(corr_y),
        float(frequency_hz),
        grad_f_te,
        grad_f_tm,
        bool(need_grad_rough),
        bool(need_grad_layers),
        bool(need_grad_frequency),
    )
    if not isinstance(out, dict) or set(out) != set(_BACKWARD_FIELDS):
        raise TypeError(
            "_channel_native.kirchhoff_table_build_backward returned invalid fields"
        )
    return out


def kirchhoff_table_build_jvp(
    s_te: torch.Tensor,
    s_tm: torch.Tensor,
    a_te: torch.Tensor,
    a_tm: torch.Tensor,
    r_diff_te: torch.Tensor,
    r_diff_tm: torch.Tensor,
    cos_i: torch.Tensor,
    phi_i: torch.Tensor,
    cos_o: torch.Tensor,
    phi_o: torch.Tensor,
    layer_thickness_m: torch.Tensor,
    layer_eps_r: torch.Tensor,
    layer_sigma_e: torch.Tensor,
    layer_mu_r: torch.Tensor,
    *,
    sigma_h: float,
    corr_x: float,
    corr_y: float,
    frequency_hz: float,
    t_layer_thickness_m: torch.Tensor | None,
    t_layer_eps_r: torch.Tensor | None,
    t_layer_sigma_e: torch.Tensor | None,
    t_sigma_h: float,
    t_corr_x: float,
    t_corr_y: float,
    t_frequency: float,
) -> dict[str, torch.Tensor]:
    """Native table-build JVP facade (ADR-015 op 4)."""

    s_te = _check_table_tensor("s_te", s_te, 4)
    s_tm = _check_table_tensor("s_tm", s_tm, 4)
    out = _required_native_op("kirchhoff_table_build_jvp")(
        s_te,
        s_tm,
        a_te.contiguous(),
        a_tm.contiguous(),
        r_diff_te.contiguous(),
        r_diff_tm.contiguous(),
        cos_i.contiguous(),
        phi_i.contiguous(),
        cos_o.contiguous(),
        phi_o.contiguous(),
        layer_thickness_m.contiguous(),
        layer_eps_r.contiguous(),
        layer_sigma_e.contiguous(),
        layer_mu_r.contiguous(),
        float(sigma_h),
        float(corr_x),
        float(corr_y),
        float(frequency_hz),
        None if t_layer_thickness_m is None else t_layer_thickness_m.contiguous(),
        None if t_layer_eps_r is None else t_layer_eps_r.contiguous(),
        None if t_layer_sigma_e is None else t_layer_sigma_e.contiguous(),
        float(t_sigma_h),
        float(t_corr_x),
        float(t_corr_y),
        float(t_frequency),
    )
    if not isinstance(out, dict) or set(out) != set(_JVP_FIELDS):
        raise TypeError(
            "_channel_native.kirchhoff_table_build_jvp returned invalid fields"
        )
    return out


def _scalar_value(tensor: torch.Tensor) -> float:
    return float(_ad_native_tensor(tensor).reshape(()).detach())


def _scalar_tangent(tensor: torch.Tensor | None) -> float:
    value = _ad_native_tangent_or_none(tensor)
    if value is None:
        return 0.0
    return float(value.reshape(()).detach())


# Fixed inputs of the build op (ADR-015 Part C): grads/tangents here fail loudly
# instead of silently detaching.
_FIXED = (
    (6, "layer_mu_r"),
    (16, "cos_i"),
    (17, "phi_i"),
    (18, "cos_o"),
    (19, "phi_o"),
)


def _backward_need_flags(needed) -> tuple[bool, bool, bool]:
    """Resolve which native grad groups the VJP must compute."""

    need_rough = bool(needed[0] or needed[1] or needed[2])
    need_layers = bool(needed[3] or needed[4] or needed[5])
    need_frequency = bool(needed[7])
    return need_rough, need_layers, need_frequency


def _backward_is_noop(
    need_rough: bool,
    need_layers: bool,
    need_frequency: bool,
    grad_f_te: torch.Tensor | None,
    grad_f_tm: torch.Tensor | None,
) -> bool:
    """True when no differentiable input is requested or no upstream grad flows."""

    no_grad_requested = not (need_rough or need_layers or need_frequency)
    no_upstream = grad_f_te is None and grad_f_tm is None
    return no_grad_requested or no_upstream


def _backward_grad_tuple(out, ctx, needed, *, need_frequency: bool) -> tuple:
    """Pack the 21-slot input-gradient tuple from the native VJP output."""

    grad_sigma_h = (
        out["grad_sigma_h"].reshape(ctx.rough_shapes[0]) if needed[0] else None
    )
    grad_corr_x = (
        out["grad_corr_x"].reshape(ctx.rough_shapes[1]) if needed[1] else None
    )
    grad_corr_y = (
        out["grad_corr_y"].reshape(ctx.rough_shapes[2]) if needed[2] else None
    )
    grad_thickness = out["grad_layer_thickness_m"] if needed[3] else None
    grad_eps = out["grad_layer_eps_r"] if needed[4] else None
    grad_sigma = out["grad_layer_sigma_e"] if needed[5] else None
    grad_frequency = (
        _ad_frequency_grad(out["grad_frequency"], ctx.frequency_meta)
        if need_frequency
        else None
    )
    return (
        grad_sigma_h,
        grad_corr_x,
        grad_corr_y,
        grad_thickness,
        grad_eps,
        grad_sigma,
        None,  # layer_mu_r
        grad_frequency,
        None,  # f_te_built
        None,  # f_tm_built
        None,  # s_te
        None,  # s_tm
        None,  # a_te
        None,  # a_tm
        None,  # r_diff_te
        None,  # r_diff_tm
        None,  # cos_i
        None,  # phi_i
        None,  # cos_o
        None,  # phi_o
        None,  # frequency_value
    )


class _KirchhoffTableBuildAdFunction(torch.autograd.Function):
    """Differentiable Kirchhoff table build (ADR-015 Part C).

    Forward passes the numpy-built ``f_te``/``f_tm`` through unchanged;
    backward/jvp dispatch the native companions against the exported f32
    structural intermediates. Differentiable inputs: ``rough_sigma_h_m`` /
    ``rough_corr_x_m`` / ``rough_corr_y_m`` and the CSR ``layer_thickness_m`` /
    ``layer_eps_r`` / ``layer_sigma_e`` slices plus the frequency scalar tensor.
    ``layer_mu_r``, the directional grids and the balance/budget intermediates
    stay fixed; requesting a fixed gradient/tangent fails loudly.
    """

    @staticmethod
    def forward(
        rough_sigma_h,
        rough_corr_x,
        rough_corr_y,
        layer_thickness_m,
        layer_eps_r,
        layer_sigma_e,
        layer_mu_r,
        frequency,
        f_te_built,
        f_tm_built,
        s_te,
        s_tm,
        a_te,
        a_tm,
        r_diff_te,
        r_diff_tm,
        cos_i,
        phi_i,
        cos_o,
        phi_o,
        frequency_value,
    ):
        # The numpy build already produced f_te_built/f_tm_built; the forward is
        # a pure pass-through so the primal table stays bit-identical.
        return f_te_built, f_tm_built

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        frequency = inputs[7]
        ctx.sigma_h = _scalar_value(inputs[0])
        ctx.corr_x = _scalar_value(inputs[1])
        ctx.corr_y = _scalar_value(inputs[2])
        ctx.frequency_value = float(inputs[20])
        ctx.rough_shapes = (
            tuple(inputs[0].shape),
            tuple(inputs[1].shape),
            tuple(inputs[2].shape),
        )
        ctx.frequency_meta = (
            (frequency.dtype, frequency.device)
            if isinstance(frequency, torch.Tensor)
            else None
        )
        # Physics tensors used by both backward and jvp (order matches the
        # facade signatures). Grids (16..19) and CSR params (3..6) included.
        saved = tuple(
            torch.autograd.forward_ad.unpack_dual(inputs[i]).primal
            for i in (10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 3, 4, 5, 6)
        )
        ctx.save_for_backward(*saved)
        ctx.save_for_forward(*saved)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_f_te, grad_f_tm):
        none_grads = (None,) * 21
        _ad_reject_fixed_inputs(
            "kirchhoff_table_build_ad", ctx.needs_input_grad, _FIXED
        )
        needed = ctx.needs_input_grad
        need_rough, need_layers, need_frequency = _backward_need_flags(needed)
        if _backward_is_noop(
            need_rough, need_layers, need_frequency, grad_f_te, grad_f_tm
        ):
            return none_grads
        saved = ctx.saved_tensors
        s_te = saved[0]
        if grad_f_te is None:
            grad_f_te = torch.zeros_like(s_te)
        if grad_f_tm is None:
            grad_f_tm = torch.zeros_like(s_te)
        out = kirchhoff_table_build_backward(
            *saved,
            sigma_h=ctx.sigma_h,
            corr_x=ctx.corr_x,
            corr_y=ctx.corr_y,
            frequency_hz=ctx.frequency_value,
            grad_f_te=grad_f_te,
            grad_f_tm=grad_f_tm,
            need_grad_rough=need_rough,
            need_grad_layers=need_layers,
            need_grad_frequency=need_frequency,
        )
        return _backward_grad_tuple(out, ctx, needed, need_frequency=need_frequency)

    @staticmethod
    def jvp(
        ctx,
        t_sigma_h,
        t_corr_x,
        t_corr_y,
        t_thickness,
        t_eps,
        t_sigma,
        t_mu_r,
        t_frequency,
        _t_f_te,
        _t_f_tm,
        _t_s_te,
        _t_s_tm,
        _t_a_te,
        _t_a_tm,
        _t_r_diff_te,
        _t_r_diff_tm,
        _t_cos_i,
        _t_phi_i,
        _t_cos_o,
        _t_phi_o,
        _t_frequency_value,
    ):
        _ad_reject_fixed_tangents(
            "kirchhoff_table_build_ad",
            (
                (t_mu_r, "layer_mu_r"),
                (_t_cos_i, "cos_i"),
                (_t_phi_i, "phi_i"),
                (_t_cos_o, "cos_o"),
                (_t_phi_o, "phi_o"),
            ),
        )
        saved = ctx.saved_tensors
        tangent_sigma_h = _scalar_tangent(t_sigma_h)
        tangent_corr_x = _scalar_tangent(t_corr_x)
        tangent_corr_y = _scalar_tangent(t_corr_y)
        tangent_frequency = _scalar_tangent(t_frequency)
        tangent_thickness = _ad_native_tangent_or_none(t_thickness)
        tangent_eps = _ad_native_tangent_or_none(t_eps)
        tangent_sigma = _ad_native_tangent_or_none(t_sigma)
        if (
            tangent_sigma_h == 0.0
            and tangent_corr_x == 0.0
            and tangent_corr_y == 0.0
            and tangent_frequency == 0.0
            and tangent_thickness is None
            and tangent_eps is None
            and tangent_sigma is None
        ):
            return None, None
        with torch_compat.disable_functorch():
            out = kirchhoff_table_build_jvp(
                *(_ad_native_tensor(value) for value in saved),
                sigma_h=ctx.sigma_h,
                corr_x=ctx.corr_x,
                corr_y=ctx.corr_y,
                frequency_hz=ctx.frequency_value,
                t_layer_thickness_m=tangent_thickness,
                t_layer_eps_r=tangent_eps,
                t_layer_sigma_e=tangent_sigma,
                t_sigma_h=tangent_sigma_h,
                t_corr_x=tangent_corr_x,
                t_corr_y=tangent_corr_y,
                t_frequency=tangent_frequency,
            )
        return out["tangent_f_te"], out["tangent_f_tm"]


def kirchhoff_table_build_ad(
    rough_sigma_h: torch.Tensor,
    rough_corr_x: torch.Tensor,
    rough_corr_y: torch.Tensor,
    layer_thickness_m: torch.Tensor,
    layer_eps_r: torch.Tensor,
    layer_sigma_e: torch.Tensor,
    layer_mu_r: torch.Tensor,
    frequency: torch.Tensor,
    *,
    table,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attach the native build adjoint to a numpy-built ``KirchhoffTable``.

    ``table`` is the float64-numpy-built :class:`KirchhoffTable` (already carries
    the primal ``f_te``/``f_tm`` and the exported ``pre_balance_lobe_*`` /
    ``normalization_applied`` / ``r_diff_*`` intermediates). The build itself is
    not re-run: the returned ``f_te``/``f_tm`` are the primal values with a
    graph node connecting them to the differentiable leaf inputs.
    """

    if table.pre_balance_lobe_te is None or table.pre_balance_lobe_tm is None:
        raise ValueError(
            "kirchhoff_table_build_ad requires the AD-saved pre-balance lobes; "
            "build the table with build_kirchhoff_table (ADR-015 Part C)"
        )
    frequency_value = _ad_frequency_value(frequency)
    a_te = table.normalization_applied[..., 0].contiguous()
    a_tm = table.normalization_applied[..., 1].contiguous()
    f_te, f_tm = _KirchhoffTableBuildAdFunction.apply(
        rough_sigma_h,
        rough_corr_x,
        rough_corr_y,
        layer_thickness_m,
        layer_eps_r,
        layer_sigma_e,
        layer_mu_r,
        frequency,
        table.f_te,
        table.f_tm,
        table.pre_balance_lobe_te,
        table.pre_balance_lobe_tm,
        a_te,
        a_tm,
        table.r_diff_te,
        table.r_diff_tm,
        table.cos_theta_i,
        table.phi_i,
        table.cos_theta_o,
        table.phi_o,
        float(frequency_value),
    )
    return f_te, f_tm

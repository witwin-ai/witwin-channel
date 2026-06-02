from __future__ import annotations

import drjit as dr

import witwin as wt
from witwin.channel._native import _extension, native_extension_available
from witwin.channel.trace.diffraction.utd import _compute_a_pm, f_utd, fresnel_integral
from witwin.channel.utils.constants import EPS, SMALL_EPS
from witwin.channel.utils.drjit_ops import ArrayInit, complex_abs_sqr, eval_complex
from witwin.channel.utils.polarization import (
    complex_dot_real,
    effective_rx_polarization,
    vector_eval,
    vector_zero,
)
from witwin.channel.utils.shadow_support import shadow_completion_weight_from_distance


def _require_radio_map_accumulate_kernel():
    ext = _extension()
    required = (
        "radiomap_accumulate_vector_power_pairs",
        "radiomap_vector_power_forward_raw",
        "radiomap_vector_power_jvp_raw",
        "radiomap_vector_power_backward_raw",
        "radiomap_matched_isb_completion_forward_raw",
        "radiomap_matched_isb_completion_jvp_raw",
        "radiomap_matched_isb_completion_backward_raw",
        "radiomap_shadow_boundary_incident_stats_forward_raw",
        "radiomap_shadow_boundary_incident_stats_jvp_raw",
        "radiomap_shadow_boundary_incident_stats_backward_raw",
    )
    missing = [name for name in required if not hasattr(ext, name)]
    if missing:
        raise RuntimeError(
            "Native radiomap accumulation kernel requires "
            + ", ".join(missing)
            + ". Rebuild the witwin.channel native extension."
        )
    return ext


def _native_vector_power_ad_available() -> bool:
    if not native_extension_available():
        return False
    ext = _extension()
    return all(
        hasattr(ext, name)
        for name in (
            "radiomap_vector_power_forward_raw",
            "radiomap_vector_power_jvp_raw",
            "radiomap_vector_power_backward_raw",
        )
    )


def _native_matched_isb_ad_available() -> bool:
    if not native_extension_available():
        return False
    ext = _extension()
    return all(
        hasattr(ext, name)
        for name in (
            "radiomap_matched_isb_completion_forward_raw",
            "radiomap_matched_isb_completion_jvp_raw",
            "radiomap_matched_isb_completion_backward_raw",
        )
    )


def _native_shadow_boundary_incident_stats_ad_available() -> bool:
    if not native_extension_available():
        return False
    ext = _extension()
    return all(
        hasattr(ext, name)
        for name in (
            "radiomap_shadow_boundary_incident_stats_forward_raw",
            "radiomap_shadow_boundary_incident_stats_jvp_raw",
            "radiomap_shadow_boundary_incident_stats_backward_raw",
        )
    )


def _zero_complex(width: int):
    return wt.Complex2f(dr.zeros(wt.Float, width), dr.zeros(wt.Float, width))


def _zero_vector(width: int):
    return wt.Vector3f(
        dr.zeros(wt.Float, width),
        dr.zeros(wt.Float, width),
        dr.zeros(wt.Float, width),
    )


def _complex_from_grad(grad_value, width: int):
    if grad_value is None:
        return _zero_complex(width)
    return wt.Complex2f(grad_value.real, grad_value.imag)


def _vector_from_grad(grad_value, width: int):
    if grad_value is None:
        return _zero_vector(width)
    return wt.Vector3f(grad_value.x, grad_value.y, grad_value.z)


def _float_from_grad(grad_value, width: int):
    if grad_value is None:
        return dr.zeros(wt.Float, width)
    return wt.Float(grad_value)


def _bool_mask_to_int(active):
    return wt.Int32(dr.select(active, wt.Int32(1), wt.Int32(0)))


def _mask_to_bool(mask):
    try:
        return mask != wt.Int32(0)
    except Exception:
        return mask


def _array_grad_enabled(value) -> bool:
    if value is None:
        return False
    try:
        return bool(dr.grad_enabled(value))
    except Exception:
        pass
    for axis in ("x", "y", "z", "real", "imag"):
        component = getattr(value, axis, None)
        if component is not None:
            try:
                if dr.grad_enabled(component):
                    return True
            except Exception:
                continue
    return False


def _symbolic_scope_active() -> bool:
    try:
        return bool(dr.flag(dr.JitFlag.SymbolicScope))
    except Exception:
        return False


def _reference_radiomap_vector_power(field_vector):
    return (
        complex_abs_sqr(field_vector["x"])
        + complex_abs_sqr(field_vector["y"])
        + complex_abs_sqr(field_vector["z"])
    )


def _reference_radiomap_matched_isb_completion(
    *,
    continued_direct,
    tx_basis,
    rx_basis,
    hard_visibility,
    interior_mask,
    incident_weight,
    incident_response,
    raw_transition_vector,
):
    side_sign = dr.select(hard_visibility > wt.Float(0.0), wt.Float(1.0), wt.Float(-1.0))
    smooth_coeff = wt.Complex2f(
        wt.Float(0.5) * (wt.Float(1.0) + side_sign * incident_response.real),
        wt.Float(0.5) * (side_sign * incident_response.imag),
    )
    hard_direct_mode = wt.Complex2f(
        continued_direct.real * hard_visibility,
        continued_direct.imag * hard_visibility,
    )
    smooth_direct_mode = wt.Complex2f(
        smooth_coeff.real * continued_direct.real - smooth_coeff.imag * continued_direct.imag,
        smooth_coeff.real * continued_direct.imag + smooth_coeff.imag * continued_direct.real,
    )
    raw_direct_mode = complex_dot_real(raw_transition_vector, tx_basis)
    direct_mode_excess = wt.Complex2f(
        raw_direct_mode.real - hard_direct_mode.real,
        raw_direct_mode.imag - hard_direct_mode.imag,
    )
    completion_mode = wt.Complex2f(
        smooth_direct_mode.real
        - hard_direct_mode.real
        - incident_weight * direct_mode_excess.real,
        smooth_direct_mode.imag
        - hard_direct_mode.imag
        - incident_weight * direct_mode_excess.imag,
    )
    interior_bool = _mask_to_bool(interior_mask)
    completion_mode = wt.Complex2f(
        dr.select(interior_bool, wt.Float(0.0), completion_mode.real),
        dr.select(interior_bool, wt.Float(0.0), completion_mode.imag),
    )
    completion_vector = {
        "x": completion_mode * tx_basis.x,
        "y": completion_mode * tx_basis.y,
        "z": completion_mode * tx_basis.z,
    }
    coherent = eval_complex(complex_dot_real(completion_vector, rx_basis))
    power = _reference_radiomap_vector_power(completion_vector)
    continued_direct_vector = {
        "x": continued_direct * tx_basis.x,
        "y": continued_direct * tx_basis.y,
        "z": continued_direct * tx_basis.z,
    }
    continued_direct_power = _reference_radiomap_vector_power(continued_direct_vector)
    transition_magnitude = dr.sqrt(
        dr.maximum(
            incident_response.real * incident_response.real
            + incident_response.imag * incident_response.imag,
            wt.Float(0.0),
        )
    )
    transition_phase = dr.atan2(incident_response.imag, incident_response.real)
    return {
        "coherent": coherent,
        "vector_coherent": vector_eval(completion_vector),
        "power": power,
        "continued_direct_power": continued_direct_power,
        "transition_magnitude": transition_magnitude,
        "transition_phase": transition_phase,
    }


def _matched_isb_aux_diagnostics(continued_direct, tx_basis, incident_response):
    continued_direct_vector = {
        "x": continued_direct * tx_basis.x,
        "y": continued_direct * tx_basis.y,
        "z": continued_direct * tx_basis.z,
    }
    continued_direct_power = _reference_radiomap_vector_power(continued_direct_vector)
    transition_magnitude = dr.sqrt(
        dr.maximum(
            incident_response.real * incident_response.real
            + incident_response.imag * incident_response.imag,
            wt.Float(0.0),
        )
    )
    transition_phase = dr.atan2(incident_response.imag, incident_response.real)
    return continued_direct_power, transition_magnitude, transition_phase


def _launch_radiomap_vector_power_forward(vec_x, vec_y, vec_z):
    ext = _require_radio_map_accumulate_kernel()
    n_rx = int(dr.width(vec_x.real))
    dr.eval(vec_x.real, vec_x.imag, vec_y.real, vec_y.imag, vec_z.real, vec_z.imag)
    return wt.Float(
        ext.radiomap_vector_power_forward_raw(
            wt.Float(vec_x.real),
            wt.Float(vec_x.imag),
            wt.Float(vec_y.real),
            wt.Float(vec_y.imag),
            wt.Float(vec_z.real),
            wt.Float(vec_z.imag),
            n_rx,
        )
    )


def _launch_radiomap_vector_power_jvp(vec_x, vec_y, vec_z, t_vec_x, t_vec_y, t_vec_z):
    ext = _require_radio_map_accumulate_kernel()
    n_rx = int(dr.width(vec_x.real))
    dr.eval(
        vec_x.real, vec_x.imag, vec_y.real, vec_y.imag, vec_z.real, vec_z.imag,
        t_vec_x.real, t_vec_x.imag, t_vec_y.real, t_vec_y.imag, t_vec_z.real, t_vec_z.imag,
    )
    return wt.Float(
        ext.radiomap_vector_power_jvp_raw(
            wt.Float(vec_x.real),
            wt.Float(vec_x.imag),
            wt.Float(vec_y.real),
            wt.Float(vec_y.imag),
            wt.Float(vec_z.real),
            wt.Float(vec_z.imag),
            wt.Float(t_vec_x.real),
            wt.Float(t_vec_x.imag),
            wt.Float(t_vec_y.real),
            wt.Float(t_vec_y.imag),
            wt.Float(t_vec_z.real),
            wt.Float(t_vec_z.imag),
            n_rx,
        )
    )


def _launch_radiomap_vector_power_backward(vec_x, vec_y, vec_z, grad_power):
    ext = _require_radio_map_accumulate_kernel()
    n_rx = int(dr.width(vec_x.real))
    grads = ext.radiomap_vector_power_backward_raw(
        wt.Float(vec_x.real),
        wt.Float(vec_x.imag),
        wt.Float(vec_y.real),
        wt.Float(vec_y.imag),
        wt.Float(vec_z.real),
        wt.Float(vec_z.imag),
        wt.Float(grad_power),
        n_rx,
    )
    return (
        wt.Complex2f(wt.Float(grads[0]), wt.Float(grads[1])),
        wt.Complex2f(wt.Float(grads[2]), wt.Float(grads[3])),
        wt.Complex2f(wt.Float(grads[4]), wt.Float(grads[5])),
    )


class _RadioMapVectorPowerOp(dr.CustomOp):
    def eval(self, vec_x, vec_y, vec_z):
        self.vec_x = vec_x
        self.vec_y = vec_y
        self.vec_z = vec_z
        return _launch_radiomap_vector_power_forward(vec_x, vec_y, vec_z)

    def forward(self):
        width = int(dr.width(self.vec_x.real))
        self.set_grad_out(
            _launch_radiomap_vector_power_jvp(
                self.vec_x,
                self.vec_y,
                self.vec_z,
                _complex_from_grad(self.grad_in("vec_x"), width),
                _complex_from_grad(self.grad_in("vec_y"), width),
                _complex_from_grad(self.grad_in("vec_z"), width),
            )
        )

    def backward(self):
        grad_x, grad_y, grad_z = _launch_radiomap_vector_power_backward(
            self.vec_x, self.vec_y, self.vec_z, self.grad_out()
        )
        self.set_grad_in("vec_x", grad_x)
        self.set_grad_in("vec_y", grad_y)
        self.set_grad_in("vec_z", grad_z)


def radiomap_vector_power(field_vector):
    if _symbolic_scope_active() or not _native_vector_power_ad_available():
        return _reference_radiomap_vector_power(field_vector)
    return dr.custom(
        _RadioMapVectorPowerOp,
        field_vector["x"],
        field_vector["y"],
        field_vector["z"],
    )


def radiomap_accumulate_vector_power_pairs(
    output_rx_idx,
    pair_vector,
    arrival_dir,
    *,
    n_output_rx: int,
    rx_polarization=None,
):
    ext = _require_radio_map_accumulate_kernel()
    n_pairs = int(dr.width(output_rx_idx))
    n_rx = int(n_output_rx)
    if n_pairs <= 0 or n_rx <= 0:
        coherent = ArrayInit.complex_zero(n_rx)
        power = dr.zeros(wt.Float, n_rx)
        vector = vector_zero(n_rx)
        dr.eval(
            coherent.real, coherent.imag, power,
            vector["x"].real, vector["x"].imag,
            vector["y"].real, vector["y"].imag,
            vector["z"].real, vector["z"].imag,
        )
        return eval_complex(coherent), power, vector_eval(vector), 0
    active_rx_pol = effective_rx_polarization(rx_polarization, (1.0, 0.0, 0.0))
    dr.eval(
        output_rx_idx,
        pair_vector["x"].real, pair_vector["x"].imag,
        pair_vector["y"].real, pair_vector["y"].imag,
        pair_vector["z"].real, pair_vector["z"].imag,
        arrival_dir.x, arrival_dir.y, arrival_dir.z,
    )
    outputs = ext.radiomap_accumulate_vector_power_pairs(
        wt.Int32(output_rx_idx),
        wt.Float(pair_vector["x"].real),
        wt.Float(pair_vector["x"].imag),
        wt.Float(pair_vector["y"].real),
        wt.Float(pair_vector["y"].imag),
        wt.Float(pair_vector["z"].real),
        wt.Float(pair_vector["z"].imag),
        wt.Float(arrival_dir.x),
        wt.Float(arrival_dir.y),
        wt.Float(arrival_dir.z),
        n_rx,
        n_pairs,
        float(active_rx_pol[0]),
        float(active_rx_pol[1]),
        float(active_rx_pol[2]),
    )
    coherent = wt.Complex2f(wt.Float(outputs[0]), wt.Float(outputs[1]))
    vector = {
        "x": wt.Complex2f(wt.Float(outputs[3]), wt.Float(outputs[4])),
        "y": wt.Complex2f(wt.Float(outputs[5]), wt.Float(outputs[6])),
        "z": wt.Complex2f(wt.Float(outputs[7]), wt.Float(outputs[8])),
    }
    return eval_complex(coherent), wt.Float(outputs[2]), vector_eval(vector), int(float(outputs[9][0]))


def _launch_radiomap_matched_isb_forward(
    continued_direct,
    tx_basis,
    rx_basis,
    hard_visibility,
    interior_mask,
    incident_weight,
    incident_response,
    raw_vec_x,
    raw_vec_y,
    raw_vec_z,
):
    ext = _require_radio_map_accumulate_kernel()
    n_rx = int(dr.width(continued_direct.real))
    outputs = ext.radiomap_matched_isb_completion_forward_raw(
        wt.Float(continued_direct.real),
        wt.Float(continued_direct.imag),
        wt.Float(tx_basis.x),
        wt.Float(tx_basis.y),
        wt.Float(tx_basis.z),
        wt.Float(rx_basis.x),
        wt.Float(rx_basis.y),
        wt.Float(rx_basis.z),
        wt.Float(hard_visibility),
        wt.Int32(interior_mask),
        wt.Float(incident_weight),
        wt.Float(incident_response.real),
        wt.Float(incident_response.imag),
        wt.Float(raw_vec_x.real),
        wt.Float(raw_vec_x.imag),
        wt.Float(raw_vec_y.real),
        wt.Float(raw_vec_y.imag),
        wt.Float(raw_vec_z.real),
        wt.Float(raw_vec_z.imag),
        n_rx,
    )
    return outputs


def _launch_radiomap_matched_isb_jvp(
    continued_direct,
    tx_basis,
    rx_basis,
    hard_visibility,
    interior_mask,
    incident_weight,
    incident_response,
    raw_vec_x,
    raw_vec_y,
    raw_vec_z,
    t_continued_direct,
    t_tx_basis,
    t_rx_basis,
    t_incident_weight,
    t_incident_response,
    t_raw_vec_x,
    t_raw_vec_y,
    t_raw_vec_z,
):
    ext = _require_radio_map_accumulate_kernel()
    n_rx = int(dr.width(continued_direct.real))
    return ext.radiomap_matched_isb_completion_jvp_raw(
        wt.Float(continued_direct.real),
        wt.Float(continued_direct.imag),
        wt.Float(tx_basis.x),
        wt.Float(tx_basis.y),
        wt.Float(tx_basis.z),
        wt.Float(rx_basis.x),
        wt.Float(rx_basis.y),
        wt.Float(rx_basis.z),
        wt.Float(hard_visibility),
        wt.Int32(interior_mask),
        wt.Float(incident_weight),
        wt.Float(incident_response.real),
        wt.Float(incident_response.imag),
        wt.Float(raw_vec_x.real),
        wt.Float(raw_vec_x.imag),
        wt.Float(raw_vec_y.real),
        wt.Float(raw_vec_y.imag),
        wt.Float(raw_vec_z.real),
        wt.Float(raw_vec_z.imag),
        wt.Float(t_continued_direct.real),
        wt.Float(t_continued_direct.imag),
        wt.Float(t_tx_basis.x),
        wt.Float(t_tx_basis.y),
        wt.Float(t_tx_basis.z),
        wt.Float(t_rx_basis.x),
        wt.Float(t_rx_basis.y),
        wt.Float(t_rx_basis.z),
        wt.Float(t_incident_weight),
        wt.Float(t_incident_response.real),
        wt.Float(t_incident_response.imag),
        wt.Float(t_raw_vec_x.real),
        wt.Float(t_raw_vec_x.imag),
        wt.Float(t_raw_vec_y.real),
        wt.Float(t_raw_vec_y.imag),
        wt.Float(t_raw_vec_z.real),
        wt.Float(t_raw_vec_z.imag),
        n_rx,
    )


def _launch_radiomap_matched_isb_backward(
    continued_direct,
    tx_basis,
    rx_basis,
    hard_visibility,
    interior_mask,
    incident_weight,
    incident_response,
    raw_vec_x,
    raw_vec_y,
    raw_vec_z,
    grad_outputs,
):
    ext = _require_radio_map_accumulate_kernel()
    n_rx = int(dr.width(continued_direct.real))
    grad_values = list(grad_outputs)
    if len(grad_values) < 12:
        grad_values.extend([None] * (12 - len(grad_values)))
    return ext.radiomap_matched_isb_completion_backward_raw(
        wt.Float(continued_direct.real),
        wt.Float(continued_direct.imag),
        wt.Float(tx_basis.x),
        wt.Float(tx_basis.y),
        wt.Float(tx_basis.z),
        wt.Float(rx_basis.x),
        wt.Float(rx_basis.y),
        wt.Float(rx_basis.z),
        wt.Float(hard_visibility),
        wt.Int32(interior_mask),
        wt.Float(incident_weight),
        wt.Float(incident_response.real),
        wt.Float(incident_response.imag),
        wt.Float(raw_vec_x.real),
        wt.Float(raw_vec_x.imag),
        wt.Float(raw_vec_y.real),
        wt.Float(raw_vec_y.imag),
        wt.Float(raw_vec_z.real),
        wt.Float(raw_vec_z.imag),
        _float_from_grad(grad_values[0], n_rx),
        _float_from_grad(grad_values[1], n_rx),
        _float_from_grad(grad_values[2], n_rx),
        _float_from_grad(grad_values[3], n_rx),
        _float_from_grad(grad_values[4], n_rx),
        _float_from_grad(grad_values[5], n_rx),
        _float_from_grad(grad_values[6], n_rx),
        _float_from_grad(grad_values[7], n_rx),
        _float_from_grad(grad_values[8], n_rx),
        _float_from_grad(grad_values[9], n_rx),
        _float_from_grad(grad_values[10], n_rx),
        _float_from_grad(grad_values[11], n_rx),
        n_rx,
    )


class _RadioMapMatchedIsbCompletionOp(dr.CustomOp):
    def eval(
        self,
        continued_direct_re,
        continued_direct_im,
        tx_basis_x,
        tx_basis_y,
        tx_basis_z,
        rx_basis_x,
        rx_basis_y,
        rx_basis_z,
        incident_weight,
        incident_response_re,
        incident_response_im,
        raw_vec_x_re,
        raw_vec_x_im,
        raw_vec_y_re,
        raw_vec_y_im,
        raw_vec_z_re,
        raw_vec_z_im,
        *,
        hard_visibility,
        interior_mask,
    ):
        self.continued_direct = wt.Complex2f(continued_direct_re, continued_direct_im)
        self.tx_basis = wt.Vector3f(tx_basis_x, tx_basis_y, tx_basis_z)
        self.rx_basis = wt.Vector3f(rx_basis_x, rx_basis_y, rx_basis_z)
        self.hard_visibility = hard_visibility
        self.interior_mask = interior_mask
        self.incident_weight = incident_weight
        self.incident_response = wt.Complex2f(incident_response_re, incident_response_im)
        self.raw_vec_x = wt.Complex2f(raw_vec_x_re, raw_vec_x_im)
        self.raw_vec_y = wt.Complex2f(raw_vec_y_re, raw_vec_y_im)
        self.raw_vec_z = wt.Complex2f(raw_vec_z_re, raw_vec_z_im)
        return _launch_radiomap_matched_isb_forward(
            self.continued_direct,
            self.tx_basis,
            self.rx_basis,
            hard_visibility,
            interior_mask,
            incident_weight,
            self.incident_response,
            self.raw_vec_x,
            self.raw_vec_y,
            self.raw_vec_z,
        )[:9]

    def forward(self):
        width = int(dr.width(self.continued_direct.real))
        t_continued_direct = wt.Complex2f(
            _float_from_grad(self.grad_in("continued_direct_re"), width),
            _float_from_grad(self.grad_in("continued_direct_im"), width),
        )
        t_tx_basis = wt.Vector3f(
            _float_from_grad(self.grad_in("tx_basis_x"), width),
            _float_from_grad(self.grad_in("tx_basis_y"), width),
            _float_from_grad(self.grad_in("tx_basis_z"), width),
        )
        t_rx_basis = wt.Vector3f(
            _float_from_grad(self.grad_in("rx_basis_x"), width),
            _float_from_grad(self.grad_in("rx_basis_y"), width),
            _float_from_grad(self.grad_in("rx_basis_z"), width),
        )
        t_incident_response = wt.Complex2f(
            _float_from_grad(self.grad_in("incident_response_re"), width),
            _float_from_grad(self.grad_in("incident_response_im"), width),
        )
        t_raw_vec_x = wt.Complex2f(
            _float_from_grad(self.grad_in("raw_vec_x_re"), width),
            _float_from_grad(self.grad_in("raw_vec_x_im"), width),
        )
        t_raw_vec_y = wt.Complex2f(
            _float_from_grad(self.grad_in("raw_vec_y_re"), width),
            _float_from_grad(self.grad_in("raw_vec_y_im"), width),
        )
        t_raw_vec_z = wt.Complex2f(
            _float_from_grad(self.grad_in("raw_vec_z_re"), width),
            _float_from_grad(self.grad_in("raw_vec_z_im"), width),
        )
        outputs = _launch_radiomap_matched_isb_jvp(
            self.continued_direct,
            self.tx_basis,
            self.rx_basis,
            self.hard_visibility,
            self.interior_mask,
            self.incident_weight,
            self.incident_response,
            self.raw_vec_x,
            self.raw_vec_y,
            self.raw_vec_z,
            t_continued_direct,
            t_tx_basis,
            t_rx_basis,
            _float_from_grad(self.grad_in("incident_weight"), width),
            t_incident_response,
            t_raw_vec_x,
            t_raw_vec_y,
            t_raw_vec_z,
        )
        self.set_grad_out(outputs[:9])

    def backward(self):
        grads = _launch_radiomap_matched_isb_backward(
            self.continued_direct,
            self.tx_basis,
            self.rx_basis,
            self.hard_visibility,
            self.interior_mask,
            self.incident_weight,
            self.incident_response,
            self.raw_vec_x,
            self.raw_vec_y,
            self.raw_vec_z,
            self.grad_out(),
        )
        self.set_grad_in("continued_direct_re", grads[0])
        self.set_grad_in("continued_direct_im", grads[1])
        self.set_grad_in("tx_basis_x", grads[2])
        self.set_grad_in("tx_basis_y", grads[3])
        self.set_grad_in("tx_basis_z", grads[4])
        self.set_grad_in("rx_basis_x", grads[5])
        self.set_grad_in("rx_basis_y", grads[6])
        self.set_grad_in("rx_basis_z", grads[7])
        self.set_grad_in("incident_weight", grads[8])
        self.set_grad_in("incident_response_re", grads[9])
        self.set_grad_in("incident_response_im", grads[10])
        self.set_grad_in("raw_vec_x_re", grads[11])
        self.set_grad_in("raw_vec_x_im", grads[12])
        self.set_grad_in("raw_vec_y_re", grads[13])
        self.set_grad_in("raw_vec_y_im", grads[14])
        self.set_grad_in("raw_vec_z_re", grads[15])
        self.set_grad_in("raw_vec_z_im", grads[16])


def radiomap_matched_isb_completion(
    *,
    continued_direct,
    tx_basis,
    rx_basis,
    hard_visibility,
    interior_mask,
    incident_weight,
    incident_response,
    raw_transition_vector,
):
    if not _native_matched_isb_ad_available():
        return _reference_radiomap_matched_isb_completion(
            continued_direct=continued_direct,
            tx_basis=tx_basis,
            rx_basis=rx_basis,
            hard_visibility=hard_visibility,
            interior_mask=interior_mask,
            incident_weight=incident_weight,
            incident_response=incident_response,
            raw_transition_vector=raw_transition_vector,
        )
    outputs = dr.custom(
        _RadioMapMatchedIsbCompletionOp,
        continued_direct.real,
        continued_direct.imag,
        tx_basis.x,
        tx_basis.y,
        tx_basis.z,
        rx_basis.x,
        rx_basis.y,
        rx_basis.z,
        incident_weight,
        incident_response.real,
        incident_response.imag,
        raw_transition_vector["x"].real,
        raw_transition_vector["x"].imag,
        raw_transition_vector["y"].real,
        raw_transition_vector["y"].imag,
        raw_transition_vector["z"].real,
        raw_transition_vector["z"].imag,
        hard_visibility=hard_visibility,
        interior_mask=_bool_mask_to_int(_mask_to_bool(interior_mask)),
    )
    continued_direct_power, transition_magnitude, transition_phase = _matched_isb_aux_diagnostics(
        continued_direct,
        tx_basis,
        incident_response,
    )
    return {
        "coherent": eval_complex(wt.Complex2f(outputs[0], outputs[1])),
        "vector_coherent": vector_eval(
            {
                "x": wt.Complex2f(outputs[3], outputs[4]),
                "y": wt.Complex2f(outputs[5], outputs[6]),
                "z": wt.Complex2f(outputs[7], outputs[8]),
            }
        ),
        "power": outputs[2],
        "continued_direct_power": continued_direct_power,
        "transition_magnitude": transition_magnitude,
        "transition_phase": transition_phase,
        "incident_weight": incident_weight,
        "hard_visibility": hard_visibility,
    }


def _reference_shadow_boundary_wedge_exterior_mask(direction_from_edge, edge_dir, n0, nn):
    direction_proj = direction_from_edge - dr.dot(direction_from_edge, edge_dir) * edge_dir
    signed_distance_0 = dr.dot(direction_proj, n0)
    signed_distance_n = dr.dot(direction_proj, nn)
    return (
        (dr.norm(direction_proj) > wt.Float(SMALL_EPS))
        & ((signed_distance_0 >= -wt.Float(SMALL_EPS)) | (signed_distance_n >= -wt.Float(SMALL_EPS)))
    )


def _reference_shadow_boundary_target_support_weight(phi, wedge_n):
    inner_distance = dr.minimum(
        phi - wedge_n * dr.pi,
        wt.Float(2.0 * dr.pi) - phi,
    )
    return shadow_completion_weight_from_distance(inner_distance, wedge_n)


def _reference_shadow_boundary_finite_factor(
    *,
    tx_pos,
    rx_pos,
    edge_pos,
    edge_dir,
    edge_line_min,
    edge_line_max,
    k: float,
):
    edge_hat = edge_dir / (dr.norm(edge_dir) + wt.Float(EPS))
    source_axial = dr.dot(tx_pos - edge_pos, edge_hat)
    target_axial = dr.dot(rx_pos - edge_pos, edge_hat)
    source_to_edge = edge_pos - tx_pos
    edge_to_target = rx_pos - edge_pos
    s_prime_proj = dr.norm(source_to_edge - dr.dot(source_to_edge, edge_hat) * edge_hat) + wt.Float(EPS)
    s_proj = dr.norm(edge_to_target - dr.dot(edge_to_target, edge_hat) * edge_hat) + wt.Float(EPS)
    stationary_u = (
        s_prime_proj * target_axial + s_proj * source_axial
    ) / (s_proj + s_prime_proj + wt.Float(EPS))
    source_offset = stationary_u - source_axial
    target_offset = target_axial - stationary_u
    source_range = dr.sqrt(s_prime_proj * s_prime_proj + source_offset * source_offset + wt.Float(EPS))
    target_range = dr.sqrt(s_proj * s_proj + target_offset * target_offset + wt.Float(EPS))
    curvature = (
        s_prime_proj * s_prime_proj / (source_range * source_range * source_range + wt.Float(EPS))
        + s_proj * s_proj / (target_range * target_range * target_range + wt.Float(EPS))
    )
    scale = dr.sqrt(dr.maximum(wt.Float(k) * curvature, wt.Float(EPS)) / dr.pi)
    delta_f = fresnel_integral(scale * (edge_line_max - stationary_u)) - fresnel_integral(
        scale * (edge_line_min - stationary_u)
    )
    return wt.Complex2f(0.5, 0.5) * dr.conj(delta_f)


def _reference_radiomap_shadow_boundary_incident_statistics(
    *,
    tx_pos,
    rx_pos,
    edge_pos,
    edge_dir,
    n0,
    n_face_n,
    wedge_n,
    edge_line_min,
    edge_line_max,
    source_visible,
    k: float,
    include_diagnostics: bool = False,
):
    n_rx = int(dr.width(rx_pos.x))
    n_edges = int(dr.width(edge_pos.x))
    zero_float = dr.zeros(wt.Float, n_rx)
    negative_one_int = dr.full(wt.Int32, -1, n_rx)
    zero_int = dr.zeros(wt.Int32, n_rx)
    if n_rx <= 0 or n_edges <= 0:
        result = {
            "sum_incident_weight": zero_float,
            "max_incident_weight": zero_float,
            "weighted_incident_response_real": zero_float,
            "weighted_incident_response_imag": zero_float,
        }
        if include_diagnostics:
            result.update(
                {
                    "argmax_edge_idx": negative_one_int,
                    "second_max_incident_weight": zero_float,
                    "support_edge_count": zero_int,
                    "argmax_margin": zero_float,
                }
            )
        return result

    pair_count = n_rx * n_edges
    pair_idx = dr.arange(wt.UInt32, pair_count)
    rx_idx = pair_idx // wt.UInt32(n_edges)
    edge_idx = pair_idx % wt.UInt32(n_edges)
    pair_rx = wt.Point3f(
        dr.gather(wt.Float, rx_pos.x, rx_idx),
        dr.gather(wt.Float, rx_pos.y, rx_idx),
        dr.gather(wt.Float, rx_pos.z, rx_idx),
    )
    zero_pair = dr.zeros(wt.Float, pair_count)
    pair_tx = wt.Point3f(
        zero_pair + tx_pos.x,
        zero_pair + tx_pos.y,
        zero_pair + tx_pos.z,
    )
    pair_edge_pos = wt.Point3f(
        dr.gather(wt.Float, edge_pos.x, edge_idx),
        dr.gather(wt.Float, edge_pos.y, edge_idx),
        dr.gather(wt.Float, edge_pos.z, edge_idx),
    )
    pair_edge_dir = wt.Vector3f(
        dr.gather(wt.Float, edge_dir.x, edge_idx),
        dr.gather(wt.Float, edge_dir.y, edge_idx),
        dr.gather(wt.Float, edge_dir.z, edge_idx),
    )
    pair_n0 = wt.Vector3f(
        dr.gather(wt.Float, n0.x, edge_idx),
        dr.gather(wt.Float, n0.y, edge_idx),
        dr.gather(wt.Float, n0.z, edge_idx),
    )
    pair_nn = wt.Vector3f(
        dr.gather(wt.Float, n_face_n.x, edge_idx),
        dr.gather(wt.Float, n_face_n.y, edge_idx),
        dr.gather(wt.Float, n_face_n.z, edge_idx),
    )
    pair_wedge_n = dr.gather(wt.Float, wedge_n, edge_idx)
    pair_line_min = dr.gather(wt.Float, edge_line_min, edge_idx)
    pair_line_max = dr.gather(wt.Float, edge_line_max, edge_idx)
    pair_source_visible = _mask_to_bool(
        dr.gather(wt.Int32, _bool_mask_to_int(_mask_to_bool(source_visible)), edge_idx)
    )

    edge_dir_hat = pair_edge_dir / (dr.norm(pair_edge_dir) + wt.Float(EPS))
    source_to_edge = pair_edge_pos - pair_tx
    source_to_edge_proj = source_to_edge - dr.dot(source_to_edge, edge_dir_hat) * edge_dir_hat
    s_prime_proj = dr.norm(source_to_edge_proj) + wt.Float(EPS)
    to_hat = dr.normalize(dr.cross(pair_n0, edge_dir_hat))
    ki_proj = source_to_edge_proj / s_prime_proj
    phi_prime = dr.pi - dr.safe_acos(dr.clip(-dr.dot(ki_proj, to_hat), -1.0, 1.0))
    phi_prime = phi_prime * (-dr.sign(-dr.dot(ki_proj, pair_n0)))
    phi_prime = phi_prime + dr.pi

    edge_to_target = pair_rx - pair_edge_pos
    edge_to_target_proj = edge_to_target - dr.dot(edge_to_target, edge_dir_hat) * edge_dir_hat
    s_proj = dr.norm(edge_to_target_proj) + wt.Float(EPS)
    ko_proj = edge_to_target_proj / s_proj
    phi = dr.pi - dr.safe_acos(dr.clip(dr.dot(ko_proj, to_hat), -1.0, 1.0))
    phi = phi * (-dr.sign(dr.dot(ko_proj, pair_n0)))
    phi = phi + dr.pi

    s = dr.norm(edge_to_target) + wt.Float(EPS)
    s_prime = dr.norm(source_to_edge) + wt.Float(EPS)
    kL = wt.Float(k) * s * s_prime * dr.rcp(s + s_prime)
    inc_a0, inc_a1 = _compute_a_pm(phi - phi_prime, pair_wedge_n)
    incident_arg = kL * dr.minimum(inc_a0, inc_a1)
    incident_transition = f_utd(incident_arg)
    transition_mag = dr.sqrt(complex_abs_sqr(incident_transition))
    incident_weight = dr.maximum(
        wt.Float(0.0),
        wt.Float(1.0) - dr.minimum(transition_mag, wt.Float(1.0)),
    )
    finite_factor = _reference_shadow_boundary_finite_factor(
        tx_pos=pair_tx,
        rx_pos=pair_rx,
        edge_pos=pair_edge_pos,
        edge_dir=pair_edge_dir,
        edge_line_min=pair_line_min,
        edge_line_max=pair_line_max,
        k=k,
    )
    finite_scale = dr.minimum(
        dr.sqrt(complex_abs_sqr(finite_factor)),
        wt.Float(1.0),
    )
    incident_transition = finite_factor * incident_transition
    incident_weight = incident_weight * finite_scale

    source_support_mask = (
        _reference_shadow_boundary_wedge_exterior_mask(
            pair_tx - pair_edge_pos,
            edge_dir_hat,
            pair_n0,
            pair_nn,
        )
        & pair_source_visible
        & (pair_wedge_n > wt.Float(1.01))
    )
    target_support_weight = _reference_shadow_boundary_target_support_weight(
        phi,
        pair_wedge_n,
    )
    incident_weight = incident_weight * dr.select(
        source_support_mask,
        target_support_weight,
        wt.Float(0.0),
    )

    sum_incident_weight = dr.block_reduce(
        dr.ReduceOp.Add,
        incident_weight,
        n_edges,
        mode="symbolic",
    )
    max_incident_weight = dr.zeros(wt.Float, n_rx)
    for edge_index in range(n_edges):
        edge_pair_idx = dr.arange(wt.UInt32, n_rx) * wt.UInt32(n_edges) + wt.UInt32(edge_index)
        edge_incident_weight = dr.gather(wt.Float, incident_weight, edge_pair_idx)
        max_incident_weight = dr.maximum(max_incident_weight, edge_incident_weight)
    weighted_incident_response_real = dr.block_reduce(
        dr.ReduceOp.Add,
        incident_weight * incident_transition.real,
        n_edges,
        mode="symbolic",
    )
    weighted_incident_response_imag = dr.block_reduce(
        dr.ReduceOp.Add,
        incident_weight * incident_transition.imag,
        n_edges,
        mode="symbolic",
    )
    result = {
        "sum_incident_weight": sum_incident_weight,
        "max_incident_weight": max_incident_weight,
        "weighted_incident_response_real": weighted_incident_response_real,
        "weighted_incident_response_imag": weighted_incident_response_imag,
    }
    if not include_diagnostics:
        return result

    support_edge_count = dr.zeros(wt.Int32, n_rx)
    argmax_edge_idx = negative_one_int
    second_max_incident_weight = dr.zeros(wt.Float, n_rx)
    for edge_index in range(n_edges):
        edge_pair_idx = dr.arange(wt.UInt32, n_rx) * wt.UInt32(n_edges) + wt.UInt32(edge_index)
        edge_incident_weight = dr.gather(wt.Float, incident_weight, edge_pair_idx)
        edge_active = edge_incident_weight > wt.Float(0.0)
        support_edge_count = support_edge_count + dr.select(
            edge_active,
            wt.Int32(1),
            wt.Int32(0),
        )
        takes_argmax = edge_active & (edge_incident_weight == max_incident_weight) & (
            (argmax_edge_idx < wt.Int32(0)) | (wt.Int32(edge_index) < argmax_edge_idx)
        )
        argmax_edge_idx = dr.select(takes_argmax, wt.Int32(edge_index), argmax_edge_idx)
    for edge_index in range(n_edges):
        edge_pair_idx = dr.arange(wt.UInt32, n_rx) * wt.UInt32(n_edges) + wt.UInt32(edge_index)
        edge_incident_weight = dr.gather(wt.Float, incident_weight, edge_pair_idx)
        second_max_incident_weight = dr.maximum(
            second_max_incident_weight,
            dr.select(
                wt.Int32(edge_index) != argmax_edge_idx,
                edge_incident_weight,
                wt.Float(0.0),
            ),
        )

    result.update(
        {
            "argmax_edge_idx": argmax_edge_idx,
            "second_max_incident_weight": second_max_incident_weight,
            "support_edge_count": support_edge_count,
            "argmax_margin": max_incident_weight - second_max_incident_weight,
        }
    )
    return result


def _shadow_boundary_incident_stats_native_ad_supported_inputs(
    edge_dir,
    n0,
    n_face_n,
    wedge_n,
    edge_line_min,
    edge_line_max,
) -> bool:
    return not any(
        _array_grad_enabled(value)
        for value in (
            edge_dir,
            n0,
            n_face_n,
            wedge_n,
            edge_line_min,
            edge_line_max,
        )
    )


def _launch_radiomap_shadow_boundary_incident_stats_forward(
    tx_pos,
    rx_pos,
    edge_pos,
    edge_dir,
    n0,
    n_face_n,
    wedge_n,
    edge_line_min,
    edge_line_max,
    source_visible,
    *,
    k: float,
):
    ext = _require_radio_map_accumulate_kernel()
    return ext.radiomap_shadow_boundary_incident_stats_forward_raw(
        wt.Float(tx_pos.x),
        wt.Float(tx_pos.y),
        wt.Float(tx_pos.z),
        wt.Float(rx_pos.x),
        wt.Float(rx_pos.y),
        wt.Float(rx_pos.z),
        wt.Float(edge_pos.x),
        wt.Float(edge_pos.y),
        wt.Float(edge_pos.z),
        wt.Float(edge_dir.x),
        wt.Float(edge_dir.y),
        wt.Float(edge_dir.z),
        wt.Float(n0.x),
        wt.Float(n0.y),
        wt.Float(n0.z),
        wt.Float(n_face_n.x),
        wt.Float(n_face_n.y),
        wt.Float(n_face_n.z),
        wt.Float(wedge_n),
        wt.Float(edge_line_min),
        wt.Float(edge_line_max),
        wt.Int32(_bool_mask_to_int(_mask_to_bool(source_visible))),
        int(dr.width(rx_pos.x)),
        int(dr.width(edge_pos.x)),
        float(k),
    )


def _shadow_boundary_incident_stats_outputs_to_dict(outputs, *, include_diagnostics: bool):
    result = {
        "sum_incident_weight": wt.Float(outputs[0]),
        "max_incident_weight": wt.Float(outputs[1]),
        "weighted_incident_response_real": wt.Float(outputs[2]),
        "weighted_incident_response_imag": wt.Float(outputs[3]),
    }
    if not include_diagnostics:
        return result
    second_max_incident_weight = wt.Float(outputs[5])
    result.update(
        {
            "argmax_edge_idx": wt.Int32(outputs[4]),
            "second_max_incident_weight": second_max_incident_weight,
            "support_edge_count": wt.Int32(outputs[6]),
            "argmax_margin": wt.Float(outputs[1]) - second_max_incident_weight,
        }
    )
    return result


def _launch_radiomap_shadow_boundary_incident_stats_jvp(
    tx_pos,
    rx_pos,
    edge_pos,
    edge_dir,
    n0,
    n_face_n,
    wedge_n,
    edge_line_min,
    edge_line_max,
    source_visible,
    argmax_edge_idx,
    t_tx_pos,
    t_rx_pos,
    t_edge_pos,
    *,
    k: float,
):
    ext = _require_radio_map_accumulate_kernel()
    return ext.radiomap_shadow_boundary_incident_stats_jvp_raw(
        wt.Float(tx_pos.x),
        wt.Float(tx_pos.y),
        wt.Float(tx_pos.z),
        wt.Float(rx_pos.x),
        wt.Float(rx_pos.y),
        wt.Float(rx_pos.z),
        wt.Float(edge_pos.x),
        wt.Float(edge_pos.y),
        wt.Float(edge_pos.z),
        wt.Float(edge_dir.x),
        wt.Float(edge_dir.y),
        wt.Float(edge_dir.z),
        wt.Float(n0.x),
        wt.Float(n0.y),
        wt.Float(n0.z),
        wt.Float(n_face_n.x),
        wt.Float(n_face_n.y),
        wt.Float(n_face_n.z),
        wt.Float(wedge_n),
        wt.Float(edge_line_min),
        wt.Float(edge_line_max),
        wt.Int32(_bool_mask_to_int(_mask_to_bool(source_visible))),
        wt.Int32(argmax_edge_idx),
        wt.Float(t_tx_pos.x),
        wt.Float(t_tx_pos.y),
        wt.Float(t_tx_pos.z),
        wt.Float(t_rx_pos.x),
        wt.Float(t_rx_pos.y),
        wt.Float(t_rx_pos.z),
        wt.Float(t_edge_pos.x),
        wt.Float(t_edge_pos.y),
        wt.Float(t_edge_pos.z),
        int(dr.width(rx_pos.x)),
        int(dr.width(edge_pos.x)),
        float(k),
    )


def _launch_radiomap_shadow_boundary_incident_stats_backward(
    tx_pos,
    rx_pos,
    edge_pos,
    edge_dir,
    n0,
    n_face_n,
    wedge_n,
    edge_line_min,
    edge_line_max,
    source_visible,
    argmax_edge_idx,
    grad_outputs,
    *,
    k: float,
):
    ext = _require_radio_map_accumulate_kernel()
    grad_values = list(grad_outputs)
    if len(grad_values) < 4:
        grad_values.extend([None] * (4 - len(grad_values)))
    return ext.radiomap_shadow_boundary_incident_stats_backward_raw(
        wt.Float(tx_pos.x),
        wt.Float(tx_pos.y),
        wt.Float(tx_pos.z),
        wt.Float(rx_pos.x),
        wt.Float(rx_pos.y),
        wt.Float(rx_pos.z),
        wt.Float(edge_pos.x),
        wt.Float(edge_pos.y),
        wt.Float(edge_pos.z),
        wt.Float(edge_dir.x),
        wt.Float(edge_dir.y),
        wt.Float(edge_dir.z),
        wt.Float(n0.x),
        wt.Float(n0.y),
        wt.Float(n0.z),
        wt.Float(n_face_n.x),
        wt.Float(n_face_n.y),
        wt.Float(n_face_n.z),
        wt.Float(wedge_n),
        wt.Float(edge_line_min),
        wt.Float(edge_line_max),
        wt.Int32(_bool_mask_to_int(_mask_to_bool(source_visible))),
        wt.Int32(argmax_edge_idx),
        _float_from_grad(grad_values[0], int(dr.width(rx_pos.x))),
        _float_from_grad(grad_values[1], int(dr.width(rx_pos.x))),
        _float_from_grad(grad_values[2], int(dr.width(rx_pos.x))),
        _float_from_grad(grad_values[3], int(dr.width(rx_pos.x))),
        int(dr.width(rx_pos.x)),
        int(dr.width(edge_pos.x)),
        float(k),
    )


class _RadioMapShadowBoundaryIncidentStatsOp(dr.CustomOp):
    def eval(
        self,
        tx_x,
        tx_y,
        tx_z,
        rx_x,
        rx_y,
        rx_z,
        edge_pos_x,
        edge_pos_y,
        edge_pos_z,
        edge_dir_x,
        edge_dir_y,
        edge_dir_z,
        n0_x,
        n0_y,
        n0_z,
        nn_x,
        nn_y,
        nn_z,
        wedge_n,
        edge_line_min,
        edge_line_max,
        source_visible,
        *,
        k: float,
    ):
        self.tx_pos = wt.Point3f(tx_x, tx_y, tx_z)
        self.rx_pos = wt.Point3f(rx_x, rx_y, rx_z)
        self.edge_pos = wt.Point3f(edge_pos_x, edge_pos_y, edge_pos_z)
        self.edge_dir = wt.Vector3f(edge_dir_x, edge_dir_y, edge_dir_z)
        self.n0 = wt.Vector3f(n0_x, n0_y, n0_z)
        self.n_face_n = wt.Vector3f(nn_x, nn_y, nn_z)
        self.wedge_n = wedge_n
        self.edge_line_min = edge_line_min
        self.edge_line_max = edge_line_max
        self.source_visible = source_visible
        self.k = float(k)
        outputs = _launch_radiomap_shadow_boundary_incident_stats_forward(
            self.tx_pos,
            self.rx_pos,
            self.edge_pos,
            self.edge_dir,
            self.n0,
            self.n_face_n,
            self.wedge_n,
            self.edge_line_min,
            self.edge_line_max,
            self.source_visible,
            k=self.k,
        )
        self.argmax_edge_idx = wt.Int32(outputs[4])
        return outputs[:4]

    def forward(self):
        tx_width = int(dr.width(self.tx_pos.x))
        rx_width = int(dr.width(self.rx_pos.x))
        edge_width = int(dr.width(self.edge_pos.x))
        outputs = _launch_radiomap_shadow_boundary_incident_stats_jvp(
            self.tx_pos,
            self.rx_pos,
            self.edge_pos,
            self.edge_dir,
            self.n0,
            self.n_face_n,
            self.wedge_n,
            self.edge_line_min,
            self.edge_line_max,
            self.source_visible,
            self.argmax_edge_idx,
            wt.Point3f(
                _float_from_grad(self.grad_in("tx_x"), tx_width),
                _float_from_grad(self.grad_in("tx_y"), tx_width),
                _float_from_grad(self.grad_in("tx_z"), tx_width),
            ),
            wt.Point3f(
                _float_from_grad(self.grad_in("rx_x"), rx_width),
                _float_from_grad(self.grad_in("rx_y"), rx_width),
                _float_from_grad(self.grad_in("rx_z"), rx_width),
            ),
            wt.Point3f(
                _float_from_grad(self.grad_in("edge_pos_x"), edge_width),
                _float_from_grad(self.grad_in("edge_pos_y"), edge_width),
                _float_from_grad(self.grad_in("edge_pos_z"), edge_width),
            ),
            k=self.k,
        )
        self.set_grad_out(outputs)

    def backward(self):
        grads = _launch_radiomap_shadow_boundary_incident_stats_backward(
            self.tx_pos,
            self.rx_pos,
            self.edge_pos,
            self.edge_dir,
            self.n0,
            self.n_face_n,
            self.wedge_n,
            self.edge_line_min,
            self.edge_line_max,
            self.source_visible,
            self.argmax_edge_idx,
            self.grad_out(),
            k=self.k,
        )
        self.set_grad_in("tx_x", grads[0])
        self.set_grad_in("tx_y", grads[1])
        self.set_grad_in("tx_z", grads[2])
        self.set_grad_in("rx_x", grads[3])
        self.set_grad_in("rx_y", grads[4])
        self.set_grad_in("rx_z", grads[5])
        self.set_grad_in("edge_pos_x", grads[6])
        self.set_grad_in("edge_pos_y", grads[7])
        self.set_grad_in("edge_pos_z", grads[8])


def radiomap_shadow_boundary_incident_statistics(
    *,
    tx_pos,
    rx_pos,
    edge_pos,
    edge_dir,
    n0,
    n_face_n,
    wedge_n,
    edge_line_min,
    edge_line_max,
    source_visible,
    k: float,
    include_diagnostics: bool = False,
):
    native_supported = (
        _native_shadow_boundary_incident_stats_ad_available()
        and _shadow_boundary_incident_stats_native_ad_supported_inputs(
            edge_dir,
            n0,
            n_face_n,
            wedge_n,
            edge_line_min,
            edge_line_max,
        )
    )
    if not native_supported:
        return _reference_radiomap_shadow_boundary_incident_statistics(
            tx_pos=tx_pos,
            rx_pos=rx_pos,
            edge_pos=edge_pos,
            edge_dir=edge_dir,
            n0=n0,
            n_face_n=n_face_n,
            wedge_n=wedge_n,
            edge_line_min=edge_line_min,
            edge_line_max=edge_line_max,
            source_visible=source_visible,
            k=k,
            include_diagnostics=include_diagnostics,
        )

    requires_ad = any(
        _array_grad_enabled(value)
        for value in (tx_pos, rx_pos, edge_pos)
    )
    if not requires_ad:
        return _shadow_boundary_incident_stats_outputs_to_dict(
            _launch_radiomap_shadow_boundary_incident_stats_forward(
                tx_pos,
                rx_pos,
                edge_pos,
                edge_dir,
                n0,
                n_face_n,
                wedge_n,
                edge_line_min,
                edge_line_max,
                source_visible,
                k=float(k),
            ),
            include_diagnostics=include_diagnostics,
        )

    outputs = dr.custom(
        _RadioMapShadowBoundaryIncidentStatsOp,
        tx_pos.x,
        tx_pos.y,
        tx_pos.z,
        rx_pos.x,
        rx_pos.y,
        rx_pos.z,
        edge_pos.x,
        edge_pos.y,
        edge_pos.z,
        edge_dir.x,
        edge_dir.y,
        edge_dir.z,
        n0.x,
        n0.y,
        n0.z,
        n_face_n.x,
        n_face_n.y,
        n_face_n.z,
        wedge_n,
        edge_line_min,
        edge_line_max,
        _bool_mask_to_int(source_visible),
        k=float(k),
    )
    result = {
        "sum_incident_weight": outputs[0],
        "max_incident_weight": outputs[1],
        "weighted_incident_response_real": outputs[2],
        "weighted_incident_response_imag": outputs[3],
    }
    if not include_diagnostics:
        return result

    diagnostic_outputs = _launch_radiomap_shadow_boundary_incident_stats_forward(
        tx_pos,
        rx_pos,
        edge_pos,
        edge_dir,
        n0,
        n_face_n,
        wedge_n,
        edge_line_min,
        edge_line_max,
        source_visible,
        k=float(k),
    )
    result.update(
        _shadow_boundary_incident_stats_outputs_to_dict(
            diagnostic_outputs,
            include_diagnostics=True,
        )
    )
    result["sum_incident_weight"] = outputs[0]
    result["max_incident_weight"] = outputs[1]
    result["weighted_incident_response_real"] = outputs[2]
    result["weighted_incident_response_imag"] = outputs[3]
    return result


__all__ = [
    "_reference_radiomap_shadow_boundary_incident_statistics",
    "radiomap_accumulate_vector_power_pairs",
    "radiomap_matched_isb_completion",
    "radiomap_shadow_boundary_incident_statistics",
    "radiomap_vector_power",
    "_reference_radiomap_matched_isb_completion",
    "_reference_radiomap_vector_power",
]

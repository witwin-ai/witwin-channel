"""Control-flow contracts for consolidated Python autograd owners.

Native companion numerics are covered by the CUDA lockstep suites.  These tests
call the private ``autograd.Function`` callbacks directly so Coverage.py also
observes their need-flag, saved-tensor, and gradient/tangent packing logic; only
the companion result is replaced with a typed sentinel dictionary.
"""

from __future__ import annotations

from contextlib import nullcontext
import inspect
from types import SimpleNamespace

import pytest
import torch

from witwin.channel.kernels import fields
from witwin.channel.kernels import scattering
from witwin.channel.montecarlo import bdpt


def _patch_ad_plumbing(monkeypatch, module) -> None:
    monkeypatch.setattr(module, "_ad_reject_fixed_inputs", lambda *args: None)
    monkeypatch.setattr(module, "_ad_reject_fixed_tangents", lambda *args: None)
    monkeypatch.setattr(module, "_ad_native_tensor", lambda value: value)
    monkeypatch.setattr(module, "_ad_native_tangent_or_none", lambda value: value)
    monkeypatch.setattr(
        module,
        "_ad_geometry_tangent",
        lambda _label, tangent, _primal: tangent,
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "_ad_frequency_tangent",
        lambda value: 0.0 if value is None else 1.0,
    )
    monkeypatch.setattr(module, "_ad_frequency_grad", lambda value, _meta: value)
    monkeypatch.setattr(module, "disable_functorch", nullcontext)


def _call_named_tangents(function, ctx, **active):
    arguments = {
        name: active.get(name)
        for name in inspect.signature(function).parameters
        if name != "ctx"
    }
    return function(ctx, **arguments)


def test_field_backward_callbacks_pack_requested_gradients(monkeypatch) -> None:
    _patch_ad_plumbing(monkeypatch, fields)
    monkeypatch.setattr(fields, "direction_cotangent", lambda _ctx, value: value)
    scalar = torch.ones(())

    free_output = {
        "grad_source": scalar,
        "grad_target": scalar,
        "grad_frequency": scalar,
    }
    monkeypatch.setattr(fields, "field_free_space_backward", lambda *a, **k: free_output)
    free_ctx = SimpleNamespace(
        needs_input_grad=(True, True, False, False, False, True, False, False),
        saved_tensors=(scalar,) * 5,
        frequency_value=3.0e9,
        frequency_meta=None,
    )
    with torch.no_grad():
        free_grads = fields._FieldFreeSpaceAdFunction.backward(
            free_ctx, scalar, scalar, scalar, scalar, scalar, scalar, scalar
        )
    assert len(free_grads) == 8
    assert free_grads[0] is scalar and free_grads[1] is scalar
    assert free_grads[5] is scalar

    reflection_output = {
        "grad_source": scalar,
        "grad_target": scalar,
        "grad_interaction_positions": scalar,
        "grad_interaction_normals": scalar,
        "grad_eps_r": scalar,
        "grad_sigma_e": scalar,
        "grad_gain": scalar,
        "grad_thickness": scalar,
        "grad_frequency": scalar,
    }
    monkeypatch.setattr(
        fields,
        "field_reflection_sequence_backward",
        lambda *a, **k: reflection_output,
    )
    requested = [False] * 15
    for index in (0, 1, 2, 3, 7, 8, 10, 11, 12):
        requested[index] = True
    reflection_ctx = SimpleNamespace(
        needs_input_grad=tuple(requested),
        saved_tensors=(scalar,) * 12,
        frequency_value=3.0e9,
        frequency_meta=None,
    )
    with torch.no_grad():
        reflection_grads = fields._FieldReflectionSequenceAdFunction.backward(
            reflection_ctx, scalar, scalar, scalar, scalar, scalar, scalar, scalar
        )
    assert len(reflection_grads) == 15
    assert reflection_grads[2] is scalar and reflection_grads[11] is scalar
    assert reflection_grads[12] is scalar


def test_scattering_chain_jvp_callbacks_pack_native_tangents(monkeypatch) -> None:
    _patch_ad_plumbing(monkeypatch, scattering)
    scalar = torch.ones(())

    monkeypatch.setattr(
        scattering,
        "scattering_chain_ensemble_eval_jvp",
        lambda *a, **k: {
            "tangent_gain": scalar,
            "tangent_amplitude": scalar,
            "tangent_length": scalar,
        },
    )
    ensemble_ctx = SimpleNamespace(
        saved_tensors=(scalar,) * 40,
        coef_value=1.0,
        threshold=-1.0,
        frequency_value=3.0e9,
    )
    ensemble = _call_named_tangents(
        scattering._ScatteringChainEnsembleEvalAdFunction.jvp,
        ensemble_ctx,
        t_c1_eps_r=scalar,
        t_coef=scalar,
    )
    assert ensemble == (scalar, scalar, scalar, None)

    monkeypatch.setattr(
        scattering,
        "scattering_chain_realization_eval_jvp",
        lambda *a, **k: {
            "tangent_total": scalar,
            "tangent_path_field": scalar,
            "tangent_path_gain": scalar,
        },
    )
    realization_ctx = SimpleNamespace(
        saved_tensors=(scalar,) * 45,
        k0_value=1.0,
        frequency_value=3.0e9,
    )
    realization = _call_named_tangents(
        scattering._ScatteringChainRealizationEvalAdFunction.jvp,
        realization_ctx,
        t_heights=scalar,
        t_k0=scalar,
    )
    assert realization == (scalar, scalar, scalar, None, None)


def test_kirchhoff_build_callbacks_pack_all_gradient_groups(monkeypatch) -> None:
    _patch_ad_plumbing(monkeypatch, scattering)
    scalar = torch.ones(())
    saved = (scalar,) * 14

    monkeypatch.setattr(
        scattering,
        "kirchhoff_table_build_jvp",
        lambda *a, **k: {"tangent_f_te": scalar, "tangent_f_tm": scalar},
    )
    jvp_ctx = SimpleNamespace(
        saved_tensors=saved,
        sigma_h=0.1,
        corr_x=0.2,
        corr_y=0.3,
        frequency_value=3.0e9,
    )
    tangents = _call_named_tangents(
        scattering._KirchhoffTableBuildAdFunction.jvp,
        jvp_ctx,
        t_sigma_h=scalar,
        t_thickness=scalar,
    )
    assert tangents == (scalar, scalar)

    backward_output = {
        "grad_sigma_h": scalar,
        "grad_corr_x": scalar,
        "grad_corr_y": scalar,
        "grad_layer_thickness_m": scalar,
        "grad_layer_eps_r": scalar,
        "grad_layer_sigma_e": scalar,
        "grad_frequency": scalar,
    }
    monkeypatch.setattr(
        scattering,
        "kirchhoff_table_build_backward",
        lambda *a, **k: backward_output,
    )
    needed = [False] * 21
    for index in (0, 1, 2, 3, 4, 5, 7):
        needed[index] = True
    backward_ctx = SimpleNamespace(
        needs_input_grad=tuple(needed),
        saved_tensors=saved,
        sigma_h=0.1,
        corr_x=0.2,
        corr_y=0.3,
        frequency_value=3.0e9,
        frequency_meta=None,
        rough_shapes=((), (), ()),
    )
    with torch.no_grad():
        gradients = scattering._KirchhoffTableBuildAdFunction.backward(
            backward_ctx, scalar, None
        )
    assert len(gradients) == 21
    assert gradients[:6] == (scalar,) * 6
    assert gradients[7] is scalar


def test_bdpt_subpath_jvp_callbacks_pack_tangent_fields(monkeypatch) -> None:
    _patch_ad_plumbing(monkeypatch, bdpt)
    scalar = torch.ones(())
    tangent_output = {
        "tangent_field_real": scalar,
        "tangent_field_imag": scalar,
        "tangent_throughput_real": scalar,
        "tangent_throughput_imag": scalar,
    }

    monkeypatch.setattr(
        bdpt,
        "bdpt_reflected_light_subpath_state_jvp",
        lambda *a, **k: tangent_output,
    )
    reflected_ctx = SimpleNamespace(
        saved_tensors=(scalar,) * 8,
        base_light={},
        intersection={},
        material_gain=scalar,
        material_valid=scalar,
        frequency_value=3.0e9,
    )
    reflected = _call_named_tangents(
        bdpt._BdptReflectedSubpathAdFunction.jvp,
        reflected_ctx,
        t_field_real=scalar,
    )
    assert reflected[bdpt._SUBPATH_DIFF_INDEX["field_real"]] is scalar

    monkeypatch.setattr(
        bdpt,
        "bdpt_transmitted_light_subpath_state_jvp",
        lambda *a, **k: tangent_output,
    )
    transmitted_ctx = SimpleNamespace(
        saved_tensors=(scalar,) * 8,
        base_light={},
        intersection={},
        face_material_id=scalar,
        layer_offset=scalar,
        layer_count=scalar,
        frequency_value=3.0e9,
    )
    transmitted = _call_named_tangents(
        bdpt._BdptTransmittedSubpathAdFunction.jvp,
        transmitted_ctx,
        t_thickness=scalar,
    )
    assert transmitted[bdpt._SUBPATH_DIFF_INDEX["throughput_imag"]] is scalar


def test_bdpt_endpoint_and_accumulate_jvp_callbacks_pack_schema(monkeypatch) -> None:
    _patch_ad_plumbing(monkeypatch, bdpt)
    scalar = torch.ones(())

    monkeypatch.setattr(
        bdpt,
        "bdpt_endpoint_connection_samples_jvp",
        lambda *a, **k: {"tangent_contribution": scalar},
    )
    endpoint_ctx = SimpleNamespace(
        saved_tensors=(scalar,) * 9,
        base_light={},
        base_sensor={},
        frequency_value=3.0e9,
        params={
            "samples_per_tx": 1,
            "mis": "none",
            "beta": 2.0,
            "strategy_count": 1,
            "max_paths": None,
        },
    )
    endpoint = bdpt._BdptEndpointConnectionAdFunction.jvp(
        endpoint_ctx,
        scalar,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    assert endpoint[bdpt._CONNECTION_CONTRIBUTION_INDEX] is scalar

    matrix_tangents = {
        "tangent_path_gain": scalar,
        "tangent_los": scalar,
        "tangent_reflection": scalar,
        "tangent_diffraction": scalar,
        "tangent_transmission": scalar,
        "tangent_scattering": scalar,
    }
    monkeypatch.setattr(
        bdpt,
        "bdpt_accumulate_connection_samples_jvp",
        lambda *a, **k: matrix_tangents,
    )
    accumulate_ctx = SimpleNamespace(
        saved_tensors=(scalar, scalar, scalar),
        base_samples={},
        tx_count=1,
        rx_count=1,
        combine_domain="power",
        bin_sums=(scalar,),
    )
    accumulated = bdpt._BdptAccumulateAdFunction.jvp(
        accumulate_ctx, scalar, None, None, None, None, None, None, None
    )
    assert accumulated[:6] == (scalar,) * 6
    assert accumulated[6] is None

def test_field_jvp_callbacks_cover_declared_component_matrix(monkeypatch) -> None:
    _patch_ad_plumbing(monkeypatch, fields)
    monkeypatch.setattr(
        fields,
        "_ad_checked_tangent",
        lambda _label, tangent, _shape: tangent,
    )
    value = torch.ones(1)

    def required_symbol(name):
        payloads = {
            "field_diffraction_wedge_jvp": {
                "tangent_field_vector": value,
                "tangent_direction": value,
            },
            "field_coupled_rd_jvp": {
                "tangent_field_vector": value,
                "tangent_coefficient": value,
                "tangent_path_field": value,
                "tangent_path_gain": value,
            },
            "field_coupled_dd_jvp": {
                "tangent_field_vector": value,
                "tangent_coefficient": value,
                "tangent_path_field": value,
                "tangent_path_gain": value,
            },
            "field_project_complex3_jvp": {
                "tangent_coefficient": value,
                "tangent_path_gain": value,
            },
            "field_source_amplitude_scale_jvp": {
                "tangent_path_field_vector": value,
            },
        }
        return lambda *args, **kwargs: payloads[name]

    monkeypatch.setattr(fields, "_required_native_op", required_symbol)
    monkeypatch.setattr(
        fields,
        "field_transmission_sequence_jvp",
        lambda *a, **k: {
            name: value for name in fields._FIELD_AD_TANGENT_FIELDS
        },
    )
    transmission_ctx = SimpleNamespace(
        saved_tensors=(value,) * 16,
        frequency_value=3.0e9,
        geometry_live=True,
    )
    transmission = _call_named_tangents(
        fields._FieldTransmissionSequenceAdFunction.jvp,
        transmission_ctx,
        t_source=value,
        t_layer_eps_r=value,
    )
    assert len(transmission) == 7

    diffraction_ctx = SimpleNamespace(
        saved_tensors=(value,) * 26,
        has_vertices=True,
        frequency_value=3.0e9,
    )
    diffraction_tangents = [None] * 26
    diffraction_tangents[1] = value
    diffraction_tangents[11] = value
    diffraction = fields._FieldDiffractionWedgeAdFunction.jvp(
        diffraction_ctx, *diffraction_tangents
    )
    assert diffraction == (value, value)

    rd_ctx = SimpleNamespace(
        saved_tensors=(value,) * 28,
        frequency_value=3.0e9,
        reverse=False,
    )
    rd_tangents = [None] * 32
    rd_tangents[0] = value
    rd_tangents[12] = value
    rd = fields._FieldCoupledRdAdFunction.jvp(rd_ctx, *rd_tangents)
    assert rd == (value, value, value, value, None)

    dd_ctx = SimpleNamespace(
        saved_tensors=(value,) * 35,
        frequency_value=3.0e9,
    )
    dd_tangents = [None] * 41
    dd_tangents[0] = value
    dd_tangents[15] = value
    dd = fields._FieldCoupledDdAdFunction.jvp(dd_ctx, *dd_tangents)
    assert dd == (value, value, value, value, None)

    monkeypatch.setattr(
        fields,
        "field_rough_reflection_scale_jvp",
        lambda *a, **k: {
            name: value for name in fields._ROUGH_SCALE_TANGENT_FIELDS
        },
    )
    rough_ctx = SimpleNamespace(
        saved_tensors=(value,) * 10,
        frequency_value=3.0e9,
    )
    rough = _call_named_tangents(
        fields._FieldRoughReflectionScaleAdFunction.jvp,
        rough_ctx,
        t_field_vector=value,
        t_positions=value,
    )
    assert len(rough) == len(fields._ROUGH_SCALE_TANGENT_FIELDS)

    project_ctx = SimpleNamespace(saved_tensors=(value,) * 3)
    assert fields._FieldProjectComplex3AdFunction.jvp(
        project_ctx, value, value, None
    ) == (value, value)

    source_ctx = SimpleNamespace(saved_tensors=(value,))
    assert fields._FieldSourceAmplitudeScaleAdFunction.jvp(
        source_ctx, value, None
    ) is value

def test_remaining_vjp_callbacks_pack_full_gradients(monkeypatch) -> None:
    _patch_ad_plumbing(monkeypatch, fields)
    value = torch.ones(1)
    material3 = torch.ones(1, 3)
    material4 = torch.ones(1, 4)

    def required_symbol(name):
        payloads = {
            "field_diffraction_wedge_backward": {
                "grad_source": value,
                "grad_target": value,
                "grad_face0_eps_r": value,
                "grad_face0_sigma_e": value,
                "grad_face0_gain": value,
                "grad_face1_eps_r": value,
                "grad_face1_sigma_e": value,
                "grad_face1_gain": value,
                "grad_frequency": value,
                "grad_vertex_v0": value,
                "grad_vertex_v1": value,
                "grad_vertex_opp0": value,
                "grad_vertex_opp1": value,
            },
            "field_coupled_rd_backward": {
                "grad_source": value,
                "grad_target": value,
                "grad_reflection_position": value,
                "grad_edge_position": value,
                "grad_eps_r": material3,
                "grad_sigma_e": material3,
                "grad_gain": material3,
                "grad_thickness": material3,
                "grad_frequency": value,
            },
            "field_coupled_dd_backward": {
                "grad_source": value,
                "grad_target": value,
                "grad_eps_r": material4,
                "grad_sigma_e": material4,
                "grad_gain": material4,
                "grad_thickness": material4,
                "grad_frequency": value,
            },
            "field_project_complex3_backward": {
                "grad_field_vector": value,
                "grad_direction": value,
            },
            "field_source_amplitude_scale_backward": {
                "grad_field_vector": value,
            },
        }
        return lambda *args, **kwargs: payloads[name]

    monkeypatch.setattr(fields, "_required_native_op", required_symbol)

    diffraction_needed = [False] * 28
    for index in (1, 2, 11, 12, 14, 16, 17, 19, 21, 22, 23, 24, 25):
        diffraction_needed[index] = True
    diffraction_ctx = SimpleNamespace(
        needs_input_grad=tuple(diffraction_needed),
        saved_tensors=(value,) * 26,
        has_vertices=True,
        frequency_value=3.0e9,
        frequency_meta=None,
    )
    with torch.no_grad():
        diffraction = fields._FieldDiffractionWedgeAdFunction.backward(
            diffraction_ctx, value, value
        )
    assert len(diffraction) == 28 and diffraction[25] is value

    rd_needed = [False] * 32
    for index in (0, 1, 2, 4, 12, 13, 15, 16, 17, 18, 20, 21, 22, 23, 25, 26, 27):
        rd_needed[index] = True
    rd_ctx = SimpleNamespace(
        needs_input_grad=tuple(rd_needed),
        saved_tensors=(value,) * 28,
        frequency_value=3.0e9,
        frequency_meta=None,
        reverse=False,
    )
    with torch.no_grad():
        rd = fields._FieldCoupledRdAdFunction.backward(
            rd_ctx, value, value, value, value, None
        )
    assert len(rd) == 32 and torch.equal(rd[26], value)

    dd_needed = [False] * 41
    for index in (0, 1, 15, 16, 18, 19, 20, 21, 23, 24, 25, 26, 28, 29, 30, 31, 33, 34, 35):
        dd_needed[index] = True
    dd_ctx = SimpleNamespace(
        needs_input_grad=tuple(dd_needed),
        saved_tensors=(value,) * 35,
        frequency_value=3.0e9,
        frequency_meta=None,
    )
    with torch.no_grad():
        dd = fields._FieldCoupledDdAdFunction.backward(
            dd_ctx, value, value, value, value, None
        )
    assert len(dd) == 41 and torch.equal(dd[34], value)

    rough_output = {
        "grad_field_vector": value,
        "grad_coefficient": value,
        "grad_path_field": value,
        "grad_path_gain": value,
        "grad_positions": value,
        "grad_normals": value,
        "grad_source": value,
        "grad_frequency": value,
    }
    monkeypatch.setattr(
        fields,
        "field_rough_reflection_scale_backward",
        lambda *a, **k: rough_output,
    )
    rough_needed = [True] * 13
    rough_needed[7] = False
    rough_ctx = SimpleNamespace(
        needs_input_grad=tuple(rough_needed),
        saved_tensors=(value,) * 10,
        frequency_value=3.0e9,
        frequency_meta=None,
    )
    with torch.no_grad():
        rough = fields._FieldRoughReflectionScaleAdFunction.backward(
            rough_ctx, value, value, value, value
        )
    assert len(rough) == 13 and rough[10] is value

    project_ctx = SimpleNamespace(
        needs_input_grad=(True, True, False),
        saved_tensors=(value,) * 3,
    )
    with torch.no_grad():
        project = fields._FieldProjectComplex3AdFunction.backward(
            project_ctx, value, value
        )
    assert project == (value, value, None)

    source_ctx = SimpleNamespace(
        needs_input_grad=(True, False),
        saved_tensors=(value,),
    )
    with torch.no_grad():
        source = fields._FieldSourceAmplitudeScaleAdFunction.backward(
            source_ctx, value
        )
    assert source == (value, None)


def test_scattering_and_bdpt_vjp_callbacks_pack_full_gradients(monkeypatch) -> None:
    _patch_ad_plumbing(monkeypatch, scattering)
    value = torch.ones(1)
    chain_output = {
        "grad_c1_eps_r": value,
        "grad_c1_sigma_e": value,
        "grad_c1_gain": value,
        "grad_c1_thickness": value,
        "grad_c2_eps_r": value,
        "grad_c2_sigma_e": value,
        "grad_c2_gain": value,
        "grad_c2_thickness": value,
        "grad_f_te": value,
        "grad_f_tm": value,
        "grad_coef": value,
        "grad_frequency": value,
    }
    monkeypatch.setattr(
        scattering,
        "scattering_chain_ensemble_eval_backward",
        lambda *a, **k: chain_output,
    )
    chain_needed = [False] * 45
    for index in (8, 9, 11, 12, 16, 17, 19, 20, 35, 36, 40, 41):
        chain_needed[index] = True
    chain_ctx = SimpleNamespace(
        needs_input_grad=tuple(chain_needed),
        saved_tensors=(value,) * 40,
        coef_value=1.0,
        coef_meta=None,
        threshold=-1.0,
        frequency_value=3.0e9,
        frequency_meta=None,
    )
    with torch.no_grad():
        chain = scattering._ScatteringChainEnsembleEvalAdFunction.backward(
            chain_ctx, value, value, value, None
        )
    assert len(chain) == 45 and chain[41] is value

    _patch_ad_plumbing(monkeypatch, bdpt)
    subpath_output = {
        "grad_light_field_real": value,
        "grad_light_field_imag": value,
        "grad_light_throughput_real": value,
        "grad_light_throughput_imag": value,
        "grad_eps_r": value,
        "grad_sigma_e": value,
        "grad_thickness": value,
        "grad_frequency": value,
    }
    monkeypatch.setattr(
        bdpt,
        "bdpt_reflected_light_subpath_state_backward",
        lambda *a, **k: subpath_output,
    )
    subpath_needed = [False] * 14
    for index in (0, 1, 2, 3, 4, 5, 7, 8):
        subpath_needed[index] = True
    subpath_ctx = SimpleNamespace(
        needs_input_grad=tuple(subpath_needed),
        saved_tensors=(value,) * 8,
        base_light={},
        intersection={},
        material_gain=value,
        material_valid=value,
        frequency_value=3.0e9,
        frequency_meta=None,
    )
    grad_outputs = [None] * len(bdpt._SUBPATH_FIELDS)
    for name in bdpt._SUBPATH_DIFF_FIELDS:
        grad_outputs[bdpt._SUBPATH_DIFF_INDEX[name]] = value
    with torch.no_grad():
        subpath = bdpt._BdptReflectedSubpathAdFunction.backward(
            subpath_ctx, *grad_outputs
        )
    assert len(subpath) == 14 and subpath[8] is value

def test_realization_and_transmission_vjp_cover_all_groups(monkeypatch) -> None:
    _patch_ad_plumbing(monkeypatch, scattering)
    value = torch.ones(1)
    realization_names = (
        "d_i",
        "d_o",
        "c1_positions",
        "c1_normals",
        "c1_eps_r",
        "c1_sigma_e",
        "c1_gain",
        "c1_thickness",
        "c2_positions",
        "c2_normals",
        "c2_eps_r",
        "c2_sigma_e",
        "c2_gain",
        "c2_thickness",
        "L1",
        "L2",
        "sp1",
        "sp2",
        "centroids",
        "heights",
        "layer_thickness",
        "layer_eps_r",
        "layer_sigma_e",
        "k0",
        "frequency",
    )
    realization_output = {f"grad_{name}": value for name in realization_names}
    monkeypatch.setattr(
        scattering,
        "scattering_chain_realization_eval_backward",
        lambda *a, **k: realization_output,
    )
    realization_needed = [False] * 49
    for index in (
        4, 5, 10, 11, 12, 13, 15, 16, 18, 19, 20, 21, 23, 24,
        28, 29, 30, 31, 32, 33, 38, 39, 40, 45, 46,
    ):
        realization_needed[index] = True
    realization_ctx = SimpleNamespace(
        needs_input_grad=tuple(realization_needed),
        saved_tensors=(value,) * 45,
        k0_value=1.0,
        k0_meta=None,
        frequency_value=3.0e9,
        frequency_meta=None,
    )
    with torch.no_grad():
        realization = scattering._ScatteringChainRealizationEvalAdFunction.backward(
            realization_ctx, None, value, value, None, None
        )
    assert len(realization) == 49
    assert realization[33] is value and realization[46] is value

    _patch_ad_plumbing(monkeypatch, fields)
    transmission_output = {
        "grad_source": value,
        "grad_target": value,
        "grad_interaction_normals": value,
        "grad_layer_thickness_m": value,
        "grad_layer_eps_r": value,
        "grad_layer_sigma_e": value,
        "grad_frequency": value,
    }
    monkeypatch.setattr(
        fields,
        "field_transmission_sequence_backward",
        lambda *a, **k: transmission_output,
    )
    transmission_needed = [False] * 18
    for index in (1, 2, 4, 12, 13, 14, 16):
        transmission_needed[index] = True
    transmission_ctx = SimpleNamespace(
        needs_input_grad=tuple(transmission_needed),
        saved_tensors=(value,) * 16,
        frequency_value=3.0e9,
        frequency_meta=None,
    )
    with torch.no_grad():
        transmission = fields._FieldTransmissionSequenceAdFunction.backward(
            transmission_ctx, value, value, value, value, value, value, None
        )
    assert len(transmission) == 18
    assert transmission[14] is value and transmission[16] is value

def test_scattering_primitive_vjp_callbacks_cover_all_groups(monkeypatch) -> None:
    _patch_ad_plumbing(monkeypatch, scattering)
    value = torch.ones(1)

    ensemble_names = (
        "wo_rows",
        "r2_rows",
        "cos_o_rows",
        "n_o",
        "t1r",
        "t2r",
        "wi_local",
        "cos_i",
        "r1",
        "a_te2",
        "a_tm2",
        "weights",
        "f_te",
        "f_tm",
        "coef",
    )
    ensemble_output = {f"grad_{name}": value for name in ensemble_names}
    monkeypatch.setattr(
        scattering,
        "scattering_ensemble_eval_backward",
        lambda *a, **k: ensemble_output,
    )
    ensemble_needed = [False] * 26
    for index in (*range(1, 13), 18, 19, 23):
        ensemble_needed[index] = True
    ensemble_ctx = SimpleNamespace(
        needs_input_grad=tuple(ensemble_needed),
        saved_tensors=(value,) * 23,
        coef_value=1.0,
        coef_meta=None,
        threshold=-1.0,
    )
    with torch.no_grad():
        ensemble = scattering._ScatteringEnsembleEvalAdFunction.backward(
            ensemble_ctx, value, value, value, None
        )
    assert len(ensemble) == 26 and ensemble[23] is value

    table_output = {
        "grad_wi": value,
        "grad_wo": value,
        "grad_f_te": value,
        "grad_f_tm": value,
    }
    monkeypatch.setattr(
        scattering,
        "scattering_table_eval_backward",
        lambda *a, **k: table_output,
    )
    table_ctx = SimpleNamespace(
        needs_input_grad=(False, True, True, True, True),
        saved_tensors=(value,) * 5,
    )
    with torch.no_grad():
        table = scattering._ScatteringTableEvalAdFunction.backward(
            table_ctx, value, value
        )
    assert table == (None, value, value, value, value)

    patch_names = (
        "d_i",
        "d_o",
        "r_te",
        "r_tm",
        "r1_rows",
        "r2_rows",
        "centroids",
        "heights",
        "k0",
    )
    patch_output = {f"grad_{name}": value for name in patch_names}
    monkeypatch.setattr(
        scattering,
        "scattering_patch_integral_eval_backward",
        lambda *a, **k: patch_output,
    )
    patch_needed = [False] * 20
    for index in (4, 5, 7, 8, 11, 12, 13, 14, 18):
        patch_needed[index] = True
    patch_ctx = SimpleNamespace(
        needs_input_grad=tuple(patch_needed),
        saved_tensors=(value,) * 18,
        k0_value=1.0,
        k0_meta=None,
    )
    with torch.no_grad():
        patch = scattering._ScatteringPatchIntegralEvalAdFunction.backward(
            patch_ctx, value, None, None
        )
    assert len(patch) == 20 and patch[18] is value

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_deterministic_field_helpers_dispatch_native() -> None:
    path_length = torch.tensor([1.0, 2.0], device="cuda", dtype=torch.float32)
    phase = fields.deterministic_phase_from_length(
        path_length,
        frequency_hz=3.0e9,
    )
    assert phase.shape == path_length.shape

    path_gain = torch.tensor([0.25, 0.5], device="cuda", dtype=torch.float32)
    exported = fields.deterministic_field_from_power_phase(path_gain, phase)
    assert exported["field_real"].shape == path_gain.shape
    assert exported["field_imag"].shape == path_gain.shape

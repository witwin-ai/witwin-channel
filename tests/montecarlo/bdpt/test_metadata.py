import pytest
import torch

from tests.support.scenes import empty_space_los_scene
from witwin.channel.runtime import validate_metadata
from witwin.channel.montecarlo.bdpt import Config, solve


def test_bdpt_metadata_reports_contract_fields():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT metadata")

    config = Config(
        samples=32, seed=123, sample_streams=2, components={"los"}, diagnostics=True
    )
    result = solve(empty_space_los_scene(), config, reference_frequency_hz=3.0e9)

    assert result.metadata["samples"] == 32
    assert result.metadata["seed"] == 123
    assert result.metadata["sample_streams"] == 2
    assert result.metadata["mis"] == "power_heuristic"
    assert result.metadata["throughput_domain"] == "complex3_jones_coherent_events"
    assert result.metadata["field_transport"] == {
        "authoritative_carrier": "complex3_jones",
        "scalar_throughput_role": "sampling_probability_proxy_only",
        "local_frame": "interaction_local_s_p_recomputed_per_event",
        "scattering": "incoherent_power_only_no_complex_field",
        "sensor_depth": "receiver_endpoint_only_always_zero",
    }
    assert (
        result.metadata["pdf_domain"] == "proposal_density_excludes_geometry_jacobian"
    )
    assert (
        result.metadata["sampled_delta_mass"]
        == "event_selection_probability_in_forward_reverse_pdf"
    )
    assert result.metadata["event_classification"]["delta_specular_reflection"] == 1
    assert (
        result.metadata["mis_capabilities"][
            "reflection_diffraction_coupled_bidirectional_pdf"
        ]
        is True
    )
    assert (
        result.metadata["mis_capabilities"]["coupled_pdf_domain"]
        == "enumerated_bidirectional_discrete_mass"
    )
    assert result.metadata["path_counts_by_strategy"]["los"] == 32 * 2 * 2
    # The LoS term connects one deterministic endpoint per transmitter
    # (audit P-1/P-5), so the valid contributions are tx * rx unique rows
    # rather than the nominal sample budget.
    assert (
        result.metadata["valid_contribution_count"]
        == result.path_gain.shape[0] * result.path_gain.shape[1]
    )
    assert result.metadata["components"]["los"] == "enabled"
    assert result.metadata["edge_policy"]["edge_diffraction"] is True
    assert result.metadata["edge_policy"]["boundary_edge_policy"] == "half_plane"
    assert result.metadata["native_capabilities"]["cuda"] is True
    assert result.metadata["variance"] is True
    assert result.metadata["ad_status"] == "none"
    validate_metadata(result.metadata["kernel"])
    assert result.metadata["kernel"]["ad_status"] == "none"
    assert result.diagnostics is not None

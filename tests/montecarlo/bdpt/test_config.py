import pytest

from witwin.channel.montecarlo.bdpt import Config


def test_bdpt_config_defaults_match_public_contract():
    config = Config()

    assert config.samples == 4096
    assert config.seed == 0
    assert config.max_depth == 3
    assert config.max_light_depth == 3
    assert config.max_diffraction_order == 1
    assert config.components == frozenset({"los", "reflection", "diffraction"})
    assert config.mis == "power_heuristic"
    assert config.power_heuristic_beta == 2.0
    assert config.coupled_paths is False
    assert config.coupled_candidate_limit == 1_000_000
    assert config.receiver_strategy == "grid_area"
    assert config.accumulation_strategy == "auto"
    assert config.sample_streams == 1
    assert config.diagnostics is False
    assert config.export_paths is False
    assert config.max_exported_paths is None
    assert config.ad_mode == "none"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"samples": 0}, "samples"),
        ({"seed": -1}, "seed"),
        ({"max_depth": -1}, "max_depth"),
        ({"max_light_depth": -1}, "max_light_depth"),
        ({"max_diffraction_order": 2}, "max_diffraction_order"),
        ({"components": {"scatter"}}, "components"),
        ({"components": set()}, "components"),
        ({"mis": "veach"}, "mis"),
        ({"power_heuristic_beta": 0.0}, "power_heuristic_beta"),
        ({"receiver_strategy": "python_loop"}, "receiver_strategy"),
        ({"accumulation_strategy": "python_loop"}, "accumulation_strategy"),
        ({"sample_streams": 0}, "sample_streams"),
        ({"max_exported_paths": -1}, "max_exported_paths"),
        # ADR-022 lifted the vjp/jvp rejection (they are supported AD modes now),
        # so an unknown ad_mode is the invalid case that must still be rejected.
        ({"ad_mode": "bogus"}, "ad_mode"),
        ({"workspace_limit_bytes": -1}, "workspace_limit_bytes"),
    ],
)
def test_bdpt_config_rejects_invalid_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Config(**kwargs)


@pytest.mark.parametrize("ad_mode", ["vjp", "jvp"])
def test_bdpt_config_accepts_adr022_ad_modes(ad_mode):
    # ADR-022 wires native BDPT AD companions, so vjp/jvp are accepted AD modes
    # and the solver metadata reports the active mode as ad_status.
    from witwin.channel.core.kernels.metadata import AdLaunchLedger
    from witwin.channel.montecarlo.bdpt.metadata import make_solver_metadata

    # components={"los"} keeps the metadata assembly free of a RayD capability
    # requirement so this stays a pure config/metadata contract check.
    config = Config(ad_mode=ad_mode, components={"los"})
    assert config.ad_mode == ad_mode

    metadata = make_solver_metadata(
        config=config,
        selected_accumulation_strategy="atomic",
        path_counts_by_strategy={},
        valid_contribution_count=0,
        reflection_available=False,
        diffraction_available=False,
        cuda_available=False,
        optix_available=False,
        workspace_bytes=0,
        variance_enabled=False,
        launch_count=1,
        effective_max_depth=config.max_depth,
        ad_ledger=AdLaunchLedger(),
    )
    assert metadata["ad_status"] == ad_mode
    assert metadata["kernel"]["ad_status"] == ad_mode


def test_bdpt_config_accepts_supported_variants_and_normalizes_components():
    config = Config(
        samples=16,
        max_light_depth=None,
        max_diffraction_order=0,
        components=["los"],
        mis="balance",
        receiver_strategy="point_sphere",
        accumulation_strategy="compact",
        sample_streams=3,
        export_paths=True,
        max_exported_paths=8,
    )

    assert config.max_light_depth == config.max_depth
    assert config.components == frozenset({"los"})


def test_bdpt_coupled_paths_require_mixed_components_and_depth_two():
    with pytest.raises(RuntimeError, match="max_depth"):
        Config(max_depth=1, coupled_paths=True)
    with pytest.raises(RuntimeError, match="reflection and diffraction"):
        Config(max_depth=2, components={"reflection"}, coupled_paths=True)
    assert Config(max_depth=2, coupled_paths=True).coupled_paths


def test_bdpt_mis_none_selects_one_diffraction_strategy():
    assert Config(components={"diffraction"}, mis="none").mis == "none"

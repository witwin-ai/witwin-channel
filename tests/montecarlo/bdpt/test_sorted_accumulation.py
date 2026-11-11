# Copyright Xingyu Chen.
# Tests the deterministic sorted BDPT accumulation and row-parallel JVP.

import pytest
import torch

import witwin.channel.montecarlo.bdpt as bdpt

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA torch is required")


def test_auto_accumulation_never_selects_staged():
    for samples in (64, 16384, 1 << 20):
        config = bdpt.Config(samples=samples, components={"los"})
        for cells in (0, 1, 64, 4096):
            for ratio in (1.0, 0.05):
                assert (
                    bdpt.select_accumulation_strategy(
                        config, grid_cells=cells, estimated_valid_ratio=ratio
                    )
                    != "staged"
                )


def _random_rows(count: int, *, tx_count: int, rx_count: int, seed: int):
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def ints(high):
        return torch.randint(0, high, (count,), device="cuda", generator=generator, dtype=torch.int32)

    components = torch.tensor([0, 1, 2, 5, 6, 7], device="cuda", dtype=torch.int32)
    tx_id = ints(tx_count + 1) - 1
    rx_id = ints(rx_count)
    component_id = components.index_select(0, ints(6).to(torch.int64))
    depth = torch.zeros((count,), device="cuda", dtype=torch.int32)
    return {
        "topology": torch.stack((tx_id, rx_id, component_id, depth), dim=1),
        "contribution": torch.rand((count,), device="cuda", generator=generator),
        "pdf": torch.ones((count,), device="cuda"),
        "mis_weight": torch.rand((count,), device="cuda", generator=generator),
        "component_id": component_id,
        "valid": torch.rand((count,), device="cuda", generator=generator) < 0.8,
        "tx_id": tx_id,
        "rx_id": rx_id,
        "grid_linear_id": rx_id.clone(),
        "light_depth": depth,
        "sensor_depth": depth.clone(),
        "path_length_m": torch.ones((count,), device="cuda"),
    }


@cuda
def test_staged_sorted_reduction_matches_atomic_and_is_reproducible():
    from witwin.channel.kernels.montecarlo import bdpt_accumulate_connection_samples

    samples = _random_rows(200_000, tx_count=3, rx_count=5, seed=1)
    atomic = bdpt_accumulate_connection_samples(
        samples, tx_count=3, rx_count=5, accumulation_strategy="atomic"
    )
    staged = bdpt_accumulate_connection_samples(
        samples, tx_count=3, rx_count=5, accumulation_strategy="staged"
    )
    repeated = bdpt_accumulate_connection_samples(
        samples, tx_count=3, rx_count=5, accumulation_strategy="staged"
    )
    for name in ("path_gain", "los", "reflection", "diffraction", "transmission", "scattering"):
        torch.testing.assert_close(staged[name], atomic[name], rtol=1.0e-6, atol=0.0)
        assert torch.equal(staged[name], repeated[name])


@cuda
def test_accumulate_jvp_is_row_parallel_and_matches_the_row_sum():
    samples = _random_rows(50_000, tx_count=2, rx_count=4, seed=2)
    tangent = torch.randn((50_000,), device="cuda")
    out = bdpt.bdpt_accumulate_connection_samples_jvp(
        samples, tx_count=2, rx_count=4, combine_domain="power", tangent_contribution=tangent
    )
    again = bdpt.bdpt_accumulate_connection_samples_jvp(
        samples, tx_count=2, rx_count=4, combine_domain="power", tangent_contribution=tangent
    )
    value = (samples["mis_weight"].double() * tangent.double()).cpu()
    keep = (samples["valid"] & (samples["tx_id"] >= 0)).cpu()
    tx = samples["tx_id"].cpu().long()
    rx = samples["rx_id"].cpu().long()
    component = samples["component_id"].cpu()
    for name, ids in (("tangent_transmission", (5,)), ("tangent_path_gain", (0, 1, 2, 5, 6))):
        mask = keep & torch.isin(component, torch.tensor(ids, dtype=torch.int32))
        expected = torch.zeros((2, 4), dtype=torch.float64)
        expected.index_put_((tx[mask], rx[mask]), value[mask], accumulate=True)
        torch.testing.assert_close(out[name].cpu().double(), expected, rtol=1.0e-5, atol=1.0e-9)
        assert torch.equal(out[name], again[name])

# Copyright Xingyu Chen.
# Kirchhoff table cache: reuse by key, fresh device storage, optional disk persistence.

"""Kirchhoff table cache: reuse by key, fresh device storage, optional disk persistence."""

import pytest
import torch

from witwin.core import SurfaceRoughness
from witwin.channel.scene import resources
from witwin.channel.scene.resources import build_kirchhoff_table, clear_kirchhoff_cache

_ROUGH = SurfaceRoughness(
    rms_height_m=1e-3, correlation_length_x_m=10e-3, correlation_length_y_m=10e-3
)
_LAYERS = [(0.1, 4.0, 0.05, 1.0)]
_FIELDS = (
    "cos_theta_i", "phi_i", "cos_theta_o", "phi_o", "f_te", "f_tm", "r_diff_te", "r_diff_tm",
    "r_diff_unpol", "normalization_applied", "sample_density", "marginal_cdf",
    "conditional_cdf", "pre_balance_lobe_te", "pre_balance_lobe_tm",
)


@pytest.fixture(autouse=True)
def _isolated_cache(monkeypatch):
    monkeypatch.delenv(resources.KIRCHHOFF_CACHE_DIR_ENV, raising=False)
    clear_kirchhoff_cache()
    yield
    clear_kirchhoff_cache()


def _counting_builder(monkeypatch) -> list[int]:
    calls: list[int] = []
    original = resources._kirchhoff_table_arrays

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(resources, "_kirchhoff_table_arrays", counted)
    return calls


def _assert_same_table(left, right) -> None:
    for name in _FIELDS:
        assert torch.equal(getattr(left, name), getattr(right, name)), name
    assert left.reciprocity_error == right.reciprocity_error


def test_same_key_reuses_host_build_with_fresh_storage(monkeypatch) -> None:
    calls = _counting_builder(monkeypatch)
    first = build_kirchhoff_table(_ROUGH, _LAYERS, 60e9, device="cpu")
    second = build_kirchhoff_table(_ROUGH, list(_LAYERS), 60e9, device="cpu")
    assert len(calls) == 1
    _assert_same_table(first, second)
    assert first.f_te.data_ptr() != second.f_te.data_ptr()


def test_cached_table_is_bitwise_the_uncached_build(monkeypatch) -> None:
    cached = build_kirchhoff_table(_ROUGH, _LAYERS, 60e9, device="cpu")
    clear_kirchhoff_cache()
    fresh = build_kirchhoff_table(_ROUGH, _LAYERS, 60e9, device="cpu")
    _assert_same_table(cached, fresh)


@pytest.mark.parametrize(
    "change",
    [
        {"frequency_hz": 61e9},
        {"layers": [(0.1, 4.5, 0.05, 1.0)]},
        {"roughness": SurfaceRoughness(
            rms_height_m=1.1e-3, correlation_length_x_m=10e-3, correlation_length_y_m=10e-3
        )},
    ],
)
def test_any_key_change_rebuilds(monkeypatch, change) -> None:
    calls = _counting_builder(monkeypatch)
    build_kirchhoff_table(_ROUGH, _LAYERS, 60e9, device="cpu")
    args = {"roughness": _ROUGH, "layers": _LAYERS, "frequency_hz": 60e9, **change}
    build_kirchhoff_table(args["roughness"], args["layers"], args["frequency_hz"], device="cpu")
    assert len(calls) == 2


def test_domain_refusal_is_not_cached() -> None:
    steep = SurfaceRoughness(
        rms_height_m=5e-3, correlation_length_x_m=10e-3, correlation_length_y_m=10e-3
    )
    for _ in range(2):
        with pytest.raises(ValueError, match="kirchhoff_domain_exceeded"):
            build_kirchhoff_table(steep, _LAYERS, 60e9, device="cpu")


def test_disk_cache_round_trip(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(resources.KIRCHHOFF_CACHE_DIR_ENV, str(tmp_path))
    built = build_kirchhoff_table(_ROUGH, _LAYERS, 60e9, device="cpu")
    files = list(tmp_path.glob("kirchhoff-*.npz"))
    assert len(files) == 1
    clear_kirchhoff_cache()

    def forbidden(*args, **kwargs):
        raise AssertionError("disk cache hit must not rebuild")

    monkeypatch.setattr(resources, "_kirchhoff_table_arrays", forbidden)
    loaded = build_kirchhoff_table(_ROUGH, _LAYERS, 60e9, device="cpu")
    _assert_same_table(built, loaded)


def test_corrupt_disk_entry_rebuilds(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(resources.KIRCHHOFF_CACHE_DIR_ENV, str(tmp_path))
    build_kirchhoff_table(_ROUGH, _LAYERS, 60e9, device="cpu")
    (entry,) = tmp_path.glob("kirchhoff-*.npz")
    entry.write_bytes(b"not an npz archive")
    clear_kirchhoff_cache()
    calls = _counting_builder(monkeypatch)
    build_kirchhoff_table(_ROUGH, _LAYERS, 60e9, device="cpu")
    assert len(calls) == 1

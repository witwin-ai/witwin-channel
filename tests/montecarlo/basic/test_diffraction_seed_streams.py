# Copyright Xingyu Chen.
# Tests that MC Basic diffraction seeds give independent lane-hashed sample streams.

from __future__ import annotations

import pytest

from witwin.channel.montecarlo.basic import lane_stream_seed


def _blocks(seeds, lane_count: int) -> set[int]:
    # The samplers draw hash(lane ^ constant ^ seed) for lane < lane_count, so two seeds
    # reuse the same hash inputs exactly when they agree above log2(lane_count) bits.
    shift = lane_count.bit_length() - 1
    return {seed >> shift for seed in seeds}


@pytest.mark.parametrize(("lane_count", "seed_count"), [(4096, 1000), (262144, 20)])
def test_nearby_seeds_draw_disjoint_lane_hash_inputs(lane_count: int, seed_count: int):
    # With a 31-bit seed two mixed seeds still share a block with probability
    # lane_count / 2**31; the seed ranges here are the ones the tests and studies use.
    seeds = range(1, seed_count + 1)
    mixed = [lane_stream_seed(seed) for seed in seeds]

    # Raw seeds below the lane count all fall in one block: every seed permutes one set.
    assert len(_blocks(seeds, lane_count)) == 1
    assert len(_blocks(mixed, lane_count)) == len(mixed)


def test_lane_stream_seed_is_a_deterministic_31_bit_value():
    values = [lane_stream_seed(seed) for seed in (0, 1, 2, 7, 2**31, 2**62)]

    assert values == [lane_stream_seed(seed) for seed in (0, 1, 2, 7, 2**31, 2**62)]
    assert all(0 <= value < 2**31 for value in values)
    assert len(set(values)) == len(values)

from __future__ import annotations

import drjit as dr
import pytest

from witwin.channel.core.scene import Scene
from witwin.core import Box, Material, Structure
from witwin.channel.montecarlo import types as wt


@pytest.mark.gpu
def test_rayd_segment_pair_visibility_matches_drjit_reference():
    scene = Scene(
        structures=[
            Structure(
                geometry=Box(
                    position=(0.0, 0.0, 0.5),
                    size=(1.0, 1.0, 1.0),
                    device="cuda",
                ),
                material=Material(),
                name="box",
            )
        ]
    )
    start = wt.Point3f(
        wt.Float([-2.0, -2.0, 0.0, 0.0]),
        wt.Float([0.2, 2.0, -2.0, 0.0]),
        wt.Float([0.5, 0.5, 0.5, 2.0]),
    )
    end = wt.Point3f(
        wt.Float([2.0, 2.0, 0.0, 0.0]),
        wt.Float([0.2, 2.0, 2.0, 0.0]),
        wt.Float([0.5, 0.5, 0.5, 2.0]),
    )
    end_offset = wt.Point3f(
        end.x,
        end.y + wt.Float([0.02, 0.02, 0.02, 0.0]),
        end.z,
    )

    ref = scene.segment_visible(start, end) & scene.segment_visible(start, end_offset)
    got = scene.segment_pair_visible(start, end, end_offset)
    dr.eval(ref, got)
    assert not bool(dr.any(ref != got))

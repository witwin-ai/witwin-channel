from __future__ import annotations

import drjit as dr
import pytest
from witwin.channel import cuda_runtime_version, native_extension_available, sample_add_one
@pytest.mark.gpu
def test_native_sample_add_one_round_trip():
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    values = dr.cuda.Float([1.0, 2.0, 3.0])
    result = sample_add_one(values)

    assert cuda_runtime_version() > 0
    assert bool(dr.all(result == dr.cuda.Float([2.0, 3.0, 4.0])))


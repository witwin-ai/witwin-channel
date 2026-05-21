import pytest
import numpy as np

from examples.deterministic_radiomap_single_cube import SingleCubeExperiment
from witwin.channel.deterministic.diffraction import postprocessing


def _line_max_adjacent_db_jumps(*, y: float, correction: bool) -> tuple[float, float]:
    experiment = SingleCubeExperiment(
        bounds=((-8.0, 8.0), (float(y) - 1e-3, float(y) + 1e-3)),
        grid_shape=(121, 1),
        forward_num_samples=192,
        max_bounces=2,
        max_diffraction_order=1,
        shadow_boundary_correction=correction,
    )
    result = experiment.solve()
    path_gain = np.asarray(result.path_gain, dtype=np.float64).reshape(1, -1)[0]
    diffraction = np.asarray(result.components["diffraction"], dtype=np.float64).reshape(1, -1)[0]
    path_gain_db = 10.0 * np.log10(np.maximum(path_gain, 1e-30))
    diffraction_db = 10.0 * np.log10(np.maximum(diffraction, 1e-30))
    return (
        float(np.max(np.abs(np.diff(path_gain_db)))),
        float(np.max(np.abs(np.diff(diffraction_db)))),
    )


@pytest.mark.gpu
def test_shadow_boundary_correction_does_not_amplify_single_cube_y4_jump():
    raw_jump, raw_diffraction_jump = _line_max_adjacent_db_jumps(y=4.0, correction=False)
    corrected_jump, corrected_diffraction_jump = _line_max_adjacent_db_jumps(y=4.0, correction=True)

    assert corrected_jump <= raw_jump + 1.0
    assert corrected_diffraction_jump <= raw_diffraction_jump - 2.0


def test_dense_shadow_boundary_guard_allows_small_workloads():
    postprocessing.validate_dense_shadow_boundary_workload(n_edges=64, n_rx=1024)


def test_dense_shadow_boundary_guard_rejects_large_dense_workloads():
    with pytest.raises(RuntimeError, match="candidate-pruned backend"):
        postprocessing.validate_dense_shadow_boundary_workload(n_edges=51631, n_rx=256)


def test_auto_shadow_boundary_backend_keeps_dense_for_small_workloads():
    backend = postprocessing.resolve_shadow_boundary_statistics_backend(
        n_edges=64,
        n_rx=1024,
        requested_backend="auto",
        native_candidate_available=False,
    )

    assert backend == "dense_native"


def test_auto_shadow_boundary_backend_uses_candidate_for_large_workloads():
    backend = postprocessing.resolve_shadow_boundary_statistics_backend(
        n_edges=51631,
        n_rx=256,
        requested_backend="auto",
        native_candidate_available=True,
    )

    assert backend == "native_candidate"


def test_auto_shadow_boundary_backend_rejects_large_without_candidate_backend():
    with pytest.raises(RuntimeError, match="candidate-pruned backend"):
        postprocessing.resolve_shadow_boundary_statistics_backend(
            n_edges=51631,
            n_rx=256,
            requested_backend="auto",
            native_candidate_available=False,
        )


def test_candidate_shadow_boundary_backend_rejects_unavailable_native_kernel():
    with pytest.raises(RuntimeError, match="native_candidate"):
        postprocessing.resolve_shadow_boundary_statistics_backend(
            n_edges=64,
            n_rx=1024,
            requested_backend="native_candidate",
            native_candidate_available=False,
        )


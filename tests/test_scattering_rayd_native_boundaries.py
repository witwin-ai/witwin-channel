from __future__ import annotations

import hashlib
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KERNELS = ROOT / "native/channel/kernels"
BINDING = ROOT / "native/channel/binding/materials.cpp"
RAYD_ROOT = Path(os.environ.get("RAYD_SOURCE_DIR", ROOT.parent.parent / "RayD"))

MOVED_TUS = {
    "scattering_chain_ensemble.cu",
    "scattering_chain_ensemble_ad.cu",
    "scattering_chain_realization.cu",
    "scattering_chain_realization_ad.cu",
    "scattering_ensemble.cu",
    "scattering_ensemble_ad.cu",
    "scattering_patch_integral.cu",
    "scattering_patch_integral_ad.cu",
    "scattering_table_eval_ad.cu",
}
TYPED_ENTRIES = {
    "scattering_chain_ensemble_eval",
    "scattering_chain_ensemble_eval_backward",
    "scattering_chain_ensemble_eval_jvp",
    "scattering_chain_realization_eval",
    "scattering_chain_realization_eval_backward",
    "scattering_chain_realization_eval_jvp",
    "scattering_table_eval",
    "scattering_table_eval_backward",
    "scattering_table_eval_jvp",
    "scattering_table_sample",
    "scattering_table_pdf",
    "scattering_ensemble_eval",
    "scattering_ensemble_eval_backward",
    "scattering_ensemble_eval_jvp",
    "scattering_patch_integral_eval",
    "scattering_patch_integral_eval_backward",
    "scattering_patch_integral_eval_jvp",
}


def _sha256(path: Path) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()


def test_current_build_uses_the_locked_rayd_scattering_surface() -> None:
    integration = RAYD_ROOT / "backends/torch/include/rayd/torch/integration.h"
    assert _sha256(integration) == (
        "57f83ea460e376166fd5ee22a8243a7c1576a290e1de99c0cbe8e86e93392e14"
    )
    integration_text = integration.read_text(encoding="utf-8-sig")
    assert 'kIntegrationApiVersion = 6;' in integration_text
    assert '"rayd.torch.integration";' in integration_text
    assert "integration_v2" not in integration_text
    assert "rayd.torch.integration.v2" not in integration_text
    assert _sha256(
        RAYD_ROOT / "backends/torch/include/rayd/torch/rf/scattering.h"
    ) == "7a29ff216f11a08256ee271ef5dcad817e4b8379d88bc07772685fa3da439aa9"
    assert _sha256(
        RAYD_ROOT / "shared/include/rayd/shared/rf/scattering_table.cuh"
    ) == "38ea9be424640301a88a97bccca9ab4bc599191ecfb0b259881ef6a300c96e38"


def test_phase10b_binding_has_one_typed_adapter_per_entry() -> None:
    source = BINDING.read_text(encoding="utf-8-sig")
    assert source.count("#include <rayd/torch/integration.h>") == 1
    assert "<<<" not in source
    for entry in TYPED_ENTRIES:
        assert source.count(f"rayd::torch::{entry}(") == 1


def test_phase10b_channel_has_no_local_scattering_numerical_owner() -> None:
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8-sig")
    assert not (KERNELS / "scattering_table.cuh").exists()
    for name in MOVED_TUS:
        assert not (KERNELS / name).exists()
        assert f"native/channel/kernels/{name}" not in cmake


def test_phase10b_retains_only_event_policy_in_scattering_tu() -> None:
    source = (KERNELS / "scattering.cu").read_text(encoding="utf-8-sig")
    assert source.count("scattering_event_kernel<<<") == 1
    assert source.count("channel_scattering_event_probabilities(") == 1
    for removed in (
        "scattering_eval_kernel",
        "scattering_pdf_kernel",
        "scattering_sample_kernel",
        "channel_scattering_table_eval",
        "channel_scattering_table_pdf",
        "channel_scattering_table_sample",
    ):
        assert removed not in source
    assert "--fmad=false" not in source


def test_phase10b_cmake_has_no_local_chain_compile_contract() -> None:
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8-sig")
    for name in (
        "scattering_chain_ensemble.cu",
        "scattering_chain_ensemble_ad.cu",
        "scattering_chain_realization.cu",
        "scattering_chain_realization_ad.cu",
    ):
        assert name not in cmake

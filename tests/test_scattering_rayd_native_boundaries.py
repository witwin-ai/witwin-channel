from __future__ import annotations

import hashlib
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KERNELS = ROOT / "native/channel_native/kernels"
BINDING = ROOT / "native/channel_native/binding/materials.cpp"
RAYD_ROOT = Path(os.environ.get("RAYD_SOURCE_DIR", ROOT.parent.parent / "RayDi"))

MOVED_TUS = {
    "scattering_ensemble.cu",
    "scattering_ensemble_ad.cu",
    "scattering_patch_integral.cu",
    "scattering_patch_integral_ad.cu",
    "scattering_table_eval_ad.cu",
}
TYPED_ENTRIES = {
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


def test_phase10a_uses_the_locked_rayd_scattering_surface() -> None:
    assert _sha256(
        RAYD_ROOT / "backends/torch/include/rayd/torch/integration_v2.h"
    ) == "9f95ad9e8e3b790d00f8e762a3e6a09252d46afb65bfc3aba7c42325836cb1fb"
    assert _sha256(
        RAYD_ROOT / "backends/torch/include/rayd/torch/rf/scattering.h"
    ) == "66d75a20be16057f03cdfb79e3b9dcc85cacec79b555cd73b019259aa510262a"
    assert _sha256(
        RAYD_ROOT / "shared/include/rayd/shared/rf/scattering_table.cuh"
    ) == "38ea9be424640301a88a97bccca9ab4bc599191ecfb0b259881ef6a300c96e38"


def test_phase10a_binding_has_one_typed_adapter_per_entry() -> None:
    source = BINDING.read_text(encoding="utf-8-sig")
    assert source.count("#include <rayd/torch/integration_v2.h>") == 1
    assert "<<<" not in source
    for entry in TYPED_ENTRIES:
        assert source.count(f"rayd::torch::{entry}(") == 1


def test_phase10a_channel_has_no_local_table_ensemble_or_patch_owner() -> None:
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8-sig")
    assert not (KERNELS / "scattering_table.cuh").exists()
    for name in MOVED_TUS:
        assert not (KERNELS / name).exists()
        assert f"native/channel_native/kernels/{name}" not in cmake


def test_phase10a_retains_only_event_policy_in_scattering_tu() -> None:
    source = (KERNELS / "scattering.cu").read_text(encoding="utf-8-sig")
    assert source.count("scattering_event_kernel<<<") == 1
    assert source.count("cn_scattering_event_probabilities(") == 1
    for removed in (
        "scattering_eval_kernel",
        "scattering_pdf_kernel",
        "scattering_sample_kernel",
        "cn_scattering_table_eval",
        "cn_scattering_table_pdf",
        "cn_scattering_table_sample",
    ):
        assert removed not in source
    assert "--fmad=false" not in source


def test_phase10a_retained_chain_consumers_use_public_table_header() -> None:
    public_include = "#include <rayd/shared/rf/scattering_table.cuh>"
    for name in ("scattering_chain_ensemble.cu", "scattering_chain_ensemble_ad.cu"):
        source = (KERNELS / name).read_text(encoding="utf-8-sig")
        assert source.count(public_include) == 1
        assert '#include "scattering_table.cuh"' not in source

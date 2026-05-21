"""Python loader for the Monte Carlo radiomap native extension."""

from __future__ import annotations

from witwin.channel._native import PACKAGE_PATH
from witwin.channel._native.loader import (
    NativeExtensionLoader,
    NativeExtensionSpec,
)


NativeExtension = NativeExtensionLoader(
    NativeExtensionSpec(
        module_name="witwin.channel._native._monte_carlo_radiomap_native",
        install_subpath=("witwin", "channel", "_native"),
        build_glob="witwin/channel/_native/monte_carlo/*",
        binary_glob="_monte_carlo_radiomap_native*",
        probe_env_var="WITWIN_MC_RADIOMAP_NATIVE_PROBE",
        probe_module_name="witwin.channel._native.montecarlo",
        error_description="witwin.channel.montecarlo native extension",
    ),
    package_path=PACKAGE_PATH,
)


__all__ = ["NativeExtension"]

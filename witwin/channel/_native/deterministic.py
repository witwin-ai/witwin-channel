"""Python loader for the bundled deterministic radiomap native extension."""

from __future__ import annotations

from witwin.channel._native import PACKAGE_PATH
from witwin.channel._native.loader import (
    NativeExtensionLoader,
    NativeExtensionSpec,
)


NativeExtension = NativeExtensionLoader(
    NativeExtensionSpec(
        module_name="witwin.channel._native._deterministic_radiomap_native",
        install_subpath=("witwin", "channel", "_native"),
        build_glob="witwin/channel/_native/deterministic/*",
        binary_glob="_deterministic_radiomap_native*",
        probe_env_var="WITWIN_DETERMINISTIC_NATIVE_PROBE",
        probe_module_name="witwin.channel._native.deterministic",
        error_description="witwin.channel.deterministic native extension",
    ),
    package_path=PACKAGE_PATH,
)


__all__ = ["NativeExtension"]

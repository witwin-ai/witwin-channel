"""Python loader for the shared channel core native extension."""

from __future__ import annotations

from witwin.channel._native import PACKAGE_PATH
from witwin.channel._native.loader import (
    NativeExtensionLoader,
    NativeExtensionSpec,
)


NativeExtension = NativeExtensionLoader(
    NativeExtensionSpec(
        module_name="witwin.channel._native._channel_utils_native",
        install_subpath=("witwin", "channel", "_native"),
        build_glob="witwin/channel/_native/channel_utils/*",
        binary_glob="_channel_utils_native*",
        probe_env_var="WITWIN_CHANNEL_UTILS_NATIVE_PROBE",
        probe_module_name="witwin.channel._native.channel_utils",
        error_description="witwin.channel.core native extension",
    ),
    package_path=PACKAGE_PATH,
)


__all__ = ["NativeExtension"]

"""Backend resolution policy for shadow-boundary post-processing.

Both the deterministic matched-ISB and Monte Carlo power-smoothing post-
processing pipelines pick between a small-workload reference backend and a
candidate-pruned native backend. They differ only in the small-backend name,
the pair-count threshold, and the user-facing error wording. This module
captures that shared decision in one place.
"""

from __future__ import annotations

from dataclasses import dataclass

from witwin.channel.core.kernels.shadow_boundary import ShadowBoundaryKernel


@dataclass(frozen=True)
class ShadowBoundaryBackendPolicy:
    """Policy that resolves a shadow-boundary backend request.

    The valid backend names are ``"auto"``, ``small_backend``, and
    ``"native_candidate"``. Error messages are caller-supplied so each
    post-processing pipeline can keep its own wording and recovery guidance.
    """

    small_backend: str
    pair_threshold: int
    too_large_message: str
    no_native_message: str
    ad_unsupported_message: str

    @property
    def _valid_backends(self) -> set[str]:
        return {"auto", self.small_backend, "native_candidate"}

    def resolve(
        self,
        *,
        requested: str,
        n_pairs: int,
        ad_enabled: bool,
        native_available: bool | None = None,
    ) -> str:
        if requested not in self._valid_backends:
            raise ValueError(
                f"shadow_boundary_backend must be one of {sorted(self._valid_backends)}."
            )
        if requested == self.small_backend:
            self.validate_small_workload(n_pairs)
            return self.small_backend
        if requested == "auto" and n_pairs <= self.pair_threshold:
            return self.small_backend
        available = (
            ShadowBoundaryKernel.available()
            if native_available is None
            else bool(native_available)
        )
        if not available:
            raise RuntimeError(self.no_native_message)
        if ad_enabled:
            raise RuntimeError(self.ad_unsupported_message)
        return "native_candidate"

    def validate_small_workload(self, n_pairs: int) -> None:
        if n_pairs > self.pair_threshold:
            raise RuntimeError(self.too_large_message.format(n_pairs=n_pairs))


__all__ = ["ShadowBoundaryBackendPolicy"]

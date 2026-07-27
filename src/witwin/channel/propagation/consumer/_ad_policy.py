"""AD admission policy for the propagation consumer (ADR-043).

Every rule here answers one question before any numerical work happens: may
this request carry this derivative at all? None of it is physics, none of it
touches a device, and all of it runs on the pre-flight of both routes, so an
unsupported AD request never reaches a native launch and never produces a
result object.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .contracts import capabilities

if TYPE_CHECKING:
    from witwin.channel.scene.compiled import CompiledScene


_CAPABILITIES = capabilities()

_MATERIAL_AD_LEAVES = (
    "eps_r",
    "sigma_e",
    "thickness_m",
    "gain",
    "layer_eps_r",
    "layer_sigma_e",
    "layer_thickness_m",
)


def has_forward_tangent(value: torch.Tensor) -> bool:
    return torch.autograd.forward_ad.unpack_dual(value).tangent is not None


def carries_ad(value: torch.Tensor | None) -> bool:
    return value is not None and (
        value.requires_grad or has_forward_tangent(value)
    )


def _primal_only_values(
    compiled: CompiledScene, request: object
) -> dict[str, torch.Tensor | None]:
    """The tensors named by ``capabilities().primal_only_ad_inputs``.

    Two of them live on the compiled scene rather than on the request: the
    relative permeabilities reach the Fresnel companions as constants and are
    rejected there, so they belong in the same pre-compute refusal as the
    request-side constants.
    """

    materials = compiled.materials
    return {
        "sources.powers_w": request.sources.powers_w,
        "sources.polarizations": request.sources.polarizations,
        "sinks.polarizations": request.sinks.polarizations,
        "sources.polarization_basis": request.sources.polarization_basis,
        "sinks.polarization_basis": request.sinks.polarization_basis,
        "materials.mu_r": materials.mu_r,
        "materials.layer_mu_r": materials.layer_mu_r,
    }


def require_primal_only_ad_inputs(
    compiled: CompiledScene, request: object
) -> None:
    """Reject AD on inputs the native field companions treat as constants.

    The native forward/backward/JVP contracts reject every one of these by
    name, but on the discovery route that rejection used to fire from inside
    ``backward()`` - after a complete ``PropagationEvaluation`` had already been
    published. That is a partial result for an unsupported request, so the same
    refusal now runs on the pre-flight of every response and every route, driven
    by the published ``primal_only_ad_inputs`` record rather than by a list
    duplicated per route.
    """

    if request.ad_mode == "none":
        return
    values = _primal_only_values(compiled, request)
    for name in _CAPABILITIES.primal_only_ad_inputs:
        if carries_ad(values[name]):
            raise NotImplementedError(
                f"{name} is primal-only; the native field companion that "
                "consumes it does not differentiate it. "
                "capabilities().primal_only_ad_inputs names every such input"
            )


def _ad_leaf_tensors(
    compiled: CompiledScene, request: object
) -> tuple[tuple[str, torch.Tensor], ...]:
    """Every tensor a caller can seed on this call, named for a refusal.

    Host attribute reads only: no device work, no allocation, and no
    synchronization, so this is free to run on the pre-flight of every call.
    """

    materials = compiled.materials
    candidates: list[tuple[str, object]] = [
        ("sources.positions_m", request.sources.positions_m),
        ("sinks.positions_m", request.sinks.positions_m),
        ("reference_frequency_hz", request.reference_frequency_hz),
    ]
    candidates.extend(_primal_only_values(compiled, request).items())
    candidates.extend(
        (f"materials.{name}", getattr(materials, name, None))
        for name in _MATERIAL_AD_LEAVES
    )
    candidates.extend(
        (f"structures[{index}].vertices", getattr(structure, "vertices", None))
        for index, structure in enumerate(compiled.structures)
    )
    return tuple(
        (name, value)
        for name, value in candidates
        if isinstance(value, torch.Tensor)
    )


def require_first_order_request(
    compiled: CompiledScene, request: object
) -> None:
    """Refuse a forward-over-reverse composition before any numerical work.

    A reverse pass cannot carry a forward tangent through the native
    companions: the gradient comes back with the correct first-order value and
    ``unpack_dual(grad).tangent is None``, so a mixed second derivative reads as
    an exact zero with no error anywhere. That is the worst shape a silent cell
    can take, and it is refused here rather than answered wrongly.

    The symmetric rule ("jvp with a requires_grad input") is deliberately NOT
    enforced: ADR-038's declared convention explicitly supports a dual built on
    a ``requires_grad`` primal, and the field facades run the same Function for
    both modes, so such a request is a legitimate first-order one.
    Reverse-over-reverse is caught instead where it becomes wrong, by
    ``_ad_first_order_only`` inside every backward.
    """

    if request.ad_mode != "vjp":
        return
    for name, value in _ad_leaf_tensors(compiled, request):
        if has_forward_tangent(value):
            raise NotImplementedError(
                f"ad_mode='vjp' with a forward dual on {name} is a "
                "second-order request; Channel is first-order only and a "
                "reverse gradient carries no tangent. "
                "capabilities().supports_higher_order_ad is False"
            )


def ad_ledger(ad_mode: str) -> object | None:
    """One AD ledger per reevaluation, or ``None`` for a primal call.

    The discovery route already builds one inside its field loop and hands it
    up through the execution sidecars. The fixed-topology route built none, so
    the inner loop a per-frame consumer runs reported no AD accounting at all;
    this is the same counter, constructed at the one place that owns the whole
    call. A primal call constructs nothing and pays nothing.
    """

    if ad_mode == "none":
        return None
    from witwin.channel.runtime.kernel_metadata import AdLaunchLedger

    return AdLaunchLedger()


def tape_bytes(ledger_bytes: int, ad_mode: str) -> int:
    """Reproduce the solver-metadata tape gate rather than the raw counter.

    ``AdLaunchLedger`` sums what every registered companion saved, and forward
    mode retains none of it past the solve. The solver metadata layer applies
    exactly this gate (``deterministic/pipeline.py``), so forwarding the raw
    sidecar number here would report retained tape for a jvp call and
    contradict the ledger's own contract.
    """

    return int(ledger_bytes) if ad_mode == "vjp" else 0


__all__ = [
    "ad_ledger",
    "carries_ad",
    "has_forward_tangent",
    "require_first_order_request",
    "require_primal_only_ad_inputs",
    "tape_bytes",
]

"""Native ADR-021 multi-bounce chain scattering kernel facades (plan 10a).

Op A (:func:`scattering_chain_ensemble_eval`) generalizes the single-bounce
ensemble op to a specular reflection chain in the power domain; Op B
(:func:`scattering_chain_realization_eval`) generalizes the phase-screen patch
integral to the coherent 2x2 Jones sandwich with a realization at the vertex.
Both are thin facades over the required native symbols and share the Duffy
quadrature-node owner ``_duffy_nodes`` in :mod:`.functional`.
"""

from __future__ import annotations

import torch

from witwin.channel_native.runtime.symbols import required_symbol as _required_native_op
from witwin.channel_native.runtime.tensor_contracts import validate_cuda_tensor

from .functional import _duffy_nodes


# ---------------------------------------------------------------------------
# ADR-021 chain scattering ops (plan 10a sections 3 and 4).
#
# Op A (``scattering_chain_ensemble_eval``) generalizes op 1 to a specular
# reflection chain C1 -> diffuse vertex v_s -> chain C2 in the power domain;
# Op B (``scattering_chain_realization_eval``) generalizes op 2 to the coherent
# 2x2 Jones sandwich with a phase-screen realization at the vertex. Both carry
# two padded per-leg blocks ``[R, Dmax, ...]`` with ``Dmax = kMaxAdDepth = 8``
# (plan 10a section 1). The facades are thin: validate the spec tables, dispatch
# the required native symbol, and assert the returned key-set exactly.
# ---------------------------------------------------------------------------

_K_MAX_AD_DEPTH = 8

_CHAIN_ENSEMBLE_OUTPUT_FIELDS = ("gain", "amplitude", "length", "keep")
_CHAIN_ENSEMBLE_TANGENT_FIELDS = (
    "tangent_gain",
    "tangent_amplitude",
    "tangent_length",
)
_CHAIN_ENSEMBLE_BACKWARD_FIELDS = (
    "grad_c1_eps_r",
    "grad_c1_sigma_e",
    "grad_c1_gain",
    "grad_c1_thickness",
    "grad_c2_eps_r",
    "grad_c2_sigma_e",
    "grad_c2_gain",
    "grad_c2_thickness",
    "grad_f_te",
    "grad_f_tm",
    "grad_coef",
    "grad_frequency",
)

_CHAIN_REALIZATION_OUTPUT_FIELDS = (
    "total",
    "path_field",
    "path_gain",
    "integral",
    "row_value",
)
_CHAIN_REALIZATION_TANGENT_FIELDS = (
    "tangent_total",
    "tangent_path_field",
    "tangent_path_gain",
)
_CHAIN_REALIZATION_BACKWARD_FIELDS = (
    "grad_heights",
    "grad_layer_thickness",
    "grad_layer_eps_r",
    "grad_layer_sigma_e",
    "grad_c1_eps_r",
    "grad_c1_sigma_e",
    "grad_c1_gain",
    "grad_c1_thickness",
    "grad_c2_eps_r",
    "grad_c2_sigma_e",
    "grad_c2_gain",
    "grad_c2_thickness",
    "grad_d_i",
    "grad_d_o",
    "grad_c1_positions",
    "grad_c1_normals",
    "grad_c2_positions",
    "grad_c2_normals",
    "grad_L1",
    "grad_L2",
    "grad_sp1",
    "grad_sp2",
    "grad_centroids",
    "grad_k0",
    "grad_frequency",
)


def _validate_chain_leg(
    prefix: str,
    positions: torch.Tensor,
    normals: torch.Tensor,
    eps_r: torch.Tensor,
    sigma_e: torch.Tensor,
    mu_r: torch.Tensor,
    gain: torch.Tensor,
    thickness: torch.Tensor,
    depth: torch.Tensor,
    rows: int,
) -> int:
    """Validate one padded specular leg block and return its ``Dmax`` width.

    The padded block is ``[R, Dmax, 3]`` positions/normals and ``[R, Dmax]``
    per-bounce Fresnel inputs with an ``[R]`` int32 per-row depth. ``Dmax`` is
    the static padded width and must not exceed ``kMaxAdDepth = 8`` (plan 10a
    section 1); the per-row ``depth`` values are trusted structural winners and
    are not read on the host (that would force a device sync).
    """

    validate_cuda_tensor(
        f"{prefix}_positions", positions, dtype=torch.float32, ndim=3, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        f"{prefix}_normals", normals, dtype=torch.float32, ndim=3, trailing_shape=(3,)
    )
    dmax = positions.shape[1]
    if dmax > _K_MAX_AD_DEPTH:
        raise ValueError(
            f"{prefix} leg depth {dmax} exceeds kMaxAdDepth={_K_MAX_AD_DEPTH}"
        )
    if tuple(positions.shape) != (rows, dmax, 3):
        raise ValueError(f"{prefix}_positions must have shape ({rows}, {dmax}, 3)")
    if tuple(normals.shape) != (rows, dmax, 3):
        raise ValueError(f"{prefix}_normals must have shape ({rows}, {dmax}, 3)")
    for name, tensor in (
        (f"{prefix}_eps_r", eps_r),
        (f"{prefix}_sigma_e", sigma_e),
        (f"{prefix}_mu_r", mu_r),
        (f"{prefix}_gain", gain),
        (f"{prefix}_thickness", thickness),
    ):
        validate_cuda_tensor(name, tensor, dtype=torch.float32, ndim=2)
        if tuple(tensor.shape) != (rows, dmax):
            raise ValueError(f"{name} must have shape ({rows}, {dmax})")
    validate_cuda_tensor(f"{prefix}_depth", depth, dtype=torch.int32, ndim=1)
    if depth.shape[0] != rows:
        raise ValueError(f"{prefix}_depth must have {rows} rows")
    return dmax


def _require_same_device(named: tuple[tuple[str, torch.Tensor], ...]) -> None:
    """Assert every named tensor shares one CUDA device (plan 10a contract)."""

    device = None
    for name, tensor in named:
        if tensor is None:
            continue
        if device is None:
            device = tensor.device
        elif tensor.device != device:
            raise ValueError(
                f"{name} must share the CUDA device of the chain row block"
            )


def scattering_chain_ensemble_eval(
    tx_pol: torch.Tensor,
    rx_pol: torch.Tensor,
    source: torch.Tensor,
    vertex: torch.Tensor,
    target: torch.Tensor,
    c1_positions: torch.Tensor,
    c1_normals: torch.Tensor,
    c1_eps_r: torch.Tensor,
    c1_sigma_e: torch.Tensor,
    c1_mu_r: torch.Tensor,
    c1_gain: torch.Tensor,
    c1_thickness: torch.Tensor,
    c1_depth: torch.Tensor,
    c2_positions: torch.Tensor,
    c2_normals: torch.Tensor,
    c2_eps_r: torch.Tensor,
    c2_sigma_e: torch.Tensor,
    c2_mu_r: torch.Tensor,
    c2_gain: torch.Tensor,
    c2_thickness: torch.Tensor,
    c2_depth: torch.Tensor,
    n_o: torch.Tensor,
    t1r: torch.Tensor,
    t2r: torch.Tensor,
    backup_axis: torch.Tensor,
    wi_local: torch.Tensor,
    cos_i: torch.Tensor,
    cos_o: torch.Tensor,
    d_i: torch.Tensor,
    d_o: torch.Tensor,
    l1: torch.Tensor,
    l2: torch.Tensor,
    weights: torch.Tensor,
    material_id: torch.Tensor,
    f_te_flat: torch.Tensor,
    f_tm_flat: torch.Tensor,
    table_offset: torch.Tensor,
    table_dims: torch.Tensor,
    material_slot: torch.Tensor,
    *,
    coef: float,
    threshold: float,
    frequency_hz: float,
) -> dict[str, torch.Tensor]:
    """Native ADR-021 Op A multi-bounce ensemble scattering rows (plan 10a s3).

    Power-domain generalization of :func:`scattering_ensemble_eval`: C1 coherent
    Jones transport of ``tx_pol`` from ``source`` to the diffuse ``vertex`` yields
    the incident coherency diagonal, the resident Kirchhoff table gives the
    outgoing diagonal, and the C2 power-domain sandwich to ``target`` plus the
    receiver projection assemble the radiometric row gain (per-vertex ``weights``
    = ``A_patch`` and ``1/(L1^2 L2^2)`` spreading, op-1 convention). The incident
    coherency is computed in-kernel from the C1 transport (supervisor ruling: no
    ``a_te2``/``a_tm2`` projection pair, no cross-pol slots). One launch per
    (tx, rx-chunk); required native op.
    """

    rows = int(tx_pol.shape[0])
    validate_cuda_tensor("tx_pol", tx_pol, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor("rx_pol", rx_pol, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    if rx_pol.shape[0] != rows:
        raise ValueError("rx_pol must match tx_pol rows")
    _validate_chain_leg(
        "c1", c1_positions, c1_normals, c1_eps_r, c1_sigma_e, c1_mu_r, c1_gain,
        c1_thickness, c1_depth, rows,
    )
    _validate_chain_leg(
        "c2", c2_positions, c2_normals, c2_eps_r, c2_sigma_e, c2_mu_r, c2_gain,
        c2_thickness, c2_depth, rows,
    )
    for name, tensor in (
        ("source", source),
        ("vertex", vertex),
        ("target", target),
        ("n_o", n_o),
        ("t1r", t1r),
        ("t2r", t2r),
        ("backup_axis", backup_axis),
        ("wi_local", wi_local),
        ("d_i", d_i),
        ("d_o", d_o),
    ):
        validate_cuda_tensor(name, tensor, dtype=torch.float32, ndim=2, trailing_shape=(3,))
        if tensor.shape[0] != rows:
            raise ValueError(f"{name} must have {rows} rows")
    for name, tensor in (
        ("cos_i", cos_i),
        ("cos_o", cos_o),
        ("l1", l1),
        ("l2", l2),
        ("weights", weights),
    ):
        validate_cuda_tensor(name, tensor, dtype=torch.float32, ndim=1)
        if tensor.shape[0] != rows:
            raise ValueError(f"{name} must have {rows} rows")
    validate_cuda_tensor("material_id", material_id, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("f_te_flat", f_te_flat, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("f_tm_flat", f_tm_flat, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("table_offset", table_offset, dtype=torch.int64, ndim=1)
    validate_cuda_tensor("table_dims", table_dims, dtype=torch.int32, ndim=2, trailing_shape=(4,))
    validate_cuda_tensor("material_slot", material_slot, dtype=torch.int32, ndim=1)
    _require_same_device(
        (
            ("tx_pol", tx_pol),
            ("source", source),
            ("c1_positions", c1_positions),
            ("c2_positions", c2_positions),
            ("d_i", d_i),
            ("f_te_flat", f_te_flat),
            ("material_slot", material_slot),
        )
    )
    out = _required_native_op("scattering_chain_ensemble_eval")(
        tx_pol,
        rx_pol,
        source,
        vertex,
        target,
        c1_positions,
        c1_normals,
        c1_eps_r,
        c1_sigma_e,
        c1_mu_r,
        c1_gain,
        c1_thickness,
        c1_depth,
        c2_positions,
        c2_normals,
        c2_eps_r,
        c2_sigma_e,
        c2_mu_r,
        c2_gain,
        c2_thickness,
        c2_depth,
        n_o,
        t1r,
        t2r,
        backup_axis,
        wi_local,
        cos_i,
        cos_o,
        d_i,
        d_o,
        l1,
        l2,
        weights,
        material_id,
        f_te_flat,
        f_tm_flat,
        table_offset,
        table_dims,
        material_slot,
        float(coef),
        float(threshold),
        float(frequency_hz),
    )
    if not isinstance(out, dict) or set(out) != set(_CHAIN_ENSEMBLE_OUTPUT_FIELDS):
        raise TypeError(
            "_channel_native.scattering_chain_ensemble_eval returned invalid fields"
        )
    return out


def scattering_chain_ensemble_eval_backward(
    tx_pol: torch.Tensor,
    rx_pol: torch.Tensor,
    source: torch.Tensor,
    vertex: torch.Tensor,
    target: torch.Tensor,
    c1_positions: torch.Tensor,
    c1_normals: torch.Tensor,
    c1_eps_r: torch.Tensor,
    c1_sigma_e: torch.Tensor,
    c1_mu_r: torch.Tensor,
    c1_gain: torch.Tensor,
    c1_thickness: torch.Tensor,
    c1_depth: torch.Tensor,
    c2_positions: torch.Tensor,
    c2_normals: torch.Tensor,
    c2_eps_r: torch.Tensor,
    c2_sigma_e: torch.Tensor,
    c2_mu_r: torch.Tensor,
    c2_gain: torch.Tensor,
    c2_thickness: torch.Tensor,
    c2_depth: torch.Tensor,
    n_o: torch.Tensor,
    t1r: torch.Tensor,
    t2r: torch.Tensor,
    backup_axis: torch.Tensor,
    wi_local: torch.Tensor,
    cos_i: torch.Tensor,
    cos_o: torch.Tensor,
    d_i: torch.Tensor,
    d_o: torch.Tensor,
    l1: torch.Tensor,
    l2: torch.Tensor,
    weights: torch.Tensor,
    material_id: torch.Tensor,
    f_te_flat: torch.Tensor,
    f_tm_flat: torch.Tensor,
    table_offset: torch.Tensor,
    table_dims: torch.Tensor,
    material_slot: torch.Tensor,
    *,
    coef: float,
    threshold: float,
    frequency_hz: float,
    grad_gain: torch.Tensor | None = None,
    grad_amplitude: torch.Tensor | None = None,
    grad_length: torch.Tensor | None = None,
    need_grad_chain1: bool = False,
    need_grad_chain2: bool = False,
    need_grad_tables: bool = False,
    need_grad_geometry: bool = False,
    need_grad_coef: bool = False,
    need_grad_frequency: bool = False,
) -> dict[str, torch.Tensor | None]:
    """VJP of :func:`scattering_chain_ensemble_eval` (ADR-021 Op A, plan 10a s3.2).

    Per-bounce chain-Fresnel grads are direct stores; the table grads use the
    16-corner atomicAdd scatter and ``grad_coef`` / ``grad_frequency`` are scalar
    atomicAdd reductions. Reverse-mode continuous chain geometry
    (``need_grad_geometry``) is a staged follow-up wave and is rejected loudly by
    the native bridge; the ``_jvp`` companion covers geometry in forward mode. The
    returned dict is exactly the twelve native VJP fields.
    """

    rows = int(tx_pol.shape[0])
    validate_cuda_tensor("tx_pol", tx_pol, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    _validate_chain_leg(
        "c1", c1_positions, c1_normals, c1_eps_r, c1_sigma_e, c1_mu_r, c1_gain,
        c1_thickness, c1_depth, rows,
    )
    _validate_chain_leg(
        "c2", c2_positions, c2_normals, c2_eps_r, c2_sigma_e, c2_mu_r, c2_gain,
        c2_thickness, c2_depth, rows,
    )
    validate_cuda_tensor("material_id", material_id, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("table_offset", table_offset, dtype=torch.int64, ndim=1)
    validate_cuda_tensor("table_dims", table_dims, dtype=torch.int32, ndim=2, trailing_shape=(4,))
    validate_cuda_tensor("material_slot", material_slot, dtype=torch.int32, ndim=1)
    out = _required_native_op("scattering_chain_ensemble_eval_backward")(
        tx_pol,
        rx_pol,
        source,
        vertex,
        target,
        c1_positions,
        c1_normals,
        c1_eps_r,
        c1_sigma_e,
        c1_mu_r,
        c1_gain,
        c1_thickness,
        c1_depth,
        c2_positions,
        c2_normals,
        c2_eps_r,
        c2_sigma_e,
        c2_mu_r,
        c2_gain,
        c2_thickness,
        c2_depth,
        n_o,
        t1r,
        t2r,
        backup_axis,
        wi_local,
        cos_i,
        cos_o,
        d_i,
        d_o,
        l1,
        l2,
        weights,
        material_id,
        f_te_flat,
        f_tm_flat,
        table_offset,
        table_dims,
        material_slot,
        float(coef),
        float(threshold),
        float(frequency_hz),
        grad_gain,
        grad_amplitude,
        grad_length,
        bool(need_grad_chain1),
        bool(need_grad_chain2),
        bool(need_grad_tables),
        bool(need_grad_geometry),
        bool(need_grad_coef),
        bool(need_grad_frequency),
    )
    if not isinstance(out, dict) or set(out) != set(_CHAIN_ENSEMBLE_BACKWARD_FIELDS):
        raise TypeError(
            "_channel_native.scattering_chain_ensemble_eval_backward returned"
            " invalid fields"
        )
    return out


def scattering_chain_ensemble_eval_jvp(
    tx_pol: torch.Tensor,
    rx_pol: torch.Tensor,
    source: torch.Tensor,
    vertex: torch.Tensor,
    target: torch.Tensor,
    c1_positions: torch.Tensor,
    c1_normals: torch.Tensor,
    c1_eps_r: torch.Tensor,
    c1_sigma_e: torch.Tensor,
    c1_mu_r: torch.Tensor,
    c1_gain: torch.Tensor,
    c1_thickness: torch.Tensor,
    c1_depth: torch.Tensor,
    c2_positions: torch.Tensor,
    c2_normals: torch.Tensor,
    c2_eps_r: torch.Tensor,
    c2_sigma_e: torch.Tensor,
    c2_mu_r: torch.Tensor,
    c2_gain: torch.Tensor,
    c2_thickness: torch.Tensor,
    c2_depth: torch.Tensor,
    n_o: torch.Tensor,
    t1r: torch.Tensor,
    t2r: torch.Tensor,
    backup_axis: torch.Tensor,
    wi_local: torch.Tensor,
    cos_i: torch.Tensor,
    cos_o: torch.Tensor,
    d_i: torch.Tensor,
    d_o: torch.Tensor,
    l1: torch.Tensor,
    l2: torch.Tensor,
    weights: torch.Tensor,
    material_id: torch.Tensor,
    f_te_flat: torch.Tensor,
    f_tm_flat: torch.Tensor,
    table_offset: torch.Tensor,
    table_dims: torch.Tensor,
    material_slot: torch.Tensor,
    *,
    coef: float,
    threshold: float,
    frequency_hz: float,
    tangent_c1_eps_r: torch.Tensor | None = None,
    tangent_c1_sigma_e: torch.Tensor | None = None,
    tangent_c1_gain: torch.Tensor | None = None,
    tangent_c1_thickness: torch.Tensor | None = None,
    tangent_c2_eps_r: torch.Tensor | None = None,
    tangent_c2_sigma_e: torch.Tensor | None = None,
    tangent_c2_gain: torch.Tensor | None = None,
    tangent_c2_thickness: torch.Tensor | None = None,
    tangent_f_te_flat: torch.Tensor | None = None,
    tangent_f_tm_flat: torch.Tensor | None = None,
    tangent_c1_positions: torch.Tensor | None = None,
    tangent_c1_normals: torch.Tensor | None = None,
    tangent_c2_positions: torch.Tensor | None = None,
    tangent_c2_normals: torch.Tensor | None = None,
    tangent_d_i: torch.Tensor | None = None,
    tangent_d_o: torch.Tensor | None = None,
    tangent_v_normal: torch.Tensor | None = None,
    tangent_l1: torch.Tensor | None = None,
    tangent_l2: torch.Tensor | None = None,
    tangent_cos_i: torch.Tensor | None = None,
    tangent_cos_o: torch.Tensor | None = None,
    tangent_coef: float = 0.0,
    tangent_frequency: float = 0.0,
) -> dict[str, torch.Tensor]:
    """JVP of :func:`scattering_chain_ensemble_eval` (ADR-021 Op A, plan 10a s3.3).

    Deterministic forward-mode dual sweep (fixed-order, no atomics). Forward-mode
    supports geometry tangents (positions/normals/d_i/d_o/n_o/L1/L2/cos_i/cos_o);
    the endpoint positions, the vertex frame axes, ``weights``, ``wi_local`` and
    the table metadata carry no tangent. A missing tangent is a zero tangent;
    ``keep`` is non-differentiable so it has no tangent output.
    """

    validate_cuda_tensor("tx_pol", tx_pol, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    out = _required_native_op("scattering_chain_ensemble_eval_jvp")(
        tx_pol,
        rx_pol,
        source,
        vertex,
        target,
        c1_positions,
        c1_normals,
        c1_eps_r,
        c1_sigma_e,
        c1_mu_r,
        c1_gain,
        c1_thickness,
        c1_depth,
        c2_positions,
        c2_normals,
        c2_eps_r,
        c2_sigma_e,
        c2_mu_r,
        c2_gain,
        c2_thickness,
        c2_depth,
        n_o,
        t1r,
        t2r,
        backup_axis,
        wi_local,
        cos_i,
        cos_o,
        d_i,
        d_o,
        l1,
        l2,
        weights,
        material_id,
        f_te_flat,
        f_tm_flat,
        table_offset,
        table_dims,
        material_slot,
        float(coef),
        float(threshold),
        float(frequency_hz),
        tangent_c1_eps_r,
        tangent_c1_sigma_e,
        tangent_c1_gain,
        tangent_c1_thickness,
        tangent_c2_eps_r,
        tangent_c2_sigma_e,
        tangent_c2_gain,
        tangent_c2_thickness,
        tangent_f_te_flat,
        tangent_f_tm_flat,
        tangent_c1_positions,
        tangent_c1_normals,
        tangent_c2_positions,
        tangent_c2_normals,
        tangent_d_i,
        tangent_d_o,
        tangent_v_normal,
        tangent_l1,
        tangent_l2,
        tangent_cos_i,
        tangent_cos_o,
        float(tangent_coef),
        float(tangent_frequency),
    )
    if not isinstance(out, dict) or set(out) != set(_CHAIN_ENSEMBLE_TANGENT_FIELDS):
        raise TypeError(
            "_channel_native.scattering_chain_ensemble_eval_jvp returned invalid fields"
        )
    return out


def scattering_chain_realization_eval(
    patch_tris: torch.Tensor,
    patch_uvs: torch.Tensor,
    rows: torch.Tensor,
    d_i: torch.Tensor,
    d_o: torch.Tensor,
    n_rows: torch.Tensor,
    source: torch.Tensor,
    vertex: torch.Tensor,
    target: torch.Tensor,
    c1_positions: torch.Tensor,
    c1_normals: torch.Tensor,
    c1_eps_r: torch.Tensor,
    c1_sigma_e: torch.Tensor,
    c1_mu_r: torch.Tensor,
    c1_gain: torch.Tensor,
    c1_thickness: torch.Tensor,
    c1_depth: torch.Tensor,
    c2_positions: torch.Tensor,
    c2_normals: torch.Tensor,
    c2_eps_r: torch.Tensor,
    c2_sigma_e: torch.Tensor,
    c2_mu_r: torch.Tensor,
    c2_gain: torch.Tensor,
    c2_thickness: torch.Tensor,
    c2_depth: torch.Tensor,
    tx_pol: torch.Tensor,
    rx_pol: torch.Tensor,
    L1: torch.Tensor,
    L2: torch.Tensor,
    sp1: torch.Tensor,
    sp2: torch.Tensor,
    centroids: torch.Tensor,
    heights: torch.Tensor,
    cos_spec: torch.Tensor,
    material_id: torch.Tensor,
    layer_offset: torch.Tensor,
    layer_count: torch.Tensor,
    layer_thickness_m: torch.Tensor,
    layer_eps_r: torch.Tensor,
    layer_sigma_e: torch.Tensor,
    layer_mu_r: torch.Tensor,
    *,
    k0: float,
    frequency_hz: float,
) -> dict[str, torch.Tensor]:
    """Native ADR-021 Op B coherent chain realization rows (plan 10a s4).

    Coherent generalization of :func:`scattering_patch_integral_eval`: the full
    2x2 Jones sandwich ``E_rx = A_2 . S_patch(d_i, d_o; h) . A_1 . e_tx`` with
    the carrier over the image-unfolded lengths, the planar spreading, the
    ``r_te/r_tm`` computed in-kernel from the resident CSR layer stack at
    ``cos_spec``, and the same Duffy 16x16 GL quadrature and two-stage
    fixed-order tree reduction as op 2. The Duffy nodes are appended by the
    facade (op-2 parity); required native op.
    """

    n = int(d_i.shape[0])
    validate_cuda_tensor("patch_tris", patch_tris, dtype=torch.float32, ndim=3, trailing_shape=(3, 3))
    validate_cuda_tensor("patch_uvs", patch_uvs, dtype=torch.float32, ndim=3, trailing_shape=(3, 2))
    validate_cuda_tensor("rows", rows, dtype=torch.int64, ndim=1)
    if rows.shape[0] != n:
        raise ValueError("rows must have one entry per chain row")
    for name, tensor in (
        ("d_i", d_i),
        ("d_o", d_o),
        ("n_rows", n_rows),
        ("source", source),
        ("vertex", vertex),
        ("target", target),
        ("tx_pol", tx_pol),
        ("rx_pol", rx_pol),
        ("centroids", centroids),
    ):
        validate_cuda_tensor(name, tensor, dtype=torch.float32, ndim=2, trailing_shape=(3,))
        if tensor.shape[0] != n:
            raise ValueError(f"{name} must have {n} rows")
    _validate_chain_leg(
        "c1", c1_positions, c1_normals, c1_eps_r, c1_sigma_e, c1_mu_r, c1_gain,
        c1_thickness, c1_depth, n,
    )
    _validate_chain_leg(
        "c2", c2_positions, c2_normals, c2_eps_r, c2_sigma_e, c2_mu_r, c2_gain,
        c2_thickness, c2_depth, n,
    )
    for name, tensor in (
        ("L1", L1),
        ("L2", L2),
        ("sp1", sp1),
        ("sp2", sp2),
        ("cos_spec", cos_spec),
    ):
        validate_cuda_tensor(name, tensor, dtype=torch.float32, ndim=1)
        if tensor.shape[0] != n:
            raise ValueError(f"{name} must have {n} rows")
    validate_cuda_tensor("heights", heights, dtype=torch.float32, ndim=2)
    validate_cuda_tensor("material_id", material_id, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("layer_offset", layer_offset, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("layer_count", layer_count, dtype=torch.int32, ndim=1)
    for name, tensor in (
        ("layer_thickness_m", layer_thickness_m),
        ("layer_eps_r", layer_eps_r),
        ("layer_sigma_e", layer_sigma_e),
        ("layer_mu_r", layer_mu_r),
    ):
        validate_cuda_tensor(name, tensor, dtype=torch.float32, ndim=1)
    _require_same_device(
        (
            ("patch_tris", patch_tris),
            ("d_i", d_i),
            ("c1_positions", c1_positions),
            ("c2_positions", c2_positions),
            ("heights", heights),
            ("layer_thickness_m", layer_thickness_m),
        )
    )
    quad_a, quad_b, quad_w = _duffy_nodes(patch_tris.device)
    out = _required_native_op("scattering_chain_realization_eval")(
        patch_tris,
        patch_uvs,
        rows,
        d_i,
        d_o,
        n_rows,
        source,
        vertex,
        target,
        c1_positions,
        c1_normals,
        c1_eps_r,
        c1_sigma_e,
        c1_mu_r,
        c1_gain,
        c1_thickness,
        c1_depth,
        c2_positions,
        c2_normals,
        c2_eps_r,
        c2_sigma_e,
        c2_mu_r,
        c2_gain,
        c2_thickness,
        c2_depth,
        tx_pol,
        rx_pol,
        L1,
        L2,
        sp1,
        sp2,
        centroids,
        heights,
        cos_spec,
        material_id,
        layer_offset,
        layer_count,
        layer_thickness_m,
        layer_eps_r,
        layer_sigma_e,
        layer_mu_r,
        quad_a,
        quad_b,
        quad_w,
        float(k0),
        float(frequency_hz),
    )
    if not isinstance(out, dict) or set(out) != set(_CHAIN_REALIZATION_OUTPUT_FIELDS):
        raise TypeError(
            "_channel_native.scattering_chain_realization_eval returned invalid fields"
        )
    return out


def scattering_chain_realization_eval_backward(
    patch_tris: torch.Tensor,
    patch_uvs: torch.Tensor,
    rows: torch.Tensor,
    d_i: torch.Tensor,
    d_o: torch.Tensor,
    n_rows: torch.Tensor,
    source: torch.Tensor,
    vertex: torch.Tensor,
    target: torch.Tensor,
    c1_positions: torch.Tensor,
    c1_normals: torch.Tensor,
    c1_eps_r: torch.Tensor,
    c1_sigma_e: torch.Tensor,
    c1_mu_r: torch.Tensor,
    c1_gain: torch.Tensor,
    c1_thickness: torch.Tensor,
    c1_depth: torch.Tensor,
    c2_positions: torch.Tensor,
    c2_normals: torch.Tensor,
    c2_eps_r: torch.Tensor,
    c2_sigma_e: torch.Tensor,
    c2_mu_r: torch.Tensor,
    c2_gain: torch.Tensor,
    c2_thickness: torch.Tensor,
    c2_depth: torch.Tensor,
    tx_pol: torch.Tensor,
    rx_pol: torch.Tensor,
    L1: torch.Tensor,
    L2: torch.Tensor,
    sp1: torch.Tensor,
    sp2: torch.Tensor,
    centroids: torch.Tensor,
    heights: torch.Tensor,
    cos_spec: torch.Tensor,
    material_id: torch.Tensor,
    layer_offset: torch.Tensor,
    layer_count: torch.Tensor,
    layer_thickness_m: torch.Tensor,
    layer_eps_r: torch.Tensor,
    layer_sigma_e: torch.Tensor,
    layer_mu_r: torch.Tensor,
    *,
    k0: float,
    frequency_hz: float,
    grad_total: torch.Tensor,
    grad_path_field: torch.Tensor | None = None,
    grad_path_gain: torch.Tensor | None = None,
    need_grad_heights: bool = False,
    need_grad_layers: bool = False,
    need_grad_chain1: bool = False,
    need_grad_chain2: bool = False,
    need_grad_geometry: bool = False,
    need_grad_k0: bool = False,
    need_grad_frequency: bool = False,
) -> dict[str, torch.Tensor | None]:
    """VJP of :func:`scattering_chain_realization_eval` (ADR-021 Op B, plan 10a s4.2).

    ``grad_total`` is the required 0-dim complex cotangent (op-2 parity);
    ``grad_path_field`` / ``grad_path_gain`` are the optional per-row cotangents
    the deterministic coherent combine (D3) backprops through. ``grad_heights``
    and the CSR layer grads use atomicAdd scatter; per-row / per-bounce grads are
    direct stores. Off-flag keys are ``None``.
    """

    n = int(d_i.shape[0])
    validate_cuda_tensor("patch_tris", patch_tris, dtype=torch.float32, ndim=3, trailing_shape=(3, 3))
    validate_cuda_tensor("rows", rows, dtype=torch.int64, ndim=1)
    validate_cuda_tensor("heights", heights, dtype=torch.float32, ndim=2)
    validate_cuda_tensor("grad_total", grad_total, dtype=torch.complex64, ndim=0)
    _validate_chain_leg(
        "c1", c1_positions, c1_normals, c1_eps_r, c1_sigma_e, c1_mu_r, c1_gain,
        c1_thickness, c1_depth, n,
    )
    _validate_chain_leg(
        "c2", c2_positions, c2_normals, c2_eps_r, c2_sigma_e, c2_mu_r, c2_gain,
        c2_thickness, c2_depth, n,
    )
    validate_cuda_tensor("material_id", material_id, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("layer_offset", layer_offset, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("layer_count", layer_count, dtype=torch.int32, ndim=1)
    quad_a, quad_b, quad_w = _duffy_nodes(patch_tris.device)
    out = _required_native_op("scattering_chain_realization_eval_backward")(
        patch_tris,
        patch_uvs,
        rows,
        d_i,
        d_o,
        n_rows,
        source,
        vertex,
        target,
        c1_positions,
        c1_normals,
        c1_eps_r,
        c1_sigma_e,
        c1_mu_r,
        c1_gain,
        c1_thickness,
        c1_depth,
        c2_positions,
        c2_normals,
        c2_eps_r,
        c2_sigma_e,
        c2_mu_r,
        c2_gain,
        c2_thickness,
        c2_depth,
        tx_pol,
        rx_pol,
        L1,
        L2,
        sp1,
        sp2,
        centroids,
        heights,
        cos_spec,
        material_id,
        layer_offset,
        layer_count,
        layer_thickness_m,
        layer_eps_r,
        layer_sigma_e,
        layer_mu_r,
        quad_a,
        quad_b,
        quad_w,
        float(k0),
        float(frequency_hz),
        grad_total,
        grad_path_field,
        grad_path_gain,
        bool(need_grad_heights),
        bool(need_grad_layers),
        bool(need_grad_chain1),
        bool(need_grad_chain2),
        bool(need_grad_geometry),
        bool(need_grad_k0),
        bool(need_grad_frequency),
    )
    if not isinstance(out, dict) or set(out) != set(_CHAIN_REALIZATION_BACKWARD_FIELDS):
        raise TypeError(
            "_channel_native.scattering_chain_realization_eval_backward returned"
            " invalid fields"
        )
    return out


def scattering_chain_realization_eval_jvp(
    patch_tris: torch.Tensor,
    patch_uvs: torch.Tensor,
    rows: torch.Tensor,
    d_i: torch.Tensor,
    d_o: torch.Tensor,
    n_rows: torch.Tensor,
    source: torch.Tensor,
    vertex: torch.Tensor,
    target: torch.Tensor,
    c1_positions: torch.Tensor,
    c1_normals: torch.Tensor,
    c1_eps_r: torch.Tensor,
    c1_sigma_e: torch.Tensor,
    c1_mu_r: torch.Tensor,
    c1_gain: torch.Tensor,
    c1_thickness: torch.Tensor,
    c1_depth: torch.Tensor,
    c2_positions: torch.Tensor,
    c2_normals: torch.Tensor,
    c2_eps_r: torch.Tensor,
    c2_sigma_e: torch.Tensor,
    c2_mu_r: torch.Tensor,
    c2_gain: torch.Tensor,
    c2_thickness: torch.Tensor,
    c2_depth: torch.Tensor,
    tx_pol: torch.Tensor,
    rx_pol: torch.Tensor,
    L1: torch.Tensor,
    L2: torch.Tensor,
    sp1: torch.Tensor,
    sp2: torch.Tensor,
    centroids: torch.Tensor,
    heights: torch.Tensor,
    cos_spec: torch.Tensor,
    material_id: torch.Tensor,
    layer_offset: torch.Tensor,
    layer_count: torch.Tensor,
    layer_thickness_m: torch.Tensor,
    layer_eps_r: torch.Tensor,
    layer_sigma_e: torch.Tensor,
    layer_mu_r: torch.Tensor,
    *,
    k0: float,
    frequency_hz: float,
    tangent_heights: torch.Tensor | None = None,
    tangent_layer_thickness: torch.Tensor | None = None,
    tangent_layer_eps_r: torch.Tensor | None = None,
    tangent_layer_sigma_e: torch.Tensor | None = None,
    tangent_c1_eps_r: torch.Tensor | None = None,
    tangent_c1_sigma_e: torch.Tensor | None = None,
    tangent_c1_gain: torch.Tensor | None = None,
    tangent_c1_thickness: torch.Tensor | None = None,
    tangent_c2_eps_r: torch.Tensor | None = None,
    tangent_c2_sigma_e: torch.Tensor | None = None,
    tangent_c2_gain: torch.Tensor | None = None,
    tangent_c2_thickness: torch.Tensor | None = None,
    tangent_d_i: torch.Tensor | None = None,
    tangent_d_o: torch.Tensor | None = None,
    tangent_c1_positions: torch.Tensor | None = None,
    tangent_c1_normals: torch.Tensor | None = None,
    tangent_c2_positions: torch.Tensor | None = None,
    tangent_c2_normals: torch.Tensor | None = None,
    tangent_L1: torch.Tensor | None = None,
    tangent_L2: torch.Tensor | None = None,
    tangent_sp1: torch.Tensor | None = None,
    tangent_sp2: torch.Tensor | None = None,
    tangent_centroids: torch.Tensor | None = None,
    tangent_k0: float = 0.0,
    tangent_frequency: float = 0.0,
) -> dict[str, torch.Tensor]:
    """JVP of :func:`scattering_chain_realization_eval` (ADR-021 Op B, plan 10a s4.3).

    Deterministic fixed-order dual sweep (no atomics). A missing tangent is a
    zero tangent. Returns the per-row and total field tangents D3 consumes.
    """

    validate_cuda_tensor("patch_tris", patch_tris, dtype=torch.float32, ndim=3, trailing_shape=(3, 3))
    quad_a, quad_b, quad_w = _duffy_nodes(patch_tris.device)
    out = _required_native_op("scattering_chain_realization_eval_jvp")(
        patch_tris,
        patch_uvs,
        rows,
        d_i,
        d_o,
        n_rows,
        source,
        vertex,
        target,
        c1_positions,
        c1_normals,
        c1_eps_r,
        c1_sigma_e,
        c1_mu_r,
        c1_gain,
        c1_thickness,
        c1_depth,
        c2_positions,
        c2_normals,
        c2_eps_r,
        c2_sigma_e,
        c2_mu_r,
        c2_gain,
        c2_thickness,
        c2_depth,
        tx_pol,
        rx_pol,
        L1,
        L2,
        sp1,
        sp2,
        centroids,
        heights,
        cos_spec,
        material_id,
        layer_offset,
        layer_count,
        layer_thickness_m,
        layer_eps_r,
        layer_sigma_e,
        layer_mu_r,
        quad_a,
        quad_b,
        quad_w,
        float(k0),
        float(frequency_hz),
        tangent_heights,
        tangent_layer_thickness,
        tangent_layer_eps_r,
        tangent_layer_sigma_e,
        tangent_c1_eps_r,
        tangent_c1_sigma_e,
        tangent_c1_gain,
        tangent_c1_thickness,
        tangent_c2_eps_r,
        tangent_c2_sigma_e,
        tangent_c2_gain,
        tangent_c2_thickness,
        tangent_d_i,
        tangent_d_o,
        tangent_c1_positions,
        tangent_c1_normals,
        tangent_c2_positions,
        tangent_c2_normals,
        tangent_L1,
        tangent_L2,
        tangent_sp1,
        tangent_sp2,
        tangent_centroids,
        float(tangent_k0),
        float(tangent_frequency),
    )
    if not isinstance(out, dict) or set(out) != set(_CHAIN_REALIZATION_TANGENT_FIELDS):
        raise TypeError(
            "_channel_native.scattering_chain_realization_eval_jvp returned invalid fields"
        )
    return out


__all__ = [
    "scattering_chain_ensemble_eval",
    "scattering_chain_ensemble_eval_backward",
    "scattering_chain_ensemble_eval_jvp",
    "scattering_chain_realization_eval",
    "scattering_chain_realization_eval_backward",
    "scattering_chain_realization_eval_jvp",
]

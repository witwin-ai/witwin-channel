"""Radio propagation tracer using mesh-based UTD simulation (Pure DrJit)."""

from __future__ import annotations

from collections.abc import Mapping

import drjit as dr
import torch
import witwin as wt

from ..config import ChannelConfig, TraceConfig, coerce_channel_config
from ..monitors.radio_map import RadioMapResult
from ..monitors.orchestration import resolve_trace_config
from ..scene import Scene
from ..utils.torch_bridge import wrap_drjit
from .output import TraceOutput, _resolve_monitor_payload
from .cache import TraceCacheManager
from .session import TraceSession, merge_monitor_overrides, normalize_monitor_overrides

_TRACE_OVERRIDE_KEYS = frozenset(TraceConfig.__dataclass_fields__) - {"diffraction_execution"}


class Tracer:
    """
    Radio propagation tracer for computing electromagnetic fields around obstacles.

    Uses 3D mesh representation with UTD (Uniform Theory of Diffraction) for
    accurate modeling of LoS, reflection, and diffraction paths.

    Keyword arguments matching TraceConfig fields (e.g. ``reflection_n_rays``,
    ``solver_mode``) override the corresponding value from *config*.
    """

    def __init__(
        self,
        frequency: float,
        scene: Scene,
        config: ChannelConfig | dict | None = None,
        **overrides,
    ):
        if scene is None:
            raise ValueError("Tracer requires a declarative Scene instance.")
        bad_keys = set(overrides) - _TRACE_OVERRIDE_KEYS
        if bad_keys:
            raise TypeError(f"Unexpected keyword arguments: {', '.join(sorted(bad_keys))}")

        self.scene = scene
        channel_config = coerce_channel_config(config)
        if overrides:
            base = channel_config.trace.to_dict()
            base.update(overrides)
            trace_config = TraceConfig.from_dict(base)
        else:
            trace_config = channel_config.trace
        self.config = ChannelConfig(trace=trace_config)
        self._resolved_trace_config = resolve_trace_config(
            frequency=float(frequency),
            config=self.config.trace,
        )

        resolved = self._resolved_trace_config
        self.frequency = resolved.frequency
        self.wavelength = resolved.wavelength
        self.k = resolved.k
        self.reflection_n_rays = resolved.reflection_n_rays
        self.reflection_max_bounces = resolved.reflection_max_bounces
        self.reflection_coef = resolved.reflection_coef
        self.min_ray_contribution_threshold = resolved.min_ray_contribution_threshold
        self.reflection_relative_permittivity = resolved.reflection_relative_permittivity
        self.reflection_conductivity = resolved.reflection_conductivity
        self.reflection_material = resolved.reflection_material
        self.diffraction_material = resolved.diffraction_material
        self.use_scene_materials_for_reflection = resolved.use_scene_materials_for_reflection
        self.use_scene_materials_for_diffraction = resolved.use_scene_materials_for_diffraction
        self.enable_rd_diffraction = resolved.enable_rd_diffraction
        self.max_diffractions = resolved.max_diffractions
        self.diffraction_state_budget = resolved.diffraction_state_budget
        self.inserted_reflection_state_budget = resolved.inserted_reflection_state_budget
        self.max_inserted_reflections_per_path = resolved.max_inserted_reflections_per_path
        self.solver_mode = resolved.solver_mode
        self.tx_polarization = resolved.tx_polarization
        self.rx_polarization = resolved.rx_polarization
        self.resolution_wavelength = resolved.resolution_wavelength
        self.cell_size = resolved.cell_size
        self._trace_caches = TraceCacheManager(
            scene=self.scene,
            trace_config=self.config.trace,
            resolved_trace_config=self._resolved_trace_config,
        )

    @staticmethod
    def _is_torch_tensor(value):
        return isinstance(value, torch.Tensor)

    @classmethod
    def _coerce_vertices(cls, vertices):
        if isinstance(vertices, wt.Point3f):
            return vertices

        if cls._is_torch_tensor(vertices):
            if vertices.ndim != 2 or vertices.shape[1] != 3:
                raise ValueError("Expected vertices torch tensor with shape (N, 3).")
            return wt.Point3f(
                wt.Float(vertices[:, 0].contiguous()),
                wt.Float(vertices[:, 1].contiguous()),
                wt.Float(vertices[:, 2].contiguous()),
            )

        rows = [tuple(vertex) for vertex in vertices]
        return wt.Point3f(
            wt.Float([float(vertex[0]) for vertex in rows]),
            wt.Float([float(vertex[1]) for vertex in rows]),
            wt.Float([float(vertex[2]) for vertex in rows]),
        )

    @classmethod
    def _coerce_faces(cls, faces):
        if isinstance(faces, wt.Vector3u):
            return faces

        if cls._is_torch_tensor(faces):
            if faces.ndim != 2 or faces.shape[1] != 3:
                raise ValueError("Expected faces torch tensor with shape (M, 3).")
            faces = faces.int()
            return wt.Vector3u(
                wt.UInt32(faces[:, 0].contiguous()),
                wt.UInt32(faces[:, 1].contiguous()),
                wt.UInt32(faces[:, 2].contiguous()),
            )

        rows = [tuple(face) for face in faces]
        return wt.Vector3u(
            wt.UInt32([int(face[0]) for face in rows]),
            wt.UInt32([int(face[1]) for face in rows]),
            wt.UInt32([int(face[2]) for face in rows]),
        )

    def _clear_trace_caches(self):
        self._trace_caches.clear()

    def _refresh_trace_caches(self):
        self._trace_caches.refresh()

    def _resolve_monitor_solver_controls(self, monitor, *, execution_intent):
        return self._trace_caches.resolve_monitor_solver_controls(
            monitor,
            execution_intent=execution_intent,
        )

    def update_scene(self, vertices, recompute_edges: bool = True):
        needs_grad = self._is_torch_tensor(vertices) and bool(vertices.requires_grad)
        vertices = self._coerce_vertices(vertices)
        if needs_grad:
            dr.enable_grad(vertices.x, vertices.y, vertices.z)
        self.scene.update_vertices(vertices, recompute_edges=recompute_edges)
        self._clear_trace_caches()
        return self.scene

    def create_intersection_func(self):
        @wrap_drjit
        def intersect(
            vertices_flat: wt.Float,
            ray_ox: wt.Float,
            ray_oy: wt.Float,
            ray_oz: wt.Float,
            ray_dx: wt.Float,
            ray_dy: wt.Float,
            ray_dz: wt.Float,
        ) -> tuple:
            n_values = dr.width(vertices_flat)
            if n_values % 3 != 0:
                raise ValueError("Flattened vertex buffer must contain xyz triplets.")

            vertices = wt.Point3f(
                vertices_flat[0::3],
                vertices_flat[1::3],
                vertices_flat[2::3],
            )
            self.update_scene(vertices, recompute_edges=False)

            ray = wt.Ray(
                wt.Point3f(ray_ox, ray_oy, ray_oz),
                wt.Vector3f(ray_dx, ray_dy, ray_dz),
            )
            si = self.scene.ray_intersect(ray)
            return si.p.x, si.p.y, si.p.z, si.n.x, si.n.y, si.n.z, si.t

        return intersect

    @staticmethod
    def _coerce_tx_pos(tx_pos):
        if isinstance(tx_pos, wt.Point3f):
            return tx_pos
        if isinstance(tx_pos, torch.Tensor):
            return wt.Point3f(tx_pos[0].item(), tx_pos[1].item(), tx_pos[2].item())
        return wt.Point3f(float(tx_pos[0]), float(tx_pos[1]), float(tx_pos[2]))

    def trace(
        self,
        tx_pos,
        *,
        monitor=None,
        monitor_overrides: Mapping[str, Mapping[str, object]] | None = None,
        verbose: bool = False,
        return_timing: bool = False,
        return_diffraction_audit: bool = False,
    ) -> TraceOutput:
        """
        Compute electromagnetic field distribution.

        Args:
            tx_pos: Transmitter position (x, y, z) - tuple, list, or wt.Point3f
            monitor: Optional FieldMonitor, PathMonitor, or iterable of them.
                When omitted, the tracer uses monitors declared on the Scene.
                When provided, the trace-time monitor list overrides Scene monitors.
            monitor_overrides: Optional mapping of monitor name -> override mapping.
                This applies trace-local `with_overrides(...)` updates without mutating
                scene-attached monitors. For example, `{"rx": {"positions": ...}}`
                overrides the receiver set for a `PathMonitor`.
            verbose: Print debug information
            return_timing: If True, include timing info in result dict
            return_diffraction_audit: If True, include diffraction audit info

        Returns:
            The traced monitor payload directly when exactly one monitor payload
            is produced, otherwise a ``dict[name, payload]`` spanning all field,
            radio-map, and path monitor outputs.
        """
        session = TraceSession(
            self,
            tx_pos=self._coerce_tx_pos(tx_pos),
            monitor=monitor,
            monitor_overrides=monitor_overrides,
            verbose=verbose,
            return_timing=return_timing,
            return_diffraction_audit=return_diffraction_audit,
        )
        return session.run()

    @staticmethod
    def _radio_map_signature(payload: RadioMapResult) -> tuple[object, ...]:
        surface = dict(payload.surface)
        receiver_sampling = dict(payload.metadata.get("receiver_sampling", {}))
        monte_carlo = dict(payload.metadata.get("monte_carlo", {}))
        accumulation_backend = dict(payload.metadata.get("accumulation_backend", {}))
        bounds = surface.get("bounds")
        if bounds is not None:
            bounds = tuple(tuple(float(value) for value in pair) for pair in bounds)
        center = surface.get("center")
        if center is not None:
            center = tuple(float(value) for value in center)
        orientation = surface.get("orientation")
        if orientation is not None:
            orientation = tuple(float(value) for value in orientation)
        size = surface.get("size")
        if size is not None:
            size = tuple(float(value) for value in size)
        sample_offsets = receiver_sampling.get("sample_offsets_local")
        if sample_offsets is not None:
            sample_offsets = tuple(
                tuple(float(value) for value in offset)
                for offset in sample_offsets
            )
        sampling_mode = str(receiver_sampling.get("sampling_mode", "deterministic"))
        return (
            tuple(int(value) for value in payload.grid_shape),
            tuple(float(value) for value in payload.cell_size),
            str(surface.get("surface_mode")),
            str(payload.receiver_model),
            surface.get("axis"),
            None if surface.get("position") is None else float(surface.get("position")),
            bounds,
            center,
            orientation,
            size,
            sampling_mode,
            receiver_sampling.get("strategy"),
            receiver_sampling.get("quadrature_mode"),
            receiver_sampling.get("samples_per_cell"),
            sample_offsets,
            receiver_sampling.get("samples_per_tx"),
            receiver_sampling.get("seed"),
            monte_carlo.get("ad_mode"),
            monte_carlo.get("rr_depth"),
            monte_carlo.get("rr_prob"),
            monte_carlo.get("stop_threshold_db"),
            accumulation_backend.get("resolved"),
        )

    @staticmethod
    def _aggregate_radio_map_rss_streaming(payloads):
        rss_tensors = []
        total_rss = None
        best_rss = None
        winner = None
        for tx_index, payload in enumerate(payloads):
            rss = payload.metric_tensor("rss")
            rss_tensors.append(rss)
            if total_rss is None:
                total_rss = rss.clone()
                best_rss = rss.clone()
                winner = torch.zeros_like(rss, dtype=torch.int32)
                continue
            total_rss = total_rss + rss
            better = rss > best_rss
            winner = torch.where(
                better,
                torch.full_like(winner, int(tx_index)),
                winner,
            )
            best_rss = torch.where(better, rss, best_rss)
        return rss_tensors, total_rss, winner

    def _apply_trace_many_radio_map_aggregation(
        self,
        sessions: list[TraceSession],
        *,
        request_labels: list[str],
    ) -> list[TraceSession]:
        def _resolved_radio_map_payload(payload):
            resolved = _resolve_monitor_payload(payload)
            return resolved if isinstance(resolved, RadioMapResult) else None

        radio_map_names = sorted(
            {
                monitor_name
                for session in sessions
                for monitor_name, payload in session.monitor_payloads.items()
                if _resolved_radio_map_payload(payload) is not None
            }
        )
        for monitor_name in radio_map_names:
            payloads = []
            payload_sessions = []
            labels = []
            for result_index, session in enumerate(sessions):
                payload = _resolved_radio_map_payload(session.monitor_payloads.get(monitor_name))
                if payload is None:
                    continue
                payloads.append(payload)
                payload_sessions.append(session)
                labels.append(str(request_labels[result_index]))
            if len(payloads) == 0:
                continue
            signature = self._radio_map_signature(payloads[0])
            for payload in payloads[1:]:
                if self._radio_map_signature(payload) != signature:
                    raise ValueError(
                        f"RadioMapMonitor '{monitor_name}' cannot be aggregated across trace_many() "
                        "because the sampled surfaces do not match."
                    )
            rss_tensors, total_rss, winner = self._aggregate_radio_map_rss_streaming(payloads)
            inf = torch.full_like(total_rss, float("inf"))
            zero = torch.zeros_like(total_rss)
            for tx_index, (payload, session, rss_tensor) in enumerate(
                zip(payloads, payload_sessions, rss_tensors)
            ):
                interference = total_rss - rss_tensor
                denominator = interference + float(payload.noise_power)
                sinr = torch.where(
                    denominator > 0.0,
                    rss_tensor / denominator,
                    torch.where(rss_tensor > 0.0, inf, zero),
                )
                metadata = dict(payload.metadata)
                metadata["aggregate_tx_count"] = int(len(payloads))
                metadata["aggregate_tx_labels"] = tuple(labels)
                metadata["aggregate_metric_source"] = "trace_many"
                metadata["aggregate_tx_index"] = int(tx_index)
                metadata["tx_stack_execution"] = {
                    "mode": "trace_many_streaming_post_aggregation",
                    "native": False,
                    "rss_stack_materialized": False,
                    "tx_count": int(len(payloads)),
                }
                session.monitor_payloads[monitor_name] = payload.with_metric_overrides(
                    sinr=sinr,
                    tx_association=winner,
                    metadata=metadata,
                )
        return sessions

    def trace_many(
        self,
        trace_requests,
        *,
        monitor=None,
        monitor_overrides: Mapping[str, Mapping[str, object]] | None = None,
        verbose: bool = False,
        return_timing: bool = False,
        return_diffraction_audit: bool = False,
    ) -> tuple[TraceOutput, ...]:
        """
        Run multiple single-transmitter traces in one call.

        Each item in ``trace_requests`` may be either a transmitter position or a
        mapping with:

        - ``tx_pos``: required transmitter position override
        - ``monitor``: optional monitor or monitor iterable override
        - ``monitor_overrides``: optional monitor-name override mapping
        - ``tx_label``: optional label used by aggregated RadioMapMonitor outputs
        """
        sessions = []
        request_labels = []
        normalized_monitor_overrides = normalize_monitor_overrides(monitor_overrides)
        for request in trace_requests:
            request_monitor = monitor
            request_monitor_overrides = normalized_monitor_overrides
            if isinstance(request, Mapping):
                if "tx_pos" not in request:
                    raise ValueError("trace_many request mappings must include 'tx_pos'.")
                request_tx_pos = request["tx_pos"]
                if "monitor" in request:
                    request_monitor = request["monitor"]
                request_monitor_overrides = merge_monitor_overrides(
                    normalized_monitor_overrides,
                    normalize_monitor_overrides(request.get("monitor_overrides")),
                )
                request_label = request.get("tx_label", request.get("label", len(sessions)))
            else:
                request_tx_pos = request
                request_label = len(sessions)
            session = TraceSession(
                self,
                tx_pos=self._coerce_tx_pos(request_tx_pos),
                monitor=request_monitor,
                monitor_overrides=request_monitor_overrides,
                verbose=verbose,
                return_timing=return_timing,
                return_diffraction_audit=return_diffraction_audit,
            )
            session.execute()
            sessions.append(session)
            request_labels.append(str(request_label))
        sessions = self._apply_trace_many_radio_map_aggregation(
            sessions,
            request_labels=request_labels,
        )
        return tuple(session.finalize_output() for session in sessions)


__all__ = ["Tracer"]

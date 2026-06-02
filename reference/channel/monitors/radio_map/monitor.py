from __future__ import annotations

from dataclasses import dataclass, field
import math

from . import options as rm_options
from . import surface as rm_surface
from . import types as rm_types
from . import validation as rm_validation


@dataclass(slots=True)
class RadioMapMonitor:
    name: str
    axis: str | None = "z"
    position: float | None = 0.0
    bounds: tuple[tuple[float, float], tuple[float, float]] | None = ((-8.0, 8.0), (-8.0, 8.0))
    center: tuple[float, float, float] | None = None
    orientation: tuple[float, float, float] | None = None
    size: tuple[float, float] | None = None
    grid_shape: tuple[int, int] | None = None
    cell_size: tuple[float, float] | None = None
    metric: rm_types.RadioMapMetric | str = rm_types.RadioMapMetric.PATH_GAIN
    combine_mode: rm_types.CombineMode | str = rm_types.CombineMode.INCOHERENT
    receiver_model: rm_types.ReceiverModel | str | None = None
    accumulation_backend: rm_types.AccumulationBackend | str = rm_types.AccumulationBackend.AUTO
    tx_power: float = 1.0
    noise_power: float | None = None
    ray_mode: str = "3d"
    max_diffractions: int | None = 1
    sampling_mode: rm_types.SamplingMode | str = rm_types.SamplingMode.DETERMINISTIC
    ad: bool | None = None
    samples_per_tx: int | None = None
    rr_depth: int | None = None
    rr_prob: float | None = None
    stop_threshold: float | None = None
    seed: int | None = 0
    quadrature_mode: str = "center"
    samples_per_cell: int | None = None
    shadow_boundary_mode: rm_types.ShadowBoundaryMode | str = rm_types.ShadowBoundaryMode.NONE
    shadow_support_cutoff_db: float | None = None
    kind: str = field(init=False, default="radio_map")
    surface_mode: rm_types.SurfaceMode = field(init=False, default=rm_types.SurfaceMode.AXIS_ALIGNED)

    def __post_init__(self):
        resolved_surface = rm_surface.resolve_surface_fields(
            axis=self.axis,
            position=self.position,
            bounds=self.bounds,
            center=self.center,
            orientation=self.orientation,
            size=self.size,
            grid_shape=self.grid_shape,
            cell_size=self.cell_size,
        )
        resolved_metric = rm_options.normalize_metric(self.metric)
        resolved_quadrature_mode, resolved_samples_per_cell = rm_surface.normalize_quadrature_mode(
            self.quadrature_mode,
            samples_per_cell=self.samples_per_cell,
        )
        self.name = str(self.name)
        self.axis = resolved_surface["axis"]
        self.position = resolved_surface["position"]
        self.bounds = resolved_surface["bounds"]
        self.center = resolved_surface["center"]
        self.orientation = resolved_surface["orientation"]
        self.size = resolved_surface["size"]
        self.grid_shape = resolved_surface["grid_shape"]
        self.cell_size = resolved_surface["cell_size"]
        self.metric = resolved_metric
        resolved_combine_mode = rm_options.normalize_combine_mode(self.combine_mode)
        self.combine_mode = resolved_combine_mode
        resolved_sampling_mode = rm_options.normalize_sampling_mode(self.sampling_mode)
        self.receiver_model = rm_options.normalize_receiver_model(
            self.receiver_model,
            combine_mode=resolved_combine_mode,
            surface_mode=resolved_surface["surface_mode"],
        )
        self.accumulation_backend = rm_options.normalize_accumulation_backend(self.accumulation_backend)
        self.tx_power = rm_options.normalize_positive_power(self.tx_power, name="tx_power")
        self.noise_power = rm_options.normalize_positive_power(self.noise_power, name="noise_power")
        self.ray_mode = rm_options.normalize_ray_mode(self.ray_mode)
        self.max_diffractions = rm_options.normalize_max_diffractions_override(self.max_diffractions)
        self.sampling_mode = resolved_sampling_mode
        self.ad = rm_options.normalize_optional_bool(self.ad, name="ad")
        resolved_samples_per_tx = rm_options.normalize_positive_int(
            self.samples_per_tx,
            name="samples_per_tx",
        )
        resolved_rr_depth = rm_options.normalize_nonnegative_int(self.rr_depth, name="rr_depth")
        resolved_rr_prob = rm_options.normalize_probability(self.rr_prob, name="rr_prob")
        if resolved_rr_depth is None:
            resolved_rr_prob = 1.0 if resolved_rr_prob is None else resolved_rr_prob
        elif resolved_rr_prob is None:
            resolved_rr_prob = 0.5
        self.samples_per_tx = (
            65536
            if resolved_sampling_mode == rm_types.SamplingMode.MONTE_CARLO and resolved_samples_per_tx is None
            else resolved_samples_per_tx
        )
        self.rr_depth = resolved_rr_depth
        self.rr_prob = resolved_rr_prob
        self.stop_threshold = rm_options.normalize_nonnegative_threshold(
            self.stop_threshold,
            name="stop_threshold",
        )
        self.seed = rm_options.normalize_seed(self.seed)
        self.quadrature_mode = resolved_quadrature_mode
        self.samples_per_cell = resolved_samples_per_cell
        resolved_shadow_boundary_mode = rm_options.normalize_shadow_boundary_mode(
            self.shadow_boundary_mode,
        )
        rm_validation.validate_shadow_boundary_contract(
            shadow_boundary_mode=resolved_shadow_boundary_mode,
            combine_mode=resolved_combine_mode,
            receiver_model=self.receiver_model,
            quadrature_mode=resolved_quadrature_mode,
        )
        self.shadow_boundary_mode = resolved_shadow_boundary_mode
        self.shadow_support_cutoff_db = rm_options.normalize_shadow_support_cutoff_db(
            self.shadow_support_cutoff_db,
        )
        self.surface_mode = resolved_surface["surface_mode"]
        rm_validation.validate_monte_carlo_contract(
            sampling_mode=self.sampling_mode,
            surface_mode=self.surface_mode,
            max_diffractions=self.max_diffractions,
            combine_mode=resolved_combine_mode,
            receiver_model=self.receiver_model,
            accumulation_backend=self.accumulation_backend,
            shadow_boundary_mode=resolved_shadow_boundary_mode,
        )

    @property
    def tangential_axes(self) -> tuple[str, str]:
        if self.surface_mode == "axis_aligned":
            from ...utils.plane_axes import tangential_axes_for_axis

            return tangential_axes_for_axis(self.axis)
        return ("u", "v")

    @property
    def spans(self) -> tuple[float, float]:
        if self.surface_mode == "axis_aligned":
            return (
                self.bounds[0][1] - self.bounds[0][0],
                self.bounds[1][1] - self.bounds[1][0],
            )
        return self.size

    def resolve_grid_shape(
        self,
        *,
        default_cell_size: float | tuple[float, float] | None = None,
    ) -> tuple[int, int]:
        if self.grid_shape is not None:
            return self.grid_shape

        resolved_default = rm_surface.normalize_optional_point2(
            default_cell_size,
            name="default_cell_size",
        )
        requested_cell_size = self.cell_size if self.cell_size is not None else resolved_default
        if requested_cell_size is None:
            raise ValueError(
                "RadioMapMonitor requires grid_shape or cell_size, or Tracer must provide a default cell_size."
            )
        span_0, span_1 = self.spans
        nx = max(1, int(math.ceil(span_0 / requested_cell_size[0])))
        ny = max(1, int(math.ceil(span_1 / requested_cell_size[1])))
        return (nx, ny)

    def resolve_cell_size(
        self,
        *,
        default_cell_size: float | tuple[float, float] | None = None,
    ) -> tuple[float, float]:
        span_0, span_1 = self.spans
        grid_shape = self.resolve_grid_shape(default_cell_size=default_cell_size)
        return (span_0 / float(grid_shape[0]), span_1 / float(grid_shape[1]))

    def with_overrides(self, **overrides) -> "RadioMapMonitor":
        return RadioMapMonitor(
            overrides.get("name", self.name),
            axis=overrides.get("axis", self.axis if self.axis is not None else "z"),
            position=overrides.get(
                "position",
                0.0 if self.position is None else self.position,
            ),
            bounds=overrides.get(
                "bounds",
                ((-1.0, 1.0), (-1.0, 1.0)) if self.bounds is None else self.bounds,
            ),
            center=overrides.get("center", self.center),
            orientation=overrides.get("orientation", self.orientation),
            size=overrides.get("size", self.size),
            grid_shape=overrides.get("grid_shape", self.grid_shape),
            cell_size=overrides.get("cell_size", self.cell_size),
            metric=overrides.get("metric", self.metric),
            combine_mode=overrides.get("combine_mode", self.combine_mode),
            receiver_model=overrides.get("receiver_model", self.receiver_model),
            accumulation_backend=overrides.get(
                "accumulation_backend",
                self.accumulation_backend,
            ),
            tx_power=overrides.get("tx_power", self.tx_power),
            noise_power=overrides.get("noise_power", self.noise_power),
            ray_mode=overrides.get("ray_mode", self.ray_mode),
            max_diffractions=overrides.get("max_diffractions", self.max_diffractions),
            sampling_mode=overrides.get("sampling_mode", self.sampling_mode),
            ad=overrides.get("ad", self.ad),
            samples_per_tx=overrides.get("samples_per_tx", self.samples_per_tx),
            rr_depth=overrides.get("rr_depth", self.rr_depth),
            rr_prob=overrides.get("rr_prob", self.rr_prob),
            stop_threshold=overrides.get("stop_threshold", self.stop_threshold),
            seed=overrides.get("seed", self.seed),
            quadrature_mode=overrides.get("quadrature_mode", self.quadrature_mode),
            samples_per_cell=overrides.get("samples_per_cell", self.samples_per_cell),
            shadow_boundary_mode=overrides.get(
                "shadow_boundary_mode",
                self.shadow_boundary_mode,
            ),
            shadow_support_cutoff_db=overrides.get(
                "shadow_support_cutoff_db",
                self.shadow_support_cutoff_db,
            ),
        )


def resolve_radio_map_monitor(monitor) -> RadioMapMonitor:
    if isinstance(monitor, RadioMapMonitor):
        return monitor
    raise TypeError("Channel monitors must be RadioMapMonitor instances.")


__all__ = [
    "RadioMapMonitor",
    "resolve_radio_map_monitor",
]

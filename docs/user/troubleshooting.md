# Troubleshooting

Start with the doctor command; it checks the things below and prints a fix for
each failure:

```bash
python -m witwin.channel doctor
```

`witwin.channel.runtime_diagnostics()` returns the same information as a
dictionary, including the trace backend each compiled scene uses. The README's
**Installation** and **Troubleshooting** sections cover installing the right
PyTorch build and the loader errors.

## OptiX or the pure-CUDA trace backend

Channel traces rays through RayD, which has two interchangeable triangle
tracers: NVIDIA OptiX (hardware ray tracing, the default when usable) and a
pure-CUDA BVH. Both produce the same discrete results; OptiX is faster on
tracing-heavy scenes. The tracer is chosen when a scene is compiled:

| `WITWIN_CHANNEL_TRACE_BACKEND` | Behaviour |
| --- | --- |
| `auto` (default) | OptiX when the driver's OptiX runtime is usable, otherwise the CUDA tracer |
| `optix` | OptiX or a clear error explaining why it is unusable |
| `cuda` | Always the CUDA tracer |

OptiX needs the driver's OptiX library (`libnvoptix.so.1` on Linux,
`nvoptix.dll` on Windows) from an R555 or newer driver. It does not need the
OptiX SDK, a display or an X server. If a solve is much slower than expected,
check which backend `doctor` reports.

## Docker

- Start containers with `--gpus all` (or `--gpus device=N`).
- The NVIDIA Container Toolkit mounts the OptiX library only with the
  `graphics` driver capability. Set
  `NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics` (or `all`); the default
  `compute,utility` silently drops OptiX and Channel runs on the CUDA tracer.
  The repository's `Dockerfile` sets this already.
- Persist the OptiX program cache between container runs (mount a volume and
  point `OPTIX_CACHE_PATH` at it) to avoid the first-solve compile in every new
  container.

## WSL2

- Install the NVIDIA Windows driver (R555 or newer) on Windows and no Linux
  NVIDIA driver inside the distribution; WSL exposes the Windows driver under
  `/usr/lib/wsl/lib`.
- Install the CUDA 12.8 PyTorch build inside WSL as on native Linux.
- Whether the driver's OptiX library is visible inside WSL depends on the driver
  release. If `doctor` reports OptiX unavailable, Channel still runs on the
  CUDA tracer. Channel has not been validated under WSL2.

## Headless servers and clusters

- No display, X server or desktop is required, only the driver and a visible
  GPU (`nvidia-smi` lists it).
- `CUDA_VISIBLE_DEVICES` selects the GPU; Channel uses PyTorch's current CUDA
  device (`torch.cuda.set_device`). Several GPUs in one process are not tested;
  run one process per GPU.
- On shared nodes, give each process its own `OPTIX_CACHE_PATH` when several
  processes start at once.
- The examples write PNGs through matplotlib's non-interactive `Agg` backend,
  which needs no display.

## First run is slow

The first solve in a fresh environment compiles OptiX programs (a few seconds;
longer with diffraction). The driver caches them on disk, so later solves and
later processes start in about a second. A GPU newer than every cubin in the
wheel JIT-compiles the PTX once in the same way.

## Scene import problems

| Symptom | Cause and fix |
| --- | --- |
| `ITU-R P.2040 material 'brick' is defined for 1-40 GHz; ... is outside every valid range` | The ITU law is not defined at that frequency. Use another material, or pass `materials={"bsdf id": PhysicalMaterial(...)}` with your own values. |
| `Mitsuba bsdf 'x' of type 'diffuse' is not a radio material` | The scene uses a visual material. Map it with `materials={"x": ...}`. |
| `unsupported Mitsuba shape type 'sphere'` | Analytic shapes are not tessellated. Export them as meshes. |
| `required glTF extensions are not supported: ['KHR_draco_mesh_compression']` | Re-export without mesh compression. |
| Many `welded_vertices` in the report | Normal for exports that split vertices along seams; welding restores shared edges. Use `weld=False` to compare. |
| Radiomap is empty in places you expect coverage | The transmitter or receiver plane may be inside a building, or below the ground. Check the scene bounds and heights. |
| A map looks transposed | Grid maps are returned as `(T, rows, cols)` for `ReceiverGrid(shape=(rows, cols))`: the first map axis follows the grid's `x_axis`, the second its `y_axis`. Show `path_gain[t].T` with `origin="lower"` for x to the right. |

## Material and frequency errors

- `frequency_offsets_hz is not supported on a scene with frequency-dependent
  materials`: the scene has dispersive materials (the default ITU presets).
  Build the presets with `dispersive=False`, or compile one scene per
  frequency.
- `ValueError` about the compile frequency of an ITU material: the scene was
  compiled outside a material's valid range; see
  [Materials and antennas](materials-and-antennas.md).
- `... does not support directional antenna patterns`: only the Path solver
  uses antenna patterns; use isotropic endpoints with the other solvers.

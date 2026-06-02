# RayD Deterministic Exact Coherent Accumulator - 2026-05-23

## Scope

This note records the first opt-in RayD path for deterministic first-order coherent diffraction accumulation:

- Channel mode: `Tuning(diffraction_execution={"accumulate_primal": "rayd_exact_coherent"})`
- RayD API: `Scene.accum_dfr_coherent_direct(full_utd_states, grid, material, options, active=True)`
- Supported workload: first-order deterministic diffraction on axis-aligned receiver grids.
- AD contract: primal-only; Channel rejects AD-sensitive inputs before calling RayD.

`auto` now uses this RayD path for supported primal workloads: first-, second-, and third-order deterministic diffraction state collections, axis-aligned receiver grid, no suffix diffraction, no `shadow_support_cutoff_db` override, and no AD-sensitive state/scene inputs. Unsupported and AD workloads keep the previous DrJit/native path.

## Benchmark Smoke

Command shape:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m tests.support.bin.profile_deterministic_munich --munich-xml E:\Code\witwin-platform\channel\reference\sionna-rt-reference-2.0.1\src\sionna\rt\scenes\munich\munich.xml --grid-size 16 --max-diffractions 1 --reflection-n-rays 1 --reflection-max-bounces 0 --edge-selection-mode all_edges --boundary-edge-policy half_plane --solver-mode fast_approximate --memory-profile memory_safe --assert-finite --json
```

| Mode | Diffraction States | Diffraction Accumulation | Total Solve | Peak Delta |
| --- | ---: | ---: | ---: | ---: |
| Legacy Channel native | 4164 | 75.069 s | 76.871 s | 4977 MiB |
| `auto` resolved to RayD exact coherent | 4164 | 0.0381 s | 5.938 s | 881 MiB |

Observed accumulation speedup on this smoke workload is about `1971x`; total solve speedup is about `12.9x`.

## Munich 256x256 Order Scaling

Command shape:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m tests.support.bin.profile_deterministic_munich --munich-xml E:\Code\witwin-platform\channel\reference\sionna-rt-reference-2.0.1\src\sionna\rt\scenes\munich\munich.xml --grid-size 256 --max-diffractions <1|2|3> --reflection-n-rays 1 --reflection-max-bounces 0 --edge-selection-mode all_edges --boundary-edge-policy half_plane --solver-mode fast_approximate --memory-profile memory_safe --diffraction-accumulate-primal auto --assert-finite --json
```

| Max Diffraction Order | States | State Preparation | RayD Accumulation | Total Solve | Peak Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 4164 | 5.496 s | 0.193 s | 5.915 s | 578 MiB |
| 2 | 4932 | 8.605 s | 0.267 s | 9.053 s | 915 MiB |
| 3 | 5629 | 10.311 s | 0.327 s | 10.831 s | 1162 MiB |

These runs use the same RayD exact coherent grid accumulator for every packed deterministic UTD state in the first-/higher-order collection. State preparation, not accumulation, is now the dominant cost.

## Accuracy Status

The RayD kernel now accepts Channel's full deterministic UTD state payload and evaluates the same native UTD vector formula path. Two parity fixes were required after the initial full-vector landing:

- Channel must pass the same `omega = material_angular_frequency(wavelength)` and transmitter polarization used by the native UTD path, otherwise RayD uses stored face operators while Channel recomputes Fresnel/vector face operators.
- RayD finite-edge stationary-point validity must require `0 < parameter < edge_length`; accepting finite but out-of-segment stationary points introduced extra strong boundary cells.

| Mode | Diffraction Sum | Nonzero Cells |
| --- | ---: | ---: |
| Legacy Channel native | `1.4506045e-08` | 143 |
| `auto` resolved to RayD exact coherent | `1.4506975e-08` | 163 |

The remaining nonzero-cell count difference is tiny boundary-tail support, not a material aggregate mismatch: per-cell diagnostics report total diffraction delta `9.3e-13` on the `16 x 16` Munich smoke. Keep AD on the existing path until a fixed-topology RayD coherent diffraction derivative is implemented and tested.

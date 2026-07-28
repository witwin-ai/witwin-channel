# BDPT Native Baselines

Initial native BDPT gates use `benchmarks/bench_bdpt_basic.py` and
`benchmarks/bench_bdpt_munich.py`. The maintained smoke budgets are:

- Empty-space LoS BDPT: no more than 1.25x MC basic LoS, with a small fixed
  50 ms tolerance for local variance.
- Reduced Munich BDPT smoke: component maps must be finite and non-empty when
  CUDA is available; metadata must report launch count and selected
  accumulation strategy, and the benchmark reports CUDA-synchronized native
  solve timing.
- Reduced Munich original/native BDPT parity: use
  `tests.support.bin.benchmark_munich_bdpt_native_vs_original`. The default
  `synthetic_reduced` scene runs original `witwin.channel` in a subprocess
  without loading the full Munich XML, records native/original timing, component
  correlations, dB delta artifacts, and strict parity gates. The maintained
  reduced gate is green for steady-state solves after one native warmup. Cold
  first-use timings include OptiX/native pipeline setup and are reported
  separately from the steady-state speed gate.
- Single-plane open-mesh original/native BDPT speed gate: use
  `tests.support.bin.benchmark_single_plane_bdpt_native_vs_original`. This script
  runs original `witwin.channel` BDPT in a subprocess and native
  `witwin.channel.montecarlo.bdpt` in-process on the same hand-authored
  open two-triangle plane scene. Strict gates require matching radiomap shape,
  nonzero native/original maps, and native median solve time faster than original
  with at least the configured minimum speedup.
- The former crude receiver-grid and point diffraction connection exporters
  were removed by Plan 13 Phase 4 after failing the four-axis reachability
  audit. Standalone BDPT diffraction now uses the opaque enumerated-path oracle;
  MC Basic retains the fused RayD sample-tape producer plus the Channel UTD
  tape accumulator. The historical timings below predate that cleanup and are
  retained as baseline evidence, not as a live ABI contract.

Regenerate local numbers with:

```powershell
conda run -n witwin2 python benchmarks/bench_bdpt_basic.py --json
conda run -n witwin2 python benchmarks/bench_bdpt_munich.py --json
conda run -n witwin2 python -m tests.support.bin.benchmark_munich_bdpt_native_vs_original --json
conda run -n witwin2 python -m tests.support.bin.benchmark_single_plane_bdpt_native_vs_original --json --strict-gates
```

Latest local smoke numbers:

- `bench_bdpt_basic.py --samples 64 --grid-size 8 --json`: BDPT LoS
  `0.0007055000023683533s`, MC basic LoS `0.0006850999998277985s`,
  selected accumulation strategy `atomic`.
- `bench_bdpt_munich.py --samples 64 --grid-size 8 --warmup-runs 1
  --repeats 3 --no-artifacts --json`: native median
  `0.0034225999988848343s`, p95 `0.0034225999988848343s`.
- `tests.support.bin.benchmark_munich_bdpt_native_vs_original --samples 16
  --grid-size 4 --max-depth 1 --warmup-runs 1 --original-timeout-seconds 240
  --json`: native solve `1.901800002087839 ms`, original solve
  `68.98979999823496 ms`, native speedup `36.27605422362835x`; LoS correlation
  `1.0000000744796738`, reflection correlation `0.9999999400453701`,
  diffraction correlation `1.0`, and total relative sum error
  `9.127168010444401e-08`. Strict Munich parity gates passed.
- Cold first-use run of the same reduced Munich gate with `--warmup-runs 0`
  remained numerically green, but measured native solve `30778.580799997144 ms`
  versus original solve `6104.1540000005625 ms` because native OptiX/RayDN
  pipeline setup is included in the timed solve.
- `tests.support.bin.benchmark_single_plane_bdpt_native_vs_original --samples
  256 --grid-size 8 --warmup-runs 1 --repeats 3 --min-speedup 1.25
  --strict-gates --json`: native median `1.8860999989556149 ms`, original
  median `2.713899993977975 ms`, native speedup `1.4388950720962492x`,
  relative sum error `2.69366456949439e-07`; strict shape/nonzero/speed and
  relative-sum-error gates passed.

# Benchmark evidence

## Baseline tool (`baseline.py`)

`baseline.py` measures fixed solves on the synthetic cities of `synthetic_city.py` (an N x N grid
of concrete boxes on a ground plane, merged into one mesh, optionally 4-way subdivided and
welded). Every case runs in `--processes` fresh processes (default 4). Each process records the
cold first solve (it includes compilation), the minimum of the timed warm solves, the peak Torch
allocation of one solve, the path count, and the CUPTI counts of one profiled solve: kernels,
stream/device/event synchronizations, H2D/D2H/D2D copy counts and bytes, memsets, and
`cudaMalloc`/`cudaFree` calls. The record keeps the minimum wall time over all processes and
lists every count that differed between processes under `unstable` (it should be empty).

```bash
source scripts/dev_env.sh
python benchmarks/baseline.py list                               # cases and tags
python benchmarks/baseline.py run --output now.json              # every case, 4 processes each
python benchmarks/baseline.py run --cases 'det.lr.*' --tags C-1 --processes 2 --output part.json
python benchmarks/baseline.py run --target installed --output installed.json  # site-packages build
python benchmarks/baseline.py table now.json                     # Markdown table of one file
python benchmarks/baseline.py compare benchmarks/baselines/m0-release-0.6.0.sm86.json now.json
python benchmarks/baseline.py merge a.json b.json --output ab.json  # min over two passes
```

Tags: `report-2.1` (one case per row of the analysis report's table 2.1), `C-1`, `C-6`, `P-1`,
`P-3`, `P-9`, `city` (4x4, 8x8, 32x32 cities and welded subdivisions, one case per solver) and
`budget-small|medium|large` (the cases `tests/budgets` enforces). Three rows of table 2.1 are
measured smaller than in the report because they launch kernels of several hundred milliseconds
to seconds, which time out other tenants of a shared GPU: Deterministic depth-3 reflection and
BDPT depth 3 run on a 16x16 grid (`det.lr.d3.g16`, `bdpt.lr.d3.g16.*`), and BDPT 256x256 depth 3
is not a case. Each record carries `longest_kernel_ms`, the longest single CUDA-runtime kernel.
`--target installed` strips the developer-override variables and `PYTHONPATH` and runs from the
temporary directory, so it measures the installed `witwin-channel`; it works only while that
version accepts the same configuration fields.

On a shared GPU run the whole command under the machine's GPU slot lock (for example
`flock <slot.lock> python benchmarks/baseline.py run ...`); workers run one at a time inside it
and cap themselves at 8 GiB. Records are labelled `"timing": "contended (shared GPU)"` unless
`--exclusive` says the run had the GPU to itself.

Wall times move with GPU contention (a contended run is 5-10x slower and looks stable); check the
`gpu_before` snapshot, compare counts first, and repeat timings back to back. The profiler does not
see OptiX launches made through the driver API, so `kernels` counts CUDA-runtime kernels only.
Recorded baselines live in `baselines/m0-*.json`.

## Plan 13 Phase 12 three-group A/B evidence

Formal Phase 12 evidence contains three independent five-pair AB/BA groups:
ADR-027 enumerated penetration, ADR-027 Monte Carlo penetration, and ADR-030
diffraction. Each process performs one discarded warmup and seven steady
solves. Acceptance uses exactly 100,000 paired bootstrap resamples.

The history is one fixed P/E/M/D/S chain. P→E is the enumerated atomic switch,
E is reused as the Monte Carlo baseline, E→M is the Monte Carlo atomic switch,
D directly follows M as the dormant ADR-030 pin, and D→S is the diffraction
live-switch/delete. Every commit uses one RayD lock, typed integration header,
gate blob, profile-contract blob, and runner blob set. The gate-freeze commit
must be an ancestor of P; it cannot be created from the measured descendants.

Formal config supplies only the six source checkouts and their common witwin2
Python. It cannot supply a native extension. Before measurement the runner
creates five initially absent isolated source/build/install trees, clones the
exact P/E/M/D/S commits, configures Release real-SM120 Ninja builds, builds and
installs them, and binds each worker to that runner-produced installation. It
retains clone/checkout/configure/build/install logs, source archives and tree
IDs, CMake cache and compiler records, installed Python manifests, native
extension bytes, and build fingerprints. E's one artifact is reused by the
enumerated candidate and Monte Carlo baseline.

Windows toolchain setup is explicit: config freezes absolute `cmd.exe` and
`vcvars64.bat` bytes alongside cl/link/nvcc. The runner invokes vcvars through a
controlled command that prints only the fixed INCLUDE/LIB/LIBPATH/PATH and VS,
Windows SDK, and UCRT allowlist. It rejects missing, duplicate, unknown, or NUL
output and retains only that allowlist manifest and its raw hash; no complete
process environment is logged. Separate raw cl/link/nvcc version probes are
retained and cross-checked against each fresh build's CMake cache and generated
CXX/CUDA compiler facts.

The stable profiling vocabulary and route requirements live only in
`benchmarks/phase13_phase12_profile_contract.json`. Nsight replay attributes
CUDA work through API correlation IDs to CUPTI kernel/memcpy activity. CPU NVTX
duration is never accepted as CUDA stage duration. Stage duration spans the
first through last correlated GPU activity, including copies before the first
kernel or after the last kernel. Kernel-active and copy-active time remain
separate; all activity must share a device, kernels must share the caller
stream, and copy streams/bytes are recorded and gated against A/B regression.

After timed workers and Nsight capture, each A/B member runs once more in its
own non-timed diagnostic process. The stable owner callables and availability
rules live in `benchmarks/phase13_phase12_diagnostic_contract.json`.
Enumerated and Monte Carlo diagnostics compare the real old/new high-level
owner arrays bitwise. Diffraction keeps the old compact/atomic target as the A
distribution and captures B's pair-major source-lane `valid`, six float32
components, device `num_paths`, and shared capacity-failure state. Replay
performs the test-only float64 ascending-state serial oracle and records
device-peak, host-peak, host-array, and retained-artifact bytes outside the
timed memory gate.

Each frozen group `correctness` object therefore carries the timed
`candidate_target_sha256`, `candidate_full_result_sha256`, and baseline hash
set plus `diagnostic_candidate_semantic_sha256` and the diagnostic baseline
semantic-hash set. Diffraction additionally freezes
`diagnostic_float64_oracle_limits` with exact max-absolute, max-relative, and
max-ULP fields. Each `resource_budgets` object keeps diagnostic device, host,
and artifact maxima separate from timed allocator maxima.

Infrastructure lands with a non-claiming gate: measured facts remain null and
formal execution fails closed. A later, independent gate-freeze commit may fill
those fields only from the controlled assembly and must then precede P.

Entry points are `tools/phase13_phase12_evidence.py`,
`benchmarks/gates/phase13_phase12.json`,
`benchmarks/phase13_phase12_diagnostic_contract.json`,
`benchmarks/schemas/phase13-phase12-evidence.schema.json`, and
`ci/check_phase13_phase12_evidence.py`.

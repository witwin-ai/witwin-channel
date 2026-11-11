# Behaviour suites

Three suites guard behaviour rather than structure. Each runs alone and as its own gate in the
`cuda` tier (`python ci/run_ci_tier.py cuda`); `cuda.unit-contract-acceptance` skips their
directories so they do not run twice. Load the extension first (`source scripts/dev_env.sh`) and
run from the repository root. On a shared GPU run each suite under the machine's GPU slot lock
(`flock <slot.lock> python -m pytest -q tests/survival`); the suites start one GPU process at a
time and cap each at 8 GiB.

| Suite | Gate | Run alone | What it asserts |
|---|---|---|---|
| `tests/survival` | `cuda.survival` | `python -m pytest -q tests/survival` | every hostile but legal input leaves the process and its CUDA context usable |
| `tests/accuracy` | `cuda.reference-accuracy` | `python -m pytest -q tests/accuracy` | solver results agree with independent float64 references and with themselves |
| `tests/budgets` | `cuda.launch-budgets` | `python -m pytest -q tests/budgets` | kernel launches, host syncs and copies stay under the stored bounds |

## Survival (`tests/survival`)

`cases.py` holds the cases; each is also a command:
`python -m tests.survival.cases --list` and `python -m tests.survival.cases <case>`.
The test runs every case in a fresh process with a timeout and requires exit code 0 and a working
`torch.zeros(1, device="cuda")` afterwards. A case passes when the solve completes or raises a
Python exception; `expect="reject"` cases must raise. A timeout is a hang.

Cases that fail today are strict expected failures in `EXPECTED_FAILURES` in `test_survival.py`,
each with its finding ID from the analysis report (S-1 ... S-8). A fix turns its case into an
unexpected pass, which fails the suite until the entry is deleted: that is the flip.

The cross-device case (S-5) needs two GPUs. It is skipped unless `CHANNEL_SURVIVAL_SECOND_GPU`
names the physical index of a second GPU, which the case then appends to `CUDA_VISIBLE_DEVICES`.
It only builds a two-triangle scene there. The two S-8 Kirchhoff cases reproduce a hang that keeps
the GPU busy for tens of seconds; they are skipped unless `CHANNEL_SURVIVAL_LONG_GPU_WORK=1`, which
only belongs on a GPU nobody else uses.

## Reference accuracy (`tests/accuracy`)

`oracles.py` is float64 NumPy/SciPy only and never imports Channel: Friis with polarization
projection, image-method reflection with half-space Fresnel coefficients (the in-plane sign is
fixed by the perfect-conductor image), the exact Sommerfeld half-plane solution, and the
Kouyoumjian-Pathak UTD wedge coefficients. Slab references use `tests/reference/em_oracle.py`.
`test_oracles.py` cross-checks the oracles on the CPU.

`test_reference.py` compares solver outputs with the oracles; `test_consistency.py` checks
reciprocity, subdivision invariance (C-6), max_depth 3 vs 4 scheme equivalence (C-1) and
four-solver agreement on LoS and a single PEC wall. Every tolerance states its reason in the test
(float32 amplitude chain plus a phase bound that grows with `k L`, UTD's own asymptotic error,
Monte Carlo counting error, summation order). A run prints a table of every measured error;
`CHANNEL_ACCURACY_REPORT=<file.json>` also writes it as JSON. Known failures are strict expected
failures with a finding ID (C-1, C-6, and A-1 ... A-4 found by this suite; see the M0 report).

## Launch budgets (`tests/budgets`)

`launch_budgets.json` stores, per case and metric, the upper bound for one warm solve: kernels,
host syncs (stream, device and event synchronize), H2D/D2H copy counts and bytes. The cases are
the `budget-small`, `budget-medium` and `budget-large` cases of `benchmarks/baseline.py`
(4x4, 8x8 and 32x32 cities, one per solver). Each test measures its case in a fresh process with
the same profiler code as the baseline tool.

Budgets only go down. After a change lowers a count, regenerate the file from a fresh measurement:

```bash
python benchmarks/baseline.py run --tags budget-small budget-medium budget-large --counts-only \
    --processes 2 --output /tmp/budgets-now.json
python benchmarks/baseline.py budgets /tmp/budgets-now.json --write tests/budgets/launch_budgets.json
```

`budgets` refuses to raise any existing bound unless `--allow-raise` is given.

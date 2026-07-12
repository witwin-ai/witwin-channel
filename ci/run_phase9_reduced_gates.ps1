param(
    [string]$Python = "python",
    [string]$ArtifactDirectory = "artifacts/phase9-ci",
    [string]$Wheel = ""
)

$ErrorActionPreference = "Stop"

& $Python -m pytest -q tests/performance
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python benchmarks/bench_solver_peak_memory.py `
    --tx 1 --rx 1024 --depth 3 --gpu-budget-gib 16 `
    --output "$ArtifactDirectory/solver_peak_memory.v1.json"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$cudaAvailable = & $Python -c "import torch; print(int(torch.cuda.is_available()))"
if ($cudaAvailable.Trim() -eq "1") {
    & $Python benchmarks/bench_solver_scaling.py `
        --solvers path,deterministic,basic,bdpt `
        --tx 1 --rx 1 --depths 1 --samples 256 `
        --warmup 0 --repeats 1 `
        --output "$ArtifactDirectory/solver_scaling.v1.json"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if ($Wheel) {
    & $Python ci/wheel_smoke.py $Wheel
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} else {
    Write-Warning "Wheel smoke gate not run: pass -Wheel <built-wheel.whl>. Source-tree import tests are not wheel evidence."
}

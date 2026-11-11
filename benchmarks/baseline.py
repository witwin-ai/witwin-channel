# Copyright Xingyu Chen.
# Record and compare per-solve wall time, launch, sync, copy and memory baselines on fixed scenes.

"""Record and compare per-solve wall time, launch, sync, copy and memory baselines on fixed scenes."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, field
import datetime
import fnmatch
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

BENCHMARKS = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARKS.parent
SCHEMA = "channel-baseline.v1"
RESULT_PREFIX = "BASELINE_RESULT "
SOLVER_MODULES = {
    "det": "witwin.channel.deterministic",
    "path": "witwin.channel.path",
    "mc": "witwin.channel.montecarlo.basic",
    "bdpt": "witwin.channel.montecarlo.bdpt",
}
COMPONENTS = {
    "los": ("los",),
    "lr": ("los", "reflection"),
    "lrd": ("los", "reflection", "diffraction"),
    "diff": ("diffraction",),
    "lt": ("los", "transmission"),
}
# Counts that must be identical between independent processes for one case.
COUNT_KEYS = (
    "kernels", "launch_api", "stream_sync", "device_sync", "event_sync", "syncs",
    "memcpy_api", "h2d_count", "h2d_bytes", "d2h_count", "d2h_bytes", "d2d_count", "d2d_bytes",
    "memset_count", "malloc_count", "free_count",
)
_LAUNCH_API = {
    "cudaLaunchKernel", "cudaLaunchKernelExC", "cudaLaunchCooperativeKernel", "cuLaunchKernel",
    "cuLaunchKernelEx", "cudaGraphLaunch", "cuGraphLaunch",
}
_MEMCPY_API_PREFIXES = ("cudaMemcpy", "cuMemcpy")


@dataclass(frozen=True, slots=True)
class Case:
    """One fixed solve: a city, a receiver set, a solver and its configuration."""

    name: str
    solver: str
    comps: str
    depth: int
    receivers: str  # "g<N>" for an N x N grid, "p<N>" for N point receivers
    city: int = 4
    subdiv: int = 0
    welded: bool = True
    samples: int | None = None
    repeats: int = 5
    tags: tuple[str, ...] = ()
    note: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def config_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"max_depth": self.depth, "components": set(COMPONENTS[self.comps])}
        if self.samples is not None:
            kwargs["samples"] = self.samples
        kwargs.update(self.extra)
        return kwargs


def _cases() -> dict[str, Case]:
    rows: list[Case] = []
    r21 = ("report-2.1",)
    for n in (32, 256, 1024):
        rows.append(Case(f"det.los.d0.g{n}", "det", "los", 0, f"g{n}", tags=r21))
    rows.append(Case("det.lr.d1.g64", "det", "lr", 1, "g64", tags=r21))
    rows.append(Case("det.lr.d1.g512", "det", "lr", 1, "g512", tags=r21))
    for d in (2, 3, 4, 5):
        rows.append(Case(f"det.lr.d{d}.g64", "det", "lr", d, "g64", repeats=3,
                         tags=r21 + (("C-1", "P-3") if d in (3, 4) else ()),
                         note="C-1: d3 exhaustive vs d4 ray discovery" if d in (3, 4) else ""))
    rows.append(Case("det.lr.d1.g128", "det", "lr", 1, "g128", tags=r21 + ("P-9",)))
    rows.append(Case("det.lr.d1.g128.s4", "det", "lr", 1, "g128", subdiv=4, tags=r21 + ("P-9",),
                     note="same geometry subdivided 4 levels (41,472 triangles), welded"))
    rows.append(Case("det.lrd.d1.g128", "det", "lrd", 1, "g128", repeats=3, tags=r21))
    rows.append(Case("det.diff.d1.g128", "det", "diff", 1, "g128", repeats=3, tags=r21))
    for sub in (0, 1, 3):
        rows.append(Case(f"det.diff.d1.g32.s{sub}", "det", "diff", 1, "g32", subdiv=sub,
                         tags=("C-6",), note="C-6 subdivision invariance (welded)"))
    rows.append(Case("det.diff.d1.g32.s1u", "det", "diff", 1, "g32", subdiv=1, welded=False,
                     tags=("C-6",), note="C-6 unwelded subdivision"))
    rows.append(Case("det.los.d0.p1", "det", "los", 0, "p1", tags=r21, note="1 Tx x 1 Rx fixed cost"))
    for n in (1, 16, 256, 1024):
        rows.append(Case(f"path.los.d0.p{n}", "path", "los", 0, f"p{n}",
                         repeats=3 if n >= 256 else 5, tags=r21 + ("P-1",)))
    for n in (1, 1024):
        rows.append(Case(f"path.lrd.d1.p{n}", "path", "lrd", 1, f"p{n}",
                         repeats=3 if n >= 256 else 5, tags=r21 + ("P-1",)))
    rows.append(Case("path.lrd.d2.p16", "path", "lrd", 2, "p16", tags=r21,
                     note="report: 872 launches, 194 syncs, 41 D2H"))
    for s, label in ((4096, "4k"), (1 << 20, "1M"), (4194304, "4M")):
        rows.append(Case(f"mc.lr.d1.g256.s{label}", "mc", "lr", 1, "g256", samples=s, tags=r21))
    for n, d, s, label, rep in ((64, 1, 4096, "4k", 5), (64, 1, 1 << 20, "1M", 5),
                                (64, 3, 4096, "4k", 3), (64, 3, 1 << 20, "1M", 3),
                                (256, 3, 4096, "4k", 2), (256, 3, 1 << 20, "1M", 2)):
        rows.append(Case(f"bdpt.lr.d{d}.g{n}.s{label}", "bdpt", "lr", d, f"g{n}", samples=s,
                         repeats=rep, tags=r21 + (("slow",) if n == 256 else ())))
    # Small / medium / large cities and their welded 4-way subdivisions, one case per solver.
    for n_side, size in ((4, "small"), (8, "medium"), (32, "large")):
        for sub in (0, 1):
            suffix = f"c{n_side}" + (f"s{sub}" if sub else "")
            tags = ("city", f"budget-{size}") if sub == 0 else ("city",)
            rep = 3 if n_side == 32 else 5
            rows.append(Case(f"det.lrd.d1.g64.{suffix}", "det", "lrd", 1, "g64", city=n_side,
                             subdiv=sub, repeats=rep, tags=tags))
            rows.append(Case(f"path.lrd.d1.p16.{suffix}", "path", "lrd", 1, "p16", city=n_side,
                             subdiv=sub, repeats=rep, tags=tags))
            rows.append(Case(f"mc.lr.d1.g64.s64k.{suffix}", "mc", "lr", 1, "g64", city=n_side,
                             subdiv=sub, samples=65536, repeats=rep, tags=tags))
            rows.append(Case(f"bdpt.lr.d1.g64.s4k.{suffix}", "bdpt", "lr", 1, "g64",
                             city=n_side, subdiv=sub, samples=4096, repeats=rep, tags=tags))
    return {case.name: case for case in rows}


CASES = _cases()


# --------------------------------------------------------------------------- worker side


def _receivers(spec: str, extent: float) -> list[Any]:
    import synthetic_city as sc

    n = int(spec[1:])
    if spec[0] == "g":
        return [sc.receiver_grid(extent, n)]
    return list(sc.receiver_points(extent, n))


def build(case: Case) -> tuple[Any, Any, Any]:
    """Return ``(solver module, scene, config)`` for ``case``."""
    import synthetic_city as sc
    from witwin.core import Scene

    city = sc.city(case.city, subdiv=case.subdiv, welded=case.welded)
    scene = Scene(structures=city.structures,
                  endpoints=[sc.transmitter(city.extent_m), *_receivers(case.receivers, city.extent_m)])
    module = importlib.import_module(SOLVER_MODULES[case.solver])
    return module, scene, module.Config(**case.config_kwargs())


def path_count(solver: str, result: Any) -> int | None:
    metadata = getattr(result, "metadata", None)
    if not isinstance(metadata, dict):
        return None
    if solver == "det":
        value = metadata.get("counts", {}).get("path_count")
    elif solver == "bdpt":
        value = metadata.get("valid_contribution_count")
    else:
        value = metadata.get("path_count")
    return None if value is None else int(value)


def _count_trace(trace_path: str) -> dict[str, int]:
    with open(trace_path, encoding="utf-8") as handle:
        events = json.load(handle).get("traceEvents", [])
    counts: Counter[str] = Counter()
    for event in events:
        category = event.get("cat")
        name = str(event.get("name", ""))
        if category == "kernel":
            counts["kernels"] += 1
        elif category in ("cuda_runtime", "cuda_driver"):
            if name in _LAUNCH_API:
                counts["launch_api"] += 1
            elif name in ("cudaStreamSynchronize", "cuStreamSynchronize"):
                counts["stream_sync"] += 1
            elif name in ("cudaDeviceSynchronize", "cuCtxSynchronize"):
                counts["device_sync"] += 1
            elif name in ("cudaEventSynchronize", "cuEventSynchronize"):
                counts["event_sync"] += 1
            elif name.startswith(_MEMCPY_API_PREFIXES):
                counts["memcpy_api"] += 1
            elif name in ("cudaMalloc", "cudaMallocAsync", "cuMemAlloc_v2"):
                counts["malloc_count"] += 1
            elif name in ("cudaFree", "cudaFreeAsync", "cuMemFree_v2"):
                counts["free_count"] += 1
        elif category == "gpu_memcpy":
            size = int(event.get("args", {}).get("bytes", 0))
            kind = "h2d" if "HtoD" in name else "d2h" if "DtoH" in name else "d2d"
            counts[f"{kind}_count"] += 1
            counts[f"{kind}_bytes"] += size
        elif category == "gpu_memset":
            counts["memset_count"] += 1
    counts["syncs"] = counts["stream_sync"] + counts["device_sync"] + counts["event_sync"]
    return {key: int(counts[key]) for key in COUNT_KEYS}


def profile_counts(fn: Any) -> dict[str, int]:
    """Run ``fn`` once under the CUDA profiler and return launch, sync and copy counts.

    Counts come from CUPTI runtime-API and activity records, so they include native-extension
    and RayD work issued through the CUDA runtime. OptiX launches issued through the driver API
    are not recorded by the Torch profiler and are therefore not in ``kernels``.
    """

    import torch
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    with tempfile.TemporaryDirectory() as tmp:
        trace = os.path.join(tmp, "trace.json")
        prof.export_chrome_trace(trace)
        counts = _count_trace(trace)
    # The two synchronize calls bracketing the profiled region are the harness's own.
    counts["device_sync"] = max(0, counts["device_sync"] - 1)
    counts["syncs"] = counts["stream_sync"] + counts["device_sync"] + counts["event_sync"]
    return counts


def measure(case: Case, *, repeats: int | None, counts_only: bool) -> dict[str, Any]:
    """Measure one case in this process; the first solve includes scene compilation."""
    import torch

    module, scene, config = build(case)

    def solve() -> Any:
        return module.solve(scene, config, reference_frequency_hz=3.5e9)

    out: dict[str, Any] = {}
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = solve()
    torch.cuda.synchronize()
    out["cold_ms"] = (time.perf_counter() - start) * 1e3
    result = solve()
    torch.cuda.synchronize()
    times = []
    for _ in range(0 if counts_only else (repeats or case.repeats)):
        start = time.perf_counter()
        result = solve()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1e3)
    out["wall_ms"] = times
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    result = solve()
    torch.cuda.synchronize()
    out["peak_alloc_mib"] = (torch.cuda.max_memory_allocated() - base) / 2**20
    out["path_count"] = path_count(case.solver, result)
    kernel_meta = result.metadata.get("kernel", {}) if isinstance(result.metadata, dict) else {}
    out["meta_launch_count"] = kernel_meta.get("launch_count") if isinstance(kernel_meta, dict) else None
    del result
    out["counts"] = profile_counts(solve)
    return out


def _worker(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(BENCHMARKS))
    if args.target == "worktree":
        sys.path.insert(0, str(REPO_ROOT))
    import torch
    import witwin.channel as channel

    case = CASES[args.case]
    info = {"channel_file": channel.__file__, "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0)}
    try:
        info.update(measure(case, repeats=args.repeats, counts_only=args.counts_only))
    except Exception as error:  # noqa: BLE001 - the error is part of the record
        info["error"] = f"{type(error).__name__}: {str(error).splitlines()[0][:300] if str(error) else ''}"
    try:
        torch.zeros(1, device="cuda").sum().item()
        info["context_ok"] = True
    except Exception:  # noqa: BLE001
        info["context_ok"] = False
    print(RESULT_PREFIX + json.dumps(info), flush=True)
    return 0


# --------------------------------------------------------------------------- driver side


def _worker_env(target: str) -> tuple[dict[str, str], str]:
    env = dict(os.environ)
    env.setdefault("CUDA_VISIBLE_DEVICES", "1")
    if target == "installed":
        for key in list(env):
            if key.startswith("WITWIN_CHANNEL_") or key == "PYTHONPATH":
                env.pop(key)
        return env, tempfile.gettempdir()
    return env, str(REPO_ROOT)


def run_worker(case: Case, *, target: str = "worktree", repeats: int | None = None,
               counts_only: bool = False, timeout_s: float = 900.0) -> dict[str, Any]:
    """Run one case in a fresh process and return its record (``error`` on crash or timeout)."""
    env, cwd = _worker_env(target)
    command = [sys.executable, str(Path(__file__).resolve()), "worker", case.name, "--target", target]
    if repeats is not None:
        command += ["--repeats", str(repeats)]
    if counts_only:
        command.append("--counts-only")
    try:
        completed = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True,
                                   timeout=timeout_s, check=False)
    except subprocess.TimeoutExpired:
        return {"error": f"timeout after {timeout_s:.0f} s", "context_ok": None}
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            record: dict[str, Any] = json.loads(line[len(RESULT_PREFIX):])
            if completed.returncode:
                record["exit_code"] = completed.returncode
            return record
    tail = (completed.stderr or completed.stdout).strip().splitlines()[-3:]
    return {"error": f"exit {completed.returncode}: {' | '.join(tail)[:400]}", "context_ok": None}


def _aggregate(case: Case, records: list[dict[str, Any]]) -> dict[str, Any]:
    good = [r for r in records if "error" not in r]
    out: dict[str, Any] = {"spec": {k: v for k, v in asdict(case).items() if k != "name"},
                           "processes": len(records)}
    if not good:
        out["error"] = records[0].get("error") if records else "no runs"
        out["context_ok"] = records[0].get("context_ok") if records else None
        return out
    per_process = [min(r["wall_ms"]) for r in good if r["wall_ms"]]
    out["wall_ms_min"] = round(min(per_process), 3) if per_process else None
    out["wall_ms_per_process_min"] = [round(x, 3) for x in per_process]
    out["cold_ms_min"] = round(min(r["cold_ms"] for r in good), 1)
    out["peak_alloc_mib"] = round(max(r["peak_alloc_mib"] for r in good), 1)
    out["path_count"] = good[0]["path_count"]
    out["meta_launch_count"] = good[0]["meta_launch_count"]
    out["counts"] = good[0]["counts"]
    unstable = sorted({key for r in good[1:] for key in COUNT_KEYS
                       if r["counts"].get(key) != good[0]["counts"].get(key)})
    if any(r["path_count"] != good[0]["path_count"] for r in good):
        unstable.append("path_count")
    out["unstable"] = unstable
    if len(good) != len(records):
        out["partial_errors"] = [r["error"] for r in records if "error" in r]
    return out


def _gpu_snapshot() -> str:
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=False).stdout.strip().replace("\n", "; ")
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"


def select(patterns: list[str] | None, tags: list[str] | None, skip_tags: list[str]) -> list[Case]:
    chosen = []
    for case in CASES.values():
        if patterns and not any(fnmatch.fnmatch(case.name, p) for p in patterns):
            continue
        if tags and not set(tags) & set(case.tags):
            continue
        if set(skip_tags) & set(case.tags):
            continue
        chosen.append(case)
    return chosen


def _git_head() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short=12", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _run(args: argparse.Namespace) -> int:
    cases = select(args.cases, args.tags, args.skip_tags)
    if not cases:
        print("no case matches", file=sys.stderr)
        return 2
    document: dict[str, Any] = {"schema": SCHEMA, "meta": {
        "target": args.target, "processes": args.processes,
        "repeats_override": args.repeats, "counts_only": args.counts_only,
        "git_head": _git_head() if args.target == "worktree" else None,
        "started": datetime.datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "cuda_visible_devices": _worker_env(args.target)[0].get("CUDA_VISIBLE_DEVICES"),
    }, "cases": {}}
    if args.target == "installed":
        try:
            from importlib.metadata import version

            document["meta"]["installed_version"] = version("witwin-channel")
        except Exception:  # noqa: BLE001
            document["meta"]["installed_version"] = None
    for case in cases:
        snapshot = _gpu_snapshot()
        records = [run_worker(case, target=args.target, repeats=args.repeats,
                              counts_only=args.counts_only, timeout_s=args.timeout)
                   for _ in range(args.processes)]
        entry = _aggregate(case, records)
        entry["gpu_before"] = snapshot
        good = [r for r in records if "error" not in r]
        if good and "channel_file" not in document["meta"]:
            document["meta"].update(channel_file=good[0]["channel_file"], torch=good[0]["torch"],
                                    gpu=good[0]["gpu"])
        document["cases"][case.name] = entry
        counts = entry.get("counts", {})
        print(f"{case.name:32s} " + (f"ERROR {entry['error']}" if "error" in entry else
              f"wall {entry['wall_ms_min']} ms  kern {counts.get('kernels')}  sync {counts.get('syncs')}  "
              f"h2d {counts.get('h2d_count')}  d2h {counts.get('d2h_count')}  "
              f"peak {entry['peak_alloc_mib']} MiB  paths {entry['path_count']}"
              + (f"  UNSTABLE {entry['unstable']}" if entry["unstable"] else "")), flush=True)
        if args.output:
            _write(args.output, document)
    document["meta"]["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
    if args.output:
        _write(args.output, document)
    return 0


def _write(path: str, document: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(document, indent=1, sort_keys=False) + "\n", encoding="utf-8")


COMPARE_METRICS = (
    ("wall_ms_min", "wall ms"), ("kernels", "kernels"), ("syncs", "syncs"),
    ("h2d_count", "H2D"), ("h2d_bytes", "H2D B"), ("d2h_count", "D2H"), ("d2h_bytes", "D2H B"),
    ("peak_alloc_mib", "peak MiB"), ("path_count", "paths"),
)


def _metric(entry: dict[str, Any], key: str) -> Any:
    if key in entry.get("counts", {}):
        return entry["counts"][key]
    return entry.get(key)


def _cell(old: Any, new: Any) -> str:
    if old is None or new is None:
        return f"{old} -> {new}"
    if old == new:
        return f"{_fmt(new)} (=)"
    ratio = f"x{new / old:.2f}" if old else "new"
    return f"{_fmt(old)} -> {_fmt(new)} ({ratio})"


def _fmt(value: Any) -> str:
    return f"{value:.3g}" if isinstance(value, float) else str(value)


def compare(old: dict[str, Any], new: dict[str, Any]) -> str:
    """Return a Markdown table of new/old ratios for every case present in both documents."""
    header = ["case", *(label for _, label in COMPARE_METRICS)]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for name, new_entry in new["cases"].items():
        old_entry = old["cases"].get(name)
        if old_entry is None:
            continue
        if "error" in old_entry or "error" in new_entry:
            lines.append(f"| {name} | old: {old_entry.get('error', 'ok')} | new: {new_entry.get('error', 'ok')} |")
            continue
        cells = [_cell(_metric(old_entry, k), _metric(new_entry, k)) for k, _ in COMPARE_METRICS]
        lines.append("| " + " | ".join([name, *cells]) + " |")
    only_old = sorted(set(old["cases"]) - set(new["cases"]))
    only_new = sorted(set(new["cases"]) - set(old["cases"]))
    if only_old:
        lines.append(f"\nonly in old: {', '.join(only_old)}")
    if only_new:
        lines.append(f"\nonly in new: {', '.join(only_new)}")
    return "\n".join(lines)


def table(document: dict[str, Any]) -> str:
    """Return a Markdown table of one document's numbers."""
    header = ["case", "wall ms", "cold ms", "kernels", "syncs", "H2D", "H2D B", "D2H", "D2H B",
              "peak MiB", "paths"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for name, e in document["cases"].items():
        if "error" in e:
            lines.append(f"| {name} | ERROR: {e['error'][:120]} | context_ok={e.get('context_ok')} |")
            continue
        c = e["counts"]
        cells = [e["wall_ms_min"], e["cold_ms_min"], c["kernels"], c["syncs"], c["h2d_count"],
                 c["h2d_bytes"], c["d2h_count"], c["d2h_bytes"], e["peak_alloc_mib"], e["path_count"]]
        lines.append("| " + " | ".join([name, *map(_fmt, cells)]) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("list", help="list the fixed cases")
    listing.add_argument("--tags", nargs="*")
    run = sub.add_parser("run", help="measure cases, each in --processes fresh processes")
    run.add_argument("--cases", nargs="*", help="fnmatch patterns over case names")
    run.add_argument("--tags", nargs="*", help="keep cases carrying any of these tags")
    run.add_argument("--skip-tags", nargs="*", default=[])
    run.add_argument("--processes", type=int, default=4)
    run.add_argument("--repeats", type=int, help="override the per-case timed repeats")
    run.add_argument("--counts-only", action="store_true", help="skip the timed repeats")
    run.add_argument("--target", choices=("worktree", "installed"), default="worktree",
                     help="worktree: this checkout plus the developer override; installed: the "
                     "site-packages witwin-channel, run from outside any checkout")
    run.add_argument("--timeout", type=float, default=900.0, help="seconds per process")
    run.add_argument("--output", help="JSON file to write (rewritten after every case)")
    cmp_ = sub.add_parser("compare", help="Markdown table of new/old ratios")
    cmp_.add_argument("old")
    cmp_.add_argument("new")
    show = sub.add_parser("table", help="Markdown table of one result file")
    show.add_argument("result")
    worker = sub.add_parser("worker", help=argparse.SUPPRESS)
    worker.add_argument("case", choices=sorted(CASES))
    worker.add_argument("--target", choices=("worktree", "installed"), default="worktree")
    worker.add_argument("--repeats", type=int)
    worker.add_argument("--counts-only", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "list":
        for case in select(None, args.tags, []):
            print(f"{case.name:32s} {','.join(case.tags):28s} {case.note}")
        return 0
    if args.command == "run":
        return _run(args)
    if args.command == "compare":
        print(compare(json.loads(Path(args.old).read_text()), json.loads(Path(args.new).read_text())))
        return 0
    if args.command == "table":
        print(table(json.loads(Path(args.result).read_text())))
        return 0
    return _worker(args)


if __name__ == "__main__":
    raise SystemExit(main())

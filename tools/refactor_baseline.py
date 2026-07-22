"""Freeze deterministic, immutable manifests for architecture refactors.

The collector is deliberately static-first: it parses Python and C++ sources
without importing ``witwin.channel`` or loading its native extension.
Solver outputs, launch ledgers, and performance measurements are produced by
their existing harnesses and can be attached with ``--runtime-artifact``.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path


SCHEMA_VERSION = 1
PACKAGE = "witwin.channel"
DEFAULT_PUBLIC_MODULES = (
    PACKAGE,
    f"{PACKAGE}.materials",
    f"{PACKAGE}.capabilities",
    f"{PACKAGE}.path",
    f"{PACKAGE}.deterministic",
    f"{PACKAGE}.montecarlo.basic",
    f"{PACKAGE}.montecarlo.bdpt",
)
REQUIRED_RUNTIME_KINDS = ("solver-results", "launch-ledger", "performance")
_SOURCE_SUFFIXES = frozenset({".cpp", ".cu", ".cuh", ".h", ".hpp"})
_SENSITIVE_KEY = re.compile(
    r"(?:^|_)(?:api_?key|authorization|credential|password|private_?key|secret|token)(?:$|_)",
    re.IGNORECASE,
)
_SAFE_ENVIRONMENT_NAMES = (
    "CONDA_PREFIX",
    "CUDA_PATH",
    "CUDA_VISIBLE_DEVICES",
    "PYTHONHASHSEED",
    "RAYD_SOURCE_DIR",
    "TORCH_CUDA_ARCH_LIST",
)
_CMAKE_CACHE_KEYS = re.compile(
    r"^(?:"
    r"CMAKE_BUILD_TYPE|CMAKE_CACHE_(?:MAJOR|MINOR|PATCH)_VERSION|CMAKE_COMMAND|"
    r"CMAKE_CUDA_ARCHITECTURES|CMAKE_GENERATOR(?:_PLATFORM|_TOOLSET)?|"
    r"CMAKE_(?:CXX|CUDA)_FLAGS(?:_(?:DEBUG|MINSIZEREL|RELEASE|RELWITHDEBINFO))?|"
    r"CMAKE_MSVC_RUNTIME_LIBRARY|RAYD_TORCH_BUILD_(?:NATIVE|PYTHON_MODULE)"
    r")$"
)
_ABSOLUTE_USER_PATH = re.compile(
    r"(?:[A-Za-z]:[\\/](?:[^\s;\"]+)|/(?:home|Users)/[^\s;\"]+)",
    re.IGNORECASE,
)


class BaselineError(RuntimeError):
    """Raised when a baseline cannot be frozen without ambiguity."""


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(_canonical_bytes(value))


def _run(
    command: Sequence[str],
    *,
    cwd: Path,
    timeout: int = 15,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=env,
    )


def _git(repo: Path, *args: str, allow_failure: bool = False) -> str | None:
    result = _run(("git", *args), cwd=repo)
    if result.returncode:
        if allow_failure:
            return None
        detail = (result.stderr or result.stdout).strip()
        raise BaselineError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout.strip()


def _git_state(repo: Path) -> dict[str, object]:
    sha = _git(repo, "rev-parse", "HEAD")
    assert sha is not None
    status_output = _git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    assert status_output is not None
    status = []
    private_claude_present = False
    for line in status_output.splitlines() if status_output else []:
        normalized = line.replace("\\", "/")
        if normalized == "?? .claude/" or normalized.startswith("?? .claude/"):
            if not private_claude_present:
                status.append("?? .claude/")
                private_claude_present = True
            continue
        status.append(line)
    return {
        "sha": sha,
        "dirty": bool(status),
        "worktree_status": status,
        "private_untracked_roots": [".claude/"] if private_claude_present else [],
    }


def _rayd_state(repo: Path, rayd_root: Path | None) -> dict[str, object]:
    candidate = rayd_root if rayd_root is not None else repo.parents[1] / "RayDi"
    if not (candidate / ".git").exists():
        return {"available": False, "dirty": None, "sha": None}
    sha = _git(candidate, "rev-parse", "HEAD", allow_failure=True)
    status = _git(
        candidate,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        allow_failure=True,
    )
    return {
        "available": sha is not None,
        "dirty": None if status is None else bool(status),
        "sha": sha,
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _first_matching_line(output: str, patterns: Sequence[str]) -> str | None:
    for line in output.splitlines():
        stripped = line.strip()
        if stripped and any(re.search(pattern, stripped) for pattern in patterns):
            return stripped
    return None


def _command_versions(repo: Path) -> dict[str, object]:
    versions: dict[str, object] = {}
    commands = {
        "cmake": (("cmake", "--version"), (r"^cmake version ",)),
        "nvcc": (("nvcc", "--version"), (r"release ", r"^Cuda compilation")),
        "msvc": (("cl", "/Bv"), (r"Compiler Version", r"Version .* for ")),
    }
    for name, (command, patterns) in commands.items():
        if shutil.which(command[0]) is None:
            versions[name] = None
            continue
        result = _run(command, cwd=repo)
        output = f"{result.stdout}\n{result.stderr}"
        versions[name] = _first_matching_line(output, patterns)

    if shutil.which("nvidia-smi") is None:
        versions["gpus"] = []
    else:
        result = _run(
            (
                "nvidia-smi",
                "--query-gpu=name,driver_version,compute_cap",
                "--format=csv,noheader,nounits",
            ),
            cwd=repo,
        )
        versions["gpus"] = (
            [line.strip() for line in result.stdout.splitlines() if line.strip()]
            if result.returncode == 0
            else []
        )
    return versions


def _torch_runtime_manifest(repo: Path) -> dict[str, object]:
    script = """
import json
import sys
import torch

cuda_available = bool(torch.cuda.is_available())
devices = []
if cuda_available:
    for index in range(torch.cuda.device_count()):
        capability = torch.cuda.get_device_capability(index)
        devices.append({
            "index": index,
            "capability": list(capability),
            "sm": capability[0] * 10 + capability[1],
        })
print(json.dumps({
    "torch_version": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "cuda_available": cuda_available,
    "devices": devices,
}, sort_keys=True))
"""
    result = _run((sys.executable, "-c", script), cwd=repo, timeout=30)
    if result.returncode:
        return {
            "status": "query-failed",
            "torch_version": _package_version("torch"),
            "cuda_runtime": None,
            "cuda_available": None,
            "devices": [],
        }
    for line in reversed(result.stdout.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "torch_version" in payload:
            return {"status": "ok", **payload}
    return {
        "status": "invalid-output",
        "torch_version": _package_version("torch"),
        "cuda_runtime": None,
        "cuda_available": None,
        "devices": [],
    }


def environment_manifest(
    repo: Path, rayd_root: Path | None = None
) -> dict[str, object]:
    """Collect version metadata without reading any environment variable value."""

    return {
        "schema_version": SCHEMA_VERSION,
        "git": _git_state(repo),
        "rayd": _rayd_state(repo, rayd_root),
        "platform": {
            "machine": platform.machine(),
            "release": platform.release(),
            "system": platform.system(),
        },
        "python": {
            "compiler": platform.python_compiler(),
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "packages": {
            name: _package_version(name)
            for name in ("cmake", "numpy", "pytest", "scikit-build-core", "torch")
        },
        "tools": _command_versions(repo),
        "torch_runtime": _torch_runtime_manifest(repo),
        "environment_variables_present": {
            name: name in os.environ for name in _SAFE_ENVIRONMENT_NAMES
        },
        "privacy": {
            "environment_values_collected": False,
            "gpu_identifiers_collected": False,
            "user_paths_collected": False,
        },
    }


def _source_files(root: Path, suffixes: frozenset[str]) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )


def _cmake_source_contract(source: str) -> dict[str, object]:
    def setting(name: str) -> str | None:
        match = re.search(
            rf"set\(\s*{re.escape(name)}\s+(?:\"([^\"]*)\"|([^\s)]+))",
            source,
            re.IGNORECASE,
        )
        return (match.group(1) or match.group(2)) if match else None

    definitions_match = re.search(
        r"target_compile_definitions\(\s*_channel\s+PRIVATE(?P<body>.*?)\)",
        source,
        re.DOTALL,
    )
    definitions = definitions_match.group("body").split() if definitions_match else []
    source_options = []
    for match in re.finditer(
        r"set_source_files_properties\(\s*(?P<path>\S+)\s+PROPERTIES\s+"
        r"COMPILE_OPTIONS\s+\"(?P<options>[^\"]*)\"\s*\)",
        source,
        re.DOTALL,
    ):
        source_options.append(
            {
                "path": match.group("path").replace("\\", "/"),
                "options": match.group("options").split(";"),
            }
        )
    return {
        "cxx_standard": setting("CMAKE_CXX_STANDARD"),
        "default_build_type": setting("CMAKE_BUILD_TYPE"),
        "default_cuda_architectures": setting(
            "CHANNEL_DEFAULT_CUDA_ARCHITECTURES"
        ),
        "default_torch_cuda_arch_list": setting(
            "CHANNEL_DEFAULT_TORCH_CUDA_ARCH_LIST"
        ),
        "target_compile_definitions": definitions,
        "per_source_compile_options": source_options,
    }


def _cmake_cache_contract(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    path = path.resolve()
    if not path.is_file():
        raise BaselineError(f"CMake cache does not exist: {path}")
    values = []
    for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        match = re.match(r"([^#/:][^:]*)\s*:[^=]+=\s*(.*)$", line)
        if match is None or not _CMAKE_CACHE_KEYS.fullmatch(match.group(1)):
            continue
        key, raw_value = match.group(1), match.group(2)
        values.append(
            {
                "key": key,
                "value": _ABSOLUTE_USER_PATH.sub("<absolute-path>", raw_value),
                "raw_sha256": _sha256_bytes(raw_value.encode("utf-8")),
            }
        )
    by_key = {str(value["key"]): str(value["value"]) for value in values}
    cmake_version_parts = [
        by_key.get(f"CMAKE_CACHE_{part}_VERSION")
        for part in ("MAJOR", "MINOR", "PATCH")
    ]
    return {
        "sha256": _sha256_file(path),
        "values": sorted(values, key=lambda item: str(item["key"])),
        "cmake_version": (
            ".".join(str(part) for part in cmake_version_parts)
            if all(part is not None for part in cmake_version_parts)
            else None
        ),
    }


_COMPILER_VALUE_KEYS = frozenset(
    {
        "CMAKE_CXX_COMPILER_ID",
        "CMAKE_CXX_COMPILER_VERSION",
        "CMAKE_CXX_COMPILER_FRONTEND_VARIANT",
        "CMAKE_CXX_SIMULATE_ID",
        "CMAKE_CXX_SIMULATE_VERSION",
        "MSVC_CXX_ARCHITECTURE_ID",
        "CMAKE_CUDA_COMPILER_ID",
        "CMAKE_CUDA_COMPILER_VERSION",
        "CMAKE_CUDA_COMPILER_FRONTEND_VARIANT",
        "CMAKE_CUDA_SIMULATE_ID",
        "CMAKE_CUDA_SIMULATE_VERSION",
        "MSVC_CUDA_ARCHITECTURE_ID",
    }
)


def _cmake_toolchain_contract(cache_path: Path | None) -> dict[str, object] | None:
    if cache_path is None:
        return None
    cache_path = cache_path.resolve()
    cmake_files = cache_path.parent / "CMakeFiles"
    compiler_files = sorted(
        path
        for name in ("CMakeCXXCompiler.cmake", "CMakeCUDACompiler.cmake")
        for path in cmake_files.glob(f"*/{name}")
    )
    entries = []
    setting_pattern = re.compile(
        r"set\(\s*([A-Za-z0-9_]+)\s+(?:\"([^\"]*)\"|([^\s)]+))"
    )
    for path in compiler_files:
        source = path.read_text(encoding="utf-8-sig", errors="replace")
        values = {}
        compiler_path = None
        for match in setting_pattern.finditer(source):
            key = match.group(1)
            value = match.group(2) or match.group(3)
            if key in _COMPILER_VALUE_KEYS:
                values[key] = value
            elif key in {"CMAKE_CXX_COMPILER", "CMAKE_CUDA_COMPILER"}:
                compiler_path = {
                    "basename": Path(value).name,
                    "value": _ABSOLUTE_USER_PATH.sub("<absolute-path>", value),
                    "raw_sha256": _sha256_bytes(value.encode("utf-8")),
                }
        entries.append(
            {
                "kind": "cuda" if "CUDA" in path.name else "cxx",
                "source": path.relative_to(cache_path.parent).as_posix(),
                "source_sha256": _sha256_file(path),
                "compiler": compiler_path,
                "values": dict(sorted(values.items())),
            }
        )
    return {"compilers": entries}


def build_manifest(
    repo: Path,
    native_binaries: Sequence[Path] = (),
    cmake_cache: Path | None = None,
) -> dict[str, object]:
    native_root = repo / "native" / "channel"
    sources = [repo / "CMakeLists.txt", repo / "pyproject.toml"]
    sources.extend(_source_files(native_root, _SOURCE_SUFFIXES))
    source_entries = [
        {
            "path": path.relative_to(repo).as_posix(),
            "sha256": _sha256_file(path),
            "size": path.stat().st_size,
        }
        for path in sorted(set(sources))
        if path.is_file()
    ]
    discovered = sorted(
        path
        for suffix in ("*.pyd", "*.so", "*.dll")
        for path in (repo / "src" / "witwin" / "channel").glob(
            f"_channel{suffix}"
        )
    )
    binaries = sorted({path.resolve() for path in (*native_binaries, *discovered)})
    binary_entries = []
    for path in binaries:
        if not path.is_file():
            raise BaselineError(f"native binary does not exist: {path}")
        try:
            label = path.relative_to(repo.resolve()).as_posix()
        except ValueError:
            label = path.name
        binary_entries.append(
            {"path": label, "sha256": _sha256_file(path), "size": path.stat().st_size}
        )
    cmake_source = (repo / "CMakeLists.txt").read_text(encoding="utf-8-sig")
    cache_contract = _cmake_cache_contract(cmake_cache)
    toolchain_contract = _cmake_toolchain_contract(cmake_cache)
    return {
        "schema_version": SCHEMA_VERSION,
        "source_fingerprint": _sha256_bytes(_canonical_bytes(source_entries)),
        "sources": source_entries,
        "native_binaries": binary_entries,
        "native_binary_fingerprint": (
            _sha256_bytes(_canonical_bytes(binary_entries)) if binary_entries else None
        ),
        "cmake_source_contract": _cmake_source_contract(cmake_source),
        "cmake_cache_contract": cache_contract,
        "cmake_toolchain_contract": toolchain_contract,
        "missing_build_inputs": sorted(
            name
            for name, missing in (
                ("cmake-cache", cache_contract is None),
                ("native-binary", not binary_entries),
            )
            if missing
        ),
    }


def _python_modules(repo: Path) -> dict[str, Path]:
    package_root = repo / "src" / "witwin" / "channel"
    modules: dict[str, Path] = {}
    for path in sorted(package_root.rglob("*.py")):
        relative = path.relative_to(repo / "src").with_suffix("")
        parts = list(relative.parts)
        if parts[-1] == "__init__":
            parts.pop()
        modules[".".join(parts)] = path
    return modules


def _read_python(path: Path) -> tuple[str, ast.Module]:
    source = path.read_text(encoding="utf-8-sig")
    return source, ast.parse(source, filename=str(path))


def _string_sequence(node: ast.AST) -> list[str] | None:
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError):
        return None
    if isinstance(value, (list, tuple)) and all(
        isinstance(item, str) for item in value
    ):
        return list(value)
    return None


def _module_all(tree: ast.Module) -> list[str]:
    exports: list[str] | None = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            exports = _string_sequence(node.value)
        elif (
            isinstance(node, ast.AugAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "__all__"
            and isinstance(node.op, ast.Add)
        ):
            extra = _string_sequence(node.value)
            if exports is not None and extra is not None:
                exports.extend(extra)
    if exports is None:
        raise BaselineError("public module has no statically resolvable __all__")
    return exports


def _resolve_import_from(module: str, is_package: bool, node: ast.ImportFrom) -> str:
    if node.level == 0:
        return node.module or ""
    package = module if is_package else module.rpartition(".")[0]
    parts = package.split(".") if package else []
    remove = node.level - 1
    if remove > len(parts):
        raise BaselineError(
            f"relative import escapes package in {module}:{node.lineno}"
        )
    base = parts[: len(parts) - remove]
    if node.module:
        base.extend(node.module.split("."))
    return ".".join(base)


def _module_index(
    module: str, path: Path, tree: ast.Module
) -> tuple[dict[str, ast.AST], dict[str, tuple[str, str | None]]]:
    definitions: dict[str, ast.AST] = {}
    imports: dict[str, tuple[str, str | None]] = {}
    is_package = path.name == "__init__.py"

    def index_import(node: ast.ImportFrom | ast.Import) -> None:
        if isinstance(node, ast.ImportFrom):
            target_module = _resolve_import_from(module, is_package, node)
            for alias in node.names:
                local = alias.asname or alias.name
                imports[local] = (target_module, alias.name)
        else:
            for alias in node.names:
                local = alias.asname or alias.name.split(".")[0]
                imports[local] = (alias.name, None)

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            definitions[node.name] = node
        elif isinstance(node, (ast.ImportFrom, ast.Import)):
            index_import(node)
        elif (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "TYPE_CHECKING"
        ):
            for child in node.body:
                if isinstance(child, (ast.ImportFrom, ast.Import)):
                    index_import(child)
    return definitions, imports


def _class_module_overrides(tree: ast.Module) -> dict[str, str]:
    """Return literal ``Class.__module__`` compatibility assignments."""
    defined_classes: set[str] = set()
    overrides: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            defined_classes.add(node.name)
            overrides.pop(node.name, None)
            continue
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Attribute)
            and node.targets[0].attr == "__module__"
            and isinstance(node.targets[0].value, ast.Name)
            and node.targets[0].value.id in defined_classes
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue
        overrides[node.targets[0].value.id] = node.value.value
    return overrides


def _decorator_names(
    node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[str]:
    return [ast.unparse(decorator) for decorator in node.decorator_list]


def _dataclass_fields(node: ast.ClassDef) -> list[dict[str, object]]:
    fields = []
    for child in node.body:
        if not isinstance(child, ast.AnnAssign) or not isinstance(
            child.target, ast.Name
        ):
            continue
        annotation = ast.unparse(child.annotation)
        if "ClassVar" in annotation:
            continue
        default_kind = "missing"
        default = None
        if child.value is not None:
            default_kind = "value"
            default = ast.unparse(child.value)
            if isinstance(child.value, ast.Call) and ast.unparse(
                child.value.func
            ).endswith("field"):
                for keyword in child.value.keywords:
                    if keyword.arg == "default_factory":
                        default_kind = "factory"
                        default = ast.unparse(keyword.value)
                    elif keyword.arg == "default":
                        default_kind = "value"
                        default = ast.unparse(keyword.value)
        fields.append(
            {
                "name": child.target.id,
                "annotation": annotation,
                "default_kind": default_kind,
                "default": default,
            }
        )
    return fields


def _describe_definition(module: str, node: ast.AST) -> dict[str, object]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return {
            "target": f"{module}.{node.name}",
            "kind": "async-function"
            if isinstance(node, ast.AsyncFunctionDef)
            else "function",
            "signature": f"({ast.unparse(node.args)})",
            "decorators": _decorator_names(node),
        }
    if not isinstance(node, ast.ClassDef):
        raise AssertionError(f"unsupported definition: {type(node)!r}")
    decorators = _decorator_names(node)
    is_dataclass = any(
        name == "dataclass"
        or name.startswith("dataclass(")
        or name.endswith(".dataclass")
        for name in decorators
    )
    bases = [ast.unparse(base) for base in node.bases]
    enum_values = []
    if any(base.rsplit(".", 1)[-1].endswith("Enum") for base in bases):
        for child in node.body:
            if (
                isinstance(child, ast.Assign)
                and len(child.targets) == 1
                and isinstance(child.targets[0], ast.Name)
            ):
                enum_values.append(
                    {"name": child.targets[0].id, "value": ast.unparse(child.value)}
                )
    return {
        "target": f"{module}.{node.name}",
        "kind": "class",
        "bases": bases,
        "decorators": decorators,
        "dataclass": is_dataclass,
        "fields": _dataclass_fields(node) if is_dataclass else [],
        "enum_values": enum_values,
        "exception_type": any(base.endswith(("Error", "Exception")) for base in bases),
    }


def api_manifest(repo: Path, public_modules: Sequence[str]) -> dict[str, object]:
    paths = _python_modules(repo)
    parsed: dict[str, tuple[Path, ast.Module]] = {}
    indices: dict[
        str, tuple[dict[str, ast.AST], dict[str, tuple[str, str | None]]]
    ] = {}
    class_module_overrides: dict[str, dict[str, str]] = {}
    for module, path in paths.items():
        _, tree = _read_python(path)
        parsed[module] = (path, tree)
        indices[module] = _module_index(module, path, tree)
        class_module_overrides[module] = _class_module_overrides(tree)

    def describe(
        module: str, name: str, seen: set[tuple[str, str]]
    ) -> dict[str, object]:
        key = (module, name)
        if key in seen:
            raise BaselineError(f"cyclic API re-export: {module}.{name}")
        seen = {*seen, key}
        if module not in indices:
            return {"target": f"{module}.{name}", "kind": "external-or-missing"}
        definitions, imports = indices[module]
        if name in definitions:
            description = _describe_definition(module, definitions[name])
            if isinstance(definitions[name], ast.ClassDef) and (
                compatibility_module := class_module_overrides[module].get(name)
            ):
                description["target"] = f"{compatibility_module}.{name}"
            return description
        if name in imports:
            target_module, target_name = imports[name]
            if target_name is None:
                return {"target": target_module, "kind": "module"}
            return describe(target_module, target_name, seen)
        return {"target": f"{module}.{name}", "kind": "unresolved"}

    modules = []
    schema_entries = []
    for module in public_modules:
        if module not in parsed:
            raise BaselineError(f"public module not found: {module}")
        path, tree = parsed[module]
        exports = _module_all(tree)
        objects = []
        for name in exports:
            description = describe(module, name, set())
            entry = {"name": name, **description}
            objects.append(entry)
            if description.get("dataclass") and (
                name in {"Config", "Result", "PathResult"}
                or name.endswith(("Config", "Result", "Table", "Samples"))
            ):
                schema_entries.append({"export": f"{module}.{name}", **description})
        modules.append(
            {
                "module": module,
                "path": path.relative_to(repo).as_posix(),
                "all": exports,
                "objects": objects,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "classification": "curated-public-modules-only",
        "modules": modules,
        "schemas": sorted(schema_entries, key=lambda item: str(item["export"])),
    }


def _import_edges(repo: Path) -> tuple[list[str], list[dict[str, object]]]:
    paths = _python_modules(repo)
    edges = []
    for module, path in sorted(paths.items()):
        _, tree = _read_python(path)
        is_package = path.name == "__init__.py"
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
                kind = "import"
            elif isinstance(node, ast.ImportFrom):
                base = _resolve_import_from(module, is_package, node)
                targets = [base] if base else []
                kind = "from"
            else:
                continue
            for target in targets:
                edges.append(
                    {
                        "source": module,
                        "target": target,
                        "line": node.lineno,
                        "kind": kind,
                        "internal": target == PACKAGE
                        or target.startswith(f"{PACKAGE}."),
                    }
                )
    return sorted(paths), sorted(
        edges,
        key=lambda edge: (str(edge["source"]), int(edge["line"]), str(edge["target"])),
    )


def _strongly_connected_components(
    modules: Sequence[str], edges: Sequence[dict[str, object]]
) -> list[list[str]]:
    adjacency = {module: [] for module in modules}
    for edge in edges:
        source, target = str(edge["source"]), str(edge["target"])
        if edge["internal"] and target in adjacency:
            adjacency[source].append(target)
    index = 0
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[list[str]] = []

    def visit(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for target in sorted(set(adjacency[node])):
            if target not in indices:
                visit(target)
                lowlinks[node] = min(lowlinks[node], lowlinks[target])
            elif target in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[target])
        if lowlinks[node] == indices[node]:
            component = []
            while True:
                item = stack.pop()
                on_stack.remove(item)
                component.append(item)
                if item == node:
                    break
            if len(component) > 1 or node in adjacency[node]:
                components.append(sorted(component))

    for module in sorted(modules):
        if module not in indices:
            visit(module)
    return sorted(components)


def _solver_area(module: str) -> str | None:
    areas = {
        f"{PACKAGE}.path": "path",
        f"{PACKAGE}.deterministic": "deterministic",
        f"{PACKAGE}.montecarlo.basic": "montecarlo.basic",
        f"{PACKAGE}.montecarlo.bdpt": "montecarlo.bdpt",
    }
    for prefix, label in areas.items():
        if module == prefix or module.startswith(f"{prefix}."):
            return label
    return None


def import_graph_manifest(repo: Path) -> dict[str, object]:
    modules, edges = _import_edges(repo)
    cross_solver = []
    direct_ops = []
    for edge in edges:
        source_area = _solver_area(str(edge["source"]))
        target_area = _solver_area(str(edge["target"]))
        if source_area and target_area and source_area != target_area:
            cross_solver.append(edge)
        if str(edge["target"]) == f"{PACKAGE}.core.kernels.ops":
            direct_ops.append(edge)
    return {
        "schema_version": SCHEMA_VERSION,
        "modules": modules,
        "edges": edges,
        "cycles": _strongly_connected_components(modules, edges),
        "solver_to_solver_imports": cross_solver,
        "direct_core_kernels_ops_imports": direct_ops,
    }


class _FunctionHashVisitor(ast.NodeVisitor):
    def __init__(self, module: str, relative_path: str) -> None:
        self.module = module
        self.relative_path = relative_path
        self.stack: list[str] = []
        self.entries: list[dict[str, object]] = []

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        qualified = ".".join((self.module, *self.stack, node.name))
        body_ast = ast.dump(
            ast.Module(body=node.body, type_ignores=[]),
            annotate_fields=True,
            include_attributes=False,
        )
        normalized_ast = ast.dump(node, annotate_fields=True, include_attributes=False)
        self.entries.append(
            {
                "path": self.relative_path,
                "qualified_name": qualified,
                "line": node.lineno,
                "end_line": node.end_lineno,
                "kind": "async-function"
                if isinstance(node, ast.AsyncFunctionDef)
                else "function",
                "body_sha256": _sha256_bytes(body_ast.encode("utf-8")),
                "normalized_ast_sha256": _sha256_bytes(normalized_ast.encode("utf-8")),
            }
        )
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self._visit_function(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()


def python_body_hashes(repo: Path) -> list[dict[str, object]]:
    entries = []
    for module, path in sorted(_python_modules(repo).items()):
        _, tree = _read_python(path)
        visitor = _FunctionHashVisitor(module, path.relative_to(repo).as_posix())
        visitor.visit(tree)
        entries.extend(visitor.entries)
    return sorted(entries, key=lambda item: (str(item["path"]), int(item["line"])))


def _matching_delimiter(
    text: str, start: int, opening: str = "(", closing: str = ")"
) -> int:
    if start >= len(text) or text[start] != opening:
        raise ValueError("start does not point at the opening delimiter")
    depth = 0
    quote: str | None = None
    escaped = False
    line_comment = False
    block_comment = False
    index = start
    while index < len(text):
        char = text[index]
        next_char = text[index + 1] if index + 1 < len(text) else ""
        if line_comment:
            if char == "\n":
                line_comment = False
        elif block_comment:
            if char == "*" and next_char == "/":
                block_comment = False
                index += 1
        elif quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char == "/" and next_char == "/":
            line_comment = True
            index += 1
        elif char == "/" and next_char == "*":
            block_comment = True
            index += 1
        elif char in {'"', "'"}:
            quote = char
        elif char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return index
        index += 1
    raise BaselineError(f"unmatched {opening!r} delimiter")


def _mask_cpp_comments(source: str) -> str:
    output = list(source)
    index = 0
    quote: str | None = None
    escaped = False
    while index < len(source):
        char = source[index]
        next_char = source[index + 1] if index + 1 < len(source) else ""
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in {'"', "'"}:
            quote = char
        elif char == "/" and next_char == "/":
            end = source.find("\n", index + 2)
            end = len(source) if end < 0 else end
            for position in range(index, end):
                output[position] = " "
            index = end - 1
        elif char == "/" and next_char == "*":
            end = source.find("*/", index + 2)
            if end < 0:
                raise BaselineError("unterminated C++ block comment")
            for position in range(index, end + 2):
                if output[position] != "\n":
                    output[position] = " "
            index = end + 1
        index += 1
    return "".join(output)


_CPP_TOKEN = re.compile(
    r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|'
    r"[A-Za-z_]\w*|(?:\d+\.\d*|\.\d+|\d+)(?:[eEpP][+-]?\d+)?[A-Za-z_]*|"
    r"::|->\*|->|<<=|>>=|<=>|==|!=|<=|>=|&&|\|\||\+\+|--|"
    r"\+=|-=|\*=|/=|%=|&=|\|=|\^=|<<|>>|\.\.\.|[^\s]"
)


def _cpp_tokens(source: str) -> list[str]:
    return _CPP_TOKEN.findall(_mask_cpp_comments(source))


_ADR033_UNCHANGED_CHANNEL_IDENTIFIERS = frozenset(
    {"channel_backward_adjoint", "channel_jvp"}
)
_ADR033_METADATA_SUFFIXES = frozenset(
    {"abi_version", "git_dirty", "git_sha"}
)
_ADR033_SHORT_MACRO_SUFFIXES = frozenset(
    {
        "BDPT_CHECK_CONNECTION_SAMPLE_ROWS",
        "BDPT_CHECK_CONNECTION_SAMPLE_TENSORS",
        "BDPT_CONNECTION_OUTPUT_POINTERS",
        "DIFFRACTION_ALLOCATE_STATE_PACK",
        "DIFFRACTION_CHECK_STATE_PACK_POWER",
        "DIFFRACTION_CHECK_STATE_PACK_SHAPES",
        "DIFFRACTION_CHECK_STATE_PACK_TENSORS",
        "DIFFRACTION_STATE_PACK_INPUT_POINTERS",
        "DIFFRACTION_STATE_PACK_OUTPUT_POINTERS",
        "DIFFRACTION_STATE_PACK_RESULTS",
        "DIFFRACTION_WEDGE_COMMON_ARGUMENTS",
        "LOS_CHECK_VISIBILITY_APPLICATION",
        "LOS_VISIBILITY_LAUNCH_ARGUMENTS",
        "MC_WALL_PRODUCT_COMMON_KERNEL_PARAMS",
        "PATH_BLOCK_APPEND",
        "PATH_BLOCK_FIELDS",
        "PATH_BLOCK_LIST",
        "PATH_BLOCK_RESERVE",
        "PATH_BLOCK_TO_DICT",
        "REFLECTION_LAUNCH_INPUT_PREFIX",
        "REFLECTION_PREPARE_LAUNCH_INPUTS",
        "SCATTERING_CHAIN_ENSEMBLE_PRIMAL_ARGS",
        "SCATTERING_CHAIN_MEDIA_ARGS",
        "SCATTERING_CHAIN_REALIZATION_PRIMAL_ARGS",
        "SCATTERING_ENSEMBLE_PRIMAL_ARGS",
        "SCATTERING_PATCH_PRIMAL_ARGS",
        "TRANSMISSION_SEQUENCE_ARGUMENTS",
    }
)


def _adr033_predecessor_identifier(identifier: str) -> str:
    """Normalize only ADR-033 identity renames for frozen numerical hashes."""

    if identifier in _ADR033_UNCHANGED_CHANNEL_IDENTIFIERS:
        return identifier
    if identifier.startswith("channel_"):
        suffix = identifier.removeprefix("channel_")
        if suffix in _ADR033_METADATA_SUFFIXES:
            return "channel" + "_native_" + suffix
        return "c" + "n_" + suffix
    if identifier.startswith("CHANNEL_"):
        suffix = identifier.removeprefix("CHANNEL_")
        if suffix in _ADR033_SHORT_MACRO_SUFFIXES:
            return "C" + "N_" + suffix
        return "CHANNEL" + "_NATIVE_" + suffix
    return identifier


def _adr033_predecessor_tokens(tokens: list[str]) -> list[str]:
    normalized: list[str] = []
    for index, token in enumerate(tokens):
        if (
            token == "channel"
            and index + 1 < len(tokens)
            and tokens[index + 1] == "::"
        ):
            normalized.append("channel" + "_native")
        else:
            normalized.append(_adr033_predecessor_identifier(token))
    return normalized


_CPP_FUNCTION = re.compile(
    r"(?m)^[ \t]*(?P<signature>"
    r"(?:template\s*<[^;{}]+>\s*)?"
    r"(?:[A-Za-z_~][\w:<>]*[\s*&]+|\[\[[^\]]+\]\]\s*|__\w+__\s*)+"
    r"(?P<name>(?:[A-Za-z_]\w*::)*~?[A-Za-z_]\w*)\s*"
    r"\((?P<params>[^;{}]*)\)\s*"
    r"(?:const\s*)?(?:noexcept(?:\s*\([^)]*\))?\s*)?"
    r"(?:->\s*[^;{}]+\s*)?)\{"
)

_CPP_MULTILINE_FUNCTION = re.compile(
    r"(?m)^[ \t]*(?P<signature>"
    r"(?:template\s*<[^;{}]+>\s*)?"
    r"(?:(?:[A-Za-z_]\w*::)*[A-Za-z_]\w*\s*<[^<>;{}()]+>\s*[\s*&]+|"
    r"[A-Za-z_~][\w:<>]*[\s*&]+|\[\[[^\]]+\]\]\s*|__\w+__\s*)+"
    r"(?P<name>(?:[A-Za-z_]\w*::)*~?[A-Za-z_]\w*)\s*"
    r"\((?P<params>[^;{}]*)\)\s*"
    r"(?:const\s*)?(?:noexcept(?:\s*\([^)]*\))?\s*)?"
    r"(?:->\s*[^;{}]+\s*)?)\{"
)


def cpp_body_hashes(
    repo: Path, *, adr033_predecessor_identity: bool = False
) -> list[dict[str, object]]:
    entries = []
    root = repo / "native" / "channel"
    for path in _source_files(root, _SOURCE_SUFFIXES):
        source = path.read_text(encoding="utf-8-sig")
        masked = _mask_cpp_comments(source)
        matches_by_open_brace = {
            match.end() - 1: match for match in _CPP_FUNCTION.finditer(masked)
        }
        for match in _CPP_MULTILINE_FUNCTION.finditer(masked):
            matches_by_open_brace.setdefault(match.end() - 1, match)
        for open_brace, match in sorted(matches_by_open_brace.items()):
            name = match.group("name")
            if name in {"catch", "for", "if", "switch", "while"}:
                continue
            close_brace = _matching_delimiter(source, open_brace, "{", "}")
            signature_tokens = _cpp_tokens(
                source[match.start("signature") : open_brace]
            )
            body_tokens = _cpp_tokens(source[open_brace + 1 : close_brace])
            if adr033_predecessor_identity:
                name = _adr033_predecessor_identifier(name)
                signature_tokens = _adr033_predecessor_tokens(signature_tokens)
                body_tokens = _adr033_predecessor_tokens(body_tokens)
            entries.append(
                {
                    "path": path.relative_to(repo).as_posix(),
                    "name": name,
                    "line": source.count("\n", 0, match.start()) + 1,
                    "token_count": len(body_tokens),
                    "signature_sha256": _sha256_bytes(
                        " ".join(signature_tokens).encode("utf-8")
                    ),
                    "body_sha256": _sha256_bytes(" ".join(body_tokens).encode("utf-8")),
                }
            )
    return sorted(entries, key=lambda item: (str(item["path"]), int(item["line"])))


def body_hash_manifest(repo: Path) -> dict[str, object]:
    python_entries = python_body_hashes(repo)
    native_entries = cpp_body_hashes(repo)
    return {
        "schema_version": SCHEMA_VERSION,
        "python_normalization": "ast.dump(include_attributes=False)",
        "native_normalization": "comments-and-whitespace-removed-token-stream",
        "python": python_entries,
        "native": native_entries,
    }


def _split_cpp_arguments(source: str) -> list[str]:
    arguments = []
    start = 0
    depths = {"(": 0, "[": 0, "{": 0, "<": 0}
    pairs = {")": "(", "]": "[", "}": "{", ">": "<"}
    quote: str | None = None
    escaped = False
    for index, char in enumerate(source):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {'"', "'"}:
            quote = char
        elif char in depths:
            depths[char] += 1
        elif char in pairs and depths[pairs[char]]:
            depths[pairs[char]] -= 1
        elif char == "," and not any(depths.values()):
            arguments.append(source[start:index].strip())
            start = index + 1
    arguments.append(source[start:].strip())
    return arguments


def _cpp_string(value: str) -> str | None:
    value = value.strip()
    if not (value.startswith('"') and value.endswith('"')):
        return None
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return None
    return parsed if isinstance(parsed, str) else None


def _find_cpp_declaration(
    sources: Sequence[tuple[Path, str]], target: str
) -> tuple[str | None, list[dict[str, object]]]:
    pattern = re.compile(
        rf"(?m)^[ \t]*(?P<return>[A-Za-z_][^\n;{{}}]*?)\s+{re.escape(target)}\s*\("
    )
    for _, source in sources:
        match = pattern.search(_mask_cpp_comments(source))
        if match is None:
            continue
        open_paren = match.end() - 1
        close_paren = _matching_delimiter(source, open_paren)
        tail = source[close_paren + 1 : close_paren + 80]
        if not re.match(r"\s*(?:const\s*)?(?:noexcept\s*)?[;{]", tail):
            continue
        parameters = []
        raw_parameters = source[open_paren + 1 : close_paren].strip()
        if raw_parameters and raw_parameters != "void":
            for raw in _split_cpp_arguments(raw_parameters):
                before_default, separator, default = raw.partition("=")
                name_match = re.search(r"([A-Za-z_]\w*)\s*$", before_default)
                parameter_name = name_match.group(1) if name_match else None
                parameter_type = (
                    before_default[: name_match.start()].strip()
                    if name_match
                    else before_default.strip()
                )
                parameters.append(
                    {
                        "name": parameter_name,
                        "type": " ".join(parameter_type.split()),
                        "default": default.strip() if separator else None,
                    }
                )
        return " ".join(match.group("return").split()), parameters
    return None, []


def _return_arity(return_type: str | None) -> int | None:
    if return_type is None:
        return None
    if return_type == "void":
        return 0
    tuple_match = re.search(r"(?:std::)?tuple\s*<(.*)>$", return_type)
    if tuple_match:
        return len(_split_cpp_arguments(tuple_match.group(1)))
    return 1


def binding_manifest(repo: Path) -> dict[str, object]:
    native_root = repo / "native" / "channel"
    paths = _source_files(native_root, _SOURCE_SUFFIXES)
    sources = [(path, path.read_text(encoding="utf-8-sig")) for path in paths]
    symbols = []
    for path, source in sources:
        search_from = 0
        while True:
            start = source.find("module.def", search_from)
            if start < 0:
                break
            open_paren = source.find("(", start + len("module.def"))
            if open_paren < 0:
                raise BaselineError(f"malformed module.def in {path}")
            close_paren = _matching_delimiter(source, open_paren)
            arguments = _split_cpp_arguments(source[open_paren + 1 : close_paren])
            search_from = close_paren + 1
            if len(arguments) < 2:
                raise BaselineError(
                    f"module.def has fewer than two arguments in {path}"
                )
            name = _cpp_string(arguments[0])
            if name is None:
                raise BaselineError(f"non-literal pybind symbol in {path}")
            target_match = re.fullmatch(r"&\s*([A-Za-z_]\w*(?:::\w+)*)", arguments[1])
            target = target_match.group(1) if target_match else arguments[1]
            return_type, parameters = _find_cpp_declaration(sources, target)
            pybind_arguments = []
            for extra in arguments[2:]:
                arg_match = re.search(
                    r"(?:pybind11|py)::arg\(\s*\"([^\"]+)\"\s*\)(?:\s*=\s*(.*))?",
                    extra,
                    re.DOTALL,
                )
                if arg_match:
                    pybind_arguments.append(
                        {
                            "name": arg_match.group(1),
                            "default": (
                                " ".join(arg_match.group(2).split())
                                if arg_match.group(2)
                                else None
                            ),
                        }
                    )
            symbols.append(
                {
                    "name": name,
                    "target": target,
                    "path": path.relative_to(repo).as_posix(),
                    "line": source.count("\n", 0, start) + 1,
                    "return_type": return_type,
                    "return_arity": _return_arity(return_type),
                    "parameters": parameters,
                    "pybind_arguments": pybind_arguments,
                    "doc": _cpp_string(arguments[2]) if len(arguments) >= 3 else None,
                }
            )
    symbols.sort(key=lambda item: str(item["name"]))
    counts: dict[str, int] = {}
    for symbol in symbols:
        name = str(symbol["name"])
        counts[name] = counts.get(name, 0) + 1
    return {
        "schema_version": SCHEMA_VERSION,
        "module": "_channel",
        "symbols": symbols,
        "duplicate_symbols": sorted(
            name for name, count in counts.items() if count > 1
        ),
    }


def _call_name(node: ast.AST) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _literal_or_source(node: ast.AST | None) -> object:
    if node is None:
        return None
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError):
        return ast.unparse(node)
    return value


def pytest_marker_manifest(repo: Path) -> dict[str, object]:
    tests_root = repo / "tests"
    entries = []
    recognized = {
        "pytest.importorskip": "importorskip",
        "pytest.mark.skip": "skip",
        "pytest.mark.skipif": "skipif",
        "pytest.mark.xfail": "xfail",
        "pytest.skip": "skip",
        "pytest.xfail": "xfail",
    }
    for path in sorted(tests_root.rglob("*.py")):
        _, tree = _read_python(path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            call_name = _call_name(node.func)
            if call_name not in recognized:
                continue
            kind = recognized[call_name]
            keywords = {
                keyword.arg: keyword.value for keyword in node.keywords if keyword.arg
            }
            reason_node = keywords.get("reason")
            condition_node = None
            if kind == "skipif":
                condition_node = node.args[0] if node.args else None
            elif kind in {"skip", "xfail"} and reason_node is None and node.args:
                reason_node = node.args[0]
            elif kind == "importorskip" and reason_node is None:
                module = _literal_or_source(node.args[0] if node.args else None)
                reason_node = ast.Constant(
                    value=f"optional module unavailable: {module}"
                )
            entry = {
                "path": path.relative_to(repo).as_posix(),
                "line": node.lineno,
                "kind": kind,
                "reason": _literal_or_source(reason_node),
                "condition": _literal_or_source(condition_node),
                "strict": _literal_or_source(keywords.get("strict")),
            }
            identity = _sha256_bytes(_canonical_bytes(entry))[:16]
            entries.append({"id": identity, **entry})
    entries.sort(
        key=lambda item: (str(item["path"]), int(item["line"]), str(item["kind"]))
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "entries": entries,
        "counts": {
            kind: sum(entry["kind"] == kind for entry in entries)
            for kind in sorted({str(entry["kind"]) for entry in entries})
        },
    }


def collect_pytest_manifest(
    repo: Path, basetemp: Path | None = None
) -> dict[str, object]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    basetemp = (
        basetemp.resolve()
        if basetemp is not None
        else (repo / "artifacts" / "pytest-baseline-collect").resolve()
    )
    try:
        basetemp_label = basetemp.relative_to(repo.resolve()).as_posix()
    except ValueError:
        raise BaselineError("pytest basetemp must be inside the repository") from None
    command = (
        sys.executable,
        "-m",
        "pytest",
        "--collect-only",
        "-q",
        "-p",
        "no:cacheprovider",
        f"--basetemp={basetemp}",
    )
    result = _run(command, cwd=repo, timeout=180, env=env)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise BaselineError(f"pytest collection failed: {detail}")
    nodeids = [
        line.strip().replace("\\", "/")
        for line in result.stdout.splitlines()
        if "::" in line and line.lstrip().startswith("tests")
    ]
    if not nodeids:
        raise BaselineError("pytest collection returned no test node ids")
    return {
        "schema_version": SCHEMA_VERSION,
        "command": [
            "python",
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            f"--basetemp={basetemp_label}",
        ],
        "count": len(nodeids),
        "nodeids": nodeids,
        "nodeids_sha256": _sha256_bytes(_canonical_bytes(nodeids)),
    }


def _reject_sensitive_keys(value: object, location: str = "$") -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            key_text = str(key)
            if _SENSITIVE_KEY.search(key_text):
                raise BaselineError(
                    f"runtime artifact contains sensitive key at {location}.{key_text}"
                )
            _reject_sensitive_keys(nested, f"{location}.{key_text}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_sensitive_keys(nested, f"{location}[{index}]")


def runtime_artifact_manifest(specifications: Sequence[str]) -> dict[str, object]:
    artifacts = []
    seen: set[str] = set()
    for specification in specifications:
        kind, separator, raw_path = specification.partition("=")
        if not separator or not re.fullmatch(r"[a-z][a-z0-9-]*", kind):
            raise BaselineError(
                "runtime artifacts must use KIND=PATH with a lowercase hyphenated kind"
            )
        if kind in seen:
            raise BaselineError(f"duplicate runtime artifact kind: {kind}")
        seen.add(kind)
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise BaselineError(f"runtime artifact does not exist: {path}")
        raw = path.read_bytes()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            raise BaselineError(
                f"runtime artifact is not valid JSON: {path.name}: {error}"
            ) from error
        _reject_sensitive_keys(payload)
        artifacts.append(
            {
                "kind": kind,
                "source_sha256": _sha256_bytes(raw),
                "canonical_sha256": _sha256_bytes(_canonical_bytes(payload)),
                "payload": payload,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "artifacts": sorted(artifacts, key=lambda item: str(item["kind"])),
        "required_kinds": list(REQUIRED_RUNTIME_KINDS),
        "missing_required_kinds": sorted(set(REQUIRED_RUNTIME_KINDS) - seen),
    }


def freeze_baseline(
    repo: Path,
    output_root: Path,
    *,
    public_modules: Sequence[str] = DEFAULT_PUBLIC_MODULES,
    runtime_artifacts: Sequence[str] = (),
    native_binaries: Sequence[Path] = (),
    cmake_cache: Path | None = None,
    rayd_root: Path | None = None,
    pytest_basetemp: Path | None = None,
    collect_tests: bool = True,
) -> Path:
    repo = repo.resolve()
    if not (repo / "pyproject.toml").is_file() or not (repo / ".git").exists():
        raise BaselineError(f"not a channel repository: {repo}")
    git_state = _git_state(repo)
    sha = str(git_state["sha"])
    output_root = output_root.resolve()
    destination = output_root / sha
    if destination.exists():
        raise BaselineError(f"immutable baseline already exists: {destination}")
    staging = output_root / f".tmp-{sha}-{os.getpid()}"
    if staging.exists():
        raise BaselineError(f"staging directory already exists: {staging}")

    api_document = api_manifest(repo, public_modules)
    documents: dict[str, object] = {
        "environment.json": environment_manifest(repo, rayd_root),
        "build.json": build_manifest(repo, native_binaries, cmake_cache),
        "api.json": api_document,
        "schemas.json": {
            "schema_version": SCHEMA_VERSION,
            "schemas": api_document["schemas"],
        },
        "bindings.json": binding_manifest(repo),
        "import_graph.json": import_graph_manifest(repo),
        "body_hashes.json": body_hash_manifest(repo),
        "pytest_markers.json": pytest_marker_manifest(repo),
        "runtime_artifacts.json": runtime_artifact_manifest(runtime_artifacts),
    }
    if collect_tests:
        documents["pytest_collection.json"] = collect_pytest_manifest(
            repo, pytest_basetemp
        )
    else:
        documents["pytest_collection.json"] = {
            "schema_version": SCHEMA_VERSION,
            "status": "not-collected",
            "count": None,
            "nodeids": [],
        }

    try:
        output_root.mkdir(parents=True, exist_ok=True)
        staging.mkdir()
        for filename, document in documents.items():
            _write_json(staging / filename, document)
        checksums = {
            filename: _sha256_file(staging / filename) for filename in sorted(documents)
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "git_sha": sha,
            "artifact_policy": "immutable-git-sha-directory",
            "collector": "tools/refactor_baseline.py",
            "files": checksums,
            "complete": (
                collect_tests
                and not documents["runtime_artifacts.json"]["missing_required_kinds"]  # type: ignore[index]
            ),
        }
        _write_json(staging / "manifest.json", manifest)
        staging.rename(destination)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root (defaults to the parent of tools/)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="artifact root (defaults to REPO/artifacts/refactor_baseline)",
    )
    parser.add_argument(
        "--public-module",
        action="append",
        dest="public_modules",
        help="replace the curated public module list; may be repeated",
    )
    parser.add_argument(
        "--runtime-artifact",
        action="append",
        default=[],
        metavar="KIND=PATH",
        help=(
            "attach JSON emitted by an existing harness; required kinds are "
            "solver-results, launch-ledger, and performance"
        ),
    )
    parser.add_argument(
        "--native-binary",
        action="append",
        type=Path,
        default=[],
        help="native extension binary to fingerprint; may be repeated",
    )
    parser.add_argument(
        "--cmake-cache",
        type=Path,
        help=(
            "CMakeCache.txt for actual compile flags; only safe allowlisted keys are "
            "stored and absolute user paths are redacted"
        ),
    )
    parser.add_argument(
        "--rayd-root",
        type=Path,
        help="RayD repository to fingerprint; its path is never written to artifacts",
    )
    parser.add_argument(
        "--no-pytest-collect",
        action="store_true",
        help="skip pytest collection for a partial development artifact",
    )
    parser.add_argument(
        "--pytest-basetemp",
        type=Path,
        help=(
            "repository-local pytest scratch directory (defaults to "
            "REPO/artifacts/pytest-baseline-collect)"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo = args.repo.resolve()
    output_root = (
        args.output_root.resolve()
        if args.output_root
        else repo / "artifacts" / "refactor_baseline"
    )
    try:
        destination = freeze_baseline(
            repo,
            output_root,
            public_modules=args.public_modules or DEFAULT_PUBLIC_MODULES,
            runtime_artifacts=args.runtime_artifact,
            native_binaries=args.native_binary,
            cmake_cache=args.cmake_cache,
            rayd_root=args.rayd_root,
            pytest_basetemp=args.pytest_basetemp,
            collect_tests=not args.no_pytest_collect,
        )
    except (BaselineError, OSError, subprocess.SubprocessError) as error:
        print(f"refactor baseline failed: {error}", file=sys.stderr)
        return 2
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

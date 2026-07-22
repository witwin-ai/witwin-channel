"""Fresh, commit-bound Channel builds used by formal Phase 12 evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import shutil

from .artifacts import (
    ArtifactStore,
    hash_external_stable,
    read_external_stable,
    reject_reparse_chain,
)
from .contracts import (
    COMPARISON_GROUPS,
    ComparisonConfig,
    EvidenceError,
    RunnerConfig,
    VariantConfig,
    controlled_environment,
)
from .workers import run_captured


_BUILD_ROLES = (
    ("P", "enumerated_penetration", "baseline"),
    ("E", "enumerated_penetration", "candidate"),
    ("M", "montecarlo_penetration", "candidate"),
    ("D", "diffraction", "baseline"),
    ("S", "diffraction", "candidate"),
)
_RAYD_COMMIT = "474c122aa3cd6b6d098675e076a73e6f485bd6be"
_RAYD_HEADER = Path("backends/torch/include/rayd/torch/integration.h")
_MSVC_ENV_KEYS = (
    "INCLUDE", "LIB", "LIBPATH", "PATH", "VCToolsInstallDir", "VCINSTALLDIR",
    "WindowsSdkDir", "WindowsSDKVersion", "UCRTVersion", "UniversalCRTSdkDir",
)


def _parse_msvc_environment_stdout(payload: bytes) -> dict[str, str]:
    if b"\x00" in payload:
        raise EvidenceError("vcvars allowlist output contains NUL")
    try:
        lines = payload.decode("utf-8", errors="strict").splitlines()
    except UnicodeDecodeError as exc:
        raise EvidenceError("vcvars allowlist output is not UTF-8") from exc
    values: dict[str, str] = {}
    for line in lines:
        if "=" not in line:
            raise EvidenceError("vcvars allowlist output contains an unexpected line")
        name, value = line.split("=", 1)
        if name not in _MSVC_ENV_KEYS or name in values or not value or value == f"!{name}!":
            raise EvidenceError("vcvars allowlist output has a missing/duplicate/unknown key")
        values[name] = value
    if tuple(values) != _MSVC_ENV_KEYS:
        raise EvidenceError("vcvars allowlist output order/key set differs")
    return values


def _capture_compiler_versions(
    config: RunnerConfig,
    *,
    root: Path,
    store: ArtifactStore,
    environment: Mapping[str, str],
    timeout_seconds: int,
) -> dict[str, object]:
    commands = {
        "cl": ([str(config.tools.cl), "/Bv"], rb"Compiler Version ([0-9.]+) for x64"),
        "link": ([str(config.tools.link), "/?"], rb"Linker Version ([0-9.]+)"),
        "nvcc": ([str(config.tools.nvcc), "--version"], rb"\bV([0-9]+\.[0-9]+\.[0-9]+)\b"),
    }
    result: dict[str, object] = {}
    for name, (argv, pattern) in commands.items():
        capture = run_captured(
            argv,
            cwd=root,
            environment=environment,
            timeout_seconds=timeout_seconds,
            store=store,
            stem=f"channel-{name}-version",
            # cl /Bv reports its version and then rejects the missing input file.
            # The retained output signature, not that incidental exit status, is
            # the version-probe contract.
            expected_returncode=None,
        )
        stdout = capture.pop("stdout_bytes")
        stderr = capture.pop("stderr_bytes")
        match = re.search(pattern, bytes(stdout) + b"\n" + bytes(stderr))
        if match is None:
            raise EvidenceError(f"{name} version probe did not emit the accepted signature")
        result[name] = {
            "version": match.group(1).decode("ascii", errors="strict"),
            "capture": capture,
        }
    cl_version = str(result["cl"]["version"])  # type: ignore[index]
    link_version = str(result["link"]["version"])  # type: ignore[index]
    nvcc_version = str(result["nvcc"]["version"])  # type: ignore[index]
    if not cl_version.startswith("19."):
        raise EvidenceError("cl version probe is not an accepted VS2022 compiler")
    if not link_version.startswith("14." + cl_version.removeprefix("19.")):
        raise EvidenceError("cl and link version probes do not identify one VS toolset")
    if not nvcc_version.startswith("12.9."):
        raise EvidenceError("nvcc version probe is not CUDA 12.9")
    return result


def _prepare_msvc_environment(
    config: RunnerConfig, *, root: Path, store: ArtifactStore, timeout_seconds: int,
) -> tuple[dict[str, str], dict[str, object]]:
    try:
        expected_vcvars = config.tools.cl.parents[6] / "Auxiliary" / "Build" / "vcvars64.bat"
    except IndexError as exc:
        raise EvidenceError("configured cl.exe path cannot identify its VS toolchain") from exc
    if config.tools.vcvars64.resolve() != expected_vcvars.resolve():
        raise EvidenceError("vcvars64.bat does not belong to the configured cl.exe toolchain")
    if config.tools.cmd.name.casefold() != "cmd.exe":
        raise EvidenceError("configured command interpreter is not cmd.exe")
    echoes = " && ".join(f"echo {name}=!{name}!" for name in _MSVC_ENV_KEYS)
    command = f'chcp 65001 >nul && call "{config.tools.vcvars64}" >nul && {echoes}'
    capture = run_captured(
        [str(config.tools.cmd), "/d", "/v:on", "/s", "/c", command],
        cwd=root,
        environment=controlled_environment(config),
        timeout_seconds=timeout_seconds,
        store=store,
        stem="channel-msvc-environment",
    )
    stdout = capture.pop("stdout_bytes")
    capture.pop("stderr_bytes")
    values = _parse_msvc_environment_stdout(stdout)
    environment = controlled_environment(config)
    environment.update(values)
    compiler_versions = _capture_compiler_versions(
        config,
        root=root,
        store=store,
        environment=environment,
        timeout_seconds=timeout_seconds,
    )
    payload = json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
    manifest = store.write_bytes(
        "builds/msvc-environment.json", payload, allow_empty=False
    )
    return environment, {
        "capture": capture,
        "manifest": manifest,
        "keys": list(_MSVC_ENV_KEYS),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "compiler_versions": compiler_versions,
    }


def parse_cuobjdump_resource_usage(payload: bytes) -> list[dict[str, object]]:
    text = payload.decode("utf-8", errors="replace")
    matches = list(
        re.finditer(
            r"(?im)^\s*Function(?:\s*:\s*(\S+)|\s+(\S+?)\s*:?)\s*$",
            text,
        )
    )
    rows: list[dict[str, object]] = []
    for index, match in enumerate(matches):
        block = text[match.end():matches[index + 1].start() if index + 1 < len(matches) else len(text)]
        values: dict[str, int] = {}
        for name, raw in re.findall(r"(?i)\b(REG|STACK|SHARED|LOCAL)\s*:\s*([0-9]+)", block):
            key = name.casefold()
            if key in values:
                raise EvidenceError("cuobjdump resource block contains a duplicate field")
            values[key] = int(raw)
        if set(values) != {"reg", "stack", "shared", "local"}:
            continue
        rows.append(
            {
                "function": match.group(1) or match.group(2),
                "registers_per_thread": values["reg"],
                "stack_bytes": values["stack"],
                "shared_bytes": values["shared"],
                "local_bytes": values["local"],
            }
        )
    if not rows:
        raise EvidenceError("cuobjdump emitted no complete compiler resource records")
    return rows


def _cache(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw or raw.startswith(("//", "#")) or "=" not in raw:
            continue
        key_and_type, value = raw.split("=", 1)
        key = key_and_type.split(":", 1)[0]
        if key in values:
            raise EvidenceError(f"CMake cache contains duplicate key: {key}")
        values[key] = value
    return values


def _same_path(actual: str, expected: Path) -> bool:
    return Path(actual).resolve() == expected.resolve()


def _validate_cache(
    path: Path,
    *,
    config: RunnerConfig,
    source: Path,
    install: Path,
) -> dict[str, str]:
    values = _cache(path)
    exact = {
        "CMAKE_BUILD_TYPE": "Release",
        "CMAKE_CUDA_ARCHITECTURES": "120-real",
        "CMAKE_GENERATOR": "Ninja",
        "CHANNEL_RELEASE_BUILD": "ON",
        "BUILD_TESTING": "OFF",
    }
    if any(values.get(name) != value for name, value in exact.items()):
        raise EvidenceError("fresh Channel CMake cache differs from fixed build policy")
    paths = {
        "CMAKE_HOME_DIRECTORY": source,
        "CMAKE_INSTALL_PREFIX": install,
        "CMAKE_CUDA_COMPILER": config.tools.nvcc,
        "CMAKE_CXX_COMPILER": config.tools.cl,
        "CMAKE_LINKER": config.tools.link,
        "CMAKE_MAKE_PROGRAM": config.tools.ninja,
        "Python_EXECUTABLE": config.variant("enumerated_penetration", "baseline").python_executable,
        "RAYD_SOURCE_DIR": config.rayd_checkout,
    }
    if any(
        name not in values or not _same_path(values[name], expected)
        for name, expected in paths.items()
    ):
        raise EvidenceError("fresh Channel CMake cache has an unbound tool/source path")
    return {name: values[name] for name in sorted(exact.keys() | paths.keys())}


def _tree_manifest(source: Path, installed: Path) -> tuple[list[dict[str, object]], str]:
    rows: list[dict[str, object]] = []
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise EvidenceError(f"Python source tree contains a link: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(source).as_posix()
        target = installed / relative
        source_hash, source_bytes = hash_external_stable(
            path, label=f"Python source {relative}", allow_empty=True
        )
        target_hash, target_bytes = hash_external_stable(
            target, label=f"installed Python source {relative}", allow_empty=True
        )
        if (source_hash, source_bytes) != (target_hash, target_bytes):
            raise EvidenceError(f"installed Python source differs from commit source: {relative}")
        rows.append({"path": relative, "sha256": source_hash, "bytes": source_bytes})
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return rows, hashlib.sha256(payload).hexdigest()


def _retain_toolchain(
    store: ArtifactStore, *, role: str, build: Path
) -> dict[str, dict[str, object]]:
    required = {
        "cmake_cache": [build / "CMakeCache.txt"],
        "configure_log": list(build.glob("CMakeFiles/CMakeConfigureLog.yaml")),
        "cxx_compiler": list(build.glob("CMakeFiles/*/CMakeCXXCompiler.cmake")),
        "cuda_compiler": list(build.glob("CMakeFiles/*/CMakeCUDACompiler.cmake")),
    }
    retained: dict[str, dict[str, object]] = {}
    for name, matches in required.items():
        if len(matches) != 1 or not matches[0].is_file():
            raise EvidenceError(f"fresh Channel build lacks one exact {name} record")
        retained[name] = store.retain_external(
            matches[0], f"builds/{role}/{name}{matches[0].suffix}",
            label=f"Channel {role} {name}", allow_empty=False,
        )
    return retained


def _prepare_rayd_source(
    config: RunnerConfig,
    *,
    commit: str,
    repository_url: str,
    expected_header_sha256: str,
    root: Path,
    store: ArtifactStore,
    timeout_seconds: int,
) -> tuple[Path, dict[str, object]]:
    if commit != _RAYD_COMMIT:
        raise EvidenceError(
            "formal Phase 12 requires the exact accepted RayD commit "
            f"{_RAYD_COMMIT}"
        )
    source = root / "rayd-source"
    if source.exists():
        raise EvidenceError("runner-owned RayD source directory was not initially absent")
    environment = controlled_environment(config)
    clone = run_captured(
        [
            str(config.tools.git), "clone", "--no-hardlinks", "--no-checkout",
            str(config.rayd_checkout), str(source),
        ],
        cwd=root,
        environment=environment,
        timeout_seconds=timeout_seconds,
        store=store,
        stem="channel-rayd-clone",
    )
    clone.pop("stdout_bytes")
    clone.pop("stderr_bytes")
    checkout = run_captured(
        [str(config.tools.git), "checkout", "--detach", commit],
        cwd=source,
        environment=environment,
        timeout_seconds=timeout_seconds,
        store=store,
        stem="channel-rayd-checkout",
    )
    checkout.pop("stdout_bytes")
    checkout.pop("stderr_bytes")
    remote = run_captured(
        [str(config.tools.git), "remote", "set-url", "origin", repository_url],
        cwd=source,
        environment=environment,
        timeout_seconds=timeout_seconds,
        store=store,
        stem="channel-rayd-origin",
    )
    remote.pop("stdout_bytes")
    remote.pop("stderr_bytes")
    tree_capture = run_captured(
        [str(config.tools.git), "rev-parse", "HEAD^{tree}"],
        cwd=source,
        environment=environment,
        timeout_seconds=timeout_seconds,
        store=store,
        stem="channel-rayd-source-tree",
    )
    tree = tree_capture.pop("stdout_bytes").decode("ascii", errors="strict").strip()
    tree_capture.pop("stderr_bytes")
    status = run_captured(
        [str(config.tools.git), "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=source,
        environment=environment,
        timeout_seconds=timeout_seconds,
        store=store,
        stem="channel-rayd-clean-status",
    )
    status_stdout = status.pop("stdout_bytes")
    status.pop("stderr_bytes")
    if status_stdout.strip():
        raise EvidenceError("runner-owned RayD source clone is not clean")
    if len(tree) != 40 or any(value not in "0123456789abcdef" for value in tree):
        raise EvidenceError("runner-owned RayD tree identity is malformed")
    archive_path = store.root / "builds" / "rayd" / "source-tree.tar"
    archive_path.parent.mkdir(parents=True, exist_ok=False)
    archive_capture = run_captured(
        [str(config.tools.git), "archive", "--format=tar", "--output", str(archive_path), "HEAD"],
        cwd=source,
        environment=environment,
        timeout_seconds=timeout_seconds,
        store=store,
        stem="channel-rayd-source-archive",
    )
    archive_capture.pop("stdout_bytes")
    archive_capture.pop("stderr_bytes")
    archive = store.inspect(
        store.relative_for_created_file(archive_path), label="runner-owned RayD source archive",
        minimum_mtime_ns=int(archive_capture["started_time_ns"]),
    )
    header_path = source / _RAYD_HEADER
    header_hash, _ = hash_external_stable(
        header_path, label="runner-owned RayD integration header", allow_empty=False
    )
    if header_hash != expected_header_sha256:
        raise EvidenceError("runner-owned RayD header differs from the Channel lock")
    header_bytes = read_external_stable(
        header_path, label="runner-owned RayD integration header", allow_empty=False
    )[0]
    if (
        b"kIntegrationApiVersion = 6" not in header_bytes
        or b'kIntegrationHeaderIdentity =\n    "rayd.torch.integration"' not in header_bytes
    ):
        raise EvidenceError("runner-owned RayD integration header is not API 6")
    header = store.retain_external(
        header_path, "builds/rayd/integration.h",
        label="runner-owned RayD integration header", allow_empty=False,
    )
    return source.resolve(), {
        "commit": commit,
        "tree": tree,
        "initially_absent": True,
        "clean": True,
        "integration_api_version": 6,
        "integration_identity": "rayd.torch.integration",
        "clone": clone,
        "checkout": checkout,
        "remote": remote,
        "tree_capture": tree_capture,
        "clean_status": status,
        "source_archive_capture": archive_capture,
        "source_archive": archive,
        "integration_header": header,
    }


def _fresh_build(
    config: RunnerConfig,
    *,
    role: str,
    input_variant: VariantConfig,
    commit: str,
    root: Path,
    store: ArtifactStore,
    timeout_seconds: int,
    build_environment: Mapping[str, str],
) -> tuple[VariantConfig, dict[str, object]]:
    role_root = root / f"{role}-{commit}"
    if role_root.exists():
        raise EvidenceError(f"fresh Channel role directory already exists: {role_root}")
    role_root.mkdir(parents=False, exist_ok=False)
    source = role_root / "source"
    build = role_root / "build"
    site_packages = role_root / "install" / "site-packages"
    if any(path.exists() for path in (source, build, site_packages)):
        raise EvidenceError("fresh Channel source/build/install directory was not absent")
    environment = dict(build_environment)
    environment.update({"TORCH_CUDA_ARCH_LIST": "12.0", "CMAKE_BUILD_PARALLEL_LEVEL": "4"})
    clone = run_captured(
        [
            str(config.tools.git), "clone", "--no-hardlinks", "--no-checkout",
            str(input_variant.checkout), str(source),
        ],
        cwd=role_root,
        environment=environment,
        timeout_seconds=timeout_seconds,
        store=store,
        stem=f"channel-{role}-clone",
    )
    clone.pop("stdout_bytes")
    clone.pop("stderr_bytes")
    checkout = run_captured(
        [str(config.tools.git), "checkout", "--detach", commit],
        cwd=source,
        environment=environment,
        timeout_seconds=timeout_seconds,
        store=store,
        stem=f"channel-{role}-checkout",
    )
    checkout.pop("stdout_bytes")
    checkout.pop("stderr_bytes")
    tree_capture = run_captured(
        [str(config.tools.git), "rev-parse", "HEAD^{tree}"],
        cwd=source,
        environment=environment,
        timeout_seconds=timeout_seconds,
        store=store,
        stem=f"channel-{role}-source-tree",
    )
    tree_stdout = tree_capture.pop("stdout_bytes").decode("ascii", errors="strict").strip()
    tree_capture.pop("stderr_bytes")
    if len(tree_stdout) != 40 or any(value not in "0123456789abcdef" for value in tree_stdout):
        raise EvidenceError("fresh Channel source tree identity is not a Git SHA")
    archive_path = store.root / "builds" / role / "source-tree.tar"
    archive_path.parent.mkdir(parents=True, exist_ok=False)
    archive_capture = run_captured(
        [str(config.tools.git), "archive", "--format=tar", "--output", str(archive_path), "HEAD"],
        cwd=source,
        environment=environment,
        timeout_seconds=timeout_seconds,
        store=store,
        stem=f"channel-{role}-source-archive",
    )
    archive_capture.pop("stdout_bytes")
    archive_capture.pop("stderr_bytes")
    archive = store.inspect(
        store.relative_for_created_file(archive_path),
        label=f"Channel {role} source archive",
        minimum_mtime_ns=int(archive_capture["started_time_ns"]),
    )
    configure_argv = [
        str(config.tools.cmake), "-S", str(source), "-B", str(build), "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CUDA_ARCHITECTURES=120-real",
        f"-DCMAKE_CUDA_COMPILER={config.tools.nvcc}",
        f"-DCMAKE_CXX_COMPILER={config.tools.cl}",
        f"-DCMAKE_LINKER={config.tools.link}",
        f"-DCMAKE_MAKE_PROGRAM={config.tools.ninja}",
        f"-DPython_EXECUTABLE={input_variant.python_executable}",
        f"-DRAYD_SOURCE_DIR={config.rayd_checkout}",
        f"-DCMAKE_INSTALL_PREFIX={site_packages}",
        "-DCHANNEL_RELEASE_BUILD=ON", "-DBUILD_TESTING=OFF",
    ]
    configure = run_captured(
        configure_argv, cwd=role_root, environment=environment,
        timeout_seconds=timeout_seconds, store=store, stem=f"channel-{role}-configure",
    )
    configure.pop("stdout_bytes")
    configure.pop("stderr_bytes")
    build_capture = run_captured(
        [str(config.tools.cmake), "--build", str(build), "--parallel", "4"],
        cwd=role_root, environment=environment, timeout_seconds=timeout_seconds,
        store=store, stem=f"channel-{role}-build",
    )
    build_capture.pop("stdout_bytes")
    build_capture.pop("stderr_bytes")
    source_package = source / "src" / "witwin"
    installed_package = site_packages / "witwin"
    site_packages.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source_package, installed_package)
    install_capture = run_captured(
        [str(config.tools.cmake), "--install", str(build), "--prefix", str(site_packages)],
        cwd=role_root, environment=environment, timeout_seconds=timeout_seconds,
        store=store, stem=f"channel-{role}-install",
    )
    install_capture.pop("stdout_bytes")
    install_capture.pop("stderr_bytes")
    cache_values = _validate_cache(
        build / "CMakeCache.txt", config=config, source=source, install=site_packages
    )
    toolchain = _retain_toolchain(store, role=role, build=build)
    extensions = list((installed_package / "channel").glob("_channel*.pyd"))
    if len(extensions) != 1:
        raise EvidenceError("fresh Channel install must contain one extension module")
    extension = extensions[0].resolve()
    resource_capture = run_captured(
        [str(config.tools.cuobjdump), "--dump-resource-usage", str(extension)],
        cwd=role_root, environment=environment, timeout_seconds=timeout_seconds,
        store=store, stem=f"channel-{role}-compiler-resources",
    )
    resource_stdout = resource_capture.pop("stdout_bytes")
    resource_capture.pop("stderr_bytes")
    compiler_resources = parse_cuobjdump_resource_usage(resource_stdout)
    fingerprints = [
        installed_package / "channel" / "runtime" / "_channel.build-fingerprint"
    ]
    fingerprints = [path for path in fingerprints if path.is_file()]
    if len(fingerprints) != 1:
        raise EvidenceError("fresh Channel install must contain one build fingerprint")
    extension_artifact = store.retain_external(
        extension, f"builds/{role}/{extension.name}",
        label=f"Channel {role} extension", allow_empty=False,
    )
    fingerprint_artifact = store.retain_external(
        fingerprints[0], f"builds/{role}/{fingerprints[0].name}",
        label=f"Channel {role} build fingerprint", allow_empty=False,
    )
    manifest_rows, manifest_sha256 = _tree_manifest(source_package, installed_package)
    manifest = store.write_bytes(
        f"builds/{role}/installed-python-manifest.json",
        json.dumps(manifest_rows, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        allow_empty=False,
    )
    post_status = run_captured(
        [str(config.tools.git), "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=source, environment=environment, timeout_seconds=timeout_seconds,
        store=store, stem=f"channel-{role}-post-build-clean-status",
    )
    post_status_stdout = post_status.pop("stdout_bytes")
    post_status.pop("stderr_bytes")
    if post_status_stdout.strip():
        raise EvidenceError(f"fresh Channel {role} source was polluted by its build")
    reject_reparse_chain(role_root, stop=root)
    runtime_variant = replace(
        input_variant,
        checkout=source,
        runner_site_packages=site_packages,
        runner_extension=extension,
    )
    record = {
        "role": role,
        "commit": commit,
        "initially_absent": True,
        "source": str(source),
        "build": str(build),
        "install": str(site_packages),
        "clone": clone,
        "checkout": checkout,
        "source_tree": tree_stdout,
        "source_tree_capture": tree_capture,
        "source_archive_capture": archive_capture,
        "source_archive": archive,
        "configure": configure,
        "build_capture": build_capture,
        "install_capture": install_capture,
        "cmake_cache_values": cache_values,
        "toolchain": toolchain,
        "installed_python_manifest": manifest,
        "installed_python_manifest_sha256": manifest_sha256,
        "extension": extension_artifact,
        "fingerprint": fingerprint_artifact,
        "post_build_clean_status": post_status,
        "compiler_resource_capture": resource_capture,
        "compiler_resources": compiler_resources,
    }
    return runtime_variant, record


def prepare_fresh_channel_builds(
    config: RunnerConfig,
    implementation: dict[str, object],
    *,
    store: ArtifactStore,
    timeout_seconds: int,
) -> tuple[RunnerConfig, dict[str, object]]:
    root = config.build_parent / store.root.name
    if config.build_parent.exists() or root.exists():
        raise EvidenceError("formal build_parent must be initially absent")
    config.build_parent.mkdir(parents=True, exist_ok=False)
    root.mkdir(parents=False, exist_ok=False)
    build_environment, msvc_environment = _prepare_msvc_environment(
        config, root=root, store=store, timeout_seconds=timeout_seconds
    )
    groups = implementation.get("groups")
    if not isinstance(groups, dict):
        raise EvidenceError("verified implementation lacks comparison groups")
    rayd_commit = str(implementation.get("rayd_commit"))
    rayd_repository_url = str(implementation.get("rayd_repository_url"))
    header_sha256 = str(implementation.get("integration_header_sha256"))
    rayd_source, rayd_record = _prepare_rayd_source(
        config, commit=rayd_commit, repository_url=rayd_repository_url,
        expected_header_sha256=header_sha256,
        root=root, store=store, timeout_seconds=timeout_seconds,
    )
    build_config = replace(config, rayd_checkout=rayd_source)
    built_by_commit: dict[str, VariantConfig] = {}
    records: dict[str, dict[str, object]] = {}
    role_by_commit: dict[str, str] = {}
    for role, group, name in _BUILD_ROLES:
        commit_key = f"{name}_commit"
        commit = str(groups[group][commit_key])  # type: ignore[index]
        if commit in built_by_commit:
            raise EvidenceError("P/E/M/D/S roles must identify five unique Channel commits")
        variant, record = _fresh_build(
            build_config, role=role,
            input_variant=build_config.variant(group, name), commit=commit,
            root=root, store=store, timeout_seconds=timeout_seconds,
            build_environment=build_environment,
        )
        built_by_commit[commit] = variant
        role_by_commit[commit] = role
        records[role] = record
    rayd_post_status = run_captured(
        [str(config.tools.git), "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=rayd_source,
        environment=controlled_environment(build_config),
        timeout_seconds=timeout_seconds,
        store=store,
        stem="channel-rayd-post-build-clean-status",
    )
    rayd_post_stdout = rayd_post_status.pop("stdout_bytes")
    rayd_post_status.pop("stderr_bytes")
    if rayd_post_stdout.strip():
        raise EvidenceError("Channel builds polluted the runner-owned RayD source")
    rayd_record["post_build_clean_status"] = rayd_post_status
    comparisons: dict[str, ComparisonConfig] = {}
    bindings: dict[str, str] = {}
    for group in COMPARISON_GROUPS:
        variants: dict[str, VariantConfig] = {}
        for name in ("baseline", "candidate"):
            commit = str(groups[group][f"{name}_commit"])  # type: ignore[index]
            if commit not in built_by_commit:
                raise EvidenceError(f"comparison commit lacks a fresh Channel build: {commit}")
            variants[name] = built_by_commit[commit]
            bindings[f"{group}:{name}"] = role_by_commit[commit]
        comparisons[group] = ComparisonConfig(
            baseline=variants["baseline"], candidate=variants["candidate"]
        )
    runtime = replace(
        build_config,
        comparisons=comparisons,
        runner_build_environment=build_environment,
    )
    return runtime, {
        "root": str(root),
        "unique_commit_count": len(records),
        "rayd_source": rayd_record,
        "msvc_environment": msvc_environment,
        "bindings": bindings,
        "records": records,
    }


def validate_channel_build_records(
    value: object,
    *,
    implementation: dict[str, object],
    store: ArtifactStore,
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != {
        "root", "unique_commit_count", "rayd_source", "msvc_environment", "bindings", "records",
        "compiler_resource_checks",
    }:
        raise EvidenceError("fresh Channel build evidence envelope is malformed")
    records = value["records"]
    bindings = value["bindings"]
    if (
        value["unique_commit_count"] != 5
        or not isinstance(records, dict)
        or set(records) != {"P", "E", "M", "D", "S"}
        or not isinstance(bindings, dict)
    ):
        raise EvidenceError("fresh Channel evidence does not contain P/E/M/D/S")
    groups = implementation.get("groups")
    if not isinstance(groups, dict):
        raise EvidenceError("implementation lacks group commits for build replay")
    expected_commits = {
        role: str(groups[group][f"{name}_commit"])  # type: ignore[index]
        for role, group, name in _BUILD_ROLES
    }
    expected_bindings: dict[str, str] = {}
    for group in COMPARISON_GROUPS:
        for name in ("baseline", "candidate"):
            commit = str(groups[group][f"{name}_commit"])  # type: ignore[index]
            roles = [role for role, role_commit in expected_commits.items() if role_commit == commit]
            if len(roles) != 1:
                raise EvidenceError("comparison commit does not map to one P/E/M/D/S build")
            expected_bindings[f"{group}:{name}"] = roles[0]
    if bindings != expected_bindings:
        raise EvidenceError("comparison variants are not bound to canonical fresh builds")
    tool_identities = implementation.get("tool_executable_identity")
    if not isinstance(tool_identities, dict):
        raise EvidenceError("implementation lacks frozen build tool identities")
    expected_tools = {
        name: str(tool_identities[name]["path"])  # type: ignore[index]
        for name in ("git", "cmake", "ninja", "nvcc", "cl", "link", "cmd", "vcvars64")
    }
    msvc = value["msvc_environment"]
    if not isinstance(msvc, dict) or set(msvc) != {
        "capture", "manifest", "keys", "sha256", "compiler_versions",
    }:
        raise EvidenceError("MSVC environment evidence is malformed")
    if msvc["keys"] != list(_MSVC_ENV_KEYS):
        raise EvidenceError("MSVC environment allowlist differs")
    manifest = msvc["manifest"]
    capture = msvc["capture"]
    if not isinstance(manifest, dict) or not isinstance(capture, dict):
        raise EvidenceError("MSVC environment artifacts are malformed")
    payload = store.read_verified(manifest, label="MSVC environment manifest")
    if manifest.get("sha256") != msvc["sha256"]:
        raise EvidenceError("MSVC environment manifest SHA differs")
    try:
        environment_values = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceError("MSVC environment manifest is not JSON") from exc
    if not isinstance(environment_values, dict) or tuple(environment_values) != tuple(sorted(_MSVC_ENV_KEYS)):
        raise EvidenceError("MSVC environment manifest key set differs")
    argv = capture.get("argv")
    if (
        not isinstance(argv, list)
        or len(argv) != 6
        or argv[:5] != [expected_tools["cmd"], "/d", "/v:on", "/s", "/c"]
        or str(expected_tools["vcvars64"]) not in str(argv[5])
        or capture.get("returncode") != 0
        or capture.get("timed_out") is not False
    ):
        raise EvidenceError("MSVC environment capture command differs")
    stdout = store.read_verified(
        capture["stdout_artifact"], label="MSVC environment allowlist stdout"
    )
    replayed_values = _parse_msvc_environment_stdout(stdout)
    if replayed_values != environment_values:
        raise EvidenceError("MSVC environment stdout differs from manifest")
    canonical_environment = json.dumps(
        replayed_values, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    if payload != canonical_environment or hashlib.sha256(payload).hexdigest() != msvc["sha256"]:
        raise EvidenceError("MSVC environment manifest is not canonical or hash-bound")
    compiler_versions = msvc["compiler_versions"]
    if not isinstance(compiler_versions, dict) or set(compiler_versions) != {"cl", "link", "nvcc"}:
        raise EvidenceError("compiler version evidence is malformed")
    version_patterns = {
        "cl": rb"Compiler Version ([0-9.]+) for x64",
        "link": rb"Linker Version ([0-9.]+)",
        "nvcc": rb"\bV([0-9]+\.[0-9]+\.[0-9]+)\b",
    }
    version_argv = {
        "cl": [expected_tools["cl"], "/Bv"],
        "link": [expected_tools["link"], "/?"],
        "nvcc": [expected_tools["nvcc"], "--version"],
    }
    replayed_versions: dict[str, str] = {}
    for name in ("cl", "link", "nvcc"):
        row = compiler_versions[name]
        if not isinstance(row, dict) or set(row) != {"version", "capture"}:
            raise EvidenceError(f"{name} version evidence is malformed")
        version_capture = row["capture"]
        if (
            not isinstance(version_capture, dict)
            or version_capture.get("argv") != version_argv[name]
            or version_capture.get("timed_out") is not False
        ):
            raise EvidenceError(f"{name} version capture differs")
        output = store.read_verified(
            version_capture["stdout_artifact"], label=f"{name} version stdout"
        ) + b"\n" + store.read_verified(
            version_capture["stderr_artifact"], label=f"{name} version stderr"
        )
        match = re.search(version_patterns[name], output)
        if match is None:
            raise EvidenceError(f"{name} version capture does not replay")
        replayed_versions[name] = match.group(1).decode("ascii", errors="strict")
        if row["version"] != replayed_versions[name]:
            raise EvidenceError(f"{name} reported version differs from its raw capture")
    if (
        not replayed_versions["cl"].startswith("19.")
        or not replayed_versions["link"].startswith(
            "14." + replayed_versions["cl"].removeprefix("19.")
        )
        or not replayed_versions["nvcc"].startswith("12.9.")
    ):
        raise EvidenceError("compiler version probes do not identify the accepted toolchain")
    rayd = value["rayd_source"]
    if not isinstance(rayd, dict) or set(rayd) != {
        "commit", "tree", "initially_absent", "clean", "integration_api_version",
        "integration_identity", "clone", "checkout", "remote", "tree_capture",
        "clean_status", "source_archive_capture", "source_archive",
        "integration_header", "post_build_clean_status",
    }:
        raise EvidenceError("runner-owned RayD source record is malformed")
    if (
        rayd["commit"] != implementation.get("rayd_commit")
        or rayd["commit"] != _RAYD_COMMIT
        or rayd["initially_absent"] is not True
        or rayd["clean"] is not True
        or rayd["integration_api_version"] != 6
        or rayd["integration_identity"] != "rayd.torch.integration"
    ):
        raise EvidenceError("runner-owned RayD source identity is not accepted")
    build_root = Path(str(value["root"]))
    rayd_source = build_root / "rayd-source"
    remote_capture = rayd["remote"]
    assert isinstance(remote_capture, dict)
    if remote_capture.get("argv") != [
        expected_tools["git"], "remote", "set-url", "origin",
        implementation.get("rayd_repository_url"),
    ]:
        raise EvidenceError("runner-owned RayD origin is not bound to the lock")
    for name in (
        "clone", "checkout", "remote", "tree_capture", "clean_status",
        "source_archive_capture", "post_build_clean_status",
    ):
        capture = rayd[name]
        if (
            not isinstance(capture, dict)
            or capture.get("returncode") != 0
            or capture.get("timed_out") is not False
        ):
            raise EvidenceError(f"runner-owned RayD {name} did not pass")
    for name in ("source_archive", "integration_header"):
        store.verify_reference(rayd[name], label=f"runner-owned RayD {name}")
    header = rayd["integration_header"]
    assert isinstance(header, dict)
    if header.get("sha256") != implementation.get("integration_header_sha256"):
        raise EvidenceError("runner-owned RayD header differs from retained lock identity")
    status_capture = rayd["clean_status"]
    assert isinstance(status_capture, dict)
    if store.read_verified(
        status_capture["stdout_artifact"], label="runner-owned RayD clean status"
    ).strip():
        raise EvidenceError("runner-owned RayD retained status is not clean")
    post_rayd_status = rayd["post_build_clean_status"]
    assert isinstance(post_rayd_status, dict)
    if store.read_verified(
        post_rayd_status["stdout_artifact"], label="runner-owned RayD post-build status"
    ).strip():
        raise EvidenceError("runner-owned RayD was polluted by Channel builds")
    for role, record in records.items():
        if not isinstance(record, dict) or set(record) != {
            "role", "commit", "initially_absent", "source", "build", "install",
            "clone", "checkout", "source_tree", "source_tree_capture",
            "source_archive_capture", "source_archive", "configure", "build_capture",
            "install_capture", "cmake_cache_values", "toolchain",
            "installed_python_manifest", "installed_python_manifest_sha256",
            "extension", "fingerprint", "post_build_clean_status",
            "compiler_resource_capture", "compiler_resources",
        }:
            raise EvidenceError(f"fresh Channel {role} build record is malformed")
        if (
            record["role"] != role
            or record["commit"] != expected_commits[role]
            or record["initially_absent"] is not True
        ):
            raise EvidenceError(f"fresh Channel {role} identity differs from history")
        for name in (
            "clone", "checkout", "source_tree_capture", "source_archive_capture",
            "configure", "build_capture", "install_capture",
            "post_build_clean_status",
            "compiler_resource_capture",
        ):
            capture = record[name]
            if (
                not isinstance(capture, dict)
                or capture.get("returncode") != 0
                or capture.get("timed_out") is not False
            ):
                raise EvidenceError(f"fresh Channel {role} {name} did not pass")
        for name in (
            "source_archive", "extension", "fingerprint", "installed_python_manifest",
        ):
            store.verify_reference(record[name], label=f"Channel {role} {name}")
        toolchain = record["toolchain"]
        if not isinstance(toolchain, dict) or set(toolchain) != {
            "cmake_cache", "configure_log", "cxx_compiler", "cuda_compiler",
        }:
            raise EvidenceError(f"fresh Channel {role} toolchain record is incomplete")
        for artifact in toolchain.values():
            store.verify_reference(artifact, label=f"Channel {role} toolchain")
        cxx_facts = store.read_verified(
            toolchain["cxx_compiler"], label=f"Channel {role} CXX compiler facts"
        ).decode("utf-8", errors="strict")
        cuda_facts = store.read_verified(
            toolchain["cuda_compiler"], label=f"Channel {role} CUDA compiler facts"
        ).decode("utf-8", errors="strict")
        cxx_version_match = re.search(
            r'CMAKE_CXX_COMPILER_VERSION "([0-9.]+)"', cxx_facts
        )
        cuda_version_match = re.search(
            r'CMAKE_CUDA_COMPILER_VERSION "([0-9.]+)"', cuda_facts
        )
        cxx_version = cxx_version_match.group(1) if cxx_version_match else ""
        cuda_version = cuda_version_match.group(1) if cuda_version_match else ""
        cl_version = replayed_versions["cl"]
        cxx_version_matches = cxx_version in {cl_version, cl_version + ".0"} or cl_version == cxx_version + ".0"
        if (
            'CMAKE_CXX_COMPILER_ID "MSVC"' not in cxx_facts
            or not cxx_version_matches
            or 'MSVC_CXX_ARCHITECTURE_ID "x64"' not in cxx_facts
            or cuda_version != replayed_versions["nvcc"]
        ):
            raise EvidenceError(f"fresh Channel {role} compiler version facts differ")
        cache_reference = toolchain["cmake_cache"]
        assert isinstance(cache_reference, dict)
        retained_cache = _cache(store.root / str(cache_reference["path"]))
        reported_cache = record["cmake_cache_values"]
        if not isinstance(reported_cache, dict) or any(
            retained_cache.get(name) != value for name, value in reported_cache.items()
        ):
            raise EvidenceError(f"fresh Channel {role} cache summary does not replay")
        source = str(record["source"])
        build = str(record["build"])
        install = str(record["install"])
        role_root = build_root / f"{role}-{record['commit']}"
        if (
            Path(source) != role_root / "source"
            or Path(build) != role_root / "build"
            or Path(install) != role_root / "install" / "site-packages"
        ):
            raise EvidenceError(f"fresh Channel {role} paths escape their isolated role root")
        configure = record["configure"]
        assert isinstance(configure, dict)
        argv = configure.get("argv")
        if (
            not isinstance(argv, list)
            or argv[:7] != [expected_tools["cmake"], "-S", source, "-B", build, "-G", "Ninja"]
            or "-DCMAKE_BUILD_TYPE=Release" not in argv
            or "-DCMAKE_CUDA_ARCHITECTURES=120-real" not in argv
            or f"-DCMAKE_CUDA_COMPILER={expected_tools['nvcc']}" not in argv
            or f"-DCMAKE_CXX_COMPILER={expected_tools['cl']}" not in argv
            or f"-DCMAKE_LINKER={expected_tools['link']}" not in argv
            or f"-DCMAKE_MAKE_PROGRAM={expected_tools['ninja']}" not in argv
            or f"-DCMAKE_INSTALL_PREFIX={install}" not in argv
            or f"-DRAYD_SOURCE_DIR={rayd_source}" not in argv
            or "-DCHANNEL_RELEASE_BUILD=ON" not in argv
            or "-DBUILD_TESTING=OFF" not in argv
        ):
            raise EvidenceError(f"fresh Channel {role} configure argv is not canonical")
        build_capture = record["build_capture"]
        install_capture = record["install_capture"]
        assert isinstance(build_capture, dict) and isinstance(install_capture, dict)
        if build_capture.get("argv") != [
            expected_tools["cmake"], "--build", build, "--parallel", "4",
        ] or install_capture.get("argv") != [
            expected_tools["cmake"], "--install", build, "--prefix", install,
        ]:
            raise EvidenceError(f"fresh Channel {role} build/install argv is not canonical")
        tree = record["source_tree"]
        manifest_sha = record["installed_python_manifest_sha256"]
        if (
            not isinstance(tree, str) or len(tree) != 40
            or not isinstance(manifest_sha, str) or len(manifest_sha) != 64
        ):
            raise EvidenceError(f"fresh Channel {role} source/install digest is malformed")
        tree_capture = record["source_tree_capture"]
        assert isinstance(tree_capture, dict)
        replayed_tree = store.read_verified(
            tree_capture["stdout_artifact"], label=f"Channel {role} source tree stdout"
        ).decode("ascii", errors="strict").strip()
        if replayed_tree != tree:
            raise EvidenceError(f"fresh Channel {role} source tree does not replay")
        manifest = record["installed_python_manifest"]
        assert isinstance(manifest, dict)
        if manifest.get("sha256") != manifest_sha:
            raise EvidenceError(f"fresh Channel {role} install manifest digest differs")
        post_status = record["post_build_clean_status"]
        assert isinstance(post_status, dict)
        if store.read_verified(
            post_status["stdout_artifact"], label=f"Channel {role} post-build status"
        ).strip():
            raise EvidenceError(f"fresh Channel {role} retained post-build status is dirty")
        resource_capture = record["compiler_resource_capture"]
        assert isinstance(resource_capture, dict)
        replayed_resources = parse_cuobjdump_resource_usage(
            store.read_verified(
                resource_capture["stdout_artifact"],
                label=f"Channel {role} compiler resource stdout",
                allow_empty=False,
            )
        )
        if replayed_resources != record["compiler_resources"]:
            raise EvidenceError(f"fresh Channel {role} compiler resources do not replay")
    return dict(value)


def compiler_resource_checks(
    channel_builds: object, gate: Mapping[str, object]
) -> dict[str, dict[str, object]]:
    if not isinstance(channel_builds, dict):
        raise EvidenceError("fresh Channel build evidence is malformed")
    records = channel_builds.get("records")
    bindings = channel_builds.get("bindings")
    if not isinstance(records, dict) or not isinstance(bindings, dict):
        raise EvidenceError("fresh Channel build evidence lacks records/bindings")
    result: dict[str, dict[str, object]] = {}
    fields = ("registers_per_thread", "stack_bytes", "shared_bytes", "local_bytes")
    for group in COMPARISON_GROUPS:
        role = str(bindings[f"{group}:candidate"])
        rows = records[role]["compiler_resources"]  # type: ignore[index]
        if not isinstance(rows, list):
            raise EvidenceError("compiler resource rows are malformed")
        budget = gate["comparison_groups"][group]["resource_budgets"][  # type: ignore[index]
            "compiler_resources"
        ]
        if not isinstance(budget, dict) or len(budget) != 1:
            raise EvidenceError("compiler resource budget must name one target kernel family")
        fragment = next(iter(budget))
        if not isinstance(fragment, str) or not fragment:
            raise EvidenceError("compiler resource target kernel family is malformed")
        matching = [row for row in rows if fragment in str(row.get("function", ""))]
        if not matching:
            raise EvidenceError(f"candidate build lacks compiler resource data for {fragment}")
        observed = {name: max(int(row[name]) for row in matching) for name in fields}
        limits = budget[fragment]
        if not isinstance(limits, dict) or set(limits) != {
            f"{name}_max" for name in fields
        }:
            raise EvidenceError("compiler resource limits are malformed")
        checks = {
            name: value <= int(limits[f"{name}_max"])
            for name, value in observed.items()
        }
        result[group] = {
            "role": role,
            "kernel_fragment": fragment,
            "matched_function_count": len(matching),
            "observed_max": observed,
            "limits": limits,
            "checks": checks,
            "passed": all(checks.values()),
        }
    return result


__all__ = [
    "compiler_resource_checks", "parse_cuobjdump_resource_usage",
    "prepare_fresh_channel_builds", "validate_channel_build_records",
]

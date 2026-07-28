"""Runner-owned release checks from fresh build directories and retained facts."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
import zipfile

from .artifacts import ArtifactStore, reject_reparse_chain
from .contracts import (
    EvidenceError,
    RunnerConfig,
    controlled_environment,
    exact_keys,
    strict_object,
)
from .workers import (
    executable_identity,
    parse_identity_stdout,
    run_captured,
    verify_checkouts,
)


CHANNEL_TIERS = ("quick", "cuda", "nightly", "release")
WHEEL_ARCHITECTURES = (
    "75-real", "80-real", "86-real", "89-real", "120-real", "120-virtual"
)
TORCH_WHEEL_ARCHITECTURES = "7.5 8.0 8.6 8.9 12.0+PTX"
FINAL_GROUP = "diffraction"
WHEEL_SMOKE_NAME = "wheel-smoke-pe-audit.v1.json"
WHEEL_NATIVE_MEMBER = "witwin/channel/_channel.cp311-win_amd64.pyd"
_FINGERPRINT_FIELDS = (
    "build_type",
    "channel_abi_version",
    "channel_git_dirty",
    "channel_git_sha",
    "compiler",
    "cuda_architectures",
    "cuda_compiler_version",
    "cuda_version",
    "cxx_abi",
    "rayd_dirty",
    "rayd_commit",
    "rayd_integration_abi_sha256",
    "rayd_integration_abi_kind",
    "rayd_integration_abi_path",
    "rayd_repository_url",
    "rayd_source_kind",
    "rayd_source_manifest_sha256",
    "torch_version",
)


def _without_payload(capture: dict[str, object]) -> dict[str, object]:
    capture.pop("stdout_bytes", None)
    capture.pop("stderr_bytes", None)
    return capture


def _capture(
    config: RunnerConfig,
    store: ArtifactStore,
    *,
    argv: list[str],
    cwd: Path,
    stem: str,
    timeout_seconds: int,
    environment: Mapping[str, str] | None = None,
) -> dict[str, object]:
    return _without_payload(
        run_captured(
            argv,
            cwd=cwd,
            environment=dict(
                environment
                or config.runner_build_environment
                or controlled_environment(config)
            ),
            timeout_seconds=timeout_seconds,
            store=store,
            stem=stem,
        )
    )


def _conda_python(config: RunnerConfig, *arguments: str) -> list[str]:
    return [str(config.tools.conda), "run", "-n", "witwin2", "python", *arguments]


def _rayd_python(config: RunnerConfig, *arguments: str) -> list[str]:
    return [str(config.tools.conda), "run", "-n", "witwin2", "python", *arguments]


def _quoted_cmake_definition(name: str, value: object) -> str:
    rendered = str(value).replace("\\", "/")
    if '"' in rendered or "\n" in rendered or "\r" in rendered:
        raise EvidenceError(f"unsafe CMake definition value for {name}")
    return f'-D{name}="{rendered}"'


def _prepare_packaged_validation_checkout(
    config: RunnerConfig,
    store: ArtifactStore,
    *,
    commit: str,
    timeout_seconds: int,
) -> dict[str, object]:
    candidate = config.variant(FINAL_GROUP, "candidate")
    if candidate.runner_extension is None or candidate.runner_site_packages is None:
        raise EvidenceError("release evidence requires the runner-built final installation")
    try:
        site_packages = candidate.runner_site_packages.resolve(strict=True)
        source_extension = candidate.runner_extension.resolve(strict=True)
    except OSError as exc:
        raise EvidenceError("runner release binding path is missing") from exc
    installed_package = site_packages / "witwin" / "channel"
    if not source_extension.is_file() or source_extension.parent != installed_package:
        raise EvidenceError("runner extension is not inside its runner-owned installation")
    source_fingerprint = (
        installed_package / "_channel.build-fingerprint"
    )
    validation = config.build_parent / f"release-validation-{commit}"
    if os.path.lexists(validation):
        raise EvidenceError("release validation checkout must be initially absent")
    validation.parent.mkdir(parents=True, exist_ok=True)
    clone = _capture(
        config,
        store,
        argv=[
            str(config.tools.git),
            "clone",
            "--no-hardlinks",
            "--no-checkout",
            str(candidate.checkout),
            str(validation),
        ],
        cwd=validation.parent,
        stem="release-validation-clone",
        timeout_seconds=timeout_seconds,
    )
    checkout = _capture(
        config,
        store,
        argv=[str(config.tools.git), "checkout", "--detach", commit],
        cwd=validation,
        stem="release-validation-checkout",
        timeout_seconds=timeout_seconds,
    )
    validation_package = validation / "src" / "witwin" / "channel"
    validation_extension = validation_package / source_extension.name
    validation_fingerprint = (
        validation_package / "_channel.build-fingerprint"
    )
    if validation_extension.exists() or validation_fingerprint.exists():
        raise EvidenceError("packaged validation overlay target already exists")
    git_exclude = validation / ".git" / "info" / "exclude"
    exclude_line = "/src/witwin/channel/_channel.build-fingerprint"
    try:
        existing_excludes = git_exclude.read_text(encoding="utf-8")
        separator = "" if not existing_excludes or existing_excludes.endswith("\n") else "\n"
        git_exclude.write_text(
            existing_excludes + separator + exclude_line + "\n", encoding="utf-8"
        )
        shutil.copyfile(source_extension, validation_extension)
        shutil.copyfile(source_fingerprint, validation_fingerprint)
        fingerprint_raw = validation_fingerprint.read_text(encoding="ascii")
    except (OSError, UnicodeError) as exc:
        raise EvidenceError("runner build fingerprint is unreadable") from exc
    match = re.fullmatch(r"([0-9a-f]{64})(?:\r?\n)?", fingerprint_raw)
    if match is None:
        raise EvidenceError("runner build fingerprint is not one canonical SHA-256")
    fingerprint = match.group(1)
    source_artifact = store.retain_external(
        source_extension,
        f"release/validation-binding/source-{source_extension.name}",
        label="runner-owned source extension",
    )
    packaged_artifact = store.retain_external(
        validation_extension,
        f"release/validation-binding/packaged-{validation_extension.name}",
        label="validation packaged extension",
    )
    fingerprint_artifact = store.retain_external(
        validation_fingerprint,
        "release/validation-binding/_channel.build-fingerprint",
        label="validation packaged fingerprint",
    )
    if (
        source_artifact["sha256"] != packaged_artifact["sha256"]
        or source_artifact["bytes"] != packaged_artifact["bytes"]
    ):
        raise EvidenceError("packaged validation extension differs from runner-built bytes")
    status = run_captured(
        [
            str(config.tools.git),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        cwd=validation,
        environment=dict(
            config.runner_build_environment or controlled_environment(config)
        ),
        timeout_seconds=timeout_seconds,
        store=store,
        stem="release-validation-clean-status",
    )
    status_stdout = status.pop("stdout_bytes")
    status.pop("stderr_bytes")
    if status_stdout.strip():
        raise EvidenceError("packaged validation overlay is not Git-clean/ignored")
    identity_capture = run_captured(
        [
            str(candidate.python_executable),
            "-I",
            str(validation / "benchmarks" / "phase13_phase12_bootstrap.py"),
            "--site-packages",
            str(validation / "src"),
            "--script",
            str(validation / "benchmarks" / "phase13_phase12_identity_probe.py"),
        ],
        cwd=validation,
        environment=controlled_environment(config),
        timeout_seconds=timeout_seconds,
        store=store,
        stem="release-validation-packaged-identity",
    )
    identity_record = parse_identity_stdout(identity_capture.pop("stdout_bytes"))
    identity_capture.pop("stderr_bytes")
    identity_build = identity_record.get("build_info")
    if (
        not isinstance(identity_build, dict)
        or identity_build.get("build_fingerprint") != fingerprint
        or Path(str(identity_record.get("extension_path"))).resolve()
        != validation_extension.resolve()
    ):
        raise EvidenceError("validation identity probe did not load the packaged runner build")
    return {
        "checkout": str(validation.resolve()),
        "commit": commit,
        "clone": clone,
        "detached_checkout": checkout,
        "clean_status": status,
        "packaged_identity_capture": identity_capture,
        "packaged_identity_record": identity_record,
        "site_packages_source": str(site_packages),
        "source_extension": source_artifact,
        "packaged_extension": packaged_artifact,
        "packaged_extension_path": str(validation_extension.resolve()),
        "packaged_fingerprint": fingerprint_artifact,
        "build_fingerprint": fingerprint,
        "local_exclude": exclude_line,
    }


def _runner_channel_environment(
    config: RunnerConfig, validation: Mapping[str, object]
) -> tuple[dict[str, str], dict[str, object]]:
    candidate = config.variant(FINAL_GROUP, "candidate")
    try:
        checkout = Path(str(validation["checkout"])).resolve(strict=True)
        source_site_packages = Path(
            str(validation["site_packages_source"])
        ).resolve(strict=True)
        extension = Path(str(validation["packaged_extension_path"])).resolve(
            strict=True
        )
        rayd_source = config.rayd_checkout.resolve(strict=True)
        python_executable = candidate.python_executable.resolve(strict=True)
    except (KeyError, OSError) as exc:
        raise EvidenceError("runner packaged validation binding is missing") from exc
    source_root = checkout / "src"
    expected_extension_parent = source_root / "witwin" / "channel"
    if extension.parent != expected_extension_parent:
        raise EvidenceError("validation extension is not in the packaged source layout")
    fingerprint = str(validation.get("build_fingerprint", ""))
    if re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None:
        raise EvidenceError("validation packaged fingerprint is invalid")
    cmake_definitions = (
        _quoted_cmake_definition("RAYD_SOURCE_DIR", rayd_source),
        _quoted_cmake_definition(
            "CMAKE_CUDA_ARCHITECTURES", ";".join(WHEEL_ARCHITECTURES)
        ),
        _quoted_cmake_definition("CMAKE_BUILD_TYPE", "Release"),
        _quoted_cmake_definition("CHANNEL_RELEASE_BUILD", "ON"),
        _quoted_cmake_definition("Python_EXECUTABLE", python_executable),
        _quoted_cmake_definition("CMAKE_CUDA_COMPILER", config.tools.nvcc),
        _quoted_cmake_definition("CMAKE_CXX_COMPILER", config.tools.cl),
        _quoted_cmake_definition("CMAKE_LINKER", config.tools.link),
        _quoted_cmake_definition("CMAKE_MAKE_PROGRAM", config.tools.ninja),
    )
    cmake_args = " ".join(cmake_definitions)
    environment = dict(
        config.runner_build_environment or controlled_environment(config)
    )
    forbidden_loader_environment = (
        "WITWIN_CHANNEL_DEVELOPER_OVERRIDE",
        "WITWIN_CHANNEL_EXTENSION_PATH",
        "WITWIN_CHANNEL_EXPECTED_FINGERPRINT",
    )
    if any(name in environment for name in forbidden_loader_environment):
        raise EvidenceError("release environment must not enable the developer loader")
    environment.update(
        {
            "PYTHONPATH": str(source_root),
            "CMAKE_ARGS": cmake_args,
            "CMAKE_GENERATOR": "Ninja",
            "CMAKE_BUILD_PARALLEL_LEVEL": "4",
            "TORCH_CUDA_ARCH_LIST": TORCH_WHEEL_ARCHITECTURES,
        }
    )
    binding = {
        "mode": "runner-owned-packaged-validation-checkout",
        "validation_checkout": str(checkout),
        "site_packages_source": str(source_site_packages),
        "extension_path": str(extension),
        "build_fingerprint": fingerprint,
        "rayd_source": str(rayd_source),
        "python_executable": str(python_executable),
        "cmake_args": cmake_args,
        "cuda_architectures": list(WHEEL_ARCHITECTURES),
        "torch_cuda_arch_list": TORCH_WHEEL_ARCHITECTURES,
    }
    return environment, binding


def _run_channel_tier(
    config: RunnerConfig,
    tier: str,
    *,
    validation: Mapping[str, object],
    timeout_seconds: int,
    store: ArtifactStore,
) -> dict[str, object]:
    candidate = config.variant(FINAL_GROUP, "candidate")
    if candidate.runner_extension is None or candidate.runner_site_packages is None:
        raise EvidenceError("release evidence requires the runner-built final installation")
    environment, binding = _runner_channel_environment(config, validation)
    validation_checkout = Path(str(validation["checkout"])).resolve(strict=True)
    capture = _capture(
        config,
        store,
        argv=_conda_python(
            config,
            "ci/run_ci_tier.py",
            tier,
            "--python",
            str(candidate.python_executable),
            "--root",
            str(validation_checkout),
        ),
        cwd=validation_checkout,
        stem=f"release-channel-{tier}",
        timeout_seconds=timeout_seconds,
        environment=environment,
    )
    capture["runner_binding"] = binding
    return capture


def _fresh_rayd_build(
    config: RunnerConfig, *, timeout_seconds: int, store: ArtifactStore
) -> tuple[Path, dict[str, dict[str, object]]]:
    build_dir = config.rayd_checkout / "backends" / "torch" / "build" / "local-120"
    if os.path.lexists(build_dir):
        raise EvidenceError(
            "formal RayD build requires absent backends/torch/build/local-120; "
            "the runner never deletes or reuses a build tree"
        )
    script = config.rayd_checkout / "scripts" / "build_local.ps1"
    build = _capture(
        config,
        store,
        argv=[
            str(config.tools.conda), "run", "-n", "witwin2",
            str(config.tools.powershell), "-NoProfile", "-ExecutionPolicy", "Bypass",
            "-File", str(script), "-Backend", "torch", "-PythonExe", "python",
            "-CudaArch", "120",
        ],
        cwd=config.rayd_checkout,
        stem="release-rayd-fresh-build",
        timeout_seconds=timeout_seconds,
    )
    if not build_dir.is_dir():
        raise EvidenceError("canonical RayD build did not create its fresh SM120 tree")
    return build_dir, {"rayd_fresh_build": build}


def _run_rayd_checks(
    config: RunnerConfig,
    build_dir: Path,
    *,
    timeout_seconds: int,
    store: ArtifactStore,
) -> dict[str, dict[str, object]]:
    direct = _capture(
        config,
        store,
        argv=_rayd_python(
            config, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"
        ),
        cwd=config.rayd_checkout,
        stem="release-rayd-direct",
        timeout_seconds=timeout_seconds,
    )
    ctest = _capture(
        config,
        store,
        argv=[
            str(config.tools.conda), "run", "-n", "witwin2",
            str(config.tools.ctest), "--test-dir", str(build_dir),
            "--build-config", "Release", "--output-on-failure", "--no-tests=error",
        ],
        cwd=config.rayd_checkout,
        stem="release-rayd-ctest",
        timeout_seconds=timeout_seconds,
    )
    return {"rayd_direct": direct, "rayd_ctest": ctest}


def _fresh_release_files(checkout: Path) -> tuple[Path, Path]:
    release_dir = checkout / "artifacts" / "release"
    reject_reparse_chain(release_dir, stop=checkout)
    wheels = sorted(release_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise EvidenceError(f"release tier must create exactly one wheel; found {len(wheels)}")
    smoke = release_dir / WHEEL_SMOKE_NAME
    if not smoke.is_file():
        raise EvidenceError("release tier did not create the stable wheel-smoke artifact")
    return wheels[0], smoke


def _load_retained_json(
    store: ArtifactStore, reference: Mapping[str, object], *, label: str
) -> dict[str, object]:
    try:
        payload = json.loads(
            store.read_verified(reference, label=label).decode("utf-8"),
            object_pairs_hook=strict_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                EvidenceError(f"non-finite JSON constant in {label}: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"retained {label} is not strict JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise EvidenceError(f"retained {label} must be an object")
    return payload


def _build_fingerprint(info: Mapping[str, object]) -> str:
    missing = [name for name in _FINGERPRINT_FIELDS if name not in info]
    if missing:
        raise EvidenceError("wheel build identity lacks fingerprint fields: " + ", ".join(missing))
    payload = {name: info[name] for name in _FINGERPRINT_FIELDS}
    try:
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise EvidenceError("wheel build identity cannot be fingerprinted") from exc
    return hashlib.sha256(canonical).hexdigest()


def _validate_wheel_smoke(
    smoke: Mapping[str, object],
    *,
    implementation: Mapping[str, object],
    wheel_sha256: str,
) -> dict[str, object]:
    exact_keys(
        smoke,
        {
            "wheel_smoke", "wheel_sha256", "distribution", "package_origin",
            "native_origin", "build_info", "pe_audit",
        },
        label="wheel smoke",
    )
    if smoke["wheel_smoke"] is not True or smoke["wheel_sha256"] != wheel_sha256:
        raise EvidenceError("wheel smoke is not bound to the retained wheel")
    info = smoke["build_info"]
    pe_audit = smoke["pe_audit"]
    if not isinstance(info, dict) or not isinstance(pe_audit, dict) or not pe_audit:
        raise EvidenceError("wheel smoke lacks build identity or a successful PE audit")
    final_history = implementation["groups"][FINAL_GROUP]  # type: ignore[index]
    assert isinstance(final_history, dict)
    expected = {
        "build_type": "Release",
        "channel_git_dirty": False,
        "channel_git_sha": final_history["candidate_commit"],
        "rayd_dirty": False,
        "rayd_commit": implementation["rayd_commit"],
        "rayd_integration_abi_kind": "source-header-sha256",
        "rayd_integration_abi_path": "backends/torch/include/rayd/torch/integration.h",
        "rayd_integration_abi_sha256": implementation["integration_header_sha256"],
        "rayd_repository_url": implementation["rayd_repository_url"],
        "cuda_architectures": list(WHEEL_ARCHITECTURES),
    }
    mismatched = [key for key, value in expected.items() if info.get(key) != value]
    if mismatched:
        raise EvidenceError("wheel build identity differs: " + ", ".join(mismatched))
    wheel_fingerprint = _build_fingerprint(info)
    if info.get("build_fingerprint") != wheel_fingerprint:
        raise EvidenceError("wheel build fingerprint is not derived from its multiarch identity")
    if wheel_fingerprint == implementation.get("final_build_fingerprint"):
        raise EvidenceError("multiarch wheel unexpectedly shares the SM120 timing fingerprint")
    return {"build_info": info, "pe_audit": pe_audit}


def _retain_rayd_build_facts(
    config: RunnerConfig,
    store: ArtifactStore,
    build_dir: Path,
    *,
    implementation: Mapping[str, object],
) -> dict[str, object]:
    cache = store.retain_external(
        build_dir / "CMakeCache.txt",
        "release/rayd-facts/CMakeCache.txt",
        label="fresh RayD CMake cache",
    )
    toolchain_files: dict[str, dict[str, object]] = {}
    for name in ("CMakeCUDACompiler.cmake", "CMakeCXXCompiler.cmake"):
        matches = list(build_dir.glob(f"CMakeFiles/*/{name}"))
        if len(matches) != 1:
            raise EvidenceError(f"fresh RayD build must expose exactly one {name}")
        toolchain_files[name] = store.retain_external(
            matches[0], f"release/rayd-facts/{name}", label=f"RayD {name}"
        )
    source = store.retain_external(
        config.rayd_checkout / "backends" / "torch" / "CMakeLists.txt",
        "release/rayd-facts/torch-CMakeLists.txt",
        label="RayD Torch CMake source",
    )
    build_script = store.retain_external(
        config.rayd_checkout / "scripts" / "build_local.ps1",
        "release/rayd-facts/build_local.ps1",
        label="RayD canonical build script",
    )
    outputs: dict[str, dict[str, object]] = {}
    candidates = sorted(
        path for path in build_dir.rglob("*")
        if path.is_file()
        and path.suffix.casefold() in {".pyd", ".dll", ".lib"}
        and "rayd" in path.name.casefold()
    )
    if not candidates:
        raise EvidenceError("fresh RayD build produced no named native artifact")
    for index, path in enumerate(candidates):
        outputs[f"{index:02d}:{path.name}"] = store.retain_external(
            path,
            f"release/rayd-facts/native/{index:02d}-{path.name}",
            label="fresh RayD native artifact",
        )
    facts = {
        "cache": cache,
        "toolchains": toolchain_files,
        "source": source,
        "build_script": build_script,
        "outputs": outputs,
    }
    _validate_rayd_build_facts(
        facts,
        store=store,
        rayd_checkout=config.rayd_checkout,
        implementation=implementation,
    )
    return facts


def _cache_values(payload: bytes) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in payload.decode("utf-8", errors="strict").splitlines():
        if not line or line.startswith(("#", "//")) or "=" not in line:
            continue
        key_type, value = line.split("=", 1)
        key = key_type.split(":", 1)[0]
        if key in values:
            raise EvidenceError(f"duplicate RayD CMake cache key: {key}")
        values[key] = value
    return values


def _validate_rayd_build_facts(
    facts: Mapping[str, object],
    *,
    store: ArtifactStore,
    rayd_checkout: Path | None,
    implementation: Mapping[str, object],
) -> None:
    exact_keys(
        facts,
        {"cache", "toolchains", "source", "build_script", "outputs"},
        label="RayD build facts",
    )
    cache_ref = facts["cache"]
    if not isinstance(cache_ref, dict):
        raise EvidenceError("RayD build cache reference is malformed")
    values = _cache_values(store.read_verified(cache_ref, label="RayD CMake cache"))
    required = {
        "CMAKE_BUILD_TYPE": "Release",
        "CMAKE_GENERATOR": "Ninja",
        "CMAKE_CUDA_ARCHITECTURES": "120",
    }
    mismatched = [name for name, expected in required.items() if values.get(name) != expected]
    if mismatched:
        raise EvidenceError("fresh RayD CMake cache mismatch: " + ", ".join(mismatched))
    home = values.get("CMAKE_HOME_DIRECTORY")
    expected_home = (
        str((rayd_checkout / "backends" / "torch").resolve())
        if rayd_checkout is not None
        else None
    )
    if not isinstance(home, str) or (
        expected_home is not None and home != expected_home
    ) or not Path(home).as_posix().casefold().endswith("/backends/torch"):
        raise EvidenceError("fresh RayD cache source is not the verified Torch backend")
    cuda_compiler = values.get("CMAKE_CUDA_COMPILER", "").replace("\\", "/").casefold()
    cxx_compiler = values.get("CMAKE_CXX_COMPILER", "").replace("\\", "/").casefold()
    tool_identities = implementation.get("tool_executable_identity")
    if not isinstance(tool_identities, dict):
        raise EvidenceError("measured implementation lacks explicit tool identities")
    configured_nvcc = str(tool_identities["nvcc"]["path"]).replace("\\", "/").casefold()  # type: ignore[index]
    configured_cl = str(tool_identities["cl"]["path"]).replace("\\", "/").casefold()  # type: ignore[index]
    configured_link = Path(str(tool_identities["link"]["path"])).resolve()  # type: ignore[index]
    python_executable = next(
        (
            values[name]
            for name in ("Python_EXECUTABLE", "Python3_EXECUTABLE")
            if values.get(name)
        ),
        "",
    ).replace("\\", "/").casefold()
    torch_dir = values.get("Torch_DIR", "").replace("\\", "/").casefold()
    if "/cuda/v12.9/" not in cuda_compiler or not cuda_compiler.endswith("/nvcc.exe"):
        raise EvidenceError("fresh RayD cache CUDA compiler is not CUDA 12.9 nvcc")
    if "/microsoft visual studio/2022/" not in cxx_compiler or not cxx_compiler.endswith("/cl.exe"):
        raise EvidenceError("fresh RayD cache CXX compiler is not Visual Studio 2022 cl.exe")
    if cuda_compiler != configured_nvcc or cxx_compiler != configured_cl:
        raise EvidenceError("fresh RayD cache compiler paths differ from frozen tool bytes")
    if configured_link.parent != Path(str(tool_identities["cl"]["path"])).resolve().parent:  # type: ignore[index]
        raise EvidenceError("frozen link.exe and cl.exe do not share the VS2022 x64 toolchain")
    if "/envs/witwin2/" not in python_executable or not python_executable.endswith("/python.exe"):
        raise EvidenceError("fresh RayD cache Python executable is not witwin2")
    if "/envs/witwin2/" not in torch_dir or not torch_dir.endswith("/torch/share/cmake/torch"):
        raise EvidenceError("fresh RayD cache Torch package is not the witwin2 package")
    toolchains = facts["toolchains"]
    if not isinstance(toolchains, dict) or set(toolchains) != {
        "CMakeCUDACompiler.cmake", "CMakeCXXCompiler.cmake"
    }:
        raise EvidenceError("fresh RayD toolchain facts are incomplete")
    cuda = store.read_verified(toolchains["CMakeCUDACompiler.cmake"], label="RayD CUDA toolchain")
    cxx = store.read_verified(toolchains["CMakeCXXCompiler.cmake"], label="RayD CXX toolchain")
    if (
        b'CMAKE_CUDA_COMPILER_VERSION "12.9' not in cuda
        or b'CMAKE_CXX_COMPILER_ID "MSVC"' not in cxx
        or b'CMAKE_CXX_COMPILER_VERSION "19.' not in cxx
        or b'MSVC_CXX_ARCHITECTURE_ID "x64"' not in cxx
    ):
        raise EvidenceError("fresh RayD build did not use CUDA 12.9 and VS2022 x64")
    source = store.verify_reference(facts["source"], label="RayD Torch CMake source")
    if source["sha256"] != implementation.get("rayd_cmake_source_sha256"):
        raise EvidenceError("retained RayD CMake source differs from the verified HEAD blob")
    build_script = store.verify_reference(
        facts["build_script"], label="RayD canonical build script"
    )
    if build_script["sha256"] != implementation.get("rayd_build_script_sha256"):
        raise EvidenceError("retained RayD build script differs from the verified HEAD blob")
    outputs = facts["outputs"]
    if not isinstance(outputs, dict) or not outputs:
        raise EvidenceError("fresh RayD build output facts are absent")
    for reference in outputs.values():
        store.verify_reference(reference, label="fresh RayD native artifact")


def _independent_wheel_audit(
    config: RunnerConfig,
    store: ArtifactStore,
    wheel: Mapping[str, object],
    *,
    timeout_seconds: int,
) -> dict[str, object]:
    candidate = config.variant(FINAL_GROUP, "candidate")
    wheel_path = store.root / str(wheel["path"])
    output = store.root / "release" / "independent-wheel-audit.json"
    capture = _capture(
        config,
        store,
        argv=[
            str(candidate.python_executable), "-I", str(candidate.checkout / "ci" / "wheel_smoke.py"),
            str(wheel_path), "--dumpbin", str(config.tools.dumpbin), "--output", str(output),
        ],
        cwd=candidate.checkout,
        stem="release-independent-wheel-audit",
        timeout_seconds=timeout_seconds,
    )
    artifact = store.inspect(
        "release/independent-wheel-audit.json", label="independent wheel audit"
    )
    payload = _load_retained_json(store, artifact, label="independent wheel audit")
    return {"capture": capture, "artifact": artifact, "payload": payload}


def _retain_wheel_native(
    store: ArtifactStore, wheel: Mapping[str, object]
) -> dict[str, object]:
    wheel_path = store.root / str(wheel["path"])
    store.verify_reference(wheel, label="release wheel")
    try:
        with zipfile.ZipFile(wheel_path) as archive:
            matching = [
                info
                for info in archive.infolist()
                if info.filename.casefold() == WHEEL_NATIVE_MEMBER.casefold()
            ]
            if (
                len(matching) != 1
                or matching[0].filename != WHEEL_NATIVE_MEMBER
                or matching[0].is_dir()
                or matching[0].flag_bits & 0x1
            ):
                raise EvidenceError(
                    "wheel must contain one canonical, unencrypted native member"
                )
            payload = archive.read(matching[0])
    except EvidenceError:
        raise
    except (OSError, KeyError, RuntimeError, zipfile.BadZipFile) as exc:
        raise EvidenceError("cannot safely retain the wheel native member") from exc
    artifact = store.write_bytes(
        f"release/wheel-native/{Path(WHEEL_NATIVE_MEMBER).name}",
        payload,
        allow_empty=False,
    )
    return {"member": WHEEL_NATIVE_MEMBER, "artifact": artifact}


def _validate_extracted_pe_audit(
    payload: Mapping[str, object],
    *,
    native: Mapping[str, object],
    wheel_pe_audit: Mapping[str, object],
    store: ArtifactStore,
) -> dict[str, object]:
    exact_keys(
        payload,
        {
            "schema_version",
            "path",
            "sha256",
            "dependencies",
            "export_count",
            "exports_sha256",
            "python_init_export",
        },
        label="extracted wheel PE audit",
    )
    native_artifact = native.get("artifact")
    if not isinstance(native_artifact, dict):
        raise EvidenceError("retained wheel native artifact is malformed")
    store.verify_reference(native_artifact, label="retained wheel native extension")
    expected_path = (store.root / str(native_artifact["path"])).resolve()
    try:
        audited_path = Path(str(payload["path"])).resolve(strict=True)
    except OSError as exc:
        raise EvidenceError("extracted wheel PE audit path is missing") from exc
    if audited_path != expected_path or payload["sha256"] != native_artifact["sha256"]:
        raise EvidenceError("PE audit did not consume the retained wheel native extension")
    normalized = {name: value for name, value in payload.items() if name != "path"}
    normalized["wheel_member"] = native.get("member")
    if normalized != wheel_pe_audit:
        raise EvidenceError("direct and wheel-smoke PE audits disagree")
    return normalized


def _independent_wheel_pe_audit(
    config: RunnerConfig,
    store: ArtifactStore,
    native: Mapping[str, object],
    *,
    wheel_pe_audit: Mapping[str, object],
    timeout_seconds: int,
) -> dict[str, object]:
    native_artifact = native.get("artifact")
    if not isinstance(native_artifact, dict):
        raise EvidenceError("retained wheel native artifact is malformed")
    native_path = store.root / str(native_artifact["path"])
    output = store.root / "release" / "independent-wheel-pe-audit.json"
    candidate = config.variant(FINAL_GROUP, "candidate")
    capture = _capture(
        config,
        store,
        argv=[
            str(candidate.python_executable),
            "-I",
            str(candidate.checkout / "ci" / "audit_windows_pe.py"),
            str(native_path),
            "--dumpbin",
            str(config.tools.dumpbin),
            "--output",
            str(output),
        ],
        cwd=candidate.checkout,
        stem="release-independent-wheel-pe-audit",
        timeout_seconds=timeout_seconds,
    )
    artifact = store.inspect(
        "release/independent-wheel-pe-audit.json",
        label="independent wheel PE audit",
    )
    payload = _load_retained_json(store, artifact, label="independent wheel PE audit")
    normalized = _validate_extracted_pe_audit(
        payload,
        native=native,
        wheel_pe_audit=wheel_pe_audit,
        store=store,
    )
    return {
        "capture": capture,
        "artifact": artifact,
        "payload": payload,
        "normalized": normalized,
    }


def _independent_cuda_arch_audit(
    config: RunnerConfig,
    store: ArtifactStore,
    extension_path: Path,
    *,
    timeout_seconds: int,
) -> dict[str, object]:
    capture = _capture(
        config,
        store,
        argv=[str(config.tools.cuobjdump), "--list-elf", "--list-ptx", str(extension_path)],
        cwd=config.variant(FINAL_GROUP, "candidate").checkout,
        stem="release-cuobjdump-architectures",
        timeout_seconds=timeout_seconds,
    )
    stdout = store.read_verified(
        capture["stdout_artifact"], label="cuobjdump stdout", allow_empty=False
    ).decode("utf-8", errors="replace")
    real = sorted({int(value) for value in re.findall(r"(?i)sm_([0-9]+)", stdout)})
    virtual = sorted({int(value) for value in re.findall(r"(?i)compute_([0-9]+)", stdout)})
    if real != [75, 80, 86, 89, 120] or virtual != [120]:
        raise EvidenceError(
            f"independent cubin/PTX architecture audit differs: real={real}, virtual={virtual}"
        )
    return {"capture": capture, "real": real, "virtual": virtual}


def _naming_audit(checkout: Path) -> dict[str, object]:
    suffixes = {".py", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".json", ".cmake"}
    forbidden = re.compile(
        r"(?i)(?:integration_v[0-9]+|rayd\.torch\.integration\.v[0-9]+|"
        r"(?:^|\W)(?:wip|next_generation|provisional_integration)(?:$|\W)|"
        r"(?:^|\W)[A-Za-z][A-Za-z0-9_]*_v2(?:$|\W)|"
        r"(?:^|\W)v2_[A-Za-z][A-Za-z0-9_]*(?:$|\W))"
    )
    protocol_exemptions = (re.compile(r"(?i)porcelain=v1\b"),)
    production_roots = ("src/", "native/", "cmake/", "ci/")
    offenders: list[str] = []
    for path in checkout.rglob("*"):
        if not path.is_file() or path.suffix.casefold() not in suffixes:
            continue
        relative = path.relative_to(checkout).as_posix()
        if not (relative.startswith(production_roots) or relative == "CMakeLists.txt"):
            continue
        if forbidden.search(relative.replace("-", "_")):
            offenders.append(relative)
            continue
        try:
            folded = path.read_text(encoding="utf-8", errors="ignore").casefold()
        except OSError as exc:
            raise EvidenceError(f"cannot scan naming policy in {path}: {exc}") from exc
        for exemption in protocol_exemptions:
            folded = exemption.sub("", folded)
        if forbidden.search(folded):
            offenders.append(relative)
    if offenders:
        raise EvidenceError(f"provisional generation naming remains: {sorted(set(offenders))}")
    return {"scanned_suffixes": sorted(suffixes), "passed": True}


def run_release_evidence(
    config: RunnerConfig,
    gate: Mapping[str, object],
    *,
    implementation: Mapping[str, object],
    timeout_seconds: int,
    store: ArtifactStore,
) -> dict[str, object]:
    candidate = config.variant(FINAL_GROUP, "candidate")
    if candidate.runner_extension is None or candidate.runner_site_packages is None:
        raise EvidenceError("release evidence requires the runner-built final installation")
    final_history = implementation["groups"][FINAL_GROUP]  # type: ignore[index]
    assert isinstance(final_history, dict)
    validation = _prepare_packaged_validation_checkout(
        config,
        store,
        commit=str(final_history["candidate_commit"]),
        timeout_seconds=timeout_seconds,
    )
    validation_checkout = Path(str(validation["checkout"])).resolve(strict=True)
    release_dir = validation_checkout / "artifacts" / "release"
    if os.path.lexists(release_dir):
        raise EvidenceError("release artifact directory must not exist before formal release checks")
    started_ns = time.time_ns()
    captures = {
        f"channel_{tier}": _run_channel_tier(
            config,
            tier,
            validation=validation,
            timeout_seconds=timeout_seconds,
            store=store,
        )
        for tier in CHANNEL_TIERS
    }
    rayd_build_dir, rayd_build_captures = _fresh_rayd_build(
        config, timeout_seconds=timeout_seconds, store=store
    )
    captures.update(rayd_build_captures)
    captures.update(
        _run_rayd_checks(
            config, rayd_build_dir, timeout_seconds=timeout_seconds, store=store
        )
    )
    validation_status = run_captured(
        [
            str(config.tools.git),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        cwd=validation_checkout,
        environment=dict(
            config.runner_build_environment or controlled_environment(config)
        ),
        timeout_seconds=timeout_seconds,
        store=store,
        stem="release-validation-post-release-clean-status",
    )
    validation_status_stdout = validation_status.pop("stdout_bytes")
    validation_status.pop("stderr_bytes")
    if validation_status_stdout.strip():
        raise EvidenceError("release tiers polluted their packaged validation checkout")
    validation["post_release_clean_status"] = validation_status
    post = verify_checkouts(config, gate)
    if post["groups"] != implementation["groups"] or post["rayd_commit"] != implementation["rayd_commit"]:
        raise EvidenceError("checkout identity changed during release validation")

    rayd_facts = _retain_rayd_build_facts(
        config, store, rayd_build_dir, implementation=implementation
    )
    wheel_source, smoke_source = _fresh_release_files(validation_checkout)
    wheel = store.retain_external(
        wheel_source,
        f"release/{wheel_source.name}",
        label="release wheel",
        minimum_mtime_ns=started_ns,
    )
    smoke = store.retain_external(
        smoke_source,
        f"release/{WHEEL_SMOKE_NAME}",
        label="wheel smoke",
        minimum_mtime_ns=started_ns,
    )
    extension = store.retain_external(
        candidate.runner_extension,
        f"release/{candidate.runner_extension.name}",
        label="final native extension",
    )
    if extension["sha256"] != implementation.get("final_extension_sha256"):
        raise EvidenceError("release native extension differs from the measured final candidate")
    supplemental_validation = _validate_wheel_smoke(
        _load_retained_json(store, smoke, label="wheel smoke"),
        implementation=implementation,
        wheel_sha256=str(wheel["sha256"]),
    )
    independent = _independent_wheel_audit(
        config, store, wheel, timeout_seconds=timeout_seconds
    )
    wheel_validation = _validate_wheel_smoke(
        independent["payload"],  # type: ignore[arg-type]
        implementation=implementation,
        wheel_sha256=str(wheel["sha256"]),
    )
    if wheel_validation != supplemental_validation:
        raise EvidenceError("release-tier smoke and independent wheel audit disagree")
    wheel_native = _retain_wheel_native(store, wheel)
    wheel_pe_audit = _independent_wheel_pe_audit(
        config,
        store,
        wheel_native,
        wheel_pe_audit=wheel_validation["pe_audit"],  # type: ignore[arg-type]
        timeout_seconds=timeout_seconds,
    )
    wheel_native_artifact = wheel_native["artifact"]
    assert isinstance(wheel_native_artifact, dict)
    if wheel_native_artifact["sha256"] == extension["sha256"]:
        raise EvidenceError(
            "multiarch wheel retained the SM120 validation overlay instead of a new build"
        )
    cuda_arch_audit = _independent_cuda_arch_audit(
        config,
        store,
        store.root / str(wheel_native_artifact["path"]),
        timeout_seconds=timeout_seconds,
    )
    derived = {
        "rayd_fresh_configure": {"build_facts": rayd_facts},
        "multiarch_wheel": {
            "artifact": wheel,
            "architectures": list(WHEEL_ARCHITECTURES),
            "native": wheel_native,
            "independent_cuda_audit": cuda_arch_audit,
        },
        "wheel_smoke": {"artifact": smoke, "independent": independent},
        "pe_audit": {
            "wheel_smoke": wheel_validation["pe_audit"],
            "extracted_native": wheel_pe_audit,
        },
        "build_fingerprint": {"value": wheel_validation["build_info"]["build_fingerprint"]},  # type: ignore[index]
        "clean_channel_checkout": {"verified": post["clean_channel_checkouts_verified"]},
        "clean_rayd_checkout": {"verified": post["clean_rayd_checkout_verified"]},
        "no_generation_or_temporary_names": _naming_audit(validation_checkout),
    }
    if set(captures) | set(derived) != set(gate["required_release_checks"]):  # type: ignore[arg-type]
        raise EvidenceError("release check set differs from the frozen policy")
    checks = [
        {"name": name, "kind": "subprocess", "capture": capture}
        for name, capture in sorted(captures.items())
    ] + [
        {"name": name, "kind": "derived", "evidence": value}
        for name, value in sorted(derived.items())
    ]
    return {
        "checks": checks,
        "wheel": wheel,
        "wheel_validation": wheel_validation,
        "native_extension": extension,
        "fresh_rayd_build": {
            "canonical_relative_path": "backends/torch/build/local-120",
            "facts": rayd_facts,
        },
        "validation_checkout": validation,
        "tool_identities": {
            name: executable_identity(getattr(config.tools, name), label=name)
            for name in (
                "conda", "cmake", "ctest", "ninja", "cuobjdump", "dumpbin",
                "powershell", "cmd", "vcvars64", "git", "nvcc", "cl", "link", "nvidia_smi",
            )
        },
    }


def _validate_runner_binding(
    capture: Mapping[str, object], *, implementation: Mapping[str, object]
) -> dict[str, object]:
    binding = capture.get("runner_binding")
    if not isinstance(binding, dict):
        raise EvidenceError("Channel release tier lacks its runner-owned binding")
    exact_keys(
        binding,
        {
            "mode",
            "validation_checkout",
            "site_packages_source",
            "extension_path",
            "build_fingerprint",
            "rayd_source",
            "python_executable",
            "cmake_args",
            "cuda_architectures",
            "torch_cuda_arch_list",
        },
        label="Channel release runner binding",
    )
    if (
        binding["mode"] != "runner-owned-packaged-validation-checkout"
        or binding["build_fingerprint"] != implementation.get("final_build_fingerprint")
        or binding["cuda_architectures"] != list(WHEEL_ARCHITECTURES)
        or binding["torch_cuda_arch_list"] != TORCH_WHEEL_ARCHITECTURES
    ):
        raise EvidenceError("Channel release tier used a non-canonical runner binding")
    validation_checkout = Path(str(binding["validation_checkout"]))
    site_packages_source = Path(str(binding["site_packages_source"]))
    extension = Path(str(binding["extension_path"]))
    rayd_source = Path(str(binding["rayd_source"]))
    python_executable = Path(str(binding["python_executable"]))
    if (
        not validation_checkout.is_absolute()
        or not site_packages_source.is_absolute()
        or not extension.is_absolute()
        or extension.parent
        != validation_checkout / "src" / "witwin" / "channel"
        or not rayd_source.is_absolute()
        or not python_executable.is_absolute()
    ):
        raise EvidenceError("Channel release tier binding paths are not absolute/contained")
    cmake_args = binding["cmake_args"]
    expected_definitions = (
        _quoted_cmake_definition("RAYD_SOURCE_DIR", rayd_source),
        _quoted_cmake_definition(
            "CMAKE_CUDA_ARCHITECTURES", ";".join(WHEEL_ARCHITECTURES)
        ),
        _quoted_cmake_definition("CMAKE_BUILD_TYPE", "Release"),
        _quoted_cmake_definition("CHANNEL_RELEASE_BUILD", "ON"),
        _quoted_cmake_definition("Python_EXECUTABLE", python_executable),
    )
    if not isinstance(cmake_args, str) or any(
        definition not in cmake_args for definition in expected_definitions
    ):
        raise EvidenceError("Channel release tier CMake binding is incomplete")
    return dict(binding)


def _validate_packaged_validation_checkout(
    value: object,
    *,
    implementation: Mapping[str, object],
    store: ArtifactStore,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise EvidenceError("packaged validation checkout evidence is malformed")
    exact_keys(
        value,
        {
            "checkout",
            "commit",
            "clone",
            "detached_checkout",
            "clean_status",
            "packaged_identity_capture",
            "packaged_identity_record",
            "post_release_clean_status",
            "site_packages_source",
            "source_extension",
            "packaged_extension",
            "packaged_extension_path",
            "packaged_fingerprint",
            "build_fingerprint",
            "local_exclude",
        },
        label="packaged validation checkout",
    )
    final_history = implementation["groups"][FINAL_GROUP]  # type: ignore[index]
    assert isinstance(final_history, dict)
    if (
        value["commit"] != final_history["candidate_commit"]
        or value["build_fingerprint"] != implementation.get("final_build_fingerprint")
        or value["local_exclude"]
        != "/src/witwin/channel/_channel.build-fingerprint"
    ):
        raise EvidenceError("packaged validation checkout identity differs")
    checkout = Path(str(value["checkout"]))
    packaged_path = Path(str(value["packaged_extension_path"]))
    source_site_packages = Path(str(value["site_packages_source"]))
    if (
        not checkout.is_absolute()
        or not source_site_packages.is_absolute()
        or packaged_path.parent
        != checkout / "src" / "witwin" / "channel"
    ):
        raise EvidenceError("packaged validation checkout paths are not canonical")
    source_extension = value["source_extension"]
    packaged_extension = value["packaged_extension"]
    packaged_fingerprint = value["packaged_fingerprint"]
    if not all(
        isinstance(row, dict)
        for row in (source_extension, packaged_extension, packaged_fingerprint)
    ):
        raise EvidenceError("packaged validation checkout artifacts are malformed")
    source = store.verify_reference(source_extension, label="runner source extension")
    packaged = store.verify_reference(
        packaged_extension, label="validation packaged extension"
    )
    if (
        source["sha256"] != implementation.get("final_extension_sha256")
        or packaged["sha256"] != source["sha256"]
        or packaged["bytes"] != source["bytes"]
    ):
        raise EvidenceError("packaged validation extension bytes differ")
    fingerprint_payload = store.read_verified(
        packaged_fingerprint,
        label="validation packaged fingerprint",
        allow_empty=False,
    )
    if fingerprint_payload not in {
        f"{value['build_fingerprint']}\n".encode("ascii"),
        str(value["build_fingerprint"]).encode("ascii"),
    }:
        raise EvidenceError("packaged validation fingerprint bytes differ")
    for name in (
        "clone",
        "detached_checkout",
        "clean_status",
        "packaged_identity_capture",
        "post_release_clean_status",
    ):
        capture = value[name]
        if (
            not isinstance(capture, dict)
            or capture.get("returncode") != 0
            or capture.get("timed_out") is not False
        ):
            raise EvidenceError(f"packaged validation capture failed: {name}")
        for stream_name in ("stdout_artifact", "stderr_artifact"):
            stream = capture.get(stream_name)
            if not isinstance(stream, dict):
                raise EvidenceError(f"packaged validation capture lacks {stream_name}")
            store.verify_reference(
                stream,
                label=f"packaged validation {name} {stream_name}",
                allow_empty=True,
            )
    for name in ("clean_status", "post_release_clean_status"):
        capture = value[name]
        assert isinstance(capture, dict)
        if store.read_verified(
            capture["stdout_artifact"],  # type: ignore[arg-type]
            label=f"packaged validation {name} stdout",
            allow_empty=True,
        ):
            raise EvidenceError("packaged validation checkout was not Git-clean")
    identity = value["packaged_identity_record"]
    if not isinstance(identity, dict):
        raise EvidenceError("packaged validation identity record is malformed")
    build_info = identity.get("build_info")
    if (
        not isinstance(build_info, dict)
        or build_info.get("build_fingerprint") != value["build_fingerprint"]
        or Path(str(identity.get("extension_path"))) != packaged_path
    ):
        raise EvidenceError("packaged validation identity probe differs")
    return dict(value)


def validate_release_report(
    release: Mapping[str, object],
    *,
    implementation: Mapping[str, object],
    gate: Mapping[str, object],
    store: ArtifactStore,
) -> dict[str, object]:
    exact_keys(
        release,
        {
            "checks",
            "wheel",
            "wheel_validation",
            "native_extension",
            "fresh_rayd_build",
            "validation_checkout",
            "tool_identities",
        },
        label="release evidence",
    )
    tool_names = {
        "conda", "cmake", "ctest", "ninja", "cuobjdump", "dumpbin",
        "powershell", "cmd", "vcvars64", "git", "nvcc", "cl", "link",
        "nvidia_smi",
    }
    release_tools = release["tool_identities"]
    implementation_tools = implementation.get("tool_executable_identity")
    if (
        not isinstance(release_tools, dict)
        or set(release_tools) != tool_names
        or not isinstance(implementation_tools, dict)
        or not tool_names <= set(implementation_tools)
        or any(release_tools[name] != implementation_tools[name] for name in tool_names)
    ):
        raise EvidenceError("release tool identities differ from frozen measured tools")
    checks = release["checks"]
    if not isinstance(checks, list) or len(checks) != len(gate["required_release_checks"]):  # type: ignore[arg-type]
        raise EvidenceError("release report has the wrong number of checks")
    by_name: dict[str, Mapping[str, object]] = {}
    for row in checks:
        if not isinstance(row, dict):
            raise EvidenceError("release check row must be an object")
        if row.get("kind") not in {"subprocess", "derived"}:
            raise EvidenceError("release check kind is not accepted")
        exact_keys(row, {"name", "kind", "capture"} if row.get("kind") == "subprocess" else {"name", "kind", "evidence"}, label="release check")
        name = str(row["name"])
        if name in by_name:
            raise EvidenceError(f"duplicate release check: {name}")
        by_name[name] = row
    if set(by_name) != set(gate["required_release_checks"]):  # type: ignore[arg-type]
        raise EvidenceError("release report check-name set is not canonical")
    naming = by_name["no_generation_or_temporary_names"].get("evidence")
    if (
        not isinstance(naming, dict)
        or naming.get("passed") is not True
        or naming.get("scanned_suffixes") != sorted(
            [".cmake", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".json", ".py"]
        )
    ):
        raise EvidenceError("release naming audit evidence is malformed")
    channel_bindings: list[dict[str, object]] = []
    for name, row in by_name.items():
        if row["kind"] != "subprocess":
            continue
        capture = row["capture"]
        if not isinstance(capture, dict) or capture.get("returncode") != 0 or capture.get("timed_out") is not False:
            raise EvidenceError(f"release subprocess failed: {name}")
        stdout = capture.get("stdout_artifact")
        stderr = capture.get("stderr_artifact")
        if not isinstance(stdout, dict) or not isinstance(stderr, dict):
            raise EvidenceError(f"release subprocess lacks retained streams: {name}")
        store.verify_reference(stdout, label=f"{name} stdout", allow_empty=True)
        store.verify_reference(stderr, label=f"{name} stderr", allow_empty=True)
        if name.startswith("channel_"):
            channel_bindings.append(
                _validate_runner_binding(capture, implementation=implementation)
            )
    if len(channel_bindings) != len(CHANNEL_TIERS) or any(
        binding != channel_bindings[0] for binding in channel_bindings[1:]
    ):
        raise EvidenceError("Channel release tiers did not share one runner-owned binding")
    validation_checkout = _validate_packaged_validation_checkout(
        release["validation_checkout"],
        implementation=implementation,
        store=store,
    )
    if channel_bindings[0]["validation_checkout"] != validation_checkout["checkout"]:
        raise EvidenceError("Channel release tiers used a different validation checkout")
    wheel = release["wheel"]
    native = release["native_extension"]
    if not isinstance(wheel, dict) or not isinstance(native, dict):
        raise EvidenceError("release artifacts are malformed")
    store.verify_reference(wheel, label="release wheel")
    store.verify_reference(native, label="release native extension")
    if native["sha256"] != implementation.get("final_extension_sha256"):
        raise EvidenceError("retained release extension differs from measured final candidate")
    smoke_evidence = by_name["wheel_smoke"].get("evidence")
    if (
        not isinstance(smoke_evidence, dict)
        or not isinstance(smoke_evidence.get("artifact"), dict)
        or not isinstance(smoke_evidence.get("independent"), dict)
    ):
        raise EvidenceError("wheel smoke artifact is missing")
    supplemental = _validate_wheel_smoke(
        _load_retained_json(store, smoke_evidence["artifact"], label="wheel smoke"),
        implementation=implementation,
        wheel_sha256=str(wheel["sha256"]),
    )
    independent = smoke_evidence["independent"]
    independent_artifact = independent.get("artifact")
    independent_capture = independent.get("capture")
    if not isinstance(independent_artifact, dict) or not isinstance(independent_capture, dict):
        raise EvidenceError("independent wheel audit artifacts are malformed")
    if independent_capture.get("returncode") != 0 or independent_capture.get("timed_out") is not False:
        raise EvidenceError("independent wheel audit subprocess failed")
    independent_payload = _load_retained_json(
        store, independent_artifact, label="independent wheel audit"
    )
    replayed = _validate_wheel_smoke(
        independent_payload,
        implementation=implementation,
        wheel_sha256=str(wheel["sha256"]),
    )
    if replayed != supplemental or replayed != release["wheel_validation"]:
        raise EvidenceError("wheel validation does not replay from retained facts")
    fresh = release["fresh_rayd_build"]
    if (
        not isinstance(fresh, dict)
        or set(fresh) != {"canonical_relative_path", "facts"}
        or fresh.get("canonical_relative_path") != "backends/torch/build/local-120"
        or not isinstance(fresh.get("facts"), dict)
    ):
        raise EvidenceError("fresh RayD build evidence is malformed")
    _validate_rayd_build_facts(
        fresh["facts"],  # type: ignore[arg-type]
        store=store,
        rayd_checkout=None,
        implementation=implementation,
    )
    multiarch = by_name["multiarch_wheel"].get("evidence")
    if (
        not isinstance(multiarch, dict)
        or set(multiarch)
        != {"artifact", "architectures", "native", "independent_cuda_audit"}
        or multiarch.get("artifact") != wheel
        or multiarch.get("architectures") != list(WHEEL_ARCHITECTURES)
        or not isinstance(multiarch.get("native"), dict)
        or not isinstance(multiarch.get("independent_cuda_audit"), dict)
    ):
        raise EvidenceError("independent CUDA architecture audit is missing")
    wheel_native = multiarch["native"]
    assert isinstance(wheel_native, dict)
    if set(wheel_native) != {"member", "artifact"} or wheel_native.get(
        "member"
    ) != WHEEL_NATIVE_MEMBER or not isinstance(wheel_native.get("artifact"), dict):
        raise EvidenceError("retained wheel native evidence is malformed")
    retained_wheel_native = store.verify_reference(
        wheel_native["artifact"],  # type: ignore[arg-type]
        label="retained wheel native extension",
    )
    if retained_wheel_native["sha256"] != replayed["pe_audit"].get("sha256"):  # type: ignore[union-attr]
        raise EvidenceError("retained wheel native bytes differ from wheel PE audit")
    if retained_wheel_native["sha256"] == native["sha256"]:
        raise EvidenceError("multiarch wheel native bytes equal the SM120 timing extension")
    arch = multiarch["independent_cuda_audit"]
    capture = arch.get("capture")
    if (
        not isinstance(capture, dict)
        or capture.get("returncode") != 0
        or capture.get("timed_out") is not False
    ):
        raise EvidenceError("cuobjdump capture is malformed")
    argv = capture.get("argv")
    if (
        not isinstance(argv, list)
        or len(argv) != 4
        or argv[1:3] != ["--list-elf", "--list-ptx"]
        or Path(str(argv[3])).resolve()
        != (store.root / str(retained_wheel_native["path"])).resolve()
    ):
        raise EvidenceError("cuobjdump did not consume the retained wheel native extension")
    stdout = store.read_verified(
        capture["stdout_artifact"], label="cuobjdump stdout", allow_empty=False
    ).decode("utf-8", errors="replace")
    real = sorted({int(value) for value in re.findall(r"(?i)sm_([0-9]+)", stdout)})
    virtual = sorted({int(value) for value in re.findall(r"(?i)compute_([0-9]+)", stdout)})
    if arch.get("real") != real or arch.get("virtual") != virtual or real != [75, 80, 86, 89, 120] or virtual != [120]:
        raise EvidenceError("independent cubin/PTX architecture audit does not replay")
    pe_evidence = by_name["pe_audit"].get("evidence")
    if (
        not isinstance(pe_evidence, dict)
        or set(pe_evidence) != {"wheel_smoke", "extracted_native"}
        or pe_evidence.get("wheel_smoke") != replayed["pe_audit"]
        or not isinstance(pe_evidence.get("extracted_native"), dict)
    ):
        raise EvidenceError("independent wheel PE evidence is malformed")
    extracted_pe = pe_evidence["extracted_native"]
    assert isinstance(extracted_pe, dict)
    if set(extracted_pe) != {"capture", "artifact", "payload", "normalized"}:
        raise EvidenceError("independent extracted PE audit schema differs")
    pe_capture = extracted_pe["capture"]
    pe_artifact = extracted_pe["artifact"]
    if (
        not isinstance(pe_capture, dict)
        or pe_capture.get("returncode") != 0
        or pe_capture.get("timed_out") is not False
        or not isinstance(pe_artifact, dict)
    ):
        raise EvidenceError("independent extracted PE audit failed")
    pe_payload = _load_retained_json(
        store, pe_artifact, label="independent wheel PE audit"
    )
    normalized_pe = _validate_extracted_pe_audit(
        pe_payload,
        native=wheel_native,
        wheel_pe_audit=replayed["pe_audit"],  # type: ignore[arg-type]
        store=store,
    )
    if (
        extracted_pe["payload"] != pe_payload
        or extracted_pe["normalized"] != normalized_pe
    ):
        raise EvidenceError("independent wheel PE audit does not replay")
    return dict(release)


__all__ = ["CHANNEL_TIERS", "run_release_evidence", "validate_release_report"]

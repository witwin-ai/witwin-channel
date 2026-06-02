"""Shared Slang runtime helpers for diffraction kernels."""

from __future__ import annotations

import locale
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


_SLANG_MODULE_CACHE: dict[str, Any] = {}
_SLANGTORCH_WINDOWS_PATCHED = False


def ensure_current_env_on_path() -> None:
    scripts_dir = os.path.join(os.path.dirname(sys.executable), "Scripts")
    if not os.path.isdir(scripts_dir):
        return
    current_path = os.environ.get("PATH", "")
    path_entries = current_path.split(os.pathsep) if current_path else []
    if scripts_dir not in path_entries:
        os.environ["PATH"] = scripts_dir + os.pathsep + current_path


def _decode_subprocess_bytes(payload: bytes) -> str:
    for encoding in ("utf-8", locale.getpreferredencoding(False), "mbcs", "cp936"):
        if not encoding:
            continue
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    return payload.decode("utf-8", errors="replace")


def _patch_slangtorch_windows_decode() -> None:
    global _SLANGTORCH_WINDOWS_PATCHED

    if _SLANGTORCH_WINDOWS_PATCHED or sys.platform != "win32":
        return

    try:
        import slangtorch.slangtorch as slangtorch_module
        import slangtorch.util.compile as compile_module
        import torch.utils.cpp_extension as cpp_extension
    except Exception:
        return

    def _run_ninja_with_locale_decode(build_directory: str, verbose: bool) -> int:
        command = ["ninja", "-v"]
        num_workers = compile_module._get_num_workers(verbose)
        if num_workers is not None:
            command.extend(["-j", str(num_workers)])

        env = os.environ.copy()
        if compile_module.IS_WINDOWS and "VSCMD_ARG_TGT_ARCH" not in env:
            from setuptools import distutils
            import distutils._msvccompiler

            plat_name = distutils.util.get_platform()
            plat_spec = compile_module.PLAT_TO_VCVARS[plat_name]
            vc_env = distutils._msvccompiler._get_vc_env(plat_spec)
            vc_env = {key.upper(): value for key, value in vc_env.items()}
            for key, value in env.items():
                upper_key = key.upper()
                if upper_key not in vc_env:
                    vc_env[upper_key] = value
            env = vc_env

        try:
            sys.stdout.flush()
            sys.stderr.flush()
            proc = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=build_directory,
                check=True,
                env=env,
            )
            stdout = _decode_subprocess_bytes(proc.stdout)
            if verbose:
                print(stdout)
            if "ninja: no work to do." in stdout:
                return compile_module.NinjaResult.NO_WORK_TO_DO
            return compile_module.NinjaResult.BUILD_SUCCESS
        except subprocess.CalledProcessError as exc:
            if verbose:
                print(_decode_subprocess_bytes(exc.stdout))
                print(_decode_subprocess_bytes(exc.stderr))
            return compile_module.NinjaResult.BUILD_FAIL

    compile_module.run_ninja = _run_ninja_with_locale_decode
    slangtorch_module.run_ninja = _run_ninja_with_locale_decode

    def _torch_run_ninja_build(build_directory: str, verbose: bool, error_prefix: str) -> None:
        command = ["ninja", "-v"]
        num_workers = cpp_extension._get_num_workers(verbose)
        if num_workers is not None:
            command.extend(["-j", str(num_workers)])

        env = os.environ.copy()
        if cpp_extension.IS_WINDOWS and "VSCMD_ARG_TGT_ARCH" not in env:
            from setuptools import distutils  # type: ignore[attr-defined]

            plat_name = distutils.util.get_platform()
            plat_spec = cpp_extension.PLAT_TO_VCVARS[plat_name]
            vc_env = {k.upper(): v for k, v in cpp_extension._get_vc_env(plat_spec).items()}
            for key, value in env.items():
                upper_key = key.upper()
                if upper_key not in vc_env:
                    vc_env[upper_key] = value
            env = vc_env

        try:
            sys.stdout.flush()
            sys.stderr.flush()
            stdout_target = 1 if verbose else subprocess.PIPE
            proc = subprocess.run(
                command,
                shell=cpp_extension.IS_WINDOWS and cpp_extension.IS_HIP_EXTENSION,
                stdout=stdout_target,
                stderr=subprocess.STDOUT,
                cwd=build_directory,
                check=True,
                env=env,
            )
            if verbose and proc.stdout:
                print(_decode_subprocess_bytes(proc.stdout))
        except subprocess.CalledProcessError as exc:
            message = error_prefix
            if exc.output:
                message += f": {_decode_subprocess_bytes(exc.output)}"
            raise RuntimeError(message) from exc

    cpp_extension._run_ninja_build = _torch_run_ninja_build
    _SLANGTORCH_WINDOWS_PATCHED = True


def load_slang_module(path: str | Path):
    try:
        import slangtorch
    except ImportError:
        return None

    ensure_current_env_on_path()
    _patch_slangtorch_windows_decode()
    module_path = str(Path(path).resolve())
    module = _SLANG_MODULE_CACHE.get(module_path)
    if module is None:
        module = slangtorch.loadModule(module_path)
        _SLANG_MODULE_CACHE[module_path] = module
    return module


def launch_shape_1d(length: int, block_size: int) -> tuple[int, int, int]:
    return ((length + block_size - 1) // block_size, 1, 1)


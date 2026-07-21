from __future__ import annotations

import re
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BINDING_ROOT = REPOSITORY_ROOT / "native/channel_native/binding"
MODULE_PATH = BINDING_ROOT / "module.cpp"
REGISTRY_HEADER_PATH = BINDING_ROOT / "registry.h"
EXPECTED_REGISTRIES = {
    "bdpt",
    "build",
    "fields",
    "materials",
    "montecarlo",
    "montecarlo_transmission",
    "path",
    "rayd",
    "runtime",
}


def test_binding_module_is_the_unique_pybind_module_owner() -> None:
    module_owners = [
        path.relative_to(REPOSITORY_ROOT).as_posix()
        for path in (REPOSITORY_ROOT / "native/channel_native").rglob("*.cpp")
        if "PYBIND11_MODULE" in path.read_text(encoding="utf-8")
    ]

    assert module_owners == ["native/channel_native/binding/module.cpp"]


def test_binding_module_only_sets_doc_and_calls_registrars() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    registry_header = REGISTRY_HEADER_PATH.read_text(encoding="utf-8")
    module_body = re.fullmatch(
        r'#include "registry\.h"\s*'
        r"PYBIND11_MODULE\(_channel_native, module\)\s*\{(?P<body>.*?)\}\s*",
        source,
        flags=re.DOTALL,
    )

    assert len(source.splitlines()) <= 300
    assert module_body is not None

    statements = [
        statement.strip()
        for statement in module_body.group("body").split(";")
        if statement.strip()
    ]
    assert statements.count(
        'module.doc() = "Channel Native Torch/CUDA extension."'
    ) == 1
    registrar_calls = statements.copy()
    registrar_calls.remove('module.doc() = "Channel Native Torch/CUDA extension."')
    declared_registrars = re.findall(
        r"void (register_[a-z0-9_]+)\(pybind11::module_ &module\);",
        registry_header,
    )
    called_registrars = [
        match.group("name")
        for call in registrar_calls
        if (
            match := re.fullmatch(
                r"(?P<name>register_[a-z0-9_]+)\(module\)", call
            )
        )
    ]

    assert len(declared_registrars) == len(set(declared_registrars))
    assert len(called_registrars) == len(registrar_calls)
    assert len(called_registrars) == len(set(called_registrars))
    assert set(called_registrars) == set(declared_registrars)


def test_binding_registries_are_declared_and_built_without_legacy_aggregators() -> None:
    cmake = (REPOSITORY_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    binding_sources = {path.stem for path in BINDING_ROOT.glob("*.cpp")}
    declared_registrars = set(
        re.findall(
            r"void (register_[a-z0-9_]+)\(pybind11::module_ &module\);",
            REGISTRY_HEADER_PATH.read_text(encoding="utf-8"),
        )
    )
    defined_registrars: list[str] = []

    assert binding_sources == EXPECTED_REGISTRIES | {"module"}
    for registry in EXPECTED_REGISTRIES:
        definitions = re.findall(
            r"void (register_[a-z0-9_]+)\(pybind11::module_ &module\)\s*\{",
            (BINDING_ROOT / f"{registry}.cpp").read_text(encoding="utf-8"),
        )
        assert definitions
        defined_registrars.extend(definitions)
        assert f"native/channel_native/binding/{registry}.cpp" in cmake

    assert len(defined_registrars) == len(set(defined_registrars))
    assert set(defined_registrars) == declared_registrars

    for legacy_source in (
        "native/channel_native/bindings.cpp",
        "native/channel_native/binding/all.cpp",
    ):
        assert not (REPOSITORY_ROOT / legacy_source).exists()
        assert legacy_source not in cmake

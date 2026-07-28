from __future__ import annotations

import ast
import hashlib
import inspect
from types import SimpleNamespace

import importlib.util

import pytest

from witwin.channel import runtime


def _body_hash(function: object) -> str:
    tree = ast.parse(inspect.getsource(function))
    node = tree.body[0]
    assert isinstance(node, ast.FunctionDef)
    body = ast.dump(
        ast.Module(body=node.body, type_ignores=[]),
        annotate_fields=True,
        include_attributes=False,
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


@pytest.fixture(autouse=True)
def _isolated_loader_state():
    runtime._clear_loader_caches()
    yield
    runtime._clear_loader_caches()


def test_runtime_owns_the_only_native_symbol_accessor():
    """No compatibility facade re-exports the loader outside ``runtime``."""

    assert runtime.native_extension is runtime.native_extension
    assert importlib.util.find_spec("witwin.channel.core") is None


def test_required_native_op_has_one_body_preserving_runtime_owner():
    function = runtime._required_native_op

    assert function.__module__ == "witwin.channel.runtime"
    assert "_required_native_op" not in runtime.__all__
    assert function.__globals__["_native_symbols"] is runtime
    assert _body_hash(function) == (
        "f60ee207119d675d9a7b6b9982131fbc44b1238eb3ce8274b1607136fbfc490d"
    )


def test_native_extension_preserves_loader_cache(monkeypatch: pytest.MonkeyPatch):
    runtime._clear_loader_caches()
    native = SimpleNamespace()
    imports: list[tuple[str, str | None]] = []

    def import_module(name: str, package: str | None = None):
        imports.append((name, package))
        return native

    monkeypatch.setattr(runtime.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(runtime.importlib, "import_module", import_module)
    monkeypatch.setattr(
        runtime,
        "_validate_extension",
        lambda module, *, packaged, expected_fingerprint: module,
    )
    monkeypatch.setattr(runtime, "_packaged_expected_fingerprint", lambda: "0" * 64)

    assert runtime.native_extension() is native
    assert runtime.native_extension() is native
    assert imports == [("._channel", "witwin.channel")]


def test_required_symbol_keeps_lookup_order_and_single_loader_call(
    monkeypatch: pytest.MonkeyPatch,
):
    kernel = object()
    loads: list[None] = []
    accesses: list[str] = []

    class Native:
        def __getattribute__(self, name: str):
            if name == "kernel":
                accesses.append(name)
                return kernel
            return object.__getattribute__(self, name)

    def load() -> object:
        loads.append(None)
        return Native()

    monkeypatch.setattr(runtime, "native_extension", load)

    assert runtime.required_symbol("kernel") is kernel
    assert loads == [None]
    assert accesses == ["kernel", "kernel"]


def test_required_symbol_uses_the_existing_error_contract(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(runtime, "native_extension", lambda: SimpleNamespace())

    with pytest.raises(
        runtime.NativeSymbolError,
        match=r"^_channel\.missing CUDA kernel is required$",
    ):
        runtime.required_symbol("missing")


def test_optional_symbol_is_not_cached(monkeypatch: pytest.MonkeyPatch):
    native = SimpleNamespace()
    monkeypatch.setattr(runtime, "native_extension", lambda: native)

    assert runtime.optional_symbol("feature") is None
    feature = object()
    native.feature = feature
    assert runtime.optional_symbol("feature") is feature


def test_has_symbol_observes_each_monkeypatched_extension(
    monkeypatch: pytest.MonkeyPatch,
):
    native = SimpleNamespace(feature=object())
    monkeypatch.setattr(runtime, "native_extension", lambda: native)
    assert runtime.has_symbol("feature") is True

    monkeypatch.setattr(runtime, "native_extension", lambda: SimpleNamespace())
    assert runtime.has_symbol("feature") is False


@pytest.mark.parametrize(
    "lookup",
    (
        runtime._required_native_op,
        runtime.required_symbol,
        runtime.optional_symbol,
        runtime.has_symbol,
    ),
)
def test_symbol_lookups_preserve_loader_errors(monkeypatch: pytest.MonkeyPatch, lookup):
    error = runtime.ExtensionLoadError("native identity rejected")

    def fail() -> object:
        raise error

    monkeypatch.setattr(runtime, "native_extension", fail)

    with pytest.raises(runtime.ExtensionLoadError) as captured:
        lookup("feature")
    assert captured.value is error


def test_required_native_op_preserves_runtime_monkeypatch_and_call_count(
    monkeypatch: pytest.MonkeyPatch,
):
    kernel = object()
    loads: list[None] = []

    def load() -> object:
        loads.append(None)
        return SimpleNamespace(kernel=kernel)

    monkeypatch.setattr(runtime, "native_extension", load)

    assert runtime._required_native_op("kernel") is kernel
    assert loads == [None]


def test_required_native_op_preserves_missing_kernel_text(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(runtime, "native_extension", lambda: None)

    with pytest.raises(
        runtime.NativeSymbolError,
        match=r"^_channel\.kernel CUDA kernel is required$",
    ):
        runtime._required_native_op("kernel")

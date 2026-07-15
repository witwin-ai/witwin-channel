from __future__ import annotations

from types import SimpleNamespace

import pytest

from witwin.channel_native.core.kernels import extension as compatibility_extension
from witwin.channel_native.core.kernels import ops
from witwin.channel_native.runtime import extension, symbols


@pytest.fixture(autouse=True)
def _isolated_loader_state():
    extension._clear_loader_caches()
    yield
    extension._clear_loader_caches()


def test_compatibility_exports_share_the_runtime_symbol_owner():
    assert compatibility_extension.native_extension is symbols.native_extension
    assert extension.native_extension is symbols.native_extension


def test_native_extension_preserves_loader_cache(monkeypatch: pytest.MonkeyPatch):
    extension._clear_loader_caches()
    native = SimpleNamespace()
    imports: list[tuple[str, str | None]] = []

    def import_module(name: str, package: str | None = None):
        imports.append((name, package))
        return native

    monkeypatch.setattr(extension.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(extension.importlib, "import_module", import_module)
    monkeypatch.setattr(
        extension,
        "_validate_extension",
        lambda module, *, packaged, expected_fingerprint: module,
    )
    monkeypatch.setattr(extension, "_packaged_expected_fingerprint", lambda: "0" * 64)

    assert symbols.native_extension() is native
    assert symbols.native_extension() is native
    assert imports == [("._channel_native", "witwin.channel_native")]


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

    monkeypatch.setattr(symbols, "native_extension", load)

    assert symbols.required_symbol("kernel") is kernel
    assert loads == [None]
    assert accesses == ["kernel", "kernel"]


def test_required_symbol_uses_the_existing_error_contract(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(symbols, "native_extension", lambda: SimpleNamespace())

    with pytest.raises(
        symbols.NativeSymbolError,
        match=r"^_channel_native\.missing CUDA kernel is required$",
    ):
        symbols.required_symbol("missing")


def test_optional_symbol_is_not_cached(monkeypatch: pytest.MonkeyPatch):
    native = SimpleNamespace()
    monkeypatch.setattr(symbols, "native_extension", lambda: native)

    assert symbols.optional_symbol("feature") is None
    feature = object()
    native.feature = feature
    assert symbols.optional_symbol("feature") is feature


def test_has_symbol_observes_each_monkeypatched_extension(
    monkeypatch: pytest.MonkeyPatch,
):
    native = SimpleNamespace(feature=object())
    monkeypatch.setattr(symbols, "native_extension", lambda: native)
    assert symbols.has_symbol("feature") is True

    monkeypatch.setattr(symbols, "native_extension", lambda: SimpleNamespace())
    assert symbols.has_symbol("feature") is False


@pytest.mark.parametrize(
    "lookup",
    (symbols.required_symbol, symbols.optional_symbol, symbols.has_symbol),
)
def test_symbol_lookups_preserve_loader_errors(monkeypatch: pytest.MonkeyPatch, lookup):
    error = extension.ExtensionLoadError("native identity rejected")

    def fail() -> object:
        raise error

    monkeypatch.setattr(symbols, "native_extension", fail)

    with pytest.raises(extension.ExtensionLoadError) as captured:
        lookup("feature")
    assert captured.value is error


def test_ops_required_symbol_preserves_local_monkeypatch_and_call_count(
    monkeypatch: pytest.MonkeyPatch,
):
    kernel = object()
    loads: list[None] = []

    def load() -> object:
        loads.append(None)
        return SimpleNamespace(kernel=kernel)

    monkeypatch.setattr(ops, "native_extension", load)

    assert ops._required_native_op("kernel") is kernel
    assert loads == [None]


def test_ops_required_symbol_preserves_missing_kernel_text(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(ops, "native_extension", lambda: None)

    with pytest.raises(
        RuntimeError, match=r"^_channel_native\.kernel CUDA kernel is required$"
    ):
        ops._required_native_op("kernel")

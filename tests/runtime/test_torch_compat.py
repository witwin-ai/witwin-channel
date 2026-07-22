from pathlib import Path
import tomllib
from types import SimpleNamespace

from packaging.specifiers import SpecifierSet
from packaging.version import Version
import pytest
import torch

from witwin.channel.runtime import torch_compat


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "src" / "witwin" / "channel_native"
_INSTALLED_TORCH = torch.__version__.split("+", maxsplit=1)[0]


def _supported_torch_specs() -> list[str]:
    with (REPO_ROOT / "ci" / "support-matrix.toml").open("rb") as stream:
        runtimes = tomllib.load(stream)["runtime"]
    return [runtime["torch_spec"] for runtime in runtimes]


@pytest.mark.parametrize("installed_torch", [_INSTALLED_TORCH])
def test_installed_torch_is_supported_and_cxx_abi_contract_holds(installed_torch):
    # A single interpreter is validated here against the support matrix. The
    # wheel-matrix tier re-runs this suite on every supported Torch build to
    # provide the multi-version coverage that one installed environment cannot.
    specs = _supported_torch_specs()
    version = Version(installed_torch)
    assert any(version in SpecifierSet(spec) for spec in specs), (
        f"installed torch {installed_torch} is outside supported specs {specs}"
    )

    value = torch.tensor([2.0])

    assert torch_compat.is_transform_wrapped_tensor(value) is False
    assert torch_compat.transform_level(value) == -1
    assert torch_compat.interpreter_stack() == ()
    assert torch_compat.uses_cxx11_abi() is bool(torch._C._GLIBCXX_USE_CXX11_ABI)


def test_single_jvp_reports_stack_and_unwraps_without_copy():
    observed: dict[str, object] = {}

    def function(value: torch.Tensor) -> torch.Tensor:
        stack = torch_compat.interpreter_stack()
        unwrapped = torch_compat.unwrap_transform_tensor(value)
        observed.update(
            wrapped=torch_compat.is_transform_wrapped_tensor(value),
            level=torch_compat.transform_level(value),
            stack_length=len(stack),
            jvp_entries=tuple(torch_compat.is_jvp_transform(item) for item in stack),
            unwrapped=unwrapped,
            unwrapped_wrapped=torch_compat.is_transform_wrapped_tensor(unwrapped),
            unwrapped_level=torch_compat.transform_level(unwrapped),
        )
        return value.square()

    primal_input = torch.tensor([2.0])
    primal, tangent = torch.func.jvp(
        function, (primal_input,), (torch.ones_like(primal_input),)
    )

    torch.testing.assert_close(primal, torch.tensor([4.0]))
    torch.testing.assert_close(tangent, torch.tensor([4.0]))
    assert observed["wrapped"] is True
    assert observed["level"] == 1
    assert observed["stack_length"] == 1
    assert observed["jvp_entries"] == (True,)
    assert observed["unwrapped_wrapped"] is False
    assert observed["unwrapped_level"] == -1
    unwrapped = observed["unwrapped"]
    assert isinstance(unwrapped, torch.Tensor)
    assert unwrapped.data_ptr() == primal_input.data_ptr()
    assert unwrapped.stride() == primal_input.stride()
    assert torch_compat.interpreter_stack() == ()


def test_composed_transform_stack_exposes_grad_and_jvp_entries():
    observed: list[tuple[bool, ...]] = []

    def outer(value: torch.Tensor) -> torch.Tensor:
        def inner(inner_value: torch.Tensor) -> torch.Tensor:
            stack = torch_compat.interpreter_stack()
            observed.append(
                tuple(torch_compat.is_jvp_transform(item) for item in stack)
            )
            return inner_value.square()

        _, tangent = torch.func.jvp(inner, (value,), (torch.ones_like(value),))
        return tangent.sum()

    gradient = torch.func.grad(outer)(torch.tensor([2.0]))

    torch.testing.assert_close(gradient, torch.tensor([2.0]))
    assert observed == [(False, True)]
    assert torch_compat.interpreter_stack() == ()


def test_disable_functorch_context_disables_and_restores_dispatch():
    observed: dict[str, object] = {}

    def disabled_add(value: torch.Tensor) -> torch.Tensor:
        with torch_compat.disable_functorch():
            result = value + 1.0
            observed["wrapped"] = torch_compat.is_transform_wrapped_tensor(result)
            observed["level"] = torch_compat.transform_level(result)
        return result

    value = torch.tensor([2.0])
    primal, tangent = torch.func.jvp(
        disabled_add, (value,), (torch.full_like(value, 3.0),)
    )

    torch.testing.assert_close(primal, torch.tensor([3.0]))
    torch.testing.assert_close(tangent, torch.zeros_like(tangent))
    assert observed == {"wrapped": False, "level": -1}

    _, restored_tangent = torch.func.jvp(
        lambda item: item + 1.0, (value,), (torch.full_like(value, 3.0),)
    )
    torch.testing.assert_close(restored_tangent, torch.tensor([3.0]))
    assert torch_compat.interpreter_stack() == ()


def test_missing_private_api_fails_loudly(monkeypatch: pytest.MonkeyPatch):
    value = torch.tensor([1.0])
    monkeypatch.setattr(
        torch_compat,
        "torch",
        SimpleNamespace(_C=SimpleNamespace()),
    )

    with pytest.raises(AttributeError):
        torch_compat.is_transform_wrapped_tensor(value)
    with pytest.raises(AttributeError):
        torch_compat.disable_functorch()
    with pytest.raises(AttributeError):
        torch_compat.uses_cxx11_abi()


def test_private_torch_api_has_a_single_production_owner():
    owner = PACKAGE_ROOT / "runtime" / "torch_compat.py"
    violations = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if path == owner:
            continue
        source = path.read_text(encoding="utf-8")
        if "torch._C" in source or "_DisableFuncTorch" in source or "._functorch" in source:
            violations.append(path.relative_to(PACKAGE_ROOT).as_posix())

    assert violations == []

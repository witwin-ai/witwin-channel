import pytest
import torch

from witwin.channel_native import ReceiverPoint, Scene


def _empty_scene() -> Scene:
    return Scene(
        structures=[],
        transmitters=[],
        receivers=[ReceiverPoint(position=torch.zeros(3))],
        frequency=3.5e9,
    )


@pytest.mark.parametrize(
    ("module_name", "message"),
    [
        ("witwin.channel_native.deterministic", "deterministic solver is reserved"),
        ("witwin.channel_native.path", "path solver is reserved"),
        ("witwin.channel_native.psdr", "PSDR solver is reserved"),
    ],
)
def test_reserved_solvers_raise_explicit_phase_errors(module_name, message):
    module = __import__(module_name, fromlist=["Config", "solve"])

    with pytest.raises(NotImplementedError, match=message):
        module.solve(_empty_scene(), module.Config())

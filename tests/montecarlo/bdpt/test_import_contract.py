import importlib
import sys


FORBIDDEN_MODULES = (
    "rayd",
    "drjit",
    "mitsuba",
    "sionna",
    "witwin.channel",
)


def test_bdpt_import_exposes_public_api_without_forbidden_modules():
    for name in FORBIDDEN_MODULES:
        sys.modules.pop(name, None)

    bdpt = importlib.import_module("witwin.channel.montecarlo.bdpt")

    assert bdpt.Config.__name__ == "Config"
    assert bdpt.Result.__name__ == "Result"
    assert bdpt.BDPTPathSamples.__name__ == "BDPTPathSamples"
    assert callable(bdpt.solve)
    for name in FORBIDDEN_MODULES:
        assert name not in sys.modules

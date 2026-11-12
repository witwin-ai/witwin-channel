# Copyright Xingyu Chen.
# Tests reserved solver imports.

def test_reserved_solver_packages_import():
    from witwin.channel import radiomap
    import witwin.channel.experimental.bdpt as bdpt
    import witwin.channel.path as path

    for name in ("Coherent", "CoherentRadioMap", "Stochastic", "RadioMap", "PathTable"):
        assert getattr(radiomap, name).__name__ == name
        assert getattr(radiomap, name).__module__ == "witwin.channel.radiomap"
    assert callable(radiomap.solve)
    assert path.Config.__name__ == "Config"
    assert path.PathResult.__name__ == "PathResult"
    assert bdpt.Config.__name__ == "Config"
    assert bdpt.Result.__name__ == "Result"
    assert bdpt.BDPTPathSamples.__name__ == "BDPTPathSamples"
    assert callable(bdpt.solve)
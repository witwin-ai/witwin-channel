def test_reserved_solver_packages_import():
    import witwin.channel_native.deterministic as deterministic
    import witwin.channel_native.montecarlo.bdpt as bdpt
    import witwin.channel_native.path as path
    import witwin.channel_native.psdr as psdr

    assert deterministic.Config.__name__ == "Config"
    assert deterministic.Result.__name__ == "Result"
    assert path.Config.__name__ == "Config"
    assert path.PathResult.__name__ == "PathResult"
    assert psdr.Config.__name__ == "Config"
    assert psdr.Result.__name__ == "Result"
    assert bdpt.Config.__name__ == "Config"
    assert bdpt.Result.__name__ == "Result"
    assert bdpt.BDPTPathSamples.__name__ == "BDPTPathSamples"
    assert callable(bdpt.solve)

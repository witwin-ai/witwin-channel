def test_reserved_solver_packages_import():
    import witwin.channel.deterministic as deterministic
    import witwin.channel.montecarlo.bdpt as bdpt
    import witwin.channel.path as path

    assert deterministic.Config.__name__ == "Config"
    assert deterministic.Result.__name__ == "Result"
    assert path.Config.__name__ == "Config"
    assert path.PathResult.__name__ == "PathResult"
    assert bdpt.Config.__name__ == "Config"
    assert bdpt.Result.__name__ == "Result"
    assert bdpt.BDPTPathSamples.__name__ == "BDPTPathSamples"
    assert callable(bdpt.solve)

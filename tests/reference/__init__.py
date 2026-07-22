"""Test-only reference implementations for ADR-010 numerical-kernel lockstep.

These modules hold the previous Torch implementations that the native CUDA
kernels replaced. They are imported only by the lockstep tests and MUST NOT be
imported from any production ``witwin.channel`` package.
"""

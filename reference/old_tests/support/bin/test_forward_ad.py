"""Test forward-mode automatic differentiation with torch -> drjit -> torch gradient flow."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.autograd.forward_ad as fwAD
import drjit as dr
import witwin as wt

from witwin.channel.utils import drjit_forward_ad, wrap_drjit_forward_ad


# =============================================================================
# Tests
# =============================================================================

def test_basic_forward_ad():
    """Test basic forward AD: y = x^2"""
    print("=" * 60)
    print("Test 1: Basic forward AD (y = x^2)")
    print("=" * 60)

    def drjit_square(x):
        return x * x

    x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    v = torch.ones_like(x)

    with fwAD.dual_level():
        dual_x = fwAD.make_dual(x, v)
        dual_y = drjit_forward_ad(drjit_square, dual_x)
        primal, tangent = fwAD.unpack_dual(dual_y)

    expected_primal = x ** 2
    expected_tangent = 2 * x * v  # dy/dx = 2x

    print(f"x = {x}")
    print(f"v = {v}")
    print(f"primal = {primal}, expected = {expected_primal}")
    print(f"tangent = {tangent}, expected = {expected_tangent}")

    passed = (
        torch.allclose(primal, expected_primal) and
        torch.allclose(tangent, expected_tangent)
    )
    print(f"PASS: {passed}")
    return passed


def test_decorator_forward_ad():
    """Test decorator syntax."""
    print("\n" + "=" * 60)
    print("Test 2: Decorator syntax")
    print("=" * 60)

    @wrap_drjit_forward_ad
    def drjit_cube(x):
        return x * x * x

    x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    v = torch.ones_like(x)

    with fwAD.dual_level():
        dual_x = fwAD.make_dual(x, v)
        dual_y = drjit_cube(dual_x)
        primal, tangent = fwAD.unpack_dual(dual_y)

    expected_primal = x ** 3
    expected_tangent = 3 * x ** 2 * v  # dy/dx = 3x^2

    print(f"x = {x}")
    print(f"primal = {primal}, expected = {expected_primal}")
    print(f"tangent = {tangent}, expected = {expected_tangent}")

    passed = (
        torch.allclose(primal, expected_primal) and
        torch.allclose(tangent, expected_tangent)
    )
    print(f"PASS: {passed}")
    return passed


def test_complex_function():
    """Test complex function: y = sin(x) * exp(-x^2)"""
    print("\n" + "=" * 60)
    print("Test 3: Complex function (y = sin(x) * exp(-x^2))")
    print("=" * 60)

    @wrap_drjit_forward_ad
    def complex_func(x):
        return dr.sin(x) * dr.exp(-x * x)

    x = torch.tensor([0.0, 0.5, 1.0], device='cuda')
    v = torch.ones_like(x)

    with fwAD.dual_level():
        dual_x = fwAD.make_dual(x, v)
        dual_y = complex_func(dual_x)
        primal, tangent = fwAD.unpack_dual(dual_y)

    expected_primal = torch.sin(x) * torch.exp(-x ** 2)
    # dy/dx = cos(x)*exp(-x^2) + sin(x)*(-2x)*exp(-x^2) = exp(-x^2)*(cos(x) - 2x*sin(x))
    expected_tangent = torch.exp(-x ** 2) * (torch.cos(x) - 2 * x * torch.sin(x)) * v

    print(f"x = {x}")
    print(f"primal = {primal}")
    print(f"expected_primal = {expected_primal}")
    print(f"tangent = {tangent}")
    print(f"expected_tangent = {expected_tangent}")

    passed = (
        torch.allclose(primal, expected_primal, atol=1e-5) and
        torch.allclose(tangent, expected_tangent, atol=1e-5)
    )
    print(f"PASS: {passed}")
    return passed


def test_multiple_inputs():
    """Test multiple inputs: f(x, y) = x^2 + x*y + y^2"""
    print("\n" + "=" * 60)
    print("Test 4: Multiple inputs (f = x^2 + xy + y^2)")
    print("=" * 60)

    @wrap_drjit_forward_ad
    def multi_func(x, y):
        return x * x + x * y + y * y

    x = torch.tensor([1.0, 2.0], device='cuda')
    y = torch.tensor([2.0, 3.0], device='cuda')
    vx = torch.ones_like(x)
    vy = torch.ones_like(y)

    with fwAD.dual_level():
        dual_x = fwAD.make_dual(x, vx)
        dual_y = fwAD.make_dual(y, vy)
        dual_f = multi_func(dual_x, dual_y)
        primal, tangent = fwAD.unpack_dual(dual_f)

    expected_primal = x ** 2 + x * y + y ** 2
    # df/dx = 2x + y, df/dy = x + 2y
    # JVP = (2x + y)*vx + (x + 2y)*vy = 2x + y + x + 2y = 3x + 3y
    expected_tangent = (2 * x + y) * vx + (x + 2 * y) * vy

    print(f"x = {x}, y = {y}")
    print(f"vx = {vx}, vy = {vy}")
    print(f"primal = {primal}, expected = {expected_primal}")
    print(f"tangent = {tangent}, expected = {expected_tangent}")

    passed = (
        torch.allclose(primal, expected_primal) and
        torch.allclose(tangent, expected_tangent)
    )
    print(f"PASS: {passed}")
    return passed


def test_partial_tangent():
    """Test when only some inputs have tangents."""
    print("\n" + "=" * 60)
    print("Test 5: Partial tangent (only x has tangent)")
    print("=" * 60)

    @wrap_drjit_forward_ad
    def multi_func(x, y):
        return x * x + x * y + y * y

    x = torch.tensor([1.0, 2.0], device='cuda')
    y = torch.tensor([2.0, 3.0], device='cuda')
    vx = torch.ones_like(x)

    with fwAD.dual_level():
        dual_x = fwAD.make_dual(x, vx)
        # y is NOT dual - no tangent
        dual_f = multi_func(dual_x, y)
        primal, tangent = fwAD.unpack_dual(dual_f)

    expected_primal = x ** 2 + x * y + y ** 2
    # df/dx = 2x + y, df/dy = 0 (y has no tangent)
    expected_tangent = (2 * x + y) * vx

    print(f"x = {x}, y = {y}")
    print(f"vx = {vx}, vy = None")
    print(f"primal = {primal}, expected = {expected_primal}")
    print(f"tangent = {tangent}, expected = {expected_tangent}")

    passed = (
        torch.allclose(primal, expected_primal) and
        torch.allclose(tangent, expected_tangent)
    )
    print(f"PASS: {passed}")
    return passed


def test_tuple_return():
    """Test function returning tuple."""
    print("\n" + "=" * 60)
    print("Test 6: Tuple return")
    print("=" * 60)

    @wrap_drjit_forward_ad
    def tuple_func(x):
        return (x * x, x * x * x)

    x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    v = torch.ones_like(x)

    with fwAD.dual_level():
        dual_x = fwAD.make_dual(x, v)
        dual_y1, dual_y2 = tuple_func(dual_x)
        primal1, tangent1 = fwAD.unpack_dual(dual_y1)
        primal2, tangent2 = fwAD.unpack_dual(dual_y2)

    expected_primal1 = x ** 2
    expected_primal2 = x ** 3
    expected_tangent1 = 2 * x * v
    expected_tangent2 = 3 * x ** 2 * v

    print(f"x = {x}")
    print(f"y1 = x^2: primal={primal1}, tangent={tangent1}")
    print(f"y2 = x^3: primal={primal2}, tangent={tangent2}")

    passed = (
        torch.allclose(primal1, expected_primal1) and
        torch.allclose(primal2, expected_primal2) and
        torch.allclose(tangent1, expected_tangent1) and
        torch.allclose(tangent2, expected_tangent2)
    )
    print(f"PASS: {passed}")
    return passed


def test_chained_operations():
    """Test chaining: PyTorch -> DrJit -> PyTorch"""
    print("\n" + "=" * 60)
    print("Test 7: Chained operations (torch -> drjit -> torch)")
    print("=" * 60)

    @wrap_drjit_forward_ad
    def drjit_middle(x):
        return dr.sin(x)

    x = torch.tensor([0.0, 0.5, 1.0], device='cuda')
    v = torch.ones_like(x)

    with fwAD.dual_level():
        dual_x = fwAD.make_dual(x, v)

        # PyTorch: y1 = x^2
        dual_y1 = dual_x ** 2

        # DrJit: y2 = sin(y1)
        dual_y2 = drjit_middle(dual_y1)

        # PyTorch: y3 = y2^2
        dual_y3 = dual_y2 ** 2

        primal, tangent = fwAD.unpack_dual(dual_y3)

    # y1 = x^2, y2 = sin(y1) = sin(x^2), y3 = y2^2 = sin(x^2)^2
    # dy3/dx = 2*sin(x^2)*cos(x^2)*2x = 4x*sin(x^2)*cos(x^2)
    expected_primal = torch.sin(x ** 2) ** 2
    expected_tangent = 4 * x * torch.sin(x ** 2) * torch.cos(x ** 2) * v

    print(f"x = {x}")
    print(f"y = sin(x^2)^2")
    print(f"primal = {primal}")
    print(f"expected_primal = {expected_primal}")
    print(f"tangent = {tangent}")
    print(f"expected_tangent = {expected_tangent}")

    passed = (
        torch.allclose(primal, expected_primal, atol=1e-5) and
        torch.allclose(tangent, expected_tangent, atol=1e-5)
    )
    print(f"PASS: {passed}")
    return passed


def test_different_tangent_vectors():
    """Test with different tangent vectors (not just ones)."""
    print("\n" + "=" * 60)
    print("Test 8: Different tangent vectors")
    print("=" * 60)

    @wrap_drjit_forward_ad
    def drjit_square(x):
        return x * x

    x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    v = torch.tensor([1.0, 0.5, 2.0], device='cuda')  # Custom tangent

    with fwAD.dual_level():
        dual_x = fwAD.make_dual(x, v)
        dual_y = drjit_square(dual_x)
        primal, tangent = fwAD.unpack_dual(dual_y)

    expected_primal = x ** 2
    expected_tangent = 2 * x * v

    print(f"x = {x}")
    print(f"v = {v}")
    print(f"primal = {primal}, expected = {expected_primal}")
    print(f"tangent = {tangent}, expected = {expected_tangent}")

    passed = (
        torch.allclose(primal, expected_primal) and
        torch.allclose(tangent, expected_tangent)
    )
    print(f"PASS: {passed}")
    return passed


def test_no_tangent():
    """Test when no inputs have tangents (outside dual_level)."""
    print("\n" + "=" * 60)
    print("Test 9: No tangent (outside dual_level)")
    print("=" * 60)

    @wrap_drjit_forward_ad
    def drjit_square(x):
        return x * x

    x = torch.tensor([1.0, 2.0, 3.0], device='cuda')

    # No dual_level - just regular forward pass
    y = drjit_square(x)

    expected = x ** 2

    print(f"x = {x}")
    print(f"y = {y}, expected = {expected}")

    passed = torch.allclose(y, expected)
    print(f"PASS: {passed}")
    return passed


def test_norm_function():
    """Test norm function: ||v|| = sqrt(x^2 + y^2 + z^2)"""
    print("\n" + "=" * 60)
    print("Test 10: Norm function")
    print("=" * 60)

    @wrap_drjit_forward_ad
    def drjit_norm(x, y, z):
        return dr.sqrt(x * x + y * y + z * z)

    vx = torch.tensor([3.0, 1.0], device='cuda')
    vy = torch.tensor([4.0, 0.0], device='cuda')
    vz = torch.tensor([0.0, 0.0], device='cuda')

    # Tangent only on x
    tx = torch.ones_like(vx)

    with fwAD.dual_level():
        dual_x = fwAD.make_dual(vx, tx)
        dual_norm = drjit_norm(dual_x, vy, vz)
        primal, tangent = fwAD.unpack_dual(dual_norm)

    expected_primal = torch.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
    # d(norm)/dx = x / norm
    expected_tangent = vx / expected_primal * tx

    print(f"vx={vx}, vy={vy}, vz={vz}")
    print(f"primal (norm) = {primal}, expected = {expected_primal}")
    print(f"tangent = {tangent}, expected = {expected_tangent}")

    passed = (
        torch.allclose(primal, expected_primal, atol=1e-5) and
        torch.allclose(tangent, expected_tangent, atol=1e-5)
    )
    print(f"PASS: {passed}")
    return passed


if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("# Forward AD: torch -> drjit -> torch")
    print("#" * 60)

    results = {}

    results['basic'] = test_basic_forward_ad()
    results['decorator'] = test_decorator_forward_ad()
    results['complex_func'] = test_complex_function()
    results['multi_input'] = test_multiple_inputs()
    results['partial_tangent'] = test_partial_tangent()
    results['tuple_return'] = test_tuple_return()
    results['chained'] = test_chained_operations()
    results['custom_tangent'] = test_different_tangent_vectors()
    results['no_tangent'] = test_no_tangent()
    results['norm'] = test_norm_function()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_passed = True
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name}: {status}")
        all_passed = all_passed and passed

    print("\n" + "-" * 60)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED")
    print("-" * 60)

    print("""
Usage:
    from witwin.channel.utils import wrap_drjit_forward_ad, drjit_forward_ad

    # Method 1: Decorator
    @wrap_drjit_forward_ad
    def my_drjit_func(x):
        return dr.sin(x) * dr.exp(-x * x)

    with fwAD.dual_level():
        dual_x = fwAD.make_dual(x, tangent)
        dual_y = my_drjit_func(dual_x)
        primal, tangent_out = fwAD.unpack_dual(dual_y)

    # Method 2: Direct call
    with fwAD.dual_level():
        dual_x = fwAD.make_dual(x, tangent)
        dual_y = drjit_forward_ad(lambda x: x * x, dual_x)
        primal, tangent_out = fwAD.unpack_dual(dual_y)
""")



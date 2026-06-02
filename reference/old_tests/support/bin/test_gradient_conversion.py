"""Test script to verify gradient preservation when converting between PyTorch and DrJit."""
import torch
import drjit as dr
import witwin as wt


def test_torch_to_drjit_to_torch():
    """Test if gradients are preserved through torch -> drjit -> torch conversion."""
    print("=" * 60)
    print("Test: PyTorch -> DrJit -> PyTorch gradient preservation")
    print("=" * 60)

    # Create PyTorch tensor with gradient tracking
    x_torch = torch.tensor([1.0, 2.0, 3.0, 4.0], device='cuda', requires_grad=True)
    print(f"\n1. Original PyTorch tensor: {x_torch}")
    print(f"   requires_grad: {x_torch.requires_grad}")

    # Convert to DrJit Float
    x_drjit = wt.Float(x_torch)
    print(f"\n2. Converted to DrJit: {x_drjit}")

    # Perform operation in DrJit
    y_drjit = x_drjit * 2.0 + 1.0
    print(f"\n3. DrJit operation (x * 2 + 1): {y_drjit}")

    # Convert back to PyTorch
    y_torch = y_drjit.torch()
    print(f"\n4. Converted back to PyTorch: {y_torch}")
    print(f"   requires_grad: {y_torch.requires_grad}")
    print(f"   grad_fn: {y_torch.grad_fn}")

    # Try backward pass
    print("\n5. Attempting backward pass...")
    try:
        loss = y_torch.sum()
        loss.backward()
        print(f"   SUCCESS! x_torch.grad = {x_torch.grad}")
    except Exception as e:
        print(f"   FAILED: {e}")

    return y_torch.requires_grad


def test_drjit_native_autodiff():
    """Test DrJit's native autodiff capability."""
    print("\n" + "=" * 60)
    print("Test: DrJit native autodiff")
    print("=" * 60)

    # Create DrJit array with gradient tracking
    x = wt.Float([1.0, 2.0, 3.0, 4.0])
    dr.enable_grad(x)
    print(f"\n1. DrJit tensor with grad enabled: {x}")
    print(f"   grad_enabled: {dr.grad_enabled(x)}")

    # Perform operations
    y = x * 2.0 + 1.0
    z = dr.sum(y)
    print(f"\n2. y = x * 2 + 1: {y}")
    print(f"   z = sum(y): {z}")

    # Backward pass
    dr.backward(z)
    grad = dr.grad(x)
    print(f"\n3. Gradient of x: {grad}")

    return True


def test_combined_autodiff():
    """Test if DrJit autodiff can work with PyTorch tensors."""
    print("\n" + "=" * 60)
    print("Test: Combined PyTorch/DrJit autodiff")
    print("=" * 60)

    # Start with PyTorch tensor
    x_torch = torch.tensor([1.0, 2.0, 3.0, 4.0], device='cuda', requires_grad=True)
    print(f"\n1. PyTorch input (requires_grad=True): {x_torch}")

    # Convert to DrJit and enable DrJit grad
    x_drjit = wt.Float(x_torch)
    dr.enable_grad(x_drjit)
    print(f"\n2. Converted to DrJit, grad_enabled: {dr.grad_enabled(x_drjit)}")

    # DrJit operations
    y_drjit = x_drjit * 2.0 + 1.0
    z = dr.sum(y_drjit)
    print(f"\n3. y = x * 2 + 1, z = sum(y) = {z}")

    # DrJit backward
    dr.backward(z)
    drjit_grad = dr.grad(x_drjit)
    print(f"\n4. DrJit gradient: {drjit_grad}")

    # Check if PyTorch tensor got gradient
    print(f"\n5. PyTorch x_torch.grad: {x_torch.grad}")
    print("   (None expected - DrJit and PyTorch autodiff are separate)")

    return True


def test_manual_gradient_bridge():
    """Test manual gradient bridging between DrJit and PyTorch."""
    print("\n" + "=" * 60)
    print("Test: Manual gradient bridge (custom autograd function)")
    print("=" * 60)

    class DrJitOp(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            # Convert to DrJit, perform operation
            x_drjit = wt.Float(x)
            y_drjit = x_drjit * 2.0 + 1.0
            y_torch = y_drjit.torch()
            # Save for backward (store the multiplier)
            ctx.save_for_backward(torch.tensor([2.0], device=x.device))
            return y_torch

        @staticmethod
        def backward(ctx, grad_output):
            multiplier, = ctx.saved_tensors
            # Gradient of (x * 2 + 1) w.r.t. x is 2
            return grad_output * multiplier

    x = torch.tensor([1.0, 2.0, 3.0, 4.0], device='cuda', requires_grad=True)
    print(f"\n1. Input: {x}")

    y = DrJitOp.apply(x)
    print(f"\n2. Output (x * 2 + 1): {y}")
    print(f"   requires_grad: {y.requires_grad}")

    loss = y.sum()
    loss.backward()
    print(f"\n3. Gradient: {x.grad}")
    print("   Expected: [2, 2, 2, 2]")

    return x.grad is not None


def test_drjit_torch_integration():
    """Test DrJit's built-in PyTorch integration for autodiff."""
    print("\n" + "=" * 60)
    print("Test: DrJit-PyTorch autodiff integration (dr.wrap)")
    print("=" * 60)

    try:
        # Use dr.wrap (wrap_ad is deprecated)
        if hasattr(dr, 'wrap'):
            print("\ndr.wrap is available")

            # Define a DrJit function wrapped for PyTorch
            @dr.wrap(source='torch', target='drjit')
            def drjit_func(x: wt.Float) -> wt.Float:
                return x * 2.0 + 1.0

            x = torch.tensor([1.0, 2.0, 3.0, 4.0], device='cuda', requires_grad=True)
            print(f"\n1. Input PyTorch tensor: {x}")

            # Call the wrapped function directly with PyTorch tensor
            y = drjit_func(x)
            print(f"\n2. Output: {y}")
            print(f"   Type: {type(y)}")
            print(f"   requires_grad: {y.requires_grad if hasattr(y, 'requires_grad') else 'N/A'}")

            # Try backward
            if isinstance(y, torch.Tensor) and y.requires_grad:
                loss = y.sum()
                loss.backward()
                print(f"\n3. Gradient: {x.grad}")
                return True
            else:
                print("\n3. Output is not a differentiable PyTorch tensor")
                return False
        else:
            print("\ndr.wrap not available")
            return False
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_direct_conversion_methods():
    """Test dr.to_torch() and direct Float3 construction."""
    print("\n" + "=" * 60)
    print("Test: Direct conversion methods")
    print("=" * 60)

    # Test 1: wt.Float with torch tensor
    print("\n--- Test 1: wt.Float(torch_tensor) ---")
    x1 = torch.tensor([1.0, 2.0, 3.0], device='cuda', requires_grad=True)
    print(f"Input: {x1}, requires_grad={x1.requires_grad}")

    x1_drjit = wt.Float(x1)
    print(f"wt.Float(x): {x1_drjit}")

    y1_drjit = x1_drjit * 2.0
    y1_torch = y1_drjit.torch()
    print(f"After ops and .torch(): {y1_torch}, requires_grad={y1_torch.requires_grad}")

    # Test 2: wt.Float3 with torch tensor
    print("\n--- Test 2: wt.Float3(torch_tensor) ---")
    x2 = torch.tensor([1.0, 2.0, 3.0], device='cuda', requires_grad=True)
    print(f"Input: {x2}, requires_grad={x2.requires_grad}")

    try:
        x2_drjit = wt.Float3(x2)
        print(f"wt.Float3(x): {x2_drjit}")
        y2_torch = x2_drjit.torch()
        print(f".torch(): {y2_torch}, requires_grad={y2_torch.requires_grad}")
    except Exception as e:
        print(f"Error: {e}")

    # Test 3: wt.Point3f with torch tensor
    print("\n--- Test 3: wt.Point3f(torch_tensors) ---")
    px = torch.tensor([1.0, 2.0], device='cuda', requires_grad=True)
    py = torch.tensor([3.0, 4.0], device='cuda', requires_grad=True)
    pz = torch.tensor([5.0, 6.0], device='cuda', requires_grad=True)
    print(f"Inputs: px={px}, py={py}, pz={pz}")

    try:
        p_drjit = wt.Point3f(wt.Float(px), wt.Float(py), wt.Float(pz))
        print(f"wt.Point3f: {p_drjit}")

        # Do operation and convert back
        p2_drjit = p_drjit * 2.0
        px_back = p2_drjit.x.torch()
        py_back = p2_drjit.y.torch()
        pz_back = p2_drjit.z.torch()
        print(f"After *2 and .torch(): x={px_back}, requires_grad={px_back.requires_grad}")
    except Exception as e:
        print(f"Error: {e}")

    # Test 4: dr.zeros / dr.ones with torch
    print("\n--- Test 4: dr.zeros -> torch ---")
    z_drjit = dr.zeros(wt.Float, 4)
    dr.enable_grad(z_drjit)
    z_drjit = z_drjit + 1.0  # Make it non-trivial
    z_torch = z_drjit.torch()
    print(f"dr.zeros + 1.0 -> .torch(): {z_torch}, requires_grad={z_torch.requires_grad}")

    # Test 5: Check if there's a global to_torch function
    print("\n--- Test 5: Check for dr.to_torch() ---")
    if hasattr(dr, 'to_torch'):
        print("dr.to_torch exists!")
        x5 = wt.Float([1.0, 2.0, 3.0])
        y5 = dr.to_torch(x5)
        print(f"dr.to_torch(wt.Float): {y5}, type={type(y5)}")
    else:
        print("dr.to_torch does NOT exist. Use array.torch() instead.")

    # Test 6: TensorXf (if available)
    print("\n--- Test 6: wt.TensorXf ---")
    try:
        t_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda', requires_grad=True)
        print(f"Input torch: {t_torch.shape}, requires_grad={t_torch.requires_grad}")

        t_drjit = wt.TensorXf(t_torch)
        print(f"wt.TensorXf: shape={t_drjit.shape}")

        t_back = t_drjit.torch()
        print(f".torch(): {t_back.shape}, requires_grad={t_back.requires_grad}")
    except Exception as e:
        print(f"Error: {e}")

    return False  # Basic conversions don't preserve grad


def test_wrap_with_vector_types():
    """Test dr.wrap with vector types like Float3, Point3f."""
    print("\n" + "=" * 60)
    print("Test: dr.wrap with vector types (Float3, Point3f)")
    print("=" * 60)

    # Test 1: Multiple wt.Float inputs
    print("\n--- Test 1: dr.wrap with multiple wt.Float ---")
    try:
        @dr.wrap(source='torch', target='drjit')
        def compute_length(x: wt.Float, y: wt.Float, z: wt.Float) -> wt.Float:
            return dr.sqrt(x*x + y*y + z*z)

        vx = torch.tensor([3.0, 0.0], device='cuda', requires_grad=True)
        vy = torch.tensor([4.0, 1.0], device='cuda', requires_grad=True)
        vz = torch.tensor([0.0, 0.0], device='cuda', requires_grad=True)

        print(f"Input: vx={vx}, vy={vy}, vz={vz}")

        result = compute_length(vx, vy, vz)
        print(f"Result (length): {result}")
        print(f"requires_grad: {result.requires_grad}")

        loss = result.sum()
        loss.backward()
        print(f"Gradients: vx.grad={vx.grad}, vy.grad={vy.grad}, vz.grad={vz.grad}")
        # For (3,4,0), length=5, grad_x = x/length = 3/5 = 0.6
        print("Expected vx.grad[0] = 3/5 = 0.6")
        success1 = True
    except Exception as e:
        print(f"Error: {e}")
        success1 = False

    # Test 2: Using tuple return
    print("\n--- Test 2: dr.wrap returning tuple of wt.Float ---")
    try:
        @dr.wrap(source='torch', target='drjit')
        def scale_components(x: wt.Float, y: wt.Float, scale: wt.Float) -> tuple:
            return (x * scale, y * scale)

        x = torch.tensor([1.0, 2.0], device='cuda', requires_grad=True)
        y = torch.tensor([3.0, 4.0], device='cuda', requires_grad=True)
        s = torch.tensor([2.0, 2.0], device='cuda', requires_grad=True)

        print(f"Input: x={x}, y={y}, scale={s}")

        rx, ry = scale_components(x, y, s)
        print(f"Result: rx={rx}, ry={ry}")
        print(f"requires_grad: rx={rx.requires_grad}, ry={ry.requires_grad}")

        loss = rx.sum() + ry.sum()
        loss.backward()
        print(f"Gradients: x.grad={x.grad}, y.grad={y.grad}, s.grad={s.grad}")
        success2 = True
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        success2 = False

    return success1 and success2


def test_utils_functions():
    """Test the new gradient-preserving functions in rfdt.utils."""
    print("\n" + "=" * 60)
    print("Test: rfdt.utils gradient-preserving functions")
    print("=" * 60)

    from witwin.channel.utils import (
        wrap_drjit, torch_to_drjit, drjit_to_torch,
        drjit_mul, drjit_add, drjit_sqrt, drjit_norm
    )

    all_passed = True

    # Test 1: torch_to_drjit identity
    print("\n--- Test 1: torch_to_drjit ---")
    x = torch.tensor([1.0, 2.0, 3.0], device='cuda', requires_grad=True)
    y = torch_to_drjit(x)
    print(f"Input: {x}, requires_grad={x.requires_grad}")
    print(f"Output: {y}, requires_grad={y.requires_grad}")
    loss = (y * 2).sum()
    loss.backward()
    print(f"Gradient: {x.grad}")
    expected = torch.tensor([2.0, 2.0, 2.0], device='cuda')
    passed = torch.allclose(x.grad, expected)
    print(f"PASS: {passed}")
    all_passed = all_passed and passed

    # Test 2: drjit_to_torch identity
    print("\n--- Test 2: drjit_to_torch ---")
    x = torch.tensor([1.0, 2.0, 3.0], device='cuda', requires_grad=True)
    y = drjit_to_torch(x)
    print(f"Input: {x}, requires_grad={x.requires_grad}")
    print(f"Output: {y}, requires_grad={y.requires_grad}")
    loss = (y * 3).sum()
    loss.backward()
    print(f"Gradient: {x.grad}")
    expected = torch.tensor([3.0, 3.0, 3.0], device='cuda')
    passed = torch.allclose(x.grad, expected)
    print(f"PASS: {passed}")
    all_passed = all_passed and passed

    # Test 3: drjit_mul
    print("\n--- Test 3: drjit_mul ---")
    a = torch.tensor([2.0, 3.0], device='cuda', requires_grad=True)
    b = torch.tensor([4.0, 5.0], device='cuda', requires_grad=True)
    c = drjit_mul(a, b)
    print(f"a={a}, b={b}, a*b={c}")
    print(f"requires_grad: {c.requires_grad}")
    c.sum().backward()
    print(f"a.grad={a.grad}, b.grad={b.grad}")
    # d(a*b)/da = b, d(a*b)/db = a
    passed = torch.allclose(a.grad, b.detach()) and torch.allclose(b.grad, a.detach())
    print(f"PASS: {passed}")
    all_passed = all_passed and passed

    # Test 4: drjit_sqrt
    print("\n--- Test 4: drjit_sqrt ---")
    x = torch.tensor([4.0, 9.0, 16.0], device='cuda', requires_grad=True)
    y = drjit_sqrt(x)
    print(f"sqrt({x}) = {y}")
    print(f"requires_grad: {y.requires_grad}")
    y.sum().backward()
    print(f"Gradient: {x.grad}")
    # d(sqrt(x))/dx = 1/(2*sqrt(x))
    expected = 1.0 / (2.0 * torch.sqrt(x.detach()))
    passed = torch.allclose(x.grad, expected)
    print(f"Expected: {expected}")
    print(f"PASS: {passed}")
    all_passed = all_passed and passed

    # Test 5: drjit_norm
    print("\n--- Test 5: drjit_norm ---")
    x = torch.tensor([3.0], device='cuda', requires_grad=True)
    y = torch.tensor([4.0], device='cuda', requires_grad=True)
    z = torch.tensor([0.0], device='cuda', requires_grad=True)
    n = drjit_norm(x, y, z)
    print(f"norm({x}, {y}, {z}) = {n}")
    print(f"requires_grad: {n.requires_grad}")
    n.sum().backward()
    print(f"Gradients: x.grad={x.grad}, y.grad={y.grad}, z.grad={z.grad}")
    # d(norm)/dx = x/norm = 3/5 = 0.6
    passed = torch.allclose(x.grad, torch.tensor([0.6], device='cuda'))
    print(f"Expected x.grad=0.6, y.grad=0.8, z.grad=0.0")
    print(f"PASS: {passed}")
    all_passed = all_passed and passed

    # Test 6: wrap_drjit decorator
    print("\n--- Test 6: @wrap_drjit decorator ---")

    @wrap_drjit
    def custom_op(a: wt.Float, b: wt.Float) -> wt.Float:
        return a * a + b * b

    a = torch.tensor([3.0], device='cuda', requires_grad=True)
    b = torch.tensor([4.0], device='cuda', requires_grad=True)
    result = custom_op(a, b)
    print(f"custom_op({a}, {b}) = a^2 + b^2 = {result}")
    print(f"requires_grad: {result.requires_grad}")
    result.sum().backward()
    print(f"Gradients: a.grad={a.grad}, b.grad={b.grad}")
    # d(a^2+b^2)/da = 2a = 6, d/db = 2b = 8
    passed = torch.allclose(a.grad, torch.tensor([6.0], device='cuda'))
    print(f"Expected a.grad=6, b.grad=8")
    print(f"PASS: {passed}")
    all_passed = all_passed and passed

    return all_passed


if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("# Gradient Preservation Test: PyTorch <-> DrJit")
    print("#" * 60)

    results = {}

    # Test 1: Basic conversion
    results['basic_conversion'] = test_torch_to_drjit_to_torch()

    # Test 2: DrJit native autodiff
    results['drjit_native'] = test_drjit_native_autodiff()

    # Test 3: Combined autodiff attempt
    results['combined'] = test_combined_autodiff()

    # Test 4: Manual gradient bridge
    results['manual_bridge'] = test_manual_gradient_bridge()

    # Test 5: DrJit-PyTorch integration
    results['drjit_torch_integration'] = test_drjit_torch_integration()

    # Test 6: Direct conversion methods
    results['direct_conversion'] = test_direct_conversion_methods()

    # Test 7: dr.wrap with vector types
    results['wrap_vector_types'] = test_wrap_with_vector_types()

    # Test 8: rfdt.utils functions
    results['utils_functions'] = test_utils_functions()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL/LIMITED"
        print(f"  {test_name}: {status}")

    print("\n" + "-" * 60)
    print("CONCLUSION:")
    print("-" * 60)
    print("""
  - Direct conversion (torch -> drjit -> torch) does NOT preserve
    PyTorch's autograd graph. The gradient computation graph is broken.

  - DrJit has its own autodiff system (dr.enable_grad, dr.backward)
    which works independently of PyTorch.

  - To bridge gradients between DrJit and PyTorch, you need:
    1. Custom torch.autograd.Function (manual gradient computation)
    2. Or use dr.wrap_ad if available in your DrJit version

  - For this project, if gradients through DrJit ops are needed,
    consider implementing custom autograd functions or using
    DrJit's native autodiff throughout the computation.
""")



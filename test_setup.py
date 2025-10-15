#!/usr/bin/env python3
"""
Test script to verify the environment setup is correct.
Run this to check that all dependencies are working properly.
"""

import sys
print("=" * 60)
print("TMM Genetic Optimization - Environment Test")
print("=" * 60)

# Test 1: Python version
print(f"\n[1/6] Python version: {sys.version}")
assert sys.version_info >= (3, 9), "Python 3.9+ required"
print("✓ Python version OK")

# Test 2: NumPy
try:
    import numpy as np
    print(f"\n[2/6] NumPy version: {np.__version__}")
    # Test basic operation
    arr = np.array([1, 2, 3])
    fft_result = np.fft.fft(arr)
    print(f"      FFT test: {arr} -> {fft_result}")
    print("✓ NumPy OK")
except Exception as e:
    print(f"✗ NumPy FAILED: {e}")
    sys.exit(1)

# Test 3: SciPy
try:
    import scipy
    from scipy.optimize import curve_fit
    print(f"\n[3/6] SciPy version: {scipy.__version__}")
    # Test curve fitting
    def func(x, a, b):
        return a * x + b
    xdata = np.array([1, 2, 3, 4])
    ydata = np.array([2.1, 4.0, 5.9, 8.1])
    popt, _ = curve_fit(func, xdata, ydata)
    print(f"      Curve fit test: y = {popt[0]:.2f}x + {popt[1]:.2f}")
    print("✓ SciPy OK")
except Exception as e:
    print(f"✗ SciPy FAILED: {e}")
    sys.exit(1)

# Test 4: Matplotlib
try:
    import matplotlib
    import matplotlib.pyplot as plt
    print(f"\n[4/6] Matplotlib version: {matplotlib.__version__}")
    print(f"      Backend: {matplotlib.get_backend()}")
    # Don't actually create plots in test
    print("✓ Matplotlib OK")
except Exception as e:
    print(f"✗ Matplotlib FAILED: {e}")
    sys.exit(1)

# Test 5: PyGAD
try:
    import pygad
    print(f"\n[5/6] PyGAD version: {pygad.__version__}")
    # Test creating a simple GA instance
    def fitness_test(ga, solution, idx):
        return -sum(solution)
    ga = pygad.GA(
        num_generations=2,
        sol_per_pop=10,
        num_parents_mating=5,
        num_genes=5,
        gene_space={'low': 0, 'high': 10},
        fitness_func=fitness_test
    )
    print(f"      GA test: Created instance with {ga.num_genes} genes")
    print("✓ PyGAD OK")
except Exception as e:
    print(f"✗ PyGAD FAILED: {e}")
    sys.exit(1)

# Test 6: TMM core
try:
    from tmm_core_test import coh_tmm
    print(f"\n[6/6] TMM Core Module")
    # Test simple calculation
    n_list = [1.0, 1.5, 1.0]
    d_list = [np.inf, 100.0, np.inf]
    result = coh_tmm('s', n_list, d_list, 0.0, 1550.0)
    print(f"      TMM test: T={result['T']:.4f}, R={result['R']:.4f}")
    print("✓ TMM Core OK")
except Exception as e:
    print(f"✗ TMM Core FAILED: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ ALL TESTS PASSED!")
print("=" * 60)
print("\nYour environment is ready to run the optimization code.")
print("\nNext steps:")
print("  1. Run Gaussian optimization: python gaussian_opt_write.py")
print("  2. Run lens optimization: python lens_opt_write_angle.py")
print("  3. See SETUP.md for detailed usage instructions")
print()

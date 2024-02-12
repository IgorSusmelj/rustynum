# bindings/python/tests/test_dot_product.py

import numpy as np
import rustynum as rnp


def test_mean_f32_small():
    a = [1.0, 2.0, 3.0, 4.0]
    a_py = rnp.NumArray(a, dtype="float32")
    result_rusty_1 = a_py.mean()
    result_rusty_2 = rnp.mean_f32(a_py)
    result_numpy = np.mean(a)

    assert np.isclose(
        result_rusty_1, result_numpy, atol=1e-6
    ), "Mean for f32 failed with error"

    assert np.isclose(
        result_rusty_2, result_numpy, atol=1e-6
    ), "Mean for f32 failed with error"


def test_mean_f32_random():
    # Generate two random f32 vectors of size 1000
    a = np.random.rand(1000).astype(np.float32)

    # Create NumArray instances
    a_py = rnp.NumArray(a.tolist(), dtype="float32")

    # Calculate and compare the mean
    result_rust = a_py.mean()
    result_numpy = np.mean(a)
    assert np.isclose(
        result_rust, result_numpy, atol=1e-6
    ), "Mean for f32 failed with error"


def test_mean_f64_random_large():
    # Generate two random f64 vectors of size 10000
    a = np.random.rand(10000).astype(np.float64)

    # Create NumArray instances
    a_py = rnp.NumArray(a.tolist(), dtype="float64")

    # Calculate and compare the mean
    result_rust = a_py.mean()
    result_numpy = np.mean(a)
    assert np.isclose(
        result_rust, result_numpy, atol=1e-9
    ), "Dot product for f64 failed with error"

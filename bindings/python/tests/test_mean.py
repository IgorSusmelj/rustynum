# bindings/python/tests/test_dot_product.py

import numpy as np
import rustynum as rnp


def test_mean_f32_small():
    a = [1.0, 2.0, 3.0, 4.0]
    a_py = rnp.NumArray(a, dtype="float32")
    result_rusty_1 = a_py.mean().item()
    result_rusty_2 = rnp.mean(a_py).item()
    result_numpy = np.mean(a).item()

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
    result_rust = a_py.mean().item()
    result_numpy = a.mean().item()

    print(result_rust, result_numpy)
    assert np.isclose(
        result_rust, result_numpy, atol=1e-9
    ), "Mean for f32 failed with error"


def test_mean_f32_axes():
    # Create a 2D array and wrap it in NumArray
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    a_py = rnp.NumArray(a.tolist(), dtype="float32")

    a = a.reshape((2, 3))
    a_py = a_py.reshape([2, 3])

    # Compute means along different axes
    result_rusty_axis0 = a_py.mean(axes=0)
    result_rusty_axis1 = a_py.mean(axes=1)
    result_numpy_axis0 = np.mean(a, axis=0)
    result_numpy_axis1 = np.mean(a, axis=1)

    print(result_rusty_axis0.tolist(), result_numpy_axis0)
    print(result_rusty_axis1.tolist(), result_numpy_axis1)

    # Check if the means are close
    assert np.allclose(
        result_rusty_axis0.tolist(), result_numpy_axis0, atol=1e-6
    ), "Mean along axis 0 for f32 failed"

    assert np.allclose(
        result_rusty_axis1.tolist(), result_numpy_axis1, atol=1e-6
    ), "Mean along axis 1 for f32 failed"


def test_mean_f64_random_large():
    # Generate two random f64 vectors of size 10000
    a = np.random.rand(10000).astype(np.float64)

    # Create NumArray instances
    a_py = rnp.NumArray(a.tolist(), dtype="float64")

    # Calculate and compare the mean
    result_rust = a_py.mean().item()
    result_numpy = a.mean().item()

    assert np.isclose(
        result_rust, result_numpy, atol=1e-12
    ), "Mean for f64 failed with error"

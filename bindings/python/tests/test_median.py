# bindings/python/tests/test_dot_product.py

import numpy as np
import rustynum as rnp


def test_median_f32_small():
    a = [1.0, 2.0, 3.0, 4.0]
    a_py = rnp.NumArray(a, dtype="float32")
    result_rusty_1 = a_py.median().item()
    result_rusty_2 = rnp.median(a_py).item()
    result_numpy = np.median(a).item()

    assert np.isclose(
        result_rusty_1, result_numpy, atol=1e-6
    ), "Mean for f32 failed with error"

    assert np.isclose(
        result_rusty_2, result_numpy, atol=1e-6
    ), "Mean for f32 failed with error"


def test_median_f32_axis():
    # Create a 2D array and wrap it in NumArray
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    a_py = rnp.NumArray(a.tolist(), dtype="float32")

    a = a.reshape((2, 3))
    a_py = a_py.reshape([2, 3])

    # Compute median along different axis
    result_rusty_axis0 = a_py.median(axis=0)
    result_rusty_axis1 = a_py.median(axis=1)
    result_numpy_axis0 = np.median(a, axis=0)
    result_numpy_axis1 = np.median(a, axis=1)

    print(result_rusty_axis0.tolist(), result_numpy_axis0)
    print(result_rusty_axis1.tolist(), result_numpy_axis1)

    # Check if the means are close
    assert np.allclose(
        result_rusty_axis0.tolist(), result_numpy_axis0, atol=1e-6
    ), "Mean along axis 0 for f32 failed"

    assert np.allclose(
        result_rusty_axis1.tolist(), result_numpy_axis1, atol=1e-6
    ), "Mean along axis 1 for f32 failed"

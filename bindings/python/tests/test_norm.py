import math

import numpy as np
import rustynum as rnp


def test_norm_f32_full_reduction():
    # Test full reduction (vector norm) for float32 arrays.
    data = [3.0, 4.0, 12.0]
    arr = rnp.NumArray(data, dtype="float32")

    # Test with p=2 (L2 norm)
    result = arr.norm(p=2)
    # Since full reduction, result is 1D with single element
    result_val = (
        result.tolist()[0] if isinstance(result.tolist(), list) else result.tolist()
    )
    expected = np.linalg.norm(np.array(data, dtype=np.float32), ord=2)
    assert math.isclose(
        result_val, expected, rel_tol=1e-6
    ), f"f32 L2 norm failed: {result_val} != {expected}"

    # Test with p=1 (L1 norm)
    result = arr.norm(p=1)
    result_val = (
        result.tolist()[0] if isinstance(result.tolist(), list) else result.tolist()
    )
    expected = np.linalg.norm(np.array(data, dtype=np.float32), ord=1)
    assert math.isclose(
        result_val, expected, rel_tol=1e-6
    ), f"f32 L1 norm failed: {result_val} != {expected}"


def test_norm_f32_axis():
    # Test norm with an axis reduction for a 2D float32 array.
    data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    arr = rnp.NumArray(data, dtype="float32")

    # Reduce along rows (axis=0): for each column, compute the norm
    result = arr.norm(p=2, axis=[0])
    # Expected: compute L2 norm along axis 0 (each column)
    expected = np.linalg.norm(np.array(data, dtype=np.float32), ord=2, axis=0)
    np.testing.assert_allclose(
        result.tolist(), expected, rtol=1e-6, err_msg="f32 norm along axis 0 failed."
    )

    # Reduce along columns (axis=1): for each row, compute the norm
    result = arr.norm(p=2, axis=[1])
    expected = np.linalg.norm(np.array(data, dtype=np.float32), ord=2, axis=1)
    np.testing.assert_allclose(
        result.tolist(), expected, rtol=1e-6, err_msg="f32 norm along axis 1 failed."
    )


def test_norm_f64_full_reduction():
    # Test full reduction (vector norm) for float64 arrays.
    data = [3.0, 4.0, 12.0]
    arr = rnp.NumArray(data, dtype="float64")

    # Test with p=2 (L2 norm)
    result = arr.norm(p=2)
    result_val = (
        result.tolist()[0] if isinstance(result.tolist(), list) else result.tolist()
    )
    expected = np.linalg.norm(np.array(data, dtype=np.float64), ord=2)
    assert math.isclose(
        result_val, expected, rel_tol=1e-12
    ), f"f64 L2 norm failed: {result_val} != {expected}"

    # Test with p=1 (L1 norm)
    result = arr.norm(p=1)
    result_val = (
        result.tolist()[0] if isinstance(result.tolist(), list) else result.tolist()
    )
    expected = np.linalg.norm(np.array(data, dtype=np.float64), ord=1)
    assert math.isclose(
        result_val, expected, rel_tol=1e-12
    ), f"f64 L1 norm failed: {result_val} != {expected}"


def test_norm_f64_axis():
    # Test norm with an axis reduction for a 2D float64 array.
    data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    arr = rnp.NumArray(data, dtype="float64")

    # Reduce along rows (axis=0)
    result = arr.norm(p=2, axis=[0])
    expected = np.linalg.norm(np.array(data, dtype=np.float64), ord=2, axis=0)
    np.testing.assert_allclose(
        result.tolist(), expected, rtol=1e-12, err_msg="f64 norm along axis 0 failed."
    )

    # Reduce along columns (axis=1)
    result = arr.norm(p=2, axis=[1])
    expected = np.linalg.norm(np.array(data, dtype=np.float64), ord=2, axis=1)
    np.testing.assert_allclose(
        result.tolist(), expected, rtol=1e-12, err_msg="f64 norm along axis 1 failed."
    )


if __name__ == "__main__":
    test_norm_f32_full_reduction()
    test_norm_f32_axis()
    test_norm_f64_full_reduction()
    test_norm_f64_axis()
    print("All norm tests passed.")

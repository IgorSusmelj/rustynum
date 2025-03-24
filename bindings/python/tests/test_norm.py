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


def test_l2_normalization():
    data = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ]

    # RustyNum implementation
    arr_rn = rnp.NumArray(data, dtype="float32")
    # Use keepdims=True to maintain the shape for broadcasting
    norms_rn = arr_rn.norm(p=2, axis=[1], keepdims=True)
    normalized_rn = arr_rn / norms_rn

    print(arr_rn.shape, norms_rn.shape, normalized_rn.shape)

    # NumPy implementation for comparison
    arr_np = np.array(data, dtype=np.float32)
    norms_np = np.linalg.norm(arr_np, ord=2, axis=1, keepdims=True)
    normalized_np = arr_np / norms_np

    # check if shape is the same
    assert normalized_rn.shape == normalized_np.shape

    np.testing.assert_allclose(
        normalized_rn.tolist(),
        normalized_np,
        rtol=1e-6,
        err_msg="L2 normalization results differ between RustyNum and NumPy",
    )

    # Verify that all rows now have unit norm
    row_norms = [np.linalg.norm(row, ord=2) for row in normalized_rn.tolist()]
    np.testing.assert_allclose(
        row_norms,
        np.ones(len(data)),
        rtol=1e-6,
        err_msg="Normalized rows do not have unit norm",
    )

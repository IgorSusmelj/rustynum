# bindings/python/tests/test_dot_product.py

import numpy as np
import pytest
import rustynum as rnp


def test_min_f32_small():
    a = [1.0, 2.0, 3.0, 4.0]
    a_py = rnp.NumArray(a, dtype="float32")
    result_rusty_1 = a_py.min()
    result_rusty_2 = rnp.min(a_py)
    result_numpy = np.min(a)

    assert np.isclose(
        result_rusty_1, result_numpy, atol=1e-6
    ), "Min for f32 failed with error"

    assert np.isclose(
        result_rusty_2, result_numpy, atol=1e-6
    ), "Min for f32 failed with error"


def test_max_f32_small():
    a = [1.0, 2.0, 3.0, 4.0]
    a_py = rnp.NumArray(a, dtype="float32")
    result_rusty_1 = a_py.max()
    result_rusty_2 = rnp.max(a_py)
    result_numpy = np.max(a)

    assert np.isclose(
        result_rusty_1, result_numpy, atol=1e-6
    ), "Max for f32 failed with error"

    assert np.isclose(
        result_rusty_2, result_numpy, atol=1e-6
    ), "Max for f32 failed with error"


def test_min_axis_2d():
    # Test min along different axis of a 2D array
    a = [[1.0, 2.0], [3.0, 4.0]]
    a_py = rnp.NumArray(a, dtype="float32")

    # Test min along axis 0 (columns)
    result_axis0 = a_py.min(axis=0)
    expected_axis0 = rnp.NumArray([1.0, 2.0], dtype="float32")
    assert result_axis0.tolist() == expected_axis0.tolist(), "Min along axis 0 failed"

    # Test min along axis 1 (rows)
    result_axis1 = a_py.min(axis=1)
    expected_axis1 = rnp.NumArray([1.0, 3.0], dtype="float32")
    assert result_axis1.tolist() == expected_axis1.tolist(), "Min along axis 1 failed"


def test_max_axis_2d():
    # Test max along different axis of a 2D array
    a = [[1.0, 2.0], [3.0, 4.0]]
    a_py = rnp.NumArray(a, dtype="float32")

    # Test max along axis 0 (columns)
    result_axis0 = a_py.max(axis=0)
    expected_axis0 = rnp.NumArray([3.0, 4.0], dtype="float32")
    assert result_axis0.tolist() == expected_axis0.tolist(), "Max along axis 0 failed"

    # Test max along axis 1 (rows)
    result_axis1 = a_py.max(axis=1)
    expected_axis1 = rnp.NumArray([2.0, 4.0], dtype="float32")
    assert result_axis1.tolist() == expected_axis1.tolist(), "Max along axis 1 failed"


def test_min_max_3d():
    # Test min/max with 3D array and multiple axis
    a = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
    a_py = rnp.NumArray(a, dtype="float32")

    # Test min along axis (0,1)
    result_min = a_py.min(axis=[0, 1])
    expected_min = rnp.NumArray([1.0, 2.0], dtype="float32")
    assert result_min.tolist() == expected_min.tolist(), "Min along axis (0,1) failed"

    # Test max along axis (1,2)
    result_max = a_py.max(axis=[1, 2])
    expected_max = rnp.NumArray([4.0, 8.0], dtype="float32")
    assert result_max.tolist() == expected_max.tolist(), "Max along axis (1,2) failed"


def test_min_max_edge_cases():
    # Test edge cases

    # Single element array
    a_single = rnp.NumArray([1.0], dtype="float32")
    assert a_single.min() == 1.0, "Min failed for single element array"
    assert a_single.max() == 1.0, "Max failed for single element array"

    # Empty axis reduction
    a_empty_axis = rnp.NumArray([[1.0], [2.0]], dtype="float32")
    result_min = a_empty_axis.min(axis=1)
    assert result_min.tolist() == [1.0, 2.0], "Min failed for empty axis reduction"

    # Test with negative values
    a_neg = rnp.NumArray([-1.0, -2.0, -3.0], dtype="float32")
    assert a_neg.min() == -3.0, "Min failed with negative values"
    assert a_neg.max() == -1.0, "Max failed with negative values"


def test_min_max_f64_small():
    # Test with float64 data type
    a = [1.0, 2.0, 3.0, 4.0]
    a_py = rnp.NumArray(a, dtype="float64")
    result_rusty_1 = a_py.min()
    result_rusty_2 = rnp.min(a_py)
    result_numpy = np.min(a)

    assert np.isclose(
        result_rusty_1, result_numpy, atol=1e-15
    ), "Min for f64 failed with error"

    assert np.isclose(
        result_rusty_2, result_numpy, atol=1e-15
    ), "Min for f64 failed with error"

    # Test max
    result_rusty_1 = a_py.max()
    result_rusty_2 = rnp.max(a_py)
    result_numpy = np.max(a)

    assert np.isclose(
        result_rusty_1, result_numpy, atol=1e-15
    ), "Max for f64 failed with error"

    assert np.isclose(
        result_rusty_2, result_numpy, atol=1e-15
    ), "Max for f64 failed with error"


def test_min_max_f64_axis():
    # Test f64 axis operations specifically to ensure exports work
    data = [[1.0, 4.0, 2.0], [3.0, 1.0, 5.0]]
    a = rnp.NumArray(data, dtype="float64")

    # Test min along axis 0
    min_axis0 = a.min(axis=0)
    expected_min = [1.0, 1.0, 2.0]
    assert min_axis0.tolist() == expected_min, "F64 min along axis 0 failed"

    # Test max along axis 1
    max_axis1 = a.max(axis=1)
    expected_max = [4.0, 5.0]
    assert max_axis1.tolist() == expected_max, "F64 max along axis 1 failed"

    # Test with multiple axes
    data_3d = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
    b = rnp.NumArray(data_3d, dtype="float64")

    min_multi_axis = b.min(axis=[0, 1])
    expected_min_multi = [1.0, 2.0]
    assert (
        min_multi_axis.tolist() == expected_min_multi
    ), "F64 min along multiple axes failed"


def test_min_max_dtype_errors():
    # Test with unsupported integer dtype
    with pytest.raises(NotImplementedError):
        rnp.NumArray([1, 2, 3, 4], dtype="int32")

    with pytest.raises(NotImplementedError):
        rnp.NumArray([1, 2, 3, 4], dtype="int64")

    # Test with invalid dtype
    with pytest.raises(ValueError):
        rnp.NumArray([1, 2, 3, 4], dtype="invalid_type")

import numpy as np
import rustynum as rnp


# test addition of scalar to numarray and then slicing
def test_add_scalar_and_slice():
    a = rnp.NumArray([1.0, 2.0, 3.0, 4.0], dtype="float32")
    result = a + 1.0
    sliced = result[1:3].tolist()
    assert sliced == [3.0, 4.0], "Addition of scalar and slicing failed"


# test addition of numarray to numarray and then slicing
def test_add_array_and_slice():
    a = rnp.NumArray([1.0, 2.0, 3.0, 4.0], dtype="float32")
    b = rnp.NumArray([4.0, 3.0, 2.0, 1.0], dtype="float32")
    result = a + b
    sliced = result[1:3].tolist()
    assert sliced == [5.0, 5.0], "Addition of NumArray and slicing failed"


# test addition of scalar to numarray and then computing the mean
def test_add_scalar_and_mean():
    a = rnp.NumArray([1.0, 2.0, 3.0, 4.0], dtype="float32")
    result = a + 1.0
    mean = result.mean().item()
    assert np.isclose(mean, 3.5, atol=1e-6), "Addition of scalar and mean failed"


def test_multiple_additions():
    result = rnp.NumArray([1.0, 2.0, 3.0, 4.0], dtype="float32")
    result_numpy = np.array([1.0, 2.0, 3.0, 4.0], dtype="float32")

    for _ in range(10):
        result = result + 1.0
        result_numpy = result_numpy + 1.0

    result = result + rnp.NumArray([1.0, 2.0, 3.0, 4.0], dtype="float32")
    result_numpy = result_numpy + np.array([1.0, 2.0, 3.0, 4.0], dtype="float32")

    assert result.tolist() == result_numpy.tolist(), "Multiple additions failed"

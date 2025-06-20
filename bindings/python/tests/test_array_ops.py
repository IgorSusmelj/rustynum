import numpy as np
import rustynum as rnp


def test_add_scalar():
    a = rnp.NumArray([1.0] * 20, dtype="float32")
    result = a + 1.0
    assert result.tolist() == [2.0] * 20, "Addition of scalar failed"


def test_add_array():
    a = rnp.NumArray([1.0] * 20, dtype="float32")
    b = rnp.NumArray([4.0] * 20, dtype="float32")
    result = a + b
    assert result.tolist() == [5.0] * 20, "Addition of NumArray failed"


def test_subtract_scalar():
    a = rnp.NumArray([1.0] * 20, dtype="float32")
    result = a - 1.0
    assert result.tolist() == [0.0] * 20, "Subtraction of scalar failed"


def test_subtract_array():
    a = rnp.NumArray([1.0] * 20, dtype="float32")
    b = rnp.NumArray([4.0] * 20, dtype="float32")
    result = a - b
    assert result.tolist() == [-3.0] * 20, "Subtraction of NumArray failed"


def test_multiply_scalar():
    a = rnp.NumArray([1.0] * 20, dtype="float32")
    result = a * 2.0
    assert result.tolist() == [2.0] * 20, "Multiplication of scalar failed"


def test_multiply_array():
    a = rnp.NumArray([1.0] * 20, dtype="float32")
    b = rnp.NumArray([4.0] * 20, dtype="float32")
    result = a * b
    assert result.tolist() == [4.0] * 20, "Multiplication of NumArray failed"


def test_inplace_multiply_scalar_f32():
    a = rnp.NumArray([2.0, 4.0, 6.0], dtype="float32")
    original_id = id(a)
    a *= 3.0
    assert id(a) == original_id, "In-place multiplication should modify the same object"
    assert a.tolist() == [6.0, 12.0, 18.0], "In-place multiplication failed for f32"


def test_inplace_multiply_scalar_f64():
    a = rnp.NumArray([2.0, 4.0, 6.0], dtype="float64")
    original_id = id(a)
    a *= 2.5
    assert id(a) == original_id, "In-place multiplication should modify the same object"
    assert a.tolist() == [5.0, 10.0, 15.0], "In-place multiplication failed for f64"


def test_inplace_multiply_scalar_u8():
    a = rnp.NumArray([2, 4, 6], dtype="uint8")
    original_id = id(a)
    a *= 3
    assert id(a) == original_id, "In-place multiplication should modify the same object"
    assert a.tolist() == [6, 12, 18], "In-place multiplication failed for uint8"


def test_inplace_multiply_preserves_shape():
    data = [[1.0, 2.0], [3.0, 4.0]]
    a = rnp.NumArray(data, dtype="float32")
    original_shape = a.shape
    a *= 2.0
    assert a.shape == original_shape, "In-place multiplication should preserve shape"
    assert a.tolist() == [[2.0, 4.0], [6.0, 8.0]], "2D in-place multiplication failed"


def test_divide_scalar():
    a = rnp.NumArray([1.0] * 20, dtype="float32")
    result = a / 2.0
    assert result.tolist() == [0.5] * 20, "Division of scalar failed"


def test_divide_array():
    a = rnp.NumArray([1.0] * 20, dtype="float32")
    b = rnp.NumArray([4.0] * 20, dtype="float32")
    result = a / b
    assert result.tolist() == [0.25] * 20, "Division of NumArray failed"


def test_slicing():
    a = rnp.NumArray([1.0] * 20, dtype="float32")
    sliced = a[1:3].tolist()
    assert sliced == [1.0, 1.0], "Slicing failed"


def test_slicing_f32():
    a = rnp.NumArray([1.0, 2.0, 3.0, 4.0], dtype="float32")
    sliced = a[1:3].tolist()
    assert sliced == [2.0, 3.0], "Slicing failed"


def test_slicing_f64():
    a = rnp.NumArray([1.0, 2.0, 3.0, 4.0], dtype="float64")
    sliced = a[1:3].tolist()
    assert sliced == [2.0, 3.0], "Slicing failed"

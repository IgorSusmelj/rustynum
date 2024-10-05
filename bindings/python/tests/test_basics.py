import numpy as np
import rustynum as rnp


def test_flat_explicit_dtype_f32():
    a = rnp.NumArray([1.0, 2.0, 3.0], dtype="float32")
    b = np.array([1.0, 2.0, 3.0], dtype="float32")
    assert np.allclose(a.tolist(), b, atol=1e-9), "Flat array creation failed"


def test_flat_explicit_dtype_u8():
    a = rnp.NumArray([1, 2, 3], dtype="uint8")
    b = np.array([1, 2, 3], dtype="uint8")
    assert np.allclose(a.tolist(), b, atol=1e-9), "Flat array creation failed"


def test_nested_explicit_dtype_f32():
    a = rnp.NumArray([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
    b = np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
    assert np.allclose(a.tolist(), b, atol=1e-9), "Nested array creation failed"


def test_nested_explicit_dtype_u8():
    a = rnp.NumArray([[1, 2], [3, 4]], dtype="uint8")
    b = np.array([[1, 2], [3, 4]], dtype="uint8")
    assert np.allclose(a.tolist(), b, atol=1e-9), "Nested array creation failed"


def test_flat_inferred_dtype_f32():
    a = rnp.NumArray([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 3.0], dtype="float32")
    assert np.allclose(a.tolist(), b, atol=1e-9), "Flat array creation failed"
    assert a.dtype == "float32", "Inferred dtype failed"


def test_numarray_creation_from_numarray_f32():
    a = rnp.NumArray([1.0, 2.0, 3.0], dtype="float32")
    b = rnp.NumArray(a)
    # Modify the original NumArray to make sure we are not just copying the reference
    a += 1
    b += 1
    assert a.tolist() == b.tolist(), "NumArray creation from NumArray failed"
    assert a.dtype == b.dtype, "NumArray creation from NumArray failed"


def test_numarray_creation_from_numarray_u8():
    a = rnp.NumArray([1, 2, 3], dtype="uint8")
    b = rnp.NumArray(a)
    # Modify the original NumArray to make sure we are not just copying the reference
    a += 1
    b += 1
    assert a.tolist() == b.tolist(), "NumArray creation from NumArray failed"
    assert a.dtype == b.dtype, "NumArray creation from NumArray failed"


def test_zeros():
    a = rnp.zeros((2, 3), dtype="float32")
    b = np.zeros((2, 3), dtype="float32")
    assert a.tolist() == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], "Zeros failed"
    assert np.allclose(a.tolist(), b, atol=1e-9), "Zeros failed"


def test_ones():
    a = rnp.ones((2, 3), dtype="float32")
    b = np.ones((2, 3), dtype="float32")
    assert a.tolist() == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], "Ones failed"
    assert np.allclose(a.tolist(), b, atol=1e-9), "Ones failed"


def test_arange():
    a = rnp.arange(0, 10, 2, dtype="float32")
    b = np.arange(0, 10, 2, dtype="float32")
    assert a.tolist() == [0.0, 2.0, 4.0, 6.0, 8.0], "Arange failed"
    assert np.allclose(a.tolist(), b, atol=1e-9), "Arange failed"


def test_linspace():
    a = rnp.linspace(0, 10, 5, dtype="float32")
    b = np.linspace(0, 10, 5, dtype="float32")
    assert a.tolist() == [0.0, 2.5, 5.0, 7.5, 10.0], "Linspace failed"
    assert np.allclose(a.tolist(), b, atol=1e-9), "Linspace failed"


def test_exp():
    a = rnp.NumArray([0.0, 1.0, 2.0, 3.0], dtype="float32")
    b = np.exp(np.array([0.0, 1.0, 2.0, 3.0], dtype="float32"))
    assert np.allclose(a.exp().tolist(), b, atol=1e-9), "Exp failed"


def test_log():
    a = rnp.NumArray([1.0, 2.0, 4.0, 8.0], dtype="float32")
    b = np.log(np.array([1.0, 2.0, 4.0, 8.0], dtype="float32"))
    assert np.allclose(a.log().tolist(), b, atol=1e-9), "Log failed"


def test_sigmoid():
    a = rnp.NumArray([0.0, 1.0, 2.0, 3.0], dtype="float32")
    b = 1 / (1 + np.exp(-np.array([0.0, 1.0, 2.0, 3.0], dtype="float32")))
    assert np.allclose(a.sigmoid().tolist(), b, atol=1e-9), "Sigmoid failed"


def test_concatenate_two_1d_arrays():
    a = rnp.NumArray([1.0, 2.0, 3.0], dtype="float32")
    b = rnp.NumArray([4.0, 5.0, 6.0], dtype="float32")
    c = np.concatenate(
        [
            np.array([1.0, 2.0, 3.0], dtype="float32"),
            np.array([4.0, 5.0, 6.0], dtype="float32"),
        ]
    )

    assert rnp.concatenate([a, b]).shape == c.shape, "Shape mismatch"
    assert np.allclose(
        rnp.concatenate([a, b]).tolist(), c, atol=1e-9
    ), "Concatenate failed"


def test_concatenate_along_axis_0():
    a = rnp.NumArray([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
    b = rnp.NumArray([[5.0, 6.0], [7.0, 8.0]], dtype="float32")
    c = np.concatenate(
        [
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32"),
            np.array([[5.0, 6.0], [7.0, 8.0]], dtype="float32"),
        ],
        axis=0,
    )

    assert rnp.concatenate([a, b], axis=0).shape == c.shape, "Shape mismatch"
    assert np.allclose(
        rnp.concatenate([a, b], axis=0).tolist(), c, atol=1e-9
    ), "Concatenate failed"


def test_concatenate_along_axis_1():
    a = rnp.NumArray([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
    b = rnp.NumArray([[5.0, 6.0], [7.0, 8.0]], dtype="float32")
    c = np.concatenate(
        [
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32"),
            np.array([[5.0, 6.0], [7.0, 8.0]], dtype="float32"),
        ],
        axis=1,
    )

    assert rnp.concatenate([a, b], axis=1).shape == c.shape, "Shape mismatch"
    assert np.allclose(
        rnp.concatenate([a, b], axis=1).tolist(), c, atol=1e-9
    ), "Concatenate failed"


def test_flip_axis_0():
    a = rnp.NumArray([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
    b = np.flip(np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32"), axis=0)
    assert np.allclose(a.flip(axis=0).tolist(), b, atol=1e-9), "Flip axis 0 failed"
    assert a.flip(axis=0).shape == b.shape, "Shape mismatch"


def test_flip_axis_1():
    a = rnp.NumArray([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
    b = np.flip(np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32"), axis=1)
    assert np.allclose(a.flip(axis=1).tolist(), b, atol=1e-9), "Flip axis 1 failed"
    assert a.flip(axis=1).shape == b.shape, "Shape mismatch"


def test_flip_multiple_axis():
    a = rnp.NumArray(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype="float32"
    )
    b = np.flip(
        np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype="float32"),
        axis=(0, 1),
    )
    assert np.allclose(
        a.flip(axis=(0, 1)).tolist(), b, atol=1e-9
    ), "Flip multiple axis failed"
    assert a.flip(axis=(0, 1)).shape == b.shape, "Shape mismatch"


def test_fancy_index_flipping_f32():
    a = rnp.NumArray([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
    b = np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32")

    assert np.allclose(
        a[:, ::-1].tolist(), b[:, ::-1], atol=1e-9
    ), "Fancy indexing failed"


def test_fancy_index_flipping_u8():
    a = rnp.NumArray([[1, 2], [3, 4]], dtype="uint8")
    b = np.array([[1, 2], [3, 4]], dtype="uint8")

    assert np.allclose(
        a[:, ::-1].tolist(), b[:, ::-1], atol=1e-9
    ), "Fancy indexing failed"


def test_fancy_index_flipping_3d_f32():
    data = [
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
        [[13.0, 14.0], [15.0, 16.0], [17.0, 18.0]],
    ]

    a = rnp.NumArray(data, dtype="float32")
    b = np.array(data, dtype="float32")

    assert np.allclose(
        a[:, :, ::-1].tolist(), b[:, :, ::-1], atol=1e-9
    ), "Fancy indexing failed"
    assert a[:, :, ::-1].shape == b[:, :, ::-1].shape, "Shape mismatch"

    assert np.allclose(
        a[:, ::-1, :].tolist(), b[:, ::-1, :], atol=1e-9
    ), "Fancy indexing failed"
    assert a[:, ::-1, :].shape == b[:, ::-1, :].shape, "Shape mismatch"


def test_fancy_index_flipping_3d_u8():
    data = [
        [[1, 2], [3, 4], [5, 6]],
        [[7, 8], [9, 10], [11, 12]],
        [[13, 14], [15, 16], [17, 18]],
    ]

    a = rnp.NumArray(data, dtype="uint8")
    b = np.array(data, dtype="uint8")

    assert np.allclose(
        a[:, :, ::-1].tolist(), b[:, :, ::-1], atol=1e-9
    ), "Fancy indexing failed"
    assert a[:, :, ::-1].shape == b[:, :, ::-1].shape, "Shape mismatch"

    assert np.allclose(
        a[:, ::-1, :].tolist(), b[:, ::-1, :], atol=1e-9
    ), "Fancy indexing failed"
    assert a[:, ::-1, :].shape == b[:, ::-1, :].shape, "Shape mismatch"

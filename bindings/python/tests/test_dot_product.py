# bindings/python/tests/test_dot_product.py

import numpy as np
import rustynum as rnp


def test_dot_product():
    # Using the generic NumArray class with dtype specified
    a = rnp.NumArray([1.0, 2.0, 3.0, 4.0], dtype="float32")
    b = rnp.NumArray([4.0, 3.0, 2.0, 1.0], dtype="float32")
    result = a.dot(b).tolist()  # Directly using the dot method on the NumArray instance
    result2 = rnp.dot(a, b).tolist()  # Using the dot function from the module
    assert np.isclose(result, 20.0, atol=1e-9), "Dot product failed"
    assert np.isclose(result2, 20.0, atol=1e-9), "Dot product failed"


def test_dot_product_f64():
    # Similar test but for float64 dtype
    a = rnp.NumArray([1.0, 2.0, 3.0, 4.0], dtype="float64")
    b = rnp.NumArray([4.0, 3.0, 2.0, 1.0], dtype="float64")
    result = a.dot(b).tolist()
    result2 = rnp.dot(a, b).tolist()
    assert np.isclose(result, 20.0, atol=1e-12), "Dot product for f64 failed"
    assert np.isclose(result2, 20.0, atol=1e-12), "Dot product for f64 failed"


def test_dot_product_f32_random():
    # Generate two random f32 vectors of size 1000
    a = np.random.rand(1000).astype(np.float32)
    b = np.random.rand(1000).astype(np.float32)

    # Create NumArray instances
    a_py = rnp.NumArray(a.tolist(), dtype="float32")
    b_py = rnp.NumArray(b.tolist(), dtype="float32")

    # Calculate and compare the dot products
    result_rust = a_py.dot(b_py).tolist()
    result_numpy = np.dot(a, b)
    assert np.isclose(
        result_rust, result_numpy, atol=1e-9
    ), "Dot product for f32 failed with error"


def test_dot_product_f64_random_large():
    # Generate two random f64 vectors of size 10000
    a = np.random.rand(10000).astype(np.float64)
    b = np.random.rand(10000).astype(np.float64)

    # Create NumArray instances
    a_py = rnp.NumArray(a.tolist(), dtype="float64")
    b_py = rnp.NumArray(b.tolist(), dtype="float64")

    # Calculate and compare the dot products
    result_rust = a_py.dot(b_py).tolist()
    result_numpy = np.dot(a, b)
    assert np.isclose(
        result_rust, result_numpy, atol=1e-12
    ), "Dot product for f64 failed with error"


def test_dot_product_f32_matrix_vector():
    # Generate a random f32 matrix and vector
    a = np.random.rand(1000 * 1000).astype(np.float32)
    b = np.random.rand(1000).astype(np.float32)

    # Create NumArray instances
    a_py = rnp.NumArray(a.tolist(), dtype="float32")
    b_py = rnp.NumArray(b.tolist(), dtype="float32")

    a = a.reshape((1000, 1000))
    a_py = a_py.reshape([1000, 1000])

    # Calculate and compare the dot products
    result_rust = a_py.dot(b_py).tolist()
    result_numpy = np.dot(a, b)
    assert np.allclose(
        result_rust, result_numpy, atol=1e-9
    ), "Dot product for f32 matrix-vector failed with error"


def test_dot_product_f64_matrix_vector():
    # Generate a random f64 matrix and vector
    a = np.random.rand(1000 * 1000).astype(np.float64)
    b = np.random.rand(1000).astype(np.float64)

    # Create NumArray instances
    a_py = rnp.NumArray(a.tolist(), dtype="float64")
    b_py = rnp.NumArray(b.tolist(), dtype="float64")

    a = a.reshape((1000, 1000))
    a_py = a_py.reshape([1000, 1000])

    # Calculate and compare the dot products
    result_rust = a_py.dot(b_py).tolist()
    result_numpy = np.dot(a, b)
    assert np.allclose(
        result_rust, result_numpy, atol=1e-12
    ), "Dot product for f64 matrix-vector failed with error"


def test_dot_product_f32_matrix_matrix():
    # Generate two random f32 matrices
    a = np.random.rand(1000 * 1000).astype(np.float32)
    b = np.random.rand(1000 * 1000).astype(np.float32)

    # Create NumArray instances
    a_py = rnp.NumArray(a.tolist(), dtype="float32")
    b_py = rnp.NumArray(b.tolist(), dtype="float32")

    a = a.reshape((1000, 1000))
    a_py = a_py.reshape([1000, 1000])
    b = b.reshape((1000, 1000))
    b_py = b_py.reshape([1000, 1000])

    # Calculate and compare the dot products
    result_rust = a_py.dot(b_py)
    result_numpy = np.dot(a, b)

    assert result_numpy.shape == result_rust.shape, "Shapes do not match"
    assert np.allclose(
        result_rust.tolist(),
        result_numpy.tolist(),
        atol=1e-9,
    ), "Dot product for f32 matrix-matrix failed with error"

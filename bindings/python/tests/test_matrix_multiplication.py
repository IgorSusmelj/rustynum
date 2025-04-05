import numpy as np
import pytest

import rustynum as rnp


def test_matmul_f32_basic():
    """
    Test basic matrix-matrix multiplication with float32 dtype.
    """
    # Define two 2x2 matrices
    a_data = [[1.0, 2.0], [3.0, 4.0]]
    b_data = [[5.0, 6.0], [7.0, 8.0]]

    # Create NumArray instances
    a = rnp.NumArray(a_data, dtype="float32")
    b = rnp.NumArray(b_data, dtype="float32")

    # Perform matrix multiplication using the matmul method and @ operator
    result_method = a.matmul(b).tolist()
    result_operator = (a @ b).tolist()

    # Expected result using NumPy
    expected = np.matmul(
        np.array(a_data, dtype=np.float32), np.array(b_data, dtype=np.float32)
    ).tolist()

    # Assert the results are close to the expected values
    assert np.allclose(
        result_method, expected, atol=1e-6
    ), "Basic matmul for float32 failed using matmul method."
    assert np.allclose(
        result_operator, expected, atol=1e-6
    ), "Basic matmul for float32 failed using @ operator."


def test_matmul_f32_random():
    """
    Test matrix-matrix multiplication with random float32 matrices.
    """
    # Generate two random 100x100 float32 matrices using nested lists
    a = np.random.rand(100, 100).astype(np.float32).tolist()
    b = np.random.rand(100, 100).astype(np.float32).tolist()

    # Create NumArray instances
    a_py = rnp.NumArray(a, dtype="float32")
    b_py = rnp.NumArray(b, dtype="float32")

    # Perform matrix multiplication using the matmul method
    result_rust = a_py.matmul(b_py).tolist()

    # Expected result using NumPy
    a_np = np.array(a, dtype=np.float32)
    b_np = np.array(b, dtype=np.float32)
    expected = np.matmul(a_np, b_np).tolist()

    # Assert the results are close to the expected values
    assert np.allclose(
        result_rust, expected, atol=1e-5
    ), "Random matmul for float32 failed."


def test_matmul_f64_basic():
    """
    Test basic matrix-matrix multiplication with float64 dtype.
    """
    # Define two 2x2 matrices as nested lists
    a_data = [[1.0, 2.0], [3.0, 4.0]]
    b_data = [[5.0, 6.0], [7.0, 8.0]]

    # Create NumArray instances
    a = rnp.NumArray(a_data, dtype="float64")
    b = rnp.NumArray(b_data, dtype="float64")

    # Perform matrix multiplication using the matmul method and @ operator
    result_method = a.matmul(b).tolist()
    result_operator = (a @ b).tolist()

    # Expected result using NumPy
    expected = np.matmul(
        np.array(a_data, dtype=np.float64), np.array(b_data, dtype=np.float64)
    ).tolist()

    # Assert the results are close to the expected values
    assert np.allclose(
        result_method, expected, atol=1e-12
    ), "Basic matmul for float64 failed using matmul method."
    assert np.allclose(
        result_operator, expected, atol=1e-12
    ), "Basic matmul for float64 failed using @ operator."


def test_matmul_f64_random():
    """
    Test matrix-matrix multiplication with random float64 matrices.
    """
    # Generate two random 200x200 float64 matrices using nested lists
    a = np.random.rand(200, 200).astype(np.float64).tolist()
    b = np.random.rand(200, 200).astype(np.float64).tolist()

    # Create NumArray instances
    a_py = rnp.NumArray(a, dtype="float64")
    b_py = rnp.NumArray(b, dtype="float64")

    # Perform matrix multiplication using the matmul method
    result_rust = a_py.matmul(b_py).tolist()

    # Expected result using NumPy
    a_np = np.array(a, dtype=np.float64)
    b_np = np.array(b, dtype=np.float64)
    expected = np.matmul(a_np, b_np).tolist()

    # Assert the results are close to the expected values
    assert np.allclose(
        result_rust, expected, atol=1e-12
    ), "Random matmul for float64 failed."


def test_matmul_empty():
    """
    Test matrix-matrix multiplication with empty matrices.
    """
    # Define two empty matrices
    a_data = []
    b_data = []

    # Create NumArray instances
    a = rnp.NumArray(a_data, dtype="float32")
    b = rnp.NumArray(b_data, dtype="float32")

    # Attempt to perform matrix multiplication and expect an assertion error
    with pytest.raises(ValueError):
        a.matmul(b)

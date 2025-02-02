import numpy as np
import pytest
import rustynum as rnp


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def setup_vector(dtype, size=1000):
    """
    Generate a 1D vector as a Python list.
    """
    a = np.random.rand(size).astype(dtype)
    return a.tolist()


def setup_matrix(dtype, rows=100, cols=100):
    """
    Generate a 2D matrix as a Python list.
    """
    a = np.random.rand(rows, cols).astype(dtype)
    return a.tolist()


# -----------------------------------------------------------------------------
# Norm functions for 1D arrays
# -----------------------------------------------------------------------------
def norm1d_rustynum(a, dtype, p):
    """
    Compute the norm (L1 if p==1, L2 if p==2) of a 1D array using RustyNum.
    """
    a_rnp = rnp.NumArray(a, dtype=dtype)
    # Full reduction; no axis provided means compute the norm over all elements.
    return a_rnp.norm(p=p)


def norm1d_numpy(a, dtype, p):
    """
    Compute the norm (L1 if p==1, L2 if p==2) of a 1D array using NumPy.
    """
    a_np = np.array(a, dtype=dtype)
    return np.linalg.norm(a_np, ord=p)


# -----------------------------------------------------------------------------
# Norm functions for 2D arrays (full reduction)
# -----------------------------------------------------------------------------
def norm2d_rustynum(a, dtype, p):
    """
    Compute the norm (L1 if p==1, L2 if p==2) of a 2D array using RustyNum.
    Providing axis=None ensures a full reduction across all elements.
    """
    a_rnp = rnp.NumArray(a, dtype=dtype)
    return a_rnp.norm(p=p, axis=None)


def norm2d_numpy(a, dtype, p):
    """
    Compute a full reduction norm by flattening the matrix and using NumPy.
    """
    a_np = np.array(a, dtype=dtype)
    a_flat = a_np.flatten()
    return np.linalg.norm(a_flat, ord=p)


# -----------------------------------------------------------------------------
# Benchmark functions for 1D norm
# -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "func,dtype,size,p",
    [
        (norm1d_rustynum, "float32", 1000, 1),
        (norm1d_rustynum, "float64", 1000, 1),
        (norm1d_numpy, "float32", 1000, 1),
        (norm1d_numpy, "float64", 1000, 1),
        (norm1d_rustynum, "float32", 10000, 1),
        (norm1d_rustynum, "float64", 10000, 1),
        (norm1d_numpy, "float32", 10000, 1),
        (norm1d_numpy, "float64", 10000, 1),
        (norm1d_rustynum, "float32", 1000, 2),
        (norm1d_rustynum, "float64", 1000, 2),
        (norm1d_numpy, "float32", 1000, 2),
        (norm1d_numpy, "float64", 1000, 2),
        (norm1d_rustynum, "float32", 10000, 2),
        (norm1d_rustynum, "float64", 10000, 2),
        (norm1d_numpy, "float32", 10000, 2),
        (norm1d_numpy, "float64", 10000, 2),
    ],
)
def test_norm_1d(benchmark, func, dtype, size, p):
    """
    Benchmark for computing the norm (L1 and L2) on a 1D array.
    """
    a = setup_vector(dtype, size)
    group_name = f"norm_1d_{dtype}_p{p}"

    benchmark.group = group_name
    benchmark.extra_info["dtype"] = dtype
    benchmark.extra_info["size"] = size
    benchmark.extra_info["p"] = p
    benchmark(func, a, dtype, p)


# -----------------------------------------------------------------------------
# Benchmark functions for 2D norm
# -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "func,dtype,size,p",
    [
        (norm2d_rustynum, "float32", (100, 100), 1),
        (norm2d_rustynum, "float64", (100, 100), 1),
        (norm2d_numpy, "float32", (100, 100), 1),
        (norm2d_numpy, "float64", (100, 100), 1),
        (norm2d_rustynum, "float32", (500, 500), 1),
        (norm2d_rustynum, "float64", (500, 500), 1),
        (norm2d_numpy, "float32", (500, 500), 1),
        (norm2d_numpy, "float64", (500, 500), 1),
        (norm2d_rustynum, "float32", (100, 100), 2),
        (norm2d_rustynum, "float64", (100, 100), 2),
        (norm2d_numpy, "float32", (100, 100), 2),
        (norm2d_numpy, "float64", (100, 100), 2),
        (norm2d_rustynum, "float32", (500, 500), 2),
        (norm2d_rustynum, "float64", (500, 500), 2),
        (norm2d_numpy, "float32", (500, 500), 2),
        (norm2d_numpy, "float64", (500, 500), 2),
    ],
)
def test_norm_2d(benchmark, func, dtype, size, p):
    """
    Benchmark for computing the norm (L1 and L2) on a 2D array.
    The matrix is flattened in the NumPy version to match the full reduction behavior.
    """
    rows, cols = size
    a = setup_matrix(dtype, rows, cols)
    group_name = f"norm_2d_{dtype}_p{p}_{rows}x{cols}"

    benchmark.group = group_name
    benchmark.extra_info["dtype"] = dtype
    benchmark.extra_info["size"] = size
    benchmark.extra_info["p"] = p
    benchmark(func, a, dtype, p)

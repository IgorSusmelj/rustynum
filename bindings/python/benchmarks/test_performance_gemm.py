import numpy as np
import pytest

import rustynum as rnp


# Helper function to generate random vectors
def setup_vector(dtype, size=1000):
    a = np.random.rand(size * size).astype(dtype)
    b = np.random.rand(size * size).astype(dtype)
    return a.tolist(), b.tolist()


# Function to perform gemm using rustynum
def gemm_rustynum(a, b, size, dtype):
    a_rnp = rnp.NumArray(a, dtype=dtype).reshape([size, size])
    b_rnp = rnp.NumArray(b, dtype=dtype).reshape([size, size])
    return a_rnp.dot(b_rnp)


# Function to perform gemv using numpy
def gemm_numpy(a, b, size, dtype):
    a_np = np.array(a, dtype=dtype).reshape((size, size))
    b_np = np.array(b, dtype=dtype).reshape((size, size))
    return np.dot(a_np, b_np)


# Parametrized test function for different libraries, data types, and sizes
@pytest.mark.parametrize(
    "func,dtype,size",
    [
        (gemm_rustynum, "float32", 500),
        (gemm_rustynum, "float64", 500),
        (gemm_numpy, "float32", 500),
        (gemm_numpy, "float64", 500),
        (gemm_rustynum, "float32", 2000),
        (gemm_rustynum, "float64", 2000),
        (gemm_numpy, "float32", 2000),
        (gemm_numpy, "float64", 2000),
    ],
)
def test_gemm(benchmark, func, dtype, size):
    a, b = setup_vector(dtype, size)
    group_name = f"{dtype}"

    benchmark.group = group_name
    benchmark.extra_info["dtype"] = dtype
    benchmark.extra_info["size"] = size
    benchmark(func, a, b, size, dtype)

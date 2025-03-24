import numpy as np
import pytest

import rustynum as rnp


# Helper function to generate random vectors
def setup_vector(dtype, size=1000):
    a = np.random.rand(size * size).astype(dtype)
    b = np.random.rand(size).astype(dtype)
    return a.tolist(), b.tolist()


# Function to perform gemv product using rustynum
def gemv_rustynum(a, b, size, dtype):
    a_rnp = rnp.NumArray(a, dtype=dtype).reshape([size, size])
    b_rnp = rnp.NumArray(b, dtype=dtype)
    return a_rnp.dot(b_rnp)


# Function to perform gemv product using numpy
def gemv_numpy(a, b, size, dtype):
    a_np = np.array(a, dtype=dtype).reshape((size, size))
    b_np = np.array(b, dtype=dtype)
    return np.dot(a_np, b_np)


# Parametrized test function for different libraries, data types, and sizes
@pytest.mark.parametrize(
    "func,dtype,size",
    [
        (gemv_rustynum, "float32", 1000),
        (gemv_rustynum, "float64", 1000),
        (gemv_numpy, "float32", 1000),
        (gemv_numpy, "float64", 1000),
        (gemv_rustynum, "float32", 10000),
        (gemv_rustynum, "float64", 10000),
        (gemv_numpy, "float32", 10000),
        (gemv_numpy, "float64", 10000),
    ],
)
def test_gemv(benchmark, func, dtype, size):
    a, b = setup_vector(dtype, size)
    group_name = f"{dtype}"

    benchmark.group = group_name
    benchmark.extra_info["dtype"] = dtype
    benchmark.extra_info["size"] = size
    benchmark(func, a, b, size, dtype)

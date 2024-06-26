import numpy as np
import pytest
import rustynum as rnp


# Helper function to generate random vectors
def setup_vector(dtype, size=1000):
    a = np.random.rand(size).astype(dtype)
    b = np.random.rand(size).astype(dtype)
    return a.tolist(), b.tolist()


# Function to perform dot product using rustynum
def dot_rustynum(a, b, dtype):
    a_rnp = rnp.NumArray(a, dtype=dtype)
    b_rnp = rnp.NumArray(b, dtype=dtype)
    return a_rnp.dot(b_rnp)


# Function to perform dot product using numpy
def dot_numpy(a, b, dtype):
    a_np = np.array(a, dtype=dtype)
    b_np = np.array(b, dtype=dtype)
    return np.dot(a_np, b_np)


# Parametrized test function for different libraries, data types, and sizes
@pytest.mark.parametrize(
    "func,dtype,size",
    [
        (dot_rustynum, "float32", 1000),
        (dot_rustynum, "float64", 1000),
        (dot_numpy, "float32", 1000),
        (dot_numpy, "float64", 1000),
        (dot_rustynum, "float32", 10000),
        (dot_rustynum, "float64", 10000),
        (dot_numpy, "float32", 10000),
        (dot_numpy, "float64", 10000),
    ],
)
def test_dot_product(benchmark, func, dtype, size):
    a, b = setup_vector(dtype, size)
    group_name = f"{dtype}"

    benchmark.group = group_name
    benchmark.extra_info["dtype"] = dtype
    benchmark.extra_info["size"] = size
    benchmark(func, a, b, dtype)

import numpy as np
import pytest

import rustynum as rnp


# Helper function to generate random vectors
def setup_vector(dtype, size=1000):
    a = np.random.rand(size).astype(dtype)
    b = np.random.rand(size).astype(dtype)
    return a.tolist(), b.tolist()


# Function to perform addition using rustynum
def add_rustynum(a, b, dtype):
    a_rnp = rnp.NumArray(a, dtype=dtype)
    b_rnp = rnp.NumArray(b, dtype=dtype)
    return a_rnp + b_rnp


# Function to perform addition using numpy
def add_numpy(a, b, dtype):
    a_np = np.array(a, dtype=dtype)
    b_np = np.array(b, dtype=dtype)
    return a_np + b_np


# Parametrized test function for different libraries, data types, and sizes
@pytest.mark.parametrize(
    "func,dtype,size",
    [
        (add_rustynum, "float32", 1000),
        (add_rustynum, "float64", 1000),
        (add_numpy, "float32", 1000),
        (add_numpy, "float64", 1000),
        (add_rustynum, "float32", 10000),
        (add_rustynum, "float64", 10000),
        (add_numpy, "float32", 10000),
        (add_numpy, "float64", 10000),
    ],
)
def test_addition(benchmark, func, dtype, size):
    a, b = setup_vector(dtype, size)
    group_name = f"{dtype}"

    benchmark.group = group_name
    benchmark.extra_info["dtype"] = dtype
    benchmark.extra_info["size"] = size
    benchmark(func, a, b, dtype)

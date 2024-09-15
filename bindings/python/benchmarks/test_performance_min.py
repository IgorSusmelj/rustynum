import numpy as np
import pytest
import rustynum as rnp


# Helper function to generate random vectors
def setup_vector(dtype, size=1000):
    a = np.random.rand(size).astype(dtype)
    return a.tolist()


# Function to perform min using rustynum
def min_rustynum(a, dtype):
    a_rnp = rnp.NumArray(a, dtype=dtype)
    return a_rnp.min()


# Function to perform min using numpy
def min_numpy(a, dtype):
    a_np = np.array(a, dtype=dtype)
    return np.min(a_np)


# Parametrized test function for different libraries, data types, and sizes
@pytest.mark.parametrize(
    "func,dtype,size",
    [
        (min_rustynum, "float32", 1000),
        (min_rustynum, "float64", 1000),
        (min_numpy, "float32", 1000),
        (min_numpy, "float64", 1000),
        (min_rustynum, "float32", 10000),
        (min_rustynum, "float64", 10000),
        (min_numpy, "float32", 10000),
        (min_numpy, "float64", 10000),
    ],
)
def test_min(benchmark, func, dtype, size):
    a = setup_vector(dtype, size)
    group_name = f"{dtype}"

    benchmark.group = group_name
    benchmark.extra_info["dtype"] = dtype
    benchmark.extra_info["size"] = size
    benchmark(func, a, dtype)

import numpy as np
import pytest
import rustynum as rnp


# Helper function to generate random vectors
def setup_vector(dtype, size=1000):
    a = np.random.rand(size).astype(dtype)
    return a.tolist()


# Function to perform median using rustynum
def median_rustynum(a, dtype):
    a_rnp = rnp.NumArray(a, dtype=dtype)
    return a_rnp.median()


# Function to perform median using numpy
def median_numpy(a, dtype):
    a_np = np.array(a, dtype=dtype)
    return np.median(a_np)


# Parametrized test function for different libraries, data types, and sizes
@pytest.mark.parametrize(
    "func,dtype,size",
    [
        (median_rustynum, "float32", 1000),
        (median_rustynum, "float64", 1000),
        (median_numpy, "float32", 1000),
        (median_numpy, "float64", 1000),
        (median_rustynum, "float32", 10000),
        (median_rustynum, "float64", 10000),
        (median_numpy, "float32", 10000),
        (median_numpy, "float64", 10000),
    ],
)
def test_median(benchmark, func, dtype, size):
    a = setup_vector(dtype, size)
    group_name = f"{dtype}"

    benchmark.group = group_name
    benchmark.extra_info["dtype"] = dtype
    benchmark.extra_info["size"] = size
    benchmark(func, a, dtype)

![RustyNum Banner](docs/assets/rustynum-banner.png?raw=true "RustyNum")

[![PyPI python](https://img.shields.io/pypi/pyversions/rustynum)](https://pypi.org/project/rustynum)
![PyPI Version](https://badge.fury.io/py/rustynum.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Test Python Bindings](https://github.com/IgorSusmelj/rustynum/actions/workflows/test_python_bindings.yml/badge.svg)
[![Documentation](https://img.shields.io/badge/docs-rustynum.com-blue)](https://rustynum.com)

---

⚠️ **Disclaimer:** _RustyNum is currently a work in progress and is not recommended for production use. Features may be unstable and subject to change._

# RustyNum

RustyNum is a NumPy compatible array library for Python that uses Rust SIMD to accelerate common operations.

RustyNum is a high-performance numerical computation library written in Rust, created to demonstrate the potential of Rust's SIMD (Single Instruction, Multiple Data) capabilities using the nightly `portable_simd` feature, and serving as a fast alternative to Numpy.

## Key Features

- **High Performance:** Utilizes Rust's `portable_simd` for accelerated numerical operations across various hardware platforms, achieving up to 2.86x faster computations for certain operations compared to Numpy.
- **Python Bindings:** Seamless integration with Python, providing a familiar Numpy-like interface.
- **Lightweight:** Minimal dependencies (no external crates are used), ensuring a small footprint and easy deployment. RustyNum Python wheels are only 300kBytes (50x smaller than Numpy wheels).

## Installation

Supported Python versions: `3.8`, `3.9`, `3.10`, `3.11`, `3.12`, `3.13`

Supported operating systems: `Windows x86`, `Linux x86`, `MacOS x86 & ARM`

For comprehensive documentation, tutorials, and API reference, visit [rustynum.com](https://rustynum.com).

### For Python

You can install RustyNum directly from PyPI:

```bash
pip install rustynum
```

> If that does not work for you please create an issue with the operating system and Python version you're using!

## Quick Start Guide (Python)

If you're familiar with Numpy, you'll quickly get used to RustyNum!

```Python
import numpy as np
import rustynum as rnp

# Using Numpy
a = np.array([1.0, 2.0, 3.0, 4.0], dtype="float32")
a = a + 2
print(a.mean())  # 4.5

# Using RustyNum
b = rnp.NumArray([1.0, 2.0, 3.0, 4.0], dtype="float32")
b = b + 2
print(b.mean().item())  # 4.5
```

### Advanced Usage

You can perform advanced operations such as matrix-vector and matrix-matrix multiplications:

```Python
# Matrix-vector dot product using Numpy
import numpy as np
import rustynum as rnp

a = np.random.rand(4 * 4).astype(np.float32)
b = np.random.rand(4).astype(np.float32)
result_numpy = np.dot(a.reshape((4, 4)), b)

# Matrix-vector dot product using RustyNum
a_rnp = rnp.NumArray(a.tolist())
b_rnp = rnp.NumArray(b.tolist())
result_rust = a_rnp.reshape([4, 4]).dot(b_rnp).tolist()

print(result_numpy)  # Example Output: [0.8383043, 1.678406, 1.4153088, 0.7959367]
print(result_rust)   # Example Output: [0.8383043, 1.678406, 1.4153088, 0.7959367]
```

## Features

RustyNum offers a variety of numerical operations and data types, with more features planned for the future.

### Supported Data Types

- float64
- float32
- uint8 (experimental)
- int32 (Planned)
- int64 (Planned)

### Supported Operations

| Operation        | NumPy Equivalent                | RustyNum Equivalent              |
| ---------------- | ------------------------------- | -------------------------------- |
| Zeros Array      | `np.zeros((2, 3))`              | `rnp.zeros((2, 3))`              |
| Ones Array       | `np.ones((2, 3))`               | `rnp.ones((2, 3))`               |
| Arange           | `np.arange(start, stop, step)`  | `rnp.arange(start, stop, step)`  |
| Linspace         | `np.linspace(start, stop, num)` | `rnp.linspace(start, stop, num)` |
| Mean             | `np.mean(a)`                    | `rnp.mean(a)`                    |
| Median           | `np.median(a)`                  | `rnp.median(a)`                  |
| Min              | `np.min(a)`                     | `rnp.min(a)`                     |
| Max              | `np.max(a)`                     | `rnp.max(a)`                     |
| Exp              | `np.exp(a)`                     | `rnp.exp(a)`                     |
| Log              | `np.log(a)`                     | `rnp.log(a)`                     |
| Sigmoid          | `1 / (1 + np.exp(-a))`          | `rnp.sigmoid(a)`                 |
| Dot Product      | `np.dot(a, b)`                  | `rnp.dot(a, b)`                  |
| Reshape          | `a.reshape((2, 3))`             | `a.reshape([2, 3])`              |
| Concatenate      | `np.concatenate([a,b], axis=0)` | `rnp.concatenate([a,b], axis=0)` |
| Element-wise Add | `a + b`                         | `a + b`                          |
| Element-wise Sub | `a - b`                         | `a - b`                          |
| Element-wise Mul | `a * b`                         | `a * b`                          |
| Element-wise Div | `a / b`                         | `a / b`                          |
| Fancy indexing   | `np.ones((2,3))[0, :]`          | `rnp.ones((2,3))[0, :]`          |
| Fancy flipping   | `np.array([1,2,3])[::-1]`       | `rnp.array([1,2,3])[::-1]`       |

### NumArray Class

Initialization

```Python
from rustynum import NumArray

# From a list
a = NumArray([1.0, 2.0, 3.0], dtype="float32")

# From another NumArray
b = NumArray(a)

# From nested lists (2D array)
c = NumArray([[1.0, 2.0], [3.0, 4.0]], dtype="float64")
```

Methods

`reshape(shape: List[int]) -> NumArray`

Reshapes the array to the specified shape.

```Python
reshaped = a.reshape([3, 1])
```

`matmul(other: NumArray) -> NumArray`

Performs matrix multiplication with another NumArray.

```Python
result = a.matmul(b)
# or
result = a @ b
```

`dot(other: NumArray) -> NumArray`

Computes the dot product with another NumArray.

```Python
dot_product = a.dot(b)
```

`mean(axis: Union[None, int, Sequence[int]] = None) -> Union[NumArray, float]`

Computes the mean along specified axis.

```Python
average = a.mean()
average_axis0 = a.mean(axis=0)
```

`median(axis: Union[None, int, Sequence[int]] = None) -> Union[NumArray, float]`

Computes the median along specified axis.

```Python
median = a.median()
median_axis0 = a.median(axis=0)
```

`min(axis: Union[None, int, Sequence[int]] = None) -> Union[NumArray, float]`

Returns the minimum value in the array.

```Python
minimum = a.min()
```

`max(axis: Union[None, int, Sequence[int]] = None) -> Union[NumArray, float]`

Returns the maximum value in the array.

```Python
maximum = a.max()
```

`tolist() -> Union[List[float], List[List[float]]]`

Converts the NumArray to a Python list.

```Python
list_representation = a.tolist()
```

### Multi-Dimensional Arrays

- Matrix-vector dot product
- Matrix-matrix dot product

## Roadmap

Planned Features:

- N-dimensional arrays
  - Useful for filters, image processing, and machine learning
- Additional operations: argmin, argmax, sort, std, var, zeros, cumsum, interp
- Integer support
- Extended shaping and reshaping capabilities
- C++ and WASM bindings

Not Planned:

- Random number generation (use the rand crate)

## Design Principles

RustyNum is built on four core principles:

1. **No 3rd Party Dependencies:** Ensuring transparency and control over the codebase.
2. **Leverage Portable SIMD:** Utilizing Rust's nightly SIMD feature for high-performance operations across platforms.
3. **First-Class Language Bindings:** Providing robust support for Python, with plans for WebAssembly and C++.
4. **Numpy-like Interface:** Offering a familiar and intuitive user experience for those coming from Python.

## Performance

### Python

RustyNum leverages Rust's `portable_simd` feature to achieve significant performance improvements in numerical computations. On a MacBook Pro M1 Pro, RustyNum outperforms Numpy in several key operations. Below are benchmark results comparing `RustyNum 0.1.4` with `Numpy 1.24.4`:

#### Benchmark Results (float32)

| Operation                   | RustyNum (us)  | Numpy (us)     | Speedup Factor |
| --------------------------- | -------------- | -------------- | -------------- |
| Mean (1000 elements)        | 8.8993         | 22.6300        | 2.54x          |
| Median (1000 elements)      | 23.6040        | 39.8451        | 1.68x          |
| Min (1000 elements)         | 10.1423        | 28.9693        | 2.86x          |
| Sigmoid (1000 elems)        | 10.6899        | 23.2486        | 2.17x          |
| Dot Product (1000 elems)    | 17.0640        | 38.2958        | 2.24x          |
| Matrix-Vector (1000x1000)   | 10,041.6093    | 24,990.2646    | 2.49x          |
| Matrix-Vector (10000x10000) | 2,731,092.0332 | 2,103,920.4830 | 0.77x          |
| Matrix-Matrix (500x500)     | 7,010.6638     | 14,878.9556    | 2.12x          |
| Matrix-Matrix (2000x2000)   | 225,595.8832   | 257,832.6334   | 1.14x          |

#### Benchmark Results (float64)

| Operation                   | RustyNum (us)  | Numpy (us)     | Speedup Factor |
| --------------------------- | -------------- | -------------- | -------------- |
| Mean (1000 elements)        | 9.1026         | 24.0636        | 2.64x          |
| Median (1000 elements)      | 24.9010        | 38.4760        | 1.54x          |
| Min (1000 elements)         | 18.2651        | 24.8170        | 1.36x          |
| Dot Product (1000 elems)    | 16.6583        | 38.8000        | 2.33x          |
| Matrix-Vector (1000x1000)   | 9,941.3305     | 23,788.9570    | 2.39x          |
| Matrix-Vector (10000x10000) | 3,635,297.4664 | 4,962,900.9084 | 1.37x          |
| Matrix-Matrix (500x500)     | 9,683.3815     | 15,866.6376    | 1.64x          |
| Matrix-Matrix (2000x2000)   | 412,333.8586   | 365,047.5000   | 0.89x          |

#### Observations

- RustyNum significantly outperforms Numpy in basic operations such as mean, median, min, and dot product, with speedup factors up to and over 2x.
- For larger operations, especially matrix-vector and matrix-matrix multiplications, Numpy currently performs better, which highlights areas for potential optimization in RustyNum.

These results demonstrate RustyNum's potential for high-performance numerical computations, particularly in operations where SIMD instructions can be fully leveraged.

### Rust

In addition to the Python bindings, RustyNum’s core library is implemented in Rust. Below is a comparison of RustyNum (rustynum_rs) with two popular Rust numerical libraries: `nalgebra 0.33.0` and `ndarray 0.16.1`. The benchmarks were conducted using the Criterion crate to measure performance across various basic operations.

#### Benchmark Results (float32)

| Input Size                                  | RustyNum  | nalgebra  | ndarray   |
| ------------------------------------------- | --------- | --------- | --------- |
| Addition (10k elements)                     | 760.53 ns | 695.73 ns | 664.29 ns |
| Vector mean (10k elements)                  | 683.83 ns | 14.602 µs | 1.2370 µs |
| Vector median (10k elements)                | 7.4175 µs | 6.8863 µs | 6.9970 µs |
| Vector Dot Product (10k elements)           | 758.65 ns | 1.1843 µs | 1.1942 µs |
| Matrix-Vector Multiplication (1k elements)  | 77.851 us | 403.39 µs | 115.75 µs |
| Matrix-Matrix Multiplication (500 elements) | 2.5526 ms | 2.9038 ms | 2.7847 ms |
| Matrix-Matrix Multiplication (1k elements)  | 17.836 ms | 21.895 ms | 22.423 ms |

#### Observations

- RustyNum is able to perform on par with nalgebra and ndarray in most operations and sometimes even outperforms them.
- There seems a significant overhead in the Python bindings. We're getting 10ms in Python in RustyNum for the matrix-vector multiplication with 1k elements vs 77us in Rust.

# Build

## Rust Crate

### Run tests

Run using

```

cargo test

```

### Create Docs

```

cargo doc --open

```

### Run Benchmarks

Run using

```

cargo bench -- <benchmark_name>

```

## Python bindings

Don't use maturin. But only setup.py

```

cd bindings/python/ && python setup.py install

```

or

```

cd bindings/python/ && python setup.py bdist_wheel

```

Then run tests using

```

pytest tests

```

or benchmarks using

```

pytest benchmarks

```

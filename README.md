# RustyNum

RustyNum is a high-performance numerical computation library written in Rust, created to demonstrate the potential of Rust's SIMD (Single Instruction, Multiple Data) capabilities using the nightly `portable_simd` feature, and serving as a fast alternative to Numpy.

## Key Features

- **High Performance:** Utilizes Rust's `portable_simd` for accelerated numerical operations across various hardware platforms, achieving up to 2.86x faster computations for certain operations compared to Numpy.
- **Python Bindings:** Seamless integration with Python, providing a familiar Numpy-like interface.
- **Lightweight:** Minimal dependencies (no external crates are used), ensuring a small footprint and easy deployment. Rustynum Python wheels are only 300kBytes (50x smaller than Numpy wheels).

## Installation

### For Python

You can install RustyNum directly from PyPI:

```bash
pip install rustynum
```

## Quick Start Guide (Python)

If you're familiar with Numpy, you'll quickly get used to RustyNum!

```Python
import numpy as np
a = np.array([1.0, 2.0, 3.0, 4.0], dtype="float32")
a = a + 2

import rustynum as rnp
b = rnp.NumArray([1.0, 2.0, 3.0, 4.0], dtype="float32")
b = b + 2

print(a.mean()) # 4.5
print(b.mean().item()) # 4.5
```

### Advanced Usage

You can perform advanced operations such as matrix-vector and matrix-matrix multiplications:

```Python

# matrix-vector dot product
import numpy as np

a = np.random.rand(4 * 4).astype(np.float32)
b = np.random.rand(4).astype(np.float32)

result_numpy = np.dot(a.reshape((4, 4)), b)

import rustynum as rnp

a_rnp = rnp.NumArray(a.tolist())
b_rnp = rnp.NumArray(b.tolist())

result_rust = a_rnp.reshape([4, 4]).dot(b_rnp).tolist()

print(result_numpy) # [0.8383043 1.678406  1.4153088 0.7959367]
print(result_rust) # [0.8383043 1.678406  1.4153088 0.7959367]
```

## Features

RustyNum offers a variety of numerical operations and data types, with more features planned for the future.

### Supported Data Types

- float64
- float32

### Supported Operations

| Operation        | NumPy Equivalent                | Rustynum Equivalent              |
| ---------------- | ------------------------------- | -------------------------------- |
| Zeros Array      | `np.zeros((2, 3))`              | `rnp.zeros((2, 3))`              |
| Ones Array       | `np.ones((2, 3))`               | `rnp.ones((2, 3))`               |
| Arange           | `np.arange(start, stop, step)`  | `rnp.arange(start, stop, step)`  |
| Linspace         | `np.linspace(start, stop, num)` | `rnp.linspace(start, stop, num)` |
| Mean             | `np.mean(a)`                    | `rnp.mean(a)`                    |
| Min              | `np.min(a)`                     | `rnp.min(a)`                     |
| Max              | `np.max(a)`                     | `rnp.max(a)`                     |
| Dot Product      | `np.dot(a, b)`                  | `rnp.dot(a, b)`                  |
| Reshape          | `a.reshape((2, 3))`             | `a.reshape([2, 3])`              |
| Element-wise Add | `a + b`                         | `a + b`                          |
| Element-wise Sub | `a - b`                         | `a - b`                          |
| Element-wise Mul | `a * b`                         | `a * b`                          |
| Element-wise Div | `a / b`                         | `a / b`                          |

### 1-Dimensional Arrays

- Dot product
- Mean
- Min/Max
- Addition (+), Subtraction (-), Multiplication (\*), Division (/)
- Reshape

### Multi-Dimensional Arrays

- Matrix-vector dot product
- Matrix-matrix dot product

## Roadmap

Planned Features:

- N-dimensional arrays
  - Useful for filters, image processing, and machine learning
- Additional operations: arange, linspace, median, argmin, argmax, sort, std, var, zeros, cumsum, interp
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

RustyNum leverages Rust's `portable_simd` feature to achieve significant performance improvements in numerical computations. On a MacBook Pro M1 Pro, RustyNum outperforms Numpy in several key operations. Below are benchmark results comparing `RustyNum 0.1.2` with `Numpy 1.24.4`:

### Benchmark Results (float32)

| Operation                   | RustyNum (us)  | Numpy (us)     | Speedup Factor |
| --------------------------- | -------------- | -------------- | -------------- |
| Mean (1000 elements)        | 8.8993         | 22.6300        | 2.54x          |
| Min (1000 elements)         | 10.1423        | 28.9693        | 2.86x          |
| Dot Product (1000 elems)    | 17.0640        | 38.2958        | 2.24x          |
| Matrix-Vector (1000x1000)   | 10,041.6093    | 24,990.2646    | 2.49x          |
| Matrix-Vector (10000x10000) | 2,731,092.0332 | 2,103,920.4830 | 0.77x          |
| Matrix-Matrix (2000x2000)   | 714,635.3832   | 264,115.1336   | 0.37x          |

### Benchmark Results (float64)

| Operation                   | RustyNum (us)  | Numpy (us)     | Speedup Factor |
| --------------------------- | -------------- | -------------- | -------------- |
| Mean (1000 elements)        | 9.1026         | 24.0636        | 2.64x          |
| Min (1000 elements)         | 18.2651        | 24.8170        | 1.36x          |
| Dot Product (1000 elems)    | 16.6583        | 38.8000        | 2.33x          |
| Matrix-Vector (1000x1000)   | 9,941.3305     | 23,788.9570    | 2.39x          |
| Matrix-Vector (10000x10000) | 3,635,297.4664 | 4,962,900.9084 | 1.37x          |
| Matrix-Matrix (2000x2000)   | 1,362,796.0666 | 331,243.1500   | 0.24x          |

### Observations

- RustyNum significantly outperforms Numpy in basic operations such as mean, min, and dot product, with speedup factors over 2x.
- For larger operations, especially matrix-vector and matrix-matrix multiplications, Numpy currently performs better, which highlights areas for potential optimization in RustyNum.

These results demonstrate RustyNum's potential for high-performance numerical computations, particularly in operations where SIMD instructions can be fully leveraged.

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

cargo criterion

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

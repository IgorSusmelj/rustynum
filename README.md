# RustyNum

A simple library for numerical computation written in Rust.

RustyNum uses portable SIMD from Rust Nightly which brings SIMD instructions across
different hardware and targets such as WASM.

We provide Python bindings that work very similar to Numpy.

## Installation

You can install rustynum directly from pypi for Python.

```bash
pip install rustynum
```

### Python Usage

If you're familiar with Numpy you will get used to RustNum quickly!

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

You can use RustyNum to compute dot products, matrix-vector or even matrix-matrix multiplications!

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

RustNum is very lightweight as it does not use any dependency except the standard lib and portable SIMD. The whole Python wheel is smaller than 300 kBytes!

### Datatypes

RustyNum supports `float64` and `float32` precision.

### Supported Operators

- 1-dim arrays

  - dot product
  - mean
  - min
  - max
  - addition (+)
  - subration (-)
  - multiply (\*)
  - division (/)
  - reshape

- multi-dim

  - matrix-vector dot product
  - matrix-matrix dot product

  **Todo:**

- N-dim arrays
  - 1-dim arrays
  - 2-dim arrays (useful for filters)
  - 3-dim arrays (useful for image processing)
  - 4-dim arrays (useful for ML)
- arange
- linspace
- median
- argmin
- argmax
- sort
- std
- var
- zeros
- cumsum
- interp
- support integers!?
- shaping and reshaping!?
- C++ bindings
- WASM bindings

**Not planned:**

- random number generation (use the rand crate and DIY)

## Design Principles

RustyNum has been designed with four principles in mind

- No 3rd party dependencies: Making it very transparent what you get
- Leverage portable SIMD: A Rust nightly feature that provides an easy interface
  to write SIMD code for fast opterations across platforms.
- First level support for bindings for languages such as Python (WebAssembly and C++ are planned)
- Numpy-like interface: Coming from Python having a Numpy like user interface

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

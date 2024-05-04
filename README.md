## Usage

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

### Design Principles

RustyNum has been designed with four principles in mind

- No 3rd party dependencies: Making it very transparent what you get
- Leverage portable SIMD: A Rust nightly feature that provides an easy interface
  to write SIMD code for fast opterations across platforms.
- First level support for bindings for languages such as Python (WebAssembly and C++ are planned)
- Numpy-like interface: Coming from Python having a Numpy like user interface

## Rust Crate

### Run tests

Run using

```
cargo test
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

```

```

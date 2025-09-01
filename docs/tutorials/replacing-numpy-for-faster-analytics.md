---
title: Replace selected NumPy calls to speed up analytics
description: Speed up mean min and dot product in Python by swapping NumPy with RustyNum. Includes timing code and results.
---
# Replacing Core NumPy Calls for Faster Analytics

Many developers rely on NumPy for array operations, statistical calculations, and linear algebra. RustyNum offers an alternative for several common NumPy routines, potentially speeding up your Python analytics. In this tutorial, you’ll see how to replace selected NumPy calls with RustyNum equivalents, measure performance differences, and integrate RustyNum into existing data workflows.

---

## Introduction

NumPy is the go-to library for Python numerical tasks, but sometimes you need extra speed. RustyNum taps into Rust’s SIMD capabilities for faster computations in certain scenarios. By strategically swapping out a few operations, you might see noticeable performance gains in your Python scripts.

---

## Why Replace NumPy Calls?

1. **Performance Gains**: RustyNum’s internal operations are optimized using low-level instructions, which can result in faster execution on supported hardware.
2. **Seamless Integration**: The interface is similar to NumPy, so transitions often involve minimal code changes.
3. **Lightweight**: RustyNum wheels are much smaller (300kB vs 15MB for NumPy), making installation and distribution simpler.

---

## Benchmark Setup

Before diving in, ensure you have both NumPy and RustyNum installed:

```bash
pip install numpy rustynum
```

To measure execution times, you can use Python’s built-in `time` module or other tools like `timeit` or IPython’s `%timeit` magic command.

---

## Key Operations

Below are three core operations you can replace with RustyNum. Let’s see how each one works with side-by-side comparisons.

### Mean

```python
import numpy as np
import rustynum as rnp
import time

# Create test data
data_np = np.random.rand(1_000_000).astype(np.float32)
data_rn = rnp.NumArray(data_np.tolist(), dtype="float32")

# NumPy timing
start_np = time.time()
mean_np = np.mean(data_np)
end_np = time.time()
numpy_duration = end_np - start_np

# RustyNum timing
start_rn = time.time()
mean_rn = data_rn.mean().item()
end_rn = time.time()
rustynum_duration = end_rn - start_rn

# Print results
print("Results Comparison:")
print("-" * 40)
print(f"NumPy mean:    {mean_np:.8f}")
print(f"RustyNum mean: {mean_rn:.8f}")
print("\nPerformance:")
print("-" * 40)
print(f"NumPy time:    {numpy_duration:.6f} seconds")
print(f"RustyNum time: {rustynum_duration:.6f} seconds")
print(f"Speedup:       {numpy_duration/rustynum_duration:.2f}x")
```

On my MacBook Pro M1, this is the output:

```
Results Comparison:
----------------------------------------
NumPy mean:    0.50181508
RustyNum mean: 0.50181526

Performance:
----------------------------------------
NumPy time:    0.000048 seconds
RustyNum time: 0.000017 seconds
Speedup:       2.83x
```

!!! note
    The computed mean is slightly different between NumPy and RustyNum. This is due to the different underlying algorithms used by the two libraries.

### Minimum

```python
import numpy as np
import rustynum as rnp
import time

# Create test data
data_np = np.random.rand(100_000).astype(np.float32)
data_rn = rnp.NumArray(data_np.tolist(), dtype="float32")

# NumPy timing
start_np = time.time()
min_np = np.min(data_np)
end_np = time.time()
numpy_duration = end_np - start_np

# RustyNum timing
start_rn = time.time()
min_rn = data_rn.min()
end_rn = time.time()
rustynum_duration = end_rn - start_rn

# Print results
print("Results Comparison:")
print("-" * 40)
print(f"NumPy min:     {min_np:.8f}")
print(f"RustyNum min:  {min_rn:.8f}")
print("\nPerformance:")
print("-" * 40)
print(f"NumPy time:    {numpy_duration:.6f} seconds")
print(f"RustyNum time: {rustynum_duration:.6f} seconds")
print(f"Speedup:       {numpy_duration/rustynum_duration:.2f}x")
```
On my MacBook Pro M1, this is the output:
```
Results Comparison:
----------------------------------------
NumPy min:     0.00001837
RustyNum min:  0.00001837

Performance:
----------------------------------------
NumPy time:    0.000033 seconds
RustyNum time: 0.000010 seconds
Speedup:       3.29x
```

### Dot Product

```python
import numpy as np
import rustynum as rnp
import time

# Create test data
matrix_np = np.random.rand(1000, 1000).astype(np.float32)
vector_np = np.random.rand(1000).astype(np.float32)
matrix_rn = rnp.NumArray(matrix_np.flatten().tolist(), dtype="float32").reshape([1000, 1000])
vector_rn = rnp.NumArray(vector_np.tolist(), dtype="float32")

# NumPy timing
start_np = time.time()
dot_np = np.dot(matrix_np, vector_np)
end_np = time.time()
numpy_duration = end_np - start_np

# RustyNum timing
start_rn = time.time()
dot_rn = matrix_rn.dot(vector_rn)
end_rn = time.time()
rustynum_duration = end_rn - start_rn

# Print results
print("Results Comparison:")
print("-" * 40)
print(f"NumPy dot[0]:    {dot_np[0]:.8f}")
print(f"RustyNum dot[0]: {dot_rn[0]:.8f}")
print("\nPerformance:")
print("-" * 40)
print(f"NumPy time:    {numpy_duration:.6f} seconds")
print(f"RustyNum time: {rustynum_duration:.6f} seconds")
print(f"Speedup:       {numpy_duration/rustynum_duration:.2f}x")
```

On my MacBook Pro M1, this is the output:

```
Results Comparison:
----------------------------------------
NumPy dot[0]:    251.67926025
RustyNum dot[0]: 251.67927551

Performance:
----------------------------------------
NumPy time:    0.000353 seconds
RustyNum time: 0.000093 seconds
Speedup:       3.79x
```

!!! note
    The computed dot product is slightly different between NumPy and RustyNum. This is due to the different underlying algorithms used by the two libraries.

---

## Next Steps

- Check out the [API Reference](../api/index.md) for a complete list of functions and classes.
- Explore more advanced examples in the upcoming [Tutorials](../index.md).
- Contribute your own ideas or ask questions on [GitHub](https://github.com/IgorSusmelj/rustynum).

---

**Further reading**: [Installation](../installation.md), [Quick Start](../quick-start.md), [API Reference](../api/index.md).
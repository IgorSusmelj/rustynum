---
title: Matrix multiplication in Python faster with RustyNum vs NumPy
description: Learn matrix vector and matrix matrix multiplication in Python using RustyNum, compare with NumPy, and measure runtime.
---
# Getting Better Matrix Operations with RustyNum

Matrix operations are at the core of many data science and engineering workflows. When performance matters, switching from traditional Python solutions to RustyNum can be a great move. In this tutorial, you’ll learn how to perform matrix-vector and matrix-matrix operations with RustyNum, compare them to NumPy, and see how SIMD acceleration can improve efficiency.

---

## Why Matrix Operations Matter

Matrix operations are fundamental to:
- Machine Learning and Deep Neural Networks
- Scientific Computing and Simulations
- Image Processing and Computer Vision
- Financial Modeling and Statistics

RustyNum leverages Rust's SIMD capabilities to provide significant performance improvements over traditional Python solutions, achieving up to 2.86x speedup for certain operations compared to NumPy.

---

## Setting Up Your Environment

If you haven’t already installed RustyNum, visit the [Installation Guide](../installation.md). Once you have RustyNum set up, verify it works by creating a small NumArray and printing its contents:

```python
import rustynum as rnp

sample = rnp.NumArray([1.0, 2.0, 3.0], dtype="float32")
print("Sample array:", sample)
```

You should see an output reflecting your new array.  

```
Sample array: [1. 2. 3.]
```

---

## Matrix-Vector Multiplication

Matrix-vector multiplication is one of the most common tasks in numerical computing. It’s often used in transformations, linear regression, and more.

### Creating a Matrix and Vector

```python
import rustynum as rnp

# Create a 4x4 matrix
matrix_data = [i for i in range(16)]  # 0 to 15
matrix = rnp.NumArray(matrix_data, dtype="float32").reshape([4, 4])

# Create a 4-element vector
vector_data = [1, 2, 3, 4]
vector = rnp.NumArray(vector_data, dtype="float32")

print("Matrix:\n", matrix)
print("Vector:\n", vector)
```

In this example, we reshape a 1D list into a 4×4 matrix. RustyNum’s `reshape` method is similar to NumPy’s.

### Performing the Multiplication

```python
result_vec = matrix.dot(vector)
print("Matrix-Vector Multiplication Result:\n", result_vec)
```

RustyNum’s `.dot()` function handles both matrix-vector and matrix-matrix products based on the shapes of the inputs.

Alternatively, you can use the `@` operator for matrix multiplication:

```python
result_vec = matrix @ vector
print("Matrix-Vector Multiplication Result:\n", result_vec)
```

!!! info "Matrix Multiplication Operator"
    RustyNum supports Python's `@` operator for matrix multiplication, which is the recommended way to perform matrix-vector and matrix-matrix operations. It follows the same rules as NumPy: the inner dimensions must match, or you'll get an error.

---

## Matrix-Matrix Multiplication

Matrix-matrix multiplication is more computationally intense than matrix-vector multiplication, and it appears in neural networks, image transformations, and various numerical algorithms.

### Creating Two Matrices

```python
import rustynum as rnp

# Create two 2D NumArrays
dataA = [1.0, 2.0, 3.0, 4.0]
dataB = [5.0, 6.0, 7.0, 8.0]

A = rnp.NumArray(dataA, dtype="float32").reshape([2, 2])
B = rnp.NumArray(dataB, dtype="float32").reshape([2, 2])

print("Matrix A:\n", A)
print("Matrix B:\n", B)
```

### Multiplying the Matrices

```python
result_matrix = A.dot(B)
print("Matrix-Matrix Multiplication Result:\n", result_matrix)
```

Similar to matrix-vector multiplication, you can use the `@` operator for matrix multiplication:

```python
result_matrix = A @ B
print("Matrix-Matrix Multiplication Result:\n", result_matrix)
```

!!! info "Matrix Multiplication Operator"
    RustyNum supports Python's `@` operator for matrix multiplication, which is the recommended way to perform matrix-vector and matrix-matrix operations. It follows the same rules as NumPy: the inner dimensions must match, or you'll get an error.


---

## Side-by-Side Comparison with NumPy

If you want to compare code or performance, you can do so easily:

```python
import numpy as np
import rustynum as rnp
import time

# NumPy multiplication
matrix_np = np.arange(16, dtype=np.float32).reshape((4, 4))
vector_np = np.array([1, 2, 3, 4], dtype=np.float32)

start_np = time.time()
result_np = np.dot(matrix_np, vector_np)
end_np = time.time()
numpy_time = end_np - start_np

# RustyNum multiplication
matrix_rn = rnp.NumArray(matrix_np.flatten().tolist(), dtype="float32").reshape([4, 4])
vector_rn = rnp.NumArray(vector_np.tolist(), dtype="float32")

start_rn = time.time()
result_rn = matrix_rn.dot(vector_rn)
end_rn = time.time()
rustynum_time = end_rn - start_rn

print("NumPy result:", result_np)
print("RustyNum result:", result_rn)
print(f"NumPy time: {numpy_time:.6f} seconds")
print(f"RustyNum time: {rustynum_time:.6f} seconds")
```

On my MacBook Pro M1, this is the output:

```
NumPy result: [ 20.  60. 100. 140.]
RustyNum result: [20.0, 60.0, 100.0, 140.0]
NumPy time: 0.000015 seconds
RustyNum time: 0.000002 seconds
```

While timing results can vary based on hardware and environment, RustyNum may offer speed improvements in certain operations.

---

## Performance Considerations

1. **SIMD Utilization**: RustyNum uses Rust’s nightly `portable_simd` feature to speed up calculations. This can lead to better throughput on supported hardware (using AVX2, AVX512 for x86, NEON for ARM etc.).
2. **Matrix Size**: Small matrices may not highlight performance gains because overhead can dominate. Larger matrices typically see more benefit.
3. **Data Types**: RustyNum currently supports float32, float64, and some integer types. Mixed data types may cause errors or reduced performance.

---

## Troubleshooting Tips

- **Shape Mismatch**: RustyNum will raise an error if matrix dimensions are incompatible. Double-check shapes with `.shape`.
- **Unsupported Data Type**: If you try to create a NumArray with an unsupported type, you might see a ValueError. Stick to float32 or float64.
- **Installation Problems**: Confirm you have the right Python version and platform by revisiting the [Installation Guide](../installation.md).

---

## Next Steps

- Check out the [API Reference](../api/index.md) for a complete list of functions and classes.
- Explore more advanced examples in the upcoming [Tutorials](../index.md).
- Contribute your own ideas or ask questions on [GitHub](https://github.com/IgorSusmelj/rustynum).

Matrix operations are a cornerstone of computational tasks, and RustyNum offers a Python-friendly path to faster, more efficient code. By tapping into Rust’s low-level optimizations, you can scale your projects without changing your entire workflow.

---

**Further reading**: [Installation](../installation.md), [Quick Start](../quick-start.md), [API Reference](../api/index.md).
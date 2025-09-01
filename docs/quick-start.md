---
title: Quick start with RustyNum in Python
description: Create arrays, compute means and dot products, and compare RustyNum with NumPy. Copy ready examples for Python.
---
# Getting Started with RustyNum

Welcome to RustyNum! This guide will help you quickly get up and running with RustyNum, from basic operations to a comparison with NumPy. If you're familiar with NumPy, you'll feel right at home.

RustyNum is a NumPy compatible array library for Python that uses Rust SIMD to accelerate common operations.

---

## 🔥 Why Use RustyNum?

RustyNum is a high-performance alternative to NumPy. With Rust’s SIMD optimization, RustyNum can significantly speed up your numerical computations, all while maintaining a familiar interface.

---

## 📘 Basic Usage

Here’s a quick example to demonstrate RustyNum's simplicity and performance:

### Example: Compute the Mean of an Array
```python
import rustynum as rnp

# Create a NumArray
a = rnp.NumArray([1.0, 2.0, 3.0, 4.0], dtype="float32")

# Add a scalar
a = a + 2

# Compute the mean
mean_value = a.mean().item()

print(mean_value)  # Output: 4.5
```

## ⚡ RustyNum vs NumPy

RustyNum is designed to be a faster alternative to NumPy for many operations. Let’s compare the syntax:

### Creating Arrays
#### NumPy
```python
import numpy as np

a = np.array([1.0, 2.0, 3.0], dtype="float32")
```

#### RustyNum
```python
import rustynum as rnp

a = rnp.NumArray([1.0, 2.0, 3.0], dtype="float32")
```

### Adding Scalars
#### NumPy
```python
a = a + 2
```

#### RustyNum
```python
a = a + 2
```

### Computing the Mean
#### NumPy
```python
mean_value = a.mean()
```

#### RustyNum
```python
mean_value = a.mean().item()
```

---

## 🛠️ Basic Operations

RustyNum supports a variety of operations. Here are a few examples to get you started:

### 1. Creating Arrays
```python
import rustynum as rnp

# Create an array of zeros
zeros_array = rnp.zeros([3, 3])

# Create an array with evenly spaced values
arange_array = rnp.arange(0, 10, 2)

# Create an array with evenly spaced values over a specified interval
linspace_array = rnp.linspace(0, 1, 5)

print(zeros_array)
print(arange_array)
print(linspace_array)
```

### 2. Element-Wise Operations
```python
# Perform element-wise addition
result = arange_array + 2

# Perform element-wise multiplication
result = arange_array * 2
```

### 3. Matrix Operations
```python
# Create a 2D NumArray
matrix = rnp.NumArray([[1.0, 2.0], [3.0, 4.0]], dtype="float32")

# Compute the dot product
vector = rnp.NumArray([1.0, 2.0], dtype="float32")
dot_product = matrix.dot(vector)

print(dot_product)
```

---

## 🔗 Learn More

Once you’re comfortable with the basics, dive deeper into RustyNum with these resources:

- **[Tutorials](tutorials/index.md)**: Explore real-world applications of RustyNum.
- **[API Reference](api/index.md)**: Detailed documentation of RustyNum's Python bindings.


For larger arrays and timings, see the comparison in [RustyNum vs NumPy performance](benchmarks/rustynum-vs-numpy.md).

---

## 📩 Need Help?

If you have any questions, check out the [GitHub Discussions](https://github.com/IgorSusmelj/rustynum/discussions) or file an issue on the [GitHub Repository](https://github.com/IgorSusmelj/rustynum/issues).

---

<div style="text-align: center;">
    <a href="tutorials/index.md" class="md-button md-button--primary">Explore Tutorials</a>
    <a href="api/index.md" class="md-button">View API Reference</a>
</div>
---
title: Scale normalize and concatenate features in Python with RustyNum
description: Faster preprocessing for ML in Python. Min max scaling L2 normalization concatenation and scikit learn integration.
---
# Streamlining Machine Learning Preprocessing with RustyNum

Data preprocessing is a key step in machine learning. Whether you're prepping large datasets for neural networks or just cleaning up smaller ones, RustyNum can help speed up vectorized operations and transformations. In this tutorial, we'll explore how to use RustyNum for several preprocessing tasks, then show how to integrate your processed data with popular Python ML libraries.

---

## Introduction

When working with data, you often need to transform and prepare it for machine learning models. This can include scaling, normalizing, feature concatenation, and more. RustyNum offers a Python-friendly interface that can accelerate these tasks by leveraging Rust's SIMD optimizations. Let's look at how it works in practice.

---

## Environment Setup

Before starting, ensure you have RustyNum installed:

```bash
pip install rustynum
```

We'll also use NumPy for comparison and scikit-learn for a quick model integration step (optional):

```bash
pip install numpy scikit-learn
```

---

## Loading and Inspecting Data

Let's start by importing all necessary libraries and creating our sample dataset:

```python
import numpy as np
import rustynum as rnp
import math
from sklearn.linear_model import LogisticRegression

# Generate random data with shape (100, 3)
data_np = np.random.rand(100, 3).astype(np.float32)

# Convert to RustyNum's NumArray
data_rn = rnp.NumArray(data_np.flatten().tolist(), dtype="float32").reshape([100, 3])

print("RustyNum data shape:", data_rn.shape)
# RustyNum data shape: (100, 3)
```

Here, we create random data in NumPy, convert it to RustyNum, and confirm its shape. If you're loading a CSV, just convert that array to RustyNum similarly.

---

## Common Preprocessing Tasks

### Scaling

Scaling adjusts the range of features so they align more closely, which can help certain algorithms converge faster. Let's perform a simple min-max scale manually with RustyNum.

1. Find the min and max for each column.  
2. Subtract the min from each element.  
3. Divide by (max - min).

```python
# Create a reusable scaling function
def min_max_scale(array):
    # Step 1: Compute column-wise min and max
    col_mins = []
    col_maxes = []
    for col_idx in range(array.shape[1]):
        col_data = array[:, col_idx]
        col_mins.append(col_data.min())
        col_maxes.append(col_data.max())
    
    # Step 2 & 3: Scale each column
    scaled_data = []
    for col_idx in range(array.shape[1]):
        col_data = array[:, col_idx]
        numerator = col_data - col_mins[col_idx]
        denominator = col_maxes[col_idx] - col_mins[col_idx] or 1.0
        scaled_col = numerator / denominator
        scaled_data.append(scaled_col.tolist())
    
    # Concatenate scaled columns
    return rnp.concatenate(
        [rnp.NumArray(col, dtype="float32").reshape([array.shape[0], 1]) for col in scaled_data],
        axis=1
    )

# Scale our data
scaled_data_rn = min_max_scale(data_rn)
print("Scaled data shape:", scaled_data_rn.shape)
# Scaled data shape: (100, 3)
print("First row after scaling:", scaled_data_rn[0, :].tolist())
# First row after scaling: [[0.9085683226585388, 0.006626238115131855, 0.5808358788490295]]
```

!!! info "Why Scale Before Normalizing?"
    Scaling and normalization serve different purposes in your preprocessing pipeline:
    
    - **Scaling** adjusts each *feature* to a common range, preventing any single feature from dominating
    - **Normalization** adjusts each *sample* to have unit length, making samples comparable regardless of magnitude
    
    This combination is particularly useful for neural networks and distance-based algorithms where both feature balance and sample comparability matter.

### Normalization

Normalization transforms each sample to have unit norm. It's helpful in tasks such as text classification or when using distance-based metrics.

```python
def l2_normalize(array):
    """
    Normalize array rows to unit L2 norm.
    """
    # Compute L2 norm along axis 1 (for each row)
    norms = array.norm(p=2, axis=[1])
    # Reshape norms to allow broadcasting (add dimension)
    norms = norms.reshape([norms.shape[0], 1])
    # Divide each row by its norm (broadcasting will work)
    return array / norms

# Create sample data
data_rn = rnp.NumArray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32")

# Normalize the data
normalized_data_rn = l2_normalize(data_rn)
print("Data after L2 normalization:", normalized_data_rn.tolist())
```

### Concatenating Features

If you have multiple feature sets (e.g., one from images, another from text), RustyNum can concatenate them into a single array:

```python
# Create two feature sets from our normalized data
features1 = normalized_data_rn[:, :2]  # First two columns (shape: [2, 2])
features2 = rnp.ones([2, 2])          # Match features1 shape: [2, 2]

# Combine them
combined = rnp.concatenate([features1, features2], axis=1)
print("Combined feature shape:", combined.shape)  # Expected: [2, 4]
```

---

## Integrating with ML Libraries

After scaling or normalizing, you can convert back to NumPy arrays for compatibility with libraries like scikit-learn:

```python
# Convert our preprocessed data back to NumPy
X_train = np.array(normalized_data_rn.tolist(), dtype=np.float32)

# Create dummy labels
y_train = np.random.randint(0, 2, size=(100,))

# Train a simple model
model = LogisticRegression()
model.fit(X_train, y_train)

print("Model coefficients:", model.coef_)
```

This approach provides a quick path for ML experimentation.

---

## Advanced Operations

### Element-wise Operations

RustyNum supports arithmetic and math functions directly on NumArray objects. For example, you can apply functions like `sigmoid`, `exp`, and `log` element-wise:

```python
# Element-wise operations example
data = rnp.NumArray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32")
print("Original data:", data.tolist())

# Apply the sigmoid function
sigmoid_data = data.sigmoid()
print("After applying sigmoid:", sigmoid_data.tolist())

# Compute the exponential and logarithm of each element
exp_data = data.exp()
log_data = data.log()
print("Exponential:", exp_data.tolist())
print("Logarithm:", log_data.tolist())
```

### Matrix Multiplication

You can perform matrix multiplication with RustyNum using the @ operator, just like with NumPy:

```python
# Matrix multiplication example
a = rnp.NumArray([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
b = rnp.NumArray([[5.0, 6.0], [7.0, 8.0]], dtype="float32")
product = a @ b
print("Matrix multiplication result:", product.tolist())
```

---

## Tips and Best Practices

1. **Batch Operations**: RustyNum is most beneficial when dealing with whole arrays instead of looping element-by-element.  
2. **Consistent Data Types**: Ensure your arrays use `float32` or `float64` as needed. Mixing types can cause errors.  
3. **Performance Testing**: For large datasets, measure performance gains with profiling tools or benchmarks.  

---

## Next Steps

- Check out the other [Tutorials](../index.md) for deeper dives into RustyNum's capabilities.  
- Review the [API Reference](../api/index.md) for more advanced methods.  
- Join the community on [GitHub Discussions](https://github.com/IgorSusmelj/rustynum/discussions) to share ideas or ask questions.

Preprocessing data can be a bottleneck in many ML pipelines. By harnessing RustyNum, you might reduce that overhead while still benefiting from a Python-friendly workflow. Give these techniques a try in your own projects and see how RustyNum fits into your machine learning stack!

---

**Further reading**: [Installation](../installation.md), [Quick Start](../quick-start.md), [API Reference](../api/index.md).
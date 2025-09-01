---
title: RustyNum vs NumPy performance benchmarks and guidance
description: Reproducible benchmarks that compare RustyNum with NumPy for mean, min, dot, and matrix multiplication in Python, with setup and tips.
---

# RustyNum vs NumPy performance

This is the canonical comparison for RustyNum and NumPy. It includes setup, a small runner you can copy, and notes on when each library is a good choice. Use it as a reference that stays current, and see the blog for dated releases that summarize new results.

RustyNum is a NumPy compatible array library for Python that uses Rust SIMD to accelerate common operations.

---

## What we measure

- Mean over large vectors
- Minimum over large vectors
- Dot product of a matrix and a vector
- Matrix multiplication of two square matrices

These cases map to common data tasks and machine learning preprocessing on a single CPU.

---

## Install and record environment

Install the packages.

```bash
python -V
pip install -U rustynum numpy
```

Record your environment when you share results.

```python
import numpy as np, platform, sys, rustynum as rnp
print("Python", sys.version.split()[0])
print("NumPy", np.__version__)
print("OS", platform.platform())
print("CPU", platform.processor())
print("RustyNum", rnp.__version__)
```

NumPy performance depends on the BLAS that ships with your wheel. OpenBLAS or MKL can change results. That is expected.

---

## Benchmark runner

The script below times four operations. It warms up first and reports medians. Sizes are modest so it runs fast during local testing.

```python
import time, math, numpy as np
import rustynum as rnp
from statistics import median

def bench(fn, repeats=7, warmup=2):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return median(times)

def as_rn(x):
    if x.dtype == np.float32:
        return rnp.NumArray(x.flatten().tolist(), dtype="float32").reshape(list(x.shape))
    elif x.dtype == np.float64:
        return rnp.NumArray(x.flatten().tolist(), dtype="float64").reshape(list(x.shape))
    else:
        raise ValueError("Use float32 or float64")

def run_suite(n=1_000_000, m=1000):
    results = []

    # 1) mean over vector
    vec32 = np.random.rand(n).astype(np.float32)
    vec32_rn = as_rn(vec32)

    t_np = bench(lambda: float(np.mean(vec32)))
    t_rn = bench(lambda: float(vec32_rn.mean().item()))
    results.append(("mean", f"{n}", t_rn, t_np, t_np / t_rn if t_rn > 0 else math.nan))

    # 2) minimum over vector
    t_np = bench(lambda: float(np.min(vec32)))
    t_rn = bench(lambda: float(vec32_rn.min()))
    results.append(("min", f"{n}", t_rn, t_np, t_np / t_rn if t_rn > 0 else math.nan))

    # 3) matrix vector dot
    mat = np.random.rand(m, m).astype(np.float32)
    vec = np.random.rand(m).astype(np.float32)
    mat_rn = as_rn(mat)
    vec_rn = as_rn(vec)

    t_np = bench(lambda: np.dot(mat, vec))
    t_rn = bench(lambda: mat_rn.dot(vec_rn))
    results.append(("matrix@vector", f"{m}x{m} · {m}", t_rn, t_np, t_np / t_rn if t_rn > 0 else math.nan))

    # 4) matrix matrix
    a = np.random.rand(m, m).astype(np.float32)
    b = np.random.rand(m, m).astype(np.float32)
    a_rn = as_rn(a)
    b_rn = as_rn(b)

    t_np = bench(lambda: a @ b)
    t_rn = bench(lambda: a_rn @ b_rn)
    results.append(("matrix@matrix", f"{m}x{m}", t_rn, t_np, t_np / t_rn if t_rn > 0 else math.nan))

    print("\nOperation, Size, RustyNum s, NumPy s, Speedup NumPy/RustyNum")
    for op, size, trn, tnp, sp in results:
        print(f"{op}, {size}, {trn:.6f}, {tnp:.6f}, {sp:.2f}x")

if __name__ == "__main__":
    run_suite(n=1_000_000, m=1000)
```

Run the file.

```bash
python benchmarks.py
```

---

## Reading the results

- Mean and minimum are memory bound. RustyNum often does well due to SIMD friendly loops.
- Matrix vector speed depends on memory access and BLAS.
- Matrix matrix can favor NumPy on large sizes with tuned BLAS. RustyNum can be close on medium sizes.

Small numeric differences are normal in floating point code.

---

## Example results

These numbers show benchmark results from RustyNum 0.1.4 vs NumPy 1.24.4 on an Apple M1 Pro laptop with float32 inputs.

| Operation     | Size      | RustyNum us | NumPy us    | Speedup NumPy over RustyNum |
| ------------- | --------- | ----------- | ----------- | --------------------------- |
| mean          | 1e3       | 8.8993      | 22.6300     | 2.54x                       |
| min           | 1e3       | 10.1423     | 28.9693     | 2.86x                       |
| matrix@vector | 1000x1000 | 10,041.6093 | 24,990.2646 | 2.49x                       |
| matrix@matrix | 500x500   | 7,010.6638  | 14,878.9556 | 2.12x                       |

Your machine will differ. Use the runner above to collect your own data.

---

## Tips for fair runs

- Stick to float32 or float64 and be consistent.
- Avoid Python loops in the hot path.
- Warm up first and report medians.
- Share CPU model, versions, and BLAS choice.

---

## When to pick RustyNum

- You want fast reductions or transforms with a small wheel.
- You need a compact dependency for packaging on servers or edge.
- You want SIMD backed methods without heavy external libraries.

## When to stay with NumPy

- You run heavy dense linear algebra that benefits from a tuned BLAS.
- You need a very wide API that RustyNum does not cover yet.

---

## Next steps

- Start with the [Quick Start](../quick-start.md).
- Try the tutorial on [Replacing Core NumPy Calls](../tutorials/replacing-numpy-for-faster-analytics.md).
- Learn matrix math in [Getting Better Matrix Operations](../tutorials/better-matrix-operations.md).
- Install or upgrade with the [Installation Guide](../installation.md).

---

**Further reading**: [Installation](../installation.md), [Quick Start](../quick-start.md), [API Reference](../api/index.md).

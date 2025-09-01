---
title: Install RustyNum on macOS Linux and Windows for Python 3.8 to 3.13
description: Step by step install guide for RustyNum on Apple Silicon Intel Linux and Windows. Verify the install and fix common issues.
---
# Installation Guide

Get started with RustyNum by following this installation guide. Whether you're using Python for data analysis or contributing to the core library, this page covers everything you need.

---

## ✅ Supported Platforms and Versions

### Supported Python Versions
- Python 3.8, 3.9, 3.10, 3.11, 3.12, 3.13

### Supported Operating Systems
- **Windows**: x86
- **Linux**: x86 & ARM
- **MacOS**: x86 & ARM (Apple Silicon support)

---

## 📦 Installation Options

### Using pip (Recommended)
The easiest way to install RustyNum is via [PyPI](https://pypi.org/project/rustynum/):

```bash
pip install rustynum
```

### Using Poetry
To install RustyNum using Poetry, run:

```bash
poetry add rustynum
```

### Using Rye
To add RustyNum to your Rye project:

```bash
rye add rustynum
```

### Using Conda
To install RustyNum in a conda environment, first activate your environment and then use pip:

```bash
conda activate your-environment
pip install rustynum
```

### Verify the Installation

Test your installation by running the following Python code:

```python
import rustynum as rnp

a = rnp.zeros([2, 3])
print(a)
```

If the installation is successful, you should see the output:

```
[[0. 0. 0.]
 [0. 0. 0.]]
```

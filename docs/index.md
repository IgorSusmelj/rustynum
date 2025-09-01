---
title: RustyNum a fast NumPy alternative written in Rust
description: RustyNum is a SIMD accelerated numerical library for Python with a NumPy like API. Learn what it is, why it is fast, and how to use it in Python.
---
![RustyNum logo and wordmark](assets/rustynum-banner.png?raw=true "RustyNum")

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "SoftwareApplication",
  "name": "RustyNum",
  "applicationCategory": "DeveloperApplication",
  "operatingSystem": "Windows, Linux, macOS",
  "softwareVersion": "latest",
  "url": "https://rustynum.com",
  "downloadUrl": "https://pypi.org/project/rustynum/",
  "programmingLanguage": "Python, Rust",
  "license": "https://opensource.org/licenses/MIT",
  "description": "SIMD accelerated numerical library for Python with a NumPy like interface."
}
</script>

# Welcome to RustyNum!

**RustyNum** is a high-performance numerical computation library written in Rust, designed to be faster, lighter, and simpler than traditional solutions. With seamless Python bindings, RustyNum empowers developers and data scientists to achieve efficient computation with minimal overhead.

!!! warning "Early Development Notice"
    RustyNum is currently in early development and has some important limitations:

    - **Limited Data Type Support**: Currently only supports `float32`, `float64`, and experimental `uint8`
    - **Basic Operation Set**: Many NumPy operations are not yet implemented
    - **Partial Multithreading**: Only matrix operations support parallel processing
    
    We're actively working on expanding these capabilities. Check our [GitHub repository](https://github.com/IgorSusmelj/rustynum) for the latest updates and planned features.

---

## 🚀 What is RustyNum?

RustyNum is built to demonstrate the potential of **Rust's SIMD capabilities** while providing a **NumPy-like interface** for Python users. Whether you're working on machine learning, data analysis, or scientific computing, RustyNum offers:

- **Up to 2.86x faster computations** than NumPy for key operations.
- **Lightweight and portable** Python wheels (300 KB vs. NumPy’s ~15 MB).
- **Minimal dependencies**, ensuring quick and easy deployment.

---

## 🏆 Key Features

### High Performance
- Utilizes Rust's `portable_simd` for lightning-fast computations.
- Optimized for matrix and vector operations, with support for advanced numerical tasks.
- Matrix operations use additional multithreading for parallelization. Multithreading is currently not supported for any other operations.

### Seamless Python Integration
- Python bindings offer a familiar interface for NumPy users.
- Compatible with popular Python versions (3.8 - 3.13).

### Lightweight and Portable
- No external Rust crates used—keeping the codebase simple and transparent.
- Tiny footprint ensures quick installations and smooth deployments.

---

## 📚 Get Started with RustyNum

Ready to explore RustyNum? Here’s how you can dive in:

1. **[Installation](installation.md)**: Install RustyNum with a single `pip` command.
2. **[Quick Start](quick-start.md)**: Learn the basics of using RustyNum.
3. **[Tutorials](tutorials/index.md)**: Explore real-world examples and advanced guides.
4. **[API Reference](api/index.md)**: Dive deep into RustyNum's Python API.

---

## 🌟 Why RustyNum?

- **Speed**: Perform computations faster than NumPy.
- **Familiarity**: Built with Python users in mind—no steep learning curve.
- **Flexibility**: Ideal for machine learning, data preprocessing, and scientific research.
- **Open Source**: Contribute and be part of a growing community.

---

## 🤝 Contribute

RustyNum is open source and powered by contributors like you! Whether you’re a developer, data scientist, or enthusiast, your input matters.

- **[GitHub Repository](https://github.com/IgorSusmelj/rustynum)**: Explore the codebase, report issues, or submit pull requests.
- **Join the Community**: Share ideas, ask questions, or propose features.

---

## 📩 Stay Updated

Follow the journey and stay updated on RustyNum's latest developments:

- Star the [GitHub repository](https://github.com/IgorSusmelj/rustynum) to support the project!

---

<div style="text-align: center;">
    <a href="quick-start" class="md-button md-button--primary">Get Started</a>
    <a href="https://github.com/IgorSusmelj/rustynum" class="md-button">Contribute on GitHub</a>
</div>
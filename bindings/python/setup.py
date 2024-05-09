from setuptools import setup, find_packages
from setuptools_rust import Binding, RustExtension

setup(
    name="rustynum",
    version="0.1.1",
    description="Python wrapper for the RustyNum library bindings",
    author="IgorSusmelj",
    author_email="isusmelj@gmail.com",
    license="MIT",
    rust_extensions=[
        RustExtension(
            "rustynum._rustynum",
            binding=Binding.PyO3,
            debug=False,
            path="Cargo.toml",
        )
    ],
    package_data={"rustynum": ["py.typed"]},
    packages=find_packages(),  # Automatically find packages
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # Add classifiers to help users find your package
        "Programming Language :: Python :: 3",
        "Programming Language :: Rust",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.8",
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "numpy>=1.25.0",
            "pytest-benchmark>=4.0.0",
        ],
    },
    tests_require=[
        "pytest>=8.0.0",
        "numpy>=1.25.0",
        "pytest-benchmark>=4.0.0",
    ],
)

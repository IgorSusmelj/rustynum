from setuptools import find_packages, setup
from setuptools_rust import Binding, RustExtension

setup(
    name="rustynum",
    version="0.1.6",
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
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Rust",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
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

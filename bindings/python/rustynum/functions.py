# bindings/python/rustynum/creation_functions.py
from typing import List, Optional, Union

from . import _rustynum
from .num_array_class import NumArray


def zeros(shape: List[int], dtype: str = "float32") -> "NumArray":
    """
    Creates a NumArray of zeros with the specified shape and dtype.

    Parameters:
        shape: Shape of the NumArray.
        dtype: Data type of the NumArray ('float32' or 'float64').

    Returns:
        A new NumArray filled with zeros.
    """
    if dtype == "float32":
        return NumArray(_rustynum.zeros_f32(shape), dtype=dtype)
    elif dtype == "float64":
        return NumArray(_rustynum.zeros_f64(shape), dtype=dtype)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def ones(shape: List[int], dtype: str = "float32") -> "NumArray":
    """
    Creates a NumArray of ones with the specified shape and dtype.

    Parameters:
        shape: Shape of the NumArray.
        dtype: Data type of the NumArray ('float32' or 'float64').

    Returns:
        A new NumArray filled with ones.
    """
    if dtype == "float32":
        return NumArray(_rustynum.ones_f32(shape), dtype=dtype)
    elif dtype == "float64":
        return NumArray(_rustynum.ones_f64(shape), dtype=dtype)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def arange(
    start: float, stop: float, step: float, dtype: str = "float32"
) -> "NumArray":
    """
    Creates a NumArray with evenly spaced values within a given interval.

    Parameters:
        start: Start of the interval.
        stop: End of the interval.
        step: Spacing between values.
        dtype: Data type of the NumArray ('float32' or 'float64').

    Returns:
        A new NumArray with evenly spaced values.
    """
    if dtype == "float32":
        return NumArray(_rustynum.arange_f32(start, stop, step), dtype=dtype)
    elif dtype == "float64":
        return NumArray(_rustynum.arange_f64(start, stop, step), dtype=dtype)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def linspace(start: float, stop: float, num: int, dtype: str = "float32") -> "NumArray":
    """
    Creates a NumArray with evenly spaced values within a given interval.

    Parameters:
        start: Start of the interval.
        stop: End of the interval.
        num: Number of samples to generate.
        dtype: Data type of the NumArray ('float32' or 'float64').

    Returns:
        A new NumArray with evenly spaced values.
    """
    if dtype == "float32":
        return NumArray(_rustynum.linspace_f32(start, stop, num), dtype=dtype)
    elif dtype == "float64":
        return NumArray(_rustynum.linspace_f64(start, stop, num), dtype=dtype)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def mean(a: "NumArray") -> float:
    if isinstance(a, NumArray):
        return a.mean()
    elif isinstance(a, (int, float)):
        return NumArray([a], dtype="float32").mean()
    else:
        raise TypeError(
            "Unsupported operand type for mean: '{}'".format(type(a).__name__)
        )


def median(a: "NumArray") -> float:
    if isinstance(a, NumArray):
        return a.median()
    elif isinstance(a, (int, float)):
        return NumArray([a], dtype="float32").median()
    else:
        raise TypeError(
            "Unsupported operand type for median: '{}'".format(type(a).__name__)
        )


def min(a: "NumArray") -> float:
    if isinstance(a, NumArray):
        return a.min()
    elif isinstance(a, (int, float)):
        return NumArray([a], dtype="float32").min()
    else:
        raise TypeError(
            "Unsupported operand type for min: '{}'".format(type(a).__name__)
        )


def max(a: "NumArray") -> float:
    if isinstance(a, NumArray):
        return a.max()
    elif isinstance(a, (int, float)):
        return NumArray([a], dtype="float32").max()
    else:
        raise TypeError(
            "Unsupported operand type for max: '{}'".format(type(a).__name__)
        )


def dot(a: "NumArray", b: "NumArray") -> Union[float, "NumArray"]:
    if isinstance(a, NumArray) and isinstance(b, NumArray):
        return a.dot(b)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        out = NumArray([a], dtype="float32").dot(NumArray([b], dtype="float32"))
        return NumArray([out], dtype="float32").item()
    else:
        raise TypeError("Both arguments must be NumArray instances.")


def exp(a: "NumArray") -> "NumArray":
    if isinstance(a, NumArray):
        return a.exp()
    elif isinstance(a, (int, float)):
        return NumArray([a], dtype="float32").exp()
    else:
        raise TypeError(
            "Unsupported operand type for exp: '{}'".format(type(a).__name__)
        )


def log(a: "NumArray") -> "NumArray":
    if isinstance(a, NumArray):
        return a.log()
    elif isinstance(a, (int, float)):
        return NumArray([a], dtype="float32").log()
    else:
        raise TypeError(
            "Unsupported operand type for log: '{}'".format(type(a).__name__)
        )


def sigmoid(a: "NumArray") -> "NumArray":
    if isinstance(a, NumArray):
        return a.sigmoid()
    elif isinstance(a, (int, float)):
        return NumArray([a], dtype="float32").sigmoid()
    else:
        raise TypeError(
            "Unsupported operand type for sigmoid: '{}'".format(type(a).__name__)
        )


def concatenate(arrays: List["NumArray"], axis: int = 0) -> "NumArray":
    # axis can be any integer, but most of the time it would only be 0 or 1
    if not all(isinstance(a, NumArray) for a in arrays):
        raise TypeError("All elements in 'arrays' must be NumArray instances.")
    if not all(a.dtype == arrays[0].dtype for a in arrays):
        raise ValueError("dtype mismatch between arrays")

    if arrays[0].dtype == "float32":
        return NumArray(
            _rustynum.concatenate_f32([a.inner for a in arrays], axis), dtype="float32"
        )
    elif arrays[0].dtype == "float64":
        return NumArray(
            _rustynum.concatenate_f64([a.inner for a in arrays], axis), dtype="float64"
        )
    else:
        raise ValueError("Unsupported dtype for concatenation")


def norm(
    a: "NumArray", p: int = 2, axis: Optional[List[int]] = None, keepdims: bool = False
) -> "NumArray":
    if a.dtype == "float32":
        return NumArray(_rustynum.norm_f32(a.inner, p, axis, keepdims), dtype="float32")
    elif a.dtype == "float64":
        return NumArray(_rustynum.norm_f64(a.inner, p, axis, keepdims), dtype="float64")
    else:
        raise ValueError(f"Unsupported dtype for norm: {a.dtype}")

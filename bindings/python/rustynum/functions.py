"""
RustyNum Functions Module

This module provides NumPy-like functions for creating and manipulating NumArray objects.
These functions offer a familiar interface for users coming from NumPy while leveraging
the performance benefits of Rust-based implementations.
"""

from typing import List, Optional, Union

from . import _rustynum
from .num_array_class import NumArray


def zeros(shape: List[int], dtype: str = "float32") -> NumArray:
    """
    Create a new array filled with zeros.

    Args:
        shape: Shape of the new array as a list of integers.
        dtype: Data type ('float32' or 'float64').

    Returns:
        New NumArray filled with zeros.

    Raises:
        ValueError: If dtype is unsupported.

    Examples:
        >>> import rustynum as rn
        >>> arr = rn.zeros([2, 3])
        >>> arr.shape
        [2, 3]
        >>> arr.tolist()
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    """
    if dtype == "float32":
        return NumArray(_rustynum.zeros_f32(shape), dtype=dtype)
    elif dtype == "float64":
        return NumArray(_rustynum.zeros_f64(shape), dtype=dtype)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def ones(shape: List[int], dtype: str = "float32") -> NumArray:
    """
    Create a new array filled with ones.

    Args:
        shape: Shape of the new array as a list of integers.
        dtype: Data type ('float32' or 'float64').

    Returns:
        New NumArray filled with ones.

    Raises:
        ValueError: If dtype is unsupported.

    Examples:
        >>> import rustynum as rn
        >>> arr = rn.ones([2, 2])
        >>> arr.tolist()
        [[1.0, 1.0], [1.0, 1.0]]
    """
    if dtype == "float32":
        return NumArray(_rustynum.ones_f32(shape), dtype=dtype)
    elif dtype == "float64":
        return NumArray(_rustynum.ones_f64(shape), dtype=dtype)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def arange(
    start: float, stop: float, step: float = 1.0, dtype: str = "float32"
) -> NumArray:
    """
    Create an array with evenly spaced values within a given interval.

    Args:
        start: Start of the interval (inclusive).
        stop: End of the interval (exclusive).
        step: Spacing between values.
        dtype: Data type ('float32' or 'float64').

    Returns:
        New NumArray with evenly spaced values.

    Raises:
        ValueError: If dtype is unsupported.

    Examples:
        >>> import rustynum as rn
        >>> arr = rn.arange(0, 10, 2)
        >>> arr.tolist()
        [0.0, 2.0, 4.0, 6.0, 8.0]
    """
    if dtype == "float32":
        return NumArray(_rustynum.arange_f32(start, stop, step), dtype=dtype)
    elif dtype == "float64":
        return NumArray(_rustynum.arange_f64(start, stop, step), dtype=dtype)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def linspace(start: float, stop: float, num: int, dtype: str = "float32") -> NumArray:
    """
    Create an array with evenly spaced values over a specified interval.

    Args:
        start: Start of the interval (inclusive).
        stop: End of the interval (inclusive).
        num: Number of samples to generate.
        dtype: Data type ('float32' or 'float64').

    Returns:
        New NumArray with evenly spaced values.

    Raises:
        ValueError: If dtype is unsupported.

    Examples:
        >>> import rustynum as rn
        >>> arr = rn.linspace(0, 1, 5)
        >>> arr.tolist()
        [0.0, 0.25, 0.5, 0.75, 1.0]
    """
    if dtype == "float32":
        return NumArray(_rustynum.linspace_f32(start, stop, num), dtype=dtype)
    elif dtype == "float64":
        return NumArray(_rustynum.linspace_f64(start, stop, num), dtype=dtype)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def mean(a: Union[NumArray, int, float]) -> Union[NumArray, float]:
    """
    Compute the arithmetic mean of array elements.

    Args:
        a: Input array or scalar value.

    Returns:
        The mean value as NumArray or float.

    Raises:
        TypeError: If input type is unsupported.

    Examples:
        >>> import rustynum as rn
        >>> arr = rn.NumArray([1, 2, 3, 4])
        >>> rn.mean(arr)
        2.5
    """
    if isinstance(a, NumArray):
        return a.mean()
    elif isinstance(a, (int, float)):
        return NumArray([a], dtype="float32").mean()
    else:
        raise TypeError(
            "Unsupported operand type for mean: '{}'".format(type(a).__name__)
        )


def median(a: Union[NumArray, int, float]) -> Union[NumArray, float]:
    """
    Compute the median of array elements.

    Args:
        a: Input array or scalar value.

    Returns:
        The median value as NumArray or float.

    Raises:
        TypeError: If input type is unsupported.
    """
    if isinstance(a, NumArray):
        return a.median()
    elif isinstance(a, (int, float)):
        return NumArray([a], dtype="float32").median()
    else:
        raise TypeError(
            "Unsupported operand type for median: '{}'".format(type(a).__name__)
        )


def min(a: Union[NumArray, int, float]) -> Union[NumArray, float]:
    """
    Return the minimum value of array elements.

    Args:
        a: Input array or scalar value.

    Returns:
        The minimum value as NumArray or float.

    Raises:
        TypeError: If input type is unsupported.
    """
    if isinstance(a, NumArray):
        return a.min()
    elif isinstance(a, (int, float)):
        return NumArray([a], dtype="float32").min()
    else:
        raise TypeError(
            "Unsupported operand type for min: '{}'".format(type(a).__name__)
        )


def max(a: Union[NumArray, int, float]) -> Union[NumArray, float]:
    """
    Return the maximum value of array elements.

    Args:
        a: Input array or scalar value.

    Returns:
        The maximum value as NumArray or float.

    Raises:
        TypeError: If input type is unsupported.
    """
    if isinstance(a, NumArray):
        return a.max()
    elif isinstance(a, (int, float)):
        return NumArray([a], dtype="float32").max()
    else:
        raise TypeError(
            "Unsupported operand type for max: '{}'".format(type(a).__name__)
        )


def dot(
    a: Union[NumArray, int, float], b: Union[NumArray, int, float]
) -> Union[float, NumArray]:
    """
    Compute dot product of two arrays or scalars.

    Args:
        a: First input array or scalar.
        b: Second input array or scalar.

    Returns:
        Dot product result as NumArray or scalar.

    Raises:
        TypeError: If input types are unsupported.

    Examples:
        >>> import rustynum as rn
        >>> a = rn.NumArray([1, 2, 3])
        >>> b = rn.NumArray([4, 5, 6])
        >>> result = rn.dot(a, b)
        >>> # result is 1*4 + 2*5 + 3*6 = 32
    """
    if isinstance(a, NumArray) and isinstance(b, NumArray):
        return a.dot(b)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        out = NumArray([a], dtype="float32").dot(NumArray([b], dtype="float32"))
        return NumArray([out], dtype="float32").item()
    else:
        raise TypeError("Both arguments must be NumArray instances or scalars.")


def exp(a: Union[NumArray, int, float]) -> NumArray:
    """
    Compute the exponential of all elements in the input.

    Args:
        a: Input array or scalar value.

    Returns:
        New NumArray with exponential of all elements.

    Raises:
        TypeError: If input type is unsupported.

    Examples:
        >>> import rustynum as rn
        >>> arr = rn.NumArray([0, 1, 2])
        >>> result = rn.exp(arr)
        >>> # result contains [1.0, 2.718..., 7.389...]
    """
    if isinstance(a, NumArray):
        return a.exp()
    elif isinstance(a, (int, float)):
        return NumArray([a], dtype="float32").exp()
    else:
        raise TypeError(
            "Unsupported operand type for exp: '{}'".format(type(a).__name__)
        )


def log(a: Union[NumArray, int, float]) -> NumArray:
    """
    Compute the natural logarithm of all elements in the input.

    Args:
        a: Input array or scalar value.

    Returns:
        New NumArray with natural logarithm of all elements.

    Raises:
        TypeError: If input type is unsupported.
    """
    if isinstance(a, NumArray):
        return a.log()
    elif isinstance(a, (int, float)):
        return NumArray([a], dtype="float32").log()
    else:
        raise TypeError(
            "Unsupported operand type for log: '{}'".format(type(a).__name__)
        )


def sigmoid(a: Union[NumArray, int, float]) -> NumArray:
    """
    Compute the sigmoid function of all elements in the input.

    The sigmoid function is defined as: 1 / (1 + exp(-x))

    Args:
        a: Input array or scalar value.

    Returns:
        New NumArray with sigmoid of all elements.

    Raises:
        TypeError: If input type is unsupported.

    Examples:
        >>> import rustynum as rn
        >>> arr = rn.NumArray([-1, 0, 1])
        >>> result = rn.sigmoid(arr)
        >>> # result contains [0.268..., 0.5, 0.731...]
    """
    if isinstance(a, NumArray):
        return a.sigmoid()
    elif isinstance(a, (int, float)):
        return NumArray([a], dtype="float32").sigmoid()
    else:
        raise TypeError(
            "Unsupported operand type for sigmoid: '{}'".format(type(a).__name__)
        )


def concatenate(arrays: List[NumArray], axis: int = 0) -> NumArray:
    """
    Join a sequence of arrays along an existing axis.

    Args:
        arrays: List of NumArray objects to concatenate.
        axis: Axis along which to concatenate (typically 0 or 1).

    Returns:
        New NumArray containing the concatenated data.

    Raises:
        TypeError: If not all elements are NumArray instances.
        ValueError: If dtypes don't match or concatenation is unsupported.

    Examples:
        >>> import rustynum as rn
        >>> a = rn.NumArray([[1, 2], [3, 4]])
        >>> b = rn.NumArray([[5, 6], [7, 8]])
        >>> result = rn.concatenate([a, b], axis=0)
        >>> result.shape
        [4, 2]
    """
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
    a: NumArray, p: int = 2, axis: Optional[List[int]] = None, keepdims: bool = False
) -> NumArray:
    """
    Compute matrix or vector norm.

    Args:
        a: Input array.
        p: Order of the norm (default: 2 for Euclidean norm).
        axis: Axis or axes along which to compute the norm.
        keepdims: Whether to keep dimensions in the result.

    Returns:
        New NumArray containing the computed norm.

    Raises:
        ValueError: If dtype is unsupported for norm computation.

    Examples:
        >>> import rustynum as rn
        >>> arr = rn.NumArray([3, 4])
        >>> result = rn.norm(arr)  # Euclidean norm
        >>> # result is 5.0 (sqrt(3^2 + 4^2))
    """
    if a.dtype == "float32":
        return NumArray(_rustynum.norm_f32(a.inner, p, axis, keepdims), dtype="float32")
    elif a.dtype == "float64":
        return NumArray(_rustynum.norm_f64(a.inner, p, axis, keepdims), dtype="float64")
    else:
        raise ValueError(f"Unsupported dtype for norm: {a.dtype}")

# bindings/python/rustynum/num_array_class.py
import itertools
import math
from typing import Any, List, Optional, Sequence, Tuple, Union

from . import _rustynum


class NumArray:
    """
    A high-performance numerical array implementation backed by Rust.

    Provides NumPy-like functionality with optimized operations for numerical computing.
    Supports multiple data types including float32, float64, and uint8.

    Examples:
        >>> # Create a 2D array
        >>> arr = NumArray([[1.0, 2.0], [3.0, 4.0]])
        >>> print(arr.shape)
        [2, 2]

        >>> # Matrix multiplication
        >>> result = arr @ arr
        >>> print(result.tolist())
        [[7.0, 10.0], [15.0, 22.0]]
    """

    # Type annotations for instance variables
    inner: Union[
        _rustynum.PyNumArrayF32, _rustynum.PyNumArrayF64, _rustynum.PyNumArrayU8
    ]
    dtype: str

    def __init__(
        self,
        data: Union[List[Any], "NumArray"],
        dtype: Optional[str] = None,
    ) -> None:
        """
        Initialize a NumArray with data and optional data type.

        Args:
            data: Nested list of numerical data or existing NumArray.
            dtype: Data type ('float32', 'float64', 'uint8'). If None, dtype is inferred.

        Raises:
            ValueError: If dtype is unsupported or data is invalid.
            NotImplementedError: If dtype is not yet implemented.

        Examples:
            >>> arr = NumArray([[1, 2], [3, 4]], dtype='float32')
            >>> arr2 = NumArray([1.0, 2.0, 3.0])  # dtype inferred as float32
        """
        if isinstance(data, NumArray):
            self.inner = data.inner
            self.dtype = data.dtype
        elif isinstance(
            data,
            (_rustynum.PyNumArrayF32, _rustynum.PyNumArrayF64, _rustynum.PyNumArrayU8),
        ):
            self.inner = data
            if isinstance(data, _rustynum.PyNumArrayF32):
                self.dtype = "float32"
            elif isinstance(data, _rustynum.PyNumArrayF64):
                self.dtype = "float64"
            elif isinstance(data, _rustynum.PyNumArrayU8):
                self.dtype = "uint8"
            else:
                self.dtype = "unknown"
        else:
            # Determine if data is nested (e.g., list of lists)
            if self._is_nested_list(data):
                # Handling for nested lists
                shape = self._infer_shape(data)
                flat_data = self._flatten(data)

                if dtype is None:
                    dtype = self._infer_dtype_from_data(flat_data)
                self.dtype = dtype

                if dtype == "float32":
                    self.inner = _rustynum.PyNumArrayF32(flat_data, shape)
                elif dtype == "float64":
                    self.inner = _rustynum.PyNumArrayF64(flat_data, shape)
                elif dtype == "uint8":
                    self.inner = _rustynum.PyNumArrayU8(flat_data, shape)
                elif dtype in ("int32", "int64"):
                    # Implement PyNumArrayInt32 and PyNumArrayInt64 similarly if needed
                    raise NotImplementedError(
                        f"dtype '{dtype}' is not yet implemented."
                    )
                else:
                    raise ValueError(f"Unsupported dtype: {dtype}")
            else:
                # Handling for non-nested lists (1D arrays)
                if isinstance(data, list):
                    flat_data = data
                else:
                    flat_data = [data]

                if dtype is None:
                    dtype = self._infer_dtype_from_data(flat_data)
                self.dtype = dtype

                if dtype == "float32":
                    self.inner = _rustynum.PyNumArrayF32(flat_data)
                elif dtype == "float64":
                    self.inner = _rustynum.PyNumArrayF64(flat_data)
                elif dtype == "uint8":
                    self.inner = _rustynum.PyNumArrayU8(flat_data)
                elif dtype in ("int32", "int64"):
                    # Implement PyNumArrayInt32 and PyNumArrayInt64 similarly if needed
                    raise NotImplementedError(
                        f"dtype '{dtype}' is not yet implemented."
                    )
                else:
                    raise ValueError(f"Unsupported dtype: {dtype}")

    @staticmethod
    def _is_nested_list(data: Any) -> bool:
        """
        Determines if the provided data is likely a nested list by checking the first element.

        Parameters:
            data: The data to check.

        Returns:
            True if data is likely a nested list, False otherwise.
        """
        if not isinstance(data, list):
            return False
        return len(data) > 0 and isinstance(data[0], list)

    @staticmethod
    def _infer_shape(data: List[Any]) -> List[int]:
        """
        Infers the shape of a nested list.

        Parameters:
            data: The nested list.

        Returns:
            A list representing the shape.
        """
        shape = []
        while isinstance(data, list):
            shape.append(len(data))
            if len(data) == 0:
                break
            data = data[0]
        return shape

    @staticmethod
    def _flatten(data: List[Any]) -> List[float]:
        """
        Flattens a nested list into a single flat list.

        Parameters:
            data: The nested list.

        Returns:
            A flat list containing all elements.
        """
        if not isinstance(data, list):
            return [data]
        return list(
            itertools.chain.from_iterable(NumArray._flatten(item) for item in data)
        )

    @staticmethod
    def _infer_dtype_from_data(data: List[Any]) -> str:
        """
        Infers the dtype from the data.

        Parameters:
            data: The flat list of data.

        Returns:
            A string representing the dtype.
        """
        first_type = type(data[0])
        if first_type is int:
            max_val = max(data[0])
            return "int32" if max_val < 2**31 else "int64"
        elif first_type is float:
            return "float32"  # Default to float32
        elif first_type is bytes or first_type is bytearray:
            return "uint8"
        else:
            raise ValueError("Unsupported data type in data.")

    @property
    def shape(self) -> List[int]:
        """
        Get the shape of the array.

        Returns:
            List of integers representing the dimensions of the array.

        Examples:
            >>> arr = NumArray([[1, 2, 3], [4, 5, 6]])
            >>> arr.shape
            [2, 3]
        """
        return self.inner.shape()

    def __getitem__(self, key: Union[int, slice, Tuple[Any, ...]]) -> "NumArray":
        """
        Get item(s) at the specified index or slice.

        Supports single-axis flipping using slice with step=-1.

        Args:
            key: Index, slice, or tuple of indices/slices for access.

        Returns:
            New NumArray with the sliced data.

        Raises:
            IndexError: If index is out of bounds.
            TypeError: If index type is unsupported.
            NotImplementedError: If slice step is not supported.
        """
        # Normalize the key to a tuple for uniform processing
        if not isinstance(key, tuple):
            key = (key,)

        # Ensure the number of indices does not exceed the number of dimensions
        if len(key) > len(self.shape):
            raise IndexError("Too many indices for NumArray")

        # Start with the current NumArray
        result = self

        # Iterate over each dimension and corresponding key
        for axis, k in enumerate(key):
            if isinstance(k, slice):
                # Check if the slice has a step of -1 (indicating a flip)
                if k.step == -1:
                    # Flip the specified axis
                    result = result.flip(axis)
                elif k.step is None or k.step == 1:
                    # Perform standard slicing
                    result = result.slice(axis, k.start, k.stop)
                else:
                    # For simplicity, only handle step=None, step=1, or step=-1
                    raise NotImplementedError(
                        "Only full slices or slice with step=-1 are supported."
                    )
            elif isinstance(k, int):
                # Handle integer indexing as before
                if k < 0:
                    k += result.shape[axis]
                if not (0 <= k < result.shape[axis]):
                    raise IndexError(
                        f"Index {k} out of bounds for axis {axis} with size {result.shape[axis]}"
                    )
                # Perform the integer indexing by slicing the data
                result = NumArray(result.inner.slice(axis, k, k + 1), dtype=self.dtype)
            else:
                raise TypeError(f"Unsupported index type: {type(k)}")

        return result

    def __str__(self) -> str:
        """Return string representation of the array data."""
        return str(self.tolist())

    def __repr__(self) -> str:
        """Return detailed string representation of the NumArray."""
        return f"NumArray(data={self.tolist()}, dtype={self.dtype})"

    def __matmul__(self, other: "NumArray") -> "NumArray":
        """
        Matrix multiplication using @ operator.

        Args:
            other: Another NumArray for matrix multiplication.

        Returns:
            New NumArray containing the result of matrix multiplication.
        """
        return self.matmul(other)

    def __rmatmul__(self, other: "NumArray") -> "NumArray":
        """
        Right matrix multiplication using @ operator.

        Args:
            other: Another NumArray for matrix multiplication.

        Returns:
            New NumArray containing the result of matrix multiplication.
        """
        return other.matmul(self)

    def item(self) -> Union[int, float]:
        """
        Extract a scalar from a single-element array.

        Returns:
            The scalar value from the array.

        Raises:
            ValueError: If array contains more than one element.

        Examples:
            >>> arr = NumArray([42.0])
            >>> arr.item()
            42.0
        """
        flat_list = self.tolist()
        if len(flat_list) == 1:
            return flat_list[0]
        else:
            raise ValueError("Can only convert an array of size 1 to a Python scalar")

    def reshape(self, shape: List[int]) -> "NumArray":
        """
        Return a new array with the specified shape.

        Args:
            shape: New shape for the NumArray.

        Returns:
            New NumArray with the reshaped data.

        Examples:
            >>> arr = NumArray([1, 2, 3, 4])
            >>> reshaped = arr.reshape([2, 2])
            >>> reshaped.shape
            [2, 2]
        """
        reshaped_inner = self.inner.reshape(shape)
        return NumArray(reshaped_inner, dtype=self.dtype)

    def matmul(self, other: "NumArray") -> "NumArray":
        """
        Compute matrix multiplication with another NumArray.

        Args:
            other: Another NumArray for matrix multiplication.

        Returns:
            New NumArray containing the result of matrix multiplication.

        Raises:
            ValueError: If dtypes don't match or arrays aren't 2D.

        Examples:
            >>> a = NumArray([[1, 2], [3, 4]])
            >>> b = NumArray([[5, 6], [7, 8]])
            >>> result = a.matmul(b)
            >>> result.tolist()
            [[19, 22], [43, 50]]
        """
        if self.dtype != other.dtype:
            raise ValueError("dtype mismatch between arrays")

        if self.dtype == "float32":
            # Ensure both arrays are 2D for matrix multiplication
            if len(self.shape) != 2 or len(other.shape) != 2:
                raise ValueError(
                    "Both NumArray instances must be 2D for matrix multiplication."
                )
            result = _rustynum.matmul_f32(self.inner, other.inner)
        elif self.dtype == "float64":
            # Ensure both arrays are 2D for matrix multiplication
            if len(self.shape) != 2 or len(other.shape) != 2:
                raise ValueError(
                    "Both NumArray instances must be 2D for matrix multiplication."
                )
            result = _rustynum.matmul_f64(self.inner, other.inner)
        else:
            raise ValueError("Unsupported dtype for matrix multiplication")

        return NumArray(result, dtype=self.dtype)

    def dot(self, other: "NumArray") -> "NumArray":
        """
        Compute dot product with another NumArray.

        Args:
            other: Another NumArray for dot product computation.

        Returns:
            New NumArray containing the result of the dot product.

        Raises:
            ValueError: If dtypes don't match or operation is unsupported.
        """
        if self.dtype != other.dtype:
            raise ValueError("dtype mismatch between arrays")

        if self.dtype == "float32":
            result = _rustynum.dot_f32(self.inner, other.inner)
        elif self.dtype == "float64":
            result = _rustynum.dot_f64(self.inner, other.inner)
        else:
            raise ValueError("Unsupported dtype for dot product")

        return NumArray(result, dtype=self.dtype)

    def mean(
        self, axis: Optional[Union[int, Sequence[int]]] = None
    ) -> Union["NumArray", float]:
        """
        Compute the arithmetic mean along specified axis.

        Args:
            axis: Axis or axes along which to compute the mean.
                  If None, compute mean of all elements.

        Returns:
            NumArray with mean values along specified axis, or scalar if axis is None.

        Examples:
            >>> arr = NumArray([[1, 2], [3, 4]])
            >>> arr.mean()  # Mean of all elements
            2.5
            >>> arr.mean(axis=0).tolist()  # Mean along axis 0
            [2.0, 3.0]
        """
        axis_list = [axis] if isinstance(axis, int) else axis
        result = (
            _rustynum.mean_f32(self.inner, axis_list)
            if self.dtype == "float32"
            else _rustynum.mean_f64(self.inner, axis_list)
        )
        return NumArray(result, dtype=self.dtype)

    def median(
        self, axis: Optional[Union[int, Sequence[int]]] = None
    ) -> Union["NumArray", float]:
        """
        Compute the median along specified axis.

        Args:
            axis: Axis or axes along which to compute the median.
                  If None, compute median of all elements.

        Returns:
            NumArray with median values along specified axis, or scalar if axis is None.
        """
        axis_list = [axis] if isinstance(axis, int) else axis
        result = (
            _rustynum.median_f32(self.inner, axis_list)
            if self.dtype == "float32"
            else _rustynum.median_f64(self.inner, axis_list)
        )
        return NumArray(result, dtype=self.dtype)

    def min(
        self, axis: Optional[Union[int, Sequence[int]]] = None
    ) -> Union["NumArray", float]:
        """
        Return the minimum along specified axis.

        Args:
            axis: Axis or axes along which to find the minimum.
                  If None, return minimum of all elements.

        Returns:
            NumArray with minimum values along specified axis, or scalar if axis is None.
        """
        if axis is None:
            return (
                _rustynum.min_f32(self.inner)
                if self.dtype == "float32"
                else _rustynum.min_f64(self.inner)
            )

        axis_list = [axis] if isinstance(axis, int) else axis
        result = (
            _rustynum.min_axis_f32(self.inner, axis_list)
            if self.dtype == "float32"
            else _rustynum.min_axis_f64(self.inner, axis_list)
        )
        return NumArray(result, dtype=self.dtype)

    def max(
        self, axis: Optional[Union[int, Sequence[int]]] = None
    ) -> Union["NumArray", float]:
        """
        Return the maximum along specified axis.

        Args:
            axis: Axis or axes along which to find the maximum.
                  If None, return maximum of all elements.

        Returns:
            NumArray with maximum values along specified axis, or scalar if axis is None.
        """
        if axis is None:
            return (
                _rustynum.max_f32(self.inner)
                if self.dtype == "float32"
                else _rustynum.max_f64(self.inner)
            )

        axis_list = [axis] if isinstance(axis, int) else axis
        result = (
            _rustynum.max_axis_f32(self.inner, axis_list)
            if self.dtype == "float32"
            else _rustynum.max_axis_f64(self.inner, axis_list)
        )
        return NumArray(result, dtype=self.dtype)

    def __imul__(self, scalar: Union[int, float]) -> "NumArray":
        """
        In-place multiplication by a scalar.

        Args:
            scalar: Scalar value to multiply by.

        Returns:
            Self (modified in-place).
        """
        self.inner.__imul__(scalar)
        return self

    def __add__(self, other: Union["NumArray", int, float]) -> "NumArray":
        """
        Add another NumArray or scalar to this array.

        Args:
            other: NumArray or scalar to add.

        Returns:
            New NumArray with the result of the addition.

        Raises:
            ValueError: If dtypes don't match between arrays.
            TypeError: If operand type is unsupported.
        """
        if isinstance(other, NumArray):
            if self.dtype != other.dtype:
                raise ValueError("dtype mismatch between arrays")
            return NumArray(self.inner.add_array(other.inner), dtype=self.dtype)
        elif isinstance(other, (int, float)):
            return NumArray(self.inner.add_scalar(other), dtype=self.dtype)
        else:
            raise TypeError(
                "Unsupported operand type for +: 'NumArray' and '{}'".format(
                    type(other).__name__
                )
            )

    def __mul__(self, other: Union["NumArray", int, float]) -> "NumArray":
        """
        Multiply this array by another NumArray or scalar.

        Args:
            other: NumArray or scalar to multiply by.

        Returns:
            New NumArray with the result of the multiplication.

        Raises:
            ValueError: If dtypes don't match between arrays.
            TypeError: If operand type is unsupported.
        """
        if isinstance(other, NumArray):
            if self.dtype != other.dtype:
                raise ValueError("dtype mismatch between arrays")
            return NumArray(self.inner.mul_array(other.inner), dtype=self.dtype)
        elif isinstance(other, (int, float)):
            return NumArray(self.inner.mul_scalar(other), dtype=self.dtype)
        else:
            raise TypeError(
                "Unsupported operand type for *: 'NumArray' and '{}'".format(
                    type(other).__name__
                )
            )

    def __sub__(self, other: Union["NumArray", int, float]) -> "NumArray":
        """
        Subtract another NumArray or scalar from this array.

        Args:
            other: NumArray or scalar to subtract.

        Returns:
            New NumArray with the result of the subtraction.

        Raises:
            ValueError: If dtypes don't match between arrays.
            TypeError: If operand type is unsupported.
        """
        if isinstance(other, NumArray):
            if self.dtype != other.dtype:
                raise ValueError("dtype mismatch between arrays")
            return NumArray(self.inner.sub_array(other.inner), dtype=self.dtype)
        elif isinstance(other, (int, float)):
            return NumArray(self.inner.sub_scalar(other), dtype=self.dtype)
        else:
            raise TypeError(
                "Unsupported operand type for -: 'NumArray' and '{}'".format(
                    type(other).__name__
                )
            )

    def __truediv__(self, other: Union["NumArray", int, float]) -> "NumArray":
        """
        Divide this array by another NumArray or scalar.

        Args:
            other: NumArray or scalar to divide by.

        Returns:
            New NumArray with the result of the division.

        Raises:
            ValueError: If dtypes don't match between arrays.
            TypeError: If operand type is unsupported.
        """
        if isinstance(other, NumArray):
            if self.dtype != other.dtype:
                raise ValueError("dtype mismatch between arrays")
            result = self.inner.div_array(other.inner)
            return NumArray(result, dtype=self.dtype)
        elif isinstance(other, (int, float)):
            return NumArray(self.inner.div_scalar(other), dtype=self.dtype)
        else:
            raise TypeError(
                "Unsupported operand type for /: 'NumArray' and '{}'".format(
                    type(other).__name__
                )
            )

    def tolist(self) -> List[Any]:
        """
        Convert the NumArray to a nested Python list.

        Returns:
            Nested list representation of the array data.

        Examples:
            >>> arr = NumArray([[1, 2], [3, 4]])
            >>> arr.tolist()
            [[1.0, 2.0], [3.0, 4.0]]
        """
        flat_list = self.inner.tolist()
        shape = self.shape

        def build_nested(flat: List[Any], shape: List[int]) -> List[Any]:
            if not shape:
                raise ValueError("Shape cannot be empty")
            if len(shape) == 1:
                return flat[: shape[0]]
            else:
                step = math.prod(shape[1:])  # Number of elements in each sublist
                return [
                    build_nested(flat[i * step : (i + 1) * step], shape[1:])
                    for i in range(shape[0])
                ]

        return build_nested(flat_list, shape)

    def exp(self) -> "NumArray":
        """
        Compute the exponential of all elements.

        Returns:
            New NumArray with exponential of all elements.

        Examples:
            >>> arr = NumArray([0, 1, 2])
            >>> result = arr.exp()
            >>> # result contains [1.0, 2.718..., 7.389...]
        """
        return NumArray(self.inner.exp(), dtype=self.dtype)

    def log(self) -> "NumArray":
        """
        Compute the natural logarithm of all elements.

        Returns:
            New NumArray with natural logarithm of all elements.
        """
        return NumArray(self.inner.log(), dtype=self.dtype)

    def sigmoid(self) -> "NumArray":
        """
        Compute the sigmoid function of all elements.

        Returns:
            New NumArray with sigmoid of all elements.

        Note:
            Sigmoid function: 1 / (1 + exp(-x))
        """
        return NumArray(self.inner.sigmoid(), dtype=self.dtype)

    def concatenate(self, other: "NumArray", axis: int) -> "NumArray":
        """
        Concatenate with another NumArray along specified axis.

        Args:
            other: Another NumArray to concatenate with.
            axis: Axis along which to concatenate.

        Returns:
            New NumArray containing the concatenated data.

        Raises:
            ValueError: If dtypes don't match or shapes are incompatible.
        """
        if self.dtype != other.dtype:
            raise ValueError("dtype mismatch between arrays")
        if self.shape[1 - axis] != other.shape[1 - axis]:
            raise ValueError("Arrays must have the same shape along the specified axis")

        if self.dtype == "float32":
            result = _rustynum.concatenate_f32([self.inner, other.inner], axis)
        elif self.dtype == "float64":
            result = _rustynum.concatenate_f64([self.inner, other.inner], axis)
        else:
            raise ValueError("Unsupported dtype for concatenation")

        return NumArray(result, dtype=self.dtype)

    def flip(self, axis: Union[int, Sequence[int]]) -> "NumArray":
        """
        Flip the array along specified axis or axes.

        Args:
            axis: Axis or axes to flip along.

        Returns:
            New NumArray with flipped data.

        Raises:
            TypeError: If axis type is invalid.
        """
        if isinstance(axis, int):
            result = self.inner.flip_axis([axis])
        elif isinstance(axis, (list, tuple)):
            result = self.inner.flip_axis(list(axis))
        else:
            raise TypeError("axis must be an integer or a sequence of integers")
        return NumArray(result, dtype=self.dtype)

    def slice(self, axis: int, start: Optional[int], stop: Optional[int]) -> "NumArray":
        """
        Slice the array along a specified axis.

        Args:
            axis: Axis to slice along.
            start: Starting index of the slice (None for beginning).
            stop: Stopping index of the slice (None for end).

        Returns:
            New NumArray with sliced data.
        """
        # Handle None values by setting defaults
        if start is None:
            start = 0
        elif start < 0:
            start += self.shape[axis]

        if stop is None:
            stop = self.shape[axis]
        elif stop < 0:
            stop += self.shape[axis]

        result = self.inner.slice(axis, start, stop)
        return NumArray(result, dtype=self.dtype)

    def norm(
        self, p: int = 2, axis: Optional[List[int]] = None, keepdims: bool = False
    ) -> "NumArray":
        """
        Compute the matrix or vector norm.

        Args:
            p: Order of the norm (default: 2 for Euclidean norm).
            axis: Axis or axes along which to compute the norm.
            keepdims: Whether to keep dimensions in the result.

        Returns:
            New NumArray containing the computed norm.

        Raises:
            ValueError: If dtype is unsupported for norm computation.
        """
        if self.dtype == "float32":
            return NumArray(
                _rustynum.norm_f32(self.inner, p, axis, keepdims), dtype="float32"
            )
        elif self.dtype == "float64":
            return NumArray(
                _rustynum.norm_f64(self.inner, p, axis, keepdims), dtype="float64"
            )
        else:
            raise ValueError(f"Unsupported dtype for norm: {self.dtype}")

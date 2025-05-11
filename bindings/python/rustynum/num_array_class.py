# bindings/python/rustynum/num_array_class.py
import itertools
import math
from typing import Any, List, Optional, Sequence, Tuple, Union

from . import _rustynum


class NumArray:
    def __init__(
        self,
        data: Union[List[Any], "NumArray"],
        dtype: Union[None, str] = None,
    ) -> None:
        """
        Initializes a NumArray object with the given data and data type.

        Parameters:
            data: Nested list of numerical data or existing NumArray.
            dtype: Data type of the numerical data ('int32', 'int64', 'float32' or 'float64'). If None, dtype is inferred.
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
        Returns the shape of the array as a tuple, similar to numpy.ndarray.shape.
        """
        # Assuming that the inner Rust object has a method to get the shape
        # You may need to implement this method in Rust if it doesn't exist
        return self.inner.shape()

    def __getitem__(
        self, key: Union[int, slice, Tuple[Any]]
    ) -> Union[List[Any], "NumArray"]:
        """
        Gets the item(s) at the specified index or slice.

        Supports single-axis flipping using slice with step=-1.

        Parameters:
            key: Index, slice, or tuple of indices/slices for access.

        Returns:
            Single item or a new NumArray with the sliced data.
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
        return str(self.tolist())

    def __repr__(self) -> str:
        return f"NumArray(data={self.tolist()}, dtype={self.dtype})"

    def __matmul__(self, other: "NumArray") -> "NumArray":
        """
        Implements the @ operator for matrix multiplication.

        Parameters:
            other: Another NumArray to compute the matrix multiplication with.

        Returns:
            A new NumArray containing the result of the matrix multiplication.
        """
        return self.matmul(other)

    def __rmatmul__(self, other: "NumArray") -> "NumArray":
        """
        Implements the @ operator for right matrix multiplication.

        Parameters:
            other: Another NumArray to compute the matrix multiplication with.

        Returns:
            A new NumArray containing the result of the matrix multiplication.
        """
        return other.matmul(self)

    def item(self):
        if len(self.tolist()) == 1:
            return self.tolist()[0]
        else:
            raise ValueError("Can only convert an array of size 1 to a Python scalar")

    def reshape(self, shape: List[int]) -> "NumArray":
        """
        Reshapes the NumArray to the specified shape.

        Parameters:
            shape: New shape for the NumArray.

        Returns:
            A new NumArray with the reshaped data.
        """
        # Ensure reshape returns a new NumArray instance
        reshaped_inner = self.inner.reshape(shape)
        return NumArray(reshaped_inner, dtype=self.dtype)

    def matmul(self, other: "NumArray") -> "NumArray":
        """
        Computes the matrix multiplication with another NumArray, similar to NumPy's matmul.

        Parameters:
            other: Another NumArray to compute the matrix multiplication with.

        Returns:
            A new NumArray containing the result of the matrix multiplication.
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
        Computes the dot product or matrix multiplication with another NumArray.

        Parameters:
            other: Another NumArray to compute the dot product with.

        Returns:
            A new NumArray containing the result of the dot product or matrix multiplication.
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
        self, axis: Union[None, int, Sequence[int]] = None
    ) -> Union["NumArray", float]:
        """
        Computes the mean of the NumArray along specified axis.

        Parameters:
            axis: Optional; Axis or axis along which to compute the mean. If None, the mean
                  of all elements is computed as a scalar.

        Returns:
            A new NumArray with the mean values along the specified axis, or a scalar if no axis are given.
        """
        axis = [axis] if isinstance(axis, int) else axis
        result = (
            _rustynum.mean_f32(self.inner, axis)
            if self.dtype == "float32"
            else _rustynum.mean_f64(self.inner, axis)
        )
        return NumArray(result, dtype=self.dtype)

    def median(
        self, axis: Union[None, int, Sequence[int]] = None
    ) -> Union["NumArray", float]:
        """
        Computes the median of the NumArray along specified axis.

        Parameters:
            axis: Optional; Axis or axis along which to compute the median. If None, the median
                  of all elements is computed as a scalar.

        Returns:
            A new NumArray with the median values along the specified axis, or a scalar if no axis are given.
        """
        axis = [axis] if isinstance(axis, int) else axis
        result = (
            _rustynum.median_f32(self.inner, axis)
            if self.dtype == "float32"
            else _rustynum.median_f64(self.inner, axis)
        )
        return NumArray(result, dtype=self.dtype)

    def median(
        self, axis: Union[None, int, Sequence[int]] = None
    ) -> Union["NumArray", float]:
        """
        Computes the median of the NumArray along specified axis.

        Parameters:
            axis: Optional; Axis or axis along which to compute the median. If None, the median
                  of all elements is computed as a scalar.

        Returns:
            A new NumArray with the median values along the specified axis, or a scalar if no axis are given.
        """
        axis = [axis] if isinstance(axis, int) else axis
        result = (
            _rustynum.median_f32(self.inner, axis)
            if self.dtype == "float32"
            else _rustynum.median_f64(self.inner, axis)
        )
        return NumArray(result, dtype=self.dtype)

    def min(
        self, axis: Union[None, int, Sequence[int]] = None
    ) -> Union["NumArray", float]:
        """
        Return the minimum along the specified axis.

        Parameters:
            axis: Optional; Axis or axis along which to find the minimum. If None,
                  the minimum of all elements is computed as a scalar.

        Returns:
            A new NumArray with the minimum values along the specified axis,
            or a scalar if no axis are given.
        """
        if axis is None:
            return (
                _rustynum.min_f32(self.inner)
                if self.dtype == "float32"
                else _rustynum.min_f64(self.inner)
            )

        axis = [axis] if isinstance(axis, int) else axis
        result = (
            _rustynum.min_axis_f32(self.inner, axis)
            if self.dtype == "float32"
            else _rustynum.min_axis_f64(self.inner, axis)
        )
        return NumArray(result, dtype=self.dtype)

    def max(
        self, axis: Union[None, int, Sequence[int]] = None
    ) -> Union["NumArray", float]:
        """
        Return the maximum along the specified axis.

        Parameters:
            axis: Optional; Axis or axis along which to find the maximum. If None,
                  the maximum of all elements is computed as a scalar.

        Returns:
            A new NumArray with the maximum values along the specified axis,
            or a scalar if no axis are given.
        """
        if axis is None:
            return (
                _rustynum.max_f32(self.inner)
                if self.dtype == "float32"
                else _rustynum.max_f64(self.inner)
            )

        axis = [axis] if isinstance(axis, int) else axis
        result = (
            _rustynum.max_axis_f32(self.inner, axis)
            if self.dtype == "float32"
            else _rustynum.max_axis_f64(self.inner, axis)
        )
        return NumArray(result, dtype=self.dtype)

    def __imul__(self, scalar: float) -> "NumArray":
        """
        In-place multiplication by a scalar.
        """
        self.inner.__imul__(scalar)
        return self

    def __add__(self, other: Union["NumArray", float]) -> "NumArray":
        """
        Adds another NumArray or a scalar to the NumArray.

        Returns:
            A new NumArray with the result of the addition.
        """
        if isinstance(other, NumArray):
            if self.dtype != other.dtype:
                raise ValueError("dtype mismatch between arrays")
            # Use add_array method from bindings for NumArray addition
            return NumArray(self.inner.add_array(other.inner), dtype=self.dtype)
        elif isinstance(other, (int, float)):
            # Use add_scalar method from bindings for scalar addition
            return NumArray(self.inner.add_scalar(other), dtype=self.dtype)
        else:
            raise TypeError(
                "Unsupported operand type for +: 'NumArray' and '{}'".format(
                    type(other).__name__
                )
            )

    def __mul__(self, other: Union["NumArray", float]) -> "NumArray":
        """
        Multiplies the NumArray by another NumArray or a scalar.

        Returns:
            A new NumArray with the result of the multiplication.
        """
        if isinstance(other, NumArray):
            if self.dtype != other.dtype:
                raise ValueError("dtype mismatch between arrays")
            # Use mul_array method from bindings for NumArray multiplication
            return NumArray(self.inner.mul_array(other.inner), dtype=self.dtype)
        elif isinstance(other, (int, float)):
            # Use mul_scalar method from bindings for scalar multiplication
            return NumArray(self.inner.mul_scalar(other), dtype=self.dtype)
        else:
            raise TypeError(
                "Unsupported operand type for *: 'NumArray' and '{}'".format(
                    type(other).__name__
                )
            )

    def __sub__(self, other: Union["NumArray", float]) -> "NumArray":
        """
        Subtracts another NumArray or a scalar from the NumArray.

        Returns:
            A new NumArray with the result of the subtraction.
        """
        if isinstance(other, NumArray):
            if self.dtype != other.dtype:
                raise ValueError("dtype mismatch between arrays")
            # Use sub_array method from bindings for NumArray subtraction
            return NumArray(self.inner.sub_array(other.inner), dtype=self.dtype)
        elif isinstance(other, (int, float)):
            # Use sub_scalar method from bindings for scalar subtraction
            return NumArray(self.inner.sub_scalar(other), dtype=self.dtype)
        else:
            raise TypeError(
                "Unsupported operand type for -: 'NumArray' and '{}'".format(
                    type(other).__name__
                )
            )

    def __truediv__(self, other: Union["NumArray", float]) -> "NumArray":
        """
        Divides the NumArray by another NumArray or a scalar.

        Returns:
            A new NumArray with the result of the division.
        """
        if isinstance(other, NumArray):
            if self.dtype != other.dtype:
                raise ValueError("dtype mismatch between arrays")
            # Use div_array method from bindings for NumArray division
            # Create new NumArray with the original shape
            result = self.inner.div_array(other.inner)
            # Important: Create NumArray with the original shape preserved
            return NumArray(result, dtype=self.dtype)
        elif isinstance(other, (int, float)):
            # Use div_scalar method from bindings for scalar division
            return NumArray(self.inner.div_scalar(other), dtype=self.dtype)
        else:
            raise TypeError(
                "Unsupported operand type for /: 'NumArray' and '{}'".format(
                    type(other).__name__
                )
            )

    def tolist(self) -> List[Any]:
        flat_list = self.inner.tolist()
        shape = self.shape

        def build_nested(flat, shape):
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
        Computes the exponential of all elements in the NumArray.

        Returns:
            A new NumArray with the exponential of all elements.
        """
        return NumArray(self.inner.exp(), dtype=self.dtype)

    def log(self) -> "NumArray":
        """
        Computes the natural logarithm of all elements in the NumArray.

        Returns:
            A new NumArray with the natural logarithm of all elements.
        """
        return NumArray(self.inner.log(), dtype=self.dtype)

    def sigmoid(self) -> "NumArray":
        """
        Computes the sigmoid of all elements in the NumArray.

        Returns:
            A new NumArray with the sigmoid of all elements.
        """
        return NumArray(self.inner.sigmoid(), dtype=self.dtype)

    def concatenate(self, other: "NumArray", axis: int) -> "NumArray":
        """
        Concatenates the NumArray with another NumArray along the specified axis.

        Parameters:
            other: Another NumArray to concatenate with.
            axis: Axis along which to concatenate.

        Returns:
            A new NumArray containing the concatenated data.
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
        Flips the NumArray along the specified axis.

        Parameters:
            axis: Axis to flip along.

        Returns:
            A new NumArray with the flipped data.
        """
        if isinstance(axis, int):
            result = self.inner.flip_axis([axis])
        elif isinstance(axis, (list, tuple)):
            result = self.inner.flip_axis(list(axis))
        else:
            raise TypeError("axis must be an integer or a sequence of integers")
        return NumArray(result, dtype=self.dtype)

    def slice(
        self, axis: int, start: Union[int, None], stop: Union[int, None]
    ) -> "NumArray":
        """
        Slices the NumArray along a specified axis.

        Parameters:
            axis: Axis to slice.
            start: Starting index of the slice.
            stop: Stopping index of the slice.

        Returns:
            A new NumArray with the sliced data.
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

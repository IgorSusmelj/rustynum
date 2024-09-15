# rustynum_py_wrapper/__init__.py
from . import _rustynum
from typing import Any, List, Sequence, Union
import itertools


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
        elif isinstance(data, (_rustynum.PyNumArray32, _rustynum.PyNumArray64)):
            self.inner = data
            self.dtype = (
                "float32" if isinstance(data, _rustynum.PyNumArray32) else "float64"
            )
        else:
            # Determine if data is nested (e.g., list of lists)
            if not self._is_nested_list(data):
                if dtype is None:
                    dtype = self._infer_dtype_from_data(data)
                self.dtype = dtype

                if dtype == "float32":
                    self.inner = _rustynum.PyNumArray32(data)
                elif dtype == "float64":
                    self.inner = _rustynum.PyNumArray64(data)
                else:
                    raise ValueError(f"Unsupported dtype: {dtype}")
            else:
                # Handling for nested lists
                shape = self._infer_shape(data)
                flat_data = self._flatten(data)

                if dtype is None:
                    dtype = self._infer_dtype_from_data(flat_data)
                self.dtype = dtype

                if dtype == "float32":
                    self.inner = _rustynum.PyNumArray32(flat_data, shape)
                elif dtype == "float64":
                    self.inner = _rustynum.PyNumArray64(flat_data, shape)
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
        if all(isinstance(x, int) for x in data):
            return "int32" if all(x < 2**31 for x in data) else "int64"
        elif all(isinstance(x, float) for x in data):
            return "float32"  # Default to float32
        else:
            raise ValueError("Mixed or unsupported data types in data.")

    @property
    def shape(self) -> List[int]:
        """
        Returns the shape of the array as a tuple, similar to numpy.ndarray.shape.
        """
        # Assuming that the inner Rust object has a method to get the shape
        # You may need to implement this method in Rust if it doesn't exist
        return self.inner.shape()

    def __getitem__(self, key: Union[int, slice]) -> Union[List[float], "NumArray"]:
        """
        Gets the item(s) at the specified index or slice.

        Parameters:
            key: Index or slice for the item(s) to get.

        Returns:
            Single item or a new NumArray with the sliced data.
        """
        if isinstance(key, slice):
            start, stop, _ = key.indices(len(self.tolist()))
            sliced_data = self.inner.slice(start, stop).tolist()
            return NumArray(sliced_data, dtype=self.dtype)
        else:
            return self.tolist()[key]

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
        self, axes: Union[None, int, Sequence[int]] = None
    ) -> Union["NumArray", float]:
        """
        Computes the mean of the NumArray along specified axes.

        Parameters:
            axes: Optional; Axis or axes along which to compute the mean. If None, the mean
                  of all elements is computed as a scalar.

        Returns:
            A new NumArray with the mean values along the specified axes, or a scalar if no axes are given.
        """
        axes = [axes] if isinstance(axes, int) else axes
        result = (
            _rustynum.mean_f32(self.inner, axes)
            if self.dtype == "float32"
            else _rustynum.mean_f64(self.inner, axes)
        )
        return NumArray(result, dtype=self.dtype)

    def min(self) -> float:
        """
        Finds the minimum value in the NumArray.

        Returns:
            The minimum value as a float.
        """
        return (
            _rustynum.min_f32(self.inner)
            if self.dtype == "float32"
            else _rustynum.min_f64(self.inner)
        )

    def max(self) -> float:
        """
        Finds the maximum value in the NumArray.

        Returns:
            The maximum value as a float.
        """
        return (
            _rustynum.max_f32(self.inner)
            if self.dtype == "float32"
            else _rustynum.max_f64(self.inner)
        )

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
            return NumArray(self.inner.div_array(other.inner), dtype=self.dtype)
        elif isinstance(other, (int, float)):
            # Use div_scalar method from bindings for scalar division
            return NumArray(self.inner.div_scalar(other), dtype=self.dtype)
        else:
            raise TypeError(
                "Unsupported operand type for /: 'NumArray' and '{}'".format(
                    type(other).__name__
                )
            )

    def tolist(self) -> Union[List[float], List[List[float]]]:
        flat_list = self.inner.tolist()
        shape = self.shape
        if len(shape) == 1:
            return flat_list  # Already a 1D list, no changes needed
        elif len(shape) == 2:
            # Reshape flat list into a list of lists (2D)
            return [
                flat_list[i * shape[1] : (i + 1) * shape[1]] for i in range(shape[0])
            ]


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

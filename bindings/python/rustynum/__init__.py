# rustynum_py_wrapper/__init__.py
from . import _rustynum
from typing import Any, List, Union


class NumArray:
    def __init__(
        self, data: Union[List[float], "NumArray"], dtype: Union[None, str] = None
    ) -> None:
        """
        Initializes a NumArray object with the given data and data type.

        Parameters:
            data: List of numerical data or existing NumArray.
            dtype: Data type of the numerical data ('float32' or 'float64'). If None, dtype is inferred.
        """
        # Infer dtype if not provided
        if dtype is None:
            dtype = "float32" if all(isinstance(x, float) for x in data) else "float64"

        self.dtype = dtype

        # Initialize inner Rust object based on dtype
        if dtype == "float32":
            self.inner: "NumArray" = (
                _rustynum.PyNumArray32(data)
                if not isinstance(data, NumArray)
                else data.inner
            )
        elif dtype == "float64":
            self.inner: "NumArray" = (
                _rustynum.PyNumArray64(data)
                if not isinstance(data, NumArray)
                else data.inner
            )
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

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

    def dot(self, other: "NumArray") -> float:
        """
        Computes the dot product with another NumArray.

        Parameters:
            other: Another NumArray to compute the dot product with.

        Returns:
            The dot product as a float.
        """
        if self.dtype != other.dtype:
            raise ValueError("dtype mismatch between arrays")
        return (
            _rustynum.dot_f32(self.inner, other.inner)
            if self.dtype == "float32"
            else _rustynum.dot_f64(self.inner, other.inner)
        )

    def mean(self) -> float:
        """
        Computes the mean of the NumArray.

        Returns:
            The mean of the NumArray as a float.
        """
        return (
            _rustynum.mean_f32(self.inner)
            if self.dtype == "float32"
            else _rustynum.mean_f64(self.inner)
        )

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

    def tolist(self) -> List[float]:
        return self.inner.tolist()


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


def dot(a: "NumArray", b: "NumArray") -> float:
    if isinstance(a, NumArray) and isinstance(b, NumArray):
        return a.dot(b)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return NumArray([a], dtype="float32").dot(NumArray([b], dtype="float32"))
    else:
        raise TypeError("Both arguments must be NumArray instances.")

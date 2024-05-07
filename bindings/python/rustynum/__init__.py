# rustynum_py_wrapper/__init__.py
from . import _rustynum
from typing import Any, List, Sequence, Union


class NumArray:
    def __init__(
        self,
        data: Union[List[float], List[int], "NumArray"],
        dtype: Union[None, str] = None,
    ) -> None:
        """
        Initializes a NumArray object with the given data and data type.

        Parameters:
            data: List of numerical data or existing NumArray.
            dtype: Data type of the numerical data ('int32', 'int64', 'float32' or 'float64'). If None, dtype is inferred.
        """

        if isinstance(data, NumArray):
            self.inner = (
                data.inner
            )  # Use the existing PyNumArray32 or PyNumArray64 object
            self.dtype = data.dtype  # Use the existing dtype
        elif isinstance(data, (_rustynum.PyNumArray32, _rustynum.PyNumArray64)):
            # Directly assign the Rust object if it's already a PyNumArray32 or PyNumArray64
            self.inner = data
            self.dtype = (
                "float32" if isinstance(data, _rustynum.PyNumArray32) else "float64"
            )
        else:
            if dtype is None:
                if all(isinstance(x, int) for x in data):
                    dtype = "int32" if all(x < 2**31 for x in data) else "int64"
                elif all(isinstance(x, float) for x in data):
                    dtype = "float32"  # Assume float32 for floating-point numbers
                else:
                    raise ValueError("Data type could not be inferred from data.")

            self.dtype = dtype

            if dtype == "float32":
                self.inner = _rustynum.PyNumArray32(
                    data
                )  # Create a new Rust NumArray32 object
            elif dtype == "float64":
                self.inner = _rustynum.PyNumArray64(
                    data
                )  # Create a new Rust NumArray64 object
            else:
                raise ValueError(f"Unsupported dtype: {dtype}")

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

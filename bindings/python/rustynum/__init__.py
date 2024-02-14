# rustynum_py_wrapper/__init__.py
from . import _rustynum


class NumArray:
    def __init__(self, data, dtype=None):
        # Infer dtype if not provided
        if dtype is None:
            dtype = "float32" if all(isinstance(x, float) for x in data) else "float64"

        self.dtype = dtype

        # Initialize inner Rust object based on dtype
        if dtype == "float32":
            self.inner = (
                _rustynum.PyNumArray32(data)
                if not isinstance(data, NumArray)
                else data.inner
            )
        elif dtype == "float64":
            self.inner = (
                _rustynum.PyNumArray64(data)
                if not isinstance(data, NumArray)
                else data.inner
            )
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

    def __getitem__(self, key):
        if isinstance(key, slice):
            # Convert Python slice to Rust start and end indices
            start, stop, _ = key.indices(len(self.tolist()))

            # Call the slice method and convert result to Python list
            sliced_data = self.inner.slice(start, stop).tolist()

            # Create a new NumArray instance with the sliced data
            return NumArray(sliced_data, dtype=self.dtype)
        else:
            # Handle single index access
            return self.tolist()[key]

    def dot(self, other):
        if self.dtype != other.dtype:
            raise ValueError("dtype mismatch between arrays")
        return (
            _rustynum.dot_f32(self.inner, other.inner)
            if self.dtype == "float32"
            else _rustynum.dot_f64(self.inner, other.inner)
        )

    def mean(self):
        return (
            _rustynum.mean_f32(self.inner)
            if self.dtype == "float32"
            else _rustynum.mean_f64(self.inner)
        )

    def min(self):
        return (
            _rustynum.min_f32(self.inner)
            if self.dtype == "float32"
            else _rustynum.min_f64(self.inner)
        )

    def max(self):
        return (
            _rustynum.max_f32(self.inner)
            if self.dtype == "float32"
            else _rustynum.max_f64(self.inner)
        )

    def __imul__(self, scalar):
        self.inner.__imul__(scalar)
        return self

    def __add__(self, other):
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

    def __mul__(self, other):
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

    def __sub__(self, other):
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

    def __truediv__(self, other):
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

    def tolist(self):
        return self.inner.tolist()


def dot_f32(a, b):
    # Ensure both inputs are NumArray instances with dtype='float32'
    if (
        isinstance(a, NumArray)
        and isinstance(b, NumArray)
        and a.dtype == "float32"
        and b.dtype == "float32"
    ):
        return a.dot(b)
    else:
        raise TypeError(
            "Both arguments must be NumArray instances with dtype='float32'."
        )


def mean_f32(a):
    # Ensure input is NumArray instance with dtype='float32'
    if isinstance(a, NumArray) and a.dtype == "float32":
        return a.mean()
    else:
        raise TypeError(
            "Both arguments must be NumArray instances with dtype='float32'."
        )


def min_f32(a):
    # Ensure input is NumArray instance with dtype='float32'
    if isinstance(a, NumArray) and a.dtype == "float32":
        return a.min()
    else:
        raise TypeError(
            "Both arguments must be NumArray instances with dtype='float32'."
        )


def max_f32(a):
    # Ensure input is NumArray instance with dtype='float32'
    if isinstance(a, NumArray) and a.dtype == "float32":
        return a.max()
    else:
        raise TypeError(
            "Both arguments must be NumArray instances with dtype='float32'."
        )


def dot_f64(a, b):
    # Ensure both inputs are NumArray instances with dtype='float64'
    if (
        isinstance(a, NumArray)
        and isinstance(b, NumArray)
        and a.dtype == "float64"
        and b.dtype == "float64"
    ):
        return a.dot(b)
    else:
        raise TypeError(
            "Both arguments must be NumArray instances with dtype='float64'."
        )


def mean_f64(a):
    # Ensure input is NumArray instance with dtype='float32'
    if isinstance(a, NumArray) and a.dtype == "float64":
        return a.mean()
    else:
        raise TypeError(
            "Both arguments must be NumArray instances with dtype='float64'."
        )


def min_f64(a):
    # Ensure input is NumArray instance with dtype='float32'
    if isinstance(a, NumArray) and a.dtype == "float64":
        return a.min()
    else:
        raise TypeError(
            "Both arguments must be NumArray instances with dtype='float64'."
        )


def max_f64(a):
    # Ensure input is NumArray instance with dtype='float32'
    if isinstance(a, NumArray) and a.dtype == "float64":
        return a.max()
    else:
        raise TypeError(
            "Both arguments must be NumArray instances with dtype='float64'."
        )

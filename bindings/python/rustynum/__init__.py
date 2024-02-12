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

    def __imul__(self, scalar):
        self.inner.__imul__(scalar)
        return self

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
    # Ensure both inputs are NumArray instances with dtype='float32'
    if isinstance(a, NumArray) and a.dtype == "float32":
        return a.mean()
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
    # Ensure both inputs are NumArray instances with dtype='float64'
    if isinstance(a, NumArray) and a.dtype == "float64":
        return a.mean()
    else:
        raise TypeError(
            "Both arguments must be NumArray instances with dtype='float64'."
        )

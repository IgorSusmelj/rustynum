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

    def dot(self, other):
        if self.dtype != other.dtype:
            raise ValueError("dtype mismatch between arrays")
        return (
            _rustynum.dot_f32(self.inner, other.inner)
            if self.dtype == "float32"
            else _rustynum.dot_f64(self.inner, other.inner)
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

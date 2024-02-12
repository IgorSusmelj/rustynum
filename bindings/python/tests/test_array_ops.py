# bindings/python/tests/test_dot_product.py

import numpy as np
import rustynum as rnp


def test_slicing_f32():
    a = rnp.NumArray([1.0, 2.0, 3.0, 4.0], dtype="float32")
    sliced = a[1:3].tolist()
    assert sliced == [2.0, 3.0], "Slicing failed"


def test_slicing_f64():
    a = rnp.NumArray([1.0, 2.0, 3.0, 4.0], dtype="float64")
    sliced = a[1:3].tolist()
    assert sliced == [2.0, 3.0], "Slicing failed"

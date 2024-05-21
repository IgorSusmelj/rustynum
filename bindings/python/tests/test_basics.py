import numpy as np
import rustynum as rnp


def test_zeros():
    a = rnp.zeros((2, 3), dtype="float32")
    b = np.zeros((2, 3), dtype="float32")
    assert a.tolist() == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], "Zeros failed"
    assert np.allclose(a.tolist(), b, atol=1e-9), "Zeros failed"


def test_ones():
    a = rnp.ones((2, 3), dtype="float32")
    b = np.ones((2, 3), dtype="float32")
    assert a.tolist() == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], "Ones failed"
    assert np.allclose(a.tolist(), b, atol=1e-9), "Ones failed"

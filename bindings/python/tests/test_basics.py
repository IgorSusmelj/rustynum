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


def test_arange():
    a = rnp.arange(0, 10, 2, dtype="float32")
    b = np.arange(0, 10, 2, dtype="float32")
    assert a.tolist() == [0.0, 2.0, 4.0, 6.0, 8.0], "Arange failed"
    assert np.allclose(a.tolist(), b, atol=1e-9), "Arange failed"


def test_linspace():
    a = rnp.linspace(0, 10, 5, dtype="float32")
    b = np.linspace(0, 10, 5, dtype="float32")
    assert a.tolist() == [0.0, 2.5, 5.0, 7.5, 10.0], "Linspace failed"
    assert np.allclose(a.tolist(), b, atol=1e-9), "Linspace failed"

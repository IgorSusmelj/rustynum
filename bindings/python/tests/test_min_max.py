# bindings/python/tests/test_dot_product.py

import numpy as np
import rustynum as rnp


def test_min_f32_small():
    a = [1.0, 2.0, 3.0, 4.0]
    a_py = rnp.NumArray(a, dtype="float32")
    result_rusty_1 = a_py.min()
    result_rusty_2 = rnp.min(a_py)
    result_numpy = np.min(a)

    assert np.isclose(
        result_rusty_1, result_numpy, atol=1e-6
    ), "Min for f32 failed with error"

    assert np.isclose(
        result_rusty_2, result_numpy, atol=1e-6
    ), "Min for f32 failed with error"


def test_max_f32_small():
    a = [1.0, 2.0, 3.0, 4.0]
    a_py = rnp.NumArray(a, dtype="float32")
    result_rusty_1 = a_py.max()
    result_rusty_2 = rnp.max(a_py)
    result_numpy = np.max(a)

    assert np.isclose(
        result_rusty_1, result_numpy, atol=1e-6
    ), "Max for f32 failed with error"

    assert np.isclose(
        result_rusty_2, result_numpy, atol=1e-6
    ), "Max for f32 failed with error"

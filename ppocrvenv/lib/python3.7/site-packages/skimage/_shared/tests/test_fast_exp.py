from ..fast_exp import fast_exp
import numpy as np


def test_fast_exp():

    X = np.linspace(-5, 0, 5000, endpoint=True)

    # Ground truth
    Y = np.exp(X)

    # Approximation at double precision
    _y_f64 = np.array([fast_exp['float64_t'](x) for x in X])

    # Approximation at single precision
    _y_f32 = np.array([fast_exp['float32_t'](x) for x in X.astype('float32')],
                      dtype='float32')

    for _y in [_y_f64, _y_f32]:

        assert np.abs(Y - _y).mean() < 3e-3

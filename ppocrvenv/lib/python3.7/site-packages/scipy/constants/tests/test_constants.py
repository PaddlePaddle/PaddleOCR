from numpy.testing import assert_equal, assert_allclose
import scipy.constants as sc


def test_convert_temperature():
    assert_equal(sc.convert_temperature(32, 'f', 'Celsius'), 0)
    assert_equal(sc.convert_temperature([0, 0], 'celsius', 'Kelvin'),
                 [273.15, 273.15])
    assert_equal(sc.convert_temperature([0, 0], 'kelvin', 'c'),
                 [-273.15, -273.15])
    assert_equal(sc.convert_temperature([32, 32], 'f', 'k'), [273.15, 273.15])
    assert_equal(sc.convert_temperature([273.15, 273.15], 'kelvin', 'F'),
                 [32, 32])
    assert_equal(sc.convert_temperature([0, 0], 'C', 'fahrenheit'), [32, 32])
    assert_allclose(sc.convert_temperature([0, 0], 'c', 'r'), [491.67, 491.67],
                    rtol=0., atol=1e-13)
    assert_allclose(sc.convert_temperature([491.67, 491.67], 'Rankine', 'C'),
                    [0., 0.], rtol=0., atol=1e-13)
    assert_allclose(sc.convert_temperature([491.67, 491.67], 'r', 'F'),
                    [32., 32.], rtol=0., atol=1e-13)
    assert_allclose(sc.convert_temperature([32, 32], 'fahrenheit', 'R'),
                    [491.67, 491.67], rtol=0., atol=1e-13)
    assert_allclose(sc.convert_temperature([273.15, 273.15], 'K', 'R'),
                    [491.67, 491.67], rtol=0., atol=1e-13)
    assert_allclose(sc.convert_temperature([491.67, 0.], 'rankine', 'kelvin'),
                    [273.15, 0.], rtol=0., atol=1e-13)


def test_lambda_to_nu():
    assert_equal(sc.lambda2nu([sc.speed_of_light, 1]), [1, sc.speed_of_light])


def test_nu_to_lambda():
    assert_equal(sc.nu2lambda([sc.speed_of_light, 1]), [1, sc.speed_of_light])


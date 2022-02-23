from pytest import raises as assert_raises
from scipy._lib._pep440 import Version, parse


def test_main_versions():
    assert Version('1.8.0') == Version('1.8.0')
    for ver in ['1.9.0', '2.0.0', '1.8.1']:
        assert Version('1.8.0') < Version(ver)

    for ver in ['1.7.0', '1.7.1', '0.9.9']:
        assert Version('1.8.0') > Version(ver)


def test_version_1_point_10():
    # regression test for gh-2998.
    assert Version('1.9.0') < Version('1.10.0')
    assert Version('1.11.0') < Version('1.11.1')
    assert Version('1.11.0') == Version('1.11.0')
    assert Version('1.99.11') < Version('1.99.12')


def test_alpha_beta_rc():
    assert Version('1.8.0rc1') == Version('1.8.0rc1')
    for ver in ['1.8.0', '1.8.0rc2']:
        assert Version('1.8.0rc1') < Version(ver)

    for ver in ['1.8.0a2', '1.8.0b3', '1.7.2rc4']:
        assert Version('1.8.0rc1') > Version(ver)

    assert Version('1.8.0b1') > Version('1.8.0a2')


def test_dev_version():
    assert Version('1.9.0.dev+Unknown') < Version('1.9.0')
    for ver in ['1.9.0', '1.9.0a1', '1.9.0b2', '1.9.0b2.dev+ffffffff', '1.9.0.dev1']:
        assert Version('1.9.0.dev+f16acvda') < Version(ver)

    assert Version('1.9.0.dev+f16acvda') == Version('1.9.0.dev+f16acvda')


def test_dev_a_b_rc_mixed():
    assert Version('1.9.0a2.dev+f16acvda') == Version('1.9.0a2.dev+f16acvda')
    assert Version('1.9.0a2.dev+6acvda54') < Version('1.9.0a2')


def test_dev0_version():
    assert Version('1.9.0.dev0+Unknown') < Version('1.9.0')
    for ver in ['1.9.0', '1.9.0a1', '1.9.0b2', '1.9.0b2.dev0+ffffffff']:
        assert Version('1.9.0.dev0+f16acvda') < Version(ver)

    assert Version('1.9.0.dev0+f16acvda') == Version('1.9.0.dev0+f16acvda')


def test_dev0_a_b_rc_mixed():
    assert Version('1.9.0a2.dev0+f16acvda') == Version('1.9.0a2.dev0+f16acvda')
    assert Version('1.9.0a2.dev0+6acvda54') < Version('1.9.0a2')


def test_raises():
    for ver in ['1,9.0', '1.7.x']:
        assert_raises(ValueError, Version, ver)

def test_legacy_version():
    # Non-PEP-440 version identifiers always compare less. For NumPy this only
    # occurs on dev builds prior to 1.10.0 which are unsupported anyway.
    assert parse('invalid') < Version('0.0.0')
    assert parse('1.9.0-f16acvda') < Version('1.0.0')

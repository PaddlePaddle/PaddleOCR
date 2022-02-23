from scipy.linalg import norm


def test_norm():
    assert norm([]) == 0.0

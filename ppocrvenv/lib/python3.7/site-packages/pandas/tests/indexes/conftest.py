import pytest


@pytest.fixture(params=[None, False])
def sort(request):
    """
    Valid values for the 'sort' parameter used in the Index
    setops methods (intersection, union, etc.)

    Caution:
        Don't confuse this one with the "sort" fixture used
        for DataFrame.append or concat. That one has
        parameters [True, False].

        We can't combine them as sort=True is not permitted
        in the Index setops methods.
    """
    return request.param


@pytest.fixture(params=["D", "3D", "-3D", "H", "2H", "-2H", "T", "2T", "S", "-3S"])
def freq_sample(request):
    """
    Valid values for 'freq' parameter used to create date_range and
    timedelta_range..
    """
    return request.param

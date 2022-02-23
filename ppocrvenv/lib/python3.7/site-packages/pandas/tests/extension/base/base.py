import pandas._testing as tm


class BaseExtensionTests:
    # classmethod and different signature is needed
    # to make inheritance compliant with mypy
    @classmethod
    def assert_equal(cls, left, right, **kwargs):
        return tm.assert_equal(left, right, **kwargs)

    @classmethod
    def assert_series_equal(cls, left, right, *args, **kwargs):
        return tm.assert_series_equal(left, right, *args, **kwargs)

    @classmethod
    def assert_frame_equal(cls, left, right, *args, **kwargs):
        return tm.assert_frame_equal(left, right, *args, **kwargs)

    @classmethod
    def assert_extension_array_equal(cls, left, right, *args, **kwargs):
        return tm.assert_extension_array_equal(left, right, *args, **kwargs)

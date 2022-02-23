"""The module contains miscellaneous helpers.
It's not considered part of the public ufoLib API.
"""
import warnings
import functools


numberTypes = (int, float)


def deprecated(msg=""):
    """Decorator factory to mark functions as deprecated with given message.

    >>> @deprecated("Enough!")
    ... def some_function():
    ...    "I just print 'hello world'."
    ...    print("hello world")
    >>> some_function()
    hello world
    >>> some_function.__doc__ == "I just print 'hello world'."
    True
    """

    def deprecated_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{func.__name__} function is a deprecated. {msg}",
                category=DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return wrapper

    return deprecated_decorator


# To be mixed with enum.Enum in UFOFormatVersion and GLIFFormatVersion
class _VersionTupleEnumMixin:
    @property
    def major(self):
        return self.value[0]

    @property
    def minor(self):
        return self.value[1]

    @classmethod
    def _missing_(cls, value):
        # allow to initialize a version enum from a single (major) integer
        if isinstance(value, int):
            return cls((value, 0))
        # or from None to obtain the current default version
        if value is None:
            return cls.default()
        return super()._missing_(value)

    def __str__(self):
        return f"{self.major}.{self.minor}"

    @classmethod
    def default(cls):
        # get the latest defined version (i.e. the max of all versions)
        return max(cls.__members__.values())

    @classmethod
    def supported_versions(cls):
        return frozenset(cls.__members__.values())


if __name__ == "__main__":
    import doctest

    doctest.testmod()

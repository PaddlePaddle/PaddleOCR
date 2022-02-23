"""
Exceptions

This module is deprecated and will be removed in joblib 0.16.
"""
# Author: Gael Varoquaux < gael dot varoquaux at normalesup dot org >
# Copyright: 2010, Gael Varoquaux
# License: BSD 3 clause


class JoblibException(Exception):
    """A simple exception with an error message that you can get to."""
    def __init__(self, *args):
        # We need to implement __init__ so that it is picked in the
        # multiple heritance hierarchy in the class created in
        # _mk_exception. Note: in Python 2, if you implement __init__
        # in your exception class you need to set .args correctly,
        # otherwise you can dump an exception instance with pickle but
        # not load it (at load time an empty .args will be passed to
        # the constructor). Also we want to be explicit and not use
        # 'super' here. Using 'super' can cause a sibling class method
        # to be called and we have no control the sibling class method
        # constructor signature in the exception returned by
        # _mk_exception.
        Exception.__init__(self, *args)

    def __repr__(self):
        if hasattr(self, 'args') and len(self.args) > 0:
            message = self.args[0]
        else:
            message = ''

        name = self.__class__.__name__
        return '%s\n%s\n%s\n%s' % (name, 75 * '_', message, 75 * '_')

    __str__ = __repr__


class TransportableException(JoblibException):
    """An exception containing all the info to wrap an original
        exception and recreate it.
    """

    def __init__(self, message, etype):
        # The next line set the .args correctly. This is needed to
        # make the exception loadable with pickle
        JoblibException.__init__(self, message, etype)
        self.message = message
        self.etype = etype

    def unwrap(self, context_message=""):
        report = """\
%s
---------------------------------------------------------------------------
Joblib worker traceback:
---------------------------------------------------------------------------
%s""" % (context_message, self.message)
        # Unwrap the exception to a JoblibException
        exception_type = _mk_exception(self.etype)[0]
        return exception_type(report)


_exception_mapping = dict()


def _mk_exception(exception, name=None):
    if issubclass(exception, JoblibException):
        # No need to wrap recursively JoblibException
        return exception, exception.__name__

    # Create an exception inheriting from both JoblibException
    # and that exception
    if name is None:
        name = exception.__name__
    this_name = 'Joblib%s' % name
    if this_name in _exception_mapping:
        # Avoid creating twice the same exception
        this_exception = _exception_mapping[this_name]
    else:
        if exception is Exception:
            # JoblibException is already a subclass of Exception. No
            # need to use multiple inheritance
            return JoblibException, this_name
        try:
            this_exception = type(
                this_name, (JoblibException, exception), {})
            _exception_mapping[this_name] = this_exception
        except TypeError:
            # This happens if "Cannot create a consistent method
            # resolution order", e.g. because 'exception' is a
            # subclass of JoblibException or 'exception' is not an
            # acceptable base class
            this_exception = JoblibException

    return this_exception, this_name


def _mk_common_exceptions():
    namespace = dict()
    import builtins as _builtin_exceptions
    common_exceptions = filter(
        lambda x: x.endswith('Error'),
        dir(_builtin_exceptions))

    for name in common_exceptions:
        obj = getattr(_builtin_exceptions, name)
        if isinstance(obj, type) and issubclass(obj, BaseException):
            this_obj, this_name = _mk_exception(obj, name=name)
            namespace[this_name] = this_obj
    return namespace


# Updating module locals so that the exceptions pickle right. AFAIK this
# works only at module-creation time
locals().update(_mk_common_exceptions())

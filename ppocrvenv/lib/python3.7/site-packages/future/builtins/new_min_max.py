import itertools

from future import utils
if utils.PY2:
    from __builtin__ import max as _builtin_max, min as _builtin_min
else:
    from builtins import max as _builtin_max, min as _builtin_min

_SENTINEL = object()


def newmin(*args, **kwargs):
    return new_min_max(_builtin_min, *args, **kwargs)


def newmax(*args, **kwargs):
    return new_min_max(_builtin_max, *args, **kwargs)


def new_min_max(_builtin_func, *args, **kwargs):
    """
    To support the argument "default" introduced in python 3.4 for min and max
    :param _builtin_func: builtin min or builtin max
    :param args:
    :param kwargs:
    :return: returns the min or max based on the arguments passed
    """

    for key, _ in kwargs.items():
        if key not in set(['key', 'default']):
            raise TypeError('Illegal argument %s', key)

    if len(args) == 0:
        raise TypeError

    if len(args) != 1 and kwargs.get('default', _SENTINEL) is not _SENTINEL:
        raise TypeError

    if len(args) == 1:
        iterator = iter(args[0])
        try:
            first = next(iterator)
        except StopIteration:
            if kwargs.get('default', _SENTINEL) is not _SENTINEL:
                return kwargs.get('default')
            else:
                raise ValueError('{}() arg is an empty sequence'.format(_builtin_func.__name__))
        else:
            iterator = itertools.chain([first], iterator)
        if kwargs.get('key') is not None:
            return _builtin_func(iterator, key=kwargs.get('key'))
        else:
            return _builtin_func(iterator)

    if len(args) > 1:
        if kwargs.get('key') is not None:
            return _builtin_func(args, key=kwargs.get('key'))
        else:
            return _builtin_func(args)

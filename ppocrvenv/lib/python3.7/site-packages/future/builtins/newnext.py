'''
This module provides a newnext() function in Python 2 that mimics the
behaviour of ``next()`` in Python 3, falling back to Python 2's behaviour for
compatibility if this fails.

``newnext(iterator)`` calls the iterator's ``__next__()`` method if it exists. If this
doesn't exist, it falls back to calling a ``next()`` method.

For example:

    >>> class Odds(object):
    ...     def __init__(self, start=1):
    ...         self.value = start - 2
    ...     def __next__(self):                 # note the Py3 interface
    ...         self.value += 2
    ...         return self.value
    ...     def __iter__(self):
    ...         return self
    ...
    >>> iterator = Odds()
    >>> next(iterator)
    1
    >>> next(iterator)
    3

If you are defining your own custom iterator class as above, it is preferable
to explicitly decorate the class with the @implements_iterator decorator from
``future.utils`` as follows:

    >>> @implements_iterator
    ... class Odds(object):
    ...     # etc
    ...     pass

This next() function is primarily for consuming iterators defined in Python 3
code elsewhere that we would like to run on Python 2 or 3.
'''

_builtin_next = next

_SENTINEL = object()

def newnext(iterator, default=_SENTINEL):
    """
    next(iterator[, default])

    Return the next item from the iterator. If default is given and the iterator
    is exhausted, it is returned instead of raising StopIteration.
    """

    # args = []
    # if default is not _SENTINEL:
    #     args.append(default)
    try:
        try:
            return iterator.__next__()
        except AttributeError:
            try:
                return iterator.next()
            except AttributeError:
                raise TypeError("'{0}' object is not an iterator".format(
                                           iterator.__class__.__name__))
    except StopIteration as e:
        if default is _SENTINEL:
            raise e
        else:
            return default


__all__ = ['newnext']

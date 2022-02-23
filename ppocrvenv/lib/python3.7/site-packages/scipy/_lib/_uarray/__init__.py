"""
.. note::
    If you are looking for overrides for NumPy-specific methods, see the
    documentation for :obj:`unumpy`. This page explains how to write
    back-ends and multimethods.

``uarray`` is built around a back-end protocol and overridable multimethods.
It is necessary to define multimethods for back-ends to be able to override them.
See the documentation of :obj:`generate_multimethod` on how to write multimethods.



Let's start with the simplest:

``__ua_domain__`` defines the back-end *domain*. The domain consists of period-
separated string consisting of the modules you extend plus the submodule. For
example, if a submodule ``module2.submodule`` extends ``module1``
(i.e., it exposes dispatchables marked as types available in ``module1``),
then the domain string should be ``"module1.module2.submodule"``.


For the purpose of this demonstration, we'll be creating an object and setting
its attributes directly. However, note that you can use a module or your own type
as a backend as well.

>>> class Backend: pass
>>> be = Backend()
>>> be.__ua_domain__ = "ua_examples"

It might be useful at this point to sidetrack to the documentation of
:obj:`generate_multimethod` to find out how to generate a multimethod
overridable by :obj:`uarray`. Needless to say, writing a backend and
creating multimethods are mostly orthogonal activities, and knowing
one doesn't necessarily require knowledge of the other, although it
is certainly helpful. We expect core API designers/specifiers to write the
multimethods, and implementors to override them. But, as is often the case,
similar people write both.

Without further ado, here's an example multimethod:

>>> import uarray as ua
>>> from uarray import Dispatchable
>>> def override_me(a, b):
...   return Dispatchable(a, int),
>>> def override_replacer(args, kwargs, dispatchables):
...     return (dispatchables[0], args[1]), {}
>>> overridden_me = ua.generate_multimethod(
...     override_me, override_replacer, "ua_examples"
... )

Next comes the part about overriding the multimethod. This requires
the ``__ua_function__`` protocol, and the ``__ua_convert__``
protocol. The ``__ua_function__`` protocol has the signature
``(method, args, kwargs)`` where ``method`` is the passed
multimethod, ``args``/``kwargs`` specify the arguments and ``dispatchables``
is the list of converted dispatchables passed in.

>>> def __ua_function__(method, args, kwargs):
...     return method.__name__, args, kwargs
>>> be.__ua_function__ = __ua_function__

The other protocol of interest is the ``__ua_convert__`` protocol. It has the
signature ``(dispatchables, coerce)``. When ``coerce`` is ``False``, conversion
between the formats should ideally be an ``O(1)`` operation, but it means that
no memory copying should be involved, only views of the existing data.

>>> def __ua_convert__(dispatchables, coerce):
...     for d in dispatchables:
...         if d.type is int:
...             if coerce and d.coercible:
...                 yield str(d.value)
...             else:
...                 yield d.value
>>> be.__ua_convert__ = __ua_convert__

Now that we have defined the backend, the next thing to do is to call the multimethod.

>>> with ua.set_backend(be):
...      overridden_me(1, "2")
('override_me', (1, '2'), {})

Note that the marked type has no effect on the actual type of the passed object.
We can also coerce the type of the input.

>>> with ua.set_backend(be, coerce=True):
...     overridden_me(1, "2")
...     overridden_me(1.0, "2")
('override_me', ('1', '2'), {})
('override_me', ('1.0', '2'), {})

Another feature is that if you remove ``__ua_convert__``, the arguments are not
converted at all and it's up to the backend to handle that.

>>> del be.__ua_convert__
>>> with ua.set_backend(be):
...     overridden_me(1, "2")
('override_me', (1, '2'), {})

You also have the option to return ``NotImplemented``, in which case processing moves on
to the next back-end, which, in this case, doesn't exist. The same applies to
``__ua_convert__``.

>>> be.__ua_function__ = lambda *a, **kw: NotImplemented
>>> with ua.set_backend(be):
...     overridden_me(1, "2")
Traceback (most recent call last):
    ...
uarray.backend.BackendNotImplementedError: ...

The last possibility is if we don't have ``__ua_convert__``, in which case the job is left
up to ``__ua_function__``, but putting things back into arrays after conversion will not be
possible.
"""

from ._backend import *

__version__ = '0.5.1+49.g4c3f1d7.scipy'

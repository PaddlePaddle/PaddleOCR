import sys

from packaging import version as _version


def ensure_python_version(min_version):
    if not isinstance(min_version, tuple):
        min_version = (min_version, )
    if sys.version_info < min_version:
        # since ensure_python_version is in the critical import path,
        # we lazy import it.
        from platform import python_version

        raise ImportError("""

You are running scikit-image on an unsupported version of Python.

Unfortunately, scikit-image 0.15 and above no longer work with your installed
version of Python (%s).  You therefore have two options: either upgrade to
Python %s, or install an older version of scikit-image.

For Python 2.7 or Python 3.4, use

 $ pip install 'scikit-image<0.15'

Please also consider updating `pip` and `setuptools`:

 $ pip install pip setuptools --upgrade

Newer versions of these tools avoid installing packages incompatible
with your version of Python.
""" % (python_version(), '.'.join([str(v) for v in min_version])))


def _check_version(actver, version, cmp_op):
    """
    Check version string of an active module against a required version.

    If dev/prerelease tags result in TypeError for string-number comparison,
    it is assumed that the dependency is satisfied.
    Users on dev branches are responsible for keeping their own packages up to
    date.

    Copyright (C) 2013  The IPython Development Team

    Distributed under the terms of the BSD License.
    """
    try:
        if cmp_op == '>':
            return _version.parse(actver) > _version.parse(version)
        elif cmp_op == '>=':
            return _version.parse(actver) >= _version.parse(version)
        elif cmp_op == '=':
            return _version.parse(actver) == _version.parse(version)
        elif cmp_op == '<':
            return _version.parse(actver) < _version.parse(version)
        else:
            return False
    except TypeError:
        return True


def get_module_version(module_name):
    """Return module version or None if version can't be retrieved."""
    mod = __import__(module_name,
                     fromlist=[module_name.rpartition('.')[-1]])
    return getattr(mod, '__version__', getattr(mod, 'VERSION', None))


def is_installed(name, version=None):
    """Test if *name* is installed.

    Parameters
    ----------
    name : str
        Name of module or "python"
    version : str, optional
        Version string to test against.
        If version is not None, checking version
        (must have an attribute named '__version__' or 'VERSION')
        Version may start with =, >=, > or < to specify the exact requirement

    Returns
    -------
    out : bool
        True if `name` is installed matching the optional version.

    Notes
    -----
    Original Copyright (C) 2009-2011 Pierre Raybaut
    Licensed under the terms of the MIT License.
    """
    if name.lower() == 'python':
        actver = sys.version[:6]
    else:
        try:
            actver = get_module_version(name)
        except ImportError:
            return False
    if version is None:
        return True
    else:
        # since version_requirements is in the critical import path,
        # we lazy import re
        import re

        match = re.search('[0-9]', version)
        assert match is not None, "Invalid version number"
        symb = version[:match.start()]
        if not symb:
            symb = '='
        assert symb in ('>=', '>', '=', '<'),\
            "Invalid version condition '%s'" % symb
        version = version[match.start():]
        return _check_version(actver, version, symb)


def require(name, version=None):
    """Return decorator that forces a requirement for a function or class.

    Parameters
    ----------
    name : str
        Name of module or "python".
    version : str, optional
        Version string to test against.
        If version is not None, checking version
        (must have an attribute named '__version__' or 'VERSION')
        Version may start with =, >=, > or < to specify the exact requirement

    Returns
    -------
    func : function
        A decorator that raises an ImportError if a function is run
        in the absence of the input dependency.
    """
    # since version_requirements is in the critical import path, we lazy import
    # functools
    import functools

    def decorator(obj):
        @functools.wraps(obj)
        def func_wrapped(*args, **kwargs):
            if is_installed(name, version):
                return obj(*args, **kwargs)
            else:
                msg = '"%s" in "%s" requires "%s'
                msg = msg % (obj, obj.__module__, name)
                if not version is None:
                    msg += " %s" % version
                raise ImportError(msg + '"')
        return func_wrapped
    return decorator


def get_module(module_name, version=None):
    """Return a module object of name *module_name* if installed.

    Parameters
    ----------
    module_name : str
        Name of module.
    version : str, optional
        Version string to test against.
        If version is not None, checking version
        (must have an attribute named '__version__' or 'VERSION')
        Version may start with =, >=, > or < to specify the exact requirement

    Returns
    -------
    mod : module or None
        Module if *module_name* is installed matching the optional version
        or None otherwise.
    """
    if not is_installed(module_name, version):
        return None
    return __import__(module_name,
                      fromlist=[module_name.rpartition('.')[-1]])

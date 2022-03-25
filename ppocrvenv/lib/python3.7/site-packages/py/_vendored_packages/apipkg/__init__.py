"""
apipkg: control the exported namespace of a Python package.

see https://pypi.python.org/pypi/apipkg

(c) holger krekel, 2009 - MIT license
"""
import os
import sys
from types import ModuleType

from .version import version as __version__  # NOQA:F401


def _py_abspath(path):
    """
    special version of abspath
    that will leave paths from jython jars alone
    """
    if path.startswith("__pyclasspath__"):

        return path
    else:
        return os.path.abspath(path)


def distribution_version(name):
    """try to get the version of the named distribution,
    returs None on failure"""
    from pkg_resources import get_distribution, DistributionNotFound

    try:
        dist = get_distribution(name)
    except DistributionNotFound:
        pass
    else:
        return dist.version


def initpkg(pkgname, exportdefs, attr=None, eager=False):
    """ initialize given package from the export definitions. """
    attr = attr or {}
    oldmod = sys.modules.get(pkgname)
    d = {}
    f = getattr(oldmod, "__file__", None)
    if f:
        f = _py_abspath(f)
    d["__file__"] = f
    if hasattr(oldmod, "__version__"):
        d["__version__"] = oldmod.__version__
    if hasattr(oldmod, "__loader__"):
        d["__loader__"] = oldmod.__loader__
    if hasattr(oldmod, "__path__"):
        d["__path__"] = [_py_abspath(p) for p in oldmod.__path__]
    if hasattr(oldmod, "__package__"):
        d["__package__"] = oldmod.__package__
    if "__doc__" not in exportdefs and getattr(oldmod, "__doc__", None):
        d["__doc__"] = oldmod.__doc__
    d["__spec__"] = getattr(oldmod, "__spec__", None)
    d.update(attr)
    if hasattr(oldmod, "__dict__"):
        oldmod.__dict__.update(d)
    mod = ApiModule(pkgname, exportdefs, implprefix=pkgname, attr=d)
    sys.modules[pkgname] = mod
    # eagerload in bypthon to avoid their monkeypatching breaking packages
    if "bpython" in sys.modules or eager:
        for module in list(sys.modules.values()):
            if isinstance(module, ApiModule):
                module.__dict__
    return mod


def importobj(modpath, attrname):
    """imports a module, then resolves the attrname on it"""
    module = __import__(modpath, None, None, ["__doc__"])
    if not attrname:
        return module

    retval = module
    names = attrname.split(".")
    for x in names:
        retval = getattr(retval, x)
    return retval


class ApiModule(ModuleType):
    """the magical lazy-loading module standing"""

    def __docget(self):
        try:
            return self.__doc
        except AttributeError:
            if "__doc__" in self.__map__:
                return self.__makeattr("__doc__")

    def __docset(self, value):
        self.__doc = value

    __doc__ = property(__docget, __docset)

    def __init__(self, name, importspec, implprefix=None, attr=None):
        self.__name__ = name
        self.__all__ = [x for x in importspec if x != "__onfirstaccess__"]
        self.__map__ = {}
        self.__implprefix__ = implprefix or name
        if attr:
            for name, val in attr.items():
                # print "setting", self.__name__, name, val
                setattr(self, name, val)
        for name, importspec in importspec.items():
            if isinstance(importspec, dict):
                subname = "{}.{}".format(self.__name__, name)
                apimod = ApiModule(subname, importspec, implprefix)
                sys.modules[subname] = apimod
                setattr(self, name, apimod)
            else:
                parts = importspec.split(":")
                modpath = parts.pop(0)
                attrname = parts and parts[0] or ""
                if modpath[0] == ".":
                    modpath = implprefix + modpath

                if not attrname:
                    subname = "{}.{}".format(self.__name__, name)
                    apimod = AliasModule(subname, modpath)
                    sys.modules[subname] = apimod
                    if "." not in name:
                        setattr(self, name, apimod)
                else:
                    self.__map__[name] = (modpath, attrname)

    def __repr__(self):
        repr_list = []
        if hasattr(self, "__version__"):
            repr_list.append("version=" + repr(self.__version__))
        if hasattr(self, "__file__"):
            repr_list.append("from " + repr(self.__file__))
        if repr_list:
            return "<ApiModule {!r} {}>".format(self.__name__, " ".join(repr_list))
        return "<ApiModule {!r}>".format(self.__name__)

    def __makeattr(self, name):
        """lazily compute value for name or raise AttributeError if unknown."""
        # print "makeattr", self.__name__, name
        target = None
        if "__onfirstaccess__" in self.__map__:
            target = self.__map__.pop("__onfirstaccess__")
            importobj(*target)()
        try:
            modpath, attrname = self.__map__[name]
        except KeyError:
            if target is not None and name != "__onfirstaccess__":
                # retry, onfirstaccess might have set attrs
                return getattr(self, name)
            raise AttributeError(name)
        else:
            result = importobj(modpath, attrname)
            setattr(self, name, result)
            try:
                del self.__map__[name]
            except KeyError:
                pass  # in a recursive-import situation a double-del can happen
            return result

    __getattr__ = __makeattr

    @property
    def __dict__(self):
        # force all the content of the module
        # to be loaded when __dict__ is read
        dictdescr = ModuleType.__dict__["__dict__"]
        dict = dictdescr.__get__(self)
        if dict is not None:
            hasattr(self, "some")
            for name in self.__all__:
                try:
                    self.__makeattr(name)
                except AttributeError:
                    pass
        return dict


def AliasModule(modname, modpath, attrname=None):
    mod = []

    def getmod():
        if not mod:
            x = importobj(modpath, None)
            if attrname is not None:
                x = getattr(x, attrname)
            mod.append(x)
        return mod[0]

    x = modpath + ("." + attrname if attrname else "")
    repr_result = "<AliasModule {!r} for {!r}>".format(modname, x)

    class AliasModule(ModuleType):
        def __repr__(self):
            return repr_result

        def __getattribute__(self, name):
            try:
                return getattr(getmod(), name)
            except ImportError:
                if modpath == "pytest" and attrname is None:
                    # hack for pylibs py.test
                    return None
                else:
                    raise

        def __setattr__(self, name, value):
            setattr(getmod(), name, value)

        def __delattr__(self, name):
            delattr(getmod(), name)

    return AliasModule(str(modname))

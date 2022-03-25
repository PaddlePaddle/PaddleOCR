import py
import sys

builtin_repr = repr

reprlib = py.builtin._tryimport('repr', 'reprlib')

class SafeRepr(reprlib.Repr):
    """ subclass of repr.Repr that limits the resulting size of repr()
        and includes information on exceptions raised during the call.
    """
    def repr(self, x):
        return self._callhelper(reprlib.Repr.repr, self, x)

    def repr_unicode(self, x, level):
        # Strictly speaking wrong on narrow builds
        def repr(u):
            if "'" not in u:
                return py.builtin._totext("'%s'") % u
            elif '"' not in u:
                return py.builtin._totext('"%s"') % u
            else:
                return py.builtin._totext("'%s'") % u.replace("'", r"\'")
        s = repr(x[:self.maxstring])
        if len(s) > self.maxstring:
            i = max(0, (self.maxstring-3)//2)
            j = max(0, self.maxstring-3-i)
            s = repr(x[:i] + x[len(x)-j:])
            s = s[:i] + '...' + s[len(s)-j:]
        return s

    def repr_instance(self, x, level):
        return self._callhelper(builtin_repr, x)

    def _callhelper(self, call, x, *args):
        try:
            # Try the vanilla repr and make sure that the result is a string
            s = call(x, *args)
        except py.builtin._sysex:
            raise
        except:
            cls, e, tb = sys.exc_info()
            exc_name = getattr(cls, '__name__', 'unknown')
            try:
                exc_info = str(e)
            except py.builtin._sysex:
                raise
            except:
                exc_info = 'unknown'
            return '<[%s("%s") raised in repr()] %s object at 0x%x>' % (
                exc_name, exc_info, x.__class__.__name__, id(x))
        else:
            if len(s) > self.maxsize:
                i = max(0, (self.maxsize-3)//2)
                j = max(0, self.maxsize-3-i)
                s = s[:i] + '...' + s[len(s)-j:]
            return s

def saferepr(obj, maxsize=240):
    """ return a size-limited safe repr-string for the given object.
    Failing __repr__ functions of user instances will be represented
    with a short exception info and 'saferepr' generally takes
    care to never raise exceptions itself.  This function is a wrapper
    around the Repr/reprlib functionality of the standard 2.6 lib.
    """
    # review exception handling
    srepr = SafeRepr()
    srepr.maxstring = maxsize
    srepr.maxsize = maxsize
    srepr.maxother = 160
    return srepr.repr(obj)

# copied from python-2.7.3's traceback.py
# CHANGES:
# - some_str is replaced, trying to create unicode strings
#
import types

def format_exception_only(etype, value):
    """Format the exception part of a traceback.

    The arguments are the exception type and value such as given by
    sys.last_type and sys.last_value. The return value is a list of
    strings, each ending in a newline.

    Normally, the list contains a single string; however, for
    SyntaxError exceptions, it contains several lines that (when
    printed) display detailed information about where the syntax
    error occurred.

    The message indicating which exception occurred is always the last
    string in the list.

    """

    # An instance should not have a meaningful value parameter, but
    # sometimes does, particularly for string exceptions, such as
    # >>> raise string1, string2  # deprecated
    #
    # Clear these out first because issubtype(string1, SyntaxError)
    # would throw another exception and mask the original problem.
    if (isinstance(etype, BaseException) or
        isinstance(etype, types.InstanceType) or
        etype is None or type(etype) is str):
        return [_format_final_exc_line(etype, value)]

    stype = etype.__name__

    if not issubclass(etype, SyntaxError):
        return [_format_final_exc_line(stype, value)]

    # It was a syntax error; show exactly where the problem was found.
    lines = []
    try:
        msg, (filename, lineno, offset, badline) = value.args
    except Exception:
        pass
    else:
        filename = filename or "<string>"
        lines.append('  File "%s", line %d\n' % (filename, lineno))
        if badline is not None:
            lines.append('    %s\n' % badline.strip())
            if offset is not None:
                caretspace = badline.rstrip('\n')[:offset].lstrip()
                # non-space whitespace (likes tabs) must be kept for alignment
                caretspace = ((c.isspace() and c or ' ') for c in caretspace)
                # only three spaces to account for offset1 == pos 0
                lines.append('   %s^\n' % ''.join(caretspace))
        value = msg

    lines.append(_format_final_exc_line(stype, value))
    return lines

def _format_final_exc_line(etype, value):
    """Return a list of a single line -- normal case for format_exception_only"""
    valuestr = _some_str(value)
    if value is None or not valuestr:
        line = "%s\n" % etype
    else:
        line = "%s: %s\n" % (etype, valuestr)
    return line

def _some_str(value):
    try:
        return unicode(value)
    except Exception:
        try:
            return str(value)
        except Exception:
            pass
    return '<unprintable %s object>' % type(value).__name__

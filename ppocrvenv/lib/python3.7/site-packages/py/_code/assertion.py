import sys
import py

BuiltinAssertionError = py.builtin.builtins.AssertionError

_reprcompare = None # if set, will be called by assert reinterp for comparison ops

def _format_explanation(explanation):
    """This formats an explanation

    Normally all embedded newlines are escaped, however there are
    three exceptions: \n{, \n} and \n~.  The first two are intended
    cover nested explanations, see function and attribute explanations
    for examples (.visit_Call(), visit_Attribute()).  The last one is
    for when one explanation needs to span multiple lines, e.g. when
    displaying diffs.
    """
    raw_lines = (explanation or '').split('\n')
    # escape newlines not followed by {, } and ~
    lines = [raw_lines[0]]
    for l in raw_lines[1:]:
        if l.startswith('{') or l.startswith('}') or l.startswith('~'):
            lines.append(l)
        else:
            lines[-1] += '\\n' + l

    result = lines[:1]
    stack = [0]
    stackcnt = [0]
    for line in lines[1:]:
        if line.startswith('{'):
            if stackcnt[-1]:
                s = 'and   '
            else:
                s = 'where '
            stack.append(len(result))
            stackcnt[-1] += 1
            stackcnt.append(0)
            result.append(' +' + '  '*(len(stack)-1) + s + line[1:])
        elif line.startswith('}'):
            assert line.startswith('}')
            stack.pop()
            stackcnt.pop()
            result[stack[-1]] += line[1:]
        else:
            assert line.startswith('~')
            result.append('  '*len(stack) + line[1:])
    assert len(stack) == 1
    return '\n'.join(result)


class AssertionError(BuiltinAssertionError):
    def __init__(self, *args):
        BuiltinAssertionError.__init__(self, *args)
        if args:
            try:
                self.msg = str(args[0])
            except py.builtin._sysex:
                raise
            except:
                self.msg = "<[broken __repr__] %s at %0xd>" %(
                    args[0].__class__, id(args[0]))
        else:
            f = py.code.Frame(sys._getframe(1))
            try:
                source = f.code.fullsource
                if source is not None:
                    try:
                        source = source.getstatement(f.lineno, assertion=True)
                    except IndexError:
                        source = None
                    else:
                        source = str(source.deindent()).strip()
            except py.error.ENOENT:
                source = None
                # this can also occur during reinterpretation, when the
                # co_filename is set to "<run>".
            if source:
                self.msg = reinterpret(source, f, should_fail=True)
            else:
                self.msg = "<could not determine information>"
            if not self.args:
                self.args = (self.msg,)

if sys.version_info > (3, 0):
    AssertionError.__module__ = "builtins"
    reinterpret_old = "old reinterpretation not available for py3"
else:
    from py._code._assertionold import interpret as reinterpret_old
from py._code._assertionnew import interpret as reinterpret

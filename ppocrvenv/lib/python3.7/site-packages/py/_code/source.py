from __future__ import generators

from bisect import bisect_right
import sys
import inspect, tokenize
import py
from types import ModuleType
cpy_compile = compile

try:
    import _ast
    from _ast import PyCF_ONLY_AST as _AST_FLAG
except ImportError:
    _AST_FLAG = 0
    _ast = None


class Source(object):
    """ a immutable object holding a source code fragment,
        possibly deindenting it.
    """
    _compilecounter = 0
    def __init__(self, *parts, **kwargs):
        self.lines = lines = []
        de = kwargs.get('deindent', True)
        rstrip = kwargs.get('rstrip', True)
        for part in parts:
            if not part:
                partlines = []
            if isinstance(part, Source):
                partlines = part.lines
            elif isinstance(part, (tuple, list)):
                partlines = [x.rstrip("\n") for x in part]
            elif isinstance(part, py.builtin._basestring):
                partlines = part.split('\n')
                if rstrip:
                    while partlines:
                        if partlines[-1].strip():
                            break
                        partlines.pop()
            else:
                partlines = getsource(part, deindent=de).lines
            if de:
                partlines = deindent(partlines)
            lines.extend(partlines)

    def __eq__(self, other):
        try:
            return self.lines == other.lines
        except AttributeError:
            if isinstance(other, str):
                return str(self) == other
            return False

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.lines[key]
        else:
            if key.step not in (None, 1):
                raise IndexError("cannot slice a Source with a step")
            return self.__getslice__(key.start, key.stop)

    def __len__(self):
        return len(self.lines)

    def __getslice__(self, start, end):
        newsource = Source()
        newsource.lines = self.lines[start:end]
        return newsource

    def strip(self):
        """ return new source object with trailing
            and leading blank lines removed.
        """
        start, end = 0, len(self)
        while start < end and not self.lines[start].strip():
            start += 1
        while end > start and not self.lines[end-1].strip():
            end -= 1
        source = Source()
        source.lines[:] = self.lines[start:end]
        return source

    def putaround(self, before='', after='', indent=' ' * 4):
        """ return a copy of the source object with
            'before' and 'after' wrapped around it.
        """
        before = Source(before)
        after = Source(after)
        newsource = Source()
        lines = [ (indent + line) for line in self.lines]
        newsource.lines = before.lines + lines +  after.lines
        return newsource

    def indent(self, indent=' ' * 4):
        """ return a copy of the source object with
            all lines indented by the given indent-string.
        """
        newsource = Source()
        newsource.lines = [(indent+line) for line in self.lines]
        return newsource

    def getstatement(self, lineno, assertion=False):
        """ return Source statement which contains the
            given linenumber (counted from 0).
        """
        start, end = self.getstatementrange(lineno, assertion)
        return self[start:end]

    def getstatementrange(self, lineno, assertion=False):
        """ return (start, end) tuple which spans the minimal
            statement region which containing the given lineno.
        """
        if not (0 <= lineno < len(self)):
            raise IndexError("lineno out of range")
        ast, start, end = getstatementrange_ast(lineno, self)
        return start, end

    def deindent(self, offset=None):
        """ return a new source object deindented by offset.
            If offset is None then guess an indentation offset from
            the first non-blank line.  Subsequent lines which have a
            lower indentation offset will be copied verbatim as
            they are assumed to be part of multilines.
        """
        # XXX maybe use the tokenizer to properly handle multiline
        #     strings etc.pp?
        newsource = Source()
        newsource.lines[:] = deindent(self.lines, offset)
        return newsource

    def isparseable(self, deindent=True):
        """ return True if source is parseable, heuristically
            deindenting it by default.
        """
        try:
            import parser
        except ImportError:
            syntax_checker = lambda x: compile(x, 'asd', 'exec')
        else:
            syntax_checker = parser.suite

        if deindent:
            source = str(self.deindent())
        else:
            source = str(self)
        try:
            #compile(source+'\n', "x", "exec")
            syntax_checker(source+'\n')
        except KeyboardInterrupt:
            raise
        except Exception:
            return False
        else:
            return True

    def __str__(self):
        return "\n".join(self.lines)

    def compile(self, filename=None, mode='exec',
                flag=generators.compiler_flag,
                dont_inherit=0, _genframe=None):
        """ return compiled code object. if filename is None
            invent an artificial filename which displays
            the source/line position of the caller frame.
        """
        if not filename or py.path.local(filename).check(file=0):
            if _genframe is None:
                _genframe = sys._getframe(1) # the caller
            fn,lineno = _genframe.f_code.co_filename, _genframe.f_lineno
            base = "<%d-codegen " % self._compilecounter
            self.__class__._compilecounter += 1
            if not filename:
                filename = base + '%s:%d>' % (fn, lineno)
            else:
                filename = base + '%r %s:%d>' % (filename, fn, lineno)
        source = "\n".join(self.lines) + '\n'
        try:
            co = cpy_compile(source, filename, mode, flag)
        except SyntaxError:
            ex = sys.exc_info()[1]
            # re-represent syntax errors from parsing python strings
            msglines = self.lines[:ex.lineno]
            if ex.offset:
                msglines.append(" "*ex.offset + '^')
            msglines.append("(code was compiled probably from here: %s)" % filename)
            newex = SyntaxError('\n'.join(msglines))
            newex.offset = ex.offset
            newex.lineno = ex.lineno
            newex.text = ex.text
            raise newex
        else:
            if flag & _AST_FLAG:
                return co
            lines = [(x + "\n") for x in self.lines]
            import linecache
            linecache.cache[filename] = (1, None, lines, filename)
            return co

#
# public API shortcut functions
#

def compile_(source, filename=None, mode='exec', flags=
            generators.compiler_flag, dont_inherit=0):
    """ compile the given source to a raw code object,
        and maintain an internal cache which allows later
        retrieval of the source code for the code object
        and any recursively created code objects.
    """
    if _ast is not None and isinstance(source, _ast.AST):
        # XXX should Source support having AST?
        return cpy_compile(source, filename, mode, flags, dont_inherit)
    _genframe = sys._getframe(1) # the caller
    s = Source(source)
    co = s.compile(filename, mode, flags, _genframe=_genframe)
    return co


def getfslineno(obj):
    """ Return source location (path, lineno) for the given object.
    If the source cannot be determined return ("", -1)
    """
    try:
        code = py.code.Code(obj)
    except TypeError:
        try:
            fn = (inspect.getsourcefile(obj) or
                  inspect.getfile(obj))
        except TypeError:
            return "", -1

        fspath = fn and py.path.local(fn) or None
        lineno = -1
        if fspath:
            try:
                _, lineno = findsource(obj)
            except IOError:
                pass
    else:
        fspath = code.path
        lineno = code.firstlineno
    assert isinstance(lineno, int)
    return fspath, lineno

#
# helper functions
#

def findsource(obj):
    try:
        sourcelines, lineno = inspect.findsource(obj)
    except py.builtin._sysex:
        raise
    except:
        return None, -1
    source = Source()
    source.lines = [line.rstrip() for line in sourcelines]
    return source, lineno

def getsource(obj, **kwargs):
    obj = py.code.getrawcode(obj)
    try:
        strsrc = inspect.getsource(obj)
    except IndentationError:
        strsrc = "\"Buggy python version consider upgrading, cannot get source\""
    assert isinstance(strsrc, str)
    return Source(strsrc, **kwargs)

def deindent(lines, offset=None):
    if offset is None:
        for line in lines:
            line = line.expandtabs()
            s = line.lstrip()
            if s:
                offset = len(line)-len(s)
                break
        else:
            offset = 0
    if offset == 0:
        return list(lines)
    newlines = []
    def readline_generator(lines):
        for line in lines:
            yield line + '\n'
        while True:
            yield ''

    it = readline_generator(lines)

    try:
        for _, _, (sline, _), (eline, _), _ in tokenize.generate_tokens(lambda: next(it)):
            if sline > len(lines):
                break # End of input reached
            if sline > len(newlines):
                line = lines[sline - 1].expandtabs()
                if line.lstrip() and line[:offset].isspace():
                    line = line[offset:] # Deindent
                newlines.append(line)

            for i in range(sline, eline):
                # Don't deindent continuing lines of
                # multiline tokens (i.e. multiline strings)
                newlines.append(lines[i])
    except (IndentationError, tokenize.TokenError):
        pass
    # Add any lines we didn't see. E.g. if an exception was raised.
    newlines.extend(lines[len(newlines):])
    return newlines


def get_statement_startend2(lineno, node):
    import ast
    # flatten all statements and except handlers into one lineno-list
    # AST's line numbers start indexing at 1
    l = []
    for x in ast.walk(node):
        if isinstance(x, _ast.stmt) or isinstance(x, _ast.ExceptHandler):
            l.append(x.lineno - 1)
            for name in "finalbody", "orelse":
                val = getattr(x, name, None)
                if val:
                    # treat the finally/orelse part as its own statement
                    l.append(val[0].lineno - 1 - 1)
    l.sort()
    insert_index = bisect_right(l, lineno)
    start = l[insert_index - 1]
    if insert_index >= len(l):
        end = None
    else:
        end = l[insert_index]
    return start, end


def getstatementrange_ast(lineno, source, assertion=False, astnode=None):
    if astnode is None:
        content = str(source)
        try:
            astnode = compile(content, "source", "exec", 1024)  # 1024 for AST
        except ValueError:
            start, end = getstatementrange_old(lineno, source, assertion)
            return None, start, end
    start, end = get_statement_startend2(lineno, astnode)
    # we need to correct the end:
    # - ast-parsing strips comments
    # - there might be empty lines
    # - we might have lesser indented code blocks at the end
    if end is None:
        end = len(source.lines)

    if end > start + 1:
        # make sure we don't span differently indented code blocks
        # by using the BlockFinder helper used which inspect.getsource() uses itself
        block_finder = inspect.BlockFinder()
        # if we start with an indented line, put blockfinder to "started" mode
        block_finder.started = source.lines[start][0].isspace()
        it = ((x + "\n") for x in source.lines[start:end])
        try:
            for tok in tokenize.generate_tokens(lambda: next(it)):
                block_finder.tokeneater(*tok)
        except (inspect.EndOfBlock, IndentationError):
            end = block_finder.last + start
        except Exception:
            pass

    # the end might still point to a comment or empty line, correct it
    while end:
        line = source.lines[end - 1].lstrip()
        if line.startswith("#") or not line:
            end -= 1
        else:
            break
    return astnode, start, end


def getstatementrange_old(lineno, source, assertion=False):
    """ return (start, end) tuple which spans the minimal
        statement region which containing the given lineno.
        raise an IndexError if no such statementrange can be found.
    """
    # XXX this logic is only used on python2.4 and below
    # 1. find the start of the statement
    from codeop import compile_command
    for start in range(lineno, -1, -1):
        if assertion:
            line = source.lines[start]
            # the following lines are not fully tested, change with care
            if 'super' in line and 'self' in line and '__init__' in line:
                raise IndexError("likely a subclass")
            if "assert" not in line and "raise" not in line:
                continue
        trylines = source.lines[start:lineno+1]
        # quick hack to prepare parsing an indented line with
        # compile_command() (which errors on "return" outside defs)
        trylines.insert(0, 'def xxx():')
        trysource = '\n '.join(trylines)
        #              ^ space here
        try:
            compile_command(trysource)
        except (SyntaxError, OverflowError, ValueError):
            continue

        # 2. find the end of the statement
        for end in range(lineno+1, len(source)+1):
            trysource = source[start:end]
            if trysource.isparseable():
                return start, end
    raise SyntaxError("no valid source range around line %d " % (lineno,))



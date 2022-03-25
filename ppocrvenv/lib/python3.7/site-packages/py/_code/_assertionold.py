import py
import sys, inspect
from compiler import parse, ast, pycodegen
from py._code.assertion import BuiltinAssertionError, _format_explanation
import types

passthroughex = py.builtin._sysex

class Failure:
    def __init__(self, node):
        self.exc, self.value, self.tb = sys.exc_info()
        self.node = node

class View(object):
    """View base class.

    If C is a subclass of View, then C(x) creates a proxy object around
    the object x.  The actual class of the proxy is not C in general,
    but a *subclass* of C determined by the rules below.  To avoid confusion
    we call view class the class of the proxy (a subclass of C, so of View)
    and object class the class of x.

    Attributes and methods not found in the proxy are automatically read on x.
    Other operations like setting attributes are performed on the proxy, as
    determined by its view class.  The object x is available from the proxy
    as its __obj__ attribute.

    The view class selection is determined by the __view__ tuples and the
    optional __viewkey__ method.  By default, the selected view class is the
    most specific subclass of C whose __view__ mentions the class of x.
    If no such subclass is found, the search proceeds with the parent
    object classes.  For example, C(True) will first look for a subclass
    of C with __view__ = (..., bool, ...) and only if it doesn't find any
    look for one with __view__ = (..., int, ...), and then ..., object,...
    If everything fails the class C itself is considered to be the default.

    Alternatively, the view class selection can be driven by another aspect
    of the object x, instead of the class of x, by overriding __viewkey__.
    See last example at the end of this module.
    """

    _viewcache = {}
    __view__ = ()

    def __new__(rootclass, obj, *args, **kwds):
        self = object.__new__(rootclass)
        self.__obj__ = obj
        self.__rootclass__ = rootclass
        key = self.__viewkey__()
        try:
            self.__class__ = self._viewcache[key]
        except KeyError:
            self.__class__ = self._selectsubclass(key)
        return self

    def __getattr__(self, attr):
        # attributes not found in the normal hierarchy rooted on View
        # are looked up in the object's real class
        return getattr(self.__obj__, attr)

    def __viewkey__(self):
        return self.__obj__.__class__

    def __matchkey__(self, key, subclasses):
        if inspect.isclass(key):
            keys = inspect.getmro(key)
        else:
            keys = [key]
        for key in keys:
            result = [C for C in subclasses if key in C.__view__]
            if result:
                return result
        return []

    def _selectsubclass(self, key):
        subclasses = list(enumsubclasses(self.__rootclass__))
        for C in subclasses:
            if not isinstance(C.__view__, tuple):
                C.__view__ = (C.__view__,)
        choices = self.__matchkey__(key, subclasses)
        if not choices:
            return self.__rootclass__
        elif len(choices) == 1:
            return choices[0]
        else:
            # combine the multiple choices
            return type('?', tuple(choices), {})

    def __repr__(self):
        return '%s(%r)' % (self.__rootclass__.__name__, self.__obj__)


def enumsubclasses(cls):
    for subcls in cls.__subclasses__():
        for subsubclass in enumsubclasses(subcls):
            yield subsubclass
    yield cls


class Interpretable(View):
    """A parse tree node with a few extra methods."""
    explanation = None

    def is_builtin(self, frame):
        return False

    def eval(self, frame):
        # fall-back for unknown expression nodes
        try:
            expr = ast.Expression(self.__obj__)
            expr.filename = '<eval>'
            self.__obj__.filename = '<eval>'
            co = pycodegen.ExpressionCodeGenerator(expr).getCode()
            result = frame.eval(co)
        except passthroughex:
            raise
        except:
            raise Failure(self)
        self.result = result
        self.explanation = self.explanation or frame.repr(self.result)

    def run(self, frame):
        # fall-back for unknown statement nodes
        try:
            expr = ast.Module(None, ast.Stmt([self.__obj__]))
            expr.filename = '<run>'
            co = pycodegen.ModuleCodeGenerator(expr).getCode()
            frame.exec_(co)
        except passthroughex:
            raise
        except:
            raise Failure(self)

    def nice_explanation(self):
        return _format_explanation(self.explanation)


class Name(Interpretable):
    __view__ = ast.Name

    def is_local(self, frame):
        source = '%r in locals() is not globals()' % self.name
        try:
            return frame.is_true(frame.eval(source))
        except passthroughex:
            raise
        except:
            return False

    def is_global(self, frame):
        source = '%r in globals()' % self.name
        try:
            return frame.is_true(frame.eval(source))
        except passthroughex:
            raise
        except:
            return False

    def is_builtin(self, frame):
        source = '%r not in locals() and %r not in globals()' % (
            self.name, self.name)
        try:
            return frame.is_true(frame.eval(source))
        except passthroughex:
            raise
        except:
            return False

    def eval(self, frame):
        super(Name, self).eval(frame)
        if not self.is_local(frame):
            self.explanation = self.name

class Compare(Interpretable):
    __view__ = ast.Compare

    def eval(self, frame):
        expr = Interpretable(self.expr)
        expr.eval(frame)
        for operation, expr2 in self.ops:
            if hasattr(self, 'result'):
                # shortcutting in chained expressions
                if not frame.is_true(self.result):
                    break
            expr2 = Interpretable(expr2)
            expr2.eval(frame)
            self.explanation = "%s %s %s" % (
                expr.explanation, operation, expr2.explanation)
            source = "__exprinfo_left %s __exprinfo_right" % operation
            try:
                self.result = frame.eval(source,
                                         __exprinfo_left=expr.result,
                                         __exprinfo_right=expr2.result)
            except passthroughex:
                raise
            except:
                raise Failure(self)
            expr = expr2

class And(Interpretable):
    __view__ = ast.And

    def eval(self, frame):
        explanations = []
        for expr in self.nodes:
            expr = Interpretable(expr)
            expr.eval(frame)
            explanations.append(expr.explanation)
            self.result = expr.result
            if not frame.is_true(expr.result):
                break
        self.explanation = '(' + ' and '.join(explanations) + ')'

class Or(Interpretable):
    __view__ = ast.Or

    def eval(self, frame):
        explanations = []
        for expr in self.nodes:
            expr = Interpretable(expr)
            expr.eval(frame)
            explanations.append(expr.explanation)
            self.result = expr.result
            if frame.is_true(expr.result):
                break
        self.explanation = '(' + ' or '.join(explanations) + ')'


# == Unary operations ==
keepalive = []
for astclass, astpattern in {
    ast.Not    : 'not __exprinfo_expr',
    ast.Invert : '(~__exprinfo_expr)',
    }.items():

    class UnaryArith(Interpretable):
        __view__ = astclass

        def eval(self, frame, astpattern=astpattern):
            expr = Interpretable(self.expr)
            expr.eval(frame)
            self.explanation = astpattern.replace('__exprinfo_expr',
                                                  expr.explanation)
            try:
                self.result = frame.eval(astpattern,
                                         __exprinfo_expr=expr.result)
            except passthroughex:
                raise
            except:
                raise Failure(self)

    keepalive.append(UnaryArith)

# == Binary operations ==
for astclass, astpattern in {
    ast.Add    : '(__exprinfo_left + __exprinfo_right)',
    ast.Sub    : '(__exprinfo_left - __exprinfo_right)',
    ast.Mul    : '(__exprinfo_left * __exprinfo_right)',
    ast.Div    : '(__exprinfo_left / __exprinfo_right)',
    ast.Mod    : '(__exprinfo_left % __exprinfo_right)',
    ast.Power  : '(__exprinfo_left ** __exprinfo_right)',
    }.items():

    class BinaryArith(Interpretable):
        __view__ = astclass

        def eval(self, frame, astpattern=astpattern):
            left = Interpretable(self.left)
            left.eval(frame)
            right = Interpretable(self.right)
            right.eval(frame)
            self.explanation = (astpattern
                                .replace('__exprinfo_left',  left .explanation)
                                .replace('__exprinfo_right', right.explanation))
            try:
                self.result = frame.eval(astpattern,
                                         __exprinfo_left=left.result,
                                         __exprinfo_right=right.result)
            except passthroughex:
                raise
            except:
                raise Failure(self)

    keepalive.append(BinaryArith)


class CallFunc(Interpretable):
    __view__ = ast.CallFunc

    def is_bool(self, frame):
        source = 'isinstance(__exprinfo_value, bool)'
        try:
            return frame.is_true(frame.eval(source,
                                            __exprinfo_value=self.result))
        except passthroughex:
            raise
        except:
            return False

    def eval(self, frame):
        node = Interpretable(self.node)
        node.eval(frame)
        explanations = []
        vars = {'__exprinfo_fn': node.result}
        source = '__exprinfo_fn('
        for a in self.args:
            if isinstance(a, ast.Keyword):
                keyword = a.name
                a = a.expr
            else:
                keyword = None
            a = Interpretable(a)
            a.eval(frame)
            argname = '__exprinfo_%d' % len(vars)
            vars[argname] = a.result
            if keyword is None:
                source += argname + ','
                explanations.append(a.explanation)
            else:
                source += '%s=%s,' % (keyword, argname)
                explanations.append('%s=%s' % (keyword, a.explanation))
        if self.star_args:
            star_args = Interpretable(self.star_args)
            star_args.eval(frame)
            argname = '__exprinfo_star'
            vars[argname] = star_args.result
            source += '*' + argname + ','
            explanations.append('*' + star_args.explanation)
        if self.dstar_args:
            dstar_args = Interpretable(self.dstar_args)
            dstar_args.eval(frame)
            argname = '__exprinfo_kwds'
            vars[argname] = dstar_args.result
            source += '**' + argname + ','
            explanations.append('**' + dstar_args.explanation)
        self.explanation = "%s(%s)" % (
            node.explanation, ', '.join(explanations))
        if source.endswith(','):
            source = source[:-1]
        source += ')'
        try:
            self.result = frame.eval(source, **vars)
        except passthroughex:
            raise
        except:
            raise Failure(self)
        if not node.is_builtin(frame) or not self.is_bool(frame):
            r = frame.repr(self.result)
            self.explanation = '%s\n{%s = %s\n}' % (r, r, self.explanation)

class Getattr(Interpretable):
    __view__ = ast.Getattr

    def eval(self, frame):
        expr = Interpretable(self.expr)
        expr.eval(frame)
        source = '__exprinfo_expr.%s' % self.attrname
        try:
            self.result = frame.eval(source, __exprinfo_expr=expr.result)
        except passthroughex:
            raise
        except:
            raise Failure(self)
        self.explanation = '%s.%s' % (expr.explanation, self.attrname)
        # if the attribute comes from the instance, its value is interesting
        source = ('hasattr(__exprinfo_expr, "__dict__") and '
                  '%r in __exprinfo_expr.__dict__' % self.attrname)
        try:
            from_instance = frame.is_true(
                frame.eval(source, __exprinfo_expr=expr.result))
        except passthroughex:
            raise
        except:
            from_instance = True
        if from_instance:
            r = frame.repr(self.result)
            self.explanation = '%s\n{%s = %s\n}' % (r, r, self.explanation)

# == Re-interpretation of full statements ==

class Assert(Interpretable):
    __view__ = ast.Assert

    def run(self, frame):
        test = Interpretable(self.test)
        test.eval(frame)
        # simplify 'assert False where False = ...'
        if (test.explanation.startswith('False\n{False = ') and
            test.explanation.endswith('\n}')):
            test.explanation = test.explanation[15:-2]
        # print the result as  'assert <explanation>'
        self.result = test.result
        self.explanation = 'assert ' + test.explanation
        if not frame.is_true(test.result):
            try:
                raise BuiltinAssertionError
            except passthroughex:
                raise
            except:
                raise Failure(self)

class Assign(Interpretable):
    __view__ = ast.Assign

    def run(self, frame):
        expr = Interpretable(self.expr)
        expr.eval(frame)
        self.result = expr.result
        self.explanation = '... = ' + expr.explanation
        # fall-back-run the rest of the assignment
        ass = ast.Assign(self.nodes, ast.Name('__exprinfo_expr'))
        mod = ast.Module(None, ast.Stmt([ass]))
        mod.filename = '<run>'
        co = pycodegen.ModuleCodeGenerator(mod).getCode()
        try:
            frame.exec_(co, __exprinfo_expr=expr.result)
        except passthroughex:
            raise
        except:
            raise Failure(self)

class Discard(Interpretable):
    __view__ = ast.Discard

    def run(self, frame):
        expr = Interpretable(self.expr)
        expr.eval(frame)
        self.result = expr.result
        self.explanation = expr.explanation

class Stmt(Interpretable):
    __view__ = ast.Stmt

    def run(self, frame):
        for stmt in self.nodes:
            stmt = Interpretable(stmt)
            stmt.run(frame)


def report_failure(e):
    explanation = e.node.nice_explanation()
    if explanation:
        explanation = ", in: " + explanation
    else:
        explanation = ""
    sys.stdout.write("%s: %s%s\n" % (e.exc.__name__, e.value, explanation))

def check(s, frame=None):
    if frame is None:
        frame = sys._getframe(1)
        frame = py.code.Frame(frame)
    expr = parse(s, 'eval')
    assert isinstance(expr, ast.Expression)
    node = Interpretable(expr.node)
    try:
        node.eval(frame)
    except passthroughex:
        raise
    except Failure:
        e = sys.exc_info()[1]
        report_failure(e)
    else:
        if not frame.is_true(node.result):
            sys.stderr.write("assertion failed: %s\n" % node.nice_explanation())


###########################################################
# API / Entry points
# #########################################################

def interpret(source, frame, should_fail=False):
    module = Interpretable(parse(source, 'exec').node)
    #print "got module", module
    if isinstance(frame, types.FrameType):
        frame = py.code.Frame(frame)
    try:
        module.run(frame)
    except Failure:
        e = sys.exc_info()[1]
        return getfailure(e)
    except passthroughex:
        raise
    except:
        import traceback
        traceback.print_exc()
    if should_fail:
        return ("(assertion failed, but when it was re-run for "
                "printing intermediate values, it did not fail.  Suggestions: "
                "compute assert expression before the assert or use --nomagic)")
    else:
        return None

def getmsg(excinfo):
    if isinstance(excinfo, tuple):
        excinfo = py.code.ExceptionInfo(excinfo)
    #frame, line = gettbline(tb)
    #frame = py.code.Frame(frame)
    #return interpret(line, frame)

    tb = excinfo.traceback[-1]
    source = str(tb.statement).strip()
    x = interpret(source, tb.frame, should_fail=True)
    if not isinstance(x, str):
        raise TypeError("interpret returned non-string %r" % (x,))
    return x

def getfailure(e):
    explanation = e.node.nice_explanation()
    if str(e.value):
        lines = explanation.split('\n')
        lines[0] += "  << %s" % (e.value,)
        explanation = '\n'.join(lines)
    text = "%s: %s" % (e.exc.__name__, explanation)
    if text.startswith('AssertionError: assert '):
        text = text[16:]
    return text

def run(s, frame=None):
    if frame is None:
        frame = sys._getframe(1)
        frame = py.code.Frame(frame)
    module = Interpretable(parse(s, 'exec').node)
    try:
        module.run(frame)
    except Failure:
        e = sys.exc_info()[1]
        report_failure(e)


if __name__ == '__main__':
    # example:
    def f():
        return 5
    def g():
        return 3
    def h(x):
        return 'never'
    check("f() * g() == 5")
    check("not f()")
    check("not (f() and g() or 0)")
    check("f() == g()")
    i = 4
    check("i == f()")
    check("len(f()) == 0")
    check("isinstance(2+3+4, float)")

    run("x = i")
    check("x == 5")

    run("assert not f(), 'oops'")
    run("a, b, c = 1, 2")
    run("a, b, c = f()")

    check("max([f(),g()]) == 4")
    check("'hello'[g()] == 'h'")
    run("'guk%d' % h(f())")

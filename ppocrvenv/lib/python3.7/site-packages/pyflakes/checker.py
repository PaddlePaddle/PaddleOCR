"""
Main module.

Implement the central Checker class.
Also, it models the Bindings and Scopes.
"""
import __future__
import ast
import bisect
import collections
import contextlib
import doctest
import functools
import os
import re
import string
import sys
import tokenize

from pyflakes import messages

PY2 = sys.version_info < (3, 0)
PY35_PLUS = sys.version_info >= (3, 5)    # Python 3.5 and above
PY36_PLUS = sys.version_info >= (3, 6)    # Python 3.6 and above
PY38_PLUS = sys.version_info >= (3, 8)
try:
    sys.pypy_version_info
    PYPY = True
except AttributeError:
    PYPY = False

builtin_vars = dir(__import__('__builtin__' if PY2 else 'builtins'))

parse_format_string = string.Formatter().parse

if PY2:
    tokenize_tokenize = tokenize.generate_tokens
else:
    tokenize_tokenize = tokenize.tokenize

if PY2:
    def getNodeType(node_class):
        # workaround str.upper() which is locale-dependent
        return str(unicode(node_class.__name__).upper())

    def get_raise_argument(node):
        return node.type

else:
    def getNodeType(node_class):
        return node_class.__name__.upper()

    def get_raise_argument(node):
        return node.exc

    # Silence `pyflakes` from reporting `undefined name 'unicode'` in Python 3.
    unicode = str

# Python >= 3.3 uses ast.Try instead of (ast.TryExcept + ast.TryFinally)
if PY2:
    def getAlternatives(n):
        if isinstance(n, (ast.If, ast.TryFinally)):
            return [n.body]
        if isinstance(n, ast.TryExcept):
            return [n.body + n.orelse] + [[hdl] for hdl in n.handlers]
else:
    def getAlternatives(n):
        if isinstance(n, ast.If):
            return [n.body]
        if isinstance(n, ast.Try):
            return [n.body + n.orelse] + [[hdl] for hdl in n.handlers]

if PY35_PLUS:
    FOR_TYPES = (ast.For, ast.AsyncFor)
    LOOP_TYPES = (ast.While, ast.For, ast.AsyncFor)
    FUNCTION_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef)
else:
    FOR_TYPES = (ast.For,)
    LOOP_TYPES = (ast.While, ast.For)
    FUNCTION_TYPES = (ast.FunctionDef,)

if PY36_PLUS:
    ANNASSIGN_TYPES = (ast.AnnAssign,)
else:
    ANNASSIGN_TYPES = ()

if PY38_PLUS:
    def _is_singleton(node):  # type: (ast.AST) -> bool
        return (
            isinstance(node, ast.Constant) and
            isinstance(node.value, (bool, type(Ellipsis), type(None)))
        )
elif not PY2:
    def _is_singleton(node):  # type: (ast.AST) -> bool
        return isinstance(node, (ast.NameConstant, ast.Ellipsis))
else:
    def _is_singleton(node):  # type: (ast.AST) -> bool
        return (
            isinstance(node, ast.Name) and
            node.id in {'True', 'False', 'Ellipsis', 'None'}
        )


def _is_tuple_constant(node):  # type: (ast.AST) -> bool
    return (
        isinstance(node, ast.Tuple) and
        all(_is_constant(elt) for elt in node.elts)
    )


if PY38_PLUS:
    def _is_constant(node):
        return isinstance(node, ast.Constant) or _is_tuple_constant(node)
else:
    _const_tps = (ast.Str, ast.Num)
    if not PY2:
        _const_tps += (ast.Bytes,)

    def _is_constant(node):
        return (
            isinstance(node, _const_tps) or
            _is_singleton(node) or
            _is_tuple_constant(node)
        )


def _is_const_non_singleton(node):  # type: (ast.AST) -> bool
    return _is_constant(node) and not _is_singleton(node)


def _is_name_or_attr(node, name):  # type: (ast.Ast, str) -> bool
    return (
        (isinstance(node, ast.Name) and node.id == name) or
        (isinstance(node, ast.Attribute) and node.attr == name)
    )


# https://github.com/python/typed_ast/blob/1.4.0/ast27/Parser/tokenizer.c#L102-L104
TYPE_COMMENT_RE = re.compile(r'^#\s*type:\s*')
# https://github.com/python/typed_ast/blob/1.4.0/ast27/Parser/tokenizer.c#L1408-L1413
ASCII_NON_ALNUM = ''.join([chr(i) for i in range(128) if not chr(i).isalnum()])
TYPE_IGNORE_RE = re.compile(
    TYPE_COMMENT_RE.pattern + r'ignore([{}]|$)'.format(ASCII_NON_ALNUM))
# https://github.com/python/typed_ast/blob/1.4.0/ast27/Grammar/Grammar#L147
TYPE_FUNC_RE = re.compile(r'^(\(.*?\))\s*->\s*(.*)$')


MAPPING_KEY_RE = re.compile(r'\(([^()]*)\)')
CONVERSION_FLAG_RE = re.compile('[#0+ -]*')
WIDTH_RE = re.compile(r'(?:\*|\d*)')
PRECISION_RE = re.compile(r'(?:\.(?:\*|\d*))?')
LENGTH_RE = re.compile('[hlL]?')
# https://docs.python.org/3/library/stdtypes.html#old-string-formatting
VALID_CONVERSIONS = frozenset('diouxXeEfFgGcrsa%')


def _must_match(regex, string, pos):
    # type: (Pattern[str], str, int) -> Match[str]
    match = regex.match(string, pos)
    assert match is not None
    return match


def parse_percent_format(s):  # type: (str) -> Tuple[PercentFormat, ...]
    """Parses the string component of a `'...' % ...` format call

    Copied from https://github.com/asottile/pyupgrade at v1.20.1
    """

    def _parse_inner():
        # type: () -> Generator[PercentFormat, None, None]
        string_start = 0
        string_end = 0
        in_fmt = False

        i = 0
        while i < len(s):
            if not in_fmt:
                try:
                    i = s.index('%', i)
                except ValueError:  # no more % fields!
                    yield s[string_start:], None
                    return
                else:
                    string_end = i
                    i += 1
                    in_fmt = True
            else:
                key_match = MAPPING_KEY_RE.match(s, i)
                if key_match:
                    key = key_match.group(1)  # type: Optional[str]
                    i = key_match.end()
                else:
                    key = None

                conversion_flag_match = _must_match(CONVERSION_FLAG_RE, s, i)
                conversion_flag = conversion_flag_match.group() or None
                i = conversion_flag_match.end()

                width_match = _must_match(WIDTH_RE, s, i)
                width = width_match.group() or None
                i = width_match.end()

                precision_match = _must_match(PRECISION_RE, s, i)
                precision = precision_match.group() or None
                i = precision_match.end()

                # length modifier is ignored
                i = _must_match(LENGTH_RE, s, i).end()

                try:
                    conversion = s[i]
                except IndexError:
                    raise ValueError('end-of-string while parsing format')
                i += 1

                fmt = (key, conversion_flag, width, precision, conversion)
                yield s[string_start:string_end], fmt

                in_fmt = False
                string_start = i

        if in_fmt:
            raise ValueError('end-of-string while parsing format')

    return tuple(_parse_inner())


class _FieldsOrder(dict):
    """Fix order of AST node fields."""

    def _get_fields(self, node_class):
        # handle iter before target, and generators before element
        fields = node_class._fields
        if 'iter' in fields:
            key_first = 'iter'.find
        elif 'generators' in fields:
            key_first = 'generators'.find
        else:
            key_first = 'value'.find
        return tuple(sorted(fields, key=key_first, reverse=True))

    def __missing__(self, node_class):
        self[node_class] = fields = self._get_fields(node_class)
        return fields


def counter(items):
    """
    Simplest required implementation of collections.Counter. Required as 2.6
    does not have Counter in collections.
    """
    results = {}
    for item in items:
        results[item] = results.get(item, 0) + 1
    return results


def iter_child_nodes(node, omit=None, _fields_order=_FieldsOrder()):
    """
    Yield all direct child nodes of *node*, that is, all fields that
    are nodes and all items of fields that are lists of nodes.

    :param node:          AST node to be iterated upon
    :param omit:          String or tuple of strings denoting the
                          attributes of the node to be omitted from
                          further parsing
    :param _fields_order: Order of AST node fields
    """
    for name in _fields_order[node.__class__]:
        if omit and name in omit:
            continue
        field = getattr(node, name, None)
        if isinstance(field, ast.AST):
            yield field
        elif isinstance(field, list):
            for item in field:
                if isinstance(item, ast.AST):
                    yield item


def convert_to_value(item):
    if isinstance(item, ast.Str):
        return item.s
    elif hasattr(ast, 'Bytes') and isinstance(item, ast.Bytes):
        return item.s
    elif isinstance(item, ast.Tuple):
        return tuple(convert_to_value(i) for i in item.elts)
    elif isinstance(item, ast.Num):
        return item.n
    elif isinstance(item, ast.Name):
        result = VariableKey(item=item)
        constants_lookup = {
            'True': True,
            'False': False,
            'None': None,
        }
        return constants_lookup.get(
            result.name,
            result,
        )
    elif (not PY2) and isinstance(item, ast.NameConstant):
        # None, True, False are nameconstants in python3, but names in 2
        return item.value
    else:
        return UnhandledKeyType()


def is_notimplemented_name_node(node):
    return isinstance(node, ast.Name) and getNodeName(node) == 'NotImplemented'


class Binding(object):
    """
    Represents the binding of a value to a name.

    The checker uses this to keep track of which names have been bound and
    which names have not. See L{Assignment} for a special type of binding that
    is checked with stricter rules.

    @ivar used: pair of (L{Scope}, node) indicating the scope and
                the node that this binding was last used.
    """

    def __init__(self, name, source):
        self.name = name
        self.source = source
        self.used = False

    def __str__(self):
        return self.name

    def __repr__(self):
        return '<%s object %r from line %r at 0x%x>' % (self.__class__.__name__,
                                                        self.name,
                                                        self.source.lineno,
                                                        id(self))

    def redefines(self, other):
        return isinstance(other, Definition) and self.name == other.name


class Definition(Binding):
    """
    A binding that defines a function or a class.
    """


class Builtin(Definition):
    """A definition created for all Python builtins."""

    def __init__(self, name):
        super(Builtin, self).__init__(name, None)

    def __repr__(self):
        return '<%s object %r at 0x%x>' % (self.__class__.__name__,
                                           self.name,
                                           id(self))


class UnhandledKeyType(object):
    """
    A dictionary key of a type that we cannot or do not check for duplicates.
    """


class VariableKey(object):
    """
    A dictionary key which is a variable.

    @ivar item: The variable AST object.
    """
    def __init__(self, item):
        self.name = item.id

    def __eq__(self, compare):
        return (
            compare.__class__ == self.__class__ and
            compare.name == self.name
        )

    def __hash__(self):
        return hash(self.name)


class Importation(Definition):
    """
    A binding created by an import statement.

    @ivar fullName: The complete name given to the import statement,
        possibly including multiple dotted components.
    @type fullName: C{str}
    """

    def __init__(self, name, source, full_name=None):
        self.fullName = full_name or name
        self.redefined = []
        super(Importation, self).__init__(name, source)

    def redefines(self, other):
        if isinstance(other, SubmoduleImportation):
            # See note in SubmoduleImportation about RedefinedWhileUnused
            return self.fullName == other.fullName
        return isinstance(other, Definition) and self.name == other.name

    def _has_alias(self):
        """Return whether importation needs an as clause."""
        return not self.fullName.split('.')[-1] == self.name

    @property
    def source_statement(self):
        """Generate a source statement equivalent to the import."""
        if self._has_alias():
            return 'import %s as %s' % (self.fullName, self.name)
        else:
            return 'import %s' % self.fullName

    def __str__(self):
        """Return import full name with alias."""
        if self._has_alias():
            return self.fullName + ' as ' + self.name
        else:
            return self.fullName


class SubmoduleImportation(Importation):
    """
    A binding created by a submodule import statement.

    A submodule import is a special case where the root module is implicitly
    imported, without an 'as' clause, and the submodule is also imported.
    Python does not restrict which attributes of the root module may be used.

    This class is only used when the submodule import is without an 'as' clause.

    pyflakes handles this case by registering the root module name in the scope,
    allowing any attribute of the root module to be accessed.

    RedefinedWhileUnused is suppressed in `redefines` unless the submodule
    name is also the same, to avoid false positives.
    """

    def __init__(self, name, source):
        # A dot should only appear in the name when it is a submodule import
        assert '.' in name and (not source or isinstance(source, ast.Import))
        package_name = name.split('.')[0]
        super(SubmoduleImportation, self).__init__(package_name, source)
        self.fullName = name

    def redefines(self, other):
        if isinstance(other, Importation):
            return self.fullName == other.fullName
        return super(SubmoduleImportation, self).redefines(other)

    def __str__(self):
        return self.fullName

    @property
    def source_statement(self):
        return 'import ' + self.fullName


class ImportationFrom(Importation):

    def __init__(self, name, source, module, real_name=None):
        self.module = module
        self.real_name = real_name or name

        if module.endswith('.'):
            full_name = module + self.real_name
        else:
            full_name = module + '.' + self.real_name

        super(ImportationFrom, self).__init__(name, source, full_name)

    def __str__(self):
        """Return import full name with alias."""
        if self.real_name != self.name:
            return self.fullName + ' as ' + self.name
        else:
            return self.fullName

    @property
    def source_statement(self):
        if self.real_name != self.name:
            return 'from %s import %s as %s' % (self.module,
                                                self.real_name,
                                                self.name)
        else:
            return 'from %s import %s' % (self.module, self.name)


class StarImportation(Importation):
    """A binding created by a 'from x import *' statement."""

    def __init__(self, name, source):
        super(StarImportation, self).__init__('*', source)
        # Each star importation needs a unique name, and
        # may not be the module name otherwise it will be deemed imported
        self.name = name + '.*'
        self.fullName = name

    @property
    def source_statement(self):
        return 'from ' + self.fullName + ' import *'

    def __str__(self):
        # When the module ends with a ., avoid the ambiguous '..*'
        if self.fullName.endswith('.'):
            return self.source_statement
        else:
            return self.name


class FutureImportation(ImportationFrom):
    """
    A binding created by a from `__future__` import statement.

    `__future__` imports are implicitly used.
    """

    def __init__(self, name, source, scope):
        super(FutureImportation, self).__init__(name, source, '__future__')
        self.used = (scope, source)


class Argument(Binding):
    """
    Represents binding a name as an argument.
    """


class Assignment(Binding):
    """
    Represents binding a name with an explicit assignment.

    The checker will raise warnings for any Assignment that isn't used. Also,
    the checker does not consider assignments in tuple/list unpacking to be
    Assignments, rather it treats them as simple Bindings.
    """


class Annotation(Binding):
    """
    Represents binding a name to a type without an associated value.

    As long as this name is not assigned a value in another binding, it is considered
    undefined for most purposes. One notable exception is using the name as a type
    annotation.
    """

    def redefines(self, other):
        """An Annotation doesn't define any name, so it cannot redefine one."""
        return False


class FunctionDefinition(Definition):
    pass


class ClassDefinition(Definition):
    pass


class ExportBinding(Binding):
    """
    A binding created by an C{__all__} assignment.  If the names in the list
    can be determined statically, they will be treated as names for export and
    additional checking applied to them.

    The only recognized C{__all__} assignment via list/tuple concatenation is in the
    following format:

        __all__ = ['a'] + ['b'] + ['c']

    Names which are imported and not otherwise used but appear in the value of
    C{__all__} will not have an unused import warning reported for them.
    """

    def __init__(self, name, source, scope):
        if '__all__' in scope and isinstance(source, ast.AugAssign):
            self.names = list(scope['__all__'].names)
        else:
            self.names = []

        def _add_to_names(container):
            for node in container.elts:
                if isinstance(node, ast.Str):
                    self.names.append(node.s)

        if isinstance(source.value, (ast.List, ast.Tuple)):
            _add_to_names(source.value)
        # If concatenating lists or tuples
        elif isinstance(source.value, ast.BinOp):
            currentValue = source.value
            while isinstance(currentValue.right, (ast.List, ast.Tuple)):
                left = currentValue.left
                right = currentValue.right
                _add_to_names(right)
                # If more lists are being added
                if isinstance(left, ast.BinOp):
                    currentValue = left
                # If just two lists are being added
                elif isinstance(left, (ast.List, ast.Tuple)):
                    _add_to_names(left)
                    # All lists accounted for - done
                    break
                # If not list concatenation
                else:
                    break
        super(ExportBinding, self).__init__(name, source)


class Scope(dict):
    importStarred = False       # set to True when import * is found

    def __repr__(self):
        scope_cls = self.__class__.__name__
        return '<%s at 0x%x %s>' % (scope_cls, id(self), dict.__repr__(self))


class ClassScope(Scope):
    pass


class FunctionScope(Scope):
    """
    I represent a name scope for a function.

    @ivar globals: Names declared 'global' in this function.
    """
    usesLocals = False
    alwaysUsed = {'__tracebackhide__', '__traceback_info__',
                  '__traceback_supplement__'}

    def __init__(self):
        super(FunctionScope, self).__init__()
        # Simplify: manage the special locals as globals
        self.globals = self.alwaysUsed.copy()
        self.returnValue = None     # First non-empty return
        self.isGenerator = False    # Detect a generator

    def unusedAssignments(self):
        """
        Return a generator for the assignments which have not been used.
        """
        for name, binding in self.items():
            if (not binding.used and
                    name != '_' and  # see issue #202
                    name not in self.globals and
                    not self.usesLocals and
                    isinstance(binding, Assignment)):
                yield name, binding


class GeneratorScope(Scope):
    pass


class ModuleScope(Scope):
    """Scope for a module."""
    _futures_allowed = True
    _annotations_future_enabled = False


class DoctestScope(ModuleScope):
    """Scope for a doctest."""


class DummyNode(object):
    """Used in place of an `ast.AST` to set error message positions"""
    def __init__(self, lineno, col_offset):
        self.lineno = lineno
        self.col_offset = col_offset


class DetectClassScopedMagic:
    names = dir()


# Globally defined names which are not attributes of the builtins module, or
# are only present on some platforms.
_MAGIC_GLOBALS = ['__file__', '__builtins__', 'WindowsError']
# module scope annotation will store in `__annotations__`, see also PEP 526.
if PY36_PLUS:
    _MAGIC_GLOBALS.append('__annotations__')


def getNodeName(node):
    # Returns node.id, or node.name, or None
    if hasattr(node, 'id'):     # One of the many nodes with an id
        return node.id
    if hasattr(node, 'name'):   # an ExceptHandler node
        return node.name
    if hasattr(node, 'rest'):   # a MatchMapping node
        return node.rest


TYPING_MODULES = frozenset(('typing', 'typing_extensions'))


def _is_typing_helper(node, is_name_match_fn, scope_stack):
    """
    Internal helper to determine whether or not something is a member of a
    typing module. This is used as part of working out whether we are within a
    type annotation context.

    Note: you probably don't want to use this function directly. Instead see the
    utils below which wrap it (`_is_typing` and `_is_any_typing_member`).
    """

    def _bare_name_is_attr(name):
        for scope in reversed(scope_stack):
            if name in scope:
                return (
                    isinstance(scope[name], ImportationFrom) and
                    scope[name].module in TYPING_MODULES and
                    is_name_match_fn(scope[name].real_name)
                )

        return False

    def _module_scope_is_typing(name):
        for scope in reversed(scope_stack):
            if name in scope:
                return (
                    isinstance(scope[name], Importation) and
                    scope[name].fullName in TYPING_MODULES
                )

        return False

    return (
        (
            isinstance(node, ast.Name) and
            _bare_name_is_attr(node.id)
        ) or (
            isinstance(node, ast.Attribute) and
            isinstance(node.value, ast.Name) and
            _module_scope_is_typing(node.value.id) and
            is_name_match_fn(node.attr)
        )
    )


def _is_typing(node, typing_attr, scope_stack):
    """
    Determine whether `node` represents the member of a typing module specified
    by `typing_attr`.

    This is used as part of working out whether we are within a type annotation
    context.
    """
    return _is_typing_helper(node, lambda x: x == typing_attr, scope_stack)


def _is_any_typing_member(node, scope_stack):
    """
    Determine whether `node` represents any member of a typing module.

    This is used as part of working out whether we are within a type annotation
    context.
    """
    return _is_typing_helper(node, lambda x: True, scope_stack)


def is_typing_overload(value, scope_stack):
    return (
        isinstance(value.source, FUNCTION_TYPES) and
        any(
            _is_typing(dec, 'overload', scope_stack)
            for dec in value.source.decorator_list
        )
    )


class AnnotationState:
    NONE = 0
    STRING = 1
    BARE = 2


def in_annotation(func):
    @functools.wraps(func)
    def in_annotation_func(self, *args, **kwargs):
        with self._enter_annotation():
            return func(self, *args, **kwargs)
    return in_annotation_func


def in_string_annotation(func):
    @functools.wraps(func)
    def in_annotation_func(self, *args, **kwargs):
        with self._enter_annotation(AnnotationState.STRING):
            return func(self, *args, **kwargs)
    return in_annotation_func


def make_tokens(code):
    # PY3: tokenize.tokenize requires readline of bytes
    if not isinstance(code, bytes):
        code = code.encode('UTF-8')
    lines = iter(code.splitlines(True))
    # next(lines, b'') is to prevent an error in pypy3
    return tuple(tokenize_tokenize(lambda: next(lines, b'')))


class _TypeableVisitor(ast.NodeVisitor):
    """Collect the line number and nodes which are deemed typeable by
    PEP 484

    https://www.python.org/dev/peps/pep-0484/#type-comments
    """
    def __init__(self):
        self.typeable_lines = []  # type: List[int]
        self.typeable_nodes = {}  # type: Dict[int, ast.AST]

    def _typeable(self, node):
        # if there is more than one typeable thing on a line last one wins
        self.typeable_lines.append(node.lineno)
        self.typeable_nodes[node.lineno] = node

        self.generic_visit(node)

    visit_Assign = visit_For = visit_FunctionDef = visit_With = _typeable
    visit_AsyncFor = visit_AsyncFunctionDef = visit_AsyncWith = _typeable


def _collect_type_comments(tree, tokens):
    visitor = _TypeableVisitor()
    visitor.visit(tree)

    type_comments = collections.defaultdict(list)
    for tp, text, start, _, _ in tokens:
        if (
                tp != tokenize.COMMENT or  # skip non comments
                not TYPE_COMMENT_RE.match(text) or  # skip non-type comments
                TYPE_IGNORE_RE.match(text)  # skip ignores
        ):
            continue

        # search for the typeable node at or before the line number of the
        # type comment.
        # if the bisection insertion point is before any nodes this is an
        # invalid type comment which is ignored.
        lineno, _ = start
        idx = bisect.bisect_right(visitor.typeable_lines, lineno)
        if idx == 0:
            continue
        node = visitor.typeable_nodes[visitor.typeable_lines[idx - 1]]
        type_comments[node].append((start, text))

    return type_comments


class Checker(object):
    """
    I check the cleanliness and sanity of Python code.

    @ivar _deferredFunctions: Tracking list used by L{deferFunction}.  Elements
        of the list are two-tuples.  The first element is the callable passed
        to L{deferFunction}.  The second element is a copy of the scope stack
        at the time L{deferFunction} was called.

    @ivar _deferredAssignments: Similar to C{_deferredFunctions}, but for
        callables which are deferred assignment checks.
    """

    _ast_node_scope = {
        ast.Module: ModuleScope,
        ast.ClassDef: ClassScope,
        ast.FunctionDef: FunctionScope,
        ast.Lambda: FunctionScope,
        ast.ListComp: GeneratorScope,
        ast.SetComp: GeneratorScope,
        ast.GeneratorExp: GeneratorScope,
        ast.DictComp: GeneratorScope,
    }
    if PY35_PLUS:
        _ast_node_scope[ast.AsyncFunctionDef] = FunctionScope

    nodeDepth = 0
    offset = None
    _in_annotation = AnnotationState.NONE
    _in_deferred = False

    builtIns = set(builtin_vars).union(_MAGIC_GLOBALS)
    _customBuiltIns = os.environ.get('PYFLAKES_BUILTINS')
    if _customBuiltIns:
        builtIns.update(_customBuiltIns.split(','))
    del _customBuiltIns

    # TODO: file_tokens= is required to perform checks on type comments,
    #       eventually make this a required positional argument.  For now it
    #       is defaulted to `()` for api compatibility.
    def __init__(self, tree, filename='(none)', builtins=None,
                 withDoctest='PYFLAKES_DOCTEST' in os.environ, file_tokens=()):
        self._nodeHandlers = {}
        self._deferredFunctions = []
        self._deferredAssignments = []
        self.deadScopes = []
        self.messages = []
        self.filename = filename
        if builtins:
            self.builtIns = self.builtIns.union(builtins)
        self.withDoctest = withDoctest
        try:
            self.scopeStack = [Checker._ast_node_scope[type(tree)]()]
        except KeyError:
            raise RuntimeError('No scope implemented for the node %r' % tree)
        self.exceptHandlers = [()]
        self.root = tree
        self._type_comments = _collect_type_comments(tree, file_tokens)
        for builtin in self.builtIns:
            self.addBinding(None, Builtin(builtin))
        self.handleChildren(tree)
        self._in_deferred = True
        self.runDeferred(self._deferredFunctions)
        # Set _deferredFunctions to None so that deferFunction will fail
        # noisily if called after we've run through the deferred functions.
        self._deferredFunctions = None
        self.runDeferred(self._deferredAssignments)
        # Set _deferredAssignments to None so that deferAssignment will fail
        # noisily if called after we've run through the deferred assignments.
        self._deferredAssignments = None
        del self.scopeStack[1:]
        self.popScope()
        self.checkDeadScopes()

    def deferFunction(self, callable):
        """
        Schedule a function handler to be called just before completion.

        This is used for handling function bodies, which must be deferred
        because code later in the file might modify the global scope. When
        `callable` is called, the scope at the time this is called will be
        restored, however it will contain any new bindings added to it.
        """
        self._deferredFunctions.append((callable, self.scopeStack[:], self.offset))

    def deferAssignment(self, callable):
        """
        Schedule an assignment handler to be called just after deferred
        function handlers.
        """
        self._deferredAssignments.append((callable, self.scopeStack[:], self.offset))

    def runDeferred(self, deferred):
        """
        Run the callables in C{deferred} using their associated scope stack.
        """
        for handler, scope, offset in deferred:
            self.scopeStack = scope
            self.offset = offset
            handler()

    def _in_doctest(self):
        return (len(self.scopeStack) >= 2 and
                isinstance(self.scopeStack[1], DoctestScope))

    @property
    def futuresAllowed(self):
        if not all(isinstance(scope, ModuleScope)
                   for scope in self.scopeStack):
            return False

        return self.scope._futures_allowed

    @futuresAllowed.setter
    def futuresAllowed(self, value):
        assert value is False
        if isinstance(self.scope, ModuleScope):
            self.scope._futures_allowed = False

    @property
    def annotationsFutureEnabled(self):
        scope = self.scopeStack[0]
        if not isinstance(scope, ModuleScope):
            return False
        return scope._annotations_future_enabled

    @annotationsFutureEnabled.setter
    def annotationsFutureEnabled(self, value):
        assert value is True
        assert isinstance(self.scope, ModuleScope)
        self.scope._annotations_future_enabled = True

    @property
    def scope(self):
        return self.scopeStack[-1]

    def popScope(self):
        self.deadScopes.append(self.scopeStack.pop())

    def checkDeadScopes(self):
        """
        Look at scopes which have been fully examined and report names in them
        which were imported but unused.
        """
        for scope in self.deadScopes:
            # imports in classes are public members
            if isinstance(scope, ClassScope):
                continue

            all_binding = scope.get('__all__')
            if all_binding and not isinstance(all_binding, ExportBinding):
                all_binding = None

            if all_binding:
                all_names = set(all_binding.names)
                undefined = [
                    name for name in all_binding.names
                    if name not in scope
                ]
            else:
                all_names = undefined = []

            if undefined:
                if not scope.importStarred and \
                   os.path.basename(self.filename) != '__init__.py':
                    # Look for possible mistakes in the export list
                    for name in undefined:
                        self.report(messages.UndefinedExport,
                                    scope['__all__'].source, name)

                # mark all import '*' as used by the undefined in __all__
                if scope.importStarred:
                    from_list = []
                    for binding in scope.values():
                        if isinstance(binding, StarImportation):
                            binding.used = all_binding
                            from_list.append(binding.fullName)
                    # report * usage, with a list of possible sources
                    from_list = ', '.join(sorted(from_list))
                    for name in undefined:
                        self.report(messages.ImportStarUsage,
                                    scope['__all__'].source, name, from_list)

            # Look for imported names that aren't used.
            for value in scope.values():
                if isinstance(value, Importation):
                    used = value.used or value.name in all_names
                    if not used:
                        messg = messages.UnusedImport
                        self.report(messg, value.source, str(value))
                    for node in value.redefined:
                        if isinstance(self.getParent(node), FOR_TYPES):
                            messg = messages.ImportShadowedByLoopVar
                        elif used:
                            continue
                        else:
                            messg = messages.RedefinedWhileUnused
                        self.report(messg, node, value.name, value.source)

    def pushScope(self, scopeClass=FunctionScope):
        self.scopeStack.append(scopeClass())

    def report(self, messageClass, *args, **kwargs):
        self.messages.append(messageClass(self.filename, *args, **kwargs))

    def getParent(self, node):
        # Lookup the first parent which is not Tuple, List or Starred
        while True:
            node = node._pyflakes_parent
            if not hasattr(node, 'elts') and not hasattr(node, 'ctx'):
                return node

    def getCommonAncestor(self, lnode, rnode, stop):
        if (
                stop in (lnode, rnode) or
                not (
                    hasattr(lnode, '_pyflakes_parent') and
                    hasattr(rnode, '_pyflakes_parent')
                )
        ):
            return None
        if lnode is rnode:
            return lnode

        if (lnode._pyflakes_depth > rnode._pyflakes_depth):
            return self.getCommonAncestor(lnode._pyflakes_parent, rnode, stop)
        if (lnode._pyflakes_depth < rnode._pyflakes_depth):
            return self.getCommonAncestor(lnode, rnode._pyflakes_parent, stop)
        return self.getCommonAncestor(
            lnode._pyflakes_parent,
            rnode._pyflakes_parent,
            stop,
        )

    def descendantOf(self, node, ancestors, stop):
        for a in ancestors:
            if self.getCommonAncestor(node, a, stop):
                return True
        return False

    def _getAncestor(self, node, ancestor_type):
        parent = node
        while True:
            if parent is self.root:
                return None
            parent = self.getParent(parent)
            if isinstance(parent, ancestor_type):
                return parent

    def getScopeNode(self, node):
        return self._getAncestor(node, tuple(Checker._ast_node_scope.keys()))

    def differentForks(self, lnode, rnode):
        """True, if lnode and rnode are located on different forks of IF/TRY"""
        ancestor = self.getCommonAncestor(lnode, rnode, self.root)
        parts = getAlternatives(ancestor)
        if parts:
            for items in parts:
                if self.descendantOf(lnode, items, ancestor) ^ \
                   self.descendantOf(rnode, items, ancestor):
                    return True
        return False

    def addBinding(self, node, value):
        """
        Called when a binding is altered.

        - `node` is the statement responsible for the change
        - `value` is the new value, a Binding instance
        """
        # assert value.source in (node, node._pyflakes_parent):
        for scope in self.scopeStack[::-1]:
            if value.name in scope:
                break
        existing = scope.get(value.name)

        if (existing and not isinstance(existing, Builtin) and
                not self.differentForks(node, existing.source)):

            parent_stmt = self.getParent(value.source)
            if isinstance(existing, Importation) and isinstance(parent_stmt, FOR_TYPES):
                self.report(messages.ImportShadowedByLoopVar,
                            node, value.name, existing.source)

            elif scope is self.scope:
                if (isinstance(parent_stmt, ast.comprehension) and
                        not isinstance(self.getParent(existing.source),
                                       (FOR_TYPES, ast.comprehension))):
                    self.report(messages.RedefinedInListComp,
                                node, value.name, existing.source)
                elif not existing.used and value.redefines(existing):
                    if value.name != '_' or isinstance(existing, Importation):
                        if not is_typing_overload(existing, self.scopeStack):
                            self.report(messages.RedefinedWhileUnused,
                                        node, value.name, existing.source)

            elif isinstance(existing, Importation) and value.redefines(existing):
                existing.redefined.append(node)

        if value.name in self.scope:
            # then assume the rebound name is used as a global or within a loop
            value.used = self.scope[value.name].used

        # don't treat annotations as assignments if there is an existing value
        # in scope
        if value.name not in self.scope or not isinstance(value, Annotation):
            self.scope[value.name] = value

    def _unknown_handler(self, node):
        # this environment variable configures whether to error on unknown
        # ast types.
        #
        # this is silent by default but the error is enabled for the pyflakes
        # testsuite.
        #
        # this allows new syntax to be added to python without *requiring*
        # changes from the pyflakes side.  but will still produce an error
        # in the pyflakes testsuite (so more specific handling can be added if
        # needed).
        if os.environ.get('PYFLAKES_ERROR_UNKNOWN'):
            raise NotImplementedError('Unexpected type: {}'.format(type(node)))
        else:
            self.handleChildren(node)

    def getNodeHandler(self, node_class):
        try:
            return self._nodeHandlers[node_class]
        except KeyError:
            nodeType = getNodeType(node_class)
        self._nodeHandlers[node_class] = handler = getattr(
            self, nodeType, self._unknown_handler,
        )
        return handler

    def handleNodeLoad(self, node):
        name = getNodeName(node)
        if not name:
            return

        in_generators = None
        importStarred = None

        # try enclosing function scopes and global scope
        for scope in self.scopeStack[-1::-1]:
            if isinstance(scope, ClassScope):
                if not PY2 and name == '__class__':
                    return
                elif in_generators is False:
                    # only generators used in a class scope can access the
                    # names of the class. this is skipped during the first
                    # iteration
                    continue

            binding = scope.get(name, None)
            if isinstance(binding, Annotation) and not self._in_postponed_annotation:
                continue

            if name == 'print' and isinstance(binding, Builtin):
                parent = self.getParent(node)
                if (isinstance(parent, ast.BinOp) and
                        isinstance(parent.op, ast.RShift)):
                    self.report(messages.InvalidPrintSyntax, node)

            try:
                scope[name].used = (self.scope, node)

                # if the name of SubImportation is same as
                # alias of other Importation and the alias
                # is used, SubImportation also should be marked as used.
                n = scope[name]
                if isinstance(n, Importation) and n._has_alias():
                    try:
                        scope[n.fullName].used = (self.scope, node)
                    except KeyError:
                        pass
            except KeyError:
                pass
            else:
                return

            importStarred = importStarred or scope.importStarred

            if in_generators is not False:
                in_generators = isinstance(scope, GeneratorScope)

        if importStarred:
            from_list = []

            for scope in self.scopeStack[-1::-1]:
                for binding in scope.values():
                    if isinstance(binding, StarImportation):
                        # mark '*' imports as used for each scope
                        binding.used = (self.scope, node)
                        from_list.append(binding.fullName)

            # report * usage, with a list of possible sources
            from_list = ', '.join(sorted(from_list))
            self.report(messages.ImportStarUsage, node, name, from_list)
            return

        if name == '__path__' and os.path.basename(self.filename) == '__init__.py':
            # the special name __path__ is valid only in packages
            return

        if name in DetectClassScopedMagic.names and isinstance(self.scope, ClassScope):
            return

        # protected with a NameError handler?
        if 'NameError' not in self.exceptHandlers[-1]:
            self.report(messages.UndefinedName, node, name)

    def handleNodeStore(self, node):
        name = getNodeName(node)
        if not name:
            return
        # if the name hasn't already been defined in the current scope
        if isinstance(self.scope, FunctionScope) and name not in self.scope:
            # for each function or module scope above us
            for scope in self.scopeStack[:-1]:
                if not isinstance(scope, (FunctionScope, ModuleScope)):
                    continue
                # if the name was defined in that scope, and the name has
                # been accessed already in the current scope, and hasn't
                # been declared global
                used = name in scope and scope[name].used
                if used and used[0] is self.scope and name not in self.scope.globals:
                    # then it's probably a mistake
                    self.report(messages.UndefinedLocal,
                                scope[name].used[1], name, scope[name].source)
                    break

        parent_stmt = self.getParent(node)
        if isinstance(parent_stmt, ANNASSIGN_TYPES) and parent_stmt.value is None:
            binding = Annotation(name, node)
        elif isinstance(parent_stmt, (FOR_TYPES, ast.comprehension)) or (
                parent_stmt != node._pyflakes_parent and
                not self.isLiteralTupleUnpacking(parent_stmt)):
            binding = Binding(name, node)
        elif name == '__all__' and isinstance(self.scope, ModuleScope):
            binding = ExportBinding(name, node._pyflakes_parent, self.scope)
        elif PY2 and isinstance(getattr(node, 'ctx', None), ast.Param):
            binding = Argument(name, self.getScopeNode(node))
        else:
            binding = Assignment(name, node)
        self.addBinding(node, binding)

    def handleNodeDelete(self, node):

        def on_conditional_branch():
            """
            Return `True` if node is part of a conditional body.
            """
            current = getattr(node, '_pyflakes_parent', None)
            while current:
                if isinstance(current, (ast.If, ast.While, ast.IfExp)):
                    return True
                current = getattr(current, '_pyflakes_parent', None)
            return False

        name = getNodeName(node)
        if not name:
            return

        if on_conditional_branch():
            # We cannot predict if this conditional branch is going to
            # be executed.
            return

        if isinstance(self.scope, FunctionScope) and name in self.scope.globals:
            self.scope.globals.remove(name)
        else:
            try:
                del self.scope[name]
            except KeyError:
                self.report(messages.UndefinedName, node, name)

    @contextlib.contextmanager
    def _enter_annotation(self, ann_type=AnnotationState.BARE):
        orig, self._in_annotation = self._in_annotation, ann_type
        try:
            yield
        finally:
            self._in_annotation = orig

    @property
    def _in_postponed_annotation(self):
        return (
            self._in_annotation == AnnotationState.STRING or
            self.annotationsFutureEnabled
        )

    def _handle_type_comments(self, node):
        for (lineno, col_offset), comment in self._type_comments.get(node, ()):
            comment = comment.split(':', 1)[1].strip()
            func_match = TYPE_FUNC_RE.match(comment)
            if func_match:
                parts = (
                    func_match.group(1).replace('*', ''),
                    func_match.group(2).strip(),
                )
            else:
                parts = (comment,)

            for part in parts:
                if PY2:
                    part = part.replace('...', 'Ellipsis')
                self.deferFunction(functools.partial(
                    self.handleStringAnnotation,
                    part, DummyNode(lineno, col_offset), lineno, col_offset,
                    messages.CommentAnnotationSyntaxError,
                ))

    def handleChildren(self, tree, omit=None):
        self._handle_type_comments(tree)
        for node in iter_child_nodes(tree, omit=omit):
            self.handleNode(node, tree)

    def isLiteralTupleUnpacking(self, node):
        if isinstance(node, ast.Assign):
            for child in node.targets + [node.value]:
                if not hasattr(child, 'elts'):
                    return False
            return True

    def isDocstring(self, node):
        """
        Determine if the given node is a docstring, as long as it is at the
        correct place in the node tree.
        """
        return isinstance(node, ast.Str) or (isinstance(node, ast.Expr) and
                                             isinstance(node.value, ast.Str))

    def getDocstring(self, node):
        if isinstance(node, ast.Expr):
            node = node.value
        if not isinstance(node, ast.Str):
            return (None, None)

        if PYPY or PY38_PLUS:
            doctest_lineno = node.lineno - 1
        else:
            # Computed incorrectly if the docstring has backslash
            doctest_lineno = node.lineno - node.s.count('\n') - 1

        return (node.s, doctest_lineno)

    def handleNode(self, node, parent):
        if node is None:
            return
        if self.offset and getattr(node, 'lineno', None) is not None:
            node.lineno += self.offset[0]
            node.col_offset += self.offset[1]
        if self.futuresAllowed and not (isinstance(node, ast.ImportFrom) or
                                        self.isDocstring(node)):
            self.futuresAllowed = False
        self.nodeDepth += 1
        node._pyflakes_depth = self.nodeDepth
        node._pyflakes_parent = parent
        try:
            handler = self.getNodeHandler(node.__class__)
            handler(node)
        finally:
            self.nodeDepth -= 1

    _getDoctestExamples = doctest.DocTestParser().get_examples

    def handleDoctests(self, node):
        try:
            if hasattr(node, 'docstring'):
                docstring = node.docstring

                # This is just a reasonable guess. In Python 3.7, docstrings no
                # longer have line numbers associated with them. This will be
                # incorrect if there are empty lines between the beginning
                # of the function and the docstring.
                node_lineno = node.lineno
                if hasattr(node, 'args'):
                    node_lineno = max([node_lineno] +
                                      [arg.lineno for arg in node.args.args])
            else:
                (docstring, node_lineno) = self.getDocstring(node.body[0])
            examples = docstring and self._getDoctestExamples(docstring)
        except (ValueError, IndexError):
            # e.g. line 6 of the docstring for <string> has inconsistent
            # leading whitespace: ...
            return
        if not examples:
            return

        # Place doctest in module scope
        saved_stack = self.scopeStack
        self.scopeStack = [self.scopeStack[0]]
        node_offset = self.offset or (0, 0)
        self.pushScope(DoctestScope)
        if '_' not in self.scopeStack[0]:
            self.addBinding(None, Builtin('_'))
        for example in examples:
            try:
                tree = ast.parse(example.source, "<doctest>")
            except SyntaxError:
                e = sys.exc_info()[1]
                if PYPY:
                    e.offset += 1
                position = (node_lineno + example.lineno + e.lineno,
                            example.indent + 4 + (e.offset or 0))
                self.report(messages.DoctestSyntaxError, node, position)
            else:
                self.offset = (node_offset[0] + node_lineno + example.lineno,
                               node_offset[1] + example.indent + 4)
                self.handleChildren(tree)
                self.offset = node_offset
        self.popScope()
        self.scopeStack = saved_stack

    @in_string_annotation
    def handleStringAnnotation(self, s, node, ref_lineno, ref_col_offset, err):
        try:
            tree = ast.parse(s)
        except SyntaxError:
            self.report(err, node, s)
            return

        body = tree.body
        if len(body) != 1 or not isinstance(body[0], ast.Expr):
            self.report(err, node, s)
            return

        parsed_annotation = tree.body[0].value
        for descendant in ast.walk(parsed_annotation):
            if (
                    'lineno' in descendant._attributes and
                    'col_offset' in descendant._attributes
            ):
                descendant.lineno = ref_lineno
                descendant.col_offset = ref_col_offset

        self.handleNode(parsed_annotation, node)

    @in_annotation
    def handleAnnotation(self, annotation, node):
        if isinstance(annotation, ast.Str):
            # Defer handling forward annotation.
            self.deferFunction(functools.partial(
                self.handleStringAnnotation,
                annotation.s,
                node,
                annotation.lineno,
                annotation.col_offset,
                messages.ForwardAnnotationSyntaxError,
            ))
        elif self.annotationsFutureEnabled:
            fn = in_annotation(Checker.handleNode)
            self.deferFunction(lambda: fn(self, annotation, node))
        else:
            self.handleNode(annotation, node)

    def ignore(self, node):
        pass

    # "stmt" type nodes
    DELETE = PRINT = FOR = ASYNCFOR = WHILE = WITH = WITHITEM = \
        ASYNCWITH = ASYNCWITHITEM = TRYFINALLY = EXEC = \
        EXPR = ASSIGN = handleChildren

    PASS = ignore

    # "expr" type nodes
    BOOLOP = UNARYOP = SET = \
        REPR = ATTRIBUTE = \
        STARRED = NAMECONSTANT = NAMEDEXPR = handleChildren

    def SUBSCRIPT(self, node):
        if _is_name_or_attr(node.value, 'Literal'):
            with self._enter_annotation(AnnotationState.NONE):
                self.handleChildren(node)
        elif _is_name_or_attr(node.value, 'Annotated'):
            self.handleNode(node.value, node)

            # py39+
            if isinstance(node.slice, ast.Tuple):
                slice_tuple = node.slice
            # <py39
            elif (
                    isinstance(node.slice, ast.Index) and
                    isinstance(node.slice.value, ast.Tuple)
            ):
                slice_tuple = node.slice.value
            else:
                slice_tuple = None

            # not a multi-arg `Annotated`
            if slice_tuple is None or len(slice_tuple.elts) < 2:
                self.handleNode(node.slice, node)
            else:
                # the first argument is the type
                self.handleNode(slice_tuple.elts[0], node)
                # the rest of the arguments are not
                with self._enter_annotation(AnnotationState.NONE):
                    for arg in slice_tuple.elts[1:]:
                        self.handleNode(arg, node)

            self.handleNode(node.ctx, node)
        else:
            if _is_any_typing_member(node.value, self.scopeStack):
                with self._enter_annotation():
                    self.handleChildren(node)
            else:
                self.handleChildren(node)

    def _handle_string_dot_format(self, node):
        try:
            placeholders = tuple(parse_format_string(node.func.value.s))
        except ValueError as e:
            self.report(messages.StringDotFormatInvalidFormat, node, e)
            return

        class state:  # py2-compatible `nonlocal`
            auto = None
            next_auto = 0

        placeholder_positional = set()
        placeholder_named = set()

        def _add_key(fmtkey):
            """Returns True if there is an error which should early-exit"""
            if fmtkey is None:  # end of string or `{` / `}` escapes
                return False

            # attributes / indices are allowed in `.format(...)`
            fmtkey, _, _ = fmtkey.partition('.')
            fmtkey, _, _ = fmtkey.partition('[')

            try:
                fmtkey = int(fmtkey)
            except ValueError:
                pass
            else:  # fmtkey was an integer
                if state.auto is True:
                    self.report(messages.StringDotFormatMixingAutomatic, node)
                    return True
                else:
                    state.auto = False

            if fmtkey == '':
                if state.auto is False:
                    self.report(messages.StringDotFormatMixingAutomatic, node)
                    return True
                else:
                    state.auto = True

                fmtkey = state.next_auto
                state.next_auto += 1

            if isinstance(fmtkey, int):
                placeholder_positional.add(fmtkey)
            else:
                placeholder_named.add(fmtkey)

            return False

        for _, fmtkey, spec, _ in placeholders:
            if _add_key(fmtkey):
                return

            # spec can also contain format specifiers
            if spec is not None:
                try:
                    spec_placeholders = tuple(parse_format_string(spec))
                except ValueError as e:
                    self.report(messages.StringDotFormatInvalidFormat, node, e)
                    return

                for _, spec_fmtkey, spec_spec, _ in spec_placeholders:
                    # can't recurse again
                    if spec_spec is not None and '{' in spec_spec:
                        self.report(
                            messages.StringDotFormatInvalidFormat,
                            node,
                            'Max string recursion exceeded',
                        )
                        return
                    if _add_key(spec_fmtkey):
                        return

        # bail early if there is *args or **kwargs
        if (
                # python 2.x *args / **kwargs
                getattr(node, 'starargs', None) or
                getattr(node, 'kwargs', None) or
                # python 3.x *args
                any(
                    isinstance(arg, getattr(ast, 'Starred', ()))
                    for arg in node.args
                ) or
                # python 3.x **kwargs
                any(kwd.arg is None for kwd in node.keywords)
        ):
            return

        substitution_positional = set(range(len(node.args)))
        substitution_named = {kwd.arg for kwd in node.keywords}

        extra_positional = substitution_positional - placeholder_positional
        extra_named = substitution_named - placeholder_named

        missing_arguments = (
            (placeholder_positional | placeholder_named) -
            (substitution_positional | substitution_named)
        )

        if extra_positional:
            self.report(
                messages.StringDotFormatExtraPositionalArguments,
                node,
                ', '.join(sorted(str(x) for x in extra_positional)),
            )
        if extra_named:
            self.report(
                messages.StringDotFormatExtraNamedArguments,
                node,
                ', '.join(sorted(extra_named)),
            )
        if missing_arguments:
            self.report(
                messages.StringDotFormatMissingArgument,
                node,
                ', '.join(sorted(str(x) for x in missing_arguments)),
            )

    def CALL(self, node):
        if (
                isinstance(node.func, ast.Attribute) and
                isinstance(node.func.value, ast.Str) and
                node.func.attr == 'format'
        ):
            self._handle_string_dot_format(node)

        omit = []
        annotated = []
        not_annotated = []

        if (
            _is_typing(node.func, 'cast', self.scopeStack) and
            len(node.args) >= 1
        ):
            with self._enter_annotation():
                self.handleNode(node.args[0], node)

        elif _is_typing(node.func, 'TypeVar', self.scopeStack):

            # TypeVar("T", "int", "str")
            omit += ["args"]
            annotated += [arg for arg in node.args[1:]]

            # TypeVar("T", bound="str")
            omit += ["keywords"]
            annotated += [k.value for k in node.keywords if k.arg == "bound"]
            not_annotated += [
                (k, ["value"] if k.arg == "bound" else None)
                for k in node.keywords
            ]

        elif _is_typing(node.func, "TypedDict", self.scopeStack):
            # TypedDict("a", {"a": int})
            if len(node.args) > 1 and isinstance(node.args[1], ast.Dict):
                omit += ["args"]
                annotated += node.args[1].values
                not_annotated += [
                    (arg, ["values"] if i == 1 else None)
                    for i, arg in enumerate(node.args)
                ]

            # TypedDict("a", a=int)
            omit += ["keywords"]
            annotated += [k.value for k in node.keywords]
            not_annotated += [(k, ["value"]) for k in node.keywords]

        elif _is_typing(node.func, "NamedTuple", self.scopeStack):
            # NamedTuple("a", [("a", int)])
            if (
                len(node.args) > 1 and
                isinstance(node.args[1], (ast.Tuple, ast.List)) and
                all(isinstance(x, (ast.Tuple, ast.List)) and
                    len(x.elts) == 2 for x in node.args[1].elts)
            ):
                omit += ["args"]
                annotated += [elt.elts[1] for elt in node.args[1].elts]
                not_annotated += [(elt.elts[0], None) for elt in node.args[1].elts]
                not_annotated += [
                    (arg, ["elts"] if i == 1 else None)
                    for i, arg in enumerate(node.args)
                ]
                not_annotated += [(elt, "elts") for elt in node.args[1].elts]

            # NamedTuple("a", a=int)
            omit += ["keywords"]
            annotated += [k.value for k in node.keywords]
            not_annotated += [(k, ["value"]) for k in node.keywords]

        if omit:
            with self._enter_annotation(AnnotationState.NONE):
                for na_node, na_omit in not_annotated:
                    self.handleChildren(na_node, omit=na_omit)
                self.handleChildren(node, omit=omit)

            with self._enter_annotation():
                for annotated_node in annotated:
                    self.handleNode(annotated_node, node)
        else:
            self.handleChildren(node)

    def _handle_percent_format(self, node):
        try:
            placeholders = parse_percent_format(node.left.s)
        except ValueError:
            self.report(
                messages.PercentFormatInvalidFormat,
                node,
                'incomplete format',
            )
            return

        named = set()
        positional_count = 0
        positional = None
        for _, placeholder in placeholders:
            if placeholder is None:
                continue
            name, _, width, precision, conversion = placeholder

            if conversion == '%':
                continue

            if conversion not in VALID_CONVERSIONS:
                self.report(
                    messages.PercentFormatUnsupportedFormatCharacter,
                    node,
                    conversion,
                )

            if positional is None and conversion:
                positional = name is None

            for part in (width, precision):
                if part is not None and '*' in part:
                    if not positional:
                        self.report(
                            messages.PercentFormatStarRequiresSequence,
                            node,
                        )
                    else:
                        positional_count += 1

            if positional and name is not None:
                self.report(
                    messages.PercentFormatMixedPositionalAndNamed,
                    node,
                )
                return
            elif not positional and name is None:
                self.report(
                    messages.PercentFormatMixedPositionalAndNamed,
                    node,
                )
                return

            if positional:
                positional_count += 1
            else:
                named.add(name)

        if (
                isinstance(node.right, (ast.List, ast.Tuple)) and
                # does not have any *splats (py35+ feature)
                not any(
                    isinstance(elt, getattr(ast, 'Starred', ()))
                    for elt in node.right.elts
                )
        ):
            substitution_count = len(node.right.elts)
            if positional and positional_count != substitution_count:
                self.report(
                    messages.PercentFormatPositionalCountMismatch,
                    node,
                    positional_count,
                    substitution_count,
                )
            elif not positional:
                self.report(messages.PercentFormatExpectedMapping, node)

        if (
                isinstance(node.right, ast.Dict) and
                all(isinstance(k, ast.Str) for k in node.right.keys)
        ):
            if positional and positional_count > 1:
                self.report(messages.PercentFormatExpectedSequence, node)
                return

            substitution_keys = {k.s for k in node.right.keys}
            extra_keys = substitution_keys - named
            missing_keys = named - substitution_keys
            if not positional and extra_keys:
                self.report(
                    messages.PercentFormatExtraNamedArguments,
                    node,
                    ', '.join(sorted(extra_keys)),
                )
            if not positional and missing_keys:
                self.report(
                    messages.PercentFormatMissingArgument,
                    node,
                    ', '.join(sorted(missing_keys)),
                )

    def BINOP(self, node):
        if (
                isinstance(node.op, ast.Mod) and
                isinstance(node.left, ast.Str)
        ):
            self._handle_percent_format(node)
        self.handleChildren(node)

    def STR(self, node):
        if self._in_annotation:
            fn = functools.partial(
                self.handleStringAnnotation,
                node.s,
                node,
                node.lineno,
                node.col_offset,
                messages.ForwardAnnotationSyntaxError,
            )
            if self._in_deferred:
                fn()
            else:
                self.deferFunction(fn)

    if PY38_PLUS:
        def CONSTANT(self, node):
            if isinstance(node.value, str):
                return self.STR(node)
    else:
        NUM = BYTES = ELLIPSIS = CONSTANT = ignore

    # "slice" type nodes
    SLICE = EXTSLICE = INDEX = handleChildren

    # expression contexts are node instances too, though being constants
    LOAD = STORE = DEL = AUGLOAD = AUGSTORE = PARAM = ignore

    # same for operators
    AND = OR = ADD = SUB = MULT = DIV = MOD = POW = LSHIFT = RSHIFT = \
        BITOR = BITXOR = BITAND = FLOORDIV = INVERT = NOT = UADD = USUB = \
        EQ = NOTEQ = LT = LTE = GT = GTE = IS = ISNOT = IN = NOTIN = \
        MATMULT = ignore

    def RAISE(self, node):
        self.handleChildren(node)

        arg = get_raise_argument(node)

        if isinstance(arg, ast.Call):
            if is_notimplemented_name_node(arg.func):
                # Handle "raise NotImplemented(...)"
                self.report(messages.RaiseNotImplemented, node)
        elif is_notimplemented_name_node(arg):
            # Handle "raise NotImplemented"
            self.report(messages.RaiseNotImplemented, node)

    # additional node types
    COMPREHENSION = KEYWORD = FORMATTEDVALUE = handleChildren

    _in_fstring = False

    def JOINEDSTR(self, node):
        if (
                # the conversion / etc. flags are parsed as f-strings without
                # placeholders
                not self._in_fstring and
                not any(isinstance(x, ast.FormattedValue) for x in node.values)
        ):
            self.report(messages.FStringMissingPlaceholders, node)

        self._in_fstring, orig = True, self._in_fstring
        try:
            self.handleChildren(node)
        finally:
            self._in_fstring = orig

    def DICT(self, node):
        # Complain if there are duplicate keys with different values
        # If they have the same value it's not going to cause potentially
        # unexpected behaviour so we'll not complain.
        keys = [
            convert_to_value(key) for key in node.keys
        ]

        key_counts = counter(keys)
        duplicate_keys = [
            key for key, count in key_counts.items()
            if count > 1
        ]

        for key in duplicate_keys:
            key_indices = [i for i, i_key in enumerate(keys) if i_key == key]

            values = counter(
                convert_to_value(node.values[index])
                for index in key_indices
            )
            if any(count == 1 for value, count in values.items()):
                for key_index in key_indices:
                    key_node = node.keys[key_index]
                    if isinstance(key, VariableKey):
                        self.report(messages.MultiValueRepeatedKeyVariable,
                                    key_node,
                                    key.name)
                    else:
                        self.report(
                            messages.MultiValueRepeatedKeyLiteral,
                            key_node,
                            key,
                        )
        self.handleChildren(node)

    def IF(self, node):
        if isinstance(node.test, ast.Tuple) and node.test.elts != []:
            self.report(messages.IfTuple, node)
        self.handleChildren(node)

    IFEXP = IF

    def ASSERT(self, node):
        if isinstance(node.test, ast.Tuple) and node.test.elts != []:
            self.report(messages.AssertTuple, node)
        self.handleChildren(node)

    def GLOBAL(self, node):
        """
        Keep track of globals declarations.
        """
        global_scope_index = 1 if self._in_doctest() else 0
        global_scope = self.scopeStack[global_scope_index]

        # Ignore 'global' statement in global scope.
        if self.scope is not global_scope:

            # One 'global' statement can bind multiple (comma-delimited) names.
            for node_name in node.names:
                node_value = Assignment(node_name, node)

                # Remove UndefinedName messages already reported for this name.
                # TODO: if the global is not used in this scope, it does not
                # become a globally defined name.  See test_unused_global.
                self.messages = [
                    m for m in self.messages if not
                    isinstance(m, messages.UndefinedName) or
                    m.message_args[0] != node_name]

                # Bind name to global scope if it doesn't exist already.
                global_scope.setdefault(node_name, node_value)

                # Bind name to non-global scopes, but as already "used".
                node_value.used = (global_scope, node)
                for scope in self.scopeStack[global_scope_index + 1:]:
                    scope[node_name] = node_value

    NONLOCAL = GLOBAL

    def GENERATOREXP(self, node):
        self.pushScope(GeneratorScope)
        self.handleChildren(node)
        self.popScope()

    LISTCOMP = handleChildren if PY2 else GENERATOREXP

    DICTCOMP = SETCOMP = GENERATOREXP

    def NAME(self, node):
        """
        Handle occurrence of Name (which can be a load/store/delete access.)
        """
        # Locate the name in locals / function / globals scopes.
        if isinstance(node.ctx, ast.Load):
            self.handleNodeLoad(node)
            if (node.id == 'locals' and isinstance(self.scope, FunctionScope) and
                    isinstance(node._pyflakes_parent, ast.Call)):
                # we are doing locals() call in current scope
                self.scope.usesLocals = True
        elif isinstance(node.ctx, ast.Store):
            self.handleNodeStore(node)
        elif PY2 and isinstance(node.ctx, ast.Param):
            self.handleNodeStore(node)
        elif isinstance(node.ctx, ast.Del):
            self.handleNodeDelete(node)
        else:
            # Unknown context
            raise RuntimeError("Got impossible expression context: %r" % (node.ctx,))

    def CONTINUE(self, node):
        # Walk the tree up until we see a loop (OK), a function or class
        # definition (not OK), for 'continue', a finally block (not OK), or
        # the top module scope (not OK)
        n = node
        while hasattr(n, '_pyflakes_parent'):
            n, n_child = n._pyflakes_parent, n
            if isinstance(n, LOOP_TYPES):
                # Doesn't apply unless it's in the loop itself
                if n_child not in n.orelse:
                    return
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
                break
            # Handle Try/TryFinally difference in Python < and >= 3.3
            if hasattr(n, 'finalbody') and isinstance(node, ast.Continue):
                if n_child in n.finalbody and not PY38_PLUS:
                    self.report(messages.ContinueInFinally, node)
                    return
        if isinstance(node, ast.Continue):
            self.report(messages.ContinueOutsideLoop, node)
        else:  # ast.Break
            self.report(messages.BreakOutsideLoop, node)

    BREAK = CONTINUE

    def RETURN(self, node):
        if isinstance(self.scope, (ClassScope, ModuleScope)):
            self.report(messages.ReturnOutsideFunction, node)
            return

        if (
            node.value and
            hasattr(self.scope, 'returnValue') and
            not self.scope.returnValue
        ):
            self.scope.returnValue = node.value
        self.handleNode(node.value, node)

    def YIELD(self, node):
        if isinstance(self.scope, (ClassScope, ModuleScope)):
            self.report(messages.YieldOutsideFunction, node)
            return

        self.scope.isGenerator = True
        self.handleNode(node.value, node)

    AWAIT = YIELDFROM = YIELD

    def FUNCTIONDEF(self, node):
        for deco in node.decorator_list:
            self.handleNode(deco, node)
        self.LAMBDA(node)
        self.addBinding(node, FunctionDefinition(node.name, node))
        # doctest does not process doctest within a doctest,
        # or in nested functions.
        if (self.withDoctest and
                not self._in_doctest() and
                not isinstance(self.scope, FunctionScope)):
            self.deferFunction(lambda: self.handleDoctests(node))

    ASYNCFUNCTIONDEF = FUNCTIONDEF

    def LAMBDA(self, node):
        args = []
        annotations = []

        if PY2:
            def addArgs(arglist):
                for arg in arglist:
                    if isinstance(arg, ast.Tuple):
                        addArgs(arg.elts)
                    else:
                        args.append(arg.id)
            addArgs(node.args.args)
            defaults = node.args.defaults
        else:
            if PY38_PLUS:
                for arg in node.args.posonlyargs:
                    args.append(arg.arg)
                    annotations.append(arg.annotation)
            for arg in node.args.args + node.args.kwonlyargs:
                args.append(arg.arg)
                annotations.append(arg.annotation)
            defaults = node.args.defaults + node.args.kw_defaults

        # Only for Python3 FunctionDefs
        is_py3_func = hasattr(node, 'returns')

        for arg_name in ('vararg', 'kwarg'):
            wildcard = getattr(node.args, arg_name)
            if not wildcard:
                continue
            args.append(wildcard if PY2 else wildcard.arg)
            if is_py3_func:
                if PY2:  # Python 2.7
                    argannotation = arg_name + 'annotation'
                    annotations.append(getattr(node.args, argannotation))
                else:     # Python >= 3.4
                    annotations.append(wildcard.annotation)

        if is_py3_func:
            annotations.append(node.returns)

        if len(set(args)) < len(args):
            for (idx, arg) in enumerate(args):
                if arg in args[:idx]:
                    self.report(messages.DuplicateArgument, node, arg)

        for annotation in annotations:
            self.handleAnnotation(annotation, node)

        for default in defaults:
            self.handleNode(default, node)

        def runFunction():

            self.pushScope()

            self.handleChildren(node, omit=['decorator_list', 'returns'])

            def checkUnusedAssignments():
                """
                Check to see if any assignments have not been used.
                """
                for name, binding in self.scope.unusedAssignments():
                    self.report(messages.UnusedVariable, binding.source, name)
            self.deferAssignment(checkUnusedAssignments)

            if PY2:
                def checkReturnWithArgumentInsideGenerator():
                    """
                    Check to see if there is any return statement with
                    arguments but the function is a generator.
                    """
                    if self.scope.isGenerator and self.scope.returnValue:
                        self.report(messages.ReturnWithArgsInsideGenerator,
                                    self.scope.returnValue)
                self.deferAssignment(checkReturnWithArgumentInsideGenerator)
            self.popScope()

        self.deferFunction(runFunction)

    def ARGUMENTS(self, node):
        self.handleChildren(node, omit=('defaults', 'kw_defaults'))
        if PY2:
            scope_node = self.getScopeNode(node)
            if node.vararg:
                self.addBinding(node, Argument(node.vararg, scope_node))
            if node.kwarg:
                self.addBinding(node, Argument(node.kwarg, scope_node))

    def ARG(self, node):
        self.addBinding(node, Argument(node.arg, self.getScopeNode(node)))

    def CLASSDEF(self, node):
        """
        Check names used in a class definition, including its decorators, base
        classes, and the body of its definition.  Additionally, add its name to
        the current scope.
        """
        for deco in node.decorator_list:
            self.handleNode(deco, node)
        for baseNode in node.bases:
            self.handleNode(baseNode, node)
        if not PY2:
            for keywordNode in node.keywords:
                self.handleNode(keywordNode, node)
        self.pushScope(ClassScope)
        # doctest does not process doctest within a doctest
        # classes within classes are processed.
        if (self.withDoctest and
                not self._in_doctest() and
                not isinstance(self.scope, FunctionScope)):
            self.deferFunction(lambda: self.handleDoctests(node))
        for stmt in node.body:
            self.handleNode(stmt, node)
        self.popScope()
        self.addBinding(node, ClassDefinition(node.name, node))

    def AUGASSIGN(self, node):
        self.handleNodeLoad(node.target)
        self.handleNode(node.value, node)
        self.handleNode(node.target, node)

    def TUPLE(self, node):
        if not PY2 and isinstance(node.ctx, ast.Store):
            # Python 3 advanced tuple unpacking: a, *b, c = d.
            # Only one starred expression is allowed, and no more than 1<<8
            # assignments are allowed before a stared expression. There is
            # also a limit of 1<<24 expressions after the starred expression,
            # which is impossible to test due to memory restrictions, but we
            # add it here anyway
            has_starred = False
            star_loc = -1
            for i, n in enumerate(node.elts):
                if isinstance(n, ast.Starred):
                    if has_starred:
                        self.report(messages.TwoStarredExpressions, node)
                        # The SyntaxError doesn't distinguish two from more
                        # than two.
                        break
                    has_starred = True
                    star_loc = i
            if star_loc >= 1 << 8 or len(node.elts) - star_loc - 1 >= 1 << 24:
                self.report(messages.TooManyExpressionsInStarredAssignment, node)
        self.handleChildren(node)

    LIST = TUPLE

    def IMPORT(self, node):
        for alias in node.names:
            if '.' in alias.name and not alias.asname:
                importation = SubmoduleImportation(alias.name, node)
            else:
                name = alias.asname or alias.name
                importation = Importation(name, node, alias.name)
            self.addBinding(node, importation)

    def IMPORTFROM(self, node):
        if node.module == '__future__':
            if not self.futuresAllowed:
                self.report(messages.LateFutureImport,
                            node, [n.name for n in node.names])
        else:
            self.futuresAllowed = False

        module = ('.' * node.level) + (node.module or '')

        for alias in node.names:
            name = alias.asname or alias.name
            if node.module == '__future__':
                importation = FutureImportation(name, node, self.scope)
                if alias.name not in __future__.all_feature_names:
                    self.report(messages.FutureFeatureNotDefined,
                                node, alias.name)
                if alias.name == 'annotations':
                    self.annotationsFutureEnabled = True
            elif alias.name == '*':
                # Only Python 2, local import * is a SyntaxWarning
                if not PY2 and not isinstance(self.scope, ModuleScope):
                    self.report(messages.ImportStarNotPermitted,
                                node, module)
                    continue

                self.scope.importStarred = True
                self.report(messages.ImportStarUsed, node, module)
                importation = StarImportation(module, node)
            else:
                importation = ImportationFrom(name, node,
                                              module, alias.name)
            self.addBinding(node, importation)

    def TRY(self, node):
        handler_names = []
        # List the exception handlers
        for i, handler in enumerate(node.handlers):
            if isinstance(handler.type, ast.Tuple):
                for exc_type in handler.type.elts:
                    handler_names.append(getNodeName(exc_type))
            elif handler.type:
                handler_names.append(getNodeName(handler.type))

            if handler.type is None and i < len(node.handlers) - 1:
                self.report(messages.DefaultExceptNotLast, handler)
        # Memorize the except handlers and process the body
        self.exceptHandlers.append(handler_names)
        for child in node.body:
            self.handleNode(child, node)
        self.exceptHandlers.pop()
        # Process the other nodes: "except:", "else:", "finally:"
        self.handleChildren(node, omit='body')

    TRYEXCEPT = TRY

    def EXCEPTHANDLER(self, node):
        if PY2 or node.name is None:
            self.handleChildren(node)
            return

        # If the name already exists in the scope, modify state of existing
        # binding.
        if node.name in self.scope:
            self.handleNodeStore(node)

        # 3.x: the name of the exception, which is not a Name node, but a
        # simple string, creates a local that is only bound within the scope of
        # the except: block. As such, temporarily remove the existing binding
        # to more accurately determine if the name is used in the except:
        # block.

        try:
            prev_definition = self.scope.pop(node.name)
        except KeyError:
            prev_definition = None

        self.handleNodeStore(node)
        self.handleChildren(node)

        # See discussion on https://github.com/PyCQA/pyflakes/pull/59

        # We're removing the local name since it's being unbound after leaving
        # the except: block and it's always unbound if the except: block is
        # never entered. This will cause an "undefined name" error raised if
        # the checked code tries to use the name afterwards.
        #
        # Unless it's been removed already. Then do nothing.

        try:
            binding = self.scope.pop(node.name)
        except KeyError:
            pass
        else:
            if not binding.used:
                self.report(messages.UnusedVariable, node, node.name)

        # Restore.
        if prev_definition:
            self.scope[node.name] = prev_definition

    def ANNASSIGN(self, node):
        self.handleNode(node.target, node)
        self.handleAnnotation(node.annotation, node)
        if node.value:
            # If the assignment has value, handle the *value* now.
            self.handleNode(node.value, node)

    def COMPARE(self, node):
        left = node.left
        for op, right in zip(node.ops, node.comparators):
            if (
                    isinstance(op, (ast.Is, ast.IsNot)) and (
                        _is_const_non_singleton(left) or
                        _is_const_non_singleton(right)
                    )
            ):
                self.report(messages.IsLiteral, node)
            left = right

        self.handleChildren(node)

    MATCH = MATCH_CASE = MATCHCLASS = MATCHOR = MATCHSEQUENCE = handleChildren
    MATCHSINGLETON = MATCHVALUE = handleChildren

    def _match_target(self, node):
        self.handleNodeStore(node)
        self.handleChildren(node)

    MATCHAS = MATCHMAPPING = MATCHSTAR = _match_target

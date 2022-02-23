#
#   Parse tree nodes
#

from __future__ import absolute_import

import cython
cython.declare(sys=object, os=object, copy=object,
               Builtin=object, error=object, warning=object, Naming=object, PyrexTypes=object,
               py_object_type=object, ModuleScope=object, LocalScope=object, ClosureScope=object,
               StructOrUnionScope=object, PyClassScope=object,
               CppClassScope=object, UtilityCode=object, EncodedString=object,
               error_type=object, _py_int_types=object)

import sys, os, copy
from itertools import chain

from . import Builtin
from .Errors import error, warning, InternalError, CompileError
from . import Naming
from . import PyrexTypes
from . import TypeSlots
from .PyrexTypes import py_object_type, error_type
from .Symtab import (ModuleScope, LocalScope, ClosureScope,
                     StructOrUnionScope, PyClassScope, CppClassScope, TemplateScope)
from .Code import UtilityCode
from .StringEncoding import EncodedString
from . import Future
from . import Options
from . import DebugFlags
from .Pythran import has_np_pythran, pythran_type, is_pythran_buffer
from ..Utils import add_metaclass


if sys.version_info[0] >= 3:
    _py_int_types = int
else:
    _py_int_types = (int, long)


def relative_position(pos):
    return (pos[0].get_filenametable_entry(), pos[1])


def embed_position(pos, docstring):
    if not Options.embed_pos_in_docstring:
        return docstring
    pos_line = u'File: %s (starting at line %s)' % relative_position(pos)
    if docstring is None:
        # unicode string
        return EncodedString(pos_line)

    # make sure we can encode the filename in the docstring encoding
    # otherwise make the docstring a unicode string
    encoding = docstring.encoding
    if encoding is not None:
        try:
            pos_line.encode(encoding)
        except UnicodeEncodeError:
            encoding = None

    if not docstring:
        # reuse the string encoding of the original docstring
        doc = EncodedString(pos_line)
    else:
        doc = EncodedString(pos_line + u'\n' + docstring)
    doc.encoding = encoding
    return doc


def analyse_type_annotation(annotation, env, assigned_value=None):
    base_type = None
    is_ambiguous = False
    explicit_pytype = explicit_ctype = False
    if annotation.is_dict_literal:
        warning(annotation.pos,
                "Dicts should no longer be used as type annotations. Use 'cython.int' etc. directly.")
        for name, value in annotation.key_value_pairs:
            if not name.is_string_literal:
                continue
            if name.value in ('type', b'type'):
                explicit_pytype = True
                if not explicit_ctype:
                    annotation = value
            elif name.value in ('ctype', b'ctype'):
                explicit_ctype = True
                annotation = value
        if explicit_pytype and explicit_ctype:
            warning(annotation.pos, "Duplicate type declarations found in signature annotation")
    arg_type = annotation.analyse_as_type(env)
    if annotation.is_name and not annotation.cython_attribute and annotation.name in ('int', 'long', 'float'):
        # Map builtin numeric Python types to C types in safe cases.
        if assigned_value is not None and arg_type is not None and not arg_type.is_pyobject:
            assigned_type = assigned_value.infer_type(env)
            if assigned_type and assigned_type.is_pyobject:
                # C type seems unsafe, e.g. due to 'None' default value  => ignore annotation type
                is_ambiguous = True
                arg_type = None
        # ignore 'int' and require 'cython.int' to avoid unsafe integer declarations
        if arg_type in (PyrexTypes.c_long_type, PyrexTypes.c_int_type, PyrexTypes.c_float_type):
            arg_type = PyrexTypes.c_double_type if annotation.name == 'float' else py_object_type
    elif arg_type is not None and annotation.is_string_literal:
        warning(annotation.pos,
                "Strings should no longer be used for type declarations. Use 'cython.int' etc. directly.")
    if arg_type is not None:
        if explicit_pytype and not explicit_ctype and not arg_type.is_pyobject:
            warning(annotation.pos,
                    "Python type declaration in signature annotation does not refer to a Python type")
        base_type = CAnalysedBaseTypeNode(
            annotation.pos, type=arg_type, is_arg=True)
    elif is_ambiguous:
        warning(annotation.pos, "Ambiguous types in annotation, ignoring")
    else:
        warning(annotation.pos, "Unknown type declaration in annotation, ignoring")
    return base_type, arg_type


def write_func_call(func, codewriter_class):
    def f(*args, **kwds):
        if len(args) > 1 and isinstance(args[1], codewriter_class):
            # here we annotate the code with this function call
            # but only if new code is generated
            node, code = args[:2]
            marker = '                    /* %s -> %s.%s %s */' % (
                ' ' * code.call_level,
                node.__class__.__name__,
                func.__name__,
                node.pos[1:])
            pristine = code.buffer.stream.tell()
            code.putln(marker)
            start = code.buffer.stream.tell()
            code.call_level += 4
            res = func(*args, **kwds)
            code.call_level -= 4
            if start == code.buffer.stream.tell():
                # no code written => undo writing marker
                code.buffer.stream.truncate(pristine)
            else:
                marker = marker.replace('->', '<-', 1)
                code.putln(marker)
            return res
        else:
            return func(*args, **kwds)
    return f


class VerboseCodeWriter(type):
    # Set this as a metaclass to trace function calls in code.
    # This slows down code generation and makes much larger files.
    def __new__(cls, name, bases, attrs):
        from types import FunctionType
        from .Code import CCodeWriter
        attrs = dict(attrs)
        for mname, m in attrs.items():
            if isinstance(m, FunctionType):
                attrs[mname] = write_func_call(m, CCodeWriter)
        return super(VerboseCodeWriter, cls).__new__(cls, name, bases, attrs)


class CheckAnalysers(type):
    """Metaclass to check that type analysis functions return a node.
    """
    methods = set(['analyse_types',
                   'analyse_expressions',
                   'analyse_target_types'])

    def __new__(cls, name, bases, attrs):
        from types import FunctionType
        def check(name, func):
            def call(*args, **kwargs):
                retval = func(*args, **kwargs)
                if retval is None:
                    print('%s %s %s' % (name, args, kwargs))
                return retval
            return call

        attrs = dict(attrs)
        for mname, m in attrs.items():
            if isinstance(m, FunctionType) and mname in cls.methods:
                attrs[mname] = check(mname, m)
        return super(CheckAnalysers, cls).__new__(cls, name, bases, attrs)


def _with_metaclass(cls):
    if DebugFlags.debug_trace_code_generation:
        return add_metaclass(VerboseCodeWriter)(cls)
    #return add_metaclass(CheckAnalysers)(cls)
    return cls


@_with_metaclass
class Node(object):
    #  pos         (string, int, int)   Source file position
    #  is_name     boolean              Is a NameNode
    #  is_literal  boolean              Is a ConstNode

    is_name = 0
    is_none = 0
    is_nonecheck = 0
    is_literal = 0
    is_terminator = 0
    is_wrapper = False  # is a DefNode wrapper for a C function
    temps = None

    # All descendants should set child_attrs to a list of the attributes
    # containing nodes considered "children" in the tree. Each such attribute
    # can either contain a single node or a list of nodes. See Visitor.py.
    child_attrs = None

    # Subset of attributes that are evaluated in the outer scope (e.g. function default arguments).
    outer_attrs = None

    cf_state = None

    # This may be an additional (or 'actual') type that will be checked when
    # this node is coerced to another type. This could be useful to set when
    # the actual type to which it can coerce is known, but you want to leave
    # the type a py_object_type
    coercion_type = None

    def __init__(self, pos, **kw):
        self.pos = pos
        self.__dict__.update(kw)

    gil_message = "Operation"

    nogil_check = None
    in_nogil_context = False  # For use only during code generation.

    def gil_error(self, env=None):
        error(self.pos, "%s not allowed without gil" % self.gil_message)

    cpp_message = "Operation"

    def cpp_check(self, env):
        if not env.is_cpp():
            self.cpp_error()

    def cpp_error(self):
        error(self.pos, "%s only allowed in c++" % self.cpp_message)

    def clone_node(self):
        """Clone the node. This is defined as a shallow copy, except for member lists
           amongst the child attributes (from get_child_accessors) which are also
           copied. Lists containing child nodes are thus seen as a way for the node
           to hold multiple children directly; the list is not treated as a separate
           level in the tree."""
        result = copy.copy(self)
        for attrname in result.child_attrs:
            value = getattr(result, attrname)
            if isinstance(value, list):
                setattr(result, attrname, [x for x in value])
        return result


    #
    #  There are 3 phases of parse tree processing, applied in order to
    #  all the statements in a given scope-block:
    #
    #  (0) analyse_declarations
    #        Make symbol table entries for all declarations at the current
    #        level, both explicit (def, cdef, etc.) and implicit (assignment
    #        to an otherwise undeclared name).
    #
    #  (1) analyse_expressions
    #         Determine the result types of expressions and fill in the
    #         'type' attribute of each ExprNode. Insert coercion nodes into the
    #         tree where needed to convert to and from Python objects.
    #         Allocate temporary locals for intermediate results. Fill
    #         in the 'result_code' attribute of each ExprNode with a C code
    #         fragment.
    #
    #  (2) generate_code
    #         Emit C code for all declarations, statements and expressions.
    #         Recursively applies the 3 processing phases to the bodies of
    #         functions.
    #

    def analyse_declarations(self, env):
        pass

    def analyse_expressions(self, env):
        raise InternalError("analyse_expressions not implemented for %s" % \
            self.__class__.__name__)

    def generate_code(self, code):
        raise InternalError("generate_code not implemented for %s" % \
            self.__class__.__name__)

    def annotate(self, code):
        # mro does the wrong thing
        if isinstance(self, BlockNode):
            self.body.annotate(code)

    def end_pos(self):
        try:
            return self._end_pos
        except AttributeError:
            pos = self.pos
            if not self.child_attrs:
                self._end_pos = pos
                return pos
            for attr in self.child_attrs:
                child = getattr(self, attr)
                # Sometimes lists, sometimes nodes
                if child is None:
                    pass
                elif isinstance(child, list):
                    for c in child:
                        pos = max(pos, c.end_pos())
                else:
                    pos = max(pos, child.end_pos())
            self._end_pos = pos
            return pos

    def dump(self, level=0, filter_out=("pos",), cutoff=100, encountered=None):
        """Debug helper method that returns a recursive string representation of this node.
        """
        if cutoff == 0:
            return "<...nesting level cutoff...>"
        if encountered is None:
            encountered = set()
        if id(self) in encountered:
            return "<%s (0x%x) -- already output>" % (self.__class__.__name__, id(self))
        encountered.add(id(self))

        def dump_child(x, level):
            if isinstance(x, Node):
                return x.dump(level, filter_out, cutoff-1, encountered)
            elif isinstance(x, list):
                return "[%s]" % ", ".join([dump_child(item, level) for item in x])
            else:
                return repr(x)

        attrs = [(key, value) for key, value in self.__dict__.items() if key not in filter_out]
        if len(attrs) == 0:
            return "<%s (0x%x)>" % (self.__class__.__name__, id(self))
        else:
            indent = "  " * level
            res = "<%s (0x%x)\n" % (self.__class__.__name__, id(self))
            for key, value in attrs:
                res += "%s  %s: %s\n" % (indent, key, dump_child(value, level + 1))
            res += "%s>" % indent
            return res

    def dump_pos(self, mark_column=False, marker='(#)'):
        """Debug helper method that returns the source code context of this node as a string.
        """
        if not self.pos:
            return u''
        source_desc, line, col = self.pos
        contents = source_desc.get_lines(encoding='ASCII', error_handling='ignore')
        # line numbers start at 1
        lines = contents[max(0, line-3):line]
        current = lines[-1]
        if mark_column:
            current = current[:col] + marker + current[col:]
        lines[-1] = current.rstrip() + u'             # <<<<<<<<<<<<<<\n'
        lines += contents[line:line+2]
        return u'"%s":%d:%d\n%s\n' % (
            source_desc.get_escaped_description(), line, col, u''.join(lines))

class CompilerDirectivesNode(Node):
    """
    Sets compiler directives for the children nodes
    """
    #  directives     {string:value}  A dictionary holding the right value for
    #                                 *all* possible directives.
    #  body           Node
    child_attrs = ["body"]

    def analyse_declarations(self, env):
        old = env.directives
        env.directives = self.directives
        self.body.analyse_declarations(env)
        env.directives = old

    def analyse_expressions(self, env):
        old = env.directives
        env.directives = self.directives
        self.body = self.body.analyse_expressions(env)
        env.directives = old
        return self

    def generate_function_definitions(self, env, code):
        env_old = env.directives
        code_old = code.globalstate.directives
        code.globalstate.directives = self.directives
        self.body.generate_function_definitions(env, code)
        env.directives = env_old
        code.globalstate.directives = code_old

    def generate_execution_code(self, code):
        old = code.globalstate.directives
        code.globalstate.directives = self.directives
        self.body.generate_execution_code(code)
        code.globalstate.directives = old

    def annotate(self, code):
        old = code.globalstate.directives
        code.globalstate.directives = self.directives
        self.body.annotate(code)
        code.globalstate.directives = old

class BlockNode(object):
    #  Mixin class for nodes representing a declaration block.

    def generate_cached_builtins_decls(self, env, code):
        entries = env.global_scope().undeclared_cached_builtins
        for entry in entries:
            code.globalstate.add_cached_builtin_decl(entry)
        del entries[:]

    def generate_lambda_definitions(self, env, code):
        for node in env.lambda_defs:
            node.generate_function_definitions(env, code)

class StatListNode(Node):
    # stats     a list of StatNode

    child_attrs = ["stats"]

    @staticmethod
    def create_analysed(pos, env, *args, **kw):
        node = StatListNode(pos, *args, **kw)
        return node  # No node-specific analysis needed

    def analyse_declarations(self, env):
        #print "StatListNode.analyse_declarations" ###
        for stat in self.stats:
            stat.analyse_declarations(env)

    def analyse_expressions(self, env):
        #print "StatListNode.analyse_expressions" ###
        self.stats = [stat.analyse_expressions(env)
                      for stat in self.stats]
        return self

    def generate_function_definitions(self, env, code):
        #print "StatListNode.generate_function_definitions" ###
        for stat in self.stats:
            stat.generate_function_definitions(env, code)

    def generate_execution_code(self, code):
        #print "StatListNode.generate_execution_code" ###
        for stat in self.stats:
            code.mark_pos(stat.pos)
            stat.generate_execution_code(code)

    def annotate(self, code):
        for stat in self.stats:
            stat.annotate(code)


class StatNode(Node):
    #
    #  Code generation for statements is split into the following subphases:
    #
    #  (1) generate_function_definitions
    #        Emit C code for the definitions of any structs,
    #        unions, enums and functions defined in the current
    #        scope-block.
    #
    #  (2) generate_execution_code
    #        Emit C code for executable statements.
    #

    def generate_function_definitions(self, env, code):
        pass

    def generate_execution_code(self, code):
        raise InternalError("generate_execution_code not implemented for %s" % \
            self.__class__.__name__)


class CDefExternNode(StatNode):
    #  include_file       string or None
    #  verbatim_include   string or None
    #  body               StatListNode

    child_attrs = ["body"]

    def analyse_declarations(self, env):
        old_cinclude_flag = env.in_cinclude
        env.in_cinclude = 1
        self.body.analyse_declarations(env)
        env.in_cinclude = old_cinclude_flag

        if self.include_file or self.verbatim_include:
            # Determine whether include should be late
            stats = self.body.stats
            if not env.directives['preliminary_late_includes_cy28']:
                late = False
            elif not stats:
                # Special case: empty 'cdef extern' blocks are early
                late = False
            else:
                late = all(isinstance(node, CVarDefNode) for node in stats)
            env.add_include_file(self.include_file, self.verbatim_include, late)

    def analyse_expressions(self, env):
        return self

    def generate_execution_code(self, code):
        pass

    def annotate(self, code):
        self.body.annotate(code)


class CDeclaratorNode(Node):
    # Part of a C declaration.
    #
    # Processing during analyse_declarations phase:
    #
    #   analyse
    #      Returns (name, type) pair where name is the
    #      CNameDeclaratorNode of the name being declared
    #      and type is the type it is being declared as.
    #
    #  calling_convention  string   Calling convention of CFuncDeclaratorNode
    #                               for which this is a base

    child_attrs = []

    calling_convention = ""

    def analyse_templates(self):
        # Only C++ functions have templates.
        return None


class CNameDeclaratorNode(CDeclaratorNode):
    #  name    string             The Cython name being declared
    #  cname   string or None     C name, if specified
    #  default ExprNode or None   the value assigned on declaration

    child_attrs = ['default']

    default = None

    def analyse(self, base_type, env, nonempty=0, visibility=None, in_pxd=False):
        if nonempty and self.name == '':
            # May have mistaken the name for the type.
            if base_type.is_ptr or base_type.is_array or base_type.is_buffer:
                error(self.pos, "Missing argument name")
            elif base_type.is_void:
                error(self.pos, "Use spam() rather than spam(void) to declare a function with no arguments.")
            else:
                self.name = base_type.declaration_code("", for_display=1, pyrex=1)
                base_type = py_object_type

        if base_type.is_fused and env.fused_to_specific:
            base_type = base_type.specialize(env.fused_to_specific)

        self.type = base_type
        return self, base_type


class CPtrDeclaratorNode(CDeclaratorNode):
    # base     CDeclaratorNode

    child_attrs = ["base"]

    def analyse_templates(self):
        return self.base.analyse_templates()

    def analyse(self, base_type, env, nonempty=0, visibility=None, in_pxd=False):
        if base_type.is_pyobject:
            error(self.pos, "Pointer base type cannot be a Python object")
        ptr_type = PyrexTypes.c_ptr_type(base_type)
        return self.base.analyse(ptr_type, env, nonempty=nonempty, visibility=visibility, in_pxd=in_pxd)


class CReferenceDeclaratorNode(CDeclaratorNode):
    # base     CDeclaratorNode

    child_attrs = ["base"]

    def analyse_templates(self):
        return self.base.analyse_templates()

    def analyse(self, base_type, env, nonempty=0, visibility=None, in_pxd=False):
        if base_type.is_pyobject:
            error(self.pos, "Reference base type cannot be a Python object")
        ref_type = PyrexTypes.c_ref_type(base_type)
        return self.base.analyse(ref_type, env, nonempty=nonempty, visibility=visibility, in_pxd=in_pxd)


class CArrayDeclaratorNode(CDeclaratorNode):
    # base        CDeclaratorNode
    # dimension   ExprNode

    child_attrs = ["base", "dimension"]

    def analyse(self, base_type, env, nonempty=0, visibility=None, in_pxd=False):
        if (base_type.is_cpp_class and base_type.is_template_type()) or base_type.is_cfunction:
            from .ExprNodes import TupleNode
            if isinstance(self.dimension, TupleNode):
                args = self.dimension.args
            else:
                args = self.dimension,
            values = [v.analyse_as_type(env) for v in args]
            if None in values:
                ix = values.index(None)
                error(args[ix].pos, "Template parameter not a type")
                base_type = error_type
            else:
                base_type = base_type.specialize_here(self.pos, values)
            return self.base.analyse(base_type, env, nonempty=nonempty, visibility=visibility, in_pxd=in_pxd)
        if self.dimension:
            self.dimension = self.dimension.analyse_const_expression(env)
            if not self.dimension.type.is_int:
                error(self.dimension.pos, "Array dimension not integer")
            size = self.dimension.get_constant_c_result_code()
            if size is not None:
                try:
                    size = int(size)
                except ValueError:
                    # runtime constant?
                    pass
        else:
            size = None
        if not base_type.is_complete():
            error(self.pos, "Array element type '%s' is incomplete" % base_type)
        if base_type.is_pyobject:
            error(self.pos, "Array element cannot be a Python object")
        if base_type.is_cfunction:
            error(self.pos, "Array element cannot be a function")
        array_type = PyrexTypes.c_array_type(base_type, size)
        return self.base.analyse(array_type, env, nonempty=nonempty, visibility=visibility, in_pxd=in_pxd)


class CFuncDeclaratorNode(CDeclaratorNode):
    # base             CDeclaratorNode
    # args             [CArgDeclNode]
    # templates        [TemplatePlaceholderType]
    # has_varargs      boolean
    # exception_value  ConstNode
    # exception_check  boolean    True if PyErr_Occurred check needed
    # nogil            boolean    Can be called without gil
    # with_gil         boolean    Acquire gil around function body
    # is_const_method  boolean    Whether this is a const method

    child_attrs = ["base", "args", "exception_value"]

    overridable = 0
    optional_arg_count = 0
    is_const_method = 0
    templates = None

    def analyse_templates(self):
        if isinstance(self.base, CArrayDeclaratorNode):
            from .ExprNodes import TupleNode, NameNode
            template_node = self.base.dimension
            if isinstance(template_node, TupleNode):
                template_nodes = template_node.args
            elif isinstance(template_node, NameNode):
                template_nodes = [template_node]
            else:
                error(template_node.pos, "Template arguments must be a list of names")
                return None
            self.templates = []
            for template in template_nodes:
                if isinstance(template, NameNode):
                    self.templates.append(PyrexTypes.TemplatePlaceholderType(template.name))
                else:
                    error(template.pos, "Template arguments must be a list of names")
            self.base = self.base.base
            return self.templates
        else:
            return None

    def analyse(self, return_type, env, nonempty=0, directive_locals=None, visibility=None, in_pxd=False):
        if directive_locals is None:
            directive_locals = {}
        if nonempty:
            nonempty -= 1
        func_type_args = []
        for i, arg_node in enumerate(self.args):
            name_declarator, type = arg_node.analyse(
                env, nonempty=nonempty,
                is_self_arg=(i == 0 and env.is_c_class_scope and 'staticmethod' not in env.directives))
            name = name_declarator.name
            if name in directive_locals:
                type_node = directive_locals[name]
                other_type = type_node.analyse_as_type(env)
                if other_type is None:
                    error(type_node.pos, "Not a type")
                elif (type is not PyrexTypes.py_object_type
                      and not type.same_as(other_type)):
                    error(self.base.pos, "Signature does not agree with previous declaration")
                    error(type_node.pos, "Previous declaration here")
                else:
                    type = other_type
            if name_declarator.cname:
                error(self.pos, "Function argument cannot have C name specification")
            if i == 0 and env.is_c_class_scope and type.is_unspecified:
                # fix the type of self
                type = env.parent_type
            # Turn *[] argument into **
            if type.is_array:
                type = PyrexTypes.c_ptr_type(type.base_type)
            # Catch attempted C-style func(void) decl
            if type.is_void:
                error(arg_node.pos, "Use spam() rather than spam(void) to declare a function with no arguments.")
            func_type_args.append(
                PyrexTypes.CFuncTypeArg(name, type, arg_node.pos))
            if arg_node.default:
                self.optional_arg_count += 1
            elif self.optional_arg_count:
                error(self.pos, "Non-default argument follows default argument")

        exc_val = None
        exc_check = 0
        if self.exception_check == '+':
            env.add_include_file('ios')         # for std::ios_base::failure
            env.add_include_file('new')         # for std::bad_alloc
            env.add_include_file('stdexcept')
            env.add_include_file('typeinfo')    # for std::bad_cast
        if (return_type.is_pyobject
                and (self.exception_value or self.exception_check)
                and self.exception_check != '+'):
            error(self.pos, "Exception clause not allowed for function returning Python object")
        else:
            if self.exception_value is None and self.exception_check and self.exception_check != '+':
                # Use an explicit exception return value to speed up exception checks.
                # Even if it is not declared, we can use the default exception value of the return type,
                # unless the function is some kind of external function that we do not control.
                if return_type.exception_value is not None and (visibility != 'extern' and not in_pxd):
                    # Extension types are more difficult because the signature must match the base type signature.
                    if not env.is_c_class_scope:
                        from .ExprNodes import ConstNode
                        self.exception_value = ConstNode(
                            self.pos, value=return_type.exception_value, type=return_type)
            if self.exception_value:
                self.exception_value = self.exception_value.analyse_const_expression(env)
                if self.exception_check == '+':
                    exc_val_type = self.exception_value.type
                    if (not exc_val_type.is_error
                            and not exc_val_type.is_pyobject
                            and not (exc_val_type.is_cfunction
                                     and not exc_val_type.return_type.is_pyobject
                                     and not exc_val_type.args)
                            and not (exc_val_type == PyrexTypes.c_char_type
                                     and self.exception_value.value == '*')):
                        error(self.exception_value.pos,
                              "Exception value must be a Python exception or cdef function with no arguments or *.")
                    exc_val = self.exception_value
                else:
                    self.exception_value = self.exception_value.coerce_to(
                        return_type, env).analyse_const_expression(env)
                    exc_val = self.exception_value.get_constant_c_result_code()
                    if exc_val is None:
                        raise InternalError(
                            "get_constant_c_result_code not implemented for %s" %
                            self.exception_value.__class__.__name__)
                    if not return_type.assignable_from(self.exception_value.type):
                        error(self.exception_value.pos,
                              "Exception value incompatible with function return type")
            exc_check = self.exception_check
        if return_type.is_cfunction:
            error(self.pos, "Function cannot return a function")
        func_type = PyrexTypes.CFuncType(
            return_type, func_type_args, self.has_varargs,
            optional_arg_count=self.optional_arg_count,
            exception_value=exc_val, exception_check=exc_check,
            calling_convention=self.base.calling_convention,
            nogil=self.nogil, with_gil=self.with_gil, is_overridable=self.overridable,
            is_const_method=self.is_const_method,
            templates=self.templates)

        if self.optional_arg_count:
            if func_type.is_fused:
                # This is a bit of a hack... When we need to create specialized CFuncTypes
                # on the fly because the cdef is defined in a pxd, we need to declare the specialized optional arg
                # struct
                def declare_opt_arg_struct(func_type, fused_cname):
                    self.declare_optional_arg_struct(func_type, env, fused_cname)

                func_type.declare_opt_arg_struct = declare_opt_arg_struct
            else:
                self.declare_optional_arg_struct(func_type, env)

        callspec = env.directives['callspec']
        if callspec:
            current = func_type.calling_convention
            if current and current != callspec:
                error(self.pos, "cannot have both '%s' and '%s' "
                      "calling conventions" % (current, callspec))
            func_type.calling_convention = callspec
        return self.base.analyse(func_type, env, visibility=visibility, in_pxd=in_pxd)

    def declare_optional_arg_struct(self, func_type, env, fused_cname=None):
        """
        Declares the optional argument struct (the struct used to hold the
        values for optional arguments). For fused cdef functions, this is
        deferred as analyse_declarations is called only once (on the fused
        cdef function).
        """
        scope = StructOrUnionScope()
        arg_count_member = '%sn' % Naming.pyrex_prefix
        scope.declare_var(arg_count_member, PyrexTypes.c_int_type, self.pos)

        for arg in func_type.args[len(func_type.args) - self.optional_arg_count:]:
            scope.declare_var(arg.name, arg.type, arg.pos, allow_pyobject=True, allow_memoryview=True)

        struct_cname = env.mangle(Naming.opt_arg_prefix, self.base.name)

        if fused_cname is not None:
            struct_cname = PyrexTypes.get_fused_cname(fused_cname, struct_cname)

        op_args_struct = env.global_scope().declare_struct_or_union(
            name=struct_cname,
            kind='struct',
            scope=scope,
            typedef_flag=0,
            pos=self.pos,
            cname=struct_cname)

        op_args_struct.defined_in_pxd = 1
        op_args_struct.used = 1

        func_type.op_arg_struct = PyrexTypes.c_ptr_type(op_args_struct.type)


class CConstDeclaratorNode(CDeclaratorNode):
    # base     CDeclaratorNode

    child_attrs = ["base"]

    def analyse(self, base_type, env, nonempty=0, visibility=None, in_pxd=False):
        if base_type.is_pyobject:
            error(self.pos,
                  "Const base type cannot be a Python object")
        const = PyrexTypes.c_const_type(base_type)
        return self.base.analyse(const, env, nonempty=nonempty, visibility=visibility, in_pxd=in_pxd)


class CArgDeclNode(Node):
    # Item in a function declaration argument list.
    #
    # base_type      CBaseTypeNode
    # declarator     CDeclaratorNode
    # not_none       boolean            Tagged with 'not None'
    # or_none        boolean            Tagged with 'or None'
    # accept_none    boolean            Resolved boolean for not_none/or_none
    # default        ExprNode or None
    # default_value  PyObjectConst      constant for default value
    # annotation     ExprNode or None   Py3 function arg annotation
    # is_self_arg    boolean            Is the "self" arg of an extension type method
    # is_type_arg    boolean            Is the "class" arg of an extension type classmethod
    # is_kw_only     boolean            Is a keyword-only argument
    # is_dynamic     boolean            Non-literal arg stored inside CyFunction

    child_attrs = ["base_type", "declarator", "default", "annotation"]
    outer_attrs = ["default", "annotation"]

    is_self_arg = 0
    is_type_arg = 0
    is_generic = 1
    kw_only = 0
    not_none = 0
    or_none = 0
    type = None
    name_declarator = None
    default_value = None
    annotation = None
    is_dynamic = 0

    def analyse(self, env, nonempty=0, is_self_arg=False):
        if is_self_arg:
            self.base_type.is_self_arg = self.is_self_arg = True
        if self.type is None:
            # The parser may misinterpret names as types. We fix that here.
            if isinstance(self.declarator, CNameDeclaratorNode) and self.declarator.name == '':
                if nonempty:
                    if self.base_type.is_basic_c_type:
                        # char, short, long called "int"
                        type = self.base_type.analyse(env, could_be_name=True)
                        arg_name = type.empty_declaration_code()
                    else:
                        arg_name = self.base_type.name
                    self.declarator.name = EncodedString(arg_name)
                    self.base_type.name = None
                    self.base_type.is_basic_c_type = False
                could_be_name = True
            else:
                could_be_name = False
            self.base_type.is_arg = True
            base_type = self.base_type.analyse(env, could_be_name=could_be_name)
            if hasattr(self.base_type, 'arg_name') and self.base_type.arg_name:
                self.declarator.name = self.base_type.arg_name

            # The parser is unable to resolve the ambiguity of [] as part of the
            # type (e.g. in buffers) or empty declarator (as with arrays).
            # This is only arises for empty multi-dimensional arrays.
            if (base_type.is_array
                    and isinstance(self.base_type, TemplatedTypeNode)
                    and isinstance(self.declarator, CArrayDeclaratorNode)):
                declarator = self.declarator
                while isinstance(declarator.base, CArrayDeclaratorNode):
                    declarator = declarator.base
                declarator.base = self.base_type.array_declarator
                base_type = base_type.base_type

            # inject type declaration from annotations
            # this is called without 'env' by AdjustDefByDirectives transform before declaration analysis
            if self.annotation and env and env.directives['annotation_typing'] and self.base_type.name is None:
                arg_type = self.inject_type_from_annotations(env)
                if arg_type is not None:
                    base_type = arg_type
            return self.declarator.analyse(base_type, env, nonempty=nonempty)
        else:
            return self.name_declarator, self.type

    def inject_type_from_annotations(self, env):
        annotation = self.annotation
        if not annotation:
            return None
        base_type, arg_type = analyse_type_annotation(annotation, env, assigned_value=self.default)
        if base_type is not None:
            self.base_type = base_type
        return arg_type

    def calculate_default_value_code(self, code):
        if self.default_value is None:
            if self.default:
                if self.default.is_literal:
                    # will not output any code, just assign the result_code
                    self.default.generate_evaluation_code(code)
                    return self.type.cast_code(self.default.result())
                self.default_value = code.get_argument_default_const(self.type)
        return self.default_value

    def annotate(self, code):
        if self.default:
            self.default.annotate(code)

    def generate_assignment_code(self, code, target=None, overloaded_assignment=False):
        default = self.default
        if default is None or default.is_literal:
            return
        if target is None:
            target = self.calculate_default_value_code(code)
        default.generate_evaluation_code(code)
        default.make_owned_reference(code)
        result = default.result() if overloaded_assignment else default.result_as(self.type)
        code.putln("%s = %s;" % (target, result))
        if self.type.is_pyobject:
            code.put_giveref(default.result())
        default.generate_post_assignment_code(code)
        default.free_temps(code)


class CBaseTypeNode(Node):
    # Abstract base class for C base type nodes.
    #
    # Processing during analyse_declarations phase:
    #
    #   analyse
    #     Returns the type.

    def analyse_as_type(self, env):
        return self.analyse(env)


class CAnalysedBaseTypeNode(Node):
    # type            type

    child_attrs = []

    def analyse(self, env, could_be_name=False):
        return self.type


class CSimpleBaseTypeNode(CBaseTypeNode):
    # name             string
    # module_path      [string]     Qualifying name components
    # is_basic_c_type  boolean
    # signed           boolean
    # longness         integer
    # complex          boolean
    # is_self_arg      boolean      Is self argument of C method
    # ##is_type_arg      boolean      Is type argument of class method

    child_attrs = []
    arg_name = None   # in case the argument name was interpreted as a type
    module_path = []
    is_basic_c_type = False
    complex = False

    def analyse(self, env, could_be_name=False):
        # Return type descriptor.
        #print "CSimpleBaseTypeNode.analyse: is_self_arg =", self.is_self_arg ###
        type = None
        if self.is_basic_c_type:
            type = PyrexTypes.simple_c_type(self.signed, self.longness, self.name)
            if not type:
                error(self.pos, "Unrecognised type modifier combination")
        elif self.name == "object" and not self.module_path:
            type = py_object_type
        elif self.name is None:
            if self.is_self_arg and env.is_c_class_scope:
                #print "CSimpleBaseTypeNode.analyse: defaulting to parent type" ###
                type = env.parent_type
            ## elif self.is_type_arg and env.is_c_class_scope:
            ##     type = Builtin.type_type
            else:
                type = py_object_type
        else:
            if self.module_path:
                # Maybe it's a nested C++ class.
                scope = env
                for item in self.module_path:
                    entry = scope.lookup(item)
                    if entry is not None and entry.is_cpp_class:
                        scope = entry.type.scope
                    else:
                        scope = None
                        break

                if scope is None:
                    # Maybe it's a cimport.
                    scope = env.find_imported_module(self.module_path, self.pos)
            else:
                scope = env

            if scope:
                if scope.is_c_class_scope:
                    scope = scope.global_scope()

                type = scope.lookup_type(self.name)
                if type is not None:
                    pass
                elif could_be_name:
                    if self.is_self_arg and env.is_c_class_scope:
                        type = env.parent_type
                    ## elif self.is_type_arg and env.is_c_class_scope:
                    ##     type = Builtin.type_type
                    else:
                        type = py_object_type
                    self.arg_name = EncodedString(self.name)
                else:
                    if self.templates:
                        if not self.name in self.templates:
                            error(self.pos, "'%s' is not a type identifier" % self.name)
                        type = PyrexTypes.TemplatePlaceholderType(self.name)
                    else:
                        error(self.pos, "'%s' is not a type identifier" % self.name)
        if type and type.is_fused and env.fused_to_specific:
            type = type.specialize(env.fused_to_specific)
        if self.complex:
            if not type.is_numeric or type.is_complex:
                error(self.pos, "can only complexify c numeric types")
            type = PyrexTypes.CComplexType(type)
            type.create_declaration_utility_code(env)
        elif type is Builtin.complex_type:
            # Special case: optimise builtin complex type into C's
            # double complex.  The parser cannot do this (as for the
            # normal scalar types) as the user may have redeclared the
            # 'complex' type.  Testing for the exact type here works.
            type = PyrexTypes.c_double_complex_type
            type.create_declaration_utility_code(env)
            self.complex = True
        if type:
            return type
        else:
            return PyrexTypes.error_type

class MemoryViewSliceTypeNode(CBaseTypeNode):

    name = 'memoryview'
    child_attrs = ['base_type_node', 'axes']

    def analyse(self, env, could_be_name=False):

        base_type = self.base_type_node.analyse(env)
        if base_type.is_error: return base_type

        from . import MemoryView

        try:
            axes_specs = MemoryView.get_axes_specs(env, self.axes)
        except CompileError as e:
            error(e.position, e.message_only)
            self.type = PyrexTypes.ErrorType()
            return self.type

        if not MemoryView.validate_axes(self.pos, axes_specs):
            self.type = error_type
        else:
            self.type = PyrexTypes.MemoryViewSliceType(base_type, axes_specs)
            self.type.validate_memslice_dtype(self.pos)
            self.use_memview_utilities(env)

        return self.type

    def use_memview_utilities(self, env):
        from . import MemoryView
        env.use_utility_code(MemoryView.view_utility_code)


class CNestedBaseTypeNode(CBaseTypeNode):
    # For C++ classes that live inside other C++ classes.

    # name             string
    # base_type        CBaseTypeNode

    child_attrs = ['base_type']

    def analyse(self, env, could_be_name=None):
        base_type = self.base_type.analyse(env)
        if base_type is PyrexTypes.error_type:
            return PyrexTypes.error_type
        if not base_type.is_cpp_class:
            error(self.pos, "'%s' is not a valid type scope" % base_type)
            return PyrexTypes.error_type
        type_entry = base_type.scope.lookup_here(self.name)
        if not type_entry or not type_entry.is_type:
            error(self.pos, "'%s.%s' is not a type identifier" % (base_type, self.name))
            return PyrexTypes.error_type
        return type_entry.type


class TemplatedTypeNode(CBaseTypeNode):
    #  After parsing:
    #  positional_args  [ExprNode]        List of positional arguments
    #  keyword_args     DictNode          Keyword arguments
    #  base_type_node   CBaseTypeNode

    #  After analysis:
    #  type             PyrexTypes.BufferType or PyrexTypes.CppClassType  ...containing the right options

    child_attrs = ["base_type_node", "positional_args",
                   "keyword_args", "dtype_node"]

    dtype_node = None

    name = None

    def analyse(self, env, could_be_name=False, base_type=None):
        if base_type is None:
            base_type = self.base_type_node.analyse(env)
        if base_type.is_error: return base_type

        if base_type.is_cpp_class and base_type.is_template_type():
            # Templated class
            if self.keyword_args and self.keyword_args.key_value_pairs:
                error(self.pos, "c++ templates cannot take keyword arguments")
                self.type = PyrexTypes.error_type
            else:
                template_types = []
                for template_node in self.positional_args:
                    type = template_node.analyse_as_type(env)
                    if type is None:
                        error(template_node.pos, "unknown type in template argument")
                        type = error_type
                    template_types.append(type)
                self.type = base_type.specialize_here(self.pos, template_types)

        elif base_type.is_pyobject:
            # Buffer
            from . import Buffer

            options = Buffer.analyse_buffer_options(
                self.pos,
                env,
                self.positional_args,
                self.keyword_args,
                base_type.buffer_defaults)

            if sys.version_info[0] < 3:
                # Py 2.x enforces byte strings as keyword arguments ...
                options = dict([(name.encode('ASCII'), value)
                                for name, value in options.items()])

            self.type = PyrexTypes.BufferType(base_type, **options)
            if has_np_pythran(env) and is_pythran_buffer(self.type):
                self.type = PyrexTypes.PythranExpr(pythran_type(self.type), self.type)

        else:
            # Array
            empty_declarator = CNameDeclaratorNode(self.pos, name="", cname=None)
            if len(self.positional_args) > 1 or self.keyword_args.key_value_pairs:
                error(self.pos, "invalid array declaration")
                self.type = PyrexTypes.error_type
            else:
                # It would be nice to merge this class with CArrayDeclaratorNode,
                # but arrays are part of the declaration, not the type...
                if not self.positional_args:
                    dimension = None
                else:
                    dimension = self.positional_args[0]
                self.array_declarator = CArrayDeclaratorNode(
                    self.pos,
                    base=empty_declarator,
                    dimension=dimension)
                self.type = self.array_declarator.analyse(base_type, env)[1]

        if self.type.is_fused and env.fused_to_specific:
            self.type = self.type.specialize(env.fused_to_specific)

        return self.type


class CComplexBaseTypeNode(CBaseTypeNode):
    # base_type   CBaseTypeNode
    # declarator  CDeclaratorNode

    child_attrs = ["base_type", "declarator"]

    def analyse(self, env, could_be_name=False):
        base = self.base_type.analyse(env, could_be_name)
        _, type = self.declarator.analyse(base, env)
        return type


class CTupleBaseTypeNode(CBaseTypeNode):
    # components [CBaseTypeNode]

    child_attrs = ["components"]

    def analyse(self, env, could_be_name=False):
        component_types = []
        for c in self.components:
            type = c.analyse(env)
            if type.is_pyobject:
                error(c.pos, "Tuple types can't (yet) contain Python objects.")
                return error_type
            component_types.append(type)
        entry = env.declare_tuple_type(self.pos, component_types)
        entry.used = True
        return entry.type


class FusedTypeNode(CBaseTypeNode):
    """
    Represents a fused type in a ctypedef statement:

        ctypedef cython.fused_type(int, long, long long) integral

    name            str                     name of this fused type
    types           [CSimpleBaseTypeNode]   is the list of types to be fused
    """

    child_attrs = []

    def analyse_declarations(self, env):
        type = self.analyse(env)
        entry = env.declare_typedef(self.name, type, self.pos)

        # Omit the typedef declaration that self.declarator would produce
        entry.in_cinclude = True

    def analyse(self, env, could_be_name=False):
        types = []
        for type_node in self.types:
            type = type_node.analyse_as_type(env)

            if not type:
                error(type_node.pos, "Not a type")
                continue

            if type in types:
                error(type_node.pos, "Type specified multiple times")
            else:
                types.append(type)

        # if len(self.types) == 1:
        #     return types[0]

        return PyrexTypes.FusedType(types, name=self.name)


class CConstTypeNode(CBaseTypeNode):
    # base_type     CBaseTypeNode

    child_attrs = ["base_type"]

    def analyse(self, env, could_be_name=False):
        base = self.base_type.analyse(env, could_be_name)
        if base.is_pyobject:
            error(self.pos,
                  "Const base type cannot be a Python object")
        return PyrexTypes.c_const_type(base)


class CVarDefNode(StatNode):
    #  C variable definition or forward/extern function declaration.
    #
    #  visibility    'private' or 'public' or 'extern'
    #  base_type     CBaseTypeNode
    #  declarators   [CDeclaratorNode]
    #  in_pxd        boolean
    #  api           boolean
    #  overridable   boolean        whether it is a cpdef
    #  modifiers     ['inline']

    #  decorators    [cython.locals(...)] or None
    #  directive_locals { string : NameNode } locals defined by cython.locals(...)

    child_attrs = ["base_type", "declarators"]

    decorators = None
    directive_locals = None

    def analyse_declarations(self, env, dest_scope=None):
        if self.directive_locals is None:
            self.directive_locals = {}
        if not dest_scope:
            dest_scope = env
        self.dest_scope = dest_scope

        if self.declarators:
            templates = self.declarators[0].analyse_templates()
        else:
            templates = None
        if templates is not None:
            if self.visibility != 'extern':
                error(self.pos, "Only extern functions allowed")
            if len(self.declarators) > 1:
                error(self.declarators[1].pos, "Can't multiply declare template types")
            env = TemplateScope('func_template', env)
            env.directives = env.outer_scope.directives
            for template_param in templates:
                env.declare_type(template_param.name, template_param, self.pos)

        base_type = self.base_type.analyse(env)

        if base_type.is_fused and not self.in_pxd and (env.is_c_class_scope or
                                                       env.is_module_scope):
            error(self.pos, "Fused types not allowed here")
            return error_type

        self.entry = None
        visibility = self.visibility

        for declarator in self.declarators:

            if (len(self.declarators) > 1
                    and not isinstance(declarator, CNameDeclaratorNode)
                    and env.directives['warn.multiple_declarators']):
                warning(
                    declarator.pos,
                    "Non-trivial type declarators in shared declaration (e.g. mix of pointers and values). "
                    "Each pointer declaration should be on its own line.", 1)

            create_extern_wrapper = (self.overridable
                                     and self.visibility == 'extern'
                                     and env.is_module_scope)
            if create_extern_wrapper:
                declarator.overridable = False
            if isinstance(declarator, CFuncDeclaratorNode):
                name_declarator, type = declarator.analyse(
                    base_type, env, directive_locals=self.directive_locals, visibility=visibility, in_pxd=self.in_pxd)
            else:
                name_declarator, type = declarator.analyse(
                    base_type, env, visibility=visibility, in_pxd=self.in_pxd)
            if not type.is_complete():
                if not (self.visibility == 'extern' and type.is_array or type.is_memoryviewslice):
                    error(declarator.pos, "Variable type '%s' is incomplete" % type)
            if self.visibility == 'extern' and type.is_pyobject:
                error(declarator.pos, "Python object cannot be declared extern")
            name = name_declarator.name
            cname = name_declarator.cname
            if name == '':
                error(declarator.pos, "Missing name in declaration.")
                return
            if type.is_reference and self.visibility != 'extern':
                error(declarator.pos, "C++ references cannot be declared; use a pointer instead")
            if type.is_cfunction:
                if 'staticmethod' in env.directives:
                    type.is_static_method = True
                self.entry = dest_scope.declare_cfunction(
                    name, type, declarator.pos,
                    cname=cname, visibility=self.visibility, in_pxd=self.in_pxd,
                    api=self.api, modifiers=self.modifiers, overridable=self.overridable)
                if self.entry is not None:
                    self.entry.directive_locals = copy.copy(self.directive_locals)
                if create_extern_wrapper:
                    self.entry.type.create_to_py_utility_code(env)
                    self.entry.create_wrapper = True
            else:
                if self.overridable:
                    warning(self.pos, "cpdef variables will not be supported in Cython 3; "
                            "currently they are no different from cdef variables", 2)
                if self.directive_locals:
                    error(self.pos, "Decorators can only be followed by functions")
                self.entry = dest_scope.declare_var(
                    name, type, declarator.pos,
                    cname=cname, visibility=visibility, in_pxd=self.in_pxd,
                    api=self.api, is_cdef=1)
                if Options.docstrings:
                    self.entry.doc = embed_position(self.pos, self.doc)


class CStructOrUnionDefNode(StatNode):
    #  name          string
    #  cname         string or None
    #  kind          "struct" or "union"
    #  typedef_flag  boolean
    #  visibility    "public" or "private"
    #  api           boolean
    #  in_pxd        boolean
    #  attributes    [CVarDefNode] or None
    #  entry         Entry
    #  packed        boolean

    child_attrs = ["attributes"]

    def declare(self, env, scope=None):
        self.entry = env.declare_struct_or_union(
            self.name, self.kind, scope, self.typedef_flag, self.pos,
            self.cname, visibility=self.visibility, api=self.api,
            packed=self.packed)

    def analyse_declarations(self, env):
        scope = None
        if self.attributes is not None:
            scope = StructOrUnionScope(self.name)
        self.declare(env, scope)
        if self.attributes is not None:
            if self.in_pxd and not env.in_cinclude:
                self.entry.defined_in_pxd = 1
            for attr in self.attributes:
                attr.analyse_declarations(env, scope)
            if self.visibility != 'extern':
                for attr in scope.var_entries:
                    type = attr.type
                    while type.is_array:
                        type = type.base_type
                    if type == self.entry.type:
                        error(attr.pos, "Struct cannot contain itself as a member.")

    def analyse_expressions(self, env):
        return self

    def generate_execution_code(self, code):
        pass


class CppClassNode(CStructOrUnionDefNode, BlockNode):

    #  name          string
    #  cname         string or None
    #  visibility    "extern"
    #  in_pxd        boolean
    #  attributes    [CVarDefNode] or None
    #  entry         Entry
    #  base_classes  [CBaseTypeNode]
    #  templates     [(string, bool)] or None
    #  decorators    [DecoratorNode] or None

    decorators = None

    def declare(self, env):
        if self.templates is None:
            template_types = None
        else:
            template_types = [PyrexTypes.TemplatePlaceholderType(template_name, not required)
                              for template_name, required in self.templates]
            num_optional_templates = sum(not required for _, required in self.templates)
            if num_optional_templates and not all(required for _, required in self.templates[:-num_optional_templates]):
                error(self.pos, "Required template parameters must precede optional template parameters.")
        self.entry = env.declare_cpp_class(
            self.name, None, self.pos, self.cname,
            base_classes=[], visibility=self.visibility, templates=template_types)

    def analyse_declarations(self, env):
        if self.templates is None:
            template_types = template_names = None
        else:
            template_names = [template_name for template_name, _ in self.templates]
            template_types = [PyrexTypes.TemplatePlaceholderType(template_name, not required)
                              for template_name, required in self.templates]
        scope = None
        if self.attributes is not None:
            scope = CppClassScope(self.name, env, templates=template_names)
        def base_ok(base_class):
            if base_class.is_cpp_class or base_class.is_struct:
                return True
            else:
                error(self.pos, "Base class '%s' not a struct or class." % base_class)
        base_class_types = filter(base_ok, [b.analyse(scope or env) for b in self.base_classes])
        self.entry = env.declare_cpp_class(
            self.name, scope, self.pos,
            self.cname, base_class_types, visibility=self.visibility, templates=template_types)
        if self.entry is None:
            return
        self.entry.is_cpp_class = 1
        if scope is not None:
            scope.type = self.entry.type
        defined_funcs = []
        def func_attributes(attributes):
            for attr in attributes:
                if isinstance(attr, CFuncDefNode):
                    yield attr
                elif isinstance(attr, CompilerDirectivesNode):
                    for sub_attr in func_attributes(attr.body.stats):
                        yield sub_attr
        if self.attributes is not None:
            if self.in_pxd and not env.in_cinclude:
                self.entry.defined_in_pxd = 1
            for attr in self.attributes:
                declare = getattr(attr, 'declare', None)
                if declare:
                    attr.declare(scope)
                attr.analyse_declarations(scope)
            for func in func_attributes(self.attributes):
                defined_funcs.append(func)
                if self.templates is not None:
                    func.template_declaration = "template <typename %s>" % ", typename ".join(template_names)
        self.body = StatListNode(self.pos, stats=defined_funcs)
        self.scope = scope

    def analyse_expressions(self, env):
        self.body = self.body.analyse_expressions(self.entry.type.scope)
        return self

    def generate_function_definitions(self, env, code):
        self.body.generate_function_definitions(self.entry.type.scope, code)

    def generate_execution_code(self, code):
        self.body.generate_execution_code(code)

    def annotate(self, code):
        self.body.annotate(code)


class CEnumDefNode(StatNode):
    #  name           string or None
    #  cname          string or None
    #  items          [CEnumDefItemNode]
    #  typedef_flag   boolean
    #  visibility     "public" or "private" or "extern"
    #  api            boolean
    #  in_pxd         boolean
    #  create_wrapper boolean
    #  entry          Entry

    child_attrs = ["items"]

    def declare(self, env):
         self.entry = env.declare_enum(
             self.name, self.pos,
             cname=self.cname, typedef_flag=self.typedef_flag,
             visibility=self.visibility, api=self.api,
             create_wrapper=self.create_wrapper)

    def analyse_declarations(self, env):
        if self.items is not None:
            if self.in_pxd and not env.in_cinclude:
                self.entry.defined_in_pxd = 1
            for item in self.items:
                item.analyse_declarations(env, self.entry)

    def analyse_expressions(self, env):
        return self

    def generate_execution_code(self, code):
        if self.visibility == 'public' or self.api:
            code.mark_pos(self.pos)
            temp = code.funcstate.allocate_temp(PyrexTypes.py_object_type, manage_ref=True)
            for item in self.entry.enum_values:
                code.putln("%s = PyInt_FromLong(%s); %s" % (
                    temp,
                    item.cname,
                    code.error_goto_if_null(temp, item.pos)))
                code.put_gotref(temp)
                code.putln('if (PyDict_SetItemString(%s, "%s", %s) < 0) %s' % (
                    Naming.moddict_cname,
                    item.name,
                    temp,
                    code.error_goto(item.pos)))
                code.put_decref_clear(temp, PyrexTypes.py_object_type)
            code.funcstate.release_temp(temp)


class CEnumDefItemNode(StatNode):
    #  name     string
    #  cname    string or None
    #  value    ExprNode or None

    child_attrs = ["value"]

    def analyse_declarations(self, env, enum_entry):
        if self.value:
            self.value = self.value.analyse_const_expression(env)
            if not self.value.type.is_int:
                self.value = self.value.coerce_to(PyrexTypes.c_int_type, env)
                self.value = self.value.analyse_const_expression(env)
        entry = env.declare_const(
            self.name, enum_entry.type,
            self.value, self.pos, cname=self.cname,
            visibility=enum_entry.visibility, api=enum_entry.api,
            create_wrapper=enum_entry.create_wrapper and enum_entry.name is None)
        enum_entry.enum_values.append(entry)
        if enum_entry.name:
            enum_entry.type.values.append(entry.name)


class CTypeDefNode(StatNode):
    #  base_type    CBaseTypeNode
    #  declarator   CDeclaratorNode
    #  visibility   "public" or "private"
    #  api          boolean
    #  in_pxd       boolean

    child_attrs = ["base_type", "declarator"]

    def analyse_declarations(self, env):
        base = self.base_type.analyse(env)
        name_declarator, type = self.declarator.analyse(
            base, env, visibility=self.visibility, in_pxd=self.in_pxd)
        name = name_declarator.name
        cname = name_declarator.cname

        entry = env.declare_typedef(
            name, type, self.pos,
            cname=cname, visibility=self.visibility, api=self.api)

        if type.is_fused:
            entry.in_cinclude = True

        if self.in_pxd and not env.in_cinclude:
            entry.defined_in_pxd = 1

    def analyse_expressions(self, env):
        return self

    def generate_execution_code(self, code):
        pass


class FuncDefNode(StatNode, BlockNode):
    #  Base class for function definition nodes.
    #
    #  return_type     PyrexType
    #  #filename        string        C name of filename string const
    #  entry           Symtab.Entry
    #  needs_closure   boolean        Whether or not this function has inner functions/classes/yield
    #  needs_outer_scope boolean      Whether or not this function requires outer scope
    #  pymethdef_required boolean     Force Python method struct generation
    #  directive_locals { string : ExprNode } locals defined by cython.locals(...)
    #  directive_returns [ExprNode] type defined by cython.returns(...)
    #  star_arg      PyArgDeclNode or None  * argument
    #  starstar_arg  PyArgDeclNode or None  ** argument
    #
    #  is_async_def  boolean          is a Coroutine function
    #
    #  has_fused_arguments  boolean
    #       Whether this cdef function has fused parameters. This is needed
    #       by AnalyseDeclarationsTransform, so it can replace CFuncDefNodes
    #       with fused argument types with a FusedCFuncDefNode

    py_func = None
    needs_closure = False
    needs_outer_scope = False
    pymethdef_required = False
    is_generator = False
    is_generator_body = False
    is_async_def = False
    modifiers = []
    has_fused_arguments = False
    star_arg = None
    starstar_arg = None
    is_cyfunction = False
    code_object = None

    def analyse_default_values(self, env):
        default_seen = 0
        for arg in self.args:
            if arg.default:
                default_seen = 1
                if arg.is_generic:
                    arg.default = arg.default.analyse_types(env)
                    arg.default = arg.default.coerce_to(arg.type, env)
                else:
                    error(arg.pos, "This argument cannot have a default value")
                    arg.default = None
            elif arg.kw_only:
                default_seen = 1
            elif default_seen:
                error(arg.pos, "Non-default argument following default argument")

    def analyse_annotation(self, env, annotation):
        # Annotations can not only contain valid Python expressions but arbitrary type references.
        if annotation is None:
            return None
        if not env.directives['annotation_typing'] or annotation.analyse_as_type(env) is None:
            annotation = annotation.analyse_types(env)
        return annotation

    def analyse_annotations(self, env):
        for arg in self.args:
            if arg.annotation:
                arg.annotation = self.analyse_annotation(env, arg.annotation)

    def align_argument_type(self, env, arg):
        # @cython.locals()
        directive_locals = self.directive_locals
        orig_type = arg.type
        if arg.name in directive_locals:
            type_node = directive_locals[arg.name]
            other_type = type_node.analyse_as_type(env)
        elif isinstance(arg, CArgDeclNode) and arg.annotation and env.directives['annotation_typing']:
            type_node = arg.annotation
            other_type = arg.inject_type_from_annotations(env)
            if other_type is None:
                return arg
        else:
            return arg
        if other_type is None:
            error(type_node.pos, "Not a type")
        elif orig_type is not py_object_type and not orig_type.same_as(other_type):
            error(arg.base_type.pos, "Signature does not agree with previous declaration")
            error(type_node.pos, "Previous declaration here")
        else:
            arg.type = other_type
        return arg

    def need_gil_acquisition(self, lenv):
        return 0

    def create_local_scope(self, env):
        genv = env
        while genv.is_py_class_scope or genv.is_c_class_scope:
            genv = genv.outer_scope
        if self.needs_closure:
            lenv = ClosureScope(name=self.entry.name,
                                outer_scope=genv,
                                parent_scope=env,
                                scope_name=self.entry.cname)
        else:
            lenv = LocalScope(name=self.entry.name,
                              outer_scope=genv,
                              parent_scope=env)
        lenv.return_type = self.return_type
        type = self.entry.type
        if type.is_cfunction:
            lenv.nogil = type.nogil and not type.with_gil
        self.local_scope = lenv
        lenv.directives = env.directives
        return lenv

    def generate_function_body(self, env, code):
        self.body.generate_execution_code(code)

    def generate_function_definitions(self, env, code):
        from . import Buffer
        if self.return_type.is_memoryviewslice:
            from . import MemoryView

        lenv = self.local_scope
        if lenv.is_closure_scope and not lenv.is_passthrough:
            outer_scope_cname = "%s->%s" % (Naming.cur_scope_cname,
                                            Naming.outer_scope_cname)
        else:
            outer_scope_cname = Naming.outer_scope_cname
        lenv.mangle_closure_cnames(outer_scope_cname)
        # Generate closure function definitions
        self.body.generate_function_definitions(lenv, code)
        # generate lambda function definitions
        self.generate_lambda_definitions(lenv, code)

        is_getbuffer_slot = (self.entry.name == "__getbuffer__" and
                             self.entry.scope.is_c_class_scope)
        is_releasebuffer_slot = (self.entry.name == "__releasebuffer__" and
                                 self.entry.scope.is_c_class_scope)
        is_buffer_slot = is_getbuffer_slot or is_releasebuffer_slot
        if is_buffer_slot:
            if 'cython_unused' not in self.modifiers:
                self.modifiers = self.modifiers + ['cython_unused']

        preprocessor_guard = self.get_preprocessor_guard()

        profile = code.globalstate.directives['profile']
        linetrace = code.globalstate.directives['linetrace']
        if profile or linetrace:
            code.globalstate.use_utility_code(
                UtilityCode.load_cached("Profile", "Profile.c"))

        # Generate C code for header and body of function
        code.enter_cfunc_scope(lenv)
        code.return_from_error_cleanup_label = code.new_label()
        code.funcstate.gil_owned = not lenv.nogil

        # ----- Top-level constants used by this function
        code.mark_pos(self.pos)
        self.generate_cached_builtins_decls(lenv, code)
        # ----- Function header
        code.putln("")

        if preprocessor_guard:
            code.putln(preprocessor_guard)

        with_pymethdef = (self.needs_assignment_synthesis(env, code) or
                          self.pymethdef_required)
        if self.py_func:
            self.py_func.generate_function_header(
                code, with_pymethdef=with_pymethdef, proto_only=True)
        self.generate_function_header(code, with_pymethdef=with_pymethdef)
        # ----- Local variable declarations
        # Find function scope
        cenv = env
        while cenv.is_py_class_scope or cenv.is_c_class_scope:
            cenv = cenv.outer_scope
        if self.needs_closure:
            code.put(lenv.scope_class.type.declaration_code(Naming.cur_scope_cname))
            code.putln(";")
        elif self.needs_outer_scope:
            if lenv.is_passthrough:
                code.put(lenv.scope_class.type.declaration_code(Naming.cur_scope_cname))
                code.putln(";")
            code.put(cenv.scope_class.type.declaration_code(Naming.outer_scope_cname))
            code.putln(";")
        self.generate_argument_declarations(lenv, code)

        for entry in lenv.var_entries:
            if not (entry.in_closure or entry.is_arg):
                code.put_var_declaration(entry)

        # Initialize the return variable __pyx_r
        init = ""
        if not self.return_type.is_void:
            if self.return_type.is_pyobject:
                init = " = NULL"
            elif self.return_type.is_memoryviewslice:
                init = ' = ' + MemoryView.memslice_entry_init

            code.putln("%s%s;" % (
                self.return_type.declaration_code(Naming.retval_cname),
                init))

        tempvardecl_code = code.insertion_point()
        self.generate_keyword_list(code)

        # ----- GIL acquisition
        acquire_gil = self.acquire_gil

        # See if we need to acquire the GIL for variable declarations, or for
        # refnanny only

        # Closures are not currently possible for cdef nogil functions,
        # but check them anyway
        have_object_args = self.needs_closure or self.needs_outer_scope
        for arg in lenv.arg_entries:
            if arg.type.is_pyobject:
                have_object_args = True
                break

        used_buffer_entries = [entry for entry in lenv.buffer_entries if entry.used]

        acquire_gil_for_var_decls_only = (
            lenv.nogil and lenv.has_with_gil_block and
            (have_object_args or used_buffer_entries))

        acquire_gil_for_refnanny_only = (
            lenv.nogil and lenv.has_with_gil_block and not
            acquire_gil_for_var_decls_only)

        use_refnanny = not lenv.nogil or lenv.has_with_gil_block

        if acquire_gil or acquire_gil_for_var_decls_only:
            code.put_ensure_gil()
            code.funcstate.gil_owned = True
        elif lenv.nogil and lenv.has_with_gil_block:
            code.declare_gilstate()

        if profile or linetrace:
            if not self.is_generator:
                # generators are traced when iterated, not at creation
                tempvardecl_code.put_trace_declarations()
                code_object = self.code_object.calculate_result_code(code) if self.code_object else None
                code.put_trace_frame_init(code_object)

        # ----- Special check for getbuffer
        if is_getbuffer_slot:
            self.getbuffer_check(code)

        # ----- set up refnanny
        if use_refnanny:
            tempvardecl_code.put_declare_refcount_context()
            code.put_setup_refcount_context(
                self.entry.name, acquire_gil=acquire_gil_for_refnanny_only)

        # ----- Automatic lead-ins for certain special functions
        if is_getbuffer_slot:
            self.getbuffer_init(code)
        # ----- Create closure scope object
        if self.needs_closure:
            tp_slot = TypeSlots.ConstructorSlot("tp_new", '__new__')
            slot_func_cname = TypeSlots.get_slot_function(lenv.scope_class.type.scope, tp_slot)
            if not slot_func_cname:
                slot_func_cname = '%s->tp_new' % lenv.scope_class.type.typeptr_cname
            code.putln("%s = (%s)%s(%s, %s, NULL);" % (
                Naming.cur_scope_cname,
                lenv.scope_class.type.empty_declaration_code(),
                slot_func_cname,
                lenv.scope_class.type.typeptr_cname,
                Naming.empty_tuple))
            code.putln("if (unlikely(!%s)) {" % Naming.cur_scope_cname)
            # Scope unconditionally DECREFed on return.
            code.putln("%s = %s;" % (
                Naming.cur_scope_cname,
                lenv.scope_class.type.cast_code("Py_None")))
            code.put_incref("Py_None", py_object_type)
            code.putln(code.error_goto(self.pos))
            code.putln("} else {")
            code.put_gotref(Naming.cur_scope_cname)
            code.putln("}")
            # Note that it is unsafe to decref the scope at this point.
        if self.needs_outer_scope:
            if self.is_cyfunction:
                code.putln("%s = (%s) __Pyx_CyFunction_GetClosure(%s);" % (
                    outer_scope_cname,
                    cenv.scope_class.type.empty_declaration_code(),
                    Naming.self_cname))
            else:
                code.putln("%s = (%s) %s;" % (
                    outer_scope_cname,
                    cenv.scope_class.type.empty_declaration_code(),
                    Naming.self_cname))
            if lenv.is_passthrough:
                code.putln("%s = %s;" % (Naming.cur_scope_cname, outer_scope_cname))
            elif self.needs_closure:
                # inner closures own a reference to their outer parent
                code.put_incref(outer_scope_cname, cenv.scope_class.type)
                code.put_giveref(outer_scope_cname)
        # ----- Trace function call
        if profile or linetrace:
            # this looks a bit late, but if we don't get here due to a
            # fatal error before hand, it's not really worth tracing
            if not self.is_generator:
                # generators are traced when iterated, not at creation
                if self.is_wrapper:
                    trace_name = self.entry.name + " (wrapper)"
                else:
                    trace_name = self.entry.name
                code.put_trace_call(
                    trace_name, self.pos, nogil=not code.funcstate.gil_owned)
            code.funcstate.can_trace = True
        # ----- Fetch arguments
        self.generate_argument_parsing_code(env, code)
        # If an argument is assigned to in the body, we must
        # incref it to properly keep track of refcounts.
        is_cdef = isinstance(self, CFuncDefNode)
        for entry in lenv.arg_entries:
            if entry.type.is_pyobject:
                if (acquire_gil or len(entry.cf_assignments) > 1) and not entry.in_closure:
                    code.put_var_incref(entry)

            # Note: defaults are always incref-ed. For def functions, we
            #       we acquire arguments from object conversion, so we have
            #       new references. If we are a cdef function, we need to
            #       incref our arguments
            elif is_cdef and entry.type.is_memoryviewslice and len(entry.cf_assignments) > 1:
                code.put_incref_memoryviewslice(entry.cname, have_gil=code.funcstate.gil_owned)
        for entry in lenv.var_entries:
            if entry.is_arg and len(entry.cf_assignments) > 1 and not entry.in_closure:
                if entry.xdecref_cleanup:
                    code.put_var_xincref(entry)
                else:
                    code.put_var_incref(entry)

        # ----- Initialise local buffer auxiliary variables
        for entry in lenv.var_entries + lenv.arg_entries:
            if entry.type.is_buffer and entry.buffer_aux.buflocal_nd_var.used:
                Buffer.put_init_vars(entry, code)

        # ----- Check and convert arguments
        self.generate_argument_type_tests(code)
        # ----- Acquire buffer arguments
        for entry in lenv.arg_entries:
            if entry.type.is_buffer:
                Buffer.put_acquire_arg_buffer(entry, code, self.pos)

        if acquire_gil_for_var_decls_only:
            code.put_release_ensured_gil()
            code.funcstate.gil_owned = False

        # -------------------------
        # ----- Function body -----
        # -------------------------
        self.generate_function_body(env, code)

        code.mark_pos(self.pos, trace=False)
        code.putln("")
        code.putln("/* function exit code */")

        # ----- Default return value
        if not self.body.is_terminator:
            if self.return_type.is_pyobject:
                #if self.return_type.is_extension_type:
                #    lhs = "(PyObject *)%s" % Naming.retval_cname
                #else:
                lhs = Naming.retval_cname
                code.put_init_to_py_none(lhs, self.return_type)
            else:
                val = self.return_type.default_value
                if val:
                    code.putln("%s = %s;" % (Naming.retval_cname, val))
                elif not self.return_type.is_void:
                    code.putln("__Pyx_pretend_to_initialize(&%s);" % Naming.retval_cname)
        # ----- Error cleanup
        if code.error_label in code.labels_used:
            if not self.body.is_terminator:
                code.put_goto(code.return_label)
            code.put_label(code.error_label)
            for cname, type in code.funcstate.all_managed_temps():
                code.put_xdecref(cname, type, have_gil=not lenv.nogil)

            # Clean up buffers -- this calls a Python function
            # so need to save and restore error state
            buffers_present = len(used_buffer_entries) > 0
            #memslice_entries = [e for e in lenv.entries.values() if e.type.is_memoryviewslice]
            if buffers_present:
                code.globalstate.use_utility_code(restore_exception_utility_code)
                code.putln("{ PyObject *__pyx_type, *__pyx_value, *__pyx_tb;")
                code.putln("__Pyx_PyThreadState_declare")
                code.putln("__Pyx_PyThreadState_assign")
                code.putln("__Pyx_ErrFetch(&__pyx_type, &__pyx_value, &__pyx_tb);")
                for entry in used_buffer_entries:
                    Buffer.put_release_buffer_code(code, entry)
                    #code.putln("%s = 0;" % entry.cname)
                code.putln("__Pyx_ErrRestore(__pyx_type, __pyx_value, __pyx_tb);}")

            if self.return_type.is_memoryviewslice:
                MemoryView.put_init_entry(Naming.retval_cname, code)
                err_val = Naming.retval_cname
            else:
                err_val = self.error_value()

            exc_check = self.caller_will_check_exceptions()
            if err_val is not None or exc_check:
                # TODO: Fix exception tracing (though currently unused by cProfile).
                # code.globalstate.use_utility_code(get_exception_tuple_utility_code)
                # code.put_trace_exception()

                if lenv.nogil and not lenv.has_with_gil_block:
                    code.putln("{")
                    code.put_ensure_gil()

                code.put_add_traceback(self.entry.qualified_name)

                if lenv.nogil and not lenv.has_with_gil_block:
                    code.put_release_ensured_gil()
                    code.putln("}")
            else:
                warning(self.entry.pos,
                        "Unraisable exception in function '%s'." %
                        self.entry.qualified_name, 0)
                code.put_unraisable(self.entry.qualified_name, lenv.nogil)
            default_retval = self.return_type.default_value
            if err_val is None and default_retval:
                err_val = default_retval
            if err_val is not None:
                if err_val != Naming.retval_cname:
                    code.putln("%s = %s;" % (Naming.retval_cname, err_val))
            elif not self.return_type.is_void:
                code.putln("__Pyx_pretend_to_initialize(&%s);" % Naming.retval_cname)

            if is_getbuffer_slot:
                self.getbuffer_error_cleanup(code)

            # If we are using the non-error cleanup section we should
            # jump past it if we have an error. The if-test below determine
            # whether this section is used.
            if buffers_present or is_getbuffer_slot or self.return_type.is_memoryviewslice:
                code.put_goto(code.return_from_error_cleanup_label)

        # ----- Non-error return cleanup
        code.put_label(code.return_label)
        for entry in used_buffer_entries:
            Buffer.put_release_buffer_code(code, entry)
        if is_getbuffer_slot:
            self.getbuffer_normal_cleanup(code)

        if self.return_type.is_memoryviewslice:
            # See if our return value is uninitialized on non-error return
            # from . import MemoryView
            # MemoryView.err_if_nogil_initialized_check(self.pos, env)
            cond = code.unlikely(self.return_type.error_condition(Naming.retval_cname))
            code.putln(
                'if (%s) {' % cond)
            if env.nogil:
                code.put_ensure_gil()
            code.putln(
                'PyErr_SetString(PyExc_TypeError, "Memoryview return value is not initialized");')
            if env.nogil:
                code.put_release_ensured_gil()
            code.putln(
                '}')

        # ----- Return cleanup for both error and no-error return
        code.put_label(code.return_from_error_cleanup_label)

        for entry in lenv.var_entries:
            if not entry.used or entry.in_closure:
                continue

            if entry.type.is_memoryviewslice:
                code.put_xdecref_memoryviewslice(entry.cname, have_gil=not lenv.nogil)
            elif entry.type.is_pyobject:
                if not entry.is_arg or len(entry.cf_assignments) > 1:
                    if entry.xdecref_cleanup:
                        code.put_var_xdecref(entry)
                    else:
                        code.put_var_decref(entry)

        # Decref any increfed args
        for entry in lenv.arg_entries:
            if entry.type.is_pyobject:
                if (acquire_gil or len(entry.cf_assignments) > 1) and not entry.in_closure:
                    code.put_var_decref(entry)
            elif (entry.type.is_memoryviewslice and
                  (not is_cdef or len(entry.cf_assignments) > 1)):
                # decref slices of def functions and acquired slices from cdef
                # functions, but not borrowed slices from cdef functions.
                code.put_xdecref_memoryviewslice(entry.cname,
                                                 have_gil=not lenv.nogil)
        if self.needs_closure:
            code.put_decref(Naming.cur_scope_cname, lenv.scope_class.type)

        # ----- Return
        # This code is duplicated in ModuleNode.generate_module_init_func
        if not lenv.nogil:
            default_retval = self.return_type.default_value
            err_val = self.error_value()
            if err_val is None and default_retval:
                err_val = default_retval  # FIXME: why is err_val not used?
            if self.return_type.is_pyobject:
                code.put_xgiveref(self.return_type.as_pyobject(Naming.retval_cname))

        if self.entry.is_special and self.entry.name == "__hash__":
            # Returning -1 for __hash__ is supposed to signal an error
            # We do as Python instances and coerce -1 into -2.
            code.putln("if (unlikely(%s == -1) && !PyErr_Occurred()) %s = -2;" % (
                Naming.retval_cname, Naming.retval_cname))

        if profile or linetrace:
            code.funcstate.can_trace = False
            if not self.is_generator:
                # generators are traced when iterated, not at creation
                if self.return_type.is_pyobject:
                    code.put_trace_return(
                        Naming.retval_cname, nogil=not code.funcstate.gil_owned)
                else:
                    code.put_trace_return(
                        "Py_None", nogil=not code.funcstate.gil_owned)

        if not lenv.nogil:
            # GIL holding function
            code.put_finish_refcount_context()

        if acquire_gil or (lenv.nogil and lenv.has_with_gil_block):
            # release the GIL (note that with-gil blocks acquire it on exit in their EnsureGILNode)
            code.put_release_ensured_gil()
            code.funcstate.gil_owned = False

        if not self.return_type.is_void:
            code.putln("return %s;" % Naming.retval_cname)

        code.putln("}")

        if preprocessor_guard:
            code.putln("#endif /*!(%s)*/" % preprocessor_guard)

        # ----- Go back and insert temp variable declarations
        tempvardecl_code.put_temp_declarations(code.funcstate)

        # ----- Python version
        code.exit_cfunc_scope()
        if self.py_func:
            self.py_func.generate_function_definitions(env, code)
        self.generate_wrapper_functions(code)

    def declare_argument(self, env, arg):
        if arg.type.is_void:
            error(arg.pos, "Invalid use of 'void'")
        elif not arg.type.is_complete() and not (arg.type.is_array or arg.type.is_memoryviewslice):
            error(arg.pos, "Argument type '%s' is incomplete" % arg.type)
        entry = env.declare_arg(arg.name, arg.type, arg.pos)
        if arg.annotation:
            entry.annotation = arg.annotation
        return entry

    def generate_arg_type_test(self, arg, code):
        # Generate type test for one argument.
        if arg.type.typeobj_is_available():
            code.globalstate.use_utility_code(
                UtilityCode.load_cached("ArgTypeTest", "FunctionArguments.c"))
            typeptr_cname = arg.type.typeptr_cname
            arg_code = "((PyObject *)%s)" % arg.entry.cname
            code.putln(
                'if (unlikely(!__Pyx_ArgTypeTest(%s, %s, %d, "%s", %s))) %s' % (
                    arg_code,
                    typeptr_cname,
                    arg.accept_none,
                    arg.name,
                    arg.type.is_builtin_type and arg.type.require_exact,
                    code.error_goto(arg.pos)))
        else:
            error(arg.pos, "Cannot test type of extern C class without type object name specification")

    def generate_arg_none_check(self, arg, code):
        # Generate None check for one argument.
        if arg.type.is_memoryviewslice:
            cname = "%s.memview" % arg.entry.cname
        else:
            cname = arg.entry.cname

        code.putln('if (unlikely(((PyObject *)%s) == Py_None)) {' % cname)
        code.putln('''PyErr_Format(PyExc_TypeError, "Argument '%%.%ds' must not be None", "%s"); %s''' % (
            max(200, len(arg.name)), arg.name,
            code.error_goto(arg.pos)))
        code.putln('}')

    def generate_wrapper_functions(self, code):
        pass

    def generate_execution_code(self, code):
        code.mark_pos(self.pos)
        # Evaluate and store argument default values
        for arg in self.args:
            if not arg.is_dynamic:
                arg.generate_assignment_code(code)

    #
    # Special code for the __getbuffer__ function
    #
    def _get_py_buffer_info(self):
        py_buffer = self.local_scope.arg_entries[1]
        try:
            # Check builtin definition of struct Py_buffer
            obj_type = py_buffer.type.base_type.scope.entries['obj'].type
        except (AttributeError, KeyError):
            # User code redeclared struct Py_buffer
            obj_type = None
        return py_buffer, obj_type

    # Old Python 3 used to support write-locks on buffer-like objects by
    # calling PyObject_GetBuffer() with a view==NULL parameter. This obscure
    # feature is obsolete, it was almost never used (only one instance in
    # `Modules/posixmodule.c` in Python 3.1) and it is now officially removed
    # (see bpo-14203). We add an extra check here to prevent legacy code from
    # from trying to use the feature and prevent segmentation faults.
    def getbuffer_check(self, code):
        py_buffer, _ = self._get_py_buffer_info()
        view = py_buffer.cname
        code.putln("if (%s == NULL) {" % view)
        code.putln("PyErr_SetString(PyExc_BufferError, "
                   "\"PyObject_GetBuffer: view==NULL argument is obsolete\");")
        code.putln("return -1;")
        code.putln("}")

    def getbuffer_init(self, code):
        py_buffer, obj_type = self._get_py_buffer_info()
        view = py_buffer.cname
        if obj_type and obj_type.is_pyobject:
            code.put_init_to_py_none("%s->obj" % view, obj_type)
            code.put_giveref("%s->obj" % view) # Do not refnanny object within structs
        else:
            code.putln("%s->obj = NULL;" % view)

    def getbuffer_error_cleanup(self, code):
        py_buffer, obj_type = self._get_py_buffer_info()
        view = py_buffer.cname
        if obj_type and obj_type.is_pyobject:
            code.putln("if (%s->obj != NULL) {" % view)
            code.put_gotref("%s->obj" % view)
            code.put_decref_clear("%s->obj" % view, obj_type)
            code.putln("}")
        else:
            code.putln("Py_CLEAR(%s->obj);" % view)

    def getbuffer_normal_cleanup(self, code):
        py_buffer, obj_type = self._get_py_buffer_info()
        view = py_buffer.cname
        if obj_type and obj_type.is_pyobject:
            code.putln("if (%s->obj == Py_None) {" % view)
            code.put_gotref("%s->obj" % view)
            code.put_decref_clear("%s->obj" % view, obj_type)
            code.putln("}")

    def get_preprocessor_guard(self):
        if not self.entry.is_special:
            return None
        name = self.entry.name
        slot = TypeSlots.method_name_to_slot.get(name)
        if not slot:
            return None
        if name == '__long__' and not self.entry.scope.lookup_here('__int__'):
            return None
        if name in ("__getbuffer__", "__releasebuffer__") and self.entry.scope.is_c_class_scope:
            return None
        return slot.preprocessor_guard_code()


class CFuncDefNode(FuncDefNode):
    #  C function definition.
    #
    #  modifiers     ['inline']
    #  visibility    'private' or 'public' or 'extern'
    #  base_type     CBaseTypeNode
    #  declarator    CDeclaratorNode
    #  cfunc_declarator  the CFuncDeclarator of this function
    #                    (this is also available through declarator or a
    #                     base thereof)
    #  body          StatListNode
    #  api           boolean
    #  decorators    [DecoratorNode]        list of decorators
    #
    #  with_gil      boolean    Acquire GIL around body
    #  type          CFuncType
    #  py_func       wrapper for calling from Python
    #  overridable   whether or not this is a cpdef function
    #  inline_in_pxd whether this is an inline function in a pxd file
    #  template_declaration  String or None   Used for c++ class methods
    #  is_const_method whether this is a const method
    #  is_static_method whether this is a static method
    #  is_c_class_method whether this is a cclass method

    child_attrs = ["base_type", "declarator", "body", "py_func_stat"]

    inline_in_pxd = False
    decorators = None
    directive_locals = None
    directive_returns = None
    override = None
    template_declaration = None
    is_const_method = False
    py_func_stat = None

    def unqualified_name(self):
        return self.entry.name

    @property
    def code_object(self):
        # share the CodeObject with the cpdef wrapper (if available)
        return self.py_func.code_object if self.py_func else None

    def analyse_declarations(self, env):
        self.is_c_class_method = env.is_c_class_scope
        if self.directive_locals is None:
            self.directive_locals = {}
        self.directive_locals.update(env.directives.get('locals', {}))
        if self.directive_returns is not None:
            base_type = self.directive_returns.analyse_as_type(env)
            if base_type is None:
                error(self.directive_returns.pos, "Not a type")
                base_type = PyrexTypes.error_type
        else:
            base_type = self.base_type.analyse(env)
        self.is_static_method = 'staticmethod' in env.directives and not env.lookup_here('staticmethod')
        # The 2 here is because we need both function and argument names.
        if isinstance(self.declarator, CFuncDeclaratorNode):
            name_declarator, type = self.declarator.analyse(
                base_type, env, nonempty=2 * (self.body is not None),
                directive_locals=self.directive_locals, visibility=self.visibility)
        else:
            name_declarator, type = self.declarator.analyse(
                base_type, env, nonempty=2 * (self.body is not None), visibility=self.visibility)
        if not type.is_cfunction:
            error(self.pos, "Suite attached to non-function declaration")
        # Remember the actual type according to the function header
        # written here, because the type in the symbol table entry
        # may be different if we're overriding a C method inherited
        # from the base type of an extension type.
        self.type = type
        type.is_overridable = self.overridable
        declarator = self.declarator
        while not hasattr(declarator, 'args'):
            declarator = declarator.base

        self.cfunc_declarator = declarator
        self.args = declarator.args

        opt_arg_count = self.cfunc_declarator.optional_arg_count
        if (self.visibility == 'public' or self.api) and opt_arg_count:
            error(self.cfunc_declarator.pos,
                  "Function with optional arguments may not be declared public or api")

        if type.exception_check == '+' and self.visibility != 'extern':
            warning(self.cfunc_declarator.pos,
                    "Only extern functions can throw C++ exceptions.")

        for formal_arg, type_arg in zip(self.args, type.args):
            self.align_argument_type(env, type_arg)
            formal_arg.type = type_arg.type
            formal_arg.name = type_arg.name
            formal_arg.cname = type_arg.cname

            self._validate_type_visibility(type_arg.type, type_arg.pos, env)

            if type_arg.type.is_fused:
                self.has_fused_arguments = True

            if type_arg.type.is_buffer and 'inline' in self.modifiers:
                warning(formal_arg.pos, "Buffer unpacking not optimized away.", 1)

            if type_arg.type.is_buffer or type_arg.type.is_pythran_expr:
                if self.type.nogil:
                    error(formal_arg.pos,
                          "Buffer may not be acquired without the GIL. Consider using memoryview slices instead.")
                elif 'inline' in self.modifiers:
                    warning(formal_arg.pos, "Buffer unpacking not optimized away.", 1)

        self._validate_type_visibility(type.return_type, self.pos, env)

        name = name_declarator.name
        cname = name_declarator.cname

        type.is_const_method = self.is_const_method
        type.is_static_method = self.is_static_method
        self.entry = env.declare_cfunction(
            name, type, self.pos,
            cname=cname, visibility=self.visibility, api=self.api,
            defining=self.body is not None, modifiers=self.modifiers,
            overridable=self.overridable)
        self.entry.inline_func_in_pxd = self.inline_in_pxd
        self.return_type = type.return_type
        if self.return_type.is_array and self.visibility != 'extern':
            error(self.pos, "Function cannot return an array")
        if self.return_type.is_cpp_class:
            self.return_type.check_nullary_constructor(self.pos, "used as a return value")

        if self.overridable and not env.is_module_scope and not self.is_static_method:
            if len(self.args) < 1 or not self.args[0].type.is_pyobject:
                # An error will be produced in the cdef function
                self.overridable = False

        self.declare_cpdef_wrapper(env)
        self.create_local_scope(env)

    def declare_cpdef_wrapper(self, env):
        if self.overridable:
            if self.is_static_method:
                # TODO(robertwb): Finish this up, perhaps via more function refactoring.
                error(self.pos, "static cpdef methods not yet supported")
            name = self.entry.name
            py_func_body = self.call_self_node(is_module_scope=env.is_module_scope)
            if self.is_static_method:
                from .ExprNodes import NameNode
                decorators = [DecoratorNode(self.pos, decorator=NameNode(self.pos, name='staticmethod'))]
                decorators[0].decorator.analyse_types(env)
            else:
                decorators = []
            self.py_func = DefNode(pos=self.pos,
                                   name=self.entry.name,
                                   args=self.args,
                                   star_arg=None,
                                   starstar_arg=None,
                                   doc=self.doc,
                                   body=py_func_body,
                                   decorators=decorators,
                                   is_wrapper=1)
            self.py_func.is_module_scope = env.is_module_scope
            self.py_func.analyse_declarations(env)
            self.py_func.entry.is_overridable = True
            self.py_func_stat = StatListNode(self.pos, stats=[self.py_func])
            self.py_func.type = PyrexTypes.py_object_type
            self.entry.as_variable = self.py_func.entry
            self.entry.used = self.entry.as_variable.used = True
            # Reset scope entry the above cfunction
            env.entries[name] = self.entry
            if (not self.entry.is_final_cmethod and
                    (not env.is_module_scope or Options.lookup_module_cpdef)):
                self.override = OverrideCheckNode(self.pos, py_func=self.py_func)
                self.body = StatListNode(self.pos, stats=[self.override, self.body])

    def _validate_type_visibility(self, type, pos, env):
        """
        Ensure that types used in cdef functions are public or api, or
        defined in a C header.
        """
        public_or_api = (self.visibility == 'public' or self.api)
        entry = getattr(type, 'entry', None)
        if public_or_api and entry and env.is_module_scope:
            if not (entry.visibility in ('public', 'extern') or
                    entry.api or entry.in_cinclude):
                error(pos, "Function declared public or api may not have private types")

    def call_self_node(self, omit_optional_args=0, is_module_scope=0):
        from . import ExprNodes
        args = self.type.args
        if omit_optional_args:
            args = args[:len(args) - self.type.optional_arg_count]
        arg_names = [arg.name for arg in args]
        if is_module_scope:
            cfunc = ExprNodes.NameNode(self.pos, name=self.entry.name)
            call_arg_names = arg_names
            skip_dispatch = Options.lookup_module_cpdef
        elif self.type.is_static_method:
            class_entry = self.entry.scope.parent_type.entry
            class_node = ExprNodes.NameNode(self.pos, name=class_entry.name)
            class_node.entry = class_entry
            cfunc = ExprNodes.AttributeNode(self.pos, obj=class_node, attribute=self.entry.name)
            # Calling static c(p)def methods on an instance disallowed.
            # TODO(robertwb): Support by passing self to check for override?
            skip_dispatch = True
        else:
            type_entry = self.type.args[0].type.entry
            type_arg = ExprNodes.NameNode(self.pos, name=type_entry.name)
            type_arg.entry = type_entry
            cfunc = ExprNodes.AttributeNode(self.pos, obj=type_arg, attribute=self.entry.name)
        skip_dispatch = not is_module_scope or Options.lookup_module_cpdef
        c_call = ExprNodes.SimpleCallNode(
            self.pos,
            function=cfunc,
            args=[ExprNodes.NameNode(self.pos, name=n) for n in arg_names],
            wrapper_call=skip_dispatch)
        return ReturnStatNode(pos=self.pos, return_type=PyrexTypes.py_object_type, value=c_call)

    def declare_arguments(self, env):
        for arg in self.type.args:
            if not arg.name:
                error(arg.pos, "Missing argument name")
            self.declare_argument(env, arg)

    def need_gil_acquisition(self, lenv):
        return self.type.with_gil

    def nogil_check(self, env):
        type = self.type
        with_gil = type.with_gil
        if type.nogil and not with_gil:
            if type.return_type.is_pyobject:
                error(self.pos,
                      "Function with Python return type cannot be declared nogil")
            for entry in self.local_scope.var_entries:
                if entry.type.is_pyobject and not entry.in_with_gil_block:
                    error(self.pos, "Function declared nogil has Python locals or temporaries")

    def analyse_expressions(self, env):
        self.local_scope.directives = env.directives
        if self.py_func_stat is not None:
            # this will also analyse the default values and the function name assignment
            self.py_func_stat = self.py_func_stat.analyse_expressions(env)
        elif self.py_func is not None:
            # this will also analyse the default values
            self.py_func = self.py_func.analyse_expressions(env)
        else:
            self.analyse_default_values(env)
            self.analyse_annotations(env)
        self.acquire_gil = self.need_gil_acquisition(self.local_scope)
        return self

    def needs_assignment_synthesis(self, env, code=None):
        return False

    def generate_function_header(self, code, with_pymethdef, with_opt_args=1, with_dispatch=1, cname=None):
        scope = self.local_scope
        arg_decls = []
        type = self.type
        for arg in type.args[:len(type.args)-type.optional_arg_count]:
            arg_decl = arg.declaration_code()
            entry = scope.lookup(arg.name)
            if not entry.cf_used:
                arg_decl = 'CYTHON_UNUSED %s' % arg_decl
            arg_decls.append(arg_decl)
        if with_dispatch and self.overridable:
            dispatch_arg = PyrexTypes.c_int_type.declaration_code(
                Naming.skip_dispatch_cname)
            if self.override:
                arg_decls.append(dispatch_arg)
            else:
                arg_decls.append('CYTHON_UNUSED %s' % dispatch_arg)
        if type.optional_arg_count and with_opt_args:
            arg_decls.append(type.op_arg_struct.declaration_code(Naming.optional_args_cname))
        if type.has_varargs:
            arg_decls.append("...")
        if not arg_decls:
            arg_decls = ["void"]
        if cname is None:
            cname = self.entry.func_cname
        entity = type.function_header_code(cname, ', '.join(arg_decls))
        if self.entry.visibility == 'private' and '::' not in cname:
            storage_class = "static "
        else:
            storage_class = ""
        dll_linkage = None
        modifiers = code.build_function_modifiers(self.entry.func_modifiers)

        header = self.return_type.declaration_code(entity, dll_linkage=dll_linkage)
        #print (storage_class, modifiers, header)
        needs_proto = self.is_c_class_method
        if self.template_declaration:
            if needs_proto:
                code.globalstate.parts['module_declarations'].putln(self.template_declaration)
            code.putln(self.template_declaration)
        if needs_proto:
            code.globalstate.parts['module_declarations'].putln(
                "%s%s%s; /* proto*/" % (storage_class, modifiers, header))
        code.putln("%s%s%s {" % (storage_class, modifiers, header))

    def generate_argument_declarations(self, env, code):
        scope = self.local_scope
        for arg in self.args:
            if arg.default:
                entry = scope.lookup(arg.name)
                if self.override or entry.cf_used:
                    result = arg.calculate_default_value_code(code)
                    code.putln('%s = %s;' % (
                        arg.type.declaration_code(arg.cname), result))

    def generate_keyword_list(self, code):
        pass

    def generate_argument_parsing_code(self, env, code):
        i = 0
        used = 0
        scope = self.local_scope
        if self.type.optional_arg_count:
            code.putln('if (%s) {' % Naming.optional_args_cname)
            for arg in self.args:
                if arg.default:
                    entry = scope.lookup(arg.name)
                    if self.override or entry.cf_used:
                        code.putln('if (%s->%sn > %s) {' %
                                   (Naming.optional_args_cname,
                                    Naming.pyrex_prefix, i))
                        declarator = arg.declarator
                        while not hasattr(declarator, 'name'):
                            declarator = declarator.base
                        code.putln('%s = %s->%s;' %
                                   (arg.cname, Naming.optional_args_cname,
                                    self.type.opt_arg_cname(declarator.name)))
                        used += 1
                    i += 1
            for _ in range(used):
                code.putln('}')
            code.putln('}')

        # Move arguments into closure if required
        def put_into_closure(entry):
            if entry.in_closure and not arg.default:
                code.putln('%s = %s;' % (entry.cname, entry.original_cname))
                code.put_var_incref(entry)
                code.put_var_giveref(entry)
        for arg in self.args:
            put_into_closure(scope.lookup_here(arg.name))


    def generate_argument_conversion_code(self, code):
        pass

    def generate_argument_type_tests(self, code):
        # Generate type tests for args whose type in a parent
        # class is a supertype of the declared type.
        for arg in self.type.args:
            if arg.needs_type_test:
                self.generate_arg_type_test(arg, code)
            elif arg.type.is_pyobject and not arg.accept_none:
                self.generate_arg_none_check(arg, code)

    def generate_execution_code(self, code):
        if code.globalstate.directives['linetrace']:
            code.mark_pos(self.pos)
            code.putln("")  # generate line tracing code
        super(CFuncDefNode, self).generate_execution_code(code)
        if self.py_func_stat:
            self.py_func_stat.generate_execution_code(code)

    def error_value(self):
        if self.return_type.is_pyobject:
            return "0"
        else:
            #return None
            return self.entry.type.exception_value

    def caller_will_check_exceptions(self):
        return self.entry.type.exception_check

    def generate_wrapper_functions(self, code):
        # If the C signature of a function has changed, we need to generate
        # wrappers to put in the slots here.
        k = 0
        entry = self.entry
        func_type = entry.type
        while entry.prev_entry is not None:
            k += 1
            entry = entry.prev_entry
            entry.func_cname = "%s%swrap_%s" % (self.entry.func_cname, Naming.pyrex_prefix, k)
            code.putln()
            self.generate_function_header(
                code, 0,
                with_dispatch=entry.type.is_overridable,
                with_opt_args=entry.type.optional_arg_count,
                cname=entry.func_cname)
            if not self.return_type.is_void:
                code.put('return ')
            args = self.type.args
            arglist = [arg.cname for arg in args[:len(args)-self.type.optional_arg_count]]
            if entry.type.is_overridable:
                arglist.append(Naming.skip_dispatch_cname)
            elif func_type.is_overridable:
                arglist.append('0')
            if entry.type.optional_arg_count:
                arglist.append(Naming.optional_args_cname)
            elif func_type.optional_arg_count:
                arglist.append('NULL')
            code.putln('%s(%s);' % (self.entry.func_cname, ', '.join(arglist)))
            code.putln('}')


class PyArgDeclNode(Node):
    # Argument which must be a Python object (used
    # for * and ** arguments).
    #
    # name        string
    # entry       Symtab.Entry
    # annotation  ExprNode or None   Py3 argument annotation
    child_attrs = []
    is_self_arg = False
    is_type_arg = False

    def generate_function_definitions(self, env, code):
        self.entry.generate_function_definitions(env, code)


class DecoratorNode(Node):
    # A decorator
    #
    # decorator    NameNode or CallNode or AttributeNode
    child_attrs = ['decorator']


class DefNode(FuncDefNode):
    # A Python function definition.
    #
    # name          string                 the Python name of the function
    # lambda_name   string                 the internal name of a lambda 'function'
    # decorators    [DecoratorNode]        list of decorators
    # args          [CArgDeclNode]         formal arguments
    # doc           EncodedString or None
    # body          StatListNode
    # return_type_annotation
    #               ExprNode or None       the Py3 return type annotation
    #
    #  The following subnode is constructed internally
    #  when the def statement is inside a Python class definition.
    #
    #  fused_py_func        DefNode     The original fused cpdef DefNode
    #                                   (in case this is a specialization)
    #  specialized_cpdefs   [DefNode]   list of specialized cpdef DefNodes
    #  py_cfunc_node  PyCFunctionNode/InnerFunctionNode   The PyCFunction to create and assign
    #
    # decorator_indirection IndirectionNode Used to remove __Pyx_Method_ClassMethod for fused functions

    child_attrs = ["args", "star_arg", "starstar_arg", "body", "decorators", "return_type_annotation"]
    outer_attrs = ["decorators", "return_type_annotation"]

    is_staticmethod = False
    is_classmethod = False

    lambda_name = None
    reqd_kw_flags_cname = "0"
    is_wrapper = 0
    no_assignment_synthesis = 0
    decorators = None
    return_type_annotation = None
    entry = None
    acquire_gil = 0
    self_in_stararg = 0
    py_cfunc_node = None
    requires_classobj = False
    defaults_struct = None # Dynamic kwrds structure name
    doc = None

    fused_py_func = False
    specialized_cpdefs = None
    py_wrapper = None
    py_wrapper_required = True
    func_cname = None

    defaults_getter = None

    def __init__(self, pos, **kwds):
        FuncDefNode.__init__(self, pos, **kwds)
        k = rk = r = 0
        for arg in self.args:
            if arg.kw_only:
                k += 1
                if not arg.default:
                    rk += 1
            if not arg.default:
                r += 1
        self.num_kwonly_args = k
        self.num_required_kw_args = rk
        self.num_required_args = r

    def as_cfunction(self, cfunc=None, scope=None, overridable=True, returns=None, except_val=None, modifiers=None,
                     nogil=False, with_gil=False):
        if self.star_arg:
            error(self.star_arg.pos, "cdef function cannot have star argument")
        if self.starstar_arg:
            error(self.starstar_arg.pos, "cdef function cannot have starstar argument")
        exception_value, exception_check = except_val or (None, False)

        if cfunc is None:
            cfunc_args = []
            for formal_arg in self.args:
                name_declarator, type = formal_arg.analyse(scope, nonempty=1)
                cfunc_args.append(PyrexTypes.CFuncTypeArg(name=name_declarator.name,
                                                          cname=None,
                                                          annotation=formal_arg.annotation,
                                                          type=py_object_type,
                                                          pos=formal_arg.pos))
            cfunc_type = PyrexTypes.CFuncType(return_type=py_object_type,
                                              args=cfunc_args,
                                              has_varargs=False,
                                              exception_value=None,
                                              exception_check=exception_check,
                                              nogil=nogil,
                                              with_gil=with_gil,
                                              is_overridable=overridable)
            cfunc = CVarDefNode(self.pos, type=cfunc_type)
        else:
            if scope is None:
                scope = cfunc.scope
            cfunc_type = cfunc.type
            if len(self.args) != len(cfunc_type.args) or cfunc_type.has_varargs:
                error(self.pos, "wrong number of arguments")
                error(cfunc.pos, "previous declaration here")
            for i, (formal_arg, type_arg) in enumerate(zip(self.args, cfunc_type.args)):
                name_declarator, type = formal_arg.analyse(scope, nonempty=1,
                                                           is_self_arg=(i == 0 and scope.is_c_class_scope))
                if type is None or type is PyrexTypes.py_object_type:
                    formal_arg.type = type_arg.type
                    formal_arg.name_declarator = name_declarator

        if exception_value is None and cfunc_type.exception_value is not None:
            from .ExprNodes import ConstNode
            exception_value = ConstNode(
                self.pos, value=cfunc_type.exception_value, type=cfunc_type.return_type)
        declarator = CFuncDeclaratorNode(self.pos,
                                         base=CNameDeclaratorNode(self.pos, name=self.name, cname=None),
                                         args=self.args,
                                         has_varargs=False,
                                         exception_check=cfunc_type.exception_check,
                                         exception_value=exception_value,
                                         with_gil=cfunc_type.with_gil,
                                         nogil=cfunc_type.nogil)
        return CFuncDefNode(self.pos,
                            modifiers=modifiers or [],
                            base_type=CAnalysedBaseTypeNode(self.pos, type=cfunc_type.return_type),
                            declarator=declarator,
                            body=self.body,
                            doc=self.doc,
                            overridable=cfunc_type.is_overridable,
                            type=cfunc_type,
                            with_gil=cfunc_type.with_gil,
                            nogil=cfunc_type.nogil,
                            visibility='private',
                            api=False,
                            directive_locals=getattr(cfunc, 'directive_locals', {}),
                            directive_returns=returns)

    def is_cdef_func_compatible(self):
        """Determines if the function's signature is compatible with a
        cdef function.  This can be used before calling
        .as_cfunction() to see if that will be successful.
        """
        if self.needs_closure:
            return False
        if self.star_arg or self.starstar_arg:
            return False
        return True

    def analyse_declarations(self, env):
        if self.decorators:
            for decorator in self.decorators:
                func = decorator.decorator
                if func.is_name:
                    self.is_classmethod |= func.name == 'classmethod'
                    self.is_staticmethod |= func.name == 'staticmethod'

        if self.is_classmethod and env.lookup_here('classmethod'):
            # classmethod() was overridden - not much we can do here ...
            self.is_classmethod = False
        if self.is_staticmethod and env.lookup_here('staticmethod'):
            # staticmethod() was overridden - not much we can do here ...
            self.is_staticmethod = False

        if self.name == '__new__' and env.is_py_class_scope:
            self.is_staticmethod = 1

        self.analyse_argument_types(env)
        if self.name == '<lambda>':
            self.declare_lambda_function(env)
        else:
            self.declare_pyfunction(env)

        self.analyse_signature(env)
        self.return_type = self.entry.signature.return_type()
        # if a signature annotation provides a more specific return object type, use it
        if self.return_type is py_object_type and self.return_type_annotation:
            if env.directives['annotation_typing'] and not self.entry.is_special:
                _, return_type = analyse_type_annotation(self.return_type_annotation, env)
                if return_type and return_type.is_pyobject:
                    self.return_type = return_type

        self.create_local_scope(env)

        self.py_wrapper = DefNodeWrapper(
            self.pos,
            target=self,
            name=self.entry.name,
            args=self.args,
            star_arg=self.star_arg,
            starstar_arg=self.starstar_arg,
            return_type=self.return_type)
        self.py_wrapper.analyse_declarations(env)

    def analyse_argument_types(self, env):
        self.directive_locals = env.directives.get('locals', {})
        allow_none_for_extension_args = env.directives['allow_none_for_extension_args']

        f2s = env.fused_to_specific
        env.fused_to_specific = None

        for arg in self.args:
            if hasattr(arg, 'name'):
                name_declarator = None
            else:
                base_type = arg.base_type.analyse(env)
                # If we hare in pythran mode and we got a buffer supported by
                # Pythran, we change this node to a fused type
                if has_np_pythran(env) and base_type.is_pythran_expr:
                    base_type = PyrexTypes.FusedType([
                        base_type,
                        #PyrexTypes.PythranExpr(pythran_type(self.type, "numpy_texpr")),
                        base_type.org_buffer])
                name_declarator, type = \
                    arg.declarator.analyse(base_type, env)
                arg.name = name_declarator.name
                arg.type = type

                if type.is_fused:
                    self.has_fused_arguments = True

            self.align_argument_type(env, arg)
            if name_declarator and name_declarator.cname:
                error(self.pos, "Python function argument cannot have C name specification")
            arg.type = arg.type.as_argument_type()
            arg.hdr_type = None
            arg.needs_conversion = 0
            arg.needs_type_test = 0
            arg.is_generic = 1
            if arg.type.is_pyobject or arg.type.is_buffer or arg.type.is_memoryviewslice:
                if arg.or_none:
                    arg.accept_none = True
                elif arg.not_none:
                    arg.accept_none = False
                elif (arg.type.is_extension_type or arg.type.is_builtin_type
                        or arg.type.is_buffer or arg.type.is_memoryviewslice):
                    if arg.default and arg.default.constant_result is None:
                        # special case: def func(MyType obj = None)
                        arg.accept_none = True
                    else:
                        # default depends on compiler directive
                        arg.accept_none = allow_none_for_extension_args
                else:
                    # probably just a plain 'object'
                    arg.accept_none = True
            else:
                arg.accept_none = True # won't be used, but must be there
                if arg.not_none:
                    error(arg.pos, "Only Python type arguments can have 'not None'")
                if arg.or_none:
                    error(arg.pos, "Only Python type arguments can have 'or None'")
        env.fused_to_specific = f2s

        if has_np_pythran(env):
            self.np_args_idx = [i for i,a in enumerate(self.args) if a.type.is_numpy_buffer]
        else:
            self.np_args_idx = []

    def analyse_signature(self, env):
        if self.entry.is_special:
            if self.decorators:
                error(self.pos, "special functions of cdef classes cannot have decorators")
            self.entry.trivial_signature = len(self.args) == 1 and not (self.star_arg or self.starstar_arg)
        elif not env.directives['always_allow_keywords'] and not (self.star_arg or self.starstar_arg):
            # Use the simpler calling signature for zero- and one-argument functions.
            if self.entry.signature is TypeSlots.pyfunction_signature:
                if len(self.args) == 0:
                    self.entry.signature = TypeSlots.pyfunction_noargs
                elif len(self.args) == 1:
                    if self.args[0].default is None and not self.args[0].kw_only:
                        self.entry.signature = TypeSlots.pyfunction_onearg
            elif self.entry.signature is TypeSlots.pymethod_signature:
                if len(self.args) == 1:
                    self.entry.signature = TypeSlots.unaryfunc
                elif len(self.args) == 2:
                    if self.args[1].default is None and not self.args[1].kw_only:
                        self.entry.signature = TypeSlots.ibinaryfunc

        sig = self.entry.signature
        nfixed = sig.num_fixed_args()
        if (sig is TypeSlots.pymethod_signature and nfixed == 1
               and len(self.args) == 0 and self.star_arg):
            # this is the only case where a diverging number of
            # arguments is not an error - when we have no explicit
            # 'self' parameter as in method(*args)
            sig = self.entry.signature = TypeSlots.pyfunction_signature # self is not 'really' used
            self.self_in_stararg = 1
            nfixed = 0

        if self.is_staticmethod and env.is_c_class_scope:
            nfixed = 0
            self.self_in_stararg = True  # FIXME: why for staticmethods?

            self.entry.signature = sig = copy.copy(sig)
            sig.fixed_arg_format = "*"
            sig.is_staticmethod = True
            sig.has_generic_args = True

        if ((self.is_classmethod or self.is_staticmethod) and
                self.has_fused_arguments and env.is_c_class_scope):
            del self.decorator_indirection.stats[:]

        for i in range(min(nfixed, len(self.args))):
            arg = self.args[i]
            arg.is_generic = 0
            if sig.is_self_arg(i) and not self.is_staticmethod:
                if self.is_classmethod:
                    arg.is_type_arg = 1
                    arg.hdr_type = arg.type = Builtin.type_type
                else:
                    arg.is_self_arg = 1
                    arg.hdr_type = arg.type = env.parent_type
                arg.needs_conversion = 0
            else:
                arg.hdr_type = sig.fixed_arg_type(i)
                if not arg.type.same_as(arg.hdr_type):
                    if arg.hdr_type.is_pyobject and arg.type.is_pyobject:
                        arg.needs_type_test = 1
                    else:
                        arg.needs_conversion = 1
            if arg.needs_conversion:
                arg.hdr_cname = Naming.arg_prefix + arg.name
            else:
                arg.hdr_cname = Naming.var_prefix + arg.name

        if nfixed > len(self.args):
            self.bad_signature()
            return
        elif nfixed < len(self.args):
            if not sig.has_generic_args:
                self.bad_signature()
            for arg in self.args:
                if arg.is_generic and (arg.type.is_extension_type or arg.type.is_builtin_type):
                    arg.needs_type_test = 1

    def bad_signature(self):
        sig = self.entry.signature
        expected_str = "%d" % sig.num_fixed_args()
        if sig.has_generic_args:
            expected_str += " or more"
        name = self.name
        if name.startswith("__") and name.endswith("__"):
            desc = "Special method"
        else:
            desc = "Method"
        error(self.pos, "%s %s has wrong number of arguments (%d declared, %s expected)" % (
            desc, self.name, len(self.args), expected_str))

    def declare_pyfunction(self, env):
        #print "DefNode.declare_pyfunction:", self.name, "in", env ###
        name = self.name
        entry = env.lookup_here(name)
        if entry:
            if entry.is_final_cmethod and not env.parent_type.is_final_type:
                error(self.pos, "Only final types can have final Python (def/cpdef) methods")
            if entry.type.is_cfunction and not entry.is_builtin_cmethod and not self.is_wrapper:
                warning(self.pos, "Overriding cdef method with def method.", 5)
        entry = env.declare_pyfunction(name, self.pos, allow_redefine=not self.is_wrapper)
        self.entry = entry
        prefix = env.next_id(env.scope_prefix)
        self.entry.pyfunc_cname = Naming.pyfunc_prefix + prefix + name
        if Options.docstrings:
            entry.doc = embed_position(self.pos, self.doc)
            entry.doc_cname = Naming.funcdoc_prefix + prefix + name
            if entry.is_special:
                if entry.name in TypeSlots.invisible or not entry.doc or (
                        entry.name in '__getattr__' and env.directives['fast_getattr']):
                    entry.wrapperbase_cname = None
                else:
                    entry.wrapperbase_cname = Naming.wrapperbase_prefix + prefix + name
        else:
            entry.doc = None

    def declare_lambda_function(self, env):
        entry = env.declare_lambda_function(self.lambda_name, self.pos)
        entry.doc = None
        self.entry = entry
        self.entry.pyfunc_cname = entry.cname

    def declare_arguments(self, env):
        for arg in self.args:
            if not arg.name:
                error(arg.pos, "Missing argument name")
            if arg.needs_conversion:
                arg.entry = env.declare_var(arg.name, arg.type, arg.pos)
                if arg.type.is_pyobject:
                    arg.entry.init = "0"
            else:
                arg.entry = self.declare_argument(env, arg)
            arg.entry.is_arg = 1
            arg.entry.used = 1
            arg.entry.is_self_arg = arg.is_self_arg
        self.declare_python_arg(env, self.star_arg)
        self.declare_python_arg(env, self.starstar_arg)

    def declare_python_arg(self, env, arg):
        if arg:
            if env.directives['infer_types'] != False:
                type = PyrexTypes.unspecified_type
            else:
                type = py_object_type
            entry = env.declare_var(arg.name, type, arg.pos)
            entry.is_arg = 1
            entry.used = 1
            entry.init = "0"
            entry.xdecref_cleanup = 1
            arg.entry = entry

    def analyse_expressions(self, env):
        self.local_scope.directives = env.directives
        self.analyse_default_values(env)
        self.analyse_annotations(env)
        if self.return_type_annotation:
            self.return_type_annotation = self.analyse_annotation(env, self.return_type_annotation)

        if not self.needs_assignment_synthesis(env) and self.decorators:
            for decorator in self.decorators[::-1]:
                decorator.decorator = decorator.decorator.analyse_expressions(env)

        self.py_wrapper.prepare_argument_coercion(env)
        return self

    def needs_assignment_synthesis(self, env, code=None):
        if self.is_staticmethod:
            return True
        if self.specialized_cpdefs or self.entry.is_fused_specialized:
            return False
        if self.no_assignment_synthesis:
            return False
        if self.entry.is_special:
            return False
        if self.entry.is_anonymous:
            return True
        if env.is_module_scope or env.is_c_class_scope:
            if code is None:
                return self.local_scope.directives['binding']
            else:
                return code.globalstate.directives['binding']
        return env.is_py_class_scope or env.is_closure_scope

    def error_value(self):
        return self.entry.signature.error_value

    def caller_will_check_exceptions(self):
        return self.entry.signature.exception_check

    def generate_function_definitions(self, env, code):
        if self.defaults_getter:
            # defaults getter must never live in class scopes, it's always a module function
            self.defaults_getter.generate_function_definitions(env.global_scope(), code)

        # Before closure cnames are mangled
        if self.py_wrapper_required:
            # func_cname might be modified by @cname
            self.py_wrapper.func_cname = self.entry.func_cname
            self.py_wrapper.generate_function_definitions(env, code)
        FuncDefNode.generate_function_definitions(self, env, code)

    def generate_function_header(self, code, with_pymethdef, proto_only=0):
        if proto_only:
            if self.py_wrapper_required:
                self.py_wrapper.generate_function_header(
                    code, with_pymethdef, True)
            return
        arg_code_list = []
        if self.entry.signature.has_dummy_arg:
            self_arg = 'PyObject *%s' % Naming.self_cname
            if not self.needs_outer_scope:
                self_arg = 'CYTHON_UNUSED ' + self_arg
            arg_code_list.append(self_arg)

        def arg_decl_code(arg):
            entry = arg.entry
            if entry.in_closure:
                cname = entry.original_cname
            else:
                cname = entry.cname
            decl = entry.type.declaration_code(cname)
            if not entry.cf_used:
                decl = 'CYTHON_UNUSED ' + decl
            return decl

        for arg in self.args:
            arg_code_list.append(arg_decl_code(arg))
        if self.star_arg:
            arg_code_list.append(arg_decl_code(self.star_arg))
        if self.starstar_arg:
            arg_code_list.append(arg_decl_code(self.starstar_arg))
        if arg_code_list:
            arg_code = ', '.join(arg_code_list)
        else:
            arg_code = 'void'  # No arguments
        dc = self.return_type.declaration_code(self.entry.pyfunc_cname)

        decls_code = code.globalstate['decls']
        preprocessor_guard = self.get_preprocessor_guard()
        if preprocessor_guard:
            decls_code.putln(preprocessor_guard)
        decls_code.putln(
            "static %s(%s); /* proto */" % (dc, arg_code))
        if preprocessor_guard:
            decls_code.putln("#endif")
        code.putln("static %s(%s) {" % (dc, arg_code))

    def generate_argument_declarations(self, env, code):
        pass

    def generate_keyword_list(self, code):
        pass

    def generate_argument_parsing_code(self, env, code):
        # Move arguments into closure if required
        def put_into_closure(entry):
            if entry.in_closure:
                code.putln('%s = %s;' % (entry.cname, entry.original_cname))
                if entry.xdecref_cleanup:
                    # mostly applies to the starstar arg - this can sometimes be NULL
                    # so must be xincrefed instead
                    code.put_var_xincref(entry)
                    code.put_var_xgiveref(entry)
                else:
                    code.put_var_incref(entry)
                    code.put_var_giveref(entry)
        for arg in self.args:
            put_into_closure(arg.entry)
        for arg in self.star_arg, self.starstar_arg:
            if arg:
                put_into_closure(arg.entry)

    def generate_argument_type_tests(self, code):
        pass


class DefNodeWrapper(FuncDefNode):
    # DefNode python wrapper code generator

    defnode = None
    target = None # Target DefNode

    def __init__(self, *args, **kwargs):
        FuncDefNode.__init__(self, *args, **kwargs)
        self.num_kwonly_args = self.target.num_kwonly_args
        self.num_required_kw_args = self.target.num_required_kw_args
        self.num_required_args = self.target.num_required_args
        self.self_in_stararg = self.target.self_in_stararg
        self.signature = None

    def analyse_declarations(self, env):
        target_entry = self.target.entry
        name = self.name
        prefix = env.next_id(env.scope_prefix)
        target_entry.func_cname = Naming.pywrap_prefix + prefix + name
        target_entry.pymethdef_cname = Naming.pymethdef_prefix + prefix + name

        self.signature = target_entry.signature

        self.np_args_idx = self.target.np_args_idx

    def prepare_argument_coercion(self, env):
        # This is only really required for Cython utility code at this time,
        # everything else can be done during code generation.  But we expand
        # all utility code here, simply because we cannot easily distinguish
        # different code types.
        for arg in self.args:
            if not arg.type.is_pyobject:
                if not arg.type.create_from_py_utility_code(env):
                    pass # will fail later
            elif arg.hdr_type and not arg.hdr_type.is_pyobject:
                if not arg.hdr_type.create_to_py_utility_code(env):
                    pass # will fail later

        if self.starstar_arg and not self.starstar_arg.entry.cf_used:
            # we will set the kwargs argument to NULL instead of a new dict
            # and must therefore correct the control flow state
            entry = self.starstar_arg.entry
            entry.xdecref_cleanup = 1
            for ass in entry.cf_assignments:
                if not ass.is_arg and ass.lhs.is_name:
                    ass.lhs.cf_maybe_null = True

    def signature_has_nongeneric_args(self):
        argcount = len(self.args)
        if argcount == 0 or (
                argcount == 1 and (self.args[0].is_self_arg or
                                   self.args[0].is_type_arg)):
            return 0
        return 1

    def signature_has_generic_args(self):
        return self.signature.has_generic_args

    def generate_function_body(self, code):
        args = []
        if self.signature.has_dummy_arg:
            args.append(Naming.self_cname)
        for arg in self.args:
            if arg.hdr_type and not (arg.type.is_memoryviewslice or
                                     arg.type.is_struct or
                                     arg.type.is_complex):
                args.append(arg.type.cast_code(arg.entry.cname))
            else:
                args.append(arg.entry.cname)
        if self.star_arg:
            args.append(self.star_arg.entry.cname)
        if self.starstar_arg:
            args.append(self.starstar_arg.entry.cname)
        args = ', '.join(args)
        if not self.return_type.is_void:
            code.put('%s = ' % Naming.retval_cname)
        code.putln('%s(%s);' % (
            self.target.entry.pyfunc_cname, args))

    def generate_function_definitions(self, env, code):
        lenv = self.target.local_scope
        # Generate C code for header and body of function
        code.mark_pos(self.pos)
        code.putln("")
        code.putln("/* Python wrapper */")
        preprocessor_guard = self.target.get_preprocessor_guard()
        if preprocessor_guard:
            code.putln(preprocessor_guard)

        code.enter_cfunc_scope(lenv)
        code.return_from_error_cleanup_label = code.new_label()

        with_pymethdef = (self.target.needs_assignment_synthesis(env, code) or
                          self.target.pymethdef_required)
        self.generate_function_header(code, with_pymethdef)
        self.generate_argument_declarations(lenv, code)
        tempvardecl_code = code.insertion_point()

        if self.return_type.is_pyobject:
            retval_init = ' = 0'
        else:
            retval_init = ''
        if not self.return_type.is_void:
            code.putln('%s%s;' % (
                self.return_type.declaration_code(Naming.retval_cname),
                retval_init))
        code.put_declare_refcount_context()
        code.put_setup_refcount_context('%s (wrapper)' % self.name)

        self.generate_argument_parsing_code(lenv, code)
        self.generate_argument_type_tests(code)
        self.generate_function_body(code)

        # ----- Go back and insert temp variable declarations
        tempvardecl_code.put_temp_declarations(code.funcstate)

        code.mark_pos(self.pos)
        code.putln("")
        code.putln("/* function exit code */")

        # ----- Error cleanup
        if code.error_label in code.labels_used:
            code.put_goto(code.return_label)
            code.put_label(code.error_label)
            for cname, type in code.funcstate.all_managed_temps():
                code.put_xdecref(cname, type)
            err_val = self.error_value()
            if err_val is not None:
                code.putln("%s = %s;" % (Naming.retval_cname, err_val))

        # ----- Non-error return cleanup
        code.put_label(code.return_label)
        for entry in lenv.var_entries:
            if entry.is_arg and entry.type.is_pyobject:
                code.put_var_decref(entry)

        code.put_finish_refcount_context()
        if not self.return_type.is_void:
            code.putln("return %s;" % Naming.retval_cname)
        code.putln('}')
        code.exit_cfunc_scope()
        if preprocessor_guard:
            code.putln("#endif /*!(%s)*/" % preprocessor_guard)

    def generate_function_header(self, code, with_pymethdef, proto_only=0):
        arg_code_list = []
        sig = self.signature

        if sig.has_dummy_arg or self.self_in_stararg:
            arg_code = "PyObject *%s" % Naming.self_cname
            if not sig.has_dummy_arg:
                arg_code = 'CYTHON_UNUSED ' + arg_code
            arg_code_list.append(arg_code)

        for arg in self.args:
            if not arg.is_generic:
                if arg.is_self_arg or arg.is_type_arg:
                    arg_code_list.append("PyObject *%s" % arg.hdr_cname)
                else:
                    arg_code_list.append(
                        arg.hdr_type.declaration_code(arg.hdr_cname))
        entry = self.target.entry
        if not entry.is_special and sig.method_flags() == [TypeSlots.method_noargs]:
            arg_code_list.append("CYTHON_UNUSED PyObject *unused")
        if entry.scope.is_c_class_scope and entry.name == "__ipow__":
            arg_code_list.append("CYTHON_UNUSED PyObject *unused")
        if sig.has_generic_args:
            arg_code_list.append(
                "PyObject *%s, PyObject *%s" % (
                    Naming.args_cname, Naming.kwds_cname))
        arg_code = ", ".join(arg_code_list)

        # Prevent warning: unused function '__pyx_pw_5numpy_7ndarray_1__getbuffer__'
        mf = ""
        if (entry.name in ("__getbuffer__", "__releasebuffer__")
                and entry.scope.is_c_class_scope):
            mf = "CYTHON_UNUSED "
            with_pymethdef = False

        dc = self.return_type.declaration_code(entry.func_cname)
        header = "static %s%s(%s)" % (mf, dc, arg_code)
        code.putln("%s; /*proto*/" % header)

        if proto_only:
            if self.target.fused_py_func:
                # If we are the specialized version of the cpdef, we still
                # want the prototype for the "fused cpdef", in case we're
                # checking to see if our method was overridden in Python
                self.target.fused_py_func.generate_function_header(
                    code, with_pymethdef, proto_only=True)
            return

        if (Options.docstrings and entry.doc and
                not self.target.fused_py_func and
                not entry.scope.is_property_scope and
                (not entry.is_special or entry.wrapperbase_cname)):
            # h_code = code.globalstate['h_code']
            docstr = entry.doc

            if docstr.is_unicode:
                docstr = docstr.as_utf8_string()

            if not (entry.is_special and entry.name in ('__getbuffer__', '__releasebuffer__')):
                code.putln('static char %s[] = %s;' % (
                    entry.doc_cname,
                    docstr.as_c_string_literal()))

            if entry.is_special:
                code.putln('#if CYTHON_COMPILING_IN_CPYTHON')
                code.putln(
                    "struct wrapperbase %s;" % entry.wrapperbase_cname)
                code.putln('#endif')

        if with_pymethdef or self.target.fused_py_func:
            code.put(
                "static PyMethodDef %s = " % entry.pymethdef_cname)
            code.put_pymethoddef(self.target.entry, ";", allow_skip=False)
        code.putln("%s {" % header)

    def generate_argument_declarations(self, env, code):
        for arg in self.args:
            if arg.is_generic:
                if arg.needs_conversion:
                    code.putln("PyObject *%s = 0;" % arg.hdr_cname)
                else:
                    code.put_var_declaration(arg.entry)
        for entry in env.var_entries:
            if entry.is_arg:
                code.put_var_declaration(entry)

    def generate_argument_parsing_code(self, env, code):
        # Generate fast equivalent of PyArg_ParseTuple call for
        # generic arguments, if any, including args/kwargs
        old_error_label = code.new_error_label()
        our_error_label = code.error_label
        end_label = code.new_label("argument_unpacking_done")

        has_kwonly_args = self.num_kwonly_args > 0
        has_star_or_kw_args = self.star_arg is not None \
            or self.starstar_arg is not None or has_kwonly_args

        for arg in self.args:
            if not arg.type.is_pyobject:
                if not arg.type.create_from_py_utility_code(env):
                    pass  # will fail later

        if not self.signature_has_generic_args():
            if has_star_or_kw_args:
                error(self.pos, "This method cannot have * or keyword arguments")
            self.generate_argument_conversion_code(code)

        elif not self.signature_has_nongeneric_args():
            # func(*args) or func(**kw) or func(*args, **kw)
            self.generate_stararg_copy_code(code)

        else:
            self.generate_tuple_and_keyword_parsing_code(self.args, end_label, code)

        code.error_label = old_error_label
        if code.label_used(our_error_label):
            if not code.label_used(end_label):
                code.put_goto(end_label)
            code.put_label(our_error_label)
            if has_star_or_kw_args:
                self.generate_arg_decref(self.star_arg, code)
                if self.starstar_arg:
                    if self.starstar_arg.entry.xdecref_cleanup:
                        code.put_var_xdecref_clear(self.starstar_arg.entry)
                    else:
                        code.put_var_decref_clear(self.starstar_arg.entry)
            code.put_add_traceback(self.target.entry.qualified_name)
            code.put_finish_refcount_context()
            code.putln("return %s;" % self.error_value())
        if code.label_used(end_label):
            code.put_label(end_label)

    def generate_arg_xdecref(self, arg, code):
        if arg:
            code.put_var_xdecref_clear(arg.entry)

    def generate_arg_decref(self, arg, code):
        if arg:
            code.put_var_decref_clear(arg.entry)

    def generate_stararg_copy_code(self, code):
        if not self.star_arg:
            code.globalstate.use_utility_code(
                UtilityCode.load_cached("RaiseArgTupleInvalid", "FunctionArguments.c"))
            code.putln("if (unlikely(PyTuple_GET_SIZE(%s) > 0)) {" %
                       Naming.args_cname)
            code.put('__Pyx_RaiseArgtupleInvalid("%s", 1, 0, 0, PyTuple_GET_SIZE(%s)); return %s;' % (
                self.name, Naming.args_cname, self.error_value()))
            code.putln("}")

        if self.starstar_arg:
            if self.star_arg or not self.starstar_arg.entry.cf_used:
                kwarg_check = "unlikely(%s)" % Naming.kwds_cname
            else:
                kwarg_check = "%s" % Naming.kwds_cname
        else:
            kwarg_check = "unlikely(%s) && unlikely(PyDict_Size(%s) > 0)" % (
                Naming.kwds_cname, Naming.kwds_cname)
        code.globalstate.use_utility_code(
            UtilityCode.load_cached("KeywordStringCheck", "FunctionArguments.c"))
        code.putln(
            "if (%s && unlikely(!__Pyx_CheckKeywordStrings(%s, \"%s\", %d))) return %s;" % (
                kwarg_check, Naming.kwds_cname, self.name,
                bool(self.starstar_arg), self.error_value()))

        if self.starstar_arg and self.starstar_arg.entry.cf_used:
            if all(ref.node.allow_null for ref in self.starstar_arg.entry.cf_references):
                code.putln("if (%s) {" % kwarg_check)
                code.putln("%s = PyDict_Copy(%s); if (unlikely(!%s)) return %s;" % (
                    self.starstar_arg.entry.cname,
                    Naming.kwds_cname,
                    self.starstar_arg.entry.cname,
                    self.error_value()))
                code.put_gotref(self.starstar_arg.entry.cname)
                code.putln("} else {")
                code.putln("%s = NULL;" % (self.starstar_arg.entry.cname,))
                code.putln("}")
                self.starstar_arg.entry.xdecref_cleanup = 1
            else:
                code.put("%s = (%s) ? PyDict_Copy(%s) : PyDict_New(); " % (
                    self.starstar_arg.entry.cname,
                    Naming.kwds_cname,
                    Naming.kwds_cname))
                code.putln("if (unlikely(!%s)) return %s;" % (
                    self.starstar_arg.entry.cname, self.error_value()))
                self.starstar_arg.entry.xdecref_cleanup = 0
                code.put_gotref(self.starstar_arg.entry.cname)

        if self.self_in_stararg and not self.target.is_staticmethod:
            # need to create a new tuple with 'self' inserted as first item
            code.put("%s = PyTuple_New(PyTuple_GET_SIZE(%s)+1); if (unlikely(!%s)) " % (
                self.star_arg.entry.cname,
                Naming.args_cname,
                self.star_arg.entry.cname))
            if self.starstar_arg and self.starstar_arg.entry.cf_used:
                code.putln("{")
                code.put_xdecref_clear(self.starstar_arg.entry.cname, py_object_type)
                code.putln("return %s;" % self.error_value())
                code.putln("}")
            else:
                code.putln("return %s;" % self.error_value())
            code.put_gotref(self.star_arg.entry.cname)
            code.put_incref(Naming.self_cname, py_object_type)
            code.put_giveref(Naming.self_cname)
            code.putln("PyTuple_SET_ITEM(%s, 0, %s);" % (
                self.star_arg.entry.cname, Naming.self_cname))
            temp = code.funcstate.allocate_temp(PyrexTypes.c_py_ssize_t_type, manage_ref=False)
            code.putln("for (%s=0; %s < PyTuple_GET_SIZE(%s); %s++) {" % (
                temp, temp, Naming.args_cname, temp))
            code.putln("PyObject* item = PyTuple_GET_ITEM(%s, %s);" % (
                Naming.args_cname, temp))
            code.put_incref("item", py_object_type)
            code.put_giveref("item")
            code.putln("PyTuple_SET_ITEM(%s, %s+1, item);" % (
                self.star_arg.entry.cname, temp))
            code.putln("}")
            code.funcstate.release_temp(temp)
            self.star_arg.entry.xdecref_cleanup = 0
        elif self.star_arg:
            code.put_incref(Naming.args_cname, py_object_type)
            code.putln("%s = %s;" % (
                self.star_arg.entry.cname,
                Naming.args_cname))
            self.star_arg.entry.xdecref_cleanup = 0

    def generate_tuple_and_keyword_parsing_code(self, args, success_label, code):
        argtuple_error_label = code.new_label("argtuple_error")

        positional_args = []
        required_kw_only_args = []
        optional_kw_only_args = []
        for arg in args:
            if arg.is_generic:
                if arg.default:
                    if not arg.is_self_arg and not arg.is_type_arg:
                        if arg.kw_only:
                            optional_kw_only_args.append(arg)
                        else:
                            positional_args.append(arg)
                elif arg.kw_only:
                    required_kw_only_args.append(arg)
                elif not arg.is_self_arg and not arg.is_type_arg:
                    positional_args.append(arg)

        # sort required kw-only args before optional ones to avoid special
        # cases in the unpacking code
        kw_only_args = required_kw_only_args + optional_kw_only_args

        min_positional_args = self.num_required_args - self.num_required_kw_args
        if len(args) > 0 and (args[0].is_self_arg or args[0].is_type_arg):
            min_positional_args -= 1
        max_positional_args = len(positional_args)
        has_fixed_positional_count = not self.star_arg and \
            min_positional_args == max_positional_args
        has_kw_only_args = bool(kw_only_args)

        if self.starstar_arg or self.star_arg:
            self.generate_stararg_init_code(max_positional_args, code)

        code.putln('{')
        all_args = tuple(positional_args) + tuple(kw_only_args)
        code.putln("static PyObject **%s[] = {%s,0};" % (
            Naming.pykwdlist_cname,
            ','.join(['&%s' % code.intern_identifier(arg.name)
                      for arg in all_args])))

        # Before being converted and assigned to the target variables,
        # borrowed references to all unpacked argument values are
        # collected into a local PyObject* array called "values",
        # regardless if they were taken from default arguments,
        # positional arguments or keyword arguments.  Note that
        # C-typed default arguments are handled at conversion time,
        # so their array value is NULL in the end if no argument
        # was passed for them.
        self.generate_argument_values_setup_code(all_args, code)

        # --- optimised code when we receive keyword arguments
        code.putln("if (%s(%s)) {" % (
            (self.num_required_kw_args > 0) and "likely" or "unlikely",
            Naming.kwds_cname))
        self.generate_keyword_unpacking_code(
            min_positional_args, max_positional_args,
            has_fixed_positional_count, has_kw_only_args,
            all_args, argtuple_error_label, code)

        # --- optimised code when we do not receive any keyword arguments
        if (self.num_required_kw_args and min_positional_args > 0) or min_positional_args == max_positional_args:
            # Python raises arg tuple related errors first, so we must
            # check the length here
            if min_positional_args == max_positional_args and not self.star_arg:
                compare = '!='
            else:
                compare = '<'
            code.putln('} else if (PyTuple_GET_SIZE(%s) %s %d) {' % (
                Naming.args_cname, compare, min_positional_args))
            code.put_goto(argtuple_error_label)

        if self.num_required_kw_args:
            # pure error case: keywords required but not passed
            if max_positional_args > min_positional_args and not self.star_arg:
                code.putln('} else if (PyTuple_GET_SIZE(%s) > %d) {' % (
                    Naming.args_cname, max_positional_args))
                code.put_goto(argtuple_error_label)
            code.putln('} else {')
            for i, arg in enumerate(kw_only_args):
                if not arg.default:
                    pystring_cname = code.intern_identifier(arg.name)
                    # required keyword-only argument missing
                    code.globalstate.use_utility_code(
                        UtilityCode.load_cached("RaiseKeywordRequired", "FunctionArguments.c"))
                    code.put('__Pyx_RaiseKeywordRequired("%s", %s); ' % (
                        self.name,
                        pystring_cname))
                    code.putln(code.error_goto(self.pos))
                    break

        else:
            # optimised tuple unpacking code
            code.putln('} else {')
            if min_positional_args == max_positional_args:
                # parse the exact number of positional arguments from
                # the args tuple
                for i, arg in enumerate(positional_args):
                    code.putln("values[%d] = PyTuple_GET_ITEM(%s, %d);" % (i, Naming.args_cname, i))
            else:
                # parse the positional arguments from the variable length
                # args tuple and reject illegal argument tuple sizes
                code.putln('switch (PyTuple_GET_SIZE(%s)) {' % Naming.args_cname)
                if self.star_arg:
                    code.putln('default:')
                reversed_args = list(enumerate(positional_args))[::-1]
                for i, arg in reversed_args:
                    if i >= min_positional_args-1:
                        if i != reversed_args[0][0]:
                            code.putln('CYTHON_FALLTHROUGH;')
                        code.put('case %2d: ' % (i+1))
                    code.putln("values[%d] = PyTuple_GET_ITEM(%s, %d);" % (i, Naming.args_cname, i))
                if min_positional_args == 0:
                    code.putln('CYTHON_FALLTHROUGH;')
                    code.put('case  0: ')
                code.putln('break;')
                if self.star_arg:
                    if min_positional_args:
                        for i in range(min_positional_args-1, -1, -1):
                            code.putln('case %2d:' % i)
                        code.put_goto(argtuple_error_label)
                else:
                    code.put('default: ')
                    code.put_goto(argtuple_error_label)
                code.putln('}')

        code.putln('}') # end of the conditional unpacking blocks

        # Convert arg values to their final type and assign them.
        # Also inject non-Python default arguments, which do cannot
        # live in the values[] array.
        for i, arg in enumerate(all_args):
            self.generate_arg_assignment(arg, "values[%d]" % i, code)

        code.putln('}') # end of the whole argument unpacking block

        if code.label_used(argtuple_error_label):
            code.put_goto(success_label)
            code.put_label(argtuple_error_label)
            code.globalstate.use_utility_code(
                UtilityCode.load_cached("RaiseArgTupleInvalid", "FunctionArguments.c"))
            code.put('__Pyx_RaiseArgtupleInvalid("%s", %d, %d, %d, PyTuple_GET_SIZE(%s)); ' % (
                self.name, has_fixed_positional_count,
                min_positional_args, max_positional_args,
                Naming.args_cname))
            code.putln(code.error_goto(self.pos))

    def generate_arg_assignment(self, arg, item, code):
        if arg.type.is_pyobject:
            # Python default arguments were already stored in 'item' at the very beginning
            if arg.is_generic:
                item = PyrexTypes.typecast(arg.type, PyrexTypes.py_object_type, item)
            entry = arg.entry
            code.putln("%s = %s;" % (entry.cname, item))
        else:
            if arg.type.from_py_function:
                if arg.default:
                    # C-typed default arguments must be handled here
                    code.putln('if (%s) {' % item)
                code.putln(arg.type.from_py_call_code(
                    item, arg.entry.cname, arg.pos, code))
                if arg.default:
                    code.putln('} else {')
                    code.putln("%s = %s;" % (
                        arg.entry.cname,
                        arg.calculate_default_value_code(code)))
                    if arg.type.is_memoryviewslice:
                        code.put_incref_memoryviewslice(arg.entry.cname,
                                                        have_gil=True)
                    code.putln('}')
            else:
                error(arg.pos, "Cannot convert Python object argument to type '%s'" % arg.type)

    def generate_stararg_init_code(self, max_positional_args, code):
        if self.starstar_arg:
            self.starstar_arg.entry.xdecref_cleanup = 0
            code.putln('%s = PyDict_New(); if (unlikely(!%s)) return %s;' % (
                self.starstar_arg.entry.cname,
                self.starstar_arg.entry.cname,
                self.error_value()))
            code.put_gotref(self.starstar_arg.entry.cname)
        if self.star_arg:
            self.star_arg.entry.xdecref_cleanup = 0
            code.putln('if (PyTuple_GET_SIZE(%s) > %d) {' % (
                Naming.args_cname,
                max_positional_args))
            code.putln('%s = PyTuple_GetSlice(%s, %d, PyTuple_GET_SIZE(%s));' % (
                self.star_arg.entry.cname, Naming.args_cname,
                max_positional_args, Naming.args_cname))
            code.putln("if (unlikely(!%s)) {" % self.star_arg.entry.cname)
            if self.starstar_arg:
                code.put_decref_clear(self.starstar_arg.entry.cname, py_object_type)
            code.put_finish_refcount_context()
            code.putln('return %s;' % self.error_value())
            code.putln('}')
            code.put_gotref(self.star_arg.entry.cname)
            code.putln('} else {')
            code.put("%s = %s; " % (self.star_arg.entry.cname, Naming.empty_tuple))
            code.put_incref(Naming.empty_tuple, py_object_type)
            code.putln('}')

    def generate_argument_values_setup_code(self, args, code):
        max_args = len(args)
        # the 'values' array collects borrowed references to arguments
        # before doing any type coercion etc.
        code.putln("PyObject* values[%d] = {%s};" % (
            max_args, ','.join('0'*max_args)))

        if self.target.defaults_struct:
            code.putln('%s *%s = __Pyx_CyFunction_Defaults(%s, %s);' % (
                self.target.defaults_struct, Naming.dynamic_args_cname,
                self.target.defaults_struct, Naming.self_cname))

        # assign borrowed Python default values to the values array,
        # so that they can be overwritten by received arguments below
        for i, arg in enumerate(args):
            if arg.default and arg.type.is_pyobject:
                default_value = arg.calculate_default_value_code(code)
                code.putln('values[%d] = %s;' % (i, arg.type.as_pyobject(default_value)))

    def generate_keyword_unpacking_code(self, min_positional_args, max_positional_args,
                                        has_fixed_positional_count, has_kw_only_args,
                                        all_args, argtuple_error_label, code):
        code.putln('Py_ssize_t kw_args;')
        code.putln('const Py_ssize_t pos_args = PyTuple_GET_SIZE(%s);' % Naming.args_cname)
        # copy the values from the args tuple and check that it's not too long
        code.putln('switch (pos_args) {')
        if self.star_arg:
            code.putln('default:')
        for i in range(max_positional_args-1, -1, -1):
            code.put('case %2d: ' % (i+1))
            code.putln("values[%d] = PyTuple_GET_ITEM(%s, %d);" % (
                i, Naming.args_cname, i))
            code.putln('CYTHON_FALLTHROUGH;')
        code.putln('case  0: break;')
        if not self.star_arg:
            code.put('default: ') # more arguments than allowed
            code.put_goto(argtuple_error_label)
        code.putln('}')

        # The code above is very often (but not always) the same as
        # the optimised non-kwargs tuple unpacking code, so we keep
        # the code block above at the very top, before the following
        # 'external' PyDict_Size() call, to make it easy for the C
        # compiler to merge the two separate tuple unpacking
        # implementations into one when they turn out to be identical.

        # If we received kwargs, fill up the positional/required
        # arguments with values from the kw dict
        code.putln('kw_args = PyDict_Size(%s);' % Naming.kwds_cname)
        if self.num_required_args or max_positional_args > 0:
            last_required_arg = -1
            for i, arg in enumerate(all_args):
                if not arg.default:
                    last_required_arg = i
            if last_required_arg < max_positional_args:
                last_required_arg = max_positional_args-1
            if max_positional_args > 0:
                code.putln('switch (pos_args) {')
            for i, arg in enumerate(all_args[:last_required_arg+1]):
                if max_positional_args > 0 and i <= max_positional_args:
                    if i != 0:
                        code.putln('CYTHON_FALLTHROUGH;')
                    if self.star_arg and i == max_positional_args:
                        code.putln('default:')
                    else:
                        code.putln('case %2d:' % i)
                pystring_cname = code.intern_identifier(arg.name)
                if arg.default:
                    if arg.kw_only:
                        # optional kw-only args are handled separately below
                        continue
                    code.putln('if (kw_args > 0) {')
                    # don't overwrite default argument
                    code.putln('PyObject* value = __Pyx_PyDict_GetItemStr(%s, %s);' % (
                        Naming.kwds_cname, pystring_cname))
                    code.putln('if (value) { values[%d] = value; kw_args--; }' % i)
                    code.putln('}')
                else:
                    code.putln('if (likely((values[%d] = __Pyx_PyDict_GetItemStr(%s, %s)) != 0)) kw_args--;' % (
                        i, Naming.kwds_cname, pystring_cname))
                    if i < min_positional_args:
                        if i == 0:
                            # special case: we know arg 0 is missing
                            code.put('else ')
                            code.put_goto(argtuple_error_label)
                        else:
                            # print the correct number of values (args or
                            # kwargs) that were passed into positional
                            # arguments up to this point
                            code.putln('else {')
                            code.globalstate.use_utility_code(
                                UtilityCode.load_cached("RaiseArgTupleInvalid", "FunctionArguments.c"))
                            code.put('__Pyx_RaiseArgtupleInvalid("%s", %d, %d, %d, %d); ' % (
                                self.name, has_fixed_positional_count,
                                min_positional_args, max_positional_args, i))
                            code.putln(code.error_goto(self.pos))
                            code.putln('}')
                    elif arg.kw_only:
                        code.putln('else {')
                        code.globalstate.use_utility_code(
                            UtilityCode.load_cached("RaiseKeywordRequired", "FunctionArguments.c"))
                        code.put('__Pyx_RaiseKeywordRequired("%s", %s); ' % (
                            self.name, pystring_cname))
                        code.putln(code.error_goto(self.pos))
                        code.putln('}')
            if max_positional_args > 0:
                code.putln('}')

        if has_kw_only_args:
            # unpack optional keyword-only arguments separately because
            # checking for interned strings in a dict is faster than iterating
            self.generate_optional_kwonly_args_unpacking_code(all_args, code)

        code.putln('if (unlikely(kw_args > 0)) {')
        # non-positional/-required kw args left in dict: default args,
        # kw-only args, **kwargs or error
        #
        # This is sort of a catch-all: except for checking required
        # arguments, this will always do the right thing for unpacking
        # keyword arguments, so that we can concentrate on optimising
        # common cases above.
        if max_positional_args == 0:
            pos_arg_count = "0"
        elif self.star_arg:
            code.putln("const Py_ssize_t used_pos_args = (pos_args < %d) ? pos_args : %d;" % (
                max_positional_args, max_positional_args))
            pos_arg_count = "used_pos_args"
        else:
            pos_arg_count = "pos_args"
        code.globalstate.use_utility_code(
            UtilityCode.load_cached("ParseKeywords", "FunctionArguments.c"))
        code.putln('if (unlikely(__Pyx_ParseOptionalKeywords(%s, %s, %s, values, %s, "%s") < 0)) %s' % (
            Naming.kwds_cname,
            Naming.pykwdlist_cname,
            self.starstar_arg and self.starstar_arg.entry.cname or '0',
            pos_arg_count,
            self.name,
            code.error_goto(self.pos)))
        code.putln('}')

    def generate_optional_kwonly_args_unpacking_code(self, all_args, code):
        optional_args = []
        first_optional_arg = -1
        for i, arg in enumerate(all_args):
            if not arg.kw_only or not arg.default:
                continue
            if not optional_args:
                first_optional_arg = i
            optional_args.append(arg.name)
        if optional_args:
            if len(optional_args) > 1:
                # if we receive more than the named kwargs, we either have **kwargs
                # (in which case we must iterate anyway) or it's an error (which we
                # also handle during iteration) => skip this part if there are more
                code.putln('if (kw_args > 0 && %s(kw_args <= %d)) {' % (
                    not self.starstar_arg and 'likely' or '',
                    len(optional_args)))
                code.putln('Py_ssize_t index;')
                # not unrolling the loop here reduces the C code overhead
                code.putln('for (index = %d; index < %d && kw_args > 0; index++) {' % (
                    first_optional_arg, first_optional_arg + len(optional_args)))
            else:
                code.putln('if (kw_args == 1) {')
                code.putln('const Py_ssize_t index = %d;' % first_optional_arg)
            code.putln('PyObject* value = __Pyx_PyDict_GetItemStr(%s, *%s[index]);' % (
                Naming.kwds_cname, Naming.pykwdlist_cname))
            code.putln('if (value) { values[index] = value; kw_args--; }')
            if len(optional_args) > 1:
                code.putln('}')
            code.putln('}')

    def generate_argument_conversion_code(self, code):
        # Generate code to convert arguments from signature type to
        # declared type, if needed.  Also copies signature arguments
        # into closure fields.
        for arg in self.args:
            if arg.needs_conversion:
                self.generate_arg_conversion(arg, code)

    def generate_arg_conversion(self, arg, code):
        # Generate conversion code for one argument.
        old_type = arg.hdr_type
        new_type = arg.type
        if old_type.is_pyobject:
            if arg.default:
                code.putln("if (%s) {" % arg.hdr_cname)
            else:
                code.putln("assert(%s); {" % arg.hdr_cname)
            self.generate_arg_conversion_from_pyobject(arg, code)
            code.putln("}")
        elif new_type.is_pyobject:
            self.generate_arg_conversion_to_pyobject(arg, code)
        else:
            if new_type.assignable_from(old_type):
                code.putln("%s = %s;" % (arg.entry.cname, arg.hdr_cname))
            else:
                error(arg.pos, "Cannot convert 1 argument from '%s' to '%s'" % (old_type, new_type))

    def generate_arg_conversion_from_pyobject(self, arg, code):
        new_type = arg.type
        # copied from CoerceFromPyTypeNode
        if new_type.from_py_function:
            code.putln(new_type.from_py_call_code(
                arg.hdr_cname,
                arg.entry.cname,
                arg.pos,
                code,
            ))
        else:
            error(arg.pos, "Cannot convert Python object argument to type '%s'" % new_type)

    def generate_arg_conversion_to_pyobject(self, arg, code):
        old_type = arg.hdr_type
        func = old_type.to_py_function
        if func:
            code.putln("%s = %s(%s); %s" % (
                arg.entry.cname,
                func,
                arg.hdr_cname,
                code.error_goto_if_null(arg.entry.cname, arg.pos)))
            code.put_var_gotref(arg.entry)
        else:
            error(arg.pos, "Cannot convert argument of type '%s' to Python object" % old_type)

    def generate_argument_type_tests(self, code):
        # Generate type tests for args whose signature
        # type is PyObject * and whose declared type is
        # a subtype thereof.
        for arg in self.args:
            if arg.needs_type_test:
                self.generate_arg_type_test(arg, code)
            elif not arg.accept_none and (arg.type.is_pyobject or
                                          arg.type.is_buffer or
                                          arg.type.is_memoryviewslice):
                self.generate_arg_none_check(arg, code)

    def error_value(self):
        return self.signature.error_value


class GeneratorDefNode(DefNode):
    # Generator function node that creates a new generator instance when called.
    #
    # gbody          GeneratorBodyDefNode   the function implementing the generator
    #

    is_generator = True
    is_coroutine = False
    is_iterable_coroutine = False
    is_asyncgen = False
    gen_type_name = 'Generator'
    needs_closure = True

    child_attrs = DefNode.child_attrs + ["gbody"]

    def __init__(self, pos, **kwargs):
        # XXX: don't actually needs a body
        kwargs['body'] = StatListNode(pos, stats=[], is_terminator=True)
        super(GeneratorDefNode, self).__init__(pos, **kwargs)

    def analyse_declarations(self, env):
        super(GeneratorDefNode, self).analyse_declarations(env)
        self.gbody.local_scope = self.local_scope
        self.gbody.analyse_declarations(env)

    def generate_function_body(self, env, code):
        body_cname = self.gbody.entry.func_cname
        name = code.intern_identifier(self.name)
        qualname = code.intern_identifier(self.qualname)
        module_name = code.intern_identifier(self.module_name)

        code.putln('{')
        code.putln('__pyx_CoroutineObject *gen = __Pyx_%s_New('
                   '(__pyx_coroutine_body_t) %s, %s, (PyObject *) %s, %s, %s, %s); %s' % (
                       self.gen_type_name,
                       body_cname, self.code_object.calculate_result_code(code) if self.code_object else 'NULL',
                       Naming.cur_scope_cname, name, qualname, module_name,
                       code.error_goto_if_null('gen', self.pos)))
        code.put_decref(Naming.cur_scope_cname, py_object_type)
        if self.requires_classobj:
            classobj_cname = 'gen->classobj'
            code.putln('%s = __Pyx_CyFunction_GetClassObj(%s);' % (
                classobj_cname, Naming.self_cname))
            code.put_incref(classobj_cname, py_object_type)
            code.put_giveref(classobj_cname)
        code.put_finish_refcount_context()
        code.putln('return (PyObject *) gen;')
        code.putln('}')

    def generate_function_definitions(self, env, code):
        env.use_utility_code(UtilityCode.load_cached(self.gen_type_name, "Coroutine.c"))
        self.gbody.generate_function_header(code, proto=True)
        super(GeneratorDefNode, self).generate_function_definitions(env, code)
        self.gbody.generate_function_definitions(env, code)


class AsyncDefNode(GeneratorDefNode):
    gen_type_name = 'Coroutine'
    is_coroutine = True


class IterableAsyncDefNode(AsyncDefNode):
    gen_type_name = 'IterableCoroutine'
    is_iterable_coroutine = True


class AsyncGenNode(AsyncDefNode):
    gen_type_name = 'AsyncGen'
    is_asyncgen = True


class GeneratorBodyDefNode(DefNode):
    # Main code body of a generator implemented as a DefNode.
    #

    is_generator_body = True
    is_inlined = False
    is_async_gen_body = False
    inlined_comprehension_type = None  # container type for inlined comprehensions

    def __init__(self, pos=None, name=None, body=None, is_async_gen_body=False):
        super(GeneratorBodyDefNode, self).__init__(
            pos=pos, body=body, name=name, is_async_gen_body=is_async_gen_body,
            doc=None, args=[], star_arg=None, starstar_arg=None)

    def declare_generator_body(self, env):
        prefix = env.next_id(env.scope_prefix)
        name = env.next_id('generator')
        cname = Naming.genbody_prefix + prefix + name
        entry = env.declare_var(None, py_object_type, self.pos,
                                cname=cname, visibility='private')
        entry.func_cname = cname
        entry.qualified_name = EncodedString(self.name)
        # Work-around for https://github.com/cython/cython/issues/1699
        # We don't currently determine whether the generator entry is used or not,
        # so mark it as used to avoid false warnings.
        entry.used = True
        self.entry = entry

    def analyse_declarations(self, env):
        self.analyse_argument_types(env)
        self.declare_generator_body(env)

    def generate_function_header(self, code, proto=False):
        header = "static PyObject *%s(__pyx_CoroutineObject *%s, CYTHON_UNUSED PyThreadState *%s, PyObject *%s)" % (
            self.entry.func_cname,
            Naming.generator_cname,
            Naming.local_tstate_cname,
            Naming.sent_value_cname)
        if proto:
            code.putln('%s; /* proto */' % header)
        else:
            code.putln('%s /* generator body */\n{' % header)

    def generate_function_definitions(self, env, code):
        lenv = self.local_scope

        # Generate closure function definitions
        self.body.generate_function_definitions(lenv, code)

        # Generate C code for header and body of function
        code.enter_cfunc_scope(lenv)
        code.return_from_error_cleanup_label = code.new_label()

        # ----- Top-level constants used by this function
        code.mark_pos(self.pos)
        self.generate_cached_builtins_decls(lenv, code)
        # ----- Function header
        code.putln("")
        self.generate_function_header(code)
        closure_init_code = code.insertion_point()
        # ----- Local variables
        code.putln("PyObject *%s = NULL;" % Naming.retval_cname)
        tempvardecl_code = code.insertion_point()
        code.put_declare_refcount_context()
        code.put_setup_refcount_context(self.entry.name or self.entry.qualified_name)
        profile = code.globalstate.directives['profile']
        linetrace = code.globalstate.directives['linetrace']
        if profile or linetrace:
            tempvardecl_code.put_trace_declarations()
            code.funcstate.can_trace = True
            code_object = self.code_object.calculate_result_code(code) if self.code_object else None
            code.put_trace_frame_init(code_object)

        # ----- Resume switch point.
        code.funcstate.init_closure_temps(lenv.scope_class.type.scope)
        resume_code = code.insertion_point()
        first_run_label = code.new_label('first_run')
        code.use_label(first_run_label)
        code.put_label(first_run_label)
        code.putln('%s' %
                   (code.error_goto_if_null(Naming.sent_value_cname, self.pos)))

        # ----- prepare target container for inlined comprehension
        if self.is_inlined and self.inlined_comprehension_type is not None:
            target_type = self.inlined_comprehension_type
            if target_type is Builtin.list_type:
                comp_init = 'PyList_New(0)'
            elif target_type is Builtin.set_type:
                comp_init = 'PySet_New(NULL)'
            elif target_type is Builtin.dict_type:
                comp_init = 'PyDict_New()'
            else:
                raise InternalError(
                    "invalid type of inlined comprehension: %s" % target_type)
            code.putln("%s = %s; %s" % (
                Naming.retval_cname, comp_init,
                code.error_goto_if_null(Naming.retval_cname, self.pos)))
            code.put_gotref(Naming.retval_cname)

        # ----- Function body
        self.generate_function_body(env, code)
        # ----- Closure initialization
        if lenv.scope_class.type.scope.var_entries:
            closure_init_code.putln('%s = %s;' % (
                lenv.scope_class.type.declaration_code(Naming.cur_scope_cname),
                lenv.scope_class.type.cast_code('%s->closure' %
                                                Naming.generator_cname)))
            # FIXME: this silences a potential "unused" warning => try to avoid unused closures in more cases
            code.putln("CYTHON_MAYBE_UNUSED_VAR(%s);" % Naming.cur_scope_cname)

        if profile or linetrace:
            code.funcstate.can_trace = False

        code.mark_pos(self.pos)
        code.putln("")
        code.putln("/* function exit code */")

        # on normal generator termination, we do not take the exception propagation
        # path: no traceback info is required and not creating it is much faster
        if not self.is_inlined and not self.body.is_terminator:
            if self.is_async_gen_body:
                code.globalstate.use_utility_code(
                    UtilityCode.load_cached("StopAsyncIteration", "Coroutine.c"))
            code.putln('PyErr_SetNone(%s);' % (
                '__Pyx_PyExc_StopAsyncIteration' if self.is_async_gen_body else 'PyExc_StopIteration'))
        # ----- Error cleanup
        if code.label_used(code.error_label):
            if not self.body.is_terminator:
                code.put_goto(code.return_label)
            code.put_label(code.error_label)
            if self.is_inlined and self.inlined_comprehension_type is not None:
                code.put_xdecref_clear(Naming.retval_cname, py_object_type)
            if Future.generator_stop in env.global_scope().context.future_directives:
                # PEP 479: turn accidental StopIteration exceptions into a RuntimeError
                code.globalstate.use_utility_code(UtilityCode.load_cached("pep479", "Coroutine.c"))
                code.putln("__Pyx_Generator_Replace_StopIteration(%d);" % bool(self.is_async_gen_body))
            for cname, type in code.funcstate.all_managed_temps():
                code.put_xdecref(cname, type)
            code.put_add_traceback(self.entry.qualified_name)

        # ----- Non-error return cleanup
        code.put_label(code.return_label)
        if self.is_inlined:
            code.put_xgiveref(Naming.retval_cname)
        else:
            code.put_xdecref_clear(Naming.retval_cname, py_object_type)
        # For Py3.7, clearing is already done below.
        code.putln("#if !CYTHON_USE_EXC_INFO_STACK")
        code.putln("__Pyx_Coroutine_ResetAndClearException(%s);" % Naming.generator_cname)
        code.putln("#endif")
        code.putln('%s->resume_label = -1;' % Naming.generator_cname)
        # clean up as early as possible to help breaking any reference cycles
        code.putln('__Pyx_Coroutine_clear((PyObject*)%s);' % Naming.generator_cname)
        if profile or linetrace:
            code.put_trace_return(Naming.retval_cname,
                                  nogil=not code.funcstate.gil_owned)
        code.put_finish_refcount_context()
        code.putln("return %s;" % Naming.retval_cname)
        code.putln("}")

        # ----- Go back and insert temp variable declarations
        tempvardecl_code.put_temp_declarations(code.funcstate)
        # ----- Generator resume code
        if profile or linetrace:
            resume_code.put_trace_call(self.entry.qualified_name, self.pos,
                                       nogil=not code.funcstate.gil_owned)
        resume_code.putln("switch (%s->resume_label) {" % (
                       Naming.generator_cname))

        resume_code.putln("case 0: goto %s;" % first_run_label)

        for i, label in code.yield_labels:
            resume_code.putln("case %d: goto %s;" % (i, label))
        resume_code.putln("default: /* CPython raises the right error here */")
        if profile or linetrace:
            resume_code.put_trace_return("Py_None",
                                         nogil=not code.funcstate.gil_owned)
        resume_code.put_finish_refcount_context()
        resume_code.putln("return NULL;")
        resume_code.putln("}")

        code.exit_cfunc_scope()


class OverrideCheckNode(StatNode):
    # A Node for dispatching to the def method if it
    # is overridden.
    #
    #  py_func
    #
    #  args
    #  func_temp
    #  body

    child_attrs = ['body']

    body = None

    def analyse_expressions(self, env):
        self.args = env.arg_entries
        if self.py_func.is_module_scope:
            first_arg = 0
        else:
            first_arg = 1
        from . import ExprNodes
        self.func_node = ExprNodes.RawCNameExprNode(self.pos, py_object_type)
        call_node = ExprNodes.SimpleCallNode(
            self.pos, function=self.func_node,
            args=[ExprNodes.NameNode(self.pos, name=arg.name)
                  for arg in self.args[first_arg:]])
        if env.return_type.is_void or env.return_type.is_returncode:
            self.body = StatListNode(self.pos, stats=[
                ExprStatNode(self.pos, expr=call_node),
                ReturnStatNode(self.pos, value=None)])
        else:
            self.body = ReturnStatNode(self.pos, value=call_node)
        self.body = self.body.analyse_expressions(env)
        return self

    def generate_execution_code(self, code):
        interned_attr_cname = code.intern_identifier(self.py_func.entry.name)
        # Check to see if we are an extension type
        if self.py_func.is_module_scope:
            self_arg = "((PyObject *)%s)" % Naming.module_cname
        else:
            self_arg = "((PyObject *)%s)" % self.args[0].cname
        code.putln("/* Check if called by wrapper */")
        code.putln("if (unlikely(%s)) ;" % Naming.skip_dispatch_cname)
        code.putln("/* Check if overridden in Python */")
        if self.py_func.is_module_scope:
            code.putln("else {")
        else:
            code.putln("else if (unlikely((Py_TYPE(%s)->tp_dictoffset != 0)"
                       " || (Py_TYPE(%s)->tp_flags & (Py_TPFLAGS_IS_ABSTRACT | Py_TPFLAGS_HEAPTYPE)))) {" % (
                self_arg, self_arg))

        code.putln("#if CYTHON_USE_DICT_VERSIONS && CYTHON_USE_PYTYPE_LOOKUP && CYTHON_USE_TYPE_SLOTS")
        code.globalstate.use_utility_code(
            UtilityCode.load_cached("PyDictVersioning", "ObjectHandling.c"))
        # TODO: remove the object dict version check by 'inlining' the getattr implementation for methods.
        # This would allow checking the dict versions around _PyType_Lookup() if it returns a descriptor,
        # and would (tada!) make this check a pure type based thing instead of supporting only a single
        # instance at a time.
        code.putln("static PY_UINT64_T %s = __PYX_DICT_VERSION_INIT, %s = __PYX_DICT_VERSION_INIT;" % (
            Naming.tp_dict_version_temp, Naming.obj_dict_version_temp))
        code.putln("if (unlikely(!__Pyx_object_dict_version_matches(%s, %s, %s))) {" % (
            self_arg, Naming.tp_dict_version_temp, Naming.obj_dict_version_temp))
        code.putln("PY_UINT64_T %s = __Pyx_get_tp_dict_version(%s);" % (
            Naming.type_dict_guard_temp, self_arg))
        code.putln("#endif")

        func_node_temp = code.funcstate.allocate_temp(py_object_type, manage_ref=True)
        self.func_node.set_cname(func_node_temp)
        # need to get attribute manually--scope would return cdef method
        code.globalstate.use_utility_code(
            UtilityCode.load_cached("PyObjectGetAttrStr", "ObjectHandling.c"))
        err = code.error_goto_if_null(func_node_temp, self.pos)
        code.putln("%s = __Pyx_PyObject_GetAttrStr(%s, %s); %s" % (
            func_node_temp, self_arg, interned_attr_cname, err))
        code.put_gotref(func_node_temp)

        is_builtin_function_or_method = "PyCFunction_Check(%s)" % func_node_temp
        is_overridden = "(PyCFunction_GET_FUNCTION(%s) != (PyCFunction)(void*)%s)" % (
            func_node_temp, self.py_func.entry.func_cname)
        code.putln("if (!%s || %s) {" % (is_builtin_function_or_method, is_overridden))
        self.body.generate_execution_code(code)
        code.putln("}")

        # NOTE: it's not 100% sure that we catch the exact versions here that were used for the lookup,
        # but it is very unlikely that the versions change during lookup, and the type dict safe guard
        # should increase the chance of detecting such a case.
        code.putln("#if CYTHON_USE_DICT_VERSIONS && CYTHON_USE_PYTYPE_LOOKUP && CYTHON_USE_TYPE_SLOTS")
        code.putln("%s = __Pyx_get_tp_dict_version(%s);" % (
            Naming.tp_dict_version_temp, self_arg))
        code.putln("%s = __Pyx_get_object_dict_version(%s);" % (
            Naming.obj_dict_version_temp, self_arg))
        # Safety check that the type dict didn't change during the lookup.  Since CPython looks up the
        # attribute (descriptor) first in the type dict and then in the instance dict or through the
        # descriptor, the only really far-away lookup when we get here is one in the type dict. So we
        # double check the type dict version before and afterwards to guard against later changes of
        # the type dict during the lookup process.
        code.putln("if (unlikely(%s != %s)) {" % (
            Naming.type_dict_guard_temp, Naming.tp_dict_version_temp))
        code.putln("%s = %s = __PYX_DICT_VERSION_INIT;" % (
            Naming.tp_dict_version_temp, Naming.obj_dict_version_temp))
        code.putln("}")
        code.putln("#endif")

        code.put_decref_clear(func_node_temp, PyrexTypes.py_object_type)
        code.funcstate.release_temp(func_node_temp)

        code.putln("#if CYTHON_USE_DICT_VERSIONS && CYTHON_USE_PYTYPE_LOOKUP && CYTHON_USE_TYPE_SLOTS")
        code.putln("}")
        code.putln("#endif")

        code.putln("}")


class ClassDefNode(StatNode, BlockNode):
    pass


class PyClassDefNode(ClassDefNode):
    #  A Python class definition.
    #
    #  name     EncodedString   Name of the class
    #  doc      string or None
    #  body     StatNode        Attribute definition code
    #  entry    Symtab.Entry
    #  scope    PyClassScope
    #  decorators    [DecoratorNode]        list of decorators or None
    #
    #  The following subnodes are constructed internally:
    #
    #  dict     DictNode   Class dictionary or Py3 namespace
    #  classobj ClassNode  Class object
    #  target   NameNode   Variable to assign class object to

    child_attrs = ["body", "dict", "metaclass", "mkw", "bases", "class_result",
                   "target", "class_cell", "decorators"]
    decorators = None
    class_result = None
    is_py3_style_class = False  # Python3 style class (kwargs)
    metaclass = None
    mkw = None

    def __init__(self, pos, name, bases, doc, body, decorators=None,
                 keyword_args=None, force_py3_semantics=False):
        StatNode.__init__(self, pos)
        self.name = name
        self.doc = doc
        self.body = body
        self.decorators = decorators
        self.bases = bases
        from . import ExprNodes
        if self.doc and Options.docstrings:
            doc = embed_position(self.pos, self.doc)
            doc_node = ExprNodes.StringNode(pos, value=doc)
        else:
            doc_node = None

        allow_py2_metaclass = not force_py3_semantics
        if keyword_args:
            allow_py2_metaclass = False
            self.is_py3_style_class = True
            if keyword_args.is_dict_literal:
                if keyword_args.key_value_pairs:
                    for i, item in list(enumerate(keyword_args.key_value_pairs))[::-1]:
                        if item.key.value == 'metaclass':
                            if self.metaclass is not None:
                                error(item.pos, "keyword argument 'metaclass' passed multiple times")
                            # special case: we already know the metaclass,
                            # so we don't need to do the "build kwargs,
                            # find metaclass" dance at runtime
                            self.metaclass = item.value
                            del keyword_args.key_value_pairs[i]
                    self.mkw = keyword_args
                else:
                    assert self.metaclass is not None
            else:
                # MergedDictNode
                self.mkw = ExprNodes.ProxyNode(keyword_args)

        if force_py3_semantics or self.bases or self.mkw or self.metaclass:
            if self.metaclass is None:
                if keyword_args and not keyword_args.is_dict_literal:
                    # **kwargs may contain 'metaclass' arg
                    mkdict = self.mkw
                else:
                    mkdict = None
                if (not mkdict and
                        self.bases.is_sequence_constructor and
                        not self.bases.args):
                    pass  # no base classes => no inherited metaclass
                else:
                    self.metaclass = ExprNodes.PyClassMetaclassNode(
                        pos, class_def_node=self)
                needs_metaclass_calculation = False
            else:
                needs_metaclass_calculation = True

            self.dict = ExprNodes.PyClassNamespaceNode(
                pos, name=name, doc=doc_node, class_def_node=self)
            self.classobj = ExprNodes.Py3ClassNode(
                pos, name=name, class_def_node=self, doc=doc_node,
                calculate_metaclass=needs_metaclass_calculation,
                allow_py2_metaclass=allow_py2_metaclass)
        else:
            # no bases, no metaclass => old style class creation
            self.dict = ExprNodes.DictNode(pos, key_value_pairs=[])
            self.classobj = ExprNodes.ClassNode(
                pos, name=name, class_def_node=self, doc=doc_node)

        self.target = ExprNodes.NameNode(pos, name=name)
        self.class_cell = ExprNodes.ClassCellInjectorNode(self.pos)

    def as_cclass(self):
        """
        Return this node as if it were declared as an extension class
        """
        if self.is_py3_style_class:
            error(self.classobj.pos, "Python3 style class could not be represented as C class")
            return

        from . import ExprNodes
        return CClassDefNode(self.pos,
                             visibility='private',
                             module_name=None,
                             class_name=self.name,
                             bases=self.bases or ExprNodes.TupleNode(self.pos, args=[]),
                             decorators=self.decorators,
                             body=self.body,
                             in_pxd=False,
                             doc=self.doc)

    def create_scope(self, env):
        genv = env
        while genv.is_py_class_scope or genv.is_c_class_scope:
            genv = genv.outer_scope
        cenv = self.scope = PyClassScope(name=self.name, outer_scope=genv)
        return cenv

    def analyse_declarations(self, env):
        class_result = self.classobj
        if self.decorators:
            from .ExprNodes import SimpleCallNode
            for decorator in self.decorators[::-1]:
                class_result = SimpleCallNode(
                    decorator.pos,
                    function=decorator.decorator,
                    args=[class_result])
            self.decorators = None
        self.class_result = class_result
        if self.bases:
            self.bases.analyse_declarations(env)
        if self.mkw:
            self.mkw.analyse_declarations(env)
        self.class_result.analyse_declarations(env)
        self.target.analyse_target_declaration(env)
        cenv = self.create_scope(env)
        cenv.directives = env.directives
        cenv.class_obj_cname = self.target.entry.cname
        self.body.analyse_declarations(cenv)

    def analyse_expressions(self, env):
        if self.bases:
            self.bases = self.bases.analyse_expressions(env)
        if self.mkw:
            self.mkw = self.mkw.analyse_expressions(env)
        if self.metaclass:
            self.metaclass = self.metaclass.analyse_expressions(env)
        self.dict = self.dict.analyse_expressions(env)
        self.class_result = self.class_result.analyse_expressions(env)
        cenv = self.scope
        self.body = self.body.analyse_expressions(cenv)
        self.target.analyse_target_expression(env, self.classobj)
        self.class_cell = self.class_cell.analyse_expressions(cenv)
        return self

    def generate_function_definitions(self, env, code):
        self.generate_lambda_definitions(self.scope, code)
        self.body.generate_function_definitions(self.scope, code)

    def generate_execution_code(self, code):
        code.mark_pos(self.pos)
        code.pyclass_stack.append(self)
        cenv = self.scope
        if self.bases:
            self.bases.generate_evaluation_code(code)
        if self.mkw:
            self.mkw.generate_evaluation_code(code)
        if self.metaclass:
            self.metaclass.generate_evaluation_code(code)
        self.dict.generate_evaluation_code(code)
        cenv.namespace_cname = cenv.class_obj_cname = self.dict.result()

        class_cell = self.class_cell
        if class_cell is not None and not class_cell.is_active:
            class_cell = None

        if class_cell is not None:
            class_cell.generate_evaluation_code(code)
        self.body.generate_execution_code(code)
        self.class_result.generate_evaluation_code(code)
        if class_cell is not None:
            class_cell.generate_injection_code(
                code, self.class_result.result())
        if class_cell is not None:
            class_cell.generate_disposal_code(code)
            class_cell.free_temps(code)

        cenv.namespace_cname = cenv.class_obj_cname = self.classobj.result()
        self.target.generate_assignment_code(self.class_result, code)
        self.dict.generate_disposal_code(code)
        self.dict.free_temps(code)
        if self.metaclass:
            self.metaclass.generate_disposal_code(code)
            self.metaclass.free_temps(code)
        if self.mkw:
            self.mkw.generate_disposal_code(code)
            self.mkw.free_temps(code)
        if self.bases:
            self.bases.generate_disposal_code(code)
            self.bases.free_temps(code)
        code.pyclass_stack.pop()


class CClassDefNode(ClassDefNode):
    #  An extension type definition.
    #
    #  visibility         'private' or 'public' or 'extern'
    #  typedef_flag       boolean
    #  api                boolean
    #  module_name        string or None    For import of extern type objects
    #  class_name         string            Unqualified name of class
    #  as_name            string or None    Name to declare as in this scope
    #  bases              TupleNode         Base class(es)
    #  objstruct_name     string or None    Specified C name of object struct
    #  typeobj_name       string or None    Specified C name of type object
    #  check_size         'warn', 'error', 'ignore'     What to do if tp_basicsize does not match
    #  in_pxd             boolean           Is in a .pxd file
    #  decorators         [DecoratorNode]   list of decorators or None
    #  doc                string or None
    #  body               StatNode or None
    #  entry              Symtab.Entry
    #  base_type          PyExtensionType or None
    #  buffer_defaults_node DictNode or None Declares defaults for a buffer
    #  buffer_defaults_pos

    child_attrs = ["body"]
    buffer_defaults_node = None
    buffer_defaults_pos = None
    typedef_flag = False
    api = False
    objstruct_name = None
    typeobj_name = None
    check_size = None
    decorators = None
    shadow = False

    def buffer_defaults(self, env):
        if not hasattr(self, '_buffer_defaults'):
            from . import Buffer
            if self.buffer_defaults_node:
                self._buffer_defaults = Buffer.analyse_buffer_options(
                    self.buffer_defaults_pos,
                    env, [], self.buffer_defaults_node,
                    need_complete=False)
            else:
                self._buffer_defaults = None
        return self._buffer_defaults

    def declare(self, env):
        if self.module_name and self.visibility != 'extern':
            module_path = self.module_name.split(".")
            home_scope = env.find_imported_module(module_path, self.pos)
            if not home_scope:
                return None
        else:
            home_scope = env

        self.entry = home_scope.declare_c_class(
            name=self.class_name,
            pos=self.pos,
            defining=0,
            implementing=0,
            module_name=self.module_name,
            base_type=None,
            objstruct_cname=self.objstruct_name,
            typeobj_cname=self.typeobj_name,
            visibility=self.visibility,
            typedef_flag=self.typedef_flag,
            check_size = self.check_size,
            api=self.api,
            buffer_defaults=self.buffer_defaults(env),
            shadow=self.shadow)

    def analyse_declarations(self, env):
        #print "CClassDefNode.analyse_declarations:", self.class_name
        #print "...visibility =", self.visibility
        #print "...module_name =", self.module_name

        if env.in_cinclude and not self.objstruct_name:
            error(self.pos, "Object struct name specification required for C class defined in 'extern from' block")
        if self.decorators:
            error(self.pos, "Decorators not allowed on cdef classes (used on type '%s')" % self.class_name)
        self.base_type = None
        # Now that module imports are cached, we need to
        # import the modules for extern classes.
        if self.module_name:
            self.module = None
            for module in env.cimported_modules:
                if module.name == self.module_name:
                    self.module = module
            if self.module is None:
                self.module = ModuleScope(self.module_name, None, env.context)
                self.module.has_extern_class = 1
                env.add_imported_module(self.module)

        if self.bases.args:
            base = self.bases.args[0]
            base_type = base.analyse_as_type(env)
            if base_type in (PyrexTypes.c_int_type, PyrexTypes.c_long_type, PyrexTypes.c_float_type):
                # Use the Python rather than C variant of these types.
                base_type = env.lookup(base_type.sign_and_name()).type
            if base_type is None:
                error(base.pos, "First base of '%s' is not an extension type" % self.class_name)
            elif base_type == PyrexTypes.py_object_type:
                base_class_scope = None
            elif not base_type.is_extension_type and \
                     not (base_type.is_builtin_type and base_type.objstruct_cname):
                error(base.pos, "'%s' is not an extension type" % base_type)
            elif not base_type.is_complete():
                error(base.pos, "Base class '%s' of type '%s' is incomplete" % (
                    base_type.name, self.class_name))
            elif base_type.scope and base_type.scope.directives and \
                     base_type.is_final_type:
                error(base.pos, "Base class '%s' of type '%s' is final" % (
                    base_type, self.class_name))
            elif base_type.is_builtin_type and \
                     base_type.name in ('tuple', 'str', 'bytes'):
                error(base.pos, "inheritance from PyVarObject types like '%s' is not currently supported"
                      % base_type.name)
            else:
                self.base_type = base_type
            if env.directives.get('freelist', 0) > 0 and base_type != PyrexTypes.py_object_type:
                warning(self.pos, "freelists cannot be used on subtypes, only the base class can manage them", 1)

        has_body = self.body is not None
        if has_body and self.base_type and not self.base_type.scope:
            # To properly initialize inherited attributes, the base type must
            # be analysed before this type.
            self.base_type.defered_declarations.append(lambda : self.analyse_declarations(env))
            return

        if self.module_name and self.visibility != 'extern':
            module_path = self.module_name.split(".")
            home_scope = env.find_imported_module(module_path, self.pos)
            if not home_scope:
                return
        else:
            home_scope = env

        if self.visibility == 'extern':
            if (self.module_name == '__builtin__' and
                    self.class_name in Builtin.builtin_types and
                    env.qualified_name[:8] != 'cpython.'): # allow overloaded names for cimporting from cpython
                warning(self.pos, "%s already a builtin Cython type" % self.class_name, 1)

        self.entry = home_scope.declare_c_class(
            name=self.class_name,
            pos=self.pos,
            defining=has_body and self.in_pxd,
            implementing=has_body and not self.in_pxd,
            module_name=self.module_name,
            base_type=self.base_type,
            objstruct_cname=self.objstruct_name,
            typeobj_cname=self.typeobj_name,
            check_size=self.check_size,
            visibility=self.visibility,
            typedef_flag=self.typedef_flag,
            api=self.api,
            buffer_defaults=self.buffer_defaults(env),
            shadow=self.shadow)

        if self.shadow:
            home_scope.lookup(self.class_name).as_variable = self.entry
        if home_scope is not env and self.visibility == 'extern':
            env.add_imported_entry(self.class_name, self.entry, self.pos)
        self.scope = scope = self.entry.type.scope
        if scope is not None:
            scope.directives = env.directives

        if self.doc and Options.docstrings:
            scope.doc = embed_position(self.pos, self.doc)

        if has_body:
            self.body.analyse_declarations(scope)
            dict_entry = self.scope.lookup_here("__dict__")
            if dict_entry and dict_entry.is_variable and (not scope.defined and not scope.implemented):
                dict_entry.getter_cname = self.scope.mangle_internal("__dict__getter")
                self.scope.declare_property("__dict__", dict_entry.doc, dict_entry.pos)
            if self.in_pxd:
                scope.defined = 1
            else:
                scope.implemented = 1

        if len(self.bases.args) > 1:
            if not has_body or self.in_pxd:
                error(self.bases.args[1].pos, "Only declare first base in declaration.")
            # At runtime, we check that the other bases are heap types
            # and that a __dict__ is added if required.
            for other_base in self.bases.args[1:]:
                if other_base.analyse_as_type(env):
                    error(other_base.pos, "Only one extension type base class allowed.")
            self.entry.type.early_init = 0
            from . import ExprNodes
            self.type_init_args = ExprNodes.TupleNode(
                self.pos,
                args=[ExprNodes.IdentifierStringNode(self.pos, value=self.class_name),
                      self.bases,
                      ExprNodes.DictNode(self.pos, key_value_pairs=[])])
        elif self.base_type:
            self.entry.type.early_init = self.base_type.is_external or self.base_type.early_init
            self.type_init_args = None
        else:
            self.entry.type.early_init = 1
            self.type_init_args = None

        env.allocate_vtable_names(self.entry)

        for thunk in self.entry.type.defered_declarations:
            thunk()

    def analyse_expressions(self, env):
        if self.body:
            scope = self.entry.type.scope
            self.body = self.body.analyse_expressions(scope)
        if self.type_init_args:
            self.type_init_args.analyse_expressions(env)
        return self

    def generate_function_definitions(self, env, code):
        if self.body:
            self.generate_lambda_definitions(self.scope, code)
            self.body.generate_function_definitions(self.scope, code)

    def generate_execution_code(self, code):
        # This is needed to generate evaluation code for
        # default values of method arguments.
        code.mark_pos(self.pos)
        if self.body:
            self.body.generate_execution_code(code)
        if not self.entry.type.early_init:
            if self.type_init_args:
                self.type_init_args.generate_evaluation_code(code)
                bases = "PyTuple_GET_ITEM(%s, 1)" % self.type_init_args.result()
                first_base = "((PyTypeObject*)PyTuple_GET_ITEM(%s, 0))" % bases
                # Let Python do the base types compatibility checking.
                trial_type = code.funcstate.allocate_temp(PyrexTypes.py_object_type, True)
                code.putln("%s = PyType_Type.tp_new(&PyType_Type, %s, NULL);" % (
                    trial_type, self.type_init_args.result()))
                code.putln(code.error_goto_if_null(trial_type, self.pos))
                code.put_gotref(trial_type)
                code.putln("if (((PyTypeObject*) %s)->tp_base != %s) {" % (
                    trial_type, first_base))
                code.putln("PyErr_Format(PyExc_TypeError, \"best base '%s' must be equal to first base '%s'\",")
                code.putln("             ((PyTypeObject*) %s)->tp_base->tp_name, %s->tp_name);" % (
                           trial_type, first_base))
                code.putln(code.error_goto(self.pos))
                code.putln("}")
                code.funcstate.release_temp(trial_type)
                code.put_incref(bases, PyrexTypes.py_object_type)
                code.put_giveref(bases)
                code.putln("%s.tp_bases = %s;" % (self.entry.type.typeobj_cname, bases))
                code.put_decref_clear(trial_type, PyrexTypes.py_object_type)
                self.type_init_args.generate_disposal_code(code)
                self.type_init_args.free_temps(code)

            self.generate_type_ready_code(self.entry, code, True)

    # Also called from ModuleNode for early init types.
    @staticmethod
    def generate_type_ready_code(entry, code, heap_type_bases=False):
        # Generate a call to PyType_Ready for an extension
        # type defined in this module.
        type = entry.type
        typeobj_cname = type.typeobj_cname
        scope = type.scope
        if not scope:  # could be None if there was an error
            return
        if entry.visibility != 'extern':
            for slot in TypeSlots.slot_table:
                slot.generate_dynamic_init_code(scope, code)
            if heap_type_bases:
                code.globalstate.use_utility_code(
                    UtilityCode.load_cached('PyType_Ready', 'ExtensionTypes.c'))
                readyfunc = "__Pyx_PyType_Ready"
            else:
                readyfunc = "PyType_Ready"
            code.putln(
                "if (%s(&%s) < 0) %s" % (
                    readyfunc,
                    typeobj_cname,
                    code.error_goto(entry.pos)))
            # Don't inherit tp_print from builtin types, restoring the
            # behavior of using tp_repr or tp_str instead.
            # ("tp_print" was renamed to "tp_vectorcall_offset" in Py3.8b1)
            code.putln("#if PY_VERSION_HEX < 0x030800B1")
            code.putln("%s.tp_print = 0;" % typeobj_cname)
            code.putln("#endif")

            # Use specialised attribute lookup for types with generic lookup but no instance dict.
            getattr_slot_func = TypeSlots.get_slot_code_by_name(scope, 'tp_getattro')
            dictoffset_slot_func = TypeSlots.get_slot_code_by_name(scope, 'tp_dictoffset')
            if getattr_slot_func == '0' and dictoffset_slot_func == '0':
                if type.is_final_type:
                    py_cfunc = "__Pyx_PyObject_GenericGetAttrNoDict"  # grepable
                    utility_func = "PyObject_GenericGetAttrNoDict"
                else:
                    py_cfunc = "__Pyx_PyObject_GenericGetAttr"
                    utility_func = "PyObject_GenericGetAttr"
                code.globalstate.use_utility_code(UtilityCode.load_cached(utility_func, "ObjectHandling.c"))

                code.putln("if ((CYTHON_USE_TYPE_SLOTS && CYTHON_USE_PYTYPE_LOOKUP) &&"
                           " likely(!%s.tp_dictoffset && %s.tp_getattro == PyObject_GenericGetAttr)) {" % (
                    typeobj_cname, typeobj_cname))
                code.putln("%s.tp_getattro = %s;" % (
                    typeobj_cname, py_cfunc))
                code.putln("}")

            # Fix special method docstrings. This is a bit of a hack, but
            # unless we let PyType_Ready create the slot wrappers we have
            # a significant performance hit. (See trac #561.)
            for func in entry.type.scope.pyfunc_entries:
                is_buffer = func.name in ('__getbuffer__', '__releasebuffer__')
                if (func.is_special and Options.docstrings and
                        func.wrapperbase_cname and not is_buffer):
                    slot = TypeSlots.method_name_to_slot.get(func.name)
                    preprocessor_guard = slot.preprocessor_guard_code() if slot else None
                    if preprocessor_guard:
                        code.putln(preprocessor_guard)
                    code.putln('#if CYTHON_COMPILING_IN_CPYTHON')
                    code.putln("{")
                    code.putln(
                        'PyObject *wrapper = PyObject_GetAttrString((PyObject *)&%s, "%s"); %s' % (
                            typeobj_cname,
                            func.name,
                            code.error_goto_if_null('wrapper', entry.pos)))
                    code.putln(
                        "if (Py_TYPE(wrapper) == &PyWrapperDescr_Type) {")
                    code.putln(
                        "%s = *((PyWrapperDescrObject *)wrapper)->d_base;" % (
                            func.wrapperbase_cname))
                    code.putln(
                        "%s.doc = %s;" % (func.wrapperbase_cname, func.doc_cname))
                    code.putln(
                        "((PyWrapperDescrObject *)wrapper)->d_base = &%s;" % (
                            func.wrapperbase_cname))
                    code.putln("}")
                    code.putln("}")
                    code.putln('#endif')
                    if preprocessor_guard:
                        code.putln('#endif')
            if type.vtable_cname:
                code.globalstate.use_utility_code(
                    UtilityCode.load_cached('SetVTable', 'ImportExport.c'))
                code.putln(
                    "if (__Pyx_SetVtable(%s.tp_dict, %s) < 0) %s" % (
                        typeobj_cname,
                        type.vtabptr_cname,
                        code.error_goto(entry.pos)))
                if heap_type_bases:
                    code.globalstate.use_utility_code(
                        UtilityCode.load_cached('MergeVTables', 'ImportExport.c'))
                    code.putln("if (__Pyx_MergeVtables(&%s) < 0) %s" % (
                        typeobj_cname,
                        code.error_goto(entry.pos)))
            if not type.scope.is_internal and not type.scope.directives.get('internal'):
                # scope.is_internal is set for types defined by
                # Cython (such as closures), the 'internal'
                # directive is set by users
                code.putln(
                    'if (PyObject_SetAttr(%s, %s, (PyObject *)&%s) < 0) %s' % (
                        Naming.module_cname,
                        code.intern_identifier(scope.class_name),
                        typeobj_cname,
                        code.error_goto(entry.pos)))
            weakref_entry = scope.lookup_here("__weakref__") if not scope.is_closure_class_scope else None
            if weakref_entry:
                if weakref_entry.type is py_object_type:
                    tp_weaklistoffset = "%s.tp_weaklistoffset" % typeobj_cname
                    if type.typedef_flag:
                        objstruct = type.objstruct_cname
                    else:
                        objstruct = "struct %s" % type.objstruct_cname
                    code.putln("if (%s == 0) %s = offsetof(%s, %s);" % (
                        tp_weaklistoffset,
                        tp_weaklistoffset,
                        objstruct,
                        weakref_entry.cname))
                else:
                    error(weakref_entry.pos, "__weakref__ slot must be of type 'object'")
            if scope.lookup_here("__reduce_cython__") if not scope.is_closure_class_scope else None:
                # Unfortunately, we cannot reliably detect whether a
                # superclass defined __reduce__ at compile time, so we must
                # do so at runtime.
                code.globalstate.use_utility_code(
                    UtilityCode.load_cached('SetupReduce', 'ExtensionTypes.c'))
                code.putln('if (__Pyx_setup_reduce((PyObject*)&%s) < 0) %s' % (
                              typeobj_cname,
                              code.error_goto(entry.pos)))
        # Generate code to initialise the typeptr of an extension
        # type defined in this module to point to its type object.
        if type.typeobj_cname:
            code.putln(
                "%s = &%s;" % (
                    type.typeptr_cname, type.typeobj_cname))

    def annotate(self, code):
        if self.type_init_args:
            self.type_init_args.annotate(code)
        if self.body:
            self.body.annotate(code)


class PropertyNode(StatNode):
    #  Definition of a property in an extension type.
    #
    #  name   string
    #  doc    EncodedString or None    Doc string
    #  entry  Symtab.Entry
    #  body   StatListNode

    child_attrs = ["body"]

    def analyse_declarations(self, env):
        self.entry = env.declare_property(self.name, self.doc, self.pos)
        self.entry.scope.directives = env.directives
        self.body.analyse_declarations(self.entry.scope)

    def analyse_expressions(self, env):
        self.body = self.body.analyse_expressions(env)
        return self

    def generate_function_definitions(self, env, code):
        self.body.generate_function_definitions(env, code)

    def generate_execution_code(self, code):
        pass

    def annotate(self, code):
        self.body.annotate(code)


class GlobalNode(StatNode):
    # Global variable declaration.
    #
    # names    [string]

    child_attrs = []

    def analyse_declarations(self, env):
        for name in self.names:
            env.declare_global(name, self.pos)

    def analyse_expressions(self, env):
        return self

    def generate_execution_code(self, code):
        pass


class NonlocalNode(StatNode):
    # Nonlocal variable declaration via the 'nonlocal' keyword.
    #
    # names    [string]

    child_attrs = []

    def analyse_declarations(self, env):
        for name in self.names:
            env.declare_nonlocal(name, self.pos)

    def analyse_expressions(self, env):
        return self

    def generate_execution_code(self, code):
        pass


class ExprStatNode(StatNode):
    #  Expression used as a statement.
    #
    #  expr   ExprNode

    child_attrs = ["expr"]

    def analyse_declarations(self, env):
        from . import ExprNodes
        expr = self.expr
        if isinstance(expr, ExprNodes.GeneralCallNode):
            func = expr.function.as_cython_attribute()
            if func == u'declare':
                args, kwds = expr.explicit_args_kwds()
                if len(args):
                    error(expr.pos, "Variable names must be specified.")
                for var, type_node in kwds.key_value_pairs:
                    type = type_node.analyse_as_type(env)
                    if type is None:
                        error(type_node.pos, "Unknown type")
                    else:
                        env.declare_var(var.value, type, var.pos, is_cdef=True)
                self.__class__ = PassStatNode
        elif getattr(expr, 'annotation', None) is not None:
            if expr.is_name:
                # non-code variable annotation, e.g. "name: type"
                expr.declare_from_annotation(env)
                self.__class__ = PassStatNode
            elif expr.is_attribute or expr.is_subscript:
                # unused expression with annotation, e.g. "a[0]: type" or "a.xyz : type"
                self.__class__ = PassStatNode

    def analyse_expressions(self, env):
        self.expr.result_is_used = False  # hint that .result() may safely be left empty
        self.expr = self.expr.analyse_expressions(env)
        # Repeat in case of node replacement.
        self.expr.result_is_used = False  # hint that .result() may safely be left empty
        return self

    def nogil_check(self, env):
        if self.expr.type.is_pyobject and self.expr.is_temp:
            self.gil_error()

    gil_message = "Discarding owned Python object"

    def generate_execution_code(self, code):
        code.mark_pos(self.pos)
        self.expr.result_is_used = False  # hint that .result() may safely be left empty
        self.expr.generate_evaluation_code(code)
        if not self.expr.is_temp and self.expr.result():
            result = self.expr.result()
            if not self.expr.type.is_void:
                result = "(void)(%s)" % result
            code.putln("%s;" % result)
        self.expr.generate_disposal_code(code)
        self.expr.free_temps(code)

    def generate_function_definitions(self, env, code):
        self.expr.generate_function_definitions(env, code)

    def annotate(self, code):
        self.expr.annotate(code)


class AssignmentNode(StatNode):
    #  Abstract base class for assignment nodes.
    #
    #  The analyse_expressions and generate_execution_code
    #  phases of assignments are split into two sub-phases
    #  each, to enable all the right hand sides of a
    #  parallel assignment to be evaluated before assigning
    #  to any of the left hand sides.

    def analyse_expressions(self, env):
        node = self.analyse_types(env)
        if isinstance(node, AssignmentNode) and not isinstance(node, ParallelAssignmentNode):
            if node.rhs.type.is_ptr and node.rhs.is_ephemeral():
                error(self.pos, "Storing unsafe C derivative of temporary Python reference")
        return node

#       def analyse_expressions(self, env):
#           self.analyse_expressions_1(env)
#           self.analyse_expressions_2(env)

    def generate_execution_code(self, code):
        code.mark_pos(self.pos)
        self.generate_rhs_evaluation_code(code)
        self.generate_assignment_code(code)


class SingleAssignmentNode(AssignmentNode):
    #  The simplest case:
    #
    #    a = b
    #
    #  lhs                      ExprNode      Left hand side
    #  rhs                      ExprNode      Right hand side
    #  first                    bool          Is this guaranteed the first assignment to lhs?
    #  is_overloaded_assignment bool          Is this assignment done via an overloaded operator=
    #  exception_check
    #  exception_value

    child_attrs = ["lhs", "rhs"]
    first = False
    is_overloaded_assignment = False
    declaration_only = False

    def analyse_declarations(self, env):
        from . import ExprNodes

        # handle declarations of the form x = cython.foo()
        if isinstance(self.rhs, ExprNodes.CallNode):
            func_name = self.rhs.function.as_cython_attribute()
            if func_name:
                args, kwds = self.rhs.explicit_args_kwds()
                if func_name in ['declare', 'typedef']:
                    if len(args) > 2:
                        error(args[2].pos, "Invalid positional argument.")
                        return
                    if kwds is not None:
                        kwdict = kwds.compile_time_value(None)
                        if func_name == 'typedef' or 'visibility' not in kwdict:
                            error(kwds.pos, "Invalid keyword argument.")
                            return
                        visibility = kwdict['visibility']
                    else:
                        visibility = 'private'
                    type = args[0].analyse_as_type(env)
                    if type is None:
                        error(args[0].pos, "Unknown type")
                        return
                    lhs = self.lhs
                    if func_name == 'declare':
                        if isinstance(lhs, ExprNodes.NameNode):
                            vars = [(lhs.name, lhs.pos)]
                        elif isinstance(lhs, ExprNodes.TupleNode):
                            vars = [(var.name, var.pos) for var in lhs.args]
                        else:
                            error(lhs.pos, "Invalid declaration")
                            return
                        for var, pos in vars:
                            env.declare_var(var, type, pos, is_cdef=True, visibility=visibility)
                        if len(args) == 2:
                            # we have a value
                            self.rhs = args[1]
                        else:
                            self.declaration_only = True
                    else:
                        self.declaration_only = True
                        if not isinstance(lhs, ExprNodes.NameNode):
                            error(lhs.pos, "Invalid declaration.")
                        env.declare_typedef(lhs.name, type, self.pos, visibility='private')

                elif func_name in ['struct', 'union']:
                    self.declaration_only = True
                    if len(args) > 0 or kwds is None:
                        error(self.rhs.pos, "Struct or union members must be given by name.")
                        return
                    members = []
                    for member, type_node in kwds.key_value_pairs:
                        type = type_node.analyse_as_type(env)
                        if type is None:
                            error(type_node.pos, "Unknown type")
                        else:
                            members.append((member.value, type, member.pos))
                    if len(members) < len(kwds.key_value_pairs):
                        return
                    if not isinstance(self.lhs, ExprNodes.NameNode):
                        error(self.lhs.pos, "Invalid declaration.")
                    name = self.lhs.name
                    scope = StructOrUnionScope(name)
                    env.declare_struct_or_union(name, func_name, scope, False, self.rhs.pos)
                    for member, type, pos in members:
                        scope.declare_var(member, type, pos)

                elif func_name == 'fused_type':
                    # dtype = cython.fused_type(...)
                    self.declaration_only = True
                    if kwds:
                        error(self.rhs.function.pos,
                              "fused_type does not take keyword arguments")

                    fusednode = FusedTypeNode(self.rhs.pos,
                                              name=self.lhs.name, types=args)
                    fusednode.analyse_declarations(env)

        if self.declaration_only:
            return
        else:
            self.lhs.analyse_target_declaration(env)

    def analyse_types(self, env, use_temp=0):
        from . import ExprNodes

        self.rhs = self.rhs.analyse_types(env)

        unrolled_assignment = self.unroll_rhs(env)
        if unrolled_assignment:
            return unrolled_assignment

        self.lhs = self.lhs.analyse_target_types(env)
        self.lhs.gil_assignment_check(env)
        unrolled_assignment = self.unroll_lhs(env)
        if unrolled_assignment:
            return unrolled_assignment

        if isinstance(self.lhs, ExprNodes.MemoryViewIndexNode):
            self.lhs.analyse_broadcast_operation(self.rhs)
            self.lhs = self.lhs.analyse_as_memview_scalar_assignment(self.rhs)
        elif self.lhs.type.is_array:
            if not isinstance(self.lhs, ExprNodes.SliceIndexNode):
                # cannot assign to C array, only to its full slice
                self.lhs = ExprNodes.SliceIndexNode(self.lhs.pos, base=self.lhs, start=None, stop=None)
                self.lhs = self.lhs.analyse_target_types(env)

        if self.lhs.type.is_cpp_class:
            op = env.lookup_operator_for_types(self.pos, '=', [self.lhs.type, self.rhs.type])
            if op:
                rhs = self.rhs
                self.is_overloaded_assignment = True
                self.exception_check = op.type.exception_check
                self.exception_value = op.type.exception_value
                if self.exception_check == '+' and self.exception_value is None:
                    env.use_utility_code(UtilityCode.load_cached("CppExceptionConversion", "CppSupport.cpp"))
            else:
                rhs = self.rhs.coerce_to(self.lhs.type, env)
        else:
            rhs = self.rhs.coerce_to(self.lhs.type, env)

        if use_temp or rhs.is_attribute or (
                not rhs.is_name and not rhs.is_literal and
                rhs.type.is_pyobject):
            # things like (cdef) attribute access are not safe (traverses pointers)
            rhs = rhs.coerce_to_temp(env)
        elif rhs.type.is_pyobject:
            rhs = rhs.coerce_to_simple(env)
        self.rhs = rhs
        return self

    def unroll(self, node, target_size, env):
        from . import ExprNodes, UtilNodes

        base = node
        start_node = stop_node = step_node = check_node = None

        if node.type.is_ctuple:
            slice_size = node.type.size

        elif node.type.is_ptr or node.type.is_array:
            while isinstance(node, ExprNodes.SliceIndexNode) and not (node.start or node.stop):
                base = node = node.base
            if isinstance(node, ExprNodes.SliceIndexNode):
                base = node.base
                start_node = node.start
                if start_node:
                    start_node = start_node.coerce_to(PyrexTypes.c_py_ssize_t_type, env)
                stop_node = node.stop
                if stop_node:
                    stop_node = stop_node.coerce_to(PyrexTypes.c_py_ssize_t_type, env)
                else:
                    if node.type.is_array and node.type.size:
                        stop_node = ExprNodes.IntNode(
                            self.pos, value=str(node.type.size),
                            constant_result=(node.type.size if isinstance(node.type.size, _py_int_types)
                                             else ExprNodes.constant_value_not_set))
                    else:
                        error(self.pos, "C array iteration requires known end index")
                        return
                step_node = None  #node.step
                if step_node:
                    step_node = step_node.coerce_to(PyrexTypes.c_py_ssize_t_type, env)

                # TODO: Factor out SliceIndexNode.generate_slice_guard_code() for use here.
                def get_const(node, none_value):
                    if node is None:
                        return none_value
                    elif node.has_constant_result():
                        return node.constant_result
                    else:
                        raise ValueError("Not a constant.")

                try:
                    slice_size = (get_const(stop_node, None) - get_const(start_node, 0)) / get_const(step_node, 1)
                except ValueError:
                    error(self.pos, "C array assignment currently requires known endpoints")
                    return

            elif node.type.is_array:
                slice_size = node.type.size
                if not isinstance(slice_size, _py_int_types):
                    return  # might still work when coercing to Python
            else:
                return

        else:
            return

        if slice_size != target_size:
            error(self.pos, "Assignment to/from slice of wrong length, expected %s, got %s" % (
                slice_size, target_size))
            return

        items = []
        base = UtilNodes.LetRefNode(base)
        refs = [base]
        if start_node and not start_node.is_literal:
            start_node = UtilNodes.LetRefNode(start_node)
            refs.append(start_node)
        if stop_node and not stop_node.is_literal:
            stop_node = UtilNodes.LetRefNode(stop_node)
            refs.append(stop_node)
        if step_node and not step_node.is_literal:
            step_node = UtilNodes.LetRefNode(step_node)
            refs.append(step_node)

        for ix in range(target_size):
            ix_node = ExprNodes.IntNode(self.pos, value=str(ix), constant_result=ix, type=PyrexTypes.c_py_ssize_t_type)
            if step_node is not None:
                if step_node.has_constant_result():
                    step_value = ix_node.constant_result * step_node.constant_result
                    ix_node = ExprNodes.IntNode(self.pos, value=str(step_value), constant_result=step_value)
                else:
                    ix_node = ExprNodes.MulNode(self.pos, operator='*', operand1=step_node, operand2=ix_node)
            if start_node is not None:
                if start_node.has_constant_result() and ix_node.has_constant_result():
                    index_value = ix_node.constant_result + start_node.constant_result
                    ix_node = ExprNodes.IntNode(self.pos, value=str(index_value), constant_result=index_value)
                else:
                    ix_node = ExprNodes.AddNode(
                        self.pos, operator='+', operand1=start_node, operand2=ix_node)
            items.append(ExprNodes.IndexNode(self.pos, base=base, index=ix_node.analyse_types(env)))
        return check_node, refs, items

    def unroll_assignments(self, refs, check_node, lhs_list, rhs_list, env):
        from . import UtilNodes
        assignments = []
        for lhs, rhs in zip(lhs_list, rhs_list):
            assignments.append(SingleAssignmentNode(self.pos, lhs=lhs, rhs=rhs, first=self.first))
        node = ParallelAssignmentNode(pos=self.pos, stats=assignments).analyse_expressions(env)
        if check_node:
            node = StatListNode(pos=self.pos, stats=[check_node, node])
        for ref in refs[::-1]:
            node = UtilNodes.LetNode(ref, node)
        return node

    def unroll_rhs(self, env):
        from . import ExprNodes
        if not isinstance(self.lhs, ExprNodes.TupleNode):
            return
        if any(arg.is_starred for arg in self.lhs.args):
            return

        unrolled = self.unroll(self.rhs, len(self.lhs.args), env)
        if not unrolled:
            return
        check_node, refs, rhs = unrolled
        return self.unroll_assignments(refs, check_node, self.lhs.args, rhs, env)

    def unroll_lhs(self, env):
        if self.lhs.type.is_ctuple:
            # Handled directly.
            return
        from . import ExprNodes
        if not isinstance(self.rhs, ExprNodes.TupleNode):
            return

        unrolled = self.unroll(self.lhs, len(self.rhs.args), env)
        if not unrolled:
            return
        check_node, refs, lhs = unrolled
        return self.unroll_assignments(refs, check_node, lhs, self.rhs.args, env)

    def generate_rhs_evaluation_code(self, code):
        self.rhs.generate_evaluation_code(code)

    def generate_assignment_code(self, code, overloaded_assignment=False):
        if self.is_overloaded_assignment:
            self.lhs.generate_assignment_code(
                self.rhs,
                code,
                overloaded_assignment=self.is_overloaded_assignment,
                exception_check=self.exception_check,
                exception_value=self.exception_value)
        else:
            self.lhs.generate_assignment_code(self.rhs, code)

    def generate_function_definitions(self, env, code):
        self.rhs.generate_function_definitions(env, code)

    def annotate(self, code):
        self.lhs.annotate(code)
        self.rhs.annotate(code)


class CascadedAssignmentNode(AssignmentNode):
    #  An assignment with multiple left hand sides:
    #
    #    a = b = c
    #
    #  lhs_list   [ExprNode]   Left hand sides
    #  rhs        ExprNode     Right hand sides
    #
    #  Used internally:
    #
    #  coerced_values       [ExprNode]   RHS coerced to all distinct LHS types
    #  cloned_values        [ExprNode]   cloned RHS value for each LHS
    #  assignment_overloads [Bool]       If each assignment uses a C++ operator=

    child_attrs = ["lhs_list", "rhs", "coerced_values", "cloned_values"]
    cloned_values = None
    coerced_values = None
    assignment_overloads = None

    def analyse_declarations(self, env):
        for lhs in self.lhs_list:
            lhs.analyse_target_declaration(env)

    def analyse_types(self, env, use_temp=0):
        from .ExprNodes import CloneNode, ProxyNode

        # collect distinct types used on the LHS
        lhs_types = set()
        for i, lhs in enumerate(self.lhs_list):
            lhs = self.lhs_list[i] = lhs.analyse_target_types(env)
            lhs.gil_assignment_check(env)
            lhs_types.add(lhs.type)

        rhs = self.rhs.analyse_types(env)
        # common special case: only one type needed on the LHS => coerce only once
        if len(lhs_types) == 1:
            # Avoid coercion for overloaded assignment operators.
            if next(iter(lhs_types)).is_cpp_class:
                op = env.lookup_operator('=', [lhs, self.rhs])
                if not op:
                    rhs = rhs.coerce_to(lhs_types.pop(), env)
            else:
                rhs = rhs.coerce_to(lhs_types.pop(), env)

        if not rhs.is_name and not rhs.is_literal and (
                use_temp or rhs.is_attribute or rhs.type.is_pyobject):
            rhs = rhs.coerce_to_temp(env)
        else:
            rhs = rhs.coerce_to_simple(env)
        self.rhs = ProxyNode(rhs) if rhs.is_temp else rhs

        # clone RHS and coerce it to all distinct LHS types
        self.coerced_values = []
        coerced_values = {}
        self.assignment_overloads = []
        for lhs in self.lhs_list:
            overloaded = lhs.type.is_cpp_class and env.lookup_operator('=', [lhs, self.rhs])
            self.assignment_overloads.append(overloaded)
            if lhs.type not in coerced_values and lhs.type != rhs.type:
                rhs = CloneNode(self.rhs)
                if not overloaded:
                    rhs = rhs.coerce_to(lhs.type, env)
                self.coerced_values.append(rhs)
                coerced_values[lhs.type] = rhs

        # clone coerced values for all LHS assignments
        self.cloned_values = []
        for lhs in self.lhs_list:
            rhs = coerced_values.get(lhs.type, self.rhs)
            self.cloned_values.append(CloneNode(rhs))
        return self

    def generate_rhs_evaluation_code(self, code):
        self.rhs.generate_evaluation_code(code)

    def generate_assignment_code(self, code, overloaded_assignment=False):
        # prepare all coercions
        for rhs in self.coerced_values:
            rhs.generate_evaluation_code(code)
        # assign clones to LHS
        for lhs, rhs, overload in zip(self.lhs_list, self.cloned_values, self.assignment_overloads):
            rhs.generate_evaluation_code(code)
            lhs.generate_assignment_code(rhs, code, overloaded_assignment=overload)
        # dispose of coerced values and original RHS
        for rhs_value in self.coerced_values:
            rhs_value.generate_disposal_code(code)
            rhs_value.free_temps(code)
        self.rhs.generate_disposal_code(code)
        self.rhs.free_temps(code)

    def generate_function_definitions(self, env, code):
        self.rhs.generate_function_definitions(env, code)

    def annotate(self, code):
        for rhs in self.coerced_values:
            rhs.annotate(code)
        for lhs, rhs in zip(self.lhs_list, self.cloned_values):
            lhs.annotate(code)
            rhs.annotate(code)
        self.rhs.annotate(code)


class ParallelAssignmentNode(AssignmentNode):
    #  A combined packing/unpacking assignment:
    #
    #    a, b, c =  d, e, f
    #
    #  This has been rearranged by the parser into
    #
    #    a = d ; b = e ; c = f
    #
    #  but we must evaluate all the right hand sides
    #  before assigning to any of the left hand sides.
    #
    #  stats     [AssignmentNode]   The constituent assignments

    child_attrs = ["stats"]

    def analyse_declarations(self, env):
        for stat in self.stats:
            stat.analyse_declarations(env)

    def analyse_expressions(self, env):
        self.stats = [stat.analyse_types(env, use_temp=1)
                      for stat in self.stats]
        return self

#    def analyse_expressions(self, env):
#        for stat in self.stats:
#            stat.analyse_expressions_1(env, use_temp=1)
#        for stat in self.stats:
#            stat.analyse_expressions_2(env)

    def generate_execution_code(self, code):
        code.mark_pos(self.pos)
        for stat in self.stats:
            stat.generate_rhs_evaluation_code(code)
        for stat in self.stats:
            stat.generate_assignment_code(code)

    def generate_function_definitions(self, env, code):
        for stat in self.stats:
            stat.generate_function_definitions(env, code)

    def annotate(self, code):
        for stat in self.stats:
            stat.annotate(code)


class InPlaceAssignmentNode(AssignmentNode):
    #  An in place arithmetic operand:
    #
    #    a += b
    #    a -= b
    #    ...
    #
    #  lhs      ExprNode      Left hand side
    #  rhs      ExprNode      Right hand side
    #  operator char          one of "+-*/%^&|"
    #
    #  This code is a bit tricky because in order to obey Python
    #  semantics the sub-expressions (e.g. indices) of the lhs must
    #  not be evaluated twice. So we must re-use the values calculated
    #  in evaluation phase for the assignment phase as well.
    #  Fortunately, the type of the lhs node is fairly constrained
    #  (it must be a NameNode, AttributeNode, or IndexNode).

    child_attrs = ["lhs", "rhs"]

    def analyse_declarations(self, env):
        self.lhs.analyse_target_declaration(env)

    def analyse_types(self, env):
        self.rhs = self.rhs.analyse_types(env)
        self.lhs = self.lhs.analyse_target_types(env)

        # When assigning to a fully indexed buffer or memoryview, coerce the rhs
        if self.lhs.is_memview_index or self.lhs.is_buffer_access:
            self.rhs = self.rhs.coerce_to(self.lhs.type, env)
        elif self.lhs.type.is_string and self.operator in '+-':
            # use pointer arithmetic for char* LHS instead of string concat
            self.rhs = self.rhs.coerce_to(PyrexTypes.c_py_ssize_t_type, env)
        return self

    def generate_execution_code(self, code):
        code.mark_pos(self.pos)
        lhs, rhs = self.lhs, self.rhs
        rhs.generate_evaluation_code(code)
        lhs.generate_subexpr_evaluation_code(code)
        c_op = self.operator
        if c_op == "//":
            c_op = "/"
        elif c_op == "**":
            error(self.pos, "No C inplace power operator")
        if lhs.is_buffer_access or lhs.is_memview_index:
            if lhs.type.is_pyobject:
                error(self.pos, "In-place operators not allowed on object buffers in this release.")
            if c_op in ('/', '%') and lhs.type.is_int and not code.globalstate.directives['cdivision']:
                error(self.pos, "In-place non-c divide operators not allowed on int buffers.")
            lhs.generate_buffer_setitem_code(rhs, code, c_op)
        elif lhs.is_memview_slice:
            error(self.pos, "Inplace operators not supported on memoryview slices")
        else:
            # C++
            # TODO: make sure overload is declared
            code.putln("%s %s= %s;" % (lhs.result(), c_op, rhs.result()))
        lhs.generate_subexpr_disposal_code(code)
        lhs.free_subexpr_temps(code)
        rhs.generate_disposal_code(code)
        rhs.free_temps(code)

    def annotate(self, code):
        self.lhs.annotate(code)
        self.rhs.annotate(code)

    def create_binop_node(self):
        from . import ExprNodes
        return ExprNodes.binop_node(self.pos, self.operator, self.lhs, self.rhs)


class PrintStatNode(StatNode):
    #  print statement
    #
    #  arg_tuple         TupleNode
    #  stream            ExprNode or None (stdout)
    #  append_newline    boolean

    child_attrs = ["arg_tuple", "stream"]

    def analyse_expressions(self, env):
        if self.stream:
            stream = self.stream.analyse_expressions(env)
            self.stream = stream.coerce_to_pyobject(env)
        arg_tuple = self.arg_tuple.analyse_expressions(env)
        self.arg_tuple = arg_tuple.coerce_to_pyobject(env)
        env.use_utility_code(printing_utility_code)
        if len(self.arg_tuple.args) == 1 and self.append_newline:
            env.use_utility_code(printing_one_utility_code)
        return self

    nogil_check = Node.gil_error
    gil_message = "Python print statement"

    def generate_execution_code(self, code):
        code.mark_pos(self.pos)
        if self.stream:
            self.stream.generate_evaluation_code(code)
            stream_result = self.stream.py_result()
        else:
            stream_result = '0'
        if len(self.arg_tuple.args) == 1 and self.append_newline:
            arg = self.arg_tuple.args[0]
            arg.generate_evaluation_code(code)

            code.putln(
                "if (__Pyx_PrintOne(%s, %s) < 0) %s" % (
                    stream_result,
                    arg.py_result(),
                    code.error_goto(self.pos)))
            arg.generate_disposal_code(code)
            arg.free_temps(code)
        else:
            self.arg_tuple.generate_evaluation_code(code)
            code.putln(
                "if (__Pyx_Print(%s, %s, %d) < 0) %s" % (
                    stream_result,
                    self.arg_tuple.py_result(),
                    self.append_newline,
                    code.error_goto(self.pos)))
            self.arg_tuple.generate_disposal_code(code)
            self.arg_tuple.free_temps(code)

        if self.stream:
            self.stream.generate_disposal_code(code)
            self.stream.free_temps(code)

    def generate_function_definitions(self, env, code):
        if self.stream:
            self.stream.generate_function_definitions(env, code)
        self.arg_tuple.generate_function_definitions(env, code)

    def annotate(self, code):
        if self.stream:
            self.stream.annotate(code)
        self.arg_tuple.annotate(code)


class ExecStatNode(StatNode):
    #  exec statement
    #
    #  args     [ExprNode]

    child_attrs = ["args"]

    def analyse_expressions(self, env):
        for i, arg in enumerate(self.args):
            arg = arg.analyse_expressions(env)
            arg = arg.coerce_to_pyobject(env)
            self.args[i] = arg
        env.use_utility_code(Builtin.pyexec_utility_code)
        return self

    nogil_check = Node.gil_error
    gil_message = "Python exec statement"

    def generate_execution_code(self, code):
        code.mark_pos(self.pos)
        args = []
        for arg in self.args:
            arg.generate_evaluation_code(code)
            args.append(arg.py_result())
        args = tuple(args + ['0', '0'][:3-len(args)])
        temp_result = code.funcstate.allocate_temp(PyrexTypes.py_object_type, manage_ref=True)
        code.putln("%s = __Pyx_PyExec3(%s, %s, %s);" % ((temp_result,) + args))
        for arg in self.args:
            arg.generate_disposal_code(code)
            arg.free_temps(code)
        code.putln(
            code.error_goto_if_null(temp_result, self.pos))
        code.put_gotref(temp_result)
        code.put_decref_clear(temp_result, py_object_type)
        code.funcstate.release_temp(temp_result)

    def annotate(self, code):
        for arg in self.args:
            arg.annotate(code)


class DelStatNode(StatNode):
    #  del statement
    #
    #  args     [ExprNode]

    child_attrs = ["args"]
    ignore_nonexisting = False

    def analyse_declarations(self, env):
        for arg in self.args:
            arg.analyse_target_declaration(env)

    def analyse_expressions(self, env):
        for i, arg in enumerate(self.args):
            arg = self.args[i] = arg.analyse_target_expression(env, None)
            if arg.type.is_pyobject or (arg.is_name and arg.type.is_memoryviewslice):
                if arg.is_name and arg.entry.is_cglobal:
                    error(arg.pos, "Deletion of global C variable")
            elif arg.type.is_ptr and arg.type.base_type.is_cpp_class:
                self.cpp_check(env)
            elif arg.type.is_cpp_class:
                error(arg.pos, "Deletion of non-heap C++ object")
            elif arg.is_subscript and arg.base.type is Builtin.bytearray_type:
                pass  # del ba[i]
            else:
                error(arg.pos, "Deletion of non-Python, non-C++ object")
            #arg.release_target_temp(env)
        return self

    def nogil_check(self, env):
        for arg in self.args:
            if arg.type.is_pyobject:
                self.gil_error()

    gil_message = "Deleting Python object"

    def generate_execution_code(self, code):
        code.mark_pos(self.pos)
        for arg in self.args:
            if (arg.type.is_pyobject or
                    arg.type.is_memoryviewslice or
                    arg.is_subscript and arg.base.type is Builtin.bytearray_type):
                arg.generate_deletion_code(
                    code, ignore_nonexisting=self.ignore_nonexisting)
            elif arg.type.is_ptr and arg.type.base_type.is_cpp_class:
                arg.generate_evaluation_code(code)
                code.putln("delete %s;" % arg.result())
                arg.generate_disposal_code(code)
                arg.free_temps(code)
            # else error reported earlier

    def annotate(self, code):
        for arg in self.args:
            arg.annotate(code)


class PassStatNode(StatNode):
    #  pass statement

    child_attrs = []

    def analyse_expressions(self, env):
        return self

    def generate_execution_code(self, code):
        pass


class IndirectionNode(StatListNode):
    """
    This adds an indirection so that the node can be shared and a subtree can
    be removed at any time by clearing self.stats.
    """

    def __init__(self, stats):
        super(IndirectionNode, self).__init__(stats[0].pos, stats=stats)


class BreakStatNode(StatNode):

    child_attrs = []
    is_terminator = True

    def analyse_expressions(self, env):
        return self

    def generate_execution_code(self, code):
        code.mark_pos(self.pos)
        if not code.break_label:
            error(self.pos, "break statement not inside loop")
        else:
            code.put_goto(code.break_label)


class ContinueStatNode(StatNode):

    child_attrs = []
    is_terminator = True

    def analyse_expressions(self, env):
        return self

    def generate_execution_code(self, code):
        if not code.continue_label:
            error(self.pos, "continue statement not inside loop")
            return
        code.mark_pos(self.pos)
        code.put_goto(code.continue_label)


class ReturnStatNode(StatNode):
    #  return statement
    #
    #  value         ExprNode or None
    #  return_type   PyrexType
    #  in_generator  return inside of generator => raise StopIteration
    #  in_async_gen  return inside of async generator

    child_attrs = ["value"]
    is_terminator = True
    in_generator = False
    in_async_gen = False

    # Whether we are in a parallel section
    in_parallel = False

    def analyse_expressions(self, env):
        return_type = env.return_type
        self.return_type = return_type
        if not return_type:
            error(self.pos, "Return not inside a function body")
            return self
        if self.value:
            if self.in_async_gen:
                error(self.pos, "Return with value in async generator")
            self.value = self.value.analyse_types(env)
            if return_type.is_void or return_type.is_returncode:
                error(self.value.pos, "Return with value in void function")
            else:
                self.value = self.value.coerce_to(env.return_type, env)
        else:
            if (not return_type.is_void
                    and not return_type.is_pyobject
                    and not return_type.is_returncode):
                error(self.pos, "Return value required")
        return self

    def nogil_check(self, env):
        if self.return_type.is_pyobject:
            self.gil_error()

    gil_message = "Returning Python object"

    def generate_execution_code(self, code):
        code.mark_pos(self.pos)
        if not self.return_type:
            # error reported earlier
            return

        value = self.value
        if self.return_type.is_pyobject:
            code.put_xdecref(Naming.retval_cname, self.return_type)
            if value and value.is_none:
                # Use specialised default handling for "return None".
                value = None

        if value:
            value.generate_evaluation_code(code)
            if self.return_type.is_memoryviewslice:
                from . import MemoryView
                MemoryView.put_acquire_memoryviewslice(
                    lhs_cname=Naming.retval_cname,
                    lhs_type=self.return_type,
                    lhs_pos=value.pos,
                    rhs=value,
                    code=code,
                    have_gil=self.in_nogil_context)
                value.generate_post_assignment_code(code)
            elif self.in_generator:
                # return value == raise StopIteration(value), but uncatchable
                code.globalstate.use_utility_code(
                    UtilityCode.load_cached("ReturnWithStopIteration", "Coroutine.c"))
                code.putln("%s = NULL; __Pyx_ReturnWithStopIteration(%s);" % (
                    Naming.retval_cname,
                    value.py_result()))
                value.generate_disposal_code(code)
            else:
                value.make_owned_reference(code)
                code.putln("%s = %s;" % (
                    Naming.retval_cname,
                    value.result_as(self.return_type)))
                value.generate_post_assignment_code(code)
            value.free_temps(code)
        else:
            if self.return_type.is_pyobject:
                if self.in_generator:
                    if self.in_async_gen:
                        code.globalstate.use_utility_code(
                            UtilityCode.load_cached("StopAsyncIteration", "Coroutine.c"))
                        code.put("PyErr_SetNone(__Pyx_PyExc_StopAsyncIteration); ")
                    code.putln("%s = NULL;" % Naming.retval_cname)
                else:
                    code.put_init_to_py_none(Naming.retval_cname, self.return_type)
            elif self.return_type.is_returncode:
                self.put_return(code, self.return_type.default_value)

        for cname, type in code.funcstate.temps_holding_reference():
            code.put_decref_clear(cname, type)

        code.put_goto(code.return_label)

    def put_return(self, code, value):
        if self.in_parallel:
            code.putln_openmp("#pragma omp critical(__pyx_returning)")
        code.putln("%s = %s;" % (Naming.retval_cname, value))

    def generate_function_definitions(self, env, code):
        if self.value is not None:
            self.value.generate_function_definitions(env, code)

    def annotate(self, code):
        if self.value:
            self.value.annotate(code)


class RaiseStatNode(StatNode):
    #  raise statement
    #
    #  exc_type    ExprNode or None
    #  exc_value   ExprNode or None
    #  exc_tb      ExprNode or None
    #  cause       ExprNode or None

    child_attrs = ["exc_type", "exc_value", "exc_tb", "cause"]
    is_terminator = True

    def analyse_expressions(self, env):
        if self.exc_type:
            exc_type = self.exc_type.analyse_types(env)
            self.exc_type = exc_type.coerce_to_pyobject(env)
        if self.exc_value:
            exc_value = self.exc_value.analyse_types(env)
            self.exc_value = exc_value.coerce_to_pyobject(env)
        if self.exc_tb:
            exc_tb = self.exc_tb.analyse_types(env)
            self.exc_tb = exc_tb.coerce_to_pyobject(env)
        if self.cause:
            cause = self.cause.analyse_types(env)
            self.cause = cause.coerce_to_pyobject(env)
        # special cases for builtin exceptions
        self.builtin_exc_name = None
        if self.exc_type and not self.exc_value and not self.exc_tb:
            exc = self.exc_type
            from . import ExprNodes
            if (isinstance(exc, ExprNodes.SimpleCallNode) and
                    not (exc.args or (exc.arg_tuple is not None and exc.arg_tuple.args))):
                exc = exc.function  # extract the exception type
            if exc.is_name and exc.entry.is_builtin:
                self.builtin_exc_name = exc.name
                if self.builtin_exc_name == 'MemoryError':
                    self.exc_type = None # has a separate implementation
        return self

    nogil_check = Node.gil_error
    gil_message = "Raising exception"

    def generate_execution_code(self, code):
        code.mark_pos(self.pos)
        if self.builtin_exc_name == 'MemoryError':
            code.putln('PyErr_NoMemory(); %s' % code.error_goto(self.pos))
            return

        if self.exc_type:
            self.exc_type.generate_evaluation_code(code)
            type_code = self.exc_type.py_result()
            if self.exc_type.is_name:
                code.globalstate.use_entry_utility_code(self.exc_type.entry)
        else:
            type_code = "0"
        if self.exc_value:
            self.exc_value.generate_evaluation_code(code)
            value_code = self.exc_value.py_result()
        else:
            value_code = "0"
        if self.exc_tb:
            self.exc_tb.generate_evaluation_code(code)
            tb_code = self.exc_tb.py_result()
        else:
            tb_code = "0"
        if self.cause:
            self.cause.generate_evaluation_code(code)
            cause_code = self.cause.py_result()
        else:
            cause_code = "0"
        code.globalstate.use_utility_code(raise_utility_code)
        code.putln(
            "__Pyx_Raise(%s, %s, %s, %s);" % (
                type_code,
                value_code,
                tb_code,
                cause_code))
        for obj in (self.exc_type, self.exc_value, self.exc_tb, self.cause):
            if obj:
                obj.generate_disposal_code(code)
                obj.free_temps(code)
        code.putln(
            code.error_goto(self.pos))

    def generate_function_definitions(self, env, code):
        if self.exc_type is not None:
            self.exc_type.generate_function_definitions(env, code)
        if self.exc_value is not None:
            self.exc_value.generate_function_definitions(env, code)
        if self.exc_tb is not None:
            self.exc_tb.generate_function_definitions(env, code)
        if self.cause is not None:
            self.cause.generate_function_definitions(env, code)

    def annotate(self, code):
        if self.exc_type:
            self.exc_type.annotate(code)
        if self.exc_value:
            self.exc_value.annotate(code)
        if self.exc_tb:
            self.exc_tb.annotate(code)
        if self.cause:
            self.cause.annotate(code)


class ReraiseStatNode(StatNode):

    child_attrs = []
    is_terminator = True

    def analyse_expressions(self, env):
        return self

    nogil_check = Node.gil_error
    gil_message = "Raising exception"

    def generate_execution_code(self, code):
        code.mark_pos(self.pos)
        vars = code.funcstate.exc_vars
        if vars:
            code.globalstate.use_utility_code(restore_exception_utility_code)
            code.put_giveref(vars[0])
            code.put_giveref(vars[1])
            # fresh exceptions may not have a traceback yet (-> finally!)
            code.put_xgiveref(vars[2])
            code.putln("__Pyx_ErrRestoreWithState(%s, %s, %s);" % tuple(vars))
            for varname in vars:
                code.put("%s = 0; " % varname)
            code.putln()
            code.putln(code.error_goto(self.pos))
        else:
            code.globalstate.use_utility_code(
                UtilityCode.load_cached("ReRaiseException", "Exceptions.c"))
            code.putln("__Pyx_ReraiseException(); %s" % code.error_goto(self.pos))

class AssertStatNode(StatNode):
    #  assert statement
    #
    #  cond    ExprNode
    #  value   ExprNode or None

    child_attrs = ["cond", "value"]

    def analyse_expressions(self, env):
        self.cond = self.cond.analyse_boolean_expression(env)
        if self.value:
            value = self.value.analyse_types(env)
            if value.type is Builtin.tuple_type or not value.type.is_builtin_type:
                # prevent tuple values from being interpreted as argument value tuples
                from .ExprNodes import TupleNode
                value = TupleNode(value.pos, args=[value], slow=True)
                self.value = value.analyse_types(env, skip_children=True).coerce_to_pyobject(env)
            else:
                self.value = value.coerce_to_pyobject(env)
        return self

    nogil_check = Node.gil_error
    gil_message = "Raising exception"

    def generate_execution_code(self, code):
        code.putln("#ifndef CYTHON_WITHOUT_ASSERTIONS")
        code.putln("if (unlikely(!Py_OptimizeFlag)) {")
        code.mark_pos(self.pos)
        self.cond.generate_evaluation_code(code)
        code.putln(
            "if (unlikely(!%s)) {" % self.cond.result())
        if self.value:
            self.value.generate_evaluation_code(code)
            code.putln(
                "PyErr_SetObject(PyExc_AssertionError, %s);" % self.value.py_result())
            self.value.generate_disposal_code(code)
            self.value.free_temps(code)
        else:
            code.putln(
                "PyErr_SetNone(PyExc_AssertionError);")
        code.putln(
            code.error_goto(self.pos))
        code.putln(
            "}")
        self.cond.generate_disposal_code(code)
        self.cond.free_temps(code)
        code.putln(
            "}")
        code.putln("#endif")

    def generate_function_definitions(self, env, code):
        self.cond.generate_function_definitions(env, code)
        if self.value is not None:
            self.value.generate_function_definitions(env, code)

    def annotate(self, code):
        self.cond.annotate(code)
        if self.value:
            self.value.annotate(code)


class IfStatNode(StatNode):
    #  if statement
    #
    #  if_clauses   [IfClauseNode]
    #  else_clause  StatNode or None

    child_attrs = ["if_clauses", "else_clause"]

    def analyse_declarations(self, env):
        for if_clause in self.if_clauses:
            if_clause.analyse_declarations(env)
        if self.else_clause:
            self.else_clause.analyse_declarations(env)

    def analyse_expressions(self, env):
        self.if_clauses = [if_clause.analyse_expressions(env) for if_clause in self.if_clauses]
        if self.else_clause:
            self.else_clause = self.else_clause.analyse_expressions(env)
        return self

    def generate_execution_code(self, code):
        code.mark_pos(self.pos)
        end_label = code.new_label()
        last = len(self.if_clauses)
        if self.else_clause:
            # If the 'else' clause is 'unlikely', then set the preceding 'if' clause to 'likely' to reflect that.
            self._set_branch_hint(self.if_clauses[-1], self.else_clause, inverse=True)
        else:
            last -= 1  # avoid redundant goto at end of last if-clause
        for i, if_clause in enumerate(self.if_clauses):
            self._set_branch_hint(if_clause, if_clause.body)
            if_clause.generate_execution_code(code, end_label, is_last=i == last)
        if self.else_clause:
            code.mark_pos(self.else_clause.pos)
            code.putln("/*else*/ {")
            self.else_clause.generate_execution_code(code)
            code.putln("}")
        code.put_label(end_label)

    def _set_branch_hint(self, clause, statements_node, inverse=False):
        if not statements_node.is_terminator:
            return
        if not isinstance(statements_node, StatListNode) or not statements_node.stats:
            return
        # Anything that unconditionally raises exceptions should be considered unlikely.
        if isinstance(statements_node.stats[-1], (RaiseStatNode, ReraiseStatNode)):
            if len(statements_node.stats) > 1:
                # Allow simple statements before the 'raise', but no conditions, loops, etc.
                non_branch_nodes = (ExprStatNode, AssignmentNode, DelStatNode, GlobalNode, NonlocalNode)
                for node in statements_node.stats[:-1]:
                    if not isinstance(node, non_branch_nodes):
                        return
            clause.branch_hint = 'likely' if inverse else 'unlikely'

    def generate_function_definitions(self, env, code):
        for clause in self.if_clauses:
            clause.generate_function_definitions(env, code)
        if self.else_clause is not None:
            self.else_clause.generate_function_definitions(env, code)

    def annotate(self, code):
        for if_clause in self.if_clauses:
            if_clause.annotate(code)
        if self.else_clause:
            self.else_clause.annotate(code)


class IfClauseNode(Node):
    #  if or elif clause in an if statement
    #
    #  condition   ExprNode
    #  body        StatNode

    child_attrs = ["condition", "body"]
    branch_hint = None

    def analyse_declarations(self, env):
        self.body.analyse_declarations(env)

    def analyse_expressions(self, env):
        self.condition = self.condition.analyse_temp_boolean_expression(env)
        self.body = self.body.analyse_expressions(env)
        return self

    def generate_execution_code(self, code, end_label, is_last):
        self.condition.generate_evaluation_code(code)
        code.mark_pos(self.pos)
        condition = self.condition.result()
        if self.branch_hint:
            condition = '%s(%s)' % (self.branch_hint, condition)
        code.putln("if (%s) {" % condition)
        self.condition.generate_disposal_code(code)
        self.condition.free_temps(code)
        self.body.generate_execution_code(code)
        code.mark_pos(self.pos, trace=False)
        if not (is_last or self.body.is_terminator):
            code.put_goto(end_label)
        code.putln("}")

    def generate_function_definitions(self, env, code):
        self.condition.generate_function_definitions(env, code)
        self.body.generate_function_definitions(env, code)

    def annotate(self, code):
        self.condition.annotate(code)
        self.body.annotate(code)


class SwitchCaseNode(StatNode):
    # Generated in the optimization of an if-elif-else node
    #
    # conditions    [ExprNode]
    # body          StatNode

    child_attrs = ['conditions', 'body']

    def generate_condition_evaluation_code(self, code):
        for cond in self.conditions:
            cond.generate_evaluation_code(code)

    def generate_execution_code(self, code):
        num_conditions = len(self.conditions)
        line_tracing_enabled = code.globalstate.directives['linetrace']
        for i, cond in enumerate(self.conditions, 1):
            code.putln("case %s:" % cond.result())
            code.mark_pos(cond.pos)  # Tracing code must appear *after* the 'case' statement.
            if line_tracing_enabled and i < num_conditions:
                # Allow fall-through after the line tracing code.
                code.putln('CYTHON_FALLTHROUGH;')
        self.body.generate_execution_code(code)
        code.mark_pos(self.pos, trace=False)
        code.putln("break;")

    def generate_function_definitions(self, env, code):
        for cond in self.conditions:
            cond.generate_function_definitions(env, code)
        self.body.generate_function_definitions(env, code)

    def annotate(self, code):
        for cond in self.conditions:
            cond.annotate(code)
        self.body.annotate(code)


class SwitchStatNode(StatNode):
    # Generated in the optimization of an if-elif-else node
    #
    # test          ExprNode
    # cases         [SwitchCaseNode]
    # else_clause   StatNode or None

    child_attrs = ['test', 'cases', 'else_clause']

    def generate_execution_code(self, code):
        self.test.generate_evaluation_code(code)
        # Make sure all conditions are evaluated before going into the switch() statement.
        # This is required in order to prevent any execution code from leaking into the space between the cases.
        for case in self.cases:
            case.generate_condition_evaluation_code(code)
        code.mark_pos(self.pos)
        code.putln("switch (%s) {" % self.test.result())
        for case in self.cases:
            case.generate_execution_code(code)
        if self.else_clause is not None:
            code.putln("default:")
            self.else_clause.generate_execution_code(code)
            code.putln("break;")
        else:
            # Always generate a default clause to prevent C compiler warnings
            # about unmatched enum values (it was not the user who decided to
            # generate the switch statement, so shouldn't be bothered).
            code.putln("default: break;")
        code.putln("}")
        self.test.generate_disposal_code(code)
        self.test.free_temps(code)

    def generate_function_definitions(self, env, code):
        self.test.generate_function_definitions(env, code)
        for case in self.cases:
            case.generate_function_definitions(env, code)
        if self.else_clause is not None:
            self.else_clause.generate_function_definitions(env, code)

    def annotate(self, code):
        self.test.annotate(code)
        for case in self.cases:
            case.annotate(code)
        if self.else_clause is not None:
            self.else_clause.annotate(code)


class LoopNode(object):
    pass


class WhileStatNode(LoopNode, StatNode):
    #  while statement
    #
    #  condition    ExprNode
    #  body         StatNode
    #  else_clause  StatNode

    child_attrs = ["condition", "body", "else_clause"]

    def analyse_declarations(self, env):
        self.body.analyse_declarations(env)
        if self.else_clause:
            self.else_clause.analyse_declarations(env)

    def analyse_expressions(self, env):
        if self.condition:
            self.condition = self.condition.analyse_temp_boolean_expression(env)
        self.body = self.body.analyse_expressions(env)
        if self.else_clause:
            self.else_clause = self.else_clause.analyse_expressions(env)
        return self

    def generate_execution_code(self, code):
        code.mark_pos(self.pos)
        old_loop_labels = code.new_loop_labels()
        code.putln(
            "while (1) {")
        if self.condition:
            self.condition.generate_evaluation_code(code)
            self.condition.generate_disposal_code(code)
            code.putln(
                "if (!%s) break;" % self.condition.result())
            self.condition.free_temps(code)
        self.body.generate_execution_code(code)
        code.put_label(code.continue_label)
        code.putln("}")
        break_label = code.break_label
        code.set_loop_labels(old_loop_labels)
        if self.else_clause:
            code.mark_pos(self.else_clause.pos)
            code.putln("/*else*/ {")
            self.else_clause.generate_execution_code(code)
            code.putln("}")
        code.put_label(break_label)

    def generate_function_definitions(self, env, code):
        if self.condition:
            self.condition.generate_function_definitions(env, code)
        self.body.generate_function_definitions(env, code)
        if self.else_clause is not None:
            self.else_clause.generate_function_definitions(env, code)

    def annotate(self, code):
        if self.condition:
            self.condition.annotate(code)
        self.body.annotate(code)
        if self.else_clause:
            self.else_clause.annotate(code)


class DictIterationNextNode(Node):
    # Helper node for calling PyDict_Next() inside of a WhileStatNode
    # and checking the dictionary size for changes.  Created in
    # Optimize.py.
    child_attrs = ['dict_obj', 'expected_size', 'pos_index_var',
                   'coerced_key_var', 'coerced_value_var', 'coerced_tuple_var',
                   'key_target', 'value_target', 'tuple_target', 'is_dict_flag']

    coerced_key_var = key_ref = None
    coerced_value_var = value_ref = None
    coerced_tuple_var = tuple_ref = None

    def __init__(self, dict_obj, expected_size, pos_index_var,
                 key_target, value_target, tuple_target, is_dict_flag):
        Node.__init__(
            self, dict_obj.pos,
            dict_obj=dict_obj,
            expected_size=expected_size,
            pos_index_var=pos_index_var,
            key_target=key_target,
            value_target=value_target,
            tuple_target=tuple_target,
            is_dict_flag=is_dict_flag,
            is_temp=True,
            type=PyrexTypes.c_bint_type)

    def analyse_expressions(self, env):
        from . import ExprNodes
        self.dict_obj = self.dict_obj.analyse_types(env)
        self.expected_size = self.expected_size.analyse_types(env)
        if self.pos_index_var:
            self.pos_index_var = self.pos_index_var.analyse_types(env)
        if self.key_target:
            self.key_target = self.key_target.analyse_target_types(env)
            self.key_ref = ExprNodes.TempNode(self.key_target.pos, PyrexTypes.py_object_type)
            self.coerced_key_var = self.key_ref.coerce_to(self.key_target.type, env)
        if self.value_target:
            self.value_target = self.value_target.analyse_target_types(env)
            self.value_ref = ExprNodes.TempNode(self.value_target.pos, type=PyrexTypes.py_object_type)
            self.coerced_value_var = self.value_ref.coerce_to(self.value_target.type, env)
        if self.tuple_target:
            self.tuple_target = self.tuple_target.analyse_target_types(env)
            self.tuple_ref = ExprNodes.TempNode(self.tuple_target.pos, PyrexTypes.py_object_type)
            self.coerced_tuple_var = self.tuple_ref.coerce_to(self.tuple_target.type, env)
        self.is_dict_flag = self.is_dict_flag.analyse_types(env)
        return self

    def generate_function_definitions(self, env, code):
        self.dict_obj.generate_function_definitions(env, code)

    def generate_execution_code(self, code):
        code.globalstate.use_utility_code(UtilityCode.load_cached("dict_iter", "Optimize.c"))
        self.dict_obj.generate_evaluation_code(code)

        assignments = []
        temp_addresses = []
        for var, result, target in [(self.key_ref, self.coerced_key_var, self.key_target),
                                    (self.value_ref, self.coerced_value_var, self.value_target),
                                    (self.tuple_ref, self.coerced_tuple_var, self.tuple_target)]:
            if target is None:
                addr = 'NULL'
            else:
                assignments.append((var, result, target))
                var.allocate(code)
                addr = '&%s' % var.result()
            temp_addresses.append(addr)

        result_temp = code.funcstate.allocate_temp(PyrexTypes.c_int_type, False)
        code.putln("%s = __Pyx_dict_iter_next(%s, %s, &%s, %s, %s, %s, %s);" % (
            result_temp,
            self.dict_obj.py_result(),
            self.expected_size.result(),
            self.pos_index_var.result(),
            temp_addresses[0],
            temp_addresses[1],
            temp_addresses[2],
            self.is_dict_flag.result()
        ))
        code.putln("if (unlikely(%s == 0)) break;" % result_temp)
        code.putln(code.error_goto_if("%s == -1" % result_temp, self.pos))
        code.funcstate.release_temp(result_temp)

        # evaluate all coercions before the assignments
        for var, result, target in assignments:
            code.put_gotref(var.result())
        for var, result, target in assignments:
            result.generate_evaluation_code(code)
        for var, result, target in assignments:
            target.generate_assignment_code(result, code)
            var.release(code)


class SetIterationNextNode(Node):
    # Helper node for calling _PySet_NextEntry() inside of a WhileStatNode
    # and checking the set size for changes.  Created in Optimize.py.
    child_attrs = ['set_obj', 'expected_size', 'pos_index_var',
                   'coerced_value_var', 'value_target', 'is_set_flag']

    coerced_value_var = value_ref = None

    def __init__(self, set_obj, expected_size, pos_index_var, value_target, is_set_flag):
        Node.__init__(
            self, set_obj.pos,
            set_obj=set_obj,
            expected_size=expected_size,
            pos_index_var=pos_index_var,
            value_target=value_target,
            is_set_flag=is_set_flag,
            is_temp=True,
            type=PyrexTypes.c_bint_type)

    def analyse_expressions(self, env):
        from . import ExprNodes
        self.set_obj = self.set_obj.analyse_types(env)
        self.expected_size = self.expected_size.analyse_types(env)
        self.pos_index_var = self.pos_index_var.analyse_types(env)
        self.value_target = self.value_target.analyse_target_types(env)
        self.value_ref = ExprNodes.TempNode(self.value_target.pos, type=PyrexTypes.py_object_type)
        self.coerced_value_var = self.value_ref.coerce_to(self.value_target.type, env)
        self.is_set_flag = self.is_set_flag.analyse_types(env)
        return self

    def generate_function_definitions(self, env, code):
        self.set_obj.generate_function_definitions(env, code)

    def generate_execution_code(self, code):
        code.globalstate.use_utility_code(UtilityCode.load_cached("set_iter", "Optimize.c"))
        self.set_obj.generate_evaluation_code(code)

        value_ref = self.value_ref
        value_ref.allocate(code)

        result_temp = code.funcstate.allocate_temp(PyrexTypes.c_int_type, False)
        code.putln("%s = __Pyx_set_iter_next(%s, %s, &%s, &%s, %s);" % (
            result_temp,
            self.set_obj.py_result(),
            self.expected_size.result(),
            self.pos_index_var.result(),
            value_ref.result(),
            self.is_set_flag.result()
        ))
        code.putln("if (unlikely(%s == 0)) break;" % result_temp)
        code.putln(code.error_goto_if("%s == -1" % result_temp, self.pos))
        code.funcstate.release_temp(result_temp)

        # evaluate all coercions before the assignments
        code.put_gotref(value_ref.result())
        self.coerced_value_var.generate_evaluation_code(code)
        self.value_target.generate_assignment_code(self.coerced_value_var, code)
        value_ref.release(code)


def ForStatNode(pos, **kw):
    if 'iterator' in kw:
        if kw['iterator'].is_async:
            return AsyncForStatNode(pos, **kw)
        else:
            return ForInStatNode(pos, **kw)
    else:
        return ForFromStatNode(pos, **kw)


class _ForInStatNode(LoopNode, StatNode):
    #  Base class of 'for-in' statements.
    #
    #  target        ExprNode
    #  iterator      IteratorNode | AIterAwaitExprNode(AsyncIteratorNode)
    #  body          StatNode
    #  else_clause   StatNode
    #  item          NextNode | AwaitExprNode(AsyncNextNode)
    #  is_async      boolean        true for 'async for' statements

    child_attrs = ["target", "item", "iterator", "body", "else_clause"]
    item = None
    is_async = False

    def _create_item_node(self):
        raise NotImplementedError("must be implemented by subclasses")

    def analyse_declarations(self, env):
        self.target.analyse_target_declaration(env)
        self.body.analyse_declarations(env)
        if self.else_clause:
            self.else_clause.analyse_declarations(env)
        self._create_item_node()

    def analyse_expressions(self, env):
        self.target = self.target.analyse_target_types(env)
        self.iterator = self.iterator.analyse_expressions(env)
        self._create_item_node()  # must rewrap self.item after analysis
        self.item = self.item.analyse_expressions(env)
        if (not self.is_async and
                (self.iterator.type.is_ptr or self.iterator.type.is_array) and
                self.target.type.assignable_from(self.iterator.type)):
            # C array slice optimization.
            pass
        else:
            self.item = self.item.coerce_to(self.target.type, env)
        self.body = self.body.analyse_expressions(env)
        if self.else_clause:
            self.else_clause = self.else_clause.analyse_expressions(env)
        return self

    def generate_execution_code(self, code):
        code.mark_pos(self.pos)
        old_loop_labels = code.new_loop_labels()
        self.iterator.generate_evaluation_code(code)
        code.putln("for (;;) {")
        self.item.generate_evaluation_code(code)
        self.target.generate_assignment_code(self.item, code)
        self.body.generate_execution_code(code)
        code.mark_pos(self.pos)
        code.put_label(code.continue_label)
        code.putln("}")
        break_label = code.break_label
        code.set_loop_labels(old_loop_labels)

        if self.else_clause:
            # in nested loops, the 'else' block can contain a
            # 'continue' statement for the outer loop, but we may need
            # to generate cleanup code before taking that path, so we
            # intercept it here
            orig_continue_label = code.continue_label
            code.continue_label = code.new_label('outer_continue')

            code.putln("/*else*/ {")
            self.else_clause.generate_execution_code(code)
            code.putln("}")

            if code.label_used(code.continue_label):
                code.put_goto(break_label)
                code.mark_pos(self.pos)
                code.put_label(code.continue_label)
                self.iterator.generate_disposal_code(code)
                code.put_goto(orig_continue_label)
            code.set_loop_labels(old_loop_labels)

        code.mark_pos(self.pos)
        if code.label_used(break_label):
            code.put_label(break_label)
        self.iterator.generate_disposal_code(code)
        self.iterator.free_temps(code)

    def generate_function_definitions(self, env, code):
        self.target.generate_function_definitions(env, code)
        self.iterator.generate_function_definitions(env, code)
        self.body.generate_function_definitions(env, code)
        if self.else_clause is not None:
            self.else_clause.generate_function_definitions(env, code)

    def annotate(self, code):
        self.target.annotate(code)
        self.iterator.annotate(code)
        self.body.annotate(code)
        if self.else_clause:
            self.else_clause.annotate(code)
        self.item.annotate(code)


class ForInStatNode(_ForInStatNode):
    #  'for' statement

    is_async = False

    def _create_item_node(self):
        from .ExprNodes import NextNode
        self.item = NextNode(self.iterator)


class AsyncForStatNode(_ForInStatNode):
    #  'async for' statement
    #
    #  iterator      AIterAwaitExprNode(AsyncIteratorNode)
    #  item          AwaitIterNextExprNode(AsyncIteratorNode)

    is_async = True

    def __init__(self, pos, **kw):
        assert 'item' not in kw
        from . import ExprNodes
        # AwaitExprNodes must appear before running MarkClosureVisitor
        kw['item'] = ExprNodes.AwaitIterNextExprNode(kw['iterator'].pos, arg=None)
        _ForInStatNode.__init__(self, pos, **kw)

    def _create_item_node(self):
        from . import ExprNodes
        self.item.arg = ExprNodes.AsyncNextNode(self.iterator)


class ForFromStatNode(LoopNode, StatNode):
    #  for name from expr rel name rel expr
    #
    #  target        NameNode
    #  bound1        ExprNode
    #  relation1     string
    #  relation2     string
    #  bound2        ExprNode
    #  step          ExprNode or None
    #  body          StatNode
    #  else_clause   StatNode or None
    #
    #  Used internally:
    #
    #  from_range         bool
    #  is_py_target       bool
    #  loopvar_node       ExprNode (usually a NameNode or temp node)
    #  py_loopvar_node    PyTempNode or None
    child_attrs = ["target", "bound1", "bound2", "step", "body", "else_clause"]

    is_py_target = False
    loopvar_node = None
    py_loopvar_node = None
    from_range = False

    gil_message = "For-loop using object bounds or target"

    def nogil_check(self, env):
        for x in (self.target, self.bound1, self.bound2):
            if x.type.is_pyobject:
                self.gil_error()

    def analyse_declarations(self, env):
        self.target.analyse_target_declaration(env)
        self.body.analyse_declarations(env)
        if self.else_clause:
            self.else_clause.analyse_declarations(env)

    def analyse_expressions(self, env):
        from . import ExprNodes
        self.target = self.target.analyse_target_types(env)
        self.bound1 = self.bound1.analyse_types(env)
        self.bound2 = self.bound2.analyse_types(env)
        if self.step is not None:
            if isinstance(self.step, ExprNodes.UnaryMinusNode):
                warning(self.step.pos, "Probable infinite loop in for-from-by statement. "
                        "Consider switching the directions of the relations.", 2)
            self.step = self.step.analyse_types(env)

        self.set_up_loop(env)
        target_type = self.target.type
        if not (target_type.is_pyobject or target_type.is_numeric):
            error(self.target.pos, "for-from loop variable must be c numeric type or Python object")

        self.body = self.body.analyse_expressions(env)
        if self.else_clause:
            self.else_clause = self.else_clause.analyse_expressions(env)
        return self

    def set_up_loop(self, env):
        from . import ExprNodes

        target_type = self.target.type
        if target_type.is_numeric:
            loop_type = target_type
        else:
            if target_type.is_enum:
                warning(self.target.pos,
                        "Integer loops over enum values are fragile. Please cast to a safe integer type instead.")
            loop_type = PyrexTypes.c_long_type if target_type.is_pyobject else PyrexTypes.c_int_type
            if not self.bound1.type.is_pyobject:
                loop_type = PyrexTypes.widest_numeric_type(loop_type, self.bound1.type)
            if not self.bound2.type.is_pyobject:
                loop_type = PyrexTypes.widest_numeric_type(loop_type, self.bound2.type)
            if self.step is not None and not self.step.type.is_pyobject:
                loop_type = PyrexTypes.widest_numeric_type(loop_type, self.step.type)
        self.bound1 = self.bound1.coerce_to(loop_type, env)
        self.bound2 = self.bound2.coerce_to(loop_type, env)
        if not self.bound2.is_literal:
            self.bound2 = self.bound2.coerce_to_temp(env)
        if self.step is not None:
            self.step = self.step.coerce_to(loop_type, env)
            if not self.step.is_literal:
                self.step = self.step.coerce_to_temp(env)

        if target_type.is_numeric or target_type.is_enum:
            self.is_py_target = False
            if isinstance(self.target, ExprNodes.BufferIndexNode):
                raise error(self.pos, "Buffer or memoryview slicing/indexing not allowed as for-loop target.")
            self.loopvar_node = self.target
            self.py_loopvar_node = None
        else:
            self.is_py_target = True
            c_loopvar_node = ExprNodes.TempNode(self.pos, loop_type, env)
            self.loopvar_node = c_loopvar_node
            self.py_loopvar_node = ExprNodes.CloneNode(c_loopvar_node).coerce_to_pyobject(env)

    def generate_execution_code(self, code):
        code.mark_pos(self.pos)
        old_loop_labels = code.new_loop_labels()
        from_range = self.from_range
        self.bound1.generate_evaluation_code(code)
        self.bound2.generate_evaluation_code(code)
        offset, incop = self.relation_table[self.relation1]
        if self.step is not None:
            self.step.generate_evaluation_code(code)
            step = self.step.result()
            incop = "%s=%s" % (incop[0], step)  # e.g. '++' => '+= STEP'
        else:
            step = '1'

        from . import ExprNodes
        if isinstance(self.loopvar_node, ExprNodes.TempNode):
            self.loopvar_node.allocate(code)
        if isinstance(self.py_loopvar_node, ExprNodes.TempNode):
            self.py_loopvar_node.allocate(code)

        loopvar_type = PyrexTypes.c_long_type if self.target.type.is_enum else self.target.type

        if from_range and not self.is_py_target:
            loopvar_name = code.funcstate.allocate_temp(loopvar_type, False)
        else:
            loopvar_name = self.loopvar_node.result()
        if loopvar_type.is_int and not loopvar_type.signed and self.relation2[0] == '>':
            # Handle the case where the endpoint of an unsigned int iteration
            # is within step of 0.
            code.putln("for (%s = %s%s + %s; %s %s %s + %s; ) { %s%s;" % (
                loopvar_name,
                self.bound1.result(), offset, step,
                loopvar_name, self.relation2, self.bound2.result(), step,
                loopvar_name, incop))
        else:
            code.putln("for (%s = %s%s; %s %s %s; %s%s) {" % (
                loopvar_name,
                self.bound1.result(), offset,
                loopvar_name, self.relation2, self.bound2.result(),
                loopvar_name, incop))

        coerced_loopvar_node = self.py_loopvar_node
        if coerced_loopvar_node is None and from_range:
            coerced_loopvar_node = ExprNodes.RawCNameExprNode(self.target.pos, loopvar_type, loopvar_name)
        if coerced_loopvar_node is not None:
            coerced_loopvar_node.generate_evaluation_code(code)
            self.target.generate_assignment_code(coerced_loopvar_node, code)

        self.body.generate_execution_code(code)
        code.put_label(code.continue_label)

        if not from_range and self.py_loopvar_node:
            # This mess is to make for..from loops with python targets behave
            # exactly like those with C targets with regards to re-assignment
            # of the loop variable.
            if self.target.entry.is_pyglobal:
                # We know target is a NameNode, this is the only ugly case.
                target_node = ExprNodes.PyTempNode(self.target.pos, None)
                target_node.allocate(code)
                interned_cname = code.intern_identifier(self.target.entry.name)
                if self.target.entry.scope.is_module_scope:
                    code.globalstate.use_utility_code(
                        UtilityCode.load_cached("GetModuleGlobalName", "ObjectHandling.c"))
                    lookup_func = '__Pyx_GetModuleGlobalName(%s, %s); %s'
                else:
                    code.globalstate.use_utility_code(
                        UtilityCode.load_cached("GetNameInClass", "ObjectHandling.c"))
                    lookup_func = '__Pyx_GetNameInClass(%s, {}, %s); %s'.format(
                        self.target.entry.scope.namespace_cname)
                code.putln(lookup_func % (
                    target_node.result(),
                    interned_cname,
                    code.error_goto_if_null(target_node.result(), self.target.pos)))
                code.put_gotref(target_node.result())
            else:
                target_node = self.target
            from_py_node = ExprNodes.CoerceFromPyTypeNode(
                self.loopvar_node.type, target_node, self.target.entry.scope)
            from_py_node.temp_code = loopvar_name
            from_py_node.generate_result_code(code)
            if self.target.entry.is_pyglobal:
                code.put_decref(target_node.result(), target_node.type)
                target_node.release(code)

        code.putln("}")

        if not from_range and self.py_loopvar_node:
            # This is potentially wasteful, but we don't want the semantics to
            # depend on whether or not the loop is a python type.
            self.py_loopvar_node.generate_evaluation_code(code)
            self.target.generate_assignment_code(self.py_loopvar_node, code)
        if from_range and not self.is_py_target:
            code.funcstate.release_temp(loopvar_name)

        break_label = code.break_label
        code.set_loop_labels(old_loop_labels)
        if self.else_clause:
            code.putln("/*else*/ {")
            self.else_clause.generate_execution_code(code)
            code.putln("}")
        code.put_label(break_label)
        self.bound1.generate_disposal_code(code)
        self.bound1.free_temps(code)
        self.bound2.generate_disposal_code(code)
        self.bound2.free_temps(code)
        if isinstance(self.loopvar_node, ExprNodes.TempNode):
            self.loopvar_node.release(code)
        if isinstance(self.py_loopvar_node, ExprNodes.TempNode):
            self.py_loopvar_node.release(code)
        if self.step is not None:
            self.step.generate_disposal_code(code)
            self.step.free_temps(code)

    relation_table = {
        # {relop : (initial offset, increment op)}
        '<=': ("",   "++"),
        '<' : ("+1", "++"),
        '>=': ("",   "--"),
        '>' : ("-1", "--"),
    }

    def generate_function_definitions(self, env, code):
        self.target.generate_function_definitions(env, code)
        self.bound1.generate_function_definitions(env, code)
        self.bound2.generate_function_definitions(env, code)
        if self.step is not None:
            self.step.generate_function_definitions(env, code)
        self.body.generate_function_definitions(env, code)
        if self.else_clause is not None:
            self.else_clause.generate_function_definitions(env, code)

    def annotate(self, code):
        self.target.annotate(code)
        self.bound1.annotate(code)
        self.bound2.annotate(code)
        if self.step:
            self.step.annotate(code)
        self.body.annotate(code)
        if self.else_clause:
            self.else_clause.annotate(code)


class WithStatNode(StatNode):
    """
    Represents a Python with statement.

    Implemented by the WithTransform as follows:

        MGR = EXPR
        EXIT = MGR.__exit__
        VALUE = MGR.__enter__()
        EXC = True
        try:
            try:
                TARGET = VALUE  # optional
                BODY
            except:
                EXC = False
                if not EXIT(*EXCINFO):
                    raise
        finally:
            if EXC:
                EXIT(None, None, None)
            MGR = EXIT = VALUE = None
    """
    #  manager          The with statement manager object
    #  target           ExprNode  the target lhs of the __enter__() call
    #  body             StatNode
    #  enter_call       ExprNode  the call to the __enter__() method
    #  exit_var         String    the cname of the __exit__() method reference

    child_attrs = ["manager", "enter_call", "target", "body"]

    enter_call = None
    target_temp = None

    def analyse_declarations(self, env):
        self.manager.analyse_declarations(env)
        self.enter_call.analyse_declarations(env)
        self.body.analyse_declarations(env)

    def analyse_expressions(self, env):
        self.manager = self.manager.analyse_types(env)
        self.enter_call = self.enter_call.analyse_types(env)
        if self.target:
            # set up target_temp before descending into body (which uses it)
            from .ExprNodes import TempNode
            self.target_temp = TempNode(self.enter_call.pos, self.enter_call.type)
        self.body = self.body.analyse_expressions(env)
        return self

    def generate_function_definitions(self, env, code):
        self.manager.generate_function_definitions(env, code)
        self.enter_call.generate_function_definitions(env, code)
        self.body.generate_function_definitions(env, code)

    def generate_execution_code(self, code):
        code.mark_pos(self.pos)
        code.putln("/*with:*/ {")
        self.manager.generate_evaluation_code(code)
        self.exit_var = code.funcstate.allocate_temp(py_object_type, manage_ref=False)
        code.globalstate.use_utility_code(
            UtilityCode.load_cached("PyObjectLookupSpecial", "ObjectHandling.c"))
        code.putln("%s = __Pyx_PyObject_LookupSpecial(%s, %s); %s" % (
            self.exit_var,
            self.manager.py_result(),
            code.intern_identifier(EncodedString('__aexit__' if self.is_async else '__exit__')),
            code.error_goto_if_null(self.exit_var, self.pos),
            ))
        code.put_gotref(self.exit_var)

        # need to free exit_var in the face of exceptions during setup
        old_error_label = code.new_error_label()
        intermediate_error_label = code.error_label

        self.enter_call.generate_evaluation_code(code)
        if self.target:
            # The temp result will be cleaned up by the WithTargetAssignmentStatNode
            # after assigning its result to the target of the 'with' statement.
            self.target_temp.allocate(code)
            self.enter_call.make_owned_reference(code)
            code.putln("%s = %s;" % (self.target_temp.result(), self.enter_call.result()))
            self.enter_call.generate_post_assignment_code(code)
        else:
            self.enter_call.generate_disposal_code(code)
        self.enter_call.free_temps(code)

        self.manager.generate_disposal_code(code)
        self.manager.free_temps(code)

        code.error_label = old_error_label
        self.body.generate_execution_code(code)

        if code.label_used(intermediate_error_label):
            step_over_label = code.new_label()
            code.put_goto(step_over_label)
            code.put_label(intermediate_error_label)
            code.put_decref_clear(self.exit_var, py_object_type)
            code.put_goto(old_error_label)
            code.put_label(step_over_label)

        code.funcstate.release_temp(self.exit_var)
        code.putln('}')


class WithTargetAssignmentStatNode(AssignmentNode):
    # The target assignment of the 'with' statement value (return
    # value of the __enter__() call).
    #
    # This is a special cased assignment that properly cleans up the RHS.
    #
    # lhs       ExprNode      the assignment target
    # rhs       ExprNode      a (coerced) TempNode for the rhs (from WithStatNode)
    # with_node WithStatNode  the surrounding with-statement

    child_attrs = ["rhs", "lhs"]
    with_node = None
    rhs = None

    def analyse_declarations(self, env):
        self.lhs.analyse_target_declaration(env)

    def analyse_expressions(self, env):
        self.lhs = self.lhs.analyse_target_types(env)
        self.lhs.gil_assignment_check(env)
        self.rhs = self.with_node.target_temp.coerce_to(self.lhs.type, env)
        return self

    def generate_execution_code(self, code):
        self.rhs.generate_evaluation_code(code)
        self.lhs.generate_assignment_code(self.rhs, code)
        self.with_node.target_temp.release(code)

    def annotate(self, code):
        self.lhs.annotate(code)
        self.rhs.annotate(code)


class TryExceptStatNode(StatNode):
    #  try .. except statement
    #
    #  body             StatNode
    #  except_clauses   [ExceptClauseNode]
    #  else_clause      StatNode or None

    child_attrs = ["body", "except_clauses", "else_clause"]
    in_generator = False

    def analyse_declarations(self, env):
        self.body.analyse_declarations(env)
        for except_clause in self.except_clauses:
            except_clause.analyse_declarations(env)
        if self.else_clause:
            self.else_clause.analyse_declarations(env)

    def analyse_expressions(self, env):
        self.body = self.body.analyse_expressions(env)
        default_clause_seen = 0
        for i, except_clause in enumerate(self.except_clauses):
            except_clause = self.except_clauses[i] = except_clause.analyse_expressions(env)
            if default_clause_seen:
                error(except_clause.pos, "default 'except:' must be last")
            if not except_clause.pattern:
                default_clause_seen = 1
        self.has_default_clause = default_clause_seen
        if self.else_clause:
            self.else_clause = self.else_clause.analyse_expressions(env)
        return self

    nogil_check = Node.gil_error
    gil_message = "Try-except statement"

    def generate_execution_code(self, code):
        code.mark_pos(self.pos)  # before changing the error label, in case of tracing errors
        code.putln("{")

        old_return_label = code.return_label
        old_break_label = code.break_label
        old_continue_label = code.continue_label
        old_error_label = code.new_error_label()
        our_error_label = code.error_label
        except_end_label = code.new_label('exception_handled')
        except_error_label = code.new_label('except_error')
        except_return_label = code.new_label('except_return')
        try_return_label = code.new_label('try_return')
        try_break_label = code.new_label('try_break') if old_break_label else None
        try_continue_label = code.new_label('try_continue') if old_continue_label else None
        try_end_label = code.new_label('try_end')

        exc_save_vars = [code.funcstate.allocate_temp(py_object_type, False)
                         for _ in range(3)]
        save_exc = code.insertion_point()
        code.putln(
            "/*try:*/ {")
        code.return_label = try_return_label
        code.break_label = try_break_label
        code.continue_label = try_continue_label
        self.body.generate_execution_code(code)
        code.mark_pos(self.pos, trace=False)
        code.putln(
            "}")
        temps_to_clean_up = code.funcstate.all_free_managed_temps()
        can_raise = code.label_used(our_error_label)

        if can_raise:
            # inject code before the try block to save away the exception state
            code.globalstate.use_utility_code(reset_exception_utility_code)
            if not self.in_generator:
                save_exc.putln("__Pyx_PyThreadState_declare")
                save_exc.putln("__Pyx_PyThreadState_assign")
            save_exc.putln("__Pyx_ExceptionSave(%s);" % (
                ', '.join(['&%s' % var for var in exc_save_vars])))
            for var in exc_save_vars:
                save_exc.put_xgotref(var)

            def restore_saved_exception():
                for name in exc_save_vars:
                    code.put_xgiveref(name)
                code.putln("__Pyx_ExceptionReset(%s);" %
                           ', '.join(exc_save_vars))
        else:
            # try block cannot raise exceptions, but we had to allocate the temps above,
            # so just keep the C compiler from complaining about them being unused
            mark_vars_used =  ["(void)%s;" % var for var in exc_save_vars]
            save_exc.putln("%s /* mark used */" % ' '.join(mark_vars_used))

            def restore_saved_exception():
                pass

        code.error_label = except_error_label
        code.return_label = except_return_label
        normal_case_terminates = self.body.is_terminator
        if self.else_clause:
            code.mark_pos(self.else_clause.pos)
            code.putln(
                "/*else:*/ {")
            self.else_clause.generate_execution_code(code)
            code.putln(
                "}")
            if not normal_case_terminates:
                normal_case_terminates = self.else_clause.is_terminator

        if can_raise:
            if not normal_case_terminates:
                for var in exc_save_vars:
                    code.put_xdecref_clear(var, py_object_type)
                code.put_goto(try_end_label)
            code.put_label(our_error_label)
            for temp_name, temp_type in temps_to_clean_up:
                code.put_xdecref_clear(temp_name, temp_type)

            outer_except = code.funcstate.current_except
            # Currently points to self, but the ExceptClauseNode would also be ok. Change if needed.
            code.funcstate.current_except = self
            for except_clause in self.except_clauses:
                except_clause.generate_handling_code(code, except_end_label)
            code.funcstate.current_except = outer_except

            if not self.has_default_clause:
                code.put_goto(except_error_label)

        for exit_label, old_label in [(except_error_label, old_error_label),
                                      (try_break_label, old_break_label),
                                      (try_continue_label, old_continue_label),
                                      (try_return_label, old_return_label),
                                      (except_return_label, old_return_label)]:
            if code.label_used(exit_label):
                if not normal_case_terminates and not code.label_used(try_end_label):
                    code.put_goto(try_end_label)
                code.put_label(exit_label)
                code.mark_pos(self.pos, trace=False)
                if can_raise:
                    restore_saved_exception()
                code.put_goto(old_label)

        if code.label_used(except_end_label):
            if not normal_case_terminates and not code.label_used(try_end_label):
                code.put_goto(try_end_label)
            code.put_label(except_end_label)
            if can_raise:
                restore_saved_exception()
        if code.label_used(try_end_label):
            code.put_label(try_end_label)
        code.putln("}")

        for cname in exc_save_vars:
            code.funcstate.release_temp(cname)

        code.return_label = old_return_label
        code.break_label = old_break_label
        code.continue_label = old_continue_label
        code.error_label = old_error_label

    def generate_function_definitions(self, env, code):
        self.body.generate_function_definitions(env, code)
        for except_clause in self.except_clauses:
            except_clause.generate_function_definitions(env, code)
        if self.else_clause is not None:
            self.else_clause.generate_function_definitions(env, code)

    def annotate(self, code):
        self.body.annotate(code)
        for except_node in self.except_clauses:
            except_node.annotate(code)
        if self.else_clause:
            self.else_clause.annotate(code)


class ExceptClauseNode(Node):
    #  Part of try ... except statement.
    #
    #  pattern        [ExprNode]
    #  target         ExprNode or None
    #  body           StatNode
    #  excinfo_target TupleNode(3*ResultRefNode) or None   optional target for exception info (not owned here!)
    #  match_flag     string             result of exception match
    #  exc_value      ExcValueNode       used internally
    #  function_name  string             qualified name of enclosing function
    #  exc_vars       (string * 3)       local exception variables
    #  is_except_as   bool               Py3-style "except ... as xyz"

    # excinfo_target is never set by the parser, but can be set by a transform
    # in order to extract more extensive information about the exception as a
    # sys.exc_info()-style tuple into a target variable

    child_attrs = ["pattern", "target", "body", "exc_value"]

    exc_value = None
    excinfo_target = None
    is_except_as = False

    def analyse_declarations(self, env):
        if self.target:
            self.target.analyse_target_declaration(env)
        self.body.analyse_declarations(env)

    def analyse_expressions(self, env):
        self.function_name = env.qualified_name
        if self.pattern:
            # normalise/unpack self.pattern into a list
            for i, pattern in enumerate(self.pattern):
                pattern = pattern.analyse_expressions(env)
                self.pattern[i] = pattern.coerce_to_pyobject(env)

        if self.target:
            from . import ExprNodes
            self.exc_value = ExprNodes.ExcValueNode(self.pos)
            self.target = self.target.analyse_target_expression(env, self.exc_value)

        self.body = self.body.analyse_expressions(env)
        return self

    def generate_handling_code(self, code, end_label):
        code.mark_pos(self.pos)

        if self.pattern:
            has_non_literals = not all(
                pattern.is_literal or pattern.is_simple() and not pattern.is_temp
                for pattern in self.pattern)

            if has_non_literals:
                # For non-trivial exception check expressions, hide the live exception from C-API calls.
                exc_vars = [code.funcstate.allocate_temp(py_object_type, manage_ref=True)
                            for _ in range(3)]
                code.globalstate.use_utility_code(UtilityCode.load_cached("PyErrFetchRestore", "Exceptions.c"))
                code.putln("__Pyx_ErrFetch(&%s, &%s, &%s);" % tuple(exc_vars))
                code.globalstate.use_utility_code(UtilityCode.load_cached("FastTypeChecks", "ModuleSetupCode.c"))
                exc_test_func = "__Pyx_PyErr_GivenExceptionMatches(%s, %%s)" % exc_vars[0]
            else:
                exc_vars = ()
                code.globalstate.use_utility_code(UtilityCode.load_cached("PyErrExceptionMatches", "Exceptions.c"))
                exc_test_func = "__Pyx_PyErr_ExceptionMatches(%s)"

            exc_tests = []
            for pattern in self.pattern:
                pattern.generate_evaluation_code(code)
                exc_tests.append(exc_test_func % pattern.py_result())

            match_flag = code.funcstate.allocate_temp(PyrexTypes.c_int_type, manage_ref=False)
            code.putln("%s = %s;" % (match_flag, ' || '.join(exc_tests)))
            for pattern in self.pattern:
                pattern.generate_disposal_code(code)
                pattern.free_temps(code)

            if has_non_literals:
                code.putln("__Pyx_ErrRestore(%s, %s, %s);" % tuple(exc_vars))
                code.putln(' '.join(["%s = 0;" % var for var in exc_vars]))
                for temp in exc_vars:
                    code.funcstate.release_temp(temp)

            code.putln(
                "if (%s) {" %
                    match_flag)
            code.funcstate.release_temp(match_flag)
        else:
            code.putln("/*except:*/ {")

        if (not getattr(self.body, 'stats', True)
                and self.excinfo_target is None
                and self.target is None):
            # most simple case: no exception variable, empty body (pass)
            # => reset the exception state, done
            code.globalstate.use_utility_code(UtilityCode.load_cached("PyErrFetchRestore", "Exceptions.c"))
            code.putln("__Pyx_ErrRestore(0,0,0);")
            code.put_goto(end_label)
            code.putln("}")
            return

        exc_vars = [code.funcstate.allocate_temp(py_object_type, manage_ref=True)
                    for _ in range(3)]
        code.put_add_traceback(self.function_name)
        # We always have to fetch the exception value even if
        # there is no target, because this also normalises the
        # exception and stores it in the thread state.
        code.globalstate.use_utility_code(get_exception_utility_code)
        exc_args = "&%s, &%s, &%s" % tuple(exc_vars)
        code.putln("if (__Pyx_GetException(%s) < 0) %s" % (
            exc_args, code.error_goto(self.pos)))
        for var in exc_vars:
            code.put_gotref(var)
        if self.target:
            self.exc_value.set_var(exc_vars[1])
            self.exc_value.generate_evaluation_code(code)
            self.target.generate_assignment_code(self.exc_value, code)
        if self.excinfo_target is not None:
            for tempvar, node in zip(exc_vars, self.excinfo_target.args):
                node.set_var(tempvar)

        old_break_label, old_continue_label = code.break_label, code.continue_label
        code.break_label = code.new_label('except_break')
        code.continue_label = code.new_label('except_continue')

        old_exc_vars = code.funcstate.exc_vars
        code.funcstate.exc_vars = exc_vars
        self.body.generate_execution_code(code)
        code.funcstate.exc_vars = old_exc_vars

        if not self.body.is_terminator:
            for var in exc_vars:
                # FIXME: XDECREF() is needed to allow re-raising (which clears the exc_vars),
                # but I don't think it's the right solution.
                code.put_xdecref_clear(var, py_object_type)
            code.put_goto(end_label)

        for new_label, old_label in [(code.break_label, old_break_label),
                                     (code.continue_label, old_continue_label)]:
            if code.label_used(new_label):
                code.put_label(new_label)
                for var in exc_vars:
                    code.put_decref_clear(var, py_object_type)
                code.put_goto(old_label)
        code.break_label = old_break_label
        code.continue_label = old_continue_label

        for temp in exc_vars:
            code.funcstate.release_temp(temp)

        code.putln(
            "}")

    def generate_function_definitions(self, env, code):
        if self.target is not None:
            self.target.generate_function_definitions(env, code)
        self.body.generate_function_definitions(env, code)

    def annotate(self, code):
        if self.pattern:
            for pattern in self.pattern:
                pattern.annotate(code)
        if self.target:
            self.target.annotate(code)
        self.body.annotate(code)


class TryFinallyStatNode(StatNode):
    #  try ... finally statement
    #
    #  body             StatNode
    #  finally_clause   StatNode
    #  finally_except_clause  deep-copy of finally_clause for exception case
    #  in_generator     inside of generator => must store away current exception also in return case
    #
    #  Each of the continue, break, return and error gotos runs
    #  into its own deep-copy of the finally block code.
    #  In addition, if we're doing an error, we save the
    #  exception on entry to the finally block and restore
    #  it on exit.

    child_attrs = ["body", "finally_clause", "finally_except_clause"]

    preserve_exception = 1

    # handle exception case, in addition to return/break/continue
    handle_error_case = True
    func_return_type = None
    finally_except_clause = None

    is_try_finally_in_nogil = False
    in_generator = False

    @staticmethod
    def create_analysed(pos, env, body, finally_clause):
        node = TryFinallyStatNode(pos, body=body, finally_clause=finally_clause)
        return node

    def analyse_declarations(self, env):
        self.body.analyse_declarations(env)
        self.finally_except_clause = copy.deepcopy(self.finally_clause)
        self.finally_except_clause.analyse_declarations(env)
        self.finally_clause.analyse_declarations(env)

    def analyse_expressions(self, env):
        self.body = self.body.analyse_expressions(env)
        self.finally_clause = self.finally_clause.analyse_expressions(env)
        self.finally_except_clause = self.finally_except_clause.analyse_expressions(env)
        if env.return_type and not env.return_type.is_void:
            self.func_return_type = env.return_type
        return self

    nogil_check = Node.gil_error
    gil_message = "Try-finally statement"

    def generate_execution_code(self, code):
        code.mark_pos(self.pos)  # before changing the error label, in case of tracing errors
        code.putln("/*try:*/ {")

        old_error_label = code.error_label
        old_labels = code.all_new_labels()
        new_labels = code.get_all_labels()
        new_error_label = code.error_label
        if not self.handle_error_case:
            code.error_label = old_error_label
        catch_label = code.new_label()

        was_in_try_finally = code.funcstate.in_try_finally
        code.funcstate.in_try_finally = 1

        self.body.generate_execution_code(code)

        code.funcstate.in_try_finally = was_in_try_finally
        code.putln("}")

        temps_to_clean_up = code.funcstate.all_free_managed_temps()
        code.mark_pos(self.finally_clause.pos)
        code.putln("/*finally:*/ {")

        # Reset labels only after writing out a potential line trace call for correct nogil error handling.
        code.set_all_labels(old_labels)

        def fresh_finally_clause(_next=[self.finally_clause]):
            # generate the original subtree once and always keep a fresh copy
            node = _next[0]
            node_copy = copy.deepcopy(node)
            if node is self.finally_clause:
                _next[0] = node_copy
            else:
                node = node_copy
            return node

        preserve_error = self.preserve_exception and code.label_used(new_error_label)
        needs_success_cleanup = not self.finally_clause.is_terminator

        if not self.body.is_terminator:
            code.putln('/*normal exit:*/{')
            fresh_finally_clause().generate_execution_code(code)
            if not self.finally_clause.is_terminator:
                code.put_goto(catch_label)
            code.putln('}')

        if preserve_error:
            code.put_label(new_error_label)
            code.putln('/*exception exit:*/{')
            if not self.in_generator:
                code.putln("__Pyx_PyThreadState_declare")
            if self.is_try_finally_in_nogil:
                code.declare_gilstate()
            if needs_success_cleanup:
                exc_lineno_cnames = tuple([
                    code.funcstate.allocate_temp(PyrexTypes.c_int_type, manage_ref=False)
                    for _ in range(2)])
                exc_filename_cname = code.funcstate.allocate_temp(
                    PyrexTypes.CPtrType(PyrexTypes.c_const_type(PyrexTypes.c_char_type)),
                    manage_ref=False)
            else:
                exc_lineno_cnames = exc_filename_cname = None
            exc_vars = tuple([
                code.funcstate.allocate_temp(py_object_type, manage_ref=False)
                for _ in range(6)])
            self.put_error_catcher(
                code, temps_to_clean_up, exc_vars, exc_lineno_cnames, exc_filename_cname)
            finally_old_labels = code.all_new_labels()

            code.putln('{')
            old_exc_vars = code.funcstate.exc_vars
            code.funcstate.exc_vars = exc_vars[:3]
            self.finally_except_clause.generate_execution_code(code)
            code.funcstate.exc_vars = old_exc_vars
            code.putln('}')

            if needs_success_cleanup:
                self.put_error_uncatcher(code, exc_vars, exc_lineno_cnames, exc_filename_cname)
                if exc_lineno_cnames:
                    for cname in exc_lineno_cnames:
                        code.funcstate.release_temp(cname)
                if exc_filename_cname:
                    code.funcstate.release_temp(exc_filename_cname)
                code.put_goto(old_error_label)

            for new_label, old_label in zip(code.get_all_labels(), finally_old_labels):
                if not code.label_used(new_label):
                    continue
                code.put_label(new_label)
                self.put_error_cleaner(code, exc_vars)
                code.put_goto(old_label)

            for cname in exc_vars:
                code.funcstate.release_temp(cname)
            code.putln('}')

        code.set_all_labels(old_labels)
        return_label = code.return_label
        exc_vars = ()

        for i, (new_label, old_label) in enumerate(zip(new_labels, old_labels)):
            if not code.label_used(new_label):
                continue
            if new_label == new_error_label and preserve_error:
                continue  # handled above

            code.putln('%s: {' % new_label)
            ret_temp = None
            if old_label == return_label:
                # return actually raises an (uncatchable) exception in generators that we must preserve
                if self.in_generator:
                    exc_vars = tuple([
                        code.funcstate.allocate_temp(py_object_type, manage_ref=False)
                        for _ in range(6)])
                    self.put_error_catcher(code, [], exc_vars)
                if not self.finally_clause.is_terminator:
                    # store away return value for later reuse
                    if (self.func_return_type and
                            not self.is_try_finally_in_nogil and
                            not isinstance(self.finally_clause, GILExitNode)):
                        ret_temp = code.funcstate.allocate_temp(
                            self.func_return_type, manage_ref=False)
                        code.putln("%s = %s;" % (ret_temp, Naming.retval_cname))
                        if self.func_return_type.is_pyobject:
                            code.putln("%s = 0;" % Naming.retval_cname)

            fresh_finally_clause().generate_execution_code(code)

            if old_label == return_label:
                if ret_temp:
                    code.putln("%s = %s;" % (Naming.retval_cname, ret_temp))
                    if self.func_return_type.is_pyobject:
                        code.putln("%s = 0;" % ret_temp)
                    code.funcstate.release_temp(ret_temp)
                if self.in_generator:
                    self.put_error_uncatcher(code, exc_vars)
                    for cname in exc_vars:
                        code.funcstate.release_temp(cname)

            if not self.finally_clause.is_terminator:
                code.put_goto(old_label)
            code.putln('}')

        # End finally
        code.put_label(catch_label)
        code.putln(
            "}")

    def generate_function_definitions(self, env, code):
        self.body.generate_function_definitions(env, code)
        self.finally_clause.generate_function_definitions(env, code)

    def put_error_catcher(self, code, temps_to_clean_up, exc_vars,
                          exc_lineno_cnames=None, exc_filename_cname=None):
        code.globalstate.use_utility_code(restore_exception_utility_code)
        code.globalstate.use_utility_code(get_exception_utility_code)
        code.globalstate.use_utility_code(swap_exception_utility_code)

        if self.is_try_finally_in_nogil:
            code.put_ensure_gil(declare_gilstate=False)
        code.putln("__Pyx_PyThreadState_assign")

        code.putln(' '.join(["%s = 0;" % var for var in exc_vars]))
        for temp_name, type in temps_to_clean_up:
            code.put_xdecref_clear(temp_name, type)

        # not using preprocessor here to avoid warnings about
        # unused utility functions and/or temps
        code.putln("if (PY_MAJOR_VERSION >= 3)"
                   " __Pyx_ExceptionSwap(&%s, &%s, &%s);" % exc_vars[3:])
        code.putln("if ((PY_MAJOR_VERSION < 3) ||"
                   # if __Pyx_GetException() fails in Py3,
                   # store the newly raised exception instead
                   " unlikely(__Pyx_GetException(&%s, &%s, &%s) < 0)) "
                   "__Pyx_ErrFetch(&%s, &%s, &%s);" % (exc_vars[:3] * 2))
        for var in exc_vars:
            code.put_xgotref(var)
        if exc_lineno_cnames:
            code.putln("%s = %s; %s = %s; %s = %s;" % (
                exc_lineno_cnames[0], Naming.lineno_cname,
                exc_lineno_cnames[1], Naming.clineno_cname,
                exc_filename_cname, Naming.filename_cname))

        if self.is_try_finally_in_nogil:
            code.put_release_ensured_gil()

    def put_error_uncatcher(self, code, exc_vars, exc_lineno_cnames=None, exc_filename_cname=None):
        code.globalstate.use_utility_code(restore_exception_utility_code)
        code.globalstate.use_utility_code(reset_exception_utility_code)

        if self.is_try_finally_in_nogil:
            code.put_ensure_gil(declare_gilstate=False)

        # not using preprocessor here to avoid warnings about
        # unused utility functions and/or temps
        code.putln("if (PY_MAJOR_VERSION >= 3) {")
        for var in exc_vars[3:]:
            code.put_xgiveref(var)
        code.putln("__Pyx_ExceptionReset(%s, %s, %s);" % exc_vars[3:])
        code.putln("}")
        for var in exc_vars[:3]:
            code.put_xgiveref(var)
        code.putln("__Pyx_ErrRestore(%s, %s, %s);" % exc_vars[:3])

        if self.is_try_finally_in_nogil:
            code.put_release_ensured_gil()

        code.putln(' '.join(["%s = 0;" % var for var in exc_vars]))
        if exc_lineno_cnames:
            code.putln("%s = %s; %s = %s; %s = %s;" % (
                Naming.lineno_cname, exc_lineno_cnames[0],
                Naming.clineno_cname, exc_lineno_cnames[1],
                Naming.filename_cname, exc_filename_cname))

    def put_error_cleaner(self, code, exc_vars):
        code.globalstate.use_utility_code(reset_exception_utility_code)
        if self.is_try_finally_in_nogil:
            code.put_ensure_gil(declare_gilstate=False)

        # not using preprocessor here to avoid warnings about
        # unused utility functions and/or temps
        code.putln("if (PY_MAJOR_VERSION >= 3) {")
        for var in exc_vars[3:]:
            code.put_xgiveref(var)
        code.putln("__Pyx_ExceptionReset(%s, %s, %s);" % exc_vars[3:])
        code.putln("}")
        for var in exc_vars[:3]:
            code.put_xdecref_clear(var, py_object_type)
        if self.is_try_finally_in_nogil:
            code.put_release_ensured_gil()
        code.putln(' '.join(["%s = 0;"]*3) % exc_vars[3:])

    def annotate(self, code):
        self.body.annotate(code)
        self.finally_clause.annotate(code)


class NogilTryFinallyStatNode(TryFinallyStatNode):
    """
    A try/finally statement that may be used in nogil code sections.
    """

    preserve_exception = False
    nogil_check = None


class GILStatNode(NogilTryFinallyStatNode):
    #  'with gil' or 'with nogil' statement
    #
    #   state   string   'gil' or 'nogil'

    state_temp = None

    def __init__(self, pos, state, body):
        self.state = state
        self.create_state_temp_if_needed(pos, state, body)
        TryFinallyStatNode.__init__(
            self, pos,
            body=body,
            finally_clause=GILExitNode(
                pos, state=state, state_temp=self.state_temp))

    def create_state_temp_if_needed(self, pos, state, body):
        from .ParseTreeTransforms import YieldNodeCollector
        collector = YieldNodeCollector()
        collector.visitchildren(body)
        if not collector.yields:
            return

        if state == 'gil':
            temp_type = PyrexTypes.c_gilstate_type
        else:
            temp_type = PyrexTypes.c_threadstate_ptr_type
        from . import ExprNodes
        self.state_temp = ExprNodes.TempNode(pos, temp_type)

    def analyse_declarations(self, env):
        env._in_with_gil_block = (self.state == 'gil')
        if self.state == 'gil':
            env.has_with_gil_block = True

        return super(GILStatNode, self).analyse_declarations(env)

    def analyse_expressions(self, env):
        env.use_utility_code(
            UtilityCode.load_cached("ForceInitThreads", "ModuleSetupCode.c"))
        was_nogil = env.nogil
        env.nogil = self.state == 'nogil'
        node = TryFinallyStatNode.analyse_expressions(self, env)
        env.nogil = was_nogil
        return node

    def generate_execution_code(self, code):
        code.mark_pos(self.pos)
        code.begin_block()
        if self.state_temp:
            self.state_temp.allocate(code)
            variable = self.state_temp.result()
        else:
            variable = None

        old_gil_config = code.funcstate.gil_owned
        if self.state == 'gil':
            code.put_ensure_gil(variable=variable)
            code.funcstate.gil_owned = True
        else:
            code.put_release_gil(variable=variable)
            code.funcstate.gil_owned = False

        TryFinallyStatNode.generate_execution_code(self, code)

        if self.state_temp:
            self.state_temp.release(code)

        code.funcstate.gil_owned = old_gil_config
        code.end_block()


class GILExitNode(StatNode):
    """
    Used as the 'finally' block in a GILStatNode

    state   string   'gil' or 'nogil'
    """

    child_attrs = []
    state_temp = None

    def analyse_expressions(self, env):
        return self

    def generate_execution_code(self, code):
        if self.state_temp:
            variable = self.state_temp.result()
        else:
            variable = None

        if self.state == 'gil':
            code.put_release_ensured_gil(variable)
        else:
            code.put_acquire_gil(variable)


class EnsureGILNode(GILExitNode):
    """
    Ensure the GIL in nogil functions for cleanup before returning.
    """

    def generate_execution_code(self, code):
        code.put_ensure_gil(declare_gilstate=False)


def cython_view_utility_code():
    from . import MemoryView
    return MemoryView.view_utility_code


utility_code_for_cimports = {
    # utility code (or inlining c) in a pxd (or pyx) file.
    # TODO: Consider a generic user-level mechanism for importing
    'cpython.array'         : lambda : UtilityCode.load_cached("ArrayAPI", "arrayarray.h"),
    'cpython.array.array'   : lambda : UtilityCode.load_cached("ArrayAPI", "arrayarray.h"),
    'cython.view'           : cython_view_utility_code,
}

utility_code_for_imports = {
    # utility code used when special modules are imported.
    # TODO: Consider a generic user-level mechanism for importing
    'asyncio': ("__Pyx_patch_asyncio", "PatchAsyncIO", "Coroutine.c"),
    'inspect': ("__Pyx_patch_inspect", "PatchInspect", "Coroutine.c"),
}


class CImportStatNode(StatNode):
    #  cimport statement
    #
    #  module_name   string           Qualified name of module being imported
    #  as_name       string or None   Name specified in "as" clause, if any
    #  is_absolute   bool             True for absolute imports, False otherwise

    child_attrs = []
    is_absolute = False

    def analyse_declarations(self, env):
        if not env.is_module_scope:
            error(self.pos, "cimport only allowed at module level")
            return
        module_scope = env.find_module(
            self.module_name, self.pos, relative_level=0 if self.is_absolute else -1)
        if "." in self.module_name:
            names = [EncodedString(name) for name in self.module_name.split(".")]
            top_name = names[0]
            top_module_scope = env.context.find_submodule(top_name)
            module_scope = top_module_scope
            for name in names[1:]:
                submodule_scope = module_scope.find_submodule(name)
                module_scope.declare_module(name, submodule_scope, self.pos)
                module_scope = submodule_scope
            if self.as_name:
                env.declare_module(self.as_name, module_scope, self.pos)
            else:
                env.add_imported_module(module_scope)
                env.declare_module(top_name, top_module_scope, self.pos)
        else:
            name = self.as_name or self.module_name
            env.declare_module(name, module_scope, self.pos)
        if self.module_name in utility_code_for_cimports:
            env.use_utility_code(utility_code_for_cimports[self.module_name]())

    def analyse_expressions(self, env):
        return self

    def generate_execution_code(self, code):
        pass


class FromCImportStatNode(StatNode):
    #  from ... cimport statement
    #
    #  module_name     string                        Qualified name of module
    #  relative_level  int or None                   Relative import: number of dots before module_name
    #  imported_names  [(pos, name, as_name, kind)]  Names to be imported

    child_attrs = []
    module_name = None
    relative_level = None
    imported_names = None

    def analyse_declarations(self, env):
        if not env.is_module_scope:
            error(self.pos, "cimport only allowed at module level")
            return
        if self.relative_level and self.relative_level > env.qualified_name.count('.'):
            error(self.pos, "relative cimport beyond main package is not allowed")
            return
        module_scope = env.find_module(self.module_name, self.pos, relative_level=self.relative_level)
        module_name = module_scope.qualified_name
        env.add_imported_module(module_scope)
        for pos, name, as_name, kind in self.imported_names:
            if name == "*":
                for local_name, entry in list(module_scope.entries.items()):
                    env.add_imported_entry(local_name, entry, pos)
            else:
                entry = module_scope.lookup(name)
                if entry:
                    if kind and not self.declaration_matches(entry, kind):
                        entry.redeclared(pos)
                    entry.used = 1
                else:
                    if kind == 'struct' or kind == 'union':
                        entry = module_scope.declare_struct_or_union(
                            name, kind=kind, scope=None, typedef_flag=0, pos=pos)
                    elif kind == 'class':
                        entry = module_scope.declare_c_class(name, pos=pos, module_name=module_name)
                    else:
                        submodule_scope = env.context.find_module(
                            name, relative_to=module_scope, pos=self.pos, absolute_fallback=False)
                        if submodule_scope.parent_module is module_scope:
                            env.declare_module(as_name or name, submodule_scope, self.pos)
                        else:
                            error(pos, "Name '%s' not declared in module '%s'" % (name, module_name))

                if entry:
                    local_name = as_name or name
                    env.add_imported_entry(local_name, entry, pos)

        if module_name.startswith('cpython') or module_name.startswith('cython'): # enough for now
            if module_name in utility_code_for_cimports:
                env.use_utility_code(utility_code_for_cimports[module_name]())
            for _, name, _, _ in self.imported_names:
                fqname = '%s.%s' % (module_name, name)
                if fqname in utility_code_for_cimports:
                    env.use_utility_code(utility_code_for_cimports[fqname]())

    def declaration_matches(self, entry, kind):
        if not entry.is_type:
            return 0
        type = entry.type
        if kind == 'class':
            if not type.is_extension_type:
                return 0
        else:
            if not type.is_struct_or_union:
                return 0
            if kind != type.kind:
                return 0
        return 1

    def analyse_expressions(self, env):
        return self

    def generate_execution_code(self, code):
        pass


class FromImportStatNode(StatNode):
    #  from ... import statement
    #
    #  module           ImportNode
    #  items            [(string, NameNode)]
    #  interned_items   [(string, NameNode, ExprNode)]
    #  item             PyTempNode            used internally
    #  import_star      boolean               used internally

    child_attrs = ["module"]
    import_star = 0

    def analyse_declarations(self, env):
        for name, target in self.items:
            if name == "*":
                if not env.is_module_scope:
                    error(self.pos, "import * only allowed at module level")
                    return
                env.has_import_star = 1
                self.import_star = 1
            else:
                target.analyse_target_declaration(env)

    def analyse_expressions(self, env):
        from . import ExprNodes
        self.module = self.module.analyse_expressions(env)
        self.item = ExprNodes.RawCNameExprNode(self.pos, py_object_type)
        self.interned_items = []
        for name, target in self.items:
            if name == '*':
                for _, entry in env.entries.items():
                    if not entry.is_type and entry.type.is_extension_type:
                        env.use_utility_code(UtilityCode.load_cached("ExtTypeTest", "ObjectHandling.c"))
                        break
            else:
                entry = env.lookup(target.name)
                # check whether or not entry is already cimported
                if (entry.is_type and entry.type.name == name
                        and hasattr(entry.type, 'module_name')):
                    if entry.type.module_name == self.module.module_name.value:
                        # cimported with absolute name
                        continue
                    try:
                        # cimported with relative name
                        module = env.find_module(self.module.module_name.value, pos=self.pos,
                                                 relative_level=self.module.level)
                        if entry.type.module_name == module.qualified_name:
                            continue
                    except AttributeError:
                        pass
                target = target.analyse_target_expression(env, None)  # FIXME?
                if target.type is py_object_type:
                    coerced_item = None
                else:
                    coerced_item = self.item.coerce_to(target.type, env)
                self.interned_items.append((name, target, coerced_item))
        return self

    def generate_execution_code(self, code):
        code.mark_pos(self.pos)
        self.module.generate_evaluation_code(code)
        if self.import_star:
            code.putln(
                'if (%s(%s) < 0) %s;' % (
                    Naming.import_star,
                    self.module.py_result(),
                    code.error_goto(self.pos)))
        item_temp = code.funcstate.allocate_temp(py_object_type, manage_ref=True)
        self.item.set_cname(item_temp)
        if self.interned_items:
            code.globalstate.use_utility_code(
                UtilityCode.load_cached("ImportFrom", "ImportExport.c"))
        for name, target, coerced_item in self.interned_items:
            code.putln(
                '%s = __Pyx_ImportFrom(%s, %s); %s' % (
                    item_temp,
                    self.module.py_result(),
                    code.intern_identifier(name),
                    code.error_goto_if_null(item_temp, self.pos)))
            code.put_gotref(item_temp)
            if coerced_item is None:
                target.generate_assignment_code(self.item, code)
            else:
                coerced_item.allocate_temp_result(code)
                coerced_item.generate_result_code(code)
                target.generate_assignment_code(coerced_item, code)
            code.put_decref_clear(item_temp, py_object_type)
        code.funcstate.release_temp(item_temp)
        self.module.generate_disposal_code(code)
        self.module.free_temps(code)


class ParallelNode(Node):
    """
    Base class for cython.parallel constructs.
    """

    nogil_check = None


class ParallelStatNode(StatNode, ParallelNode):
    """
    Base class for 'with cython.parallel.parallel():' and 'for i in prange():'.

    assignments     { Entry(var) : (var.pos, inplace_operator_or_None) }
                    assignments to variables in this parallel section

    parent          parent ParallelStatNode or None
    is_parallel     indicates whether this node is OpenMP parallel
                    (true for #pragma omp parallel for and
                              #pragma omp parallel)

    is_parallel is true for:

        #pragma omp parallel
        #pragma omp parallel for

    sections, but NOT for

        #pragma omp for

    We need this to determine the sharing attributes.

    privatization_insertion_point   a code insertion point used to make temps
                                    private (esp. the "nsteps" temp)

    args         tuple          the arguments passed to the parallel construct
    kwargs       DictNode       the keyword arguments passed to the parallel
                                construct (replaced by its compile time value)
    """

    child_attrs = ['body', 'num_threads']

    body = None

    is_prange = False
    is_nested_prange = False

    error_label_used = False

    num_threads = None
    chunksize = None

    parallel_exc = (
        Naming.parallel_exc_type,
        Naming.parallel_exc_value,
        Naming.parallel_exc_tb,
    )

    parallel_pos_info = (
        Naming.parallel_filename,
        Naming.parallel_lineno,
        Naming.parallel_clineno,
    )

    pos_info = (
        Naming.filename_cname,
        Naming.lineno_cname,
        Naming.clineno_cname,
    )

    critical_section_counter = 0

    def __init__(self, pos, **kwargs):
        super(ParallelStatNode, self).__init__(pos, **kwargs)

        # All assignments in this scope
        self.assignments = kwargs.get('assignments') or {}

        # All seen closure cnames and their temporary cnames
        self.seen_closure_vars = set()

        # Dict of variables that should be declared (first|last|)private or
        # reduction { Entry: (op, lastprivate) }.
        # If op is not None, it's a reduction.
        self.privates = {}

        # [NameNode]
        self.assigned_nodes = []

    def analyse_declarations(self, env):
        self.body.analyse_declarations(env)

        self.num_threads = None

        if self.kwargs:
            # Try to find num_threads and chunksize keyword arguments
            pairs = []
            seen = set()
            for dictitem in self.kwargs.key_value_pairs:
                if dictitem.key.value in seen:
                    error(self.pos, "Duplicate keyword argument found: %s" % dictitem.key.value)
                seen.add(dictitem.key.value)
                if dictitem.key.value == 'num_threads':
                    if not dictitem.value.is_none:
                       self.num_threads = dictitem.value
                elif self.is_prange and dictitem.key.value == 'chunksize':
                    if not dictitem.value.is_none:
                        self.chunksize = dictitem.value
                else:
                    pairs.append(dictitem)

            self.kwargs.key_value_pairs = pairs

            try:
                self.kwargs = self.kwargs.compile_time_value(env)
            except Exception as e:
                error(self.kwargs.pos, "Only compile-time values may be "
                                       "supplied as keyword arguments")
        else:
            self.kwargs = {}

        for kw, val in self.kwargs.items():
            if kw not in self.valid_keyword_arguments:
                error(self.pos, "Invalid keyword argument: %s" % kw)
            else:
                setattr(self, kw, val)

    def analyse_expressions(self, env):
        if self.num_threads:
            self.num_threads = self.num_threads.analyse_expressions(env)

        if self.chunksize:
            self.chunksize = self.chunksize.analyse_expressions(env)

        self.body = self.body.analyse_expressions(env)
        self.analyse_sharing_attributes(env)

        if self.num_threads is not None:
            if self.parent and self.parent.num_threads is not None and not self.parent.is_prange:
                error(self.pos, "num_threads already declared in outer section")
            elif self.parent and not self.parent.is_prange:
                error(self.pos, "num_threads must be declared in the parent parallel section")
            elif (self.num_threads.type.is_int and
                    self.num_threads.is_literal and
                    self.num_threads.compile_time_value(env) <= 0):
                error(self.pos, "argument to num_threads must be greater than 0")

            if not self.num_threads.is_simple() or self.num_threads.type.is_pyobject:
                self.num_threads = self.num_threads.coerce_to(
                    PyrexTypes.c_int_type, env).coerce_to_temp(env)
        return self

    def analyse_sharing_attributes(self, env):
        """
        Analyse the privates for this block and set them in self.privates.
        This should be called in a post-order fashion during the
        analyse_expressions phase
        """
        for entry, (pos, op) in self.assignments.items():

            if self.is_prange and not self.is_parallel:
                # closely nested prange in a with parallel block, disallow
                # assigning to privates in the with parallel block (we
                # consider it too implicit and magicky for users)
                if entry in self.parent.assignments:
                    error(pos, "Cannot assign to private of outer parallel block")
                    continue

            if not self.is_prange and op:
                # Again possible, but considered to magicky
                error(pos, "Reductions not allowed for parallel blocks")
                continue

            # By default all variables should have the same values as if
            # executed sequentially
            lastprivate = True
            self.propagate_var_privatization(entry, pos, op, lastprivate)

    def propagate_var_privatization(self, entry, pos, op, lastprivate):
        """
        Propagate the sharing attributes of a variable. If the privatization is
        determined by a parent scope, done propagate further.

        If we are a prange, we propagate our sharing attributes outwards to
        other pranges. If we are a prange in parallel block and the parallel
        block does not determine the variable private, we propagate to the
        parent of the parent. Recursion stops at parallel blocks, as they have
        no concept of lastprivate or reduction.

        So the following cases propagate:

            sum is a reduction for all loops:

                for i in prange(n):
                    for j in prange(n):
                        for k in prange(n):
                            sum += i * j * k

            sum is a reduction for both loops, local_var is private to the
            parallel with block:

                for i in prange(n):
                    with parallel:
                        local_var = ... # private to the parallel
                        for j in prange(n):
                            sum += i * j

        Nested with parallel blocks are disallowed, because they wouldn't
        allow you to propagate lastprivates or reductions:

            #pragma omp parallel for lastprivate(i)
            for i in prange(n):

                sum = 0

                #pragma omp parallel private(j, sum)
                with parallel:

                    #pragma omp parallel
                    with parallel:

                        #pragma omp for lastprivate(j) reduction(+:sum)
                        for j in prange(n):
                            sum += i

                    # sum and j are well-defined here

                # sum and j are undefined here

            # sum and j are undefined here
        """
        self.privates[entry] = (op, lastprivate)

        if entry.type.is_memoryviewslice:
            error(pos, "Memoryview slices can only be shared in parallel sections")
            return

        if self.is_prange:
            if not self.is_parallel and entry not in self.parent.assignments:
                # Parent is a parallel with block
                parent = self.parent.parent
            else:
                parent = self.parent

            # We don't need to propagate privates, only reductions and
            # lastprivates
            if parent and (op or lastprivate):
                parent.propagate_var_privatization(entry, pos, op, lastprivate)

    def _allocate_closure_temp(self, code, entry):
        """
        Helper function that allocate a temporary for a closure variable that
        is assigned to.
        """
        if self.parent:
            return self.parent._allocate_closure_temp(code, entry)

        if entry.cname in self.seen_closure_vars:
            return entry.cname

        cname = code.funcstate.allocate_temp(entry.type, True)

        # Add both the actual cname and the temp cname, as the actual cname
        # will be replaced with the temp cname on the entry
        self.seen_closure_vars.add(entry.cname)
        self.seen_closure_vars.add(cname)

        self.modified_entries.append((entry, entry.cname))
        code.putln("%s = %s;" % (cname, entry.cname))
        entry.cname = cname

    def initialize_privates_to_nan(self, code, exclude=None):
        first = True

        for entry, (op, lastprivate) in sorted(self.privates.items()):
            if not op and (not exclude or entry != exclude):
                invalid_value = entry.type.invalid_value()

                if invalid_value:
                    if first:
                        code.putln("/* Initialize private variables to "
                                   "invalid values */")
                        first = False
                    code.putln("%s = %s;" % (entry.cname,
                                             entry.type.cast_code(invalid_value)))

    def evaluate_before_block(self, code, expr):
        c = self.begin_of_parallel_control_block_point_after_decls
        # we need to set the owner to ourselves temporarily, as
        # allocate_temp may generate a comment in the middle of our pragma
        # otherwise when DebugFlags.debug_temp_code_comments is in effect
        owner = c.funcstate.owner
        c.funcstate.owner = c
        expr.generate_evaluation_code(c)
        c.funcstate.owner = owner

        return expr.result()

    def put_num_threads(self, code):
        """
        Write self.num_threads if set as the num_threads OpenMP directive
        """
        if self.num_threads is not None:
            code.put(" num_threads(%s)" % self.evaluate_before_block(code, self.num_threads))


    def declare_closure_privates(self, code):
        """
        If a variable is in a scope object, we need to allocate a temp and
        assign the value from the temp to the variable in the scope object
        after the parallel section. This kind of copying should be done only
        in the outermost parallel section.
        """
        self.modified_entries = []

        for entry in sorted(self.assignments):
            if entry.from_closure or entry.in_closure:
                self._allocate_closure_temp(code, entry)

    def release_closure_privates(self, code):
        """
        Release any temps used for variables in scope objects. As this is the
        outermost parallel block, we don't need to delete the cnames from
        self.seen_closure_vars.
        """
        for entry, original_cname in self.modified_entries:
            code.putln("%s = %s;" % (original_cname, entry.cname))
            code.funcstate.release_temp(entry.cname)
            entry.cname = original_cname

    def privatize_temps(self, code, exclude_temps=()):
        """
        Make any used temporaries private. Before the relevant code block
        code.start_collecting_temps() should have been called.
        """
        c = self.privatization_insertion_point
        self.privatization_insertion_point = None

        if self.is_parallel:
            self.temps = temps = code.funcstate.stop_collecting_temps()
            privates, firstprivates = [], []
            for temp, type in sorted(temps):
                if type.is_pyobject or type.is_memoryviewslice:
                    firstprivates.append(temp)
                else:
                    privates.append(temp)

            if privates:
                c.put(" private(%s)" % ", ".join(privates))
            if firstprivates:
                c.put(" firstprivate(%s)" % ", ".join(firstprivates))

            if self.breaking_label_used:
                shared_vars = [Naming.parallel_why]
                if self.error_label_used:
                    shared_vars.extend(self.parallel_exc)
                    c.put(" private(%s, %s, %s)" % self.pos_info)

                c.put(" shared(%s)" % ', '.join(shared_vars))

    def cleanup_temps(self, code):
        # Now clean up any memoryview slice and object temporaries
        if self.is_parallel and not self.is_nested_prange:
            code.putln("/* Clean up any temporaries */")
            for temp, type in sorted(self.temps):
                if type.is_memoryviewslice:
                    code.put_xdecref_memoryviewslice(temp, have_gil=False)
                elif type.is_pyobject:
                    code.put_xdecref(temp, type)
                    code.putln("%s = NULL;" % temp)

    def setup_parallel_control_flow_block(self, code):
        """
        Sets up a block that surrounds the parallel block to determine
        how the parallel section was exited. Any kind of return is
        trapped (break, continue, return, exceptions). This is the idea:

        {
            int why = 0;

            #pragma omp parallel
            {
                return # -> goto new_return_label;
                goto end_parallel;

            new_return_label:
                why = 3;
                goto end_parallel;

            end_parallel:;
                #pragma omp flush(why) # we need to flush for every iteration
            }

            if (why == 3)
                goto old_return_label;
        }
        """
        self.old_loop_labels = code.new_loop_labels()
        self.old_error_label = code.new_error_label()
        self.old_return_label = code.return_label
        code.return_label = code.new_label(name="return")

        code.begin_block() # parallel control flow block
        self.begin_of_parallel_control_block_point = code.insertion_point()
        self.begin_of_parallel_control_block_point_after_decls = code.insertion_point()

        self.undef_builtin_expect_apple_gcc_bug(code)

    def begin_parallel_block(self, code):
        """
        Each OpenMP thread in a parallel section that contains a with gil block
        must have the thread-state initialized. The call to
        PyGILState_Release() then deallocates our threadstate. If we wouldn't
        do this, each with gil block would allocate and deallocate one, thereby
        losing exception information before it can be saved before leaving the
        parallel section.
        """
        self.begin_of_parallel_block = code.insertion_point()

    def end_parallel_block(self, code):
        """
        To ensure all OpenMP threads have thread states, we ensure the GIL
        in each thread (which creates a thread state if it doesn't exist),
        after which we release the GIL.
        On exit, reacquire the GIL and release the thread state.

        If compiled without OpenMP support (at the C level), then we still have
        to acquire the GIL to decref any object temporaries.
        """
        begin_code = self.begin_of_parallel_block
        self.begin_of_parallel_block = None

        if self.error_label_used:
            end_code = code

            begin_code.putln("#ifdef _OPENMP")
            begin_code.put_ensure_gil(declare_gilstate=True)
            begin_code.putln("Py_BEGIN_ALLOW_THREADS")
            begin_code.putln("#endif /* _OPENMP */")

            end_code.putln("#ifdef _OPENMP")
            end_code.putln("Py_END_ALLOW_THREADS")
            end_code.putln("#else")
            end_code.put_safe("{\n")
            end_code.put_ensure_gil()
            end_code.putln("#endif /* _OPENMP */")
            self.cleanup_temps(end_code)
            end_code.put_release_ensured_gil()
            end_code.putln("#ifndef _OPENMP")
            end_code.put_safe("}\n")
            end_code.putln("#endif /* _OPENMP */")

    def trap_parallel_exit(self, code, should_flush=False):
        """
        Trap any kind of return inside a parallel construct. 'should_flush'
        indicates whether the variable should be flushed, which is needed by
        prange to skip the loop. It also indicates whether we need to register
        a continue (we need this for parallel blocks, but not for prange
        loops, as it is a direct jump there).

        It uses the same mechanism as try/finally:
            1 continue
            2 break
            3 return
            4 error
        """
        save_lastprivates_label = code.new_label()
        dont_return_label = code.new_label()

        self.any_label_used = False
        self.breaking_label_used = False
        self.error_label_used = False

        self.parallel_private_temps = []

        all_labels = code.get_all_labels()

        # Figure this out before starting to generate any code
        for label in all_labels:
            if code.label_used(label):
                self.breaking_label_used = (self.breaking_label_used or
                                            label != code.continue_label)
                self.any_label_used = True

        if self.any_label_used:
            code.put_goto(dont_return_label)

        for i, label in enumerate(all_labels):
            if not code.label_used(label):
                continue

            is_continue_label = label == code.continue_label

            code.put_label(label)

            if not (should_flush and is_continue_label):
                if label == code.error_label:
                    self.error_label_used = True
                    self.fetch_parallel_exception(code)

                code.putln("%s = %d;" % (Naming.parallel_why, i + 1))

            if (self.breaking_label_used and self.is_prange and not
                    is_continue_label):
                code.put_goto(save_lastprivates_label)
            else:
                code.put_goto(dont_return_label)

        if self.any_label_used:
            if self.is_prange and self.breaking_label_used:
                # Don't rely on lastprivate, save our lastprivates
                code.put_label(save_lastprivates_label)
                self.save_parallel_vars(code)

            code.put_label(dont_return_label)

            if should_flush and self.breaking_label_used:
                code.putln_openmp("#pragma omp flush(%s)" % Naming.parallel_why)

    def save_parallel_vars(self, code):
        """
        The following shenanigans are instated when we break, return or
        propagate errors from a prange. In this case we cannot rely on
        lastprivate() to do its job, as no iterations may have executed yet
        in the last thread, leaving the values undefined. It is most likely
        that the breaking thread has well-defined values of the lastprivate
        variables, so we keep those values.
        """
        section_name = "__pyx_parallel_lastprivates%d" % self.critical_section_counter
        code.putln_openmp("#pragma omp critical(%s)" % section_name)
        ParallelStatNode.critical_section_counter += 1

        code.begin_block() # begin critical section

        c = self.begin_of_parallel_control_block_point

        temp_count = 0
        for entry, (op, lastprivate) in sorted(self.privates.items()):
            if not lastprivate or entry.type.is_pyobject:
                continue

            type_decl = entry.type.empty_declaration_code()
            temp_cname = "__pyx_parallel_temp%d" % temp_count
            private_cname = entry.cname

            temp_count += 1

            invalid_value = entry.type.invalid_value()
            if invalid_value:
                init = ' = ' + entry.type.cast_code(invalid_value)
            else:
                init = ''
            # Declare the parallel private in the outer block
            c.putln("%s %s%s;" % (type_decl, temp_cname, init))

            # Initialize before escaping
            code.putln("%s = %s;" % (temp_cname, private_cname))

            self.parallel_private_temps.append((temp_cname, private_cname))

        code.end_block() # end critical section

    def fetch_parallel_exception(self, code):
        """
        As each OpenMP thread may raise an exception, we need to fetch that
        exception from the threadstate and save it for after the parallel
        section where it can be re-raised in the master thread.

        Although it would seem that __pyx_filename, __pyx_lineno and
        __pyx_clineno are only assigned to under exception conditions (i.e.,
        when we have the GIL), and thus should be allowed to be shared without
        any race condition, they are in fact subject to the same race
        conditions that they were previously when they were global variables
        and functions were allowed to release the GIL:

            thread A                thread B
                acquire
                set lineno
                release
                                        acquire
                                        set lineno
                                        release
                acquire
                fetch exception
                release
                                        skip the fetch

                deallocate threadstate  deallocate threadstate
        """
        code.begin_block()
        code.put_ensure_gil(declare_gilstate=True)

        code.putln_openmp("#pragma omp flush(%s)" % Naming.parallel_exc_type)
        code.putln(
            "if (!%s) {" % Naming.parallel_exc_type)

        code.putln("__Pyx_ErrFetchWithState(&%s, &%s, &%s);" % self.parallel_exc)
        pos_info = chain(*zip(self.parallel_pos_info, self.pos_info))
        code.funcstate.uses_error_indicator = True
        code.putln("%s = %s; %s = %s; %s = %s;" % tuple(pos_info))
        code.put_gotref(Naming.parallel_exc_type)

        code.putln(
            "}")

        code.put_release_ensured_gil()
        code.end_block()

    def restore_parallel_exception(self, code):
        "Re-raise a parallel exception"
        code.begin_block()
        code.put_ensure_gil(declare_gilstate=True)

        code.put_giveref(Naming.parallel_exc_type)
        code.putln("__Pyx_ErrRestoreWithState(%s, %s, %s);" % self.parallel_exc)
        pos_info = chain(*zip(self.pos_info, self.parallel_pos_info))
        code.putln("%s = %s; %s = %s; %s = %s;" % tuple(pos_info))

        code.put_release_ensured_gil()
        code.end_block()

    def restore_labels(self, code):
        """
        Restore all old labels. Call this before the 'else' clause to for
        loops and always before ending the parallel control flow block.
        """
        code.set_all_labels(self.old_loop_labels + (self.old_return_label,
                                                    self.old_error_label))

    def end_parallel_control_flow_block(
            self, code, break_=False, continue_=False, return_=False):
        """
        This ends the parallel control flow block and based on how the parallel
        section was exited, takes the corresponding action. The break_ and
        continue_ parameters indicate whether these should be propagated
        outwards:

            for i in prange(...):
                with cython.parallel.parallel():
                    continue

        Here break should be trapped in the parallel block, and propagated to
        the for loop.
        """
        c = self.begin_of_parallel_control_block_point
        self.begin_of_parallel_control_block_point = None
        self.begin_of_parallel_control_block_point_after_decls = None

        if self.num_threads is not None:
            # FIXME: is it the right place? should not normally produce code.
            self.num_threads.generate_disposal_code(code)
            self.num_threads.free_temps(code)

        # Firstly, always prefer errors over returning, continue or break
        if self.error_label_used:
            c.putln("const char *%s = NULL; int %s = 0, %s = 0;" % self.parallel_pos_info)
            c.putln("PyObject *%s = NULL, *%s = NULL, *%s = NULL;" % self.parallel_exc)

            code.putln(
                "if (%s) {" % Naming.parallel_exc_type)
            code.putln("/* This may have been overridden by a continue, "
                       "break or return in another thread. Prefer the error. */")
            code.putln("%s = 4;" % Naming.parallel_why)
            code.putln(
                "}")

        if continue_:
            any_label_used = self.any_label_used
        else:
            any_label_used = self.breaking_label_used

        if any_label_used:
            # __pyx_parallel_why is used, declare and initialize
            c.putln("int %s;" % Naming.parallel_why)
            c.putln("%s = 0;" % Naming.parallel_why)

            code.putln(
                "if (%s) {" % Naming.parallel_why)

            for temp_cname, private_cname in self.parallel_private_temps:
                code.putln("%s = %s;" % (private_cname, temp_cname))

            code.putln("switch (%s) {" % Naming.parallel_why)
            if continue_:
                code.put("    case 1: ")
                code.put_goto(code.continue_label)

            if break_:
                code.put("    case 2: ")
                code.put_goto(code.break_label)

            if return_:
                code.put("    case 3: ")
                code.put_goto(code.return_label)

            if self.error_label_used:
                code.globalstate.use_utility_code(restore_exception_utility_code)
                code.putln("    case 4:")
                self.restore_parallel_exception(code)
                code.put_goto(code.error_label)

            code.putln("}") # end switch
            code.putln(
                "}") # end if

        code.end_block() # end parallel control flow block
        self.redef_builtin_expect_apple_gcc_bug(code)

    # FIXME: improve with version number for OS X Lion
    buggy_platform_macro_condition = "(defined(__APPLE__) || defined(__OSX__))"
    have_expect_condition = "(defined(__GNUC__) && " \
                             "(__GNUC__ > 2 || (__GNUC__ == 2 && (__GNUC_MINOR__ > 95))))"
    redef_condition = "(%s && %s)" % (buggy_platform_macro_condition, have_expect_condition)

    def undef_builtin_expect_apple_gcc_bug(self, code):
        """
        A bug on OS X Lion disallows __builtin_expect macros. This code avoids them
        """
        if not self.parent:
            code.undef_builtin_expect(self.redef_condition)

    def redef_builtin_expect_apple_gcc_bug(self, code):
        if not self.parent:
            code.redef_builtin_expect(self.redef_condition)


class ParallelWithBlockNode(ParallelStatNode):
    """
    This node represents a 'with cython.parallel.parallel():' block
    """

    valid_keyword_arguments = ['num_threads']

    num_threads = None

    def analyse_declarations(self, env):
        super(ParallelWithBlockNode, self).analyse_declarations(env)
        if self.args:
            error(self.pos, "cython.parallel.parallel() does not take "
                            "positional arguments")

    def generate_execution_code(self, code):
        self.declare_closure_privates(code)
        self.setup_parallel_control_flow_block(code)

        code.putln("#ifdef _OPENMP")
        code.put("#pragma omp parallel ")

        if self.privates:
            privates = [e.cname for e in self.privates
                        if not e.type.is_pyobject]
            code.put('private(%s)' % ', '.join(sorted(privates)))

        self.privatization_insertion_point = code.insertion_point()
        self.put_num_threads(code)
        code.putln("")

        code.putln("#endif /* _OPENMP */")

        code.begin_block()  # parallel block
        self.begin_parallel_block(code)
        self.initialize_privates_to_nan(code)
        code.funcstate.start_collecting_temps()
        self.body.generate_execution_code(code)
        self.trap_parallel_exit(code)
        self.privatize_temps(code)
        self.end_parallel_block(code)
        code.end_block()  # end parallel block

        continue_ = code.label_used(code.continue_label)
        break_ = code.label_used(code.break_label)
        return_ = code.label_used(code.return_label)

        self.restore_labels(code)
        self.end_parallel_control_flow_block(code, break_=break_,
                                             continue_=continue_,
                                             return_=return_)
        self.release_closure_privates(code)


class ParallelRangeNode(ParallelStatNode):
    """
    This node represents a 'for i in cython.parallel.prange():' construct.

    target       NameNode       the target iteration variable
    else_clause  Node or None   the else clause of this loop
    """

    child_attrs = ['body', 'target', 'else_clause', 'args', 'num_threads',
                   'chunksize']

    body = target = else_clause = args = None

    start = stop = step = None

    is_prange = True

    nogil = None
    schedule = None

    valid_keyword_arguments = ['schedule', 'nogil', 'num_threads', 'chunksize']

    def __init__(self, pos, **kwds):
        super(ParallelRangeNode, self).__init__(pos, **kwds)
        # Pretend to be a ForInStatNode for control flow analysis
        self.iterator = PassStatNode(pos)

    def analyse_declarations(self, env):
        super(ParallelRangeNode, self).analyse_declarations(env)
        self.target.analyse_target_declaration(env)
        if self.else_clause is not None:
            self.else_clause.analyse_declarations(env)

        if not self.args or len(self.args) > 3:
            error(self.pos, "Invalid number of positional arguments to prange")
            return

        if len(self.args) == 1:
            self.stop, = self.args
        elif len(self.args) == 2:
            self.start, self.stop = self.args
        else:
            self.start, self.stop, self.step = self.args

        if hasattr(self.schedule, 'decode'):
            self.schedule = self.schedule.decode('ascii')

        if self.schedule not in (None, 'static', 'dynamic', 'guided', 'runtime'):
            error(self.pos, "Invalid schedule argument to prange: %s" % (self.schedule,))

    def analyse_expressions(self, env):
        was_nogil = env.nogil
        if self.nogil:
            env.nogil = True

        if self.target is None:
            error(self.pos, "prange() can only be used as part of a for loop")
            return self

        self.target = self.target.analyse_target_types(env)

        if not self.target.type.is_numeric:
            # Not a valid type, assume one for now anyway

            if not self.target.type.is_pyobject:
                # nogil_check will catch the is_pyobject case
                error(self.target.pos,
                      "Must be of numeric type, not %s" % self.target.type)

            self.index_type = PyrexTypes.c_py_ssize_t_type
        else:
            self.index_type = self.target.type

        # Setup start, stop and step, allocating temps if needed
        self.names = 'start', 'stop', 'step'
        start_stop_step = self.start, self.stop, self.step

        for node, name in zip(start_stop_step, self.names):
            if node is not None:
                node.analyse_types(env)
                if not node.type.is_numeric:
                    error(node.pos, "%s argument must be numeric" % name)
                    continue

                if not node.is_literal:
                    node = node.coerce_to_temp(env)
                    setattr(self, name, node)

                # As we range from 0 to nsteps, computing the index along the
                # way, we need a fitting type for 'i' and 'nsteps'
                self.index_type = PyrexTypes.widest_numeric_type(
                    self.index_type, node.type)

        if self.else_clause is not None:
            self.else_clause = self.else_clause.analyse_expressions(env)

        # Although not actually an assignment in this scope, it should be
        # treated as such to ensure it is unpacked if a closure temp, and to
        # ensure lastprivate behaviour and propagation. If the target index is
        # not a NameNode, it won't have an entry, and an error was issued by
        # ParallelRangeTransform
        if hasattr(self.target, 'entry'):
            self.assignments[self.target.entry] = self.target.pos, None

        node = super(ParallelRangeNode, self).analyse_expressions(env)

        if node.chunksize:
            if not node.schedule:
                error(node.chunksize.pos,
                      "Must provide schedule with chunksize")
            elif node.schedule == 'runtime':
                error(node.chunksize.pos,
                      "Chunksize not valid for the schedule runtime")
            elif (node.chunksize.type.is_int and
                  node.chunksize.is_literal and
                  node.chunksize.compile_time_value(env) <= 0):
                error(node.chunksize.pos, "Chunksize must not be negative")

            node.chunksize = node.chunksize.coerce_to(
                PyrexTypes.c_int_type, env).coerce_to_temp(env)

        if node.nogil:
            env.nogil = was_nogil

        node.is_nested_prange = node.parent and node.parent.is_prange
        if node.is_nested_prange:
            parent = node
            while parent.parent and parent.parent.is_prange:
                parent = parent.parent

            parent.assignments.update(node.assignments)
            parent.privates.update(node.privates)
            parent.assigned_nodes.extend(node.assigned_nodes)
        return node

    def nogil_check(self, env):
        names = 'start', 'stop', 'step', 'target'
        nodes = self.start, self.stop, self.step, self.target
        for name, node in zip(names, nodes):
            if node is not None and node.type.is_pyobject:
                error(node.pos, "%s may not be a Python object "
                                "as we don't have the GIL" % name)

    def generate_execution_code(self, code):
        """
        Generate code in the following steps

            1)  copy any closure variables determined thread-private
                into temporaries

            2)  allocate temps for start, stop and step

            3)  generate a loop that calculates the total number of steps,
                which then computes the target iteration variable for every step:

                    for i in prange(start, stop, step):
                        ...

                becomes

                    nsteps = (stop - start) / step;
                    i = start;

                    #pragma omp parallel for lastprivate(i)
                    for (temp = 0; temp < nsteps; temp++) {
                        i = start + step * temp;
                        ...
                    }

                Note that accumulation of 'i' would have a data dependency
                between iterations.

                Also, you can't do this

                    for (i = start; i < stop; i += step)
                        ...

                as the '<' operator should become '>' for descending loops.
                'for i from x < i < y:' does not suffer from this problem
                as the relational operator is known at compile time!

            4) release our temps and write back any private closure variables
        """
        self.declare_closure_privates(code)

        # This can only be a NameNode
        target_index_cname = self.target.entry.cname

        # This will be used as the dict to format our code strings, holding
        # the start, stop , step, temps and target cnames
        fmt_dict = {
            'target': target_index_cname,
            'target_type': self.target.type.empty_declaration_code()
        }

        # Setup start, stop and step, allocating temps if needed
        start_stop_step = self.start, self.stop, self.step
        defaults = '0', '0', '1'
        for node, name, default in zip(start_stop_step, self.names, defaults):
            if node is None:
                result = default
            elif node.is_literal:
                result = node.get_constant_c_result_code()
            else:
                node.generate_evaluation_code(code)
                result = node.result()

            fmt_dict[name] = result

        fmt_dict['i'] = code.funcstate.allocate_temp(self.index_type, False)
        fmt_dict['nsteps'] = code.funcstate.allocate_temp(self.index_type, False)

        # TODO: check if the step is 0 and if so, raise an exception in a
        # 'with gil' block. For now, just abort
        code.putln("if ((%(step)s == 0)) abort();" % fmt_dict)

        self.setup_parallel_control_flow_block(code) # parallel control flow block

        # Note: nsteps is private in an outer scope if present
        code.putln("%(nsteps)s = (%(stop)s - %(start)s + %(step)s - %(step)s/abs(%(step)s)) / %(step)s;" % fmt_dict)

        # The target iteration variable might not be initialized, do it only if
        # we are executing at least 1 iteration, otherwise we should leave the
        # target unaffected. The target iteration variable is firstprivate to
        # shut up compiler warnings caused by lastprivate, as the compiler
        # erroneously believes that nsteps may be <= 0, leaving the private
        # target index uninitialized
        code.putln("if (%(nsteps)s > 0)" % fmt_dict)
        code.begin_block() # if block
        self.generate_loop(code, fmt_dict)
        code.end_block() # end if block

        self.restore_labels(code)

        if self.else_clause:
            if self.breaking_label_used:
                code.put("if (%s < 2)" % Naming.parallel_why)

            code.begin_block() # else block
            code.putln("/* else */")
            self.else_clause.generate_execution_code(code)
            code.end_block() # end else block

        # ------ cleanup ------
        self.end_parallel_control_flow_block(code) # end parallel control flow block

        # And finally, release our privates and write back any closure
        # variables
        for temp in start_stop_step + (self.chunksize,):
            if temp is not None:
                temp.generate_disposal_code(code)
                temp.free_temps(code)

        code.funcstate.release_temp(fmt_dict['i'])
        code.funcstate.release_temp(fmt_dict['nsteps'])

        self.release_closure_privates(code)

    def generate_loop(self, code, fmt_dict):
        if self.is_nested_prange:
            code.putln("#if 0")
        else:
            code.putln("#ifdef _OPENMP")

        if not self.is_parallel:
            code.put("#pragma omp for")
            self.privatization_insertion_point = code.insertion_point()
            reduction_codepoint = self.parent.privatization_insertion_point
        else:
            code.put("#pragma omp parallel")
            self.privatization_insertion_point = code.insertion_point()
            reduction_codepoint = self.privatization_insertion_point
            code.putln("")
            code.putln("#endif /* _OPENMP */")

            code.begin_block() # pragma omp parallel begin block

            # Initialize the GIL if needed for this thread
            self.begin_parallel_block(code)

            if self.is_nested_prange:
                code.putln("#if 0")
            else:
                code.putln("#ifdef _OPENMP")
            code.put("#pragma omp for")

        for entry, (op, lastprivate) in sorted(self.privates.items()):
            # Don't declare the index variable as a reduction
            if op and op in "+*-&^|" and entry != self.target.entry:
                if entry.type.is_pyobject:
                    error(self.pos, "Python objects cannot be reductions")
                else:
                    #code.put(" reduction(%s:%s)" % (op, entry.cname))
                    # This is the only way reductions + nesting works in gcc4.5
                    reduction_codepoint.put(
                                " reduction(%s:%s)" % (op, entry.cname))
            else:
                if entry == self.target.entry:
                    code.put(" firstprivate(%s)" % entry.cname)
                    code.put(" lastprivate(%s)" % entry.cname)
                    continue

                if not entry.type.is_pyobject:
                    if lastprivate:
                        private = 'lastprivate'
                    else:
                        private = 'private'

                    code.put(" %s(%s)" % (private, entry.cname))

        if self.schedule:
            if self.chunksize:
                chunksize = ", %s" % self.evaluate_before_block(code, self.chunksize)
            else:
                chunksize = ""

            code.put(" schedule(%s%s)" % (self.schedule, chunksize))

        self.put_num_threads(reduction_codepoint)

        code.putln("")
        code.putln("#endif /* _OPENMP */")

        code.put("for (%(i)s = 0; %(i)s < %(nsteps)s; %(i)s++)" % fmt_dict)
        code.begin_block()  # for loop block

        guard_around_body_codepoint = code.insertion_point()

        # Start if guard block around the body. This may be unnecessary, but
        # at least it doesn't spoil indentation
        code.begin_block()

        code.putln("%(target)s = (%(target_type)s)(%(start)s + %(step)s * %(i)s);" % fmt_dict)
        self.initialize_privates_to_nan(code, exclude=self.target.entry)

        if self.is_parallel and not self.is_nested_prange:
            # nested pranges are not omp'ified, temps go to outer loops
            code.funcstate.start_collecting_temps()

        self.body.generate_execution_code(code)
        self.trap_parallel_exit(code, should_flush=True)
        if self.is_parallel and not self.is_nested_prange:
            # nested pranges are not omp'ified, temps go to outer loops
            self.privatize_temps(code)

        if self.breaking_label_used:
            # Put a guard around the loop body in case return, break or
            # exceptions might be used
            guard_around_body_codepoint.putln("if (%s < 2)" % Naming.parallel_why)

        code.end_block()  # end guard around loop body
        code.end_block()  # end for loop block

        if self.is_parallel:
            # Release the GIL and deallocate the thread state
            self.end_parallel_block(code)
            code.end_block()  # pragma omp parallel end block


class CnameDecoratorNode(StatNode):
    """
    This node is for the cname decorator in CythonUtilityCode:

        @cname('the_cname')
        cdef func(...):
            ...

    In case of a cdef class the cname specifies the objstruct_cname.

    node        the node to which the cname decorator is applied
    cname       the cname the node should get
    """

    child_attrs = ['node']

    def analyse_declarations(self, env):
        self.node.analyse_declarations(env)

        node = self.node
        if isinstance(node, CompilerDirectivesNode):
            node = node.body.stats[0]

        self.is_function = isinstance(node, FuncDefNode)
        is_struct_or_enum = isinstance(node, (CStructOrUnionDefNode, CEnumDefNode))
        e = node.entry

        if self.is_function:
            e.cname = self.cname
            e.func_cname = self.cname
            e.used = True
            if e.pyfunc_cname and '.' in e.pyfunc_cname:
                e.pyfunc_cname = self.mangle(e.pyfunc_cname)
        elif is_struct_or_enum:
            e.cname = e.type.cname = self.cname
        else:
            scope = node.scope

            e.cname = self.cname
            e.type.objstruct_cname = self.cname + '_obj'
            e.type.typeobj_cname = Naming.typeobj_prefix + self.cname
            e.type.typeptr_cname = self.cname + '_type'
            e.type.scope.namespace_cname = e.type.typeptr_cname

            e.as_variable.cname = e.type.typeptr_cname

            scope.scope_prefix = self.cname + "_"

            for name, entry in scope.entries.items():
                if entry.func_cname:
                    entry.func_cname = self.mangle(entry.cname)
                if entry.pyfunc_cname:
                    entry.pyfunc_cname = self.mangle(entry.pyfunc_cname)

    def mangle(self, cname):
        if '.' in cname:
            # remove __pyx_base from func_cname
            cname = cname.split('.')[-1]
        return '%s_%s' % (self.cname, cname)

    def analyse_expressions(self, env):
        self.node = self.node.analyse_expressions(env)
        return self

    def generate_function_definitions(self, env, code):
        "Ensure a prototype for every @cname method in the right place"
        if self.is_function and env.is_c_class_scope:
            # method in cdef class, generate a prototype in the header
            h_code = code.globalstate['utility_code_proto']

            if isinstance(self.node, DefNode):
                self.node.generate_function_header(
                    h_code, with_pymethdef=False, proto_only=True)
            else:
                from . import ModuleNode
                entry = self.node.entry
                cname = entry.cname
                entry.cname = entry.func_cname

                ModuleNode.generate_cfunction_declaration(
                    entry,
                    env.global_scope(),
                    h_code,
                    definition=True)

                entry.cname = cname

        self.node.generate_function_definitions(env, code)

    def generate_execution_code(self, code):
        self.node.generate_execution_code(code)


#------------------------------------------------------------------------------------
#
#  Runtime support code
#
#------------------------------------------------------------------------------------

if Options.gcc_branch_hints:
    branch_prediction_macros = """
/* Test for GCC > 2.95 */
#if defined(__GNUC__) \
    && (__GNUC__ > 2 || (__GNUC__ == 2 && (__GNUC_MINOR__ > 95)))
  #define likely(x)   __builtin_expect(!!(x), 1)
  #define unlikely(x) __builtin_expect(!!(x), 0)
#else /* !__GNUC__ or GCC < 2.95 */
  #define likely(x)   (x)
  #define unlikely(x) (x)
#endif /* __GNUC__ */
"""
else:
    branch_prediction_macros = """
#define likely(x)   (x)
#define unlikely(x) (x)
"""

#------------------------------------------------------------------------------------

printing_utility_code = UtilityCode.load_cached("Print", "Printing.c")
printing_one_utility_code = UtilityCode.load_cached("PrintOne", "Printing.c")

#------------------------------------------------------------------------------------

# Exception raising code
#
# Exceptions are raised by __Pyx_Raise() and stored as plain
# type/value/tb in PyThreadState->curexc_*.  When being caught by an
# 'except' statement, curexc_* is moved over to exc_* by
# __Pyx_GetException()

restore_exception_utility_code = UtilityCode.load_cached("PyErrFetchRestore", "Exceptions.c")
raise_utility_code = UtilityCode.load_cached("RaiseException", "Exceptions.c")
get_exception_utility_code = UtilityCode.load_cached("GetException", "Exceptions.c")
swap_exception_utility_code = UtilityCode.load_cached("SwapException", "Exceptions.c")
reset_exception_utility_code = UtilityCode.load_cached("SaveResetException", "Exceptions.c")
traceback_utility_code = UtilityCode.load_cached("AddTraceback", "Exceptions.c")

#------------------------------------------------------------------------------------

get_exception_tuple_utility_code = UtilityCode(
    proto="""
static PyObject *__Pyx_GetExceptionTuple(PyThreadState *__pyx_tstate); /*proto*/
""",
    # I doubt that calling __Pyx_GetException() here is correct as it moves
    # the exception from tstate->curexc_* to tstate->exc_*, which prevents
    # exception handlers later on from receiving it.
    # NOTE: "__pyx_tstate" may be used by __Pyx_GetException() macro
    impl = """
static PyObject *__Pyx_GetExceptionTuple(CYTHON_UNUSED PyThreadState *__pyx_tstate) {
    PyObject *type = NULL, *value = NULL, *tb = NULL;
    if (__Pyx_GetException(&type, &value, &tb) == 0) {
        PyObject* exc_info = PyTuple_New(3);
        if (exc_info) {
            Py_INCREF(type);
            Py_INCREF(value);
            Py_INCREF(tb);
            PyTuple_SET_ITEM(exc_info, 0, type);
            PyTuple_SET_ITEM(exc_info, 1, value);
            PyTuple_SET_ITEM(exc_info, 2, tb);
            return exc_info;
        }
    }
    return NULL;
}
""",
    requires=[get_exception_utility_code])

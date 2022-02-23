#
#   Cython/Python language types
#

from __future__ import absolute_import

import copy
import hashlib
import re

try:
    reduce
except NameError:
    from functools import reduce

from Cython.Utils import cached_function
from .Code import UtilityCode, LazyUtilityCode, TempitaUtilityCode
from . import StringEncoding
from . import Naming

from .Errors import error, warning


class BaseType(object):
    #
    #  Base class for all Cython types including pseudo-types.

    # List of attribute names of any subtypes
    subtypes = []
    _empty_declaration = None
    _specialization_name = None
    default_format_spec = None

    def can_coerce_to_pyobject(self, env):
        return False

    def can_coerce_from_pyobject(self, env):
        return False

    def can_coerce_to_pystring(self, env, format_spec=None):
        return False

    def convert_to_pystring(self, cvalue, code, format_spec=None):
        raise NotImplementedError("C types that support string formatting must override this method")

    def cast_code(self, expr_code):
        return "((%s)%s)" % (self.empty_declaration_code(), expr_code)

    def empty_declaration_code(self):
        if self._empty_declaration is None:
            self._empty_declaration = self.declaration_code('')
        return self._empty_declaration

    def specialization_name(self):
        if self._specialization_name is None:
            # This is not entirely robust.
            common_subs = (self.empty_declaration_code()
                           .replace("unsigned ", "unsigned_")
                           .replace("long long", "long_long")
                           .replace(" ", "__"))
            self._specialization_name = re.sub(
                '[^a-zA-Z0-9_]', lambda x: '_%x_' % ord(x.group(0)), common_subs)
        return self._specialization_name

    def base_declaration_code(self, base_code, entity_code):
        if entity_code:
            return "%s %s" % (base_code, entity_code)
        else:
            return base_code

    def __deepcopy__(self, memo):
        """
        Types never need to be copied, if we do copy, Unfortunate Things
        Will Happen!
        """
        return self

    def get_fused_types(self, result=None, seen=None, subtypes=None):
        subtypes = subtypes or self.subtypes
        if not subtypes:
            return None

        if result is None:
            result = []
            seen = set()

        for attr in subtypes:
            list_or_subtype = getattr(self, attr)
            if list_or_subtype:
                if isinstance(list_or_subtype, BaseType):
                    list_or_subtype.get_fused_types(result, seen)
                else:
                    for subtype in list_or_subtype:
                        subtype.get_fused_types(result, seen)

        return result

    def specialize_fused(self, env):
        if env.fused_to_specific:
            return self.specialize(env.fused_to_specific)

        return self

    @property
    def is_fused(self):
        """
        Whether this type or any of its subtypes is a fused type
        """
        # Add this indirection for the is_fused property to allow overriding
        # get_fused_types in subclasses.
        return self.get_fused_types()

    def deduce_template_params(self, actual):
        """
        Deduce any template params in this (argument) type given the actual
        argument type.

        http://en.cppreference.com/w/cpp/language/function_template#Template_argument_deduction
        """
        return {}

    def __lt__(self, other):
        """
        For sorting. The sorting order should correspond to the preference of
        conversion from Python types.

        Override to provide something sensible. This is only implemented so that
        python 3 doesn't trip
        """
        return id(type(self)) < id(type(other))

    def py_type_name(self):
        """
        Return the name of the Python type that can coerce to this type.
        """

    def typeof_name(self):
        """
        Return the string with which fused python functions can be indexed.
        """
        if self.is_builtin_type or self.py_type_name() == 'object':
            index_name = self.py_type_name()
        else:
            index_name = str(self)

        return index_name

    def check_for_null_code(self, cname):
        """
        Return the code for a NULL-check in case an UnboundLocalError should
        be raised if an entry of this type is referenced before assignment.
        Returns None if no check should be performed.
        """
        return None

    def invalid_value(self):
        """
        Returns the most invalid value an object of this type can assume as a
        C expression string. Returns None if no such value exists.
        """


class PyrexType(BaseType):
    #
    #  Base class for all Cython types
    #
    #  is_pyobject           boolean     Is a Python object type
    #  is_extension_type     boolean     Is a Python extension type
    #  is_final_type         boolean     Is a final extension type
    #  is_numeric            boolean     Is a C numeric type
    #  is_int                boolean     Is a C integer type
    #  is_float              boolean     Is a C floating point type
    #  is_complex            boolean     Is a C complex type
    #  is_void               boolean     Is the C void type
    #  is_array              boolean     Is a C array type
    #  is_ptr                boolean     Is a C pointer type
    #  is_null_ptr           boolean     Is the type of NULL
    #  is_reference          boolean     Is a C reference type
    #  is_const              boolean     Is a C const type.
    #  is_cfunction          boolean     Is a C function type
    #  is_struct_or_union    boolean     Is a C struct or union type
    #  is_struct             boolean     Is a C struct type
    #  is_enum               boolean     Is a C enum type
    #  is_typedef            boolean     Is a typedef type
    #  is_string             boolean     Is a C char * type
    #  is_pyunicode_ptr      boolean     Is a C PyUNICODE * type
    #  is_cpp_string         boolean     Is a C++ std::string type
    #  is_unicode_char       boolean     Is either Py_UCS4 or Py_UNICODE
    #  is_returncode         boolean     Is used only to signal exceptions
    #  is_error              boolean     Is the dummy error type
    #  is_buffer             boolean     Is buffer access type
    #  is_pythran_expr       boolean     Is Pythran expr
    #  is_numpy_buffer       boolean     Is Numpy array buffer
    #  has_attributes        boolean     Has C dot-selectable attributes
    #  default_value         string      Initial value that can be assigned before first user assignment.
    #  declaration_value     string      The value statically assigned on declaration (if any).
    #  entry                 Entry       The Entry for this type
    #
    #  declaration_code(entity_code,
    #      for_display = 0, dll_linkage = None, pyrex = 0)
    #    Returns a code fragment for the declaration of an entity
    #    of this type, given a code fragment for the entity.
    #    * If for_display, this is for reading by a human in an error
    #      message; otherwise it must be valid C code.
    #    * If dll_linkage is not None, it must be 'DL_EXPORT' or
    #      'DL_IMPORT', and will be added to the base type part of
    #      the declaration.
    #    * If pyrex = 1, this is for use in a 'cdef extern'
    #      statement of a Cython include file.
    #
    #  assignable_from(src_type)
    #    Tests whether a variable of this type can be
    #    assigned a value of type src_type.
    #
    #  same_as(other_type)
    #    Tests whether this type represents the same type
    #    as other_type.
    #
    #  as_argument_type():
    #    Coerces array and C function types into pointer type for use as
    #    a formal argument type.
    #

    is_pyobject = 0
    is_unspecified = 0
    is_extension_type = 0
    is_final_type = 0
    is_builtin_type = 0
    is_numeric = 0
    is_int = 0
    is_float = 0
    is_complex = 0
    is_void = 0
    is_array = 0
    is_ptr = 0
    is_null_ptr = 0
    is_reference = 0
    is_const = 0
    is_cfunction = 0
    is_struct_or_union = 0
    is_cpp_class = 0
    is_cpp_string = 0
    is_struct = 0
    is_enum = 0
    is_typedef = 0
    is_string = 0
    is_pyunicode_ptr = 0
    is_unicode_char = 0
    is_returncode = 0
    is_error = 0
    is_buffer = 0
    is_ctuple = 0
    is_memoryviewslice = 0
    is_pythran_expr = 0
    is_numpy_buffer = 0
    has_attributes = 0
    default_value = ""
    declaration_value = ""

    def resolve(self):
        # If a typedef, returns the base type.
        return self

    def specialize(self, values):
        # TODO(danilo): Override wherever it makes sense.
        return self

    def literal_code(self, value):
        # Returns a C code fragment representing a literal
        # value of this type.
        return str(value)

    def __str__(self):
        return self.declaration_code("", for_display = 1).strip()

    def same_as(self, other_type, **kwds):
        return self.same_as_resolved_type(other_type.resolve(), **kwds)

    def same_as_resolved_type(self, other_type):
        return self == other_type or other_type is error_type

    def subtype_of(self, other_type):
        return self.subtype_of_resolved_type(other_type.resolve())

    def subtype_of_resolved_type(self, other_type):
        return self.same_as(other_type)

    def assignable_from(self, src_type):
        return self.assignable_from_resolved_type(src_type.resolve())

    def assignable_from_resolved_type(self, src_type):
        return self.same_as(src_type)

    def as_argument_type(self):
        return self

    def is_complete(self):
        # A type is incomplete if it is an unsized array,
        # a struct whose attributes are not defined, etc.
        return 1

    def is_simple_buffer_dtype(self):
        return (self.is_int or self.is_float or self.is_complex or self.is_pyobject or
                self.is_extension_type or self.is_ptr)

    def struct_nesting_depth(self):
        # Returns the number levels of nested structs. This is
        # used for constructing a stack for walking the run-time
        # type information of the struct.
        return 1

    def global_init_code(self, entry, code):
        # abstract
        pass

    def needs_nonecheck(self):
        return 0

    def _assign_from_py_code(self, source_code, result_code, error_pos, code,
                             from_py_function=None, error_condition=None, extra_args=None):
        args = ', ' + ', '.join('%s' % arg for arg in extra_args) if extra_args else ''
        convert_call = "%s(%s%s)" % (
            from_py_function or self.from_py_function,
            source_code,
            args,
        )
        if self.is_enum:
            convert_call = typecast(self, c_long_type, convert_call)
        return '%s = %s; %s' % (
            result_code,
            convert_call,
            code.error_goto_if(error_condition or self.error_condition(result_code), error_pos))


def public_decl(base_code, dll_linkage):
    if dll_linkage:
        return "%s(%s)" % (dll_linkage, base_code.replace(',', ' __PYX_COMMA '))
    else:
        return base_code

def create_typedef_type(name, base_type, cname, is_external=0, namespace=None):
    is_fused = base_type.is_fused
    if base_type.is_complex or is_fused:
        if is_external:
            if is_fused:
                msg = "Fused"
            else:
                msg = "Complex"

            raise ValueError("%s external typedefs not supported" % msg)

        return base_type
    else:
        return CTypedefType(name, base_type, cname, is_external, namespace)


class CTypedefType(BaseType):
    #
    #  Pseudo-type defined with a ctypedef statement in a
    #  'cdef extern from' block.
    #  Delegates most attribute lookups to the base type.
    #  (Anything not defined here or in the BaseType is delegated.)
    #
    #  qualified_name      string
    #  typedef_name        string
    #  typedef_cname       string
    #  typedef_base_type   PyrexType
    #  typedef_is_external bool

    is_typedef = 1
    typedef_is_external = 0

    to_py_utility_code = None
    from_py_utility_code = None

    subtypes = ['typedef_base_type']

    def __init__(self, name, base_type, cname, is_external=0, namespace=None):
        assert not base_type.is_complex
        self.typedef_name = name
        self.typedef_cname = cname
        self.typedef_base_type = base_type
        self.typedef_is_external = is_external
        self.typedef_namespace = namespace

    def invalid_value(self):
        return self.typedef_base_type.invalid_value()

    def resolve(self):
        return self.typedef_base_type.resolve()

    def declaration_code(self, entity_code,
            for_display = 0, dll_linkage = None, pyrex = 0):
        if pyrex or for_display:
            base_code = self.typedef_name
        else:
            base_code = public_decl(self.typedef_cname, dll_linkage)
        if self.typedef_namespace is not None and not pyrex:
            base_code = "%s::%s" % (self.typedef_namespace.empty_declaration_code(), base_code)
        return self.base_declaration_code(base_code, entity_code)

    def as_argument_type(self):
        return self

    def cast_code(self, expr_code):
        # If self is really an array (rather than pointer), we can't cast.
        # For example, the gmp mpz_t.
        if self.typedef_base_type.is_array:
            base_type = self.typedef_base_type.base_type
            return CPtrType(base_type).cast_code(expr_code)
        else:
            return BaseType.cast_code(self, expr_code)

    def specialize(self, values):
        base_type = self.typedef_base_type.specialize(values)
        namespace = self.typedef_namespace.specialize(values) if self.typedef_namespace else None
        if base_type is self.typedef_base_type and namespace is self.typedef_namespace:
            return self
        else:
            return create_typedef_type(self.typedef_name, base_type, self.typedef_cname,
                                0, namespace)

    def __repr__(self):
        return "<CTypedefType %s>" % self.typedef_cname

    def __str__(self):
        return self.typedef_name

    def _create_utility_code(self, template_utility_code,
                             template_function_name):
        type_name = type_identifier(self.typedef_cname)
        utility_code = template_utility_code.specialize(
            type     = self.typedef_cname,
            TypeName = type_name)
        function_name = template_function_name % type_name
        return utility_code, function_name

    def create_to_py_utility_code(self, env):
        if self.typedef_is_external:
            if not self.to_py_utility_code:
                base_type = self.typedef_base_type
                if type(base_type) is CIntType:
                    self.to_py_function = "__Pyx_PyInt_From_" + self.specialization_name()
                    env.use_utility_code(TempitaUtilityCode.load_cached(
                        "CIntToPy", "TypeConversion.c",
                        context={"TYPE": self.empty_declaration_code(),
                                 "TO_PY_FUNCTION": self.to_py_function}))
                    return True
                elif base_type.is_float:
                    pass # XXX implement!
                elif base_type.is_complex:
                    pass # XXX implement!
                    pass
                elif base_type.is_cpp_string:
                    cname = "__pyx_convert_PyObject_string_to_py_%s" % type_identifier(self)
                    context = {
                        'cname': cname,
                        'type': self.typedef_cname,
                    }
                    from .UtilityCode import CythonUtilityCode
                    env.use_utility_code(CythonUtilityCode.load(
                        "string.to_py", "CppConvert.pyx", context=context))
                    self.to_py_function = cname
                    return True
            if self.to_py_utility_code:
                env.use_utility_code(self.to_py_utility_code)
                return True
        # delegation
        return self.typedef_base_type.create_to_py_utility_code(env)

    def create_from_py_utility_code(self, env):
        if self.typedef_is_external:
            if not self.from_py_utility_code:
                base_type = self.typedef_base_type
                if type(base_type) is CIntType:
                    self.from_py_function = "__Pyx_PyInt_As_" + self.specialization_name()
                    env.use_utility_code(TempitaUtilityCode.load_cached(
                        "CIntFromPy", "TypeConversion.c",
                        context={"TYPE": self.empty_declaration_code(),
                                 "FROM_PY_FUNCTION": self.from_py_function}))
                    return True
                elif base_type.is_float:
                    pass # XXX implement!
                elif base_type.is_complex:
                    pass # XXX implement!
                elif base_type.is_cpp_string:
                    cname = '__pyx_convert_string_from_py_%s' % type_identifier(self)
                    context = {
                        'cname': cname,
                        'type': self.typedef_cname,
                    }
                    from .UtilityCode import CythonUtilityCode
                    env.use_utility_code(CythonUtilityCode.load(
                        "string.from_py", "CppConvert.pyx", context=context))
                    self.from_py_function = cname
                    return True
            if self.from_py_utility_code:
                env.use_utility_code(self.from_py_utility_code)
                return True
        # delegation
        return self.typedef_base_type.create_from_py_utility_code(env)

    def to_py_call_code(self, source_code, result_code, result_type, to_py_function=None):
        if to_py_function is None:
            to_py_function = self.to_py_function
        return self.typedef_base_type.to_py_call_code(
            source_code, result_code, result_type, to_py_function)

    def from_py_call_code(self, source_code, result_code, error_pos, code,
                          from_py_function=None, error_condition=None):
        return self.typedef_base_type.from_py_call_code(
            source_code, result_code, error_pos, code,
            from_py_function or self.from_py_function,
            error_condition or self.error_condition(result_code)
        )

    def overflow_check_binop(self, binop, env, const_rhs=False):
        env.use_utility_code(UtilityCode.load("Common", "Overflow.c"))
        type = self.empty_declaration_code()
        name = self.specialization_name()
        if binop == "lshift":
            env.use_utility_code(TempitaUtilityCode.load_cached(
                "LeftShift", "Overflow.c",
                context={'TYPE': type, 'NAME': name, 'SIGNED': self.signed}))
        else:
            if const_rhs:
                binop += "_const"
            _load_overflow_base(env)
            env.use_utility_code(TempitaUtilityCode.load_cached(
                "SizeCheck", "Overflow.c",
                context={'TYPE': type, 'NAME': name}))
            env.use_utility_code(TempitaUtilityCode.load_cached(
                "Binop", "Overflow.c",
                context={'TYPE': type, 'NAME': name, 'BINOP': binop}))
        return "__Pyx_%s_%s_checking_overflow" % (binop, name)

    def error_condition(self, result_code):
        if self.typedef_is_external:
            if self.exception_value:
                condition = "(%s == %s)" % (
                    result_code, self.cast_code(self.exception_value))
                if self.exception_check:
                    condition += " && PyErr_Occurred()"
                return condition
        # delegation
        return self.typedef_base_type.error_condition(result_code)

    def __getattr__(self, name):
        return getattr(self.typedef_base_type, name)

    def py_type_name(self):
        return self.typedef_base_type.py_type_name()

    def can_coerce_to_pyobject(self, env):
        return self.typedef_base_type.can_coerce_to_pyobject(env)

    def can_coerce_from_pyobject(self, env):
        return self.typedef_base_type.can_coerce_from_pyobject(env)


class MemoryViewSliceType(PyrexType):

    is_memoryviewslice = 1

    has_attributes = 1
    scope = None

    # These are special cased in Defnode
    from_py_function = None
    to_py_function = None

    exception_value = None
    exception_check = True

    subtypes = ['dtype']

    def __init__(self, base_dtype, axes):
        """
        MemoryViewSliceType(base, axes)

        Base is the C base type; axes is a list of (access, packing) strings,
        where access is one of 'full', 'direct' or 'ptr' and packing is one of
        'contig', 'strided' or 'follow'.  There is one (access, packing) tuple
        for each dimension.

        the access specifiers determine whether the array data contains
        pointers that need to be dereferenced along that axis when
        retrieving/setting:

        'direct' -- No pointers stored in this dimension.
        'ptr' -- Pointer stored in this dimension.
        'full' -- Check along this dimension, don't assume either.

        the packing specifiers specify how the array elements are layed-out
        in memory.

        'contig' -- The data is contiguous in memory along this dimension.
                At most one dimension may be specified as 'contig'.
        'strided' -- The data isn't contiguous along this dimension.
        'follow' -- Used for C/Fortran contiguous arrays, a 'follow' dimension
            has its stride automatically computed from extents of the other
            dimensions to ensure C or Fortran memory layout.

        C-contiguous memory has 'direct' as the access spec, 'contig' as the
        *last* axis' packing spec and 'follow' for all other packing specs.

        Fortran-contiguous memory has 'direct' as the access spec, 'contig' as
        the *first* axis' packing spec and 'follow' for all other packing
        specs.
        """
        from . import Buffer, MemoryView

        self.dtype = base_dtype
        self.axes = axes
        self.ndim = len(axes)
        self.flags = MemoryView.get_buf_flags(self.axes)

        self.is_c_contig, self.is_f_contig = MemoryView.is_cf_contig(self.axes)
        assert not (self.is_c_contig and self.is_f_contig)

        self.mode = MemoryView.get_mode(axes)
        self.writable_needed = False

        if not self.dtype.is_fused:
            self.dtype_name = Buffer.mangle_dtype_name(self.dtype)

    def __hash__(self):
        return hash(self.__class__) ^ hash(self.dtype) ^ hash(tuple(self.axes))

    def __eq__(self, other):
        if isinstance(other, BaseType):
            return self.same_as_resolved_type(other)
        else:
            return False

    def same_as_resolved_type(self, other_type):
        return ((other_type.is_memoryviewslice and
            #self.writable_needed == other_type.writable_needed and  # FIXME: should be only uni-directional
            self.dtype.same_as(other_type.dtype) and
            self.axes == other_type.axes) or
            other_type is error_type)

    def needs_nonecheck(self):
        return True

    def is_complete(self):
        # incomplete since the underlying struct doesn't have a cython.memoryview object.
        return 0

    def declaration_code(self, entity_code,
            for_display = 0, dll_linkage = None, pyrex = 0):
        # XXX: we put these guards in for now...
        assert not pyrex
        assert not dll_linkage
        from . import MemoryView
        base_code = str(self) if for_display else MemoryView.memviewslice_cname
        return self.base_declaration_code(
                base_code,
                entity_code)

    def attributes_known(self):
        if self.scope is None:
            from . import Symtab

            self.scope = scope = Symtab.CClassScope(
                    'mvs_class_'+self.specialization_suffix(),
                    None,
                    visibility='extern')

            scope.parent_type = self
            scope.directives = {}

            scope.declare_var('_data', c_char_ptr_type, None,
                              cname='data', is_cdef=1)

        return True

    def declare_attribute(self, attribute, env, pos):
        from . import MemoryView, Options

        scope = self.scope

        if attribute == 'shape':
            scope.declare_var('shape',
                    c_array_type(c_py_ssize_t_type,
                                 Options.buffer_max_dims),
                    pos,
                    cname='shape',
                    is_cdef=1)

        elif attribute == 'strides':
            scope.declare_var('strides',
                    c_array_type(c_py_ssize_t_type,
                                 Options.buffer_max_dims),
                    pos,
                    cname='strides',
                    is_cdef=1)

        elif attribute == 'suboffsets':
            scope.declare_var('suboffsets',
                    c_array_type(c_py_ssize_t_type,
                                 Options.buffer_max_dims),
                    pos,
                    cname='suboffsets',
                    is_cdef=1)

        elif attribute in ("copy", "copy_fortran"):
            ndim = len(self.axes)

            follow_dim = [('direct', 'follow')]
            contig_dim = [('direct', 'contig')]
            to_axes_c = follow_dim * (ndim - 1) + contig_dim
            to_axes_f = contig_dim + follow_dim * (ndim -1)

            dtype = self.dtype
            if dtype.is_const:
                dtype = dtype.const_base_type

            to_memview_c = MemoryViewSliceType(dtype, to_axes_c)
            to_memview_f = MemoryViewSliceType(dtype, to_axes_f)

            for to_memview, cython_name in [(to_memview_c, "copy"),
                                            (to_memview_f, "copy_fortran")]:
                copy_func_type = CFuncType(
                    to_memview,
                    [CFuncTypeArg("memviewslice", self, None)])
                copy_cname = MemoryView.copy_c_or_fortran_cname(to_memview)

                entry = scope.declare_cfunction(
                    cython_name,
                    copy_func_type, pos=pos, defining=1,
                    cname=copy_cname)

                utility = MemoryView.get_copy_new_utility(pos, self, to_memview)
                env.use_utility_code(utility)

            MemoryView.use_cython_array_utility_code(env)

        elif attribute in ("is_c_contig", "is_f_contig"):
            # is_c_contig and is_f_contig functions
            for (c_or_f, cython_name) in (('C', 'is_c_contig'), ('F', 'is_f_contig')):

                is_contig_name = MemoryView.get_is_contig_func_name(c_or_f, self.ndim)

                cfunctype = CFuncType(
                        return_type=c_bint_type,
                        args=[CFuncTypeArg("memviewslice", self, None)],
                        exception_value="-1",
                )

                entry = scope.declare_cfunction(cython_name,
                            cfunctype,
                            pos=pos,
                            defining=1,
                            cname=is_contig_name)

                entry.utility_code_definition = MemoryView.get_is_contig_utility(c_or_f, self.ndim)

        return True

    def get_entry(self, node, cname=None, type=None):
        from . import MemoryView, Symtab

        if cname is None:
            assert node.is_simple() or node.is_temp or node.is_elemental
            cname = node.result()

        if type is None:
            type = node.type

        entry = Symtab.Entry(cname, cname, type, node.pos)
        return MemoryView.MemoryViewSliceBufferEntry(entry)

    def conforms_to(self, dst, broadcast=False, copying=False):
        """
        Returns True if src conforms to dst, False otherwise.

        If conformable, the types are the same, the ndims are equal, and each axis spec is conformable.

        Any packing/access spec is conformable to itself.

        'direct' and 'ptr' are conformable to 'full'.
        'contig' and 'follow' are conformable to 'strided'.
        Any other combo is not conformable.
        """
        from . import MemoryView

        src = self

        #if not copying and self.writable_needed and not dst.writable_needed:
        #    return False

        src_dtype, dst_dtype = src.dtype, dst.dtype
        if dst_dtype.is_const:
            # Requesting read-only views is always ok => consider only the non-const base type.
            dst_dtype = dst_dtype.const_base_type
            if src_dtype.is_const:
                # When assigning between read-only views, compare only the non-const base types.
                src_dtype = src_dtype.const_base_type
        elif copying and src_dtype.is_const:
            # Copying by value => ignore const on source.
            src_dtype = src_dtype.const_base_type

        if src_dtype != dst_dtype:
            return False

        if src.ndim != dst.ndim:
            if broadcast:
                src, dst = MemoryView.broadcast_types(src, dst)
            else:
                return False

        for src_spec, dst_spec in zip(src.axes, dst.axes):
            src_access, src_packing = src_spec
            dst_access, dst_packing = dst_spec
            if src_access != dst_access and dst_access != 'full':
                return False
            if src_packing != dst_packing and dst_packing != 'strided' and not copying:
                return False

        return True

    def valid_dtype(self, dtype, i=0):
        """
        Return whether type dtype can be used as the base type of a
        memoryview slice.

        We support structs, numeric types and objects
        """
        if dtype.is_complex and dtype.real_type.is_int:
            return False

        if dtype.is_struct and dtype.kind == 'struct':
            for member in dtype.scope.var_entries:
                if not self.valid_dtype(member.type):
                    return False

            return True

        return (
            dtype.is_error or
            # Pointers are not valid (yet)
            # (dtype.is_ptr and valid_memslice_dtype(dtype.base_type)) or
            (dtype.is_array and i < 8 and self.valid_dtype(dtype.base_type, i + 1)) or
            dtype.is_numeric or
            dtype.is_pyobject or
            dtype.is_fused or  # accept this as it will be replaced by specializations later
            (dtype.is_typedef and self.valid_dtype(dtype.typedef_base_type))
        )

    def validate_memslice_dtype(self, pos):
        if not self.valid_dtype(self.dtype):
            error(pos, "Invalid base type for memoryview slice: %s" % self.dtype)

    def assert_direct_dims(self, pos):
        for access, packing in self.axes:
            if access != 'direct':
                error(pos, "All dimensions must be direct")
                return False
        return True

    def transpose(self, pos):
        if not self.assert_direct_dims(pos):
            return error_type
        return MemoryViewSliceType(self.dtype, self.axes[::-1])

    def specialization_name(self):
        return '%s_%s' % (
            super(MemoryViewSliceType,self).specialization_name(),
            self.specialization_suffix())

    def specialization_suffix(self):
        return "%s_%s" % (self.axes_to_name(), self.dtype_name)

    def can_coerce_to_pyobject(self, env):
        return True

    def can_coerce_from_pyobject(self, env):
        return True

    def check_for_null_code(self, cname):
        return cname + '.memview'

    def create_from_py_utility_code(self, env):
        from . import MemoryView, Buffer

        # We don't have 'code', so use a LazyUtilityCode with a callback.
        def lazy_utility_callback(code):
            context['dtype_typeinfo'] = Buffer.get_type_information_cname(code, self.dtype)
            return TempitaUtilityCode.load(
                "ObjectToMemviewSlice", "MemoryView_C.c", context=context)

        env.use_utility_code(MemoryView.memviewslice_init_code)
        env.use_utility_code(LazyUtilityCode(lazy_utility_callback))

        if self.is_c_contig:
            c_or_f_flag = "__Pyx_IS_C_CONTIG"
        elif self.is_f_contig:
            c_or_f_flag = "__Pyx_IS_F_CONTIG"
        else:
            c_or_f_flag = "0"

        suffix = self.specialization_suffix()
        funcname = "__Pyx_PyObject_to_MemoryviewSlice_" + suffix

        context = dict(
            MemoryView.context,
            buf_flag = self.flags,
            ndim = self.ndim,
            axes_specs = ', '.join(self.axes_to_code()),
            dtype_typedecl = self.dtype.empty_declaration_code(),
            struct_nesting_depth = self.dtype.struct_nesting_depth(),
            c_or_f_flag = c_or_f_flag,
            funcname = funcname,
        )

        self.from_py_function = funcname
        return True

    def from_py_call_code(self, source_code, result_code, error_pos, code,
                          from_py_function=None, error_condition=None):
        # NOTE: auto-detection of readonly buffers is disabled:
        # writable = self.writable_needed or not self.dtype.is_const
        writable = not self.dtype.is_const
        return self._assign_from_py_code(
            source_code, result_code, error_pos, code, from_py_function, error_condition,
            extra_args=['PyBUF_WRITABLE' if writable else '0'])

    def create_to_py_utility_code(self, env):
        self._dtype_to_py_func, self._dtype_from_py_func = self.dtype_object_conversion_funcs(env)
        return True

    def to_py_call_code(self, source_code, result_code, result_type, to_py_function=None):
        assert self._dtype_to_py_func
        assert self._dtype_from_py_func

        to_py_func = "(PyObject *(*)(char *)) " + self._dtype_to_py_func
        from_py_func = "(int (*)(char *, PyObject *)) " + self._dtype_from_py_func

        tup = (result_code, source_code, self.ndim, to_py_func, from_py_func, self.dtype.is_pyobject)
        return "%s = __pyx_memoryview_fromslice(%s, %s, %s, %s, %d);" % tup

    def dtype_object_conversion_funcs(self, env):
        get_function = "__pyx_memview_get_%s" % self.dtype_name
        set_function = "__pyx_memview_set_%s" % self.dtype_name

        context = dict(
            get_function = get_function,
            set_function = set_function,
        )

        if self.dtype.is_pyobject:
            utility_name = "MemviewObjectToObject"
        else:
            self.dtype.create_to_py_utility_code(env)
            to_py_function = self.dtype.to_py_function

            from_py_function = None
            if not self.dtype.is_const:
                self.dtype.create_from_py_utility_code(env)
                from_py_function = self.dtype.from_py_function

            if not (to_py_function or from_py_function):
                return "NULL", "NULL"
            if not to_py_function:
                get_function = "NULL"
            if not from_py_function:
                set_function = "NULL"

            utility_name = "MemviewDtypeToObject"
            error_condition = (self.dtype.error_condition('value') or
                               'PyErr_Occurred()')
            context.update(
                to_py_function=to_py_function,
                from_py_function=from_py_function,
                dtype=self.dtype.empty_declaration_code(),
                error_condition=error_condition,
            )

        utility = TempitaUtilityCode.load_cached(
            utility_name, "MemoryView_C.c", context=context)
        env.use_utility_code(utility)
        return get_function, set_function

    def axes_to_code(self):
        """Return a list of code constants for each axis"""
        from . import MemoryView
        d = MemoryView._spec_to_const
        return ["(%s | %s)" % (d[a], d[p]) for a, p in self.axes]

    def axes_to_name(self):
        """Return an abbreviated name for our axes"""
        from . import MemoryView
        d = MemoryView._spec_to_abbrev
        return "".join(["%s%s" % (d[a], d[p]) for a, p in self.axes])

    def error_condition(self, result_code):
        return "!%s.memview" % result_code

    def __str__(self):
        from . import MemoryView

        axes_code_list = []
        for idx, (access, packing) in enumerate(self.axes):
            flag = MemoryView.get_memoryview_flag(access, packing)
            if flag == "strided":
                axes_code_list.append(":")
            else:
                if flag == 'contiguous':
                    have_follow = [p for a, p in self.axes[idx - 1:idx + 2]
                                         if p == 'follow']
                    if have_follow or self.ndim == 1:
                        flag = '1'

                axes_code_list.append("::" + flag)

        if self.dtype.is_pyobject:
            dtype_name = self.dtype.name
        else:
            dtype_name = self.dtype

        return "%s[%s]" % (dtype_name, ", ".join(axes_code_list))

    def specialize(self, values):
        """This does not validate the base type!!"""
        dtype = self.dtype.specialize(values)
        if dtype is not self.dtype:
            return MemoryViewSliceType(dtype, self.axes)

        return self

    def cast_code(self, expr_code):
        return expr_code


class BufferType(BaseType):
    #
    #  Delegates most attribute lookups to the base type.
    #  (Anything not defined here or in the BaseType is delegated.)
    #
    # dtype            PyrexType
    # ndim             int
    # mode             str
    # negative_indices bool
    # cast             bool
    # is_buffer        bool
    # writable         bool

    is_buffer = 1
    writable = True

    subtypes = ['dtype']

    def __init__(self, base, dtype, ndim, mode, negative_indices, cast):
        self.base = base
        self.dtype = dtype
        self.ndim = ndim
        self.buffer_ptr_type = CPtrType(dtype)
        self.mode = mode
        self.negative_indices = negative_indices
        self.cast = cast
        self.is_numpy_buffer = self.base.name == "ndarray"

    def can_coerce_to_pyobject(self,env):
        return True

    def can_coerce_from_pyobject(self,env):
        return True

    def as_argument_type(self):
        return self

    def specialize(self, values):
        dtype = self.dtype.specialize(values)
        if dtype is not self.dtype:
            return BufferType(self.base, dtype, self.ndim, self.mode,
                              self.negative_indices, self.cast)
        return self

    def get_entry(self, node):
        from . import Buffer
        assert node.is_name
        return Buffer.BufferEntry(node.entry)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def __repr__(self):
        return "<BufferType %r>" % self.base

    def __str__(self):
        # avoid ', ', as fused functions split the signature string on ', '
        cast_str = ''
        if self.cast:
            cast_str = ',cast=True'

        return "%s[%s,ndim=%d%s]" % (self.base, self.dtype, self.ndim,
                                      cast_str)

    def assignable_from(self, other_type):
        if other_type.is_buffer:
            return (self.same_as(other_type, compare_base=False) and
                    self.base.assignable_from(other_type.base))

        return self.base.assignable_from(other_type)

    def same_as(self, other_type, compare_base=True):
        if not other_type.is_buffer:
            return other_type.same_as(self.base)

        return (self.dtype.same_as(other_type.dtype) and
                self.ndim == other_type.ndim and
                self.mode == other_type.mode and
                self.cast == other_type.cast and
                (not compare_base or self.base.same_as(other_type.base)))


class PyObjectType(PyrexType):
    #
    #  Base class for all Python object types (reference-counted).
    #
    #  buffer_defaults  dict or None     Default options for bu

    name = "object"
    is_pyobject = 1
    default_value = "0"
    declaration_value = "0"
    buffer_defaults = None
    is_extern = False
    is_subclassed = False
    is_gc_simple = False

    def __str__(self):
        return "Python object"

    def __repr__(self):
        return "<PyObjectType>"

    def can_coerce_to_pyobject(self, env):
        return True

    def can_coerce_from_pyobject(self, env):
        return True

    def default_coerced_ctype(self):
        """The default C type that this Python type coerces to, or None."""
        return None

    def assignable_from(self, src_type):
        # except for pointers, conversion will be attempted
        return not src_type.is_ptr or src_type.is_string or src_type.is_pyunicode_ptr

    def declaration_code(self, entity_code,
            for_display = 0, dll_linkage = None, pyrex = 0):
        if pyrex or for_display:
            base_code = "object"
        else:
            base_code = public_decl("PyObject", dll_linkage)
            entity_code = "*%s" % entity_code
        return self.base_declaration_code(base_code, entity_code)

    def as_pyobject(self, cname):
        if (not self.is_complete()) or self.is_extension_type:
            return "(PyObject *)" + cname
        else:
            return cname

    def py_type_name(self):
        return "object"

    def __lt__(self, other):
        """
        Make sure we sort highest, as instance checking on py_type_name
        ('object') is always true
        """
        return False

    def global_init_code(self, entry, code):
        code.put_init_var_to_py_none(entry, nanny=False)

    def check_for_null_code(self, cname):
        return cname


builtin_types_that_cannot_create_refcycles = set([
    'bool', 'int', 'long', 'float', 'complex',
    'bytearray', 'bytes', 'unicode', 'str', 'basestring'
])


class BuiltinObjectType(PyObjectType):
    #  objstruct_cname  string           Name of PyObject struct

    is_builtin_type = 1
    has_attributes = 1
    base_type = None
    module_name = '__builtin__'
    require_exact = 1

    # fields that let it look like an extension type
    vtabslot_cname = None
    vtabstruct_cname = None
    vtabptr_cname = None
    typedef_flag = True
    is_external = True
    decl_type = 'PyObject'

    def __init__(self, name, cname, objstruct_cname=None):
        self.name = name
        self.cname = cname
        self.typeptr_cname = "(&%s)" % cname
        self.objstruct_cname = objstruct_cname
        self.is_gc_simple = name in builtin_types_that_cannot_create_refcycles
        if name == 'type':
            # Special case the type type, as many C API calls (and other
            # libraries) actually expect a PyTypeObject* for type arguments.
            self.decl_type = objstruct_cname
        if name == 'Exception':
            self.require_exact = 0

    def set_scope(self, scope):
        self.scope = scope
        if scope:
            scope.parent_type = self

    def __str__(self):
        return "%s object" % self.name

    def __repr__(self):
        return "<%s>"% self.cname

    def default_coerced_ctype(self):
        if self.name in ('bytes', 'bytearray'):
            return c_char_ptr_type
        elif self.name == 'bool':
            return c_bint_type
        elif self.name == 'float':
            return c_double_type
        return None

    def assignable_from(self, src_type):
        if isinstance(src_type, BuiltinObjectType):
            if self.name == 'basestring':
                return src_type.name in ('str', 'unicode', 'basestring')
            else:
                return src_type.name == self.name
        elif src_type.is_extension_type:
            # FIXME: This is an ugly special case that we currently
            # keep supporting.  It allows users to specify builtin
            # types as external extension types, while keeping them
            # compatible with the real builtin types.  We already
            # generate a warning for it.  Big TODO: remove!
            return (src_type.module_name == '__builtin__' and
                    src_type.name == self.name)
        else:
            return True

    def typeobj_is_available(self):
        return True

    def attributes_known(self):
        return True

    def subtype_of(self, type):
        return type.is_pyobject and type.assignable_from(self)

    def type_check_function(self, exact=True):
        type_name = self.name
        if type_name == 'str':
            type_check = 'PyString_Check'
        elif type_name == 'basestring':
            type_check = '__Pyx_PyBaseString_Check'
        elif type_name == 'Exception':
            type_check = '__Pyx_PyException_Check'
        elif type_name == 'bytearray':
            type_check = 'PyByteArray_Check'
        elif type_name == 'frozenset':
            type_check = 'PyFrozenSet_Check'
        else:
            type_check = 'Py%s_Check' % type_name.capitalize()
        if exact and type_name not in ('bool', 'slice', 'Exception'):
            type_check += 'Exact'
        return type_check

    def isinstance_code(self, arg):
        return '%s(%s)' % (self.type_check_function(exact=False), arg)

    def type_test_code(self, arg, notnone=False, exact=True):
        type_check = self.type_check_function(exact=exact)
        check = 'likely(%s(%s))' % (type_check, arg)
        if not notnone:
            check += '||((%s) == Py_None)' % arg
        if self.name == 'basestring':
            name = '(PY_MAJOR_VERSION < 3 ? "basestring" : "str")'
            space_for_name = 16
        else:
            name = '"%s"' % self.name
            # avoid wasting too much space but limit number of different format strings
            space_for_name = (len(self.name) // 16 + 1) * 16
        error = '(PyErr_Format(PyExc_TypeError, "Expected %%.%ds, got %%.200s", %s, Py_TYPE(%s)->tp_name), 0)' % (
            space_for_name, name, arg)
        return check + '||' + error

    def declaration_code(self, entity_code,
            for_display = 0, dll_linkage = None, pyrex = 0):
        if pyrex or for_display:
            base_code = self.name
        else:
            base_code = public_decl(self.decl_type, dll_linkage)
            entity_code = "*%s" % entity_code
        return self.base_declaration_code(base_code, entity_code)

    def as_pyobject(self, cname):
        if self.decl_type == 'PyObject':
            return cname
        else:
            return "(PyObject *)" + cname

    def cast_code(self, expr_code, to_object_struct = False):
        return "((%s*)%s)" % (
            to_object_struct and self.objstruct_cname or self.decl_type, # self.objstruct_cname may be None
            expr_code)

    def py_type_name(self):
        return self.name



class PyExtensionType(PyObjectType):
    #
    #  A Python extension type.
    #
    #  name             string
    #  scope            CClassScope      Attribute namespace
    #  visibility       string
    #  typedef_flag     boolean
    #  base_type        PyExtensionType or None
    #  module_name      string or None   Qualified name of defining module
    #  objstruct_cname  string           Name of PyObject struct
    #  objtypedef_cname string           Name of PyObject struct typedef
    #  typeobj_cname    string or None   C code fragment referring to type object
    #  typeptr_cname    string or None   Name of pointer to external type object
    #  vtabslot_cname   string           Name of C method table member
    #  vtabstruct_cname string           Name of C method table struct
    #  vtabptr_cname    string           Name of pointer to C method table
    #  vtable_cname     string           Name of C method table definition
    #  early_init       boolean          Whether to initialize early (as opposed to during module execution).
    #  defered_declarations [thunk]      Used to declare class hierarchies in order
    #  check_size       'warn', 'error', 'ignore'    What to do if tp_basicsize does not match

    is_extension_type = 1
    has_attributes = 1
    early_init = 1

    objtypedef_cname = None

    def __init__(self, name, typedef_flag, base_type, is_external=0, check_size=None):
        self.name = name
        self.scope = None
        self.typedef_flag = typedef_flag
        if base_type is not None:
            base_type.is_subclassed = True
        self.base_type = base_type
        self.module_name = None
        self.objstruct_cname = None
        self.typeobj_cname = None
        self.typeptr_cname = None
        self.vtabslot_cname = None
        self.vtabstruct_cname = None
        self.vtabptr_cname = None
        self.vtable_cname = None
        self.is_external = is_external
        self.check_size = check_size or 'warn'
        self.defered_declarations = []

    def set_scope(self, scope):
        self.scope = scope
        if scope:
            scope.parent_type = self

    def needs_nonecheck(self):
        return True

    def subtype_of_resolved_type(self, other_type):
        if other_type.is_extension_type or other_type.is_builtin_type:
            return self is other_type or (
                self.base_type and self.base_type.subtype_of(other_type))
        else:
            return other_type is py_object_type

    def typeobj_is_available(self):
        # Do we have a pointer to the type object?
        return self.typeptr_cname

    def typeobj_is_imported(self):
        # If we don't know the C name of the type object but we do
        # know which module it's defined in, it will be imported.
        return self.typeobj_cname is None and self.module_name is not None

    def assignable_from(self, src_type):
        if self == src_type:
            return True
        if isinstance(src_type, PyExtensionType):
            if src_type.base_type is not None:
                return self.assignable_from(src_type.base_type)
        if isinstance(src_type, BuiltinObjectType):
            # FIXME: This is an ugly special case that we currently
            # keep supporting.  It allows users to specify builtin
            # types as external extension types, while keeping them
            # compatible with the real builtin types.  We already
            # generate a warning for it.  Big TODO: remove!
            return (self.module_name == '__builtin__' and
                    self.name == src_type.name)
        return False

    def declaration_code(self, entity_code,
            for_display = 0, dll_linkage = None, pyrex = 0, deref = 0):
        if pyrex or for_display:
            base_code = self.name
        else:
            if self.typedef_flag:
                objstruct = self.objstruct_cname
            else:
                objstruct = "struct %s" % self.objstruct_cname
            base_code = public_decl(objstruct, dll_linkage)
            if deref:
                assert not entity_code
            else:
                entity_code = "*%s" % entity_code
        return self.base_declaration_code(base_code, entity_code)

    def type_test_code(self, py_arg, notnone=False):

        none_check = "((%s) == Py_None)" % py_arg
        type_check = "likely(__Pyx_TypeTest(%s, %s))" % (
            py_arg, self.typeptr_cname)
        if notnone:
            return type_check
        else:
            return "likely(%s || %s)" % (none_check, type_check)

    def attributes_known(self):
        return self.scope is not None

    def __str__(self):
        return self.name

    def __repr__(self):
        return "<PyExtensionType %s%s>" % (self.scope.class_name,
            ("", " typedef")[self.typedef_flag])

    def py_type_name(self):
        if not self.module_name:
            return self.name

        return "__import__(%r, None, None, ['']).%s" % (self.module_name,
                                                        self.name)

class CType(PyrexType):
    #
    #  Base class for all C types (non-reference-counted).
    #
    #  to_py_function     string     C function for converting to Python object
    #  from_py_function   string     C function for constructing from Python object
    #

    to_py_function = None
    from_py_function = None
    exception_value = None
    exception_check = 1

    def create_to_py_utility_code(self, env):
        return self.to_py_function is not None

    def create_from_py_utility_code(self, env):
        return self.from_py_function is not None

    def can_coerce_to_pyobject(self, env):
        return self.create_to_py_utility_code(env)

    def can_coerce_from_pyobject(self, env):
        return self.create_from_py_utility_code(env)

    def error_condition(self, result_code):
        conds = []
        if self.is_string or self.is_pyunicode_ptr:
            conds.append("(!%s)" % result_code)
        elif self.exception_value is not None:
            conds.append("(%s == (%s)%s)" % (result_code, self.sign_and_name(), self.exception_value))
        if self.exception_check:
            conds.append("PyErr_Occurred()")
        if len(conds) > 0:
            return " && ".join(conds)
        else:
            return 0

    def to_py_call_code(self, source_code, result_code, result_type, to_py_function=None):
        func = self.to_py_function if to_py_function is None else to_py_function
        assert func
        if self.is_string or self.is_cpp_string:
            if result_type.is_builtin_type:
                result_type_name = result_type.name
                if result_type_name in ('bytes', 'str', 'unicode'):
                    func = func.replace("Object", result_type_name.title(), 1)
                elif result_type_name == 'bytearray':
                    func = func.replace("Object", "ByteArray", 1)
        return '%s = %s(%s)' % (
            result_code,
            func,
            source_code or 'NULL')

    def from_py_call_code(self, source_code, result_code, error_pos, code,
                          from_py_function=None, error_condition=None):
        return self._assign_from_py_code(
            source_code, result_code, error_pos, code, from_py_function, error_condition)



class PythranExpr(CType):
    # Pythran object of a given type

    to_py_function = "__Pyx_pythran_to_python"
    is_pythran_expr = True
    writable = True
    has_attributes = 1

    def __init__(self, pythran_type, org_buffer=None):
        self.org_buffer = org_buffer
        self.pythran_type = pythran_type
        self.name = self.pythran_type
        self.cname = self.pythran_type
        self.from_py_function = "from_python<%s>" % (self.pythran_type)
        self.scope = None

    def declaration_code(self, entity_code, for_display=0, dll_linkage=None, pyrex=0):
        assert not pyrex
        return "%s %s" % (self.cname, entity_code)

    def attributes_known(self):
        if self.scope is None:
            from . import Symtab
            # FIXME: fake C scope, might be better represented by a struct or C++ class scope
            self.scope = scope = Symtab.CClassScope('', None, visibility="extern")
            scope.parent_type = self
            scope.directives = {}
            scope.declare_var("shape", CPtrType(c_long_type), None, cname="_shape", is_cdef=True)
            scope.declare_var("ndim", c_long_type, None, cname="value", is_cdef=True)

        return True

    def __eq__(self, other):
        return isinstance(other, PythranExpr) and self.pythran_type == other.pythran_type

    def __ne__(self, other):
        return not (isinstance(other, PythranExpr) and self.pythran_type == other.pythran_type)

    def __hash__(self):
        return hash(self.pythran_type)


class CConstType(BaseType):

    is_const = 1

    def __init__(self, const_base_type):
        self.const_base_type = const_base_type
        if const_base_type.has_attributes and const_base_type.scope is not None:
            from . import Symtab
            self.scope = Symtab.CConstScope(const_base_type.scope)

    def __repr__(self):
        return "<CConstType %s>" % repr(self.const_base_type)

    def __str__(self):
        return self.declaration_code("", for_display=1)

    def declaration_code(self, entity_code,
            for_display = 0, dll_linkage = None, pyrex = 0):
        if for_display or pyrex:
            return "const " + self.const_base_type.declaration_code(entity_code, for_display, dll_linkage, pyrex)
        else:
            return self.const_base_type.declaration_code("const %s" % entity_code, for_display, dll_linkage, pyrex)

    def specialize(self, values):
        base_type = self.const_base_type.specialize(values)
        if base_type == self.const_base_type:
            return self
        else:
            return CConstType(base_type)

    def deduce_template_params(self, actual):
        return self.const_base_type.deduce_template_params(actual)

    def can_coerce_to_pyobject(self, env):
        return self.const_base_type.can_coerce_to_pyobject(env)

    def can_coerce_from_pyobject(self, env):
        return self.const_base_type.can_coerce_from_pyobject(env)

    def create_to_py_utility_code(self, env):
        if self.const_base_type.create_to_py_utility_code(env):
            self.to_py_function = self.const_base_type.to_py_function
            return True

    def same_as_resolved_type(self, other_type):
        if other_type.is_const:
            return self.const_base_type.same_as_resolved_type(other_type.const_base_type)
        # Accept const LHS <- non-const RHS.
        return self.const_base_type.same_as_resolved_type(other_type)

    def __getattr__(self, name):
        return getattr(self.const_base_type, name)


class FusedType(CType):
    """
    Represents a Fused Type. All it needs to do is keep track of the types
    it aggregates, as it will be replaced with its specific version wherever
    needed.

    See http://wiki.cython.org/enhancements/fusedtypes

    types           [PyrexType]             is the list of types to be fused
    name            str                     the name of the ctypedef
    """

    is_fused = 1
    exception_check = 0

    def __init__(self, types, name=None):
        # Use list rather than set to preserve order (list should be short).
        flattened_types = []
        for t in types:
            if t.is_fused:
                # recursively merge in subtypes
                for subtype in t.types:
                    if subtype not in flattened_types:
                        flattened_types.append(subtype)
            elif t not in flattened_types:
                flattened_types.append(t)
        self.types = flattened_types
        self.name = name

    def declaration_code(self, entity_code, for_display = 0,
                         dll_linkage = None, pyrex = 0):
        if pyrex or for_display:
            return self.name

        raise Exception("This may never happen, please report a bug")

    def __repr__(self):
        return 'FusedType(name=%r)' % self.name

    def specialize(self, values):
        return values[self]

    def get_fused_types(self, result=None, seen=None):
        if result is None:
            return [self]

        if self not in seen:
            result.append(self)
            seen.add(self)


class CVoidType(CType):
    #
    #   C "void" type
    #

    is_void = 1
    to_py_function = "__Pyx_void_to_None"

    def __repr__(self):
        return "<CVoidType>"

    def declaration_code(self, entity_code,
            for_display = 0, dll_linkage = None, pyrex = 0):
        if pyrex or for_display:
            base_code = "void"
        else:
            base_code = public_decl("void", dll_linkage)
        return self.base_declaration_code(base_code, entity_code)

    def is_complete(self):
        return 0

class InvisibleVoidType(CVoidType):
    #
    #   For use with C++ constructors and destructors return types.
    #   Acts like void, but does not print out a declaration.
    #
    def declaration_code(self, entity_code,
            for_display = 0, dll_linkage = None, pyrex = 0):
        if pyrex or for_display:
            base_code = "[void]"
        else:
            base_code = public_decl("", dll_linkage)
        return self.base_declaration_code(base_code, entity_code)


class CNumericType(CType):
    #
    #   Base class for all C numeric types.
    #
    #   rank      integer     Relative size
    #   signed    integer     0 = unsigned, 1 = unspecified, 2 = explicitly signed
    #

    is_numeric = 1
    default_value = "0"
    has_attributes = True
    scope = None

    sign_words = ("unsigned ", "", "signed ")

    def __init__(self, rank, signed = 1):
        self.rank = rank
        if rank > 0 and signed == SIGNED:
            # Signed is meaningless for anything but char, and complicates
            # type promotion.
            signed = 1
        self.signed = signed

    def sign_and_name(self):
        s = self.sign_words[self.signed]
        n = rank_to_type_name[self.rank]
        return s + n

    def __repr__(self):
        return "<CNumericType %s>" % self.sign_and_name()

    def declaration_code(self, entity_code,
            for_display = 0, dll_linkage = None, pyrex = 0):
        type_name = self.sign_and_name()
        if pyrex or for_display:
            base_code = type_name.replace('PY_LONG_LONG', 'long long')
        else:
            base_code = public_decl(type_name, dll_linkage)
        return self.base_declaration_code(base_code, entity_code)

    def attributes_known(self):
        if self.scope is None:
            from . import Symtab
            self.scope = scope = Symtab.CClassScope(
                    '',
                    None,
                    visibility="extern")
            scope.parent_type = self
            scope.directives = {}
            scope.declare_cfunction(
                    "conjugate",
                    CFuncType(self, [CFuncTypeArg("self", self, None)], nogil=True),
                    pos=None,
                    defining=1,
                    cname=" ")
        return True

    def __lt__(self, other):
        """Sort based on rank, preferring signed over unsigned"""
        if other.is_numeric:
            return self.rank > other.rank and self.signed >= other.signed

        # Prefer numeric types over others
        return True

    def py_type_name(self):
        if self.rank <= 4:
            return "(int, long)"
        return "float"


class ForbidUseClass:
    def __repr__(self):
        raise RuntimeError()
    def __str__(self):
        raise RuntimeError()
ForbidUse = ForbidUseClass()


class CIntLike(object):
    """Mixin for shared behaviour of C integers and enums.
    """
    to_py_function = None
    from_py_function = None
    to_pyunicode_utility = None
    default_format_spec = 'd'

    def can_coerce_to_pyobject(self, env):
        return True

    def can_coerce_from_pyobject(self, env):
        return True

    def create_to_py_utility_code(self, env):
        if type(self).to_py_function is None:
            self.to_py_function = "__Pyx_PyInt_From_" + self.specialization_name()
            env.use_utility_code(TempitaUtilityCode.load_cached(
                "CIntToPy", "TypeConversion.c",
                context={"TYPE": self.empty_declaration_code(),
                         "TO_PY_FUNCTION": self.to_py_function}))
        return True

    def create_from_py_utility_code(self, env):
        if type(self).from_py_function is None:
            self.from_py_function = "__Pyx_PyInt_As_" + self.specialization_name()
            env.use_utility_code(TempitaUtilityCode.load_cached(
                "CIntFromPy", "TypeConversion.c",
                context={"TYPE": self.empty_declaration_code(),
                         "FROM_PY_FUNCTION": self.from_py_function}))
        return True

    @staticmethod
    def _parse_format(format_spec):
        padding = ' '
        if not format_spec:
            return ('d', 0, padding)
        format_type = format_spec[-1]
        if format_type in ('o', 'd', 'x', 'X'):
            prefix = format_spec[:-1]
        elif format_type.isdigit():
            format_type = 'd'
            prefix = format_spec
        else:
            return (None, 0, padding)
        if not prefix:
            return (format_type, 0, padding)
        if prefix[0] == '-':
            prefix = prefix[1:]
        if prefix and prefix[0] == '0':
            padding = '0'
            prefix = prefix.lstrip('0')
        if prefix.isdigit():
            return (format_type, int(prefix), padding)
        return (None, 0, padding)

    def can_coerce_to_pystring(self, env, format_spec=None):
        format_type, width, padding = self._parse_format(format_spec)
        return format_type is not None and width <= 2**30

    def convert_to_pystring(self, cvalue, code, format_spec=None):
        if self.to_pyunicode_utility is None:
            utility_code_name = "__Pyx_PyUnicode_From_" + self.specialization_name()
            to_pyunicode_utility = TempitaUtilityCode.load_cached(
                "CIntToPyUnicode", "TypeConversion.c",
                context={"TYPE": self.empty_declaration_code(),
                         "TO_PY_FUNCTION": utility_code_name})
            self.to_pyunicode_utility = (utility_code_name, to_pyunicode_utility)
        else:
            utility_code_name, to_pyunicode_utility = self.to_pyunicode_utility
        code.globalstate.use_utility_code(to_pyunicode_utility)
        format_type, width, padding_char = self._parse_format(format_spec)
        return "%s(%s, %d, '%s', '%s')" % (utility_code_name, cvalue, width, padding_char, format_type)


class CIntType(CIntLike, CNumericType):

    is_int = 1
    typedef_flag = 0
    exception_value = -1

    def get_to_py_type_conversion(self):
        if self.rank < list(rank_to_type_name).index('int'):
            # This assumes sizeof(short) < sizeof(int)
            return "PyInt_FromLong"
        else:
            # Py{Int|Long}_From[Unsigned]Long[Long]
            Prefix = "Int"
            SignWord = ""
            TypeName = "Long"
            if not self.signed:
                Prefix = "Long"
                SignWord = "Unsigned"
            if self.rank >= list(rank_to_type_name).index('PY_LONG_LONG'):
                Prefix = "Long"
                TypeName = "LongLong"
            return "Py%s_From%s%s" % (Prefix, SignWord, TypeName)

    def assignable_from_resolved_type(self, src_type):
        return src_type.is_int or src_type.is_enum or src_type is error_type

    def invalid_value(self):
        if rank_to_type_name[int(self.rank)] == 'char':
            return "'?'"
        else:
            # We do not really know the size of the type, so return
            # a 32-bit literal and rely on casting to final type. It will
            # be negative for signed ints, which is good.
            return "0xbad0bad0"

    def overflow_check_binop(self, binop, env, const_rhs=False):
        env.use_utility_code(UtilityCode.load("Common", "Overflow.c"))
        type = self.empty_declaration_code()
        name = self.specialization_name()
        if binop == "lshift":
            env.use_utility_code(TempitaUtilityCode.load_cached(
                "LeftShift", "Overflow.c",
                context={'TYPE': type, 'NAME': name, 'SIGNED': self.signed}))
        else:
            if const_rhs:
                binop += "_const"
            if type in ('int', 'long', 'long long'):
                env.use_utility_code(TempitaUtilityCode.load_cached(
                    "BaseCaseSigned", "Overflow.c",
                    context={'INT': type, 'NAME': name}))
            elif type in ('unsigned int', 'unsigned long', 'unsigned long long'):
                env.use_utility_code(TempitaUtilityCode.load_cached(
                    "BaseCaseUnsigned", "Overflow.c",
                    context={'UINT': type, 'NAME': name}))
            elif self.rank <= 1:
                # sizeof(short) < sizeof(int)
                return "__Pyx_%s_%s_no_overflow" % (binop, name)
            else:
                _load_overflow_base(env)
                env.use_utility_code(TempitaUtilityCode.load_cached(
                    "SizeCheck", "Overflow.c",
                    context={'TYPE': type, 'NAME': name}))
                env.use_utility_code(TempitaUtilityCode.load_cached(
                    "Binop", "Overflow.c",
                    context={'TYPE': type, 'NAME': name, 'BINOP': binop}))
        return "__Pyx_%s_%s_checking_overflow" % (binop, name)


def _load_overflow_base(env):
    env.use_utility_code(UtilityCode.load("Common", "Overflow.c"))
    for type in ('int', 'long', 'long long'):
        env.use_utility_code(TempitaUtilityCode.load_cached(
            "BaseCaseSigned", "Overflow.c",
            context={'INT': type, 'NAME': type.replace(' ', '_')}))
    for type in ('unsigned int', 'unsigned long', 'unsigned long long'):
        env.use_utility_code(TempitaUtilityCode.load_cached(
            "BaseCaseUnsigned", "Overflow.c",
            context={'UINT': type, 'NAME': type.replace(' ', '_')}))


class CAnonEnumType(CIntType):

    is_enum = 1

    def sign_and_name(self):
        return 'int'


class CReturnCodeType(CIntType):

    to_py_function = "__Pyx_Owned_Py_None"

    is_returncode = True
    exception_check = False
    default_format_spec = ''

    def can_coerce_to_pystring(self, env, format_spec=None):
        return not format_spec

    def convert_to_pystring(self, cvalue, code, format_spec=None):
        return "__Pyx_NewRef(%s)" % code.globalstate.get_py_string_const(StringEncoding.EncodedString("None")).cname


class CBIntType(CIntType):

    to_py_function = "__Pyx_PyBool_FromLong"
    from_py_function = "__Pyx_PyObject_IsTrue"
    exception_check = 1  # for C++ bool
    default_format_spec = ''

    def can_coerce_to_pystring(self, env, format_spec=None):
        return not format_spec or super(CBIntType, self).can_coerce_to_pystring(env, format_spec)

    def convert_to_pystring(self, cvalue, code, format_spec=None):
        if format_spec:
            return super(CBIntType, self).convert_to_pystring(cvalue, code, format_spec)
        # NOTE: no caching here as the string constant cnames depend on the current module
        utility_code_name = "__Pyx_PyUnicode_FromBInt_" + self.specialization_name()
        to_pyunicode_utility = TempitaUtilityCode.load_cached(
            "CBIntToPyUnicode", "TypeConversion.c", context={
                "TRUE_CONST":  code.globalstate.get_py_string_const(StringEncoding.EncodedString("True")).cname,
                "FALSE_CONST": code.globalstate.get_py_string_const(StringEncoding.EncodedString("False")).cname,
                "TO_PY_FUNCTION": utility_code_name,
            })
        code.globalstate.use_utility_code(to_pyunicode_utility)
        return "%s(%s)" % (utility_code_name, cvalue)

    def declaration_code(self, entity_code,
            for_display = 0, dll_linkage = None, pyrex = 0):
        if for_display:
            base_code = 'bool'
        elif pyrex:
            base_code = 'bint'
        else:
            base_code = public_decl('int', dll_linkage)
        return self.base_declaration_code(base_code, entity_code)

    def __repr__(self):
        return "<CNumericType bint>"

    def __str__(self):
        return 'bint'

    def py_type_name(self):
        return "bool"


class CPyUCS4IntType(CIntType):
    # Py_UCS4

    is_unicode_char = True

    # Py_UCS4 coerces from and to single character unicode strings (or
    # at most two characters on 16bit Unicode builds), but we also
    # allow Python integers as input.  The value range for Py_UCS4
    # is 0..1114111, which is checked when converting from an integer
    # value.

    to_py_function = "PyUnicode_FromOrdinal"
    from_py_function = "__Pyx_PyObject_AsPy_UCS4"

    def can_coerce_to_pystring(self, env, format_spec=None):
        return False  # does the right thing anyway

    def create_from_py_utility_code(self, env):
        env.use_utility_code(UtilityCode.load_cached("ObjectAsUCS4", "TypeConversion.c"))
        return True

    def sign_and_name(self):
        return "Py_UCS4"


class CPyUnicodeIntType(CIntType):
    # Py_UNICODE

    is_unicode_char = True

    # Py_UNICODE coerces from and to single character unicode strings,
    # but we also allow Python integers as input.  The value range for
    # Py_UNICODE is 0..1114111, which is checked when converting from
    # an integer value.

    to_py_function = "PyUnicode_FromOrdinal"
    from_py_function = "__Pyx_PyObject_AsPy_UNICODE"

    def can_coerce_to_pystring(self, env, format_spec=None):
        return False  # does the right thing anyway

    def create_from_py_utility_code(self, env):
        env.use_utility_code(UtilityCode.load_cached("ObjectAsPyUnicode", "TypeConversion.c"))
        return True

    def sign_and_name(self):
        return "Py_UNICODE"


class CPyHashTType(CIntType):

    to_py_function = "__Pyx_PyInt_FromHash_t"
    from_py_function = "__Pyx_PyInt_AsHash_t"

    def sign_and_name(self):
        return "Py_hash_t"

class CPySSizeTType(CIntType):

    to_py_function = "PyInt_FromSsize_t"
    from_py_function = "__Pyx_PyIndex_AsSsize_t"

    def sign_and_name(self):
        return "Py_ssize_t"

class CSSizeTType(CIntType):

    to_py_function = "PyInt_FromSsize_t"
    from_py_function = "PyInt_AsSsize_t"

    def sign_and_name(self):
        return "Py_ssize_t"

class CSizeTType(CIntType):

    to_py_function = "__Pyx_PyInt_FromSize_t"

    def sign_and_name(self):
        return "size_t"

class CPtrdiffTType(CIntType):

    def sign_and_name(self):
        return "ptrdiff_t"


class CFloatType(CNumericType):

    is_float = 1
    to_py_function = "PyFloat_FromDouble"
    from_py_function = "__pyx_PyFloat_AsDouble"

    exception_value = -1

    def __init__(self, rank, math_h_modifier = ''):
        CNumericType.__init__(self, rank, 1)
        self.math_h_modifier = math_h_modifier
        if rank == RANK_FLOAT:
            self.from_py_function = "__pyx_PyFloat_AsFloat"

    def assignable_from_resolved_type(self, src_type):
        return (src_type.is_numeric and not src_type.is_complex) or src_type is error_type

    def invalid_value(self):
        return Naming.PYX_NAN

class CComplexType(CNumericType):

    is_complex = 1
    to_py_function = "__pyx_PyComplex_FromComplex"
    has_attributes = 1
    scope = None

    def __init__(self, real_type):
        while real_type.is_typedef and not real_type.typedef_is_external:
            real_type = real_type.typedef_base_type
        self.funcsuffix = "_%s" % real_type.specialization_name()
        if real_type.is_float:
            self.math_h_modifier = real_type.math_h_modifier
        else:
            self.math_h_modifier = "_UNUSED"

        self.real_type = real_type
        CNumericType.__init__(self, real_type.rank + 0.5, real_type.signed)
        self.binops = {}
        self.from_parts = "%s_from_parts" % self.specialization_name()
        self.default_value = "%s(0, 0)" % self.from_parts

    def __eq__(self, other):
        if isinstance(self, CComplexType) and isinstance(other, CComplexType):
            return self.real_type == other.real_type
        else:
            return False

    def __ne__(self, other):
        if isinstance(self, CComplexType) and isinstance(other, CComplexType):
            return self.real_type != other.real_type
        else:
            return True

    def __lt__(self, other):
        if isinstance(self, CComplexType) and isinstance(other, CComplexType):
            return self.real_type < other.real_type
        else:
            # this is arbitrary, but it makes sure we always have
            # *some* kind of order
            return False

    def __hash__(self):
        return ~hash(self.real_type)

    def declaration_code(self, entity_code,
            for_display = 0, dll_linkage = None, pyrex = 0):
        if pyrex or for_display:
            real_code = self.real_type.declaration_code("", for_display, dll_linkage, pyrex)
            base_code = "%s complex" % real_code
        else:
            base_code = public_decl(self.sign_and_name(), dll_linkage)
        return self.base_declaration_code(base_code, entity_code)

    def sign_and_name(self):
        real_type_name = self.real_type.specialization_name()
        real_type_name = real_type_name.replace('long__double','long_double')
        real_type_name = real_type_name.replace('PY_LONG_LONG','long_long')
        return Naming.type_prefix + real_type_name + "_complex"

    def assignable_from(self, src_type):
        # Temporary hack/feature disabling, see #441
        if (not src_type.is_complex and src_type.is_numeric and src_type.is_typedef
            and src_type.typedef_is_external):
             return False
        elif src_type.is_pyobject:
            return True
        else:
            return super(CComplexType, self).assignable_from(src_type)

    def assignable_from_resolved_type(self, src_type):
        return (src_type.is_complex and self.real_type.assignable_from_resolved_type(src_type.real_type)
                    or src_type.is_numeric and self.real_type.assignable_from_resolved_type(src_type)
                    or src_type is error_type)

    def attributes_known(self):
        if self.scope is None:
            from . import Symtab
            self.scope = scope = Symtab.CClassScope(
                    '',
                    None,
                    visibility="extern")
            scope.parent_type = self
            scope.directives = {}
            scope.declare_var("real", self.real_type, None, cname="real", is_cdef=True)
            scope.declare_var("imag", self.real_type, None, cname="imag", is_cdef=True)
            scope.declare_cfunction(
                    "conjugate",
                    CFuncType(self, [CFuncTypeArg("self", self, None)], nogil=True),
                    pos=None,
                    defining=1,
                    cname="__Pyx_c_conj%s" % self.funcsuffix)

        return True

    def _utility_code_context(self):
        return {
            'type': self.empty_declaration_code(),
            'type_name': self.specialization_name(),
            'real_type': self.real_type.empty_declaration_code(),
            'func_suffix': self.funcsuffix,
            'm': self.math_h_modifier,
            'is_float': int(self.real_type.is_float)
        }

    def create_declaration_utility_code(self, env):
        # This must always be run, because a single CComplexType instance can be shared
        # across multiple compilations (the one created in the module scope)
        env.use_utility_code(UtilityCode.load_cached('Header', 'Complex.c'))
        env.use_utility_code(UtilityCode.load_cached('RealImag', 'Complex.c'))
        env.use_utility_code(TempitaUtilityCode.load_cached(
            'Declarations', 'Complex.c', self._utility_code_context()))
        env.use_utility_code(TempitaUtilityCode.load_cached(
            'Arithmetic', 'Complex.c', self._utility_code_context()))
        return True

    def can_coerce_to_pyobject(self, env):
        return True

    def can_coerce_from_pyobject(self, env):
        return True

    def create_to_py_utility_code(self, env):
        env.use_utility_code(UtilityCode.load_cached('ToPy', 'Complex.c'))
        return True

    def create_from_py_utility_code(self, env):
        env.use_utility_code(TempitaUtilityCode.load_cached(
            'FromPy', 'Complex.c', self._utility_code_context()))
        self.from_py_function = "__Pyx_PyComplex_As_" + self.specialization_name()
        return True

    def lookup_op(self, nargs, op):
        try:
            return self.binops[nargs, op]
        except KeyError:
            pass
        try:
            op_name = complex_ops[nargs, op]
            self.binops[nargs, op] = func_name = "__Pyx_c_%s%s" % (op_name, self.funcsuffix)
            return func_name
        except KeyError:
            return None

    def unary_op(self, op):
        return self.lookup_op(1, op)

    def binary_op(self, op):
        return self.lookup_op(2, op)

    def py_type_name(self):
        return "complex"

    def cast_code(self, expr_code):
        return expr_code

complex_ops = {
    (1, '-'): 'neg',
    (1, 'zero'): 'is_zero',
    (2, '+'): 'sum',
    (2, '-'): 'diff',
    (2, '*'): 'prod',
    (2, '/'): 'quot',
    (2, '**'): 'pow',
    (2, '=='): 'eq',
}


class CPyTSSTType(CType):
    #
    #   PEP-539 "Py_tss_t" type
    #

    declaration_value = "Py_tss_NEEDS_INIT"

    def __repr__(self):
        return "<Py_tss_t>"

    def declaration_code(self, entity_code,
                         for_display=0, dll_linkage=None, pyrex=0):
        if pyrex or for_display:
            base_code = "Py_tss_t"
        else:
            base_code = public_decl("Py_tss_t", dll_linkage)
        return self.base_declaration_code(base_code, entity_code)


class CPointerBaseType(CType):
    # common base type for pointer/array types
    #
    #  base_type     CType              Reference type

    subtypes = ['base_type']

    def __init__(self, base_type):
        self.base_type = base_type
        if base_type.is_const:
            base_type = base_type.const_base_type
        for char_type in (c_char_type, c_uchar_type, c_schar_type):
            if base_type.same_as(char_type):
                self.is_string = 1
                break
        else:
            if base_type.same_as(c_py_unicode_type):
                self.is_pyunicode_ptr = 1

        if self.is_string and not base_type.is_error:
            if base_type.signed == 2:
                self.to_py_function = "__Pyx_PyObject_FromCString"
                if self.is_ptr:
                    self.from_py_function = "__Pyx_PyObject_As%sSString"
            elif base_type.signed:
                self.to_py_function = "__Pyx_PyObject_FromString"
                if self.is_ptr:
                    self.from_py_function = "__Pyx_PyObject_As%sString"
            else:
                self.to_py_function = "__Pyx_PyObject_FromCString"
                if self.is_ptr:
                    self.from_py_function = "__Pyx_PyObject_As%sUString"
            if self.is_ptr:
                self.from_py_function %= '' if self.base_type.is_const else 'Writable'
            self.exception_value = "NULL"
        elif self.is_pyunicode_ptr and not base_type.is_error:
            self.to_py_function = "__Pyx_PyUnicode_FromUnicode"
            if self.is_ptr:
                self.from_py_function = "__Pyx_PyUnicode_AsUnicode"
            self.exception_value = "NULL"

    def py_type_name(self):
        if self.is_string:
            return "bytes"
        elif self.is_pyunicode_ptr:
            return "unicode"
        else:
            return super(CPointerBaseType, self).py_type_name()

    def literal_code(self, value):
        if self.is_string:
            assert isinstance(value, str)
            return '"%s"' % StringEncoding.escape_byte_string(value)


class CArrayType(CPointerBaseType):
    #  base_type     CType              Element type
    #  size          integer or None    Number of elements

    is_array = 1
    to_tuple_function = None

    def __init__(self, base_type, size):
        super(CArrayType, self).__init__(base_type)
        self.size = size

    def __eq__(self, other):
        if isinstance(other, CType) and other.is_array and self.size == other.size:
            return self.base_type.same_as(other.base_type)
        return False

    def __hash__(self):
        return hash(self.base_type) + 28 # arbitrarily chosen offset

    def __repr__(self):
        return "<CArrayType %s %s>" % (self.size, repr(self.base_type))

    def same_as_resolved_type(self, other_type):
        return ((other_type.is_array and
            self.base_type.same_as(other_type.base_type))
                or other_type is error_type)

    def assignable_from_resolved_type(self, src_type):
        # C arrays are assigned by value, either Python containers or C arrays/pointers
        if src_type.is_pyobject:
            return True
        if src_type.is_ptr or src_type.is_array:
            return self.base_type.assignable_from(src_type.base_type)
        return False

    def element_ptr_type(self):
        return c_ptr_type(self.base_type)

    def declaration_code(self, entity_code,
            for_display = 0, dll_linkage = None, pyrex = 0):
        if self.size is not None:
            dimension_code = self.size
        else:
            dimension_code = ""
        if entity_code.startswith("*"):
            entity_code = "(%s)" % entity_code
        return self.base_type.declaration_code(
            "%s[%s]" % (entity_code, dimension_code),
            for_display, dll_linkage, pyrex)

    def as_argument_type(self):
        return c_ptr_type(self.base_type)

    def is_complete(self):
        return self.size is not None

    def specialize(self, values):
        base_type = self.base_type.specialize(values)
        if base_type == self.base_type:
            return self
        else:
            return CArrayType(base_type, self.size)

    def deduce_template_params(self, actual):
        if isinstance(actual, CArrayType):
            return self.base_type.deduce_template_params(actual.base_type)
        else:
            return {}

    def can_coerce_to_pyobject(self, env):
        return self.base_type.can_coerce_to_pyobject(env)

    def can_coerce_from_pyobject(self, env):
        return self.base_type.can_coerce_from_pyobject(env)

    def create_to_py_utility_code(self, env):
        if self.to_py_function is not None:
            return self.to_py_function
        if not self.base_type.create_to_py_utility_code(env):
            return False

        safe_typename = self.base_type.specialization_name()
        to_py_function = "__Pyx_carray_to_py_%s" % safe_typename
        to_tuple_function = "__Pyx_carray_to_tuple_%s" % safe_typename

        from .UtilityCode import CythonUtilityCode
        context = {
            'cname': to_py_function,
            'to_tuple_cname': to_tuple_function,
            'base_type': self.base_type,
        }
        env.use_utility_code(CythonUtilityCode.load(
            "carray.to_py", "CConvert.pyx",
            outer_module_scope=env.global_scope(),  # need access to types declared in module
            context=context, compiler_directives=dict(env.global_scope().directives)))
        self.to_tuple_function = to_tuple_function
        self.to_py_function = to_py_function
        return True

    def to_py_call_code(self, source_code, result_code, result_type, to_py_function=None):
        func = self.to_py_function if to_py_function is None else to_py_function
        if self.is_string or self.is_pyunicode_ptr:
            return '%s = %s(%s)' % (
                result_code,
                func,
                source_code)
        target_is_tuple = result_type.is_builtin_type and result_type.name == 'tuple'
        return '%s = %s(%s, %s)' % (
            result_code,
            self.to_tuple_function if target_is_tuple else func,
            source_code,
            self.size)

    def create_from_py_utility_code(self, env):
        if self.from_py_function is not None:
            return self.from_py_function
        if not self.base_type.create_from_py_utility_code(env):
            return False

        from_py_function = "__Pyx_carray_from_py_%s" % self.base_type.specialization_name()

        from .UtilityCode import CythonUtilityCode
        context = {
            'cname': from_py_function,
            'base_type': self.base_type,
        }
        env.use_utility_code(CythonUtilityCode.load(
            "carray.from_py", "CConvert.pyx",
            outer_module_scope=env.global_scope(),  # need access to types declared in module
            context=context, compiler_directives=dict(env.global_scope().directives)))
        self.from_py_function = from_py_function
        return True

    def from_py_call_code(self, source_code, result_code, error_pos, code,
                          from_py_function=None, error_condition=None):
        assert not error_condition, '%s: %s' % (error_pos, error_condition)
        call_code = "%s(%s, %s, %s)" % (
            from_py_function or self.from_py_function,
            source_code, result_code, self.size)
        return code.error_goto_if_neg(call_code, error_pos)


class CPtrType(CPointerBaseType):
    #  base_type     CType              Reference type

    is_ptr = 1
    default_value = "0"

    def __hash__(self):
        return hash(self.base_type) + 27 # arbitrarily chosen offset

    def __eq__(self, other):
        if isinstance(other, CType) and other.is_ptr:
            return self.base_type.same_as(other.base_type)
        return False

    def __ne__(self, other):
        return not (self == other)

    def __repr__(self):
        return "<CPtrType %s>" % repr(self.base_type)

    def same_as_resolved_type(self, other_type):
        return ((other_type.is_ptr and
            self.base_type.same_as(other_type.base_type))
                or other_type is error_type)

    def declaration_code(self, entity_code,
            for_display = 0, dll_linkage = None, pyrex = 0):
        #print "CPtrType.declaration_code: pointer to", self.base_type ###
        return self.base_type.declaration_code(
            "*%s" % entity_code,
            for_display, dll_linkage, pyrex)

    def assignable_from_resolved_type(self, other_type):
        if other_type is error_type:
            return 1
        if other_type.is_null_ptr:
            return 1
        if self.base_type.is_const:
            self = CPtrType(self.base_type.const_base_type)
        if self.base_type.is_cfunction:
            if other_type.is_ptr:
                other_type = other_type.base_type.resolve()
            if other_type.is_cfunction:
                return self.base_type.pointer_assignable_from_resolved_type(other_type)
            else:
                return 0
        if (self.base_type.is_cpp_class and other_type.is_ptr
                and other_type.base_type.is_cpp_class and other_type.base_type.is_subclass(self.base_type)):
            return 1
        if other_type.is_array or other_type.is_ptr:
            return self.base_type.is_void or self.base_type.same_as(other_type.base_type)
        return 0

    def specialize(self, values):
        base_type = self.base_type.specialize(values)
        if base_type == self.base_type:
            return self
        else:
            return CPtrType(base_type)

    def deduce_template_params(self, actual):
        if isinstance(actual, CPtrType):
            return self.base_type.deduce_template_params(actual.base_type)
        else:
            return {}

    def invalid_value(self):
        return "1"

    def find_cpp_operation_type(self, operator, operand_type=None):
        if self.base_type.is_cpp_class:
            return self.base_type.find_cpp_operation_type(operator, operand_type)
        return None


class CNullPtrType(CPtrType):

    is_null_ptr = 1


class CReferenceType(BaseType):

    is_reference = 1
    is_fake_reference = 0

    def __init__(self, base_type):
        self.ref_base_type = base_type

    def __repr__(self):
        return "<CReferenceType %s>" % repr(self.ref_base_type)

    def __str__(self):
        return "%s &" % self.ref_base_type

    def declaration_code(self, entity_code,
            for_display = 0, dll_linkage = None, pyrex = 0):
        #print "CReferenceType.declaration_code: pointer to", self.base_type ###
        return self.ref_base_type.declaration_code(
            "&%s" % entity_code,
            for_display, dll_linkage, pyrex)

    def specialize(self, values):
        base_type = self.ref_base_type.specialize(values)
        if base_type == self.ref_base_type:
            return self
        else:
            return type(self)(base_type)

    def deduce_template_params(self, actual):
        return self.ref_base_type.deduce_template_params(actual)

    def __getattr__(self, name):
        return getattr(self.ref_base_type, name)


class CFakeReferenceType(CReferenceType):

    is_fake_reference = 1

    def __repr__(self):
        return "<CFakeReferenceType %s>" % repr(self.ref_base_type)

    def __str__(self):
        return "%s [&]" % self.ref_base_type

    def declaration_code(self, entity_code,
            for_display = 0, dll_linkage = None, pyrex = 0):
        #print "CReferenceType.declaration_code: pointer to", self.base_type ###
        return "__Pyx_FakeReference<%s> %s" % (self.ref_base_type.empty_declaration_code(), entity_code)


class CFuncType(CType):
    #  return_type      CType
    #  args             [CFuncTypeArg]
    #  has_varargs      boolean
    #  exception_value  string
    #  exception_check  boolean    True if PyErr_Occurred check needed
    #  calling_convention  string  Function calling convention
    #  nogil            boolean    Can be called without gil
    #  with_gil         boolean    Acquire gil around function body
    #  templates        [string] or None
    #  cached_specialized_types [CFuncType]   cached specialized versions of the CFuncType if defined in a pxd
    #  from_fused       boolean    Indicates whether this is a specialized
    #                              C function
    #  is_strict_signature boolean  function refuses to accept coerced arguments
    #                               (used for optimisation overrides)
    #  is_const_method  boolean
    #  is_static_method boolean

    is_cfunction = 1
    original_sig = None
    cached_specialized_types = None
    from_fused = False
    is_const_method = False

    subtypes = ['return_type', 'args']

    def __init__(self, return_type, args, has_varargs = 0,
            exception_value = None, exception_check = 0, calling_convention = "",
            nogil = 0, with_gil = 0, is_overridable = 0, optional_arg_count = 0,
            is_const_method = False, is_static_method=False,
            templates = None, is_strict_signature = False):
        self.return_type = return_type
        self.args = args
        self.has_varargs = has_varargs
        self.optional_arg_count = optional_arg_count
        self.exception_value = exception_value
        self.exception_check = exception_check
        self.calling_convention = calling_convention
        self.nogil = nogil
        self.with_gil = with_gil
        self.is_overridable = is_overridable
        self.is_const_method = is_const_method
        self.is_static_method = is_static_method
        self.templates = templates
        self.is_strict_signature = is_strict_signature

    def __repr__(self):
        arg_reprs = list(map(repr, self.args))
        if self.has_varargs:
            arg_reprs.append("...")
        if self.exception_value:
            except_clause = " %r" % self.exception_value
        else:
            except_clause = ""
        if self.exception_check:
            except_clause += "?"
        return "<CFuncType %s %s[%s]%s>" % (
            repr(self.return_type),
            self.calling_convention_prefix(),
            ",".join(arg_reprs),
            except_clause)

    def with_with_gil(self, with_gil):
        if with_gil == self.with_gil:
            return self
        else:
            return CFuncType(
                self.return_type, self.args, self.has_varargs,
                self.exception_value, self.exception_check,
                self.calling_convention, self.nogil,
                with_gil,
                self.is_overridable, self.optional_arg_count,
                self.is_const_method, self.is_static_method,
                self.templates, self.is_strict_signature)

    def calling_convention_prefix(self):
        cc = self.calling_convention
        if cc:
            return cc + " "
        else:
            return ""

    def as_argument_type(self):
        return c_ptr_type(self)

    def same_c_signature_as(self, other_type, as_cmethod = 0):
        return self.same_c_signature_as_resolved_type(
            other_type.resolve(), as_cmethod)

    def same_c_signature_as_resolved_type(self, other_type, as_cmethod=False, as_pxd_definition=False,
                                          exact_semantics=True):
        # If 'exact_semantics' is false, allow any equivalent C signatures
        # if the Cython semantics are compatible, i.e. the same or wider for 'other_type'.

        #print "CFuncType.same_c_signature_as_resolved_type:", \
        #    self, other_type, "as_cmethod =", as_cmethod ###
        if other_type is error_type:
            return 1
        if not other_type.is_cfunction:
            return 0
        if self.is_overridable != other_type.is_overridable:
            return 0
        nargs = len(self.args)
        if nargs != len(other_type.args):
            return 0
        # When comparing C method signatures, the first argument
        # is exempt from compatibility checking (the proper check
        # is performed elsewhere).
        for i in range(as_cmethod, nargs):
            if not self.args[i].type.same_as(other_type.args[i].type):
                return 0
        if self.has_varargs != other_type.has_varargs:
            return 0
        if self.optional_arg_count != other_type.optional_arg_count:
            return 0
        if as_pxd_definition:
            # A narrowing of the return type declared in the pxd is allowed.
            if not self.return_type.subtype_of_resolved_type(other_type.return_type):
                return 0
        else:
            if not self.return_type.same_as(other_type.return_type):
                return 0
        if not self.same_calling_convention_as(other_type):
            return 0
        if exact_semantics:
            if self.exception_check != other_type.exception_check:
                return 0
            if not self._same_exception_value(other_type.exception_value):
                return 0
        elif not self._is_exception_compatible_with(other_type):
            return 0
        return 1

    def _same_exception_value(self, other_exc_value):
        if self.exception_value == other_exc_value:
            return 1
        if self.exception_check != '+':
            return 0
        if not self.exception_value or not other_exc_value:
            return 0
        if self.exception_value.type != other_exc_value.type:
            return 0
        if self.exception_value.entry and other_exc_value.entry:
            if self.exception_value.entry.cname != other_exc_value.entry.cname:
                return 0
        if self.exception_value.name != other_exc_value.name:
            return 0
        return 1

    def compatible_signature_with(self, other_type, as_cmethod = 0):
        return self.compatible_signature_with_resolved_type(other_type.resolve(), as_cmethod)

    def compatible_signature_with_resolved_type(self, other_type, as_cmethod):
        #print "CFuncType.same_c_signature_as_resolved_type:", \
        #    self, other_type, "as_cmethod =", as_cmethod ###
        if other_type is error_type:
            return 1
        if not other_type.is_cfunction:
            return 0
        if not self.is_overridable and other_type.is_overridable:
            return 0
        nargs = len(self.args)
        if nargs - self.optional_arg_count != len(other_type.args) - other_type.optional_arg_count:
            return 0
        if self.optional_arg_count < other_type.optional_arg_count:
            return 0
        # When comparing C method signatures, the first argument
        # is exempt from compatibility checking (the proper check
        # is performed elsewhere).
        for i in range(as_cmethod, len(other_type.args)):
            if not self.args[i].type.same_as(
                other_type.args[i].type):
                    return 0
        if self.has_varargs != other_type.has_varargs:
            return 0
        if not self.return_type.subtype_of_resolved_type(other_type.return_type):
            return 0
        if not self.same_calling_convention_as(other_type):
            return 0
        if self.nogil != other_type.nogil:
            return 0
        if not self._is_exception_compatible_with(other_type):
            return 0
        self.original_sig = other_type.original_sig or other_type
        return 1

    def _is_exception_compatible_with(self, other_type):
        # narrower exception checks are ok, but prevent mismatches
        if self.exception_check == '+' and other_type.exception_check != '+':
            # must catch C++ exceptions if we raise them
            return 0
        if not other_type.exception_check or other_type.exception_value is not None:
            # if other does not *always* check exceptions, self must comply
            if not self._same_exception_value(other_type.exception_value):
                return 0
            if self.exception_check and self.exception_check != other_type.exception_check:
                # a redundant exception check doesn't make functions incompatible, but a missing one does
                return 0
        return 1

    def narrower_c_signature_than(self, other_type, as_cmethod = 0):
        return self.narrower_c_signature_than_resolved_type(other_type.resolve(), as_cmethod)

    def narrower_c_signature_than_resolved_type(self, other_type, as_cmethod):
        if other_type is error_type:
            return 1
        if not other_type.is_cfunction:
            return 0
        nargs = len(self.args)
        if nargs != len(other_type.args):
            return 0
        for i in range(as_cmethod, nargs):
            if not self.args[i].type.subtype_of_resolved_type(other_type.args[i].type):
                return 0
            else:
                self.args[i].needs_type_test = other_type.args[i].needs_type_test \
                        or not self.args[i].type.same_as(other_type.args[i].type)
        if self.has_varargs != other_type.has_varargs:
            return 0
        if self.optional_arg_count != other_type.optional_arg_count:
            return 0
        if not self.return_type.subtype_of_resolved_type(other_type.return_type):
            return 0
        if not self.exception_check and other_type.exception_check:
            # a redundant exception check doesn't make functions incompatible, but a missing one does
            return 0
        if not self._same_exception_value(other_type.exception_value):
            return 0
        return 1

    def same_calling_convention_as(self, other):
        ## XXX Under discussion ...
        ## callspec_words = ("__stdcall", "__cdecl", "__fastcall")
        ## cs1 = self.calling_convention
        ## cs2 = other.calling_convention
        ## if (cs1 in callspec_words or
        ##     cs2 in callspec_words):
        ##     return cs1 == cs2
        ## else:
        ##     return True
        sc1 = self.calling_convention == '__stdcall'
        sc2 = other.calling_convention == '__stdcall'
        return sc1 == sc2

    def same_as_resolved_type(self, other_type, as_cmethod=False):
        return self.same_c_signature_as_resolved_type(other_type, as_cmethod=as_cmethod) \
            and self.nogil == other_type.nogil

    def pointer_assignable_from_resolved_type(self, rhs_type):
        # Accept compatible exception/nogil declarations for the RHS.
        if rhs_type is error_type:
            return 1
        if not rhs_type.is_cfunction:
            return 0
        return rhs_type.same_c_signature_as_resolved_type(self, exact_semantics=False) \
            and not (self.nogil and not rhs_type.nogil)

    def declaration_code(self, entity_code,
                         for_display = 0, dll_linkage = None, pyrex = 0,
                         with_calling_convention = 1):
        arg_decl_list = []
        for arg in self.args[:len(self.args)-self.optional_arg_count]:
            arg_decl_list.append(
                arg.type.declaration_code("", for_display, pyrex = pyrex))
        if self.is_overridable:
            arg_decl_list.append("int %s" % Naming.skip_dispatch_cname)
        if self.optional_arg_count:
            arg_decl_list.append(self.op_arg_struct.declaration_code(Naming.optional_args_cname))
        if self.has_varargs:
            arg_decl_list.append("...")
        arg_decl_code = ", ".join(arg_decl_list)
        if not arg_decl_code and not pyrex:
            arg_decl_code = "void"
        trailer = ""
        if (pyrex or for_display) and not self.return_type.is_pyobject:
            if self.exception_value and self.exception_check:
                trailer = " except? %s" % self.exception_value
            elif self.exception_value:
                trailer = " except %s" % self.exception_value
            elif self.exception_check == '+':
                trailer = " except +"
            elif self.exception_check and for_display:
                # not spelled out by default, unless for human eyes
                trailer = " except *"
            if self.nogil:
                trailer += " nogil"
        if not with_calling_convention:
            cc = ''
        else:
            cc = self.calling_convention_prefix()
            if (not entity_code and cc) or entity_code.startswith("*"):
                entity_code = "(%s%s)" % (cc, entity_code)
                cc = ""
        if self.is_const_method:
            trailer += " const"
        return self.return_type.declaration_code(
            "%s%s(%s)%s" % (cc, entity_code, arg_decl_code, trailer),
            for_display, dll_linkage, pyrex)

    def function_header_code(self, func_name, arg_code):
        if self.is_const_method:
            trailer = " const"
        else:
            trailer = ""
        return "%s%s(%s)%s" % (self.calling_convention_prefix(),
            func_name, arg_code, trailer)

    def signature_string(self):
        s = self.empty_declaration_code()
        return s

    def signature_cast_string(self):
        s = self.declaration_code("(*)", with_calling_convention=False)
        return '(%s)' % s

    def specialize(self, values):
        result = CFuncType(self.return_type.specialize(values),
                           [arg.specialize(values) for arg in self.args],
                           has_varargs = self.has_varargs,
                           exception_value = self.exception_value,
                           exception_check = self.exception_check,
                           calling_convention = self.calling_convention,
                           nogil = self.nogil,
                           with_gil = self.with_gil,
                           is_overridable = self.is_overridable,
                           optional_arg_count = self.optional_arg_count,
                           is_const_method = self.is_const_method,
                           is_static_method = self.is_static_method,
                           templates = self.templates)

        result.from_fused = self.is_fused
        return result

    def opt_arg_cname(self, arg_name):
        return self.op_arg_struct.base_type.scope.lookup(arg_name).cname

    # Methods that deal with Fused Types
    # All but map_with_specific_entries should be called only on functions
    # with fused types (and not on their corresponding specific versions).

    def get_all_specialized_permutations(self, fused_types=None):
        """
        Permute all the types. For every specific instance of a fused type, we
        want all other specific instances of all other fused types.

        It returns an iterable of two-tuples of the cname that should prefix
        the cname of the function, and a dict mapping any fused types to their
        respective specific types.
        """
        assert self.is_fused

        if fused_types is None:
            fused_types = self.get_fused_types()

        return get_all_specialized_permutations(fused_types)

    def get_all_specialized_function_types(self):
        """
        Get all the specific function types of this one.
        """
        assert self.is_fused

        if self.entry.fused_cfunction:
            return [n.type for n in self.entry.fused_cfunction.nodes]
        elif self.cached_specialized_types is not None:
            return self.cached_specialized_types

        result = []
        permutations = self.get_all_specialized_permutations()

        new_cfunc_entries = []
        for cname, fused_to_specific in permutations:
            new_func_type = self.entry.type.specialize(fused_to_specific)

            if self.optional_arg_count:
                # Remember, this method is set by CFuncDeclaratorNode
                self.declare_opt_arg_struct(new_func_type, cname)

            new_entry = copy.deepcopy(self.entry)
            new_func_type.specialize_entry(new_entry, cname)

            new_entry.type = new_func_type
            new_func_type.entry = new_entry
            result.append(new_func_type)

            new_cfunc_entries.append(new_entry)

        cfunc_entries = self.entry.scope.cfunc_entries
        try:
            cindex = cfunc_entries.index(self.entry)
        except ValueError:
            cfunc_entries.extend(new_cfunc_entries)
        else:
            cfunc_entries[cindex:cindex+1] = new_cfunc_entries

        self.cached_specialized_types = result

        return result

    def get_fused_types(self, result=None, seen=None, subtypes=None):
        """Return fused types in the order they appear as parameter types"""
        return super(CFuncType, self).get_fused_types(result, seen,
                                                      subtypes=['args'])

    def specialize_entry(self, entry, cname):
        assert not self.is_fused
        specialize_entry(entry, cname)

    def can_coerce_to_pyobject(self, env):
        # duplicating the decisions from create_to_py_utility_code() here avoids writing out unused code
        if self.has_varargs or self.optional_arg_count:
            return False
        if self.to_py_function is not None:
            return self.to_py_function
        for arg in self.args:
            if not arg.type.is_pyobject and not arg.type.can_coerce_to_pyobject(env):
                return False
        if not self.return_type.is_pyobject and not self.return_type.can_coerce_to_pyobject(env):
            return False
        return True

    def create_to_py_utility_code(self, env):
        # FIXME: it seems we're trying to coerce in more cases than we should
        if self.to_py_function is not None:
            return self.to_py_function
        if not self.can_coerce_to_pyobject(env):
            return False
        from .UtilityCode import CythonUtilityCode
        safe_typename = re.sub('[^a-zA-Z0-9]', '__', self.declaration_code("", pyrex=1))
        to_py_function = "__Pyx_CFunc_%s_to_py" % safe_typename

        for arg in self.args:
            if not arg.type.is_pyobject and not arg.type.create_from_py_utility_code(env):
                return False
        if not self.return_type.is_pyobject and not self.return_type.create_to_py_utility_code(env):
            return False

        def declared_type(ctype):
            type_displayname = str(ctype.declaration_code("", for_display=True))
            if ctype.is_pyobject:
                arg_ctype = type_name = type_displayname
                if ctype.is_builtin_type:
                    arg_ctype = ctype.name
                elif not ctype.is_extension_type:
                    type_name = 'object'
                    type_displayname = None
                else:
                    type_displayname = repr(type_displayname)
            elif ctype is c_bint_type:
                type_name = arg_ctype = 'bint'
            else:
                type_name = arg_ctype = type_displayname
                if ctype is c_double_type:
                    type_displayname = 'float'
                else:
                    type_displayname = repr(type_displayname)
            return type_name, arg_ctype, type_displayname

        class Arg(object):
            def __init__(self, arg_name, arg_type):
                self.name = arg_name
                self.type = arg_type
                self.type_cname, self.ctype, self.type_displayname = declared_type(arg_type)

        if self.return_type.is_void:
            except_clause = 'except *'
        elif self.return_type.is_pyobject:
            except_clause = ''
        elif self.exception_value:
            except_clause = ('except? %s' if self.exception_check else 'except %s') % self.exception_value
        else:
            except_clause = 'except *'

        context = {
            'cname': to_py_function,
            'args': [Arg(arg.name or 'arg%s' % ix, arg.type) for ix, arg in enumerate(self.args)],
            'return_type': Arg('return', self.return_type),
            'except_clause': except_clause,
        }
        # FIXME: directives come from first defining environment and do not adapt for reuse
        env.use_utility_code(CythonUtilityCode.load(
            "cfunc.to_py", "CConvert.pyx",
            outer_module_scope=env.global_scope(),  # need access to types declared in module
            context=context, compiler_directives=dict(env.global_scope().directives)))
        self.to_py_function = to_py_function
        return True


def specialize_entry(entry, cname):
    """
    Specialize an entry of a copied fused function or method
    """
    entry.is_fused_specialized = True
    entry.name = get_fused_cname(cname, entry.name)

    if entry.is_cmethod:
        entry.cname = entry.name
        if entry.is_inherited:
            entry.cname = StringEncoding.EncodedString(
                    "%s.%s" % (Naming.obj_base_cname, entry.cname))
    else:
        entry.cname = get_fused_cname(cname, entry.cname)

    if entry.func_cname:
        entry.func_cname = get_fused_cname(cname, entry.func_cname)

def get_fused_cname(fused_cname, orig_cname):
    """
    Given the fused cname id and an original cname, return a specialized cname
    """
    assert fused_cname and orig_cname
    return StringEncoding.EncodedString('%s%s%s' % (Naming.fused_func_prefix,
                                                    fused_cname, orig_cname))

def unique(somelist):
    seen = set()
    result = []
    for obj in somelist:
        if obj not in seen:
            result.append(obj)
            seen.add(obj)

    return result

def get_all_specialized_permutations(fused_types):
    return _get_all_specialized_permutations(unique(fused_types))

def _get_all_specialized_permutations(fused_types, id="", f2s=()):
    fused_type, = fused_types[0].get_fused_types()
    result = []

    for newid, specific_type in enumerate(fused_type.types):
        # f2s = dict(f2s, **{ fused_type: specific_type })
        f2s = dict(f2s)
        f2s.update({ fused_type: specific_type })

        if id:
            cname = '%s_%s' % (id, newid)
        else:
            cname = str(newid)

        if len(fused_types) > 1:
            result.extend(_get_all_specialized_permutations(
                                            fused_types[1:], cname, f2s))
        else:
            result.append((cname, f2s))

    return result

def specialization_signature_string(fused_compound_type, fused_to_specific):
    """
    Return the signature for a specialization of a fused type. e.g.

        floating[:] ->
            'float' or 'double'

        cdef fused ft:
            float[:]
            double[:]

        ft ->
            'float[:]' or 'double[:]'

        integral func(floating) ->
            'int (*func)(float)' or ...
    """
    fused_types = fused_compound_type.get_fused_types()
    if len(fused_types) == 1:
        fused_type = fused_types[0]
    else:
        fused_type = fused_compound_type

    return fused_type.specialize(fused_to_specific).typeof_name()


def get_specialized_types(type):
    """
    Return a list of specialized types in their declared order.
    """
    assert type.is_fused

    if isinstance(type, FusedType):
        result = list(type.types)
        for specialized_type in result:
            specialized_type.specialization_string = specialized_type.typeof_name()
    else:
        result = []
        for cname, f2s in get_all_specialized_permutations(type.get_fused_types()):
            specialized_type = type.specialize(f2s)
            specialized_type.specialization_string = (
                            specialization_signature_string(type, f2s))
            result.append(specialized_type)

    return result


class CFuncTypeArg(BaseType):
    #  name       string
    #  cname      string
    #  type       PyrexType
    #  pos        source file position

    # FIXME: is this the right setup? should None be allowed here?
    not_none = False
    or_none = False
    accept_none = True
    accept_builtin_subtypes = False
    annotation = None

    subtypes = ['type']

    def __init__(self, name, type, pos, cname=None, annotation=None):
        self.name = name
        if cname is not None:
            self.cname = cname
        else:
            self.cname = Naming.var_prefix + name
        if annotation is not None:
            self.annotation = annotation
        self.type = type
        self.pos = pos
        self.needs_type_test = False # TODO: should these defaults be set in analyse_types()?

    def __repr__(self):
        return "%s:%s" % (self.name, repr(self.type))

    def declaration_code(self, for_display = 0):
        return self.type.declaration_code(self.cname, for_display)

    def specialize(self, values):
        return CFuncTypeArg(self.name, self.type.specialize(values), self.pos, self.cname)


class ToPyStructUtilityCode(object):

    requires = None

    def __init__(self, type, forward_decl, env):
        self.type = type
        self.header = "static PyObject* %s(%s)" % (type.to_py_function,
                                                   type.declaration_code('s'))
        self.forward_decl = forward_decl
        self.env = env

    def __eq__(self, other):
        return isinstance(other, ToPyStructUtilityCode) and self.header == other.header

    def __hash__(self):
        return hash(self.header)

    def get_tree(self, **kwargs):
        pass

    def put_code(self, output):
        code = output['utility_code_def']
        proto = output['utility_code_proto']

        code.putln("%s {" % self.header)
        code.putln("PyObject* res;")
        code.putln("PyObject* member;")
        code.putln("res = __Pyx_PyDict_NewPresized(%d); if (unlikely(!res)) return NULL;" %
                   len(self.type.scope.var_entries))
        for member in self.type.scope.var_entries:
            nameconst_cname = code.get_py_string_const(member.name, identifier=True)
            code.putln("%s; if (unlikely(!member)) goto bad;" % (
                member.type.to_py_call_code('s.%s' % member.cname, 'member', member.type)))
            code.putln("if (unlikely(PyDict_SetItem(res, %s, member) < 0)) goto bad;" % nameconst_cname)
            code.putln("Py_DECREF(member);")
        code.putln("return res;")
        code.putln("bad:")
        code.putln("Py_XDECREF(member);")
        code.putln("Py_DECREF(res);")
        code.putln("return NULL;")
        code.putln("}")

        # This is a bit of a hack, we need a forward declaration
        # due to the way things are ordered in the module...
        if self.forward_decl:
            proto.putln(self.type.empty_declaration_code() + ';')
        proto.putln(self.header + ";")

    def inject_tree_and_scope_into(self, module_node):
        pass


class CStructOrUnionType(CType):
    #  name          string
    #  cname         string
    #  kind          string              "struct" or "union"
    #  scope         StructOrUnionScope, or None if incomplete
    #  typedef_flag  boolean
    #  packed        boolean

    # entry          Entry

    is_struct_or_union = 1
    has_attributes = 1
    exception_check = True

    def __init__(self, name, kind, scope, typedef_flag, cname, packed=False):
        self.name = name
        self.cname = cname
        self.kind = kind
        self.scope = scope
        self.typedef_flag = typedef_flag
        self.is_struct = kind == 'struct'
        self.to_py_function = "%s_to_py_%s" % (
            Naming.convert_func_prefix, self.specialization_name())
        self.from_py_function = "%s_from_py_%s" % (
            Naming.convert_func_prefix, self.specialization_name())
        self.exception_check = True
        self._convert_to_py_code = None
        self._convert_from_py_code = None
        self.packed = packed

    def can_coerce_to_pyobject(self, env):
        if self._convert_to_py_code is False:
            return None  # tri-state-ish

        if env.outer_scope is None:
            return False

        if self._convert_to_py_code is None:
            is_union = not self.is_struct
            unsafe_union_types = set()
            safe_union_types = set()
            for member in self.scope.var_entries:
                member_type = member.type
                if not member_type.can_coerce_to_pyobject(env):
                    self.to_py_function = None
                    self._convert_to_py_code = False
                    return False
                if is_union:
                    if member_type.is_ptr or member_type.is_cpp_class:
                        unsafe_union_types.add(member_type)
                    else:
                        safe_union_types.add(member_type)

            if unsafe_union_types and (safe_union_types or len(unsafe_union_types) > 1):
                # unsafe mix of safe and unsafe to convert types
                self.from_py_function = None
                self._convert_from_py_code = False
                return False

        return True

    def create_to_py_utility_code(self, env):
        if not self.can_coerce_to_pyobject(env):
            return False

        if self._convert_to_py_code is None:
            for member in self.scope.var_entries:
                member.type.create_to_py_utility_code(env)
            forward_decl = self.entry.visibility != 'extern' and not self.typedef_flag
            self._convert_to_py_code = ToPyStructUtilityCode(self, forward_decl, env)

        env.use_utility_code(self._convert_to_py_code)
        return True

    def can_coerce_from_pyobject(self, env):
        if env.outer_scope is None or self._convert_from_py_code is False:
            return False
        for member in self.scope.var_entries:
            if not member.type.can_coerce_from_pyobject(env):
                return False
        return True

    def create_from_py_utility_code(self, env):
        if env.outer_scope is None:
            return False

        if self._convert_from_py_code is False:
            return None  # tri-state-ish

        if self._convert_from_py_code is None:
            if not self.scope.var_entries:
                # There are obviously missing fields; don't allow instantiation
                # where absolutely no content is provided.
                return False

            for member in self.scope.var_entries:
                if not member.type.create_from_py_utility_code(env):
                    self.from_py_function = None
                    self._convert_from_py_code = False
                    return False

            context = dict(
                struct_type=self,
                var_entries=self.scope.var_entries,
                funcname=self.from_py_function,
            )
            from .UtilityCode import CythonUtilityCode
            self._convert_from_py_code = CythonUtilityCode.load(
                "FromPyStructUtility" if self.is_struct else "FromPyUnionUtility",
                "CConvert.pyx",
                outer_module_scope=env.global_scope(),  # need access to types declared in module
                context=context)

        env.use_utility_code(self._convert_from_py_code)
        return True

    def __repr__(self):
        return "<CStructOrUnionType %s %s%s>" % (
            self.name, self.cname,
            ("", " typedef")[self.typedef_flag])

    def declaration_code(self, entity_code,
                         for_display=0, dll_linkage=None, pyrex=0):
        if pyrex or for_display:
            base_code = self.name
        else:
            if self.typedef_flag:
                base_code = self.cname
            else:
                base_code = "%s %s" % (self.kind, self.cname)
            base_code = public_decl(base_code, dll_linkage)
        return self.base_declaration_code(base_code, entity_code)

    def __eq__(self, other):
        try:
            return (isinstance(other, CStructOrUnionType) and
                    self.name == other.name)
        except AttributeError:
            return False

    def __lt__(self, other):
        try:
            return self.name < other.name
        except AttributeError:
            # this is arbitrary, but it makes sure we always have
            # *some* kind of order
            return False

    def __hash__(self):
        return hash(self.cname) ^ hash(self.kind)

    def is_complete(self):
        return self.scope is not None

    def attributes_known(self):
        return self.is_complete()

    def can_be_complex(self):
        # Does the struct consist of exactly two identical floats?
        fields = self.scope.var_entries
        if len(fields) != 2: return False
        a, b = fields
        return (a.type.is_float and b.type.is_float and
                a.type.empty_declaration_code() ==
                b.type.empty_declaration_code())

    def struct_nesting_depth(self):
        child_depths = [x.type.struct_nesting_depth()
                        for x in self.scope.var_entries]
        return max(child_depths) + 1

    def cast_code(self, expr_code):
        if self.is_struct:
            return expr_code
        return super(CStructOrUnionType, self).cast_code(expr_code)

cpp_string_conversions = ("std::string",)

builtin_cpp_conversions = {
    # type                element template params
    "std::pair":          2,
    "std::vector":        1,
    "std::list":          1,
    "std::set":           1,
    "std::unordered_set": 1,
    "std::map":           2,
    "std::unordered_map": 2,
    "std::complex":       1,
}

class CppClassType(CType):
    #  name          string
    #  cname         string
    #  scope         CppClassScope
    #  templates     [string] or None

    is_cpp_class = 1
    has_attributes = 1
    exception_check = True
    namespace = None

    # For struct-like declaration.
    kind = "struct"
    packed = False
    typedef_flag = False

    subtypes = ['templates']

    def __init__(self, name, scope, cname, base_classes, templates=None, template_type=None):
        self.name = name
        self.cname = cname
        self.scope = scope
        self.base_classes = base_classes
        self.operators = []
        self.templates = templates
        self.template_type = template_type
        self.num_optional_templates = sum(is_optional_template_param(T) for T in templates or ())
        if templates:
            self.specializations = {tuple(zip(templates, templates)): self}
        else:
            self.specializations = {}
        self.is_cpp_string = cname in cpp_string_conversions

    def use_conversion_utility(self, from_or_to):
        pass

    def maybe_unordered(self):
        if 'unordered' in self.cname:
            return 'unordered_'
        else:
            return ''

    def can_coerce_from_pyobject(self, env):
        if self.cname in builtin_cpp_conversions:
            template_count = builtin_cpp_conversions[self.cname]
            for ix, T in enumerate(self.templates or []):
                if ix >= template_count:
                    break
                if T.is_pyobject or not T.can_coerce_from_pyobject(env):
                    return False
            return True
        elif self.cname in cpp_string_conversions:
            return True
        return False

    def create_from_py_utility_code(self, env):
        if self.from_py_function is not None:
            return True
        if self.cname in builtin_cpp_conversions or self.cname in cpp_string_conversions:
            X = "XYZABC"
            tags = []
            context = {}
            for ix, T in enumerate(self.templates or []):
                if ix >= builtin_cpp_conversions[self.cname]:
                    break
                if T.is_pyobject or not T.create_from_py_utility_code(env):
                    return False
                tags.append(T.specialization_name())
                context[X[ix]] = T

            if self.cname in cpp_string_conversions:
                cls = 'string'
                tags = type_identifier(self),
            else:
                cls = self.cname[5:]
            cname = '__pyx_convert_%s_from_py_%s' % (cls, '__and_'.join(tags))
            context.update({
                'cname': cname,
                'maybe_unordered': self.maybe_unordered(),
                'type': self.cname,
            })
            from .UtilityCode import CythonUtilityCode
            env.use_utility_code(CythonUtilityCode.load(
                cls.replace('unordered_', '') + ".from_py", "CppConvert.pyx",
                context=context, compiler_directives=env.directives))
            self.from_py_function = cname
            return True

    def can_coerce_to_pyobject(self, env):
        if self.cname in builtin_cpp_conversions or self.cname in cpp_string_conversions:
            for ix, T in enumerate(self.templates or []):
                if ix >= builtin_cpp_conversions[self.cname]:
                    break
                if T.is_pyobject or not T.can_coerce_to_pyobject(env):
                    return False
            return True


    def create_to_py_utility_code(self, env):
        if self.to_py_function is not None:
            return True
        if self.cname in builtin_cpp_conversions or self.cname in cpp_string_conversions:
            X = "XYZABC"
            tags = []
            context = {}
            for ix, T in enumerate(self.templates or []):
                if ix >= builtin_cpp_conversions[self.cname]:
                    break
                if not T.create_to_py_utility_code(env):
                    return False
                tags.append(T.specialization_name())
                context[X[ix]] = T

            if self.cname in cpp_string_conversions:
                cls = 'string'
                prefix = 'PyObject_'  # gets specialised by explicit type casts in CoerceToPyTypeNode
                tags = type_identifier(self),
            else:
                cls = self.cname[5:]
                prefix = ''
            cname = "__pyx_convert_%s%s_to_py_%s" % (prefix, cls, "____".join(tags))
            context.update({
                'cname': cname,
                'maybe_unordered': self.maybe_unordered(),
                'type': self.cname,
            })
            from .UtilityCode import CythonUtilityCode
            env.use_utility_code(CythonUtilityCode.load(
                cls.replace('unordered_', '') + ".to_py", "CppConvert.pyx",
                context=context, compiler_directives=env.directives))
            self.to_py_function = cname
            return True

    def is_template_type(self):
        return self.templates is not None and self.template_type is None

    def get_fused_types(self, result=None, seen=None):
        if result is None:
            result = []
            seen = set()
        if self.namespace:
            self.namespace.get_fused_types(result, seen)
        if self.templates:
            for T in self.templates:
                T.get_fused_types(result, seen)
        return result

    def specialize_here(self, pos, template_values=None):
        if not self.is_template_type():
            error(pos, "'%s' type is not a template" % self)
            return error_type
        if len(self.templates) - self.num_optional_templates <= len(template_values) < len(self.templates):
            num_defaults = len(self.templates) - len(template_values)
            partial_specialization = self.declaration_code('', template_params=template_values)
            # Most of the time we don't need to declare anything typed to these
            # default template arguments, but when we do there's no way in C++
            # to reference this directly.  However, it is common convention to
            # provide a typedef in the template class that resolves to each
            # template type.  For now, allow the user to specify this name as
            # the template parameter.
            # TODO: Allow typedefs in cpp classes and search for it in this
            # classes scope as a concrete name we could use.
            template_values = template_values + [
                TemplatePlaceholderType(
                    "%s::%s" % (partial_specialization, param.name), True)
                for param in self.templates[-num_defaults:]]
        if len(self.templates) != len(template_values):
            error(pos, "%s templated type receives %d arguments, got %d" %
                  (self.name, len(self.templates), len(template_values)))
            return error_type
        has_object_template_param = False
        for value in template_values:
            if value.is_pyobject:
                has_object_template_param = True
                error(pos,
                      "Python object type '%s' cannot be used as a template argument" % value)
        if has_object_template_param:
            return error_type
        return self.specialize(dict(zip(self.templates, template_values)))

    def specialize(self, values):
        if not self.templates and not self.namespace:
            return self
        if self.templates is None:
            self.templates = []
        key = tuple(values.items())
        if key in self.specializations:
            return self.specializations[key]
        template_values = [t.specialize(values) for t in self.templates]
        specialized = self.specializations[key] = \
            CppClassType(self.name, None, self.cname, [], template_values, template_type=self)
        # Need to do these *after* self.specializations[key] is set
        # to avoid infinite recursion on circular references.
        specialized.base_classes = [b.specialize(values) for b in self.base_classes]
        if self.namespace is not None:
            specialized.namespace = self.namespace.specialize(values)
        specialized.scope = self.scope.specialize(values, specialized)
        if self.cname == 'std::vector':
          # vector<bool> is special cased in the C++ standard, and its
          # accessors do not necessarily return references to the underlying
          # elements (which may be bit-packed).
          # http://www.cplusplus.com/reference/vector/vector-bool/
          # Here we pretend that the various methods return bool values
          # (as the actual returned values are coercable to such, and
          # we don't support call expressions as lvalues).
          T = values.get(self.templates[0], None)
          if T and not T.is_fused and T.empty_declaration_code() == 'bool':
            for bit_ref_returner in ('at', 'back', 'front'):
              if bit_ref_returner in specialized.scope.entries:
                specialized.scope.entries[bit_ref_returner].type.return_type = T
        return specialized

    def deduce_template_params(self, actual):
        if actual.is_const:
            actual = actual.const_base_type
        if actual.is_reference:
            actual = actual.ref_base_type
        if self == actual:
            return {}
        elif actual.is_cpp_class:
            self_template_type = self
            while getattr(self_template_type, 'template_type', None):
                self_template_type = self_template_type.template_type
            def all_bases(cls):
                yield cls
                for parent in cls.base_classes:
                    for base in all_bases(parent):
                        yield base
            for actual_base in all_bases(actual):
                template_type = actual_base
                while getattr(template_type, 'template_type', None):
                    template_type = template_type.template_type
                    if (self_template_type.empty_declaration_code()
                            == template_type.empty_declaration_code()):
                        return reduce(
                            merge_template_deductions,
                            [formal_param.deduce_template_params(actual_param)
                             for (formal_param, actual_param)
                             in zip(self.templates, actual_base.templates)],
                            {})
        else:
            return {}

    def declaration_code(self, entity_code,
            for_display = 0, dll_linkage = None, pyrex = 0,
            template_params = None):
        if template_params is None:
            template_params = self.templates
        if self.templates:
            template_strings = [param.declaration_code('', for_display, None, pyrex)
                                for param in template_params
                                if not is_optional_template_param(param) and not param.is_fused]
            if for_display:
                brackets = "[%s]"
            else:
                brackets = "<%s> "
            templates = brackets % ",".join(template_strings)
        else:
            templates = ""
        if pyrex or for_display:
            base_code = "%s%s" % (self.name, templates)
        else:
            base_code = "%s%s" % (self.cname, templates)
            if self.namespace is not None:
                base_code = "%s::%s" % (self.namespace.empty_declaration_code(), base_code)
            base_code = public_decl(base_code, dll_linkage)
        return self.base_declaration_code(base_code, entity_code)

    def is_subclass(self, other_type):
        if self.same_as_resolved_type(other_type):
            return 1
        for base_class in self.base_classes:
            if base_class.is_subclass(other_type):
                return 1
        return 0

    def subclass_dist(self, super_type):
        if self.same_as_resolved_type(super_type):
            return 0
        elif not self.base_classes:
            return float('inf')
        else:
            return 1 + min(b.subclass_dist(super_type) for b in self.base_classes)

    def same_as_resolved_type(self, other_type):
        if other_type.is_cpp_class:
            if self == other_type:
                return 1
            # This messy logic is needed due to GH Issue #1852.
            elif (self.cname == other_type.cname and
                    (self.template_type and other_type.template_type
                     or self.templates
                     or other_type.templates)):
                if self.templates == other_type.templates:
                    return 1
                for t1, t2 in zip(self.templates, other_type.templates):
                    if is_optional_template_param(t1) and is_optional_template_param(t2):
                        break
                    if not t1.same_as_resolved_type(t2):
                        return 0
                return 1
        return 0

    def assignable_from_resolved_type(self, other_type):
        # TODO: handle operator=(...) here?
        if other_type is error_type:
            return True
        elif other_type.is_cpp_class:
            return other_type.is_subclass(self)
        elif other_type.is_string and self.cname in cpp_string_conversions:
            return True

    def attributes_known(self):
        return self.scope is not None

    def find_cpp_operation_type(self, operator, operand_type=None):
        operands = [self]
        if operand_type is not None:
            operands.append(operand_type)
        # pos == None => no errors
        operator_entry = self.scope.lookup_operator_for_types(None, operator, operands)
        if not operator_entry:
            return None
        func_type = operator_entry.type
        if func_type.is_ptr:
            func_type = func_type.base_type
        return func_type.return_type

    def get_constructor(self, pos):
        constructor = self.scope.lookup('<init>')
        if constructor is not None:
            return constructor

        # Otherwise: automatically declare no-args default constructor.
        # Make it "nogil" if the base classes allow it.
        nogil = True
        for base in self.base_classes:
            base_constructor = base.scope.lookup('<init>')
            if base_constructor and not base_constructor.type.nogil:
                nogil = False
                break

        func_type = CFuncType(self, [], exception_check='+', nogil=nogil)
        return self.scope.declare_cfunction(u'<init>', func_type, pos)

    def check_nullary_constructor(self, pos, msg="stack allocated"):
        constructor = self.scope.lookup(u'<init>')
        if constructor is not None and best_match([], constructor.all_alternatives()) is None:
            error(pos, "C++ class must have a nullary constructor to be %s" % msg)


class TemplatePlaceholderType(CType):

    def __init__(self, name, optional=False):
        self.name = name
        self.optional = optional

    def declaration_code(self, entity_code,
            for_display = 0, dll_linkage = None, pyrex = 0):
        if entity_code:
            return self.name + " " + entity_code
        else:
            return self.name

    def specialize(self, values):
        if self in values:
            return values[self]
        else:
            return self

    def deduce_template_params(self, actual):
        return {self: actual}

    def same_as_resolved_type(self, other_type):
        if isinstance(other_type, TemplatePlaceholderType):
            return self.name == other_type.name
        else:
            return 0

    def __hash__(self):
        return hash(self.name)

    def __cmp__(self, other):
        if isinstance(other, TemplatePlaceholderType):
            return cmp(self.name, other.name)
        else:
            return cmp(type(self), type(other))

    def __eq__(self, other):
        if isinstance(other, TemplatePlaceholderType):
            return self.name == other.name
        else:
            return False

def is_optional_template_param(type):
    return isinstance(type, TemplatePlaceholderType) and type.optional


class CEnumType(CIntLike, CType):
    #  name           string
    #  cname          string or None
    #  typedef_flag   boolean
    #  values         [string], populated during declaration analysis

    is_enum = 1
    signed = 1
    rank = -1 # Ranks below any integer type

    def __init__(self, name, cname, typedef_flag, namespace=None):
        self.name = name
        self.cname = cname
        self.values = []
        self.typedef_flag = typedef_flag
        self.namespace = namespace
        self.default_value = "(%s) 0" % self.empty_declaration_code()

    def __str__(self):
        return self.name

    def __repr__(self):
        return "<CEnumType %s %s%s>" % (self.name, self.cname,
            ("", " typedef")[self.typedef_flag])

    def declaration_code(self, entity_code,
            for_display = 0, dll_linkage = None, pyrex = 0):
        if pyrex or for_display:
            base_code = self.name
        else:
            if self.namespace:
                base_code = "%s::%s" % (
                    self.namespace.empty_declaration_code(), self.cname)
            elif self.typedef_flag:
                base_code = self.cname
            else:
                base_code = "enum %s" % self.cname
            base_code = public_decl(base_code, dll_linkage)
        return self.base_declaration_code(base_code, entity_code)

    def specialize(self, values):
        if self.namespace:
            namespace = self.namespace.specialize(values)
            if namespace != self.namespace:
                return CEnumType(
                    self.name, self.cname, self.typedef_flag, namespace)
        return self

    def create_type_wrapper(self, env):
        from .UtilityCode import CythonUtilityCode
        env.use_utility_code(CythonUtilityCode.load(
            "EnumType", "CpdefEnums.pyx",
            context={"name": self.name,
                     "items": tuple(self.values)},
            outer_module_scope=env.global_scope()))


class CTupleType(CType):
    # components [PyrexType]

    is_ctuple = True

    def __init__(self, cname, components):
        self.cname = cname
        self.components = components
        self.size = len(components)
        self.to_py_function = "%s_to_py_%s" % (Naming.convert_func_prefix, self.cname)
        self.from_py_function = "%s_from_py_%s" % (Naming.convert_func_prefix, self.cname)
        self.exception_check = True
        self._convert_to_py_code = None
        self._convert_from_py_code = None

    def __str__(self):
        return "(%s)" % ", ".join(str(c) for c in self.components)

    def declaration_code(self, entity_code,
            for_display = 0, dll_linkage = None, pyrex = 0):
        if pyrex or for_display:
            return str(self)
        else:
            return self.base_declaration_code(self.cname, entity_code)

    def can_coerce_to_pyobject(self, env):
        for component in self.components:
            if not component.can_coerce_to_pyobject(env):
                return False
        return True

    def can_coerce_from_pyobject(self, env):
        for component in self.components:
            if not component.can_coerce_from_pyobject(env):
                return False
        return True

    def create_to_py_utility_code(self, env):
        if self._convert_to_py_code is False:
            return None  # tri-state-ish

        if self._convert_to_py_code is None:
            for component in self.components:
                if not component.create_to_py_utility_code(env):
                    self.to_py_function = None
                    self._convert_to_py_code = False
                    return False

            context = dict(
                struct_type_decl=self.empty_declaration_code(),
                components=self.components,
                funcname=self.to_py_function,
                size=len(self.components)
            )
            self._convert_to_py_code = TempitaUtilityCode.load(
                "ToPyCTupleUtility", "TypeConversion.c", context=context)

        env.use_utility_code(self._convert_to_py_code)
        return True

    def create_from_py_utility_code(self, env):
        if self._convert_from_py_code is False:
            return None  # tri-state-ish

        if self._convert_from_py_code is None:
            for component in self.components:
                if not component.create_from_py_utility_code(env):
                    self.from_py_function = None
                    self._convert_from_py_code = False
                    return False

            context = dict(
                struct_type_decl=self.empty_declaration_code(),
                components=self.components,
                funcname=self.from_py_function,
                size=len(self.components)
            )
            self._convert_from_py_code = TempitaUtilityCode.load(
                "FromPyCTupleUtility", "TypeConversion.c", context=context)

        env.use_utility_code(self._convert_from_py_code)
        return True

    def cast_code(self, expr_code):
        return expr_code


def c_tuple_type(components):
    components = tuple(components)
    cname = Naming.ctuple_type_prefix + type_list_identifier(components)
    tuple_type = CTupleType(cname, components)
    return tuple_type


class UnspecifiedType(PyrexType):
    # Used as a placeholder until the type can be determined.

    is_unspecified = 1

    def declaration_code(self, entity_code,
            for_display = 0, dll_linkage = None, pyrex = 0):
        return "<unspecified>"

    def same_as_resolved_type(self, other_type):
        return False


class ErrorType(PyrexType):
    # Used to prevent propagation of error messages.

    is_error = 1
    exception_value = "0"
    exception_check    = 0
    to_py_function = "dummy"
    from_py_function = "dummy"

    def create_to_py_utility_code(self, env):
        return True

    def create_from_py_utility_code(self, env):
        return True

    def declaration_code(self, entity_code,
            for_display = 0, dll_linkage = None, pyrex = 0):
        return "<error>"

    def same_as_resolved_type(self, other_type):
        return 1

    def error_condition(self, result_code):
        return "dummy"


rank_to_type_name = (
    "char",         # 0
    "short",        # 1
    "int",          # 2
    "long",         # 3
    "PY_LONG_LONG", # 4
    "float",        # 5
    "double",       # 6
    "long double",  # 7
)

_rank_to_type_name = list(rank_to_type_name)
RANK_INT  = _rank_to_type_name.index('int')
RANK_LONG = _rank_to_type_name.index('long')
RANK_FLOAT = _rank_to_type_name.index('float')
UNSIGNED = 0
SIGNED = 2

error_type =    ErrorType()
unspecified_type = UnspecifiedType()

py_object_type = PyObjectType()

c_void_type =        CVoidType()

c_uchar_type =       CIntType(0, UNSIGNED)
c_ushort_type =      CIntType(1, UNSIGNED)
c_uint_type =        CIntType(2, UNSIGNED)
c_ulong_type =       CIntType(3, UNSIGNED)
c_ulonglong_type =   CIntType(4, UNSIGNED)

c_char_type =        CIntType(0)
c_short_type =       CIntType(1)
c_int_type =         CIntType(2)
c_long_type =        CIntType(3)
c_longlong_type =    CIntType(4)

c_schar_type =       CIntType(0, SIGNED)
c_sshort_type =      CIntType(1, SIGNED)
c_sint_type =        CIntType(2, SIGNED)
c_slong_type =       CIntType(3, SIGNED)
c_slonglong_type =   CIntType(4, SIGNED)

c_float_type =       CFloatType(5, math_h_modifier='f')
c_double_type =      CFloatType(6)
c_longdouble_type =  CFloatType(7, math_h_modifier='l')

c_float_complex_type =      CComplexType(c_float_type)
c_double_complex_type =     CComplexType(c_double_type)
c_longdouble_complex_type = CComplexType(c_longdouble_type)

c_anon_enum_type =   CAnonEnumType(-1)
c_returncode_type =  CReturnCodeType(RANK_INT)
c_bint_type =        CBIntType(RANK_INT)
c_py_unicode_type =  CPyUnicodeIntType(RANK_INT-0.5, UNSIGNED)
c_py_ucs4_type =     CPyUCS4IntType(RANK_LONG-0.5, UNSIGNED)
c_py_hash_t_type =   CPyHashTType(RANK_LONG+0.5, SIGNED)
c_py_ssize_t_type =  CPySSizeTType(RANK_LONG+0.5, SIGNED)
c_ssize_t_type =     CSSizeTType(RANK_LONG+0.5, SIGNED)
c_size_t_type =      CSizeTType(RANK_LONG+0.5, UNSIGNED)
c_ptrdiff_t_type =   CPtrdiffTType(RANK_LONG+0.75, SIGNED)

c_null_ptr_type =     CNullPtrType(c_void_type)
c_void_ptr_type =     CPtrType(c_void_type)
c_void_ptr_ptr_type = CPtrType(c_void_ptr_type)
c_char_ptr_type =     CPtrType(c_char_type)
c_const_char_ptr_type = CPtrType(CConstType(c_char_type))
c_uchar_ptr_type =    CPtrType(c_uchar_type)
c_const_uchar_ptr_type = CPtrType(CConstType(c_uchar_type))
c_char_ptr_ptr_type = CPtrType(c_char_ptr_type)
c_int_ptr_type =      CPtrType(c_int_type)
c_py_unicode_ptr_type = CPtrType(c_py_unicode_type)
c_const_py_unicode_ptr_type = CPtrType(CConstType(c_py_unicode_type))
c_py_ssize_t_ptr_type =  CPtrType(c_py_ssize_t_type)
c_ssize_t_ptr_type =  CPtrType(c_ssize_t_type)
c_size_t_ptr_type =  CPtrType(c_size_t_type)

# GIL state
c_gilstate_type = CEnumType("PyGILState_STATE", "PyGILState_STATE", True)
c_threadstate_type = CStructOrUnionType("PyThreadState", "struct", None, 1, "PyThreadState")
c_threadstate_ptr_type = CPtrType(c_threadstate_type)

# PEP-539 "Py_tss_t" type
c_pytss_t_type = CPyTSSTType()

# the Py_buffer type is defined in Builtin.py
c_py_buffer_type = CStructOrUnionType("Py_buffer", "struct", None, 1, "Py_buffer")
c_py_buffer_ptr_type = CPtrType(c_py_buffer_type)

# Not sure whether the unsigned versions and 'long long' should be in there
# long long requires C99 and might be slow, and would always get preferred
# when specialization happens through calling and not indexing
cy_integral_type = FusedType([c_short_type, c_int_type, c_long_type],
                             name="integral")
# Omitting long double as it might be slow
cy_floating_type = FusedType([c_float_type, c_double_type], name="floating")
cy_numeric_type = FusedType([c_short_type,
                             c_int_type,
                             c_long_type,
                             c_float_type,
                             c_double_type,
                             c_float_complex_type,
                             c_double_complex_type], name="numeric")

# buffer-related structs
c_buf_diminfo_type =  CStructOrUnionType("__Pyx_Buf_DimInfo", "struct",
                                      None, 1, "__Pyx_Buf_DimInfo")
c_pyx_buffer_type = CStructOrUnionType("__Pyx_Buffer", "struct", None, 1, "__Pyx_Buffer")
c_pyx_buffer_ptr_type = CPtrType(c_pyx_buffer_type)
c_pyx_buffer_nd_type = CStructOrUnionType("__Pyx_LocalBuf_ND", "struct",
                                      None, 1, "__Pyx_LocalBuf_ND")

cython_memoryview_type = CStructOrUnionType("__pyx_memoryview_obj", "struct",
                                      None, 0, "__pyx_memoryview_obj")

memoryviewslice_type = CStructOrUnionType("memoryviewslice", "struct",
                                          None, 1, "__Pyx_memviewslice")

modifiers_and_name_to_type = {
    #(signed, longness, name) : type
    (0,  0, "char"): c_uchar_type,
    (1,  0, "char"): c_char_type,
    (2,  0, "char"): c_schar_type,

    (0, -1, "int"): c_ushort_type,
    (0,  0, "int"): c_uint_type,
    (0,  1, "int"): c_ulong_type,
    (0,  2, "int"): c_ulonglong_type,

    (1, -1, "int"): c_short_type,
    (1,  0, "int"): c_int_type,
    (1,  1, "int"): c_long_type,
    (1,  2, "int"): c_longlong_type,

    (2, -1, "int"): c_sshort_type,
    (2,  0, "int"): c_sint_type,
    (2,  1, "int"): c_slong_type,
    (2,  2, "int"): c_slonglong_type,

    (1,  0, "float"):  c_float_type,
    (1,  0, "double"): c_double_type,
    (1,  1, "double"): c_longdouble_type,

    (1,  0, "complex"):  c_double_complex_type,  # C: float, Python: double => Python wins
    (1,  0, "floatcomplex"):  c_float_complex_type,
    (1,  0, "doublecomplex"): c_double_complex_type,
    (1,  1, "doublecomplex"): c_longdouble_complex_type,

    #
    (1,  0, "void"): c_void_type,
    (1,  0, "Py_tss_t"): c_pytss_t_type,

    (1,  0, "bint"):       c_bint_type,
    (0,  0, "Py_UNICODE"): c_py_unicode_type,
    (0,  0, "Py_UCS4"):    c_py_ucs4_type,
    (2,  0, "Py_hash_t"):  c_py_hash_t_type,
    (2,  0, "Py_ssize_t"): c_py_ssize_t_type,
    (2,  0, "ssize_t") :   c_ssize_t_type,
    (0,  0, "size_t") :    c_size_t_type,
    (2,  0, "ptrdiff_t") : c_ptrdiff_t_type,

    (1,  0, "object"): py_object_type,
}

def is_promotion(src_type, dst_type):
    # It's hard to find a hard definition of promotion, but empirical
    # evidence suggests that the below is all that's allowed.
    if src_type.is_numeric:
        if dst_type.same_as(c_int_type):
            unsigned = (not src_type.signed)
            return (src_type.is_enum or
                    (src_type.is_int and
                     unsigned + src_type.rank < dst_type.rank))
        elif dst_type.same_as(c_double_type):
            return src_type.is_float and src_type.rank <= dst_type.rank
    return False

def best_match(arg_types, functions, pos=None, env=None, args=None):
    """
    Given a list args of arguments and a list of functions, choose one
    to call which seems to be the "best" fit for this list of arguments.
    This function is used, e.g., when deciding which overloaded method
    to dispatch for C++ classes.

    We first eliminate functions based on arity, and if only one
    function has the correct arity, we return it. Otherwise, we weight
    functions based on how much work must be done to convert the
    arguments, with the following priorities:
      * identical types or pointers to identical types
      * promotions
      * non-Python types
    That is, we prefer functions where no arguments need converted,
    and failing that, functions where only promotions are required, and
    so on.

    If no function is deemed a good fit, or if two or more functions have
    the same weight, we return None (as there is no best match). If pos
    is not None, we also generate an error.
    """
    # TODO: args should be a list of types, not a list of Nodes.
    actual_nargs = len(arg_types)

    candidates = []
    errors = []
    for func in functions:
        error_mesg = ""
        func_type = func.type
        if func_type.is_ptr:
            func_type = func_type.base_type
        # Check function type
        if not func_type.is_cfunction:
            if not func_type.is_error and pos is not None:
                error_mesg = "Calling non-function type '%s'" % func_type
            errors.append((func, error_mesg))
            continue
        # Check no. of args
        max_nargs = len(func_type.args)
        min_nargs = max_nargs - func_type.optional_arg_count
        if actual_nargs < min_nargs or \
            (not func_type.has_varargs and actual_nargs > max_nargs):
            if max_nargs == min_nargs and not func_type.has_varargs:
                expectation = max_nargs
            elif actual_nargs < min_nargs:
                expectation = "at least %s" % min_nargs
            else:
                expectation = "at most %s" % max_nargs
            error_mesg = "Call with wrong number of arguments (expected %s, got %s)" \
                         % (expectation, actual_nargs)
            errors.append((func, error_mesg))
            continue
        if func_type.templates:
            deductions = reduce(
                merge_template_deductions,
                [pattern.type.deduce_template_params(actual) for (pattern, actual) in zip(func_type.args, arg_types)],
                {})
            if deductions is None:
                errors.append((func, "Unable to deduce type parameters for %s given (%s)" % (func_type, ', '.join(map(str, arg_types)))))
            elif len(deductions) < len(func_type.templates):
                errors.append((func, "Unable to deduce type parameter %s" % (
                    ", ".join([param.name for param in set(func_type.templates) - set(deductions.keys())]))))
            else:
                type_list = [deductions[param] for param in func_type.templates]
                from .Symtab import Entry
                specialization = Entry(
                    name = func.name + "[%s]" % ",".join([str(t) for t in type_list]),
                    cname = func.cname + "<%s>" % ",".join([t.empty_declaration_code() for t in type_list]),
                    type = func_type.specialize(deductions),
                    pos = func.pos)
                candidates.append((specialization, specialization.type))
        else:
            candidates.append((func, func_type))

    # Optimize the most common case of no overloading...
    if len(candidates) == 1:
        return candidates[0][0]
    elif len(candidates) == 0:
        if pos is not None:
            func, errmsg = errors[0]
            if len(errors) == 1 or [1 for func, e in errors if e == errmsg]:
                error(pos, errmsg)
            else:
                error(pos, "no suitable method found")
        return None

    possibilities = []
    bad_types = []
    needed_coercions = {}

    for index, (func, func_type) in enumerate(candidates):
        score = [0,0,0,0,0,0,0]
        for i in range(min(actual_nargs, len(func_type.args))):
            src_type = arg_types[i]
            dst_type = func_type.args[i].type

            assignable = dst_type.assignable_from(src_type)

            # Now take care of unprefixed string literals. So when you call a cdef
            # function that takes a char *, the coercion will mean that the
            # type will simply become bytes. We need to do this coercion
            # manually for overloaded and fused functions
            if not assignable:
                c_src_type = None
                if src_type.is_pyobject:
                    if src_type.is_builtin_type and src_type.name == 'str' and dst_type.resolve().is_string:
                        c_src_type = dst_type.resolve()
                    else:
                        c_src_type = src_type.default_coerced_ctype()
                elif src_type.is_pythran_expr:
                        c_src_type = src_type.org_buffer

                if c_src_type is not None:
                    assignable = dst_type.assignable_from(c_src_type)
                    if assignable:
                        src_type = c_src_type
                        needed_coercions[func] = (i, dst_type)

            if assignable:
                if src_type == dst_type or dst_type.same_as(src_type):
                    pass  # score 0
                elif func_type.is_strict_signature:
                    break  # exact match requested but not found
                elif is_promotion(src_type, dst_type):
                    score[2] += 1
                elif ((src_type.is_int and dst_type.is_int) or
                      (src_type.is_float and dst_type.is_float)):
                    score[2] += abs(dst_type.rank + (not dst_type.signed) -
                                    (src_type.rank + (not src_type.signed))) + 1
                elif dst_type.is_ptr and src_type.is_ptr:
                    if dst_type.base_type == c_void_type:
                        score[4] += 1
                    elif src_type.base_type.is_cpp_class and src_type.base_type.is_subclass(dst_type.base_type):
                        score[6] += src_type.base_type.subclass_dist(dst_type.base_type)
                    else:
                        score[5] += 1
                elif not src_type.is_pyobject:
                    score[1] += 1
                else:
                    score[0] += 1
            else:
                error_mesg = "Invalid conversion from '%s' to '%s'" % (src_type, dst_type)
                bad_types.append((func, error_mesg))
                break
        else:
            possibilities.append((score, index, func))  # so we can sort it

    if possibilities:
        possibilities.sort()
        if len(possibilities) > 1:
            score1 = possibilities[0][0]
            score2 = possibilities[1][0]
            if score1 == score2:
                if pos is not None:
                    error(pos, "ambiguous overloaded method")
                return None

        function = possibilities[0][-1]

        if function in needed_coercions and env:
            arg_i, coerce_to_type = needed_coercions[function]
            args[arg_i] = args[arg_i].coerce_to(coerce_to_type, env)

        return function

    if pos is not None:
        if len(bad_types) == 1:
            error(pos, bad_types[0][1])
        else:
            error(pos, "no suitable method found")

    return None

def merge_template_deductions(a, b):
    if a is None or b is None:
        return None
    all = a
    for param, value in b.items():
        if param in all:
            if a[param] != b[param]:
                return None
        else:
            all[param] = value
    return all


def widest_numeric_type(type1, type2):
    """Given two numeric types, return the narrowest type encompassing both of them.
    """
    if type1.is_reference:
        type1 = type1.ref_base_type
    if type2.is_reference:
        type2 = type2.ref_base_type
    if type1.is_const:
        type1 = type1.const_base_type
    if type2.is_const:
        type2 = type2.const_base_type
    if type1 == type2:
        widest_type = type1
    elif type1.is_complex or type2.is_complex:
        def real_type(ntype):
            if ntype.is_complex:
                return ntype.real_type
            return ntype
        widest_type = CComplexType(
            widest_numeric_type(
                real_type(type1),
                real_type(type2)))
    elif type1.is_enum and type2.is_enum:
        widest_type = c_int_type
    elif type1.rank < type2.rank:
        widest_type = type2
    elif type1.rank > type2.rank:
        widest_type = type1
    elif type1.signed < type2.signed:
        widest_type = type1
    elif type1.signed > type2.signed:
        widest_type = type2
    elif type1.is_typedef > type2.is_typedef:
        widest_type = type1
    else:
        widest_type = type2
    return widest_type


def numeric_type_fits(small_type, large_type):
    return widest_numeric_type(small_type, large_type) == large_type


def independent_spanning_type(type1, type2):
    # Return a type assignable independently from both type1 and
    # type2, but do not require any interoperability between the two.
    # For example, in "True * 2", it is safe to assume an integer
    # result type (so spanning_type() will do the right thing),
    # whereas "x = True or 2" must evaluate to a type that can hold
    # both a boolean value and an integer, so this function works
    # better.
    if type1.is_reference ^ type2.is_reference:
        if type1.is_reference:
            type1 = type1.ref_base_type
        else:
            type2 = type2.ref_base_type
    if type1 == type2:
        return type1
    elif (type1 is c_bint_type or type2 is c_bint_type) and (type1.is_numeric and type2.is_numeric):
        # special case: if one of the results is a bint and the other
        # is another C integer, we must prevent returning a numeric
        # type so that we do not lose the ability to coerce to a
        # Python bool if we have to.
        return py_object_type
    span_type = _spanning_type(type1, type2)
    if span_type is None:
        return error_type
    return span_type

def spanning_type(type1, type2):
    # Return a type assignable from both type1 and type2, or
    # py_object_type if no better type is found.  Assumes that the
    # code that calls this will try a coercion afterwards, which will
    # fail if the types cannot actually coerce to a py_object_type.
    if type1 == type2:
        return type1
    elif type1 is py_object_type or type2 is py_object_type:
        return py_object_type
    elif type1 is c_py_unicode_type or type2 is c_py_unicode_type:
        # Py_UNICODE behaves more like a string than an int
        return py_object_type
    span_type = _spanning_type(type1, type2)
    if span_type is None:
        return py_object_type
    return span_type

def _spanning_type(type1, type2):
    if type1.is_numeric and type2.is_numeric:
        return widest_numeric_type(type1, type2)
    elif type1.is_builtin_type and type1.name == 'float' and type2.is_numeric:
        return widest_numeric_type(c_double_type, type2)
    elif type2.is_builtin_type and type2.name == 'float' and type1.is_numeric:
        return widest_numeric_type(type1, c_double_type)
    elif type1.is_extension_type and type2.is_extension_type:
        return widest_extension_type(type1, type2)
    elif type1.is_pyobject or type2.is_pyobject:
        return py_object_type
    elif type1.assignable_from(type2):
        if type1.is_extension_type and type1.typeobj_is_imported():
            # external types are unsafe, so we use PyObject instead
            return py_object_type
        return type1
    elif type2.assignable_from(type1):
        if type2.is_extension_type and type2.typeobj_is_imported():
            # external types are unsafe, so we use PyObject instead
            return py_object_type
        return type2
    elif type1.is_ptr and type2.is_ptr:
        if type1.base_type.is_cpp_class and type2.base_type.is_cpp_class:
            common_base = widest_cpp_type(type1.base_type, type2.base_type)
            if common_base:
                return CPtrType(common_base)
        # incompatible pointers, void* will do as a result
        return c_void_ptr_type
    else:
        return None

def widest_extension_type(type1, type2):
    if type1.typeobj_is_imported() or type2.typeobj_is_imported():
        return py_object_type
    while True:
        if type1.subtype_of(type2):
            return type2
        elif type2.subtype_of(type1):
            return type1
        type1, type2 = type1.base_type, type2.base_type
        if type1 is None or type2 is None:
            return py_object_type

def widest_cpp_type(type1, type2):
    @cached_function
    def bases(type):
        all = set()
        for base in type.base_classes:
            all.add(base)
            all.update(bases(base))
        return all
    common_bases = bases(type1).intersection(bases(type2))
    common_bases_bases = reduce(set.union, [bases(b) for b in common_bases], set())
    candidates = [b for b in common_bases if b not in common_bases_bases]
    if len(candidates) == 1:
        return candidates[0]
    else:
        # Fall back to void* for now.
        return None


def simple_c_type(signed, longness, name):
    # Find type descriptor for simple type given name and modifiers.
    # Returns None if arguments don't make sense.
    return modifiers_and_name_to_type.get((signed, longness, name))

def parse_basic_type(name):
    base = None
    if name.startswith('p_'):
        base = parse_basic_type(name[2:])
    elif name.startswith('p'):
        base = parse_basic_type(name[1:])
    elif name.endswith('*'):
        base = parse_basic_type(name[:-1])
    if base:
        return CPtrType(base)
    #
    basic_type = simple_c_type(1, 0, name)
    if basic_type:
        return basic_type
    #
    signed = 1
    longness = 0
    if name == 'Py_UNICODE':
        signed = 0
    elif name == 'Py_UCS4':
        signed = 0
    elif name == 'Py_hash_t':
        signed = 2
    elif name == 'Py_ssize_t':
        signed = 2
    elif name == 'ssize_t':
        signed = 2
    elif name == 'size_t':
        signed = 0
    else:
        if name.startswith('u'):
            name = name[1:]
            signed = 0
        elif (name.startswith('s') and
              not name.startswith('short')):
            name = name[1:]
            signed = 2
        longness = 0
        while name.startswith('short'):
            name = name.replace('short', '', 1).strip()
            longness -= 1
        while name.startswith('long'):
            name = name.replace('long', '', 1).strip()
            longness += 1
        if longness != 0 and not name:
            name = 'int'
    return simple_c_type(signed, longness, name)

def c_array_type(base_type, size):
    # Construct a C array type.
    if base_type is error_type:
        return error_type
    else:
        return CArrayType(base_type, size)

def c_ptr_type(base_type):
    # Construct a C pointer type.
    if base_type is error_type:
        return error_type
    elif base_type.is_reference:
        return CPtrType(base_type.ref_base_type)
    else:
        return CPtrType(base_type)

def c_ref_type(base_type):
    # Construct a C reference type
    if base_type is error_type:
        return error_type
    else:
        return CReferenceType(base_type)

def c_const_type(base_type):
    # Construct a C const type.
    if base_type is error_type:
        return error_type
    else:
        return CConstType(base_type)

def same_type(type1, type2):
    return type1.same_as(type2)

def assignable_from(type1, type2):
    return type1.assignable_from(type2)

def typecast(to_type, from_type, expr_code):
    #  Return expr_code cast to a C type which can be
    #  assigned to to_type, assuming its existing C type
    #  is from_type.
    if (to_type is from_type or
            (not to_type.is_pyobject and assignable_from(to_type, from_type))):
        return expr_code
    elif (to_type is py_object_type and from_type and
            from_type.is_builtin_type and from_type.name != 'type'):
        # no cast needed, builtins are PyObject* already
        return expr_code
    else:
        #print "typecast: to", to_type, "from", from_type ###
        return to_type.cast_code(expr_code)

def type_list_identifier(types):
    return cap_length('__and_'.join(type_identifier(type) for type in types))

_type_identifier_cache = {}
def type_identifier(type):
    decl = type.empty_declaration_code()
    safe = _type_identifier_cache.get(decl)
    if safe is None:
        safe = decl
        safe = re.sub(' +', ' ', safe)
        safe = re.sub(' ([^a-zA-Z0-9_])', r'\1', safe)
        safe = re.sub('([^a-zA-Z0-9_]) ', r'\1', safe)
        safe = (safe.replace('__', '__dunder')
                    .replace('const ', '__const_')
                    .replace(' ', '__space_')
                    .replace('*', '__ptr')
                    .replace('&', '__ref')
                    .replace('[', '__lArr')
                    .replace(']', '__rArr')
                    .replace('<', '__lAng')
                    .replace('>', '__rAng')
                    .replace('(', '__lParen')
                    .replace(')', '__rParen')
                    .replace(',', '__comma_')
                    .replace('::', '__in_'))
        safe = cap_length(re.sub('[^a-zA-Z0-9_]', lambda x: '__%X' % ord(x.group(0)), safe))
        _type_identifier_cache[decl] = safe
    return safe

def cap_length(s, max_prefix=63, max_len=1024):
    if len(s) <= max_prefix:
        return s
    hash_prefix = hashlib.sha256(s.encode('ascii')).hexdigest()[:6]
    return '%s__%s__etc' % (hash_prefix, s[:max_len-17])

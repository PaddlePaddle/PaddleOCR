# cython: language_level=3

from __future__ import absolute_import

from .PyrexTypes import CType, CTypedefType, CStructOrUnionType

import cython

try:
    import pythran
    pythran_is_pre_0_9 = tuple(map(int, pythran.__version__.split('.')[0:2])) < (0, 9)
    pythran_is_pre_0_9_6 = tuple(map(int, pythran.__version__.split('.')[0:3])) < (0, 9, 6)
except ImportError:
    pythran = None
    pythran_is_pre_0_9 = True
    pythran_is_pre_0_9_6 = True

if pythran_is_pre_0_9_6:
    pythran_builtins = '__builtin__'
else:
    pythran_builtins = 'builtins'


# Pythran/Numpy specific operations

def has_np_pythran(env):
    if env is None:
        return False
    directives = getattr(env, 'directives', None)
    return (directives and directives.get('np_pythran', False))

@cython.ccall
def is_pythran_supported_dtype(type_):
    if isinstance(type_, CTypedefType):
        return is_pythran_supported_type(type_.typedef_base_type)
    return type_.is_numeric


def pythran_type(Ty, ptype="ndarray"):
    if Ty.is_buffer:
        ndim,dtype = Ty.ndim, Ty.dtype
        if isinstance(dtype, CStructOrUnionType):
            ctype = dtype.cname
        elif isinstance(dtype, CType):
            ctype = dtype.sign_and_name()
        elif isinstance(dtype, CTypedefType):
            ctype = dtype.typedef_cname
        else:
            raise ValueError("unsupported type %s!" % dtype)
        if pythran_is_pre_0_9:
            return "pythonic::types::%s<%s,%d>" % (ptype,ctype, ndim)
        else:
            return "pythonic::types::%s<%s,pythonic::types::pshape<%s>>" % (ptype,ctype, ",".join(("long",)*ndim))
    if Ty.is_pythran_expr:
        return Ty.pythran_type
    #if Ty.is_none:
    #    return "decltype(pythonic::builtins::None)"
    if Ty.is_numeric:
        return Ty.sign_and_name()
    raise ValueError("unsupported pythran type %s (%s)" % (Ty, type(Ty)))


@cython.cfunc
def type_remove_ref(ty):
    return "typename std::remove_reference<%s>::type" % ty


def pythran_binop_type(op, tA, tB):
    if op == '**':
        return 'decltype(pythonic::numpy::functor::power{}(std::declval<%s>(), std::declval<%s>()))' % (
            pythran_type(tA), pythran_type(tB))
    else:
        return "decltype(std::declval<%s>() %s std::declval<%s>())" % (
            pythran_type(tA), op, pythran_type(tB))


def pythran_unaryop_type(op, type_):
    return "decltype(%sstd::declval<%s>())" % (
        op, pythran_type(type_))


@cython.cfunc
def _index_access(index_code, indices):
    indexing = ",".join([index_code(idx) for idx in indices])
    return ('[%s]' if len(indices) == 1 else '(%s)') % indexing


def _index_type_code(index_with_type):
    idx, index_type = index_with_type
    if idx.is_slice:
        n = 2 + int(not idx.step.is_none)
        return "pythonic::%s::functor::slice{}(%s)" % (
            pythran_builtins,
            ",".join(["0"]*n))
    elif index_type.is_int:
        return "std::declval<%s>()" % index_type.sign_and_name()
    elif index_type.is_pythran_expr:
        return "std::declval<%s>()" % index_type.pythran_type
    raise ValueError("unsupported indexing type %s!" % index_type)


def _index_code(idx):
    if idx.is_slice:
        values = idx.start, idx.stop, idx.step
        if idx.step.is_none:
            func = "contiguous_slice"
            values = values[:2]
        else:
            func = "slice"
        return "pythonic::types::%s(%s)" % (
            func, ",".join((v.pythran_result() for v in values)))
    elif idx.type.is_int:
        return to_pythran(idx)
    elif idx.type.is_pythran_expr:
        return idx.pythran_result()
    raise ValueError("unsupported indexing type %s" % idx.type)


def pythran_indexing_type(type_, indices):
    return type_remove_ref("decltype(std::declval<%s>()%s)" % (
        pythran_type(type_),
        _index_access(_index_type_code, indices),
    ))


def pythran_indexing_code(indices):
    return _index_access(_index_code, indices)

def np_func_to_list(func):
    if not func.is_numpy_attribute:
        return []
    return np_func_to_list(func.obj) + [func.attribute]

if pythran is None:
    def pythran_is_numpy_func_supported(name):
        return False
else:
    def pythran_is_numpy_func_supported(func):
        CurF = pythran.tables.MODULES['numpy']
        FL = np_func_to_list(func)
        for F in FL:
            CurF = CurF.get(F, None)
            if CurF is None:
                return False
        return True

def pythran_functor(func):
    func = np_func_to_list(func)
    submodules = "::".join(func[:-1] + ["functor"])
    return "pythonic::numpy::%s::%s" % (submodules, func[-1])

def pythran_func_type(func, args):
    args = ",".join(("std::declval<%s>()" % pythran_type(a.type) for a in args))
    return "decltype(%s{}(%s))" % (pythran_functor(func), args)


@cython.ccall
def to_pythran(op, ptype=None):
    op_type = op.type
    if op_type.is_int:
        # Make sure that integer literals always have exactly the type that the templates expect.
        return op_type.cast_code(op.result())
    if is_type(op_type, ["is_pythran_expr", "is_numeric", "is_float", "is_complex"]):
        return op.result()
    if op.is_none:
        return "pythonic::%s::None" % pythran_builtins
    if ptype is None:
        ptype = pythran_type(op_type)

    assert op.type.is_pyobject
    return "from_python<%s>(%s)" % (ptype, op.py_result())


@cython.cfunc
def is_type(type_, types):
    for attr in types:
        if getattr(type_, attr, False):
            return True
    return False


def is_pythran_supported_node_or_none(node):
    return node.is_none or is_pythran_supported_type(node.type)


@cython.ccall
def is_pythran_supported_type(type_):
    pythran_supported = (
        "is_pythran_expr", "is_int", "is_numeric", "is_float", "is_none", "is_complex")
    return is_type(type_, pythran_supported) or is_pythran_expr(type_)


def is_pythran_supported_operation_type(type_):
    pythran_supported = (
        "is_pythran_expr", "is_int", "is_numeric", "is_float", "is_complex")
    return is_type(type_,pythran_supported) or is_pythran_expr(type_)


@cython.ccall
def is_pythran_expr(type_):
    return type_.is_pythran_expr


def is_pythran_buffer(type_):
    return (type_.is_numpy_buffer and is_pythran_supported_dtype(type_.dtype) and
            type_.mode in ("c", "strided") and not type_.cast)

def pythran_get_func_include_file(func):
    func = np_func_to_list(func)
    return "pythonic/numpy/%s.hpp" % "/".join(func)

def include_pythran_generic(env):
    # Generic files
    env.add_include_file("pythonic/core.hpp")
    env.add_include_file("pythonic/python/core.hpp")
    env.add_include_file("pythonic/types/bool.hpp")
    env.add_include_file("pythonic/types/ndarray.hpp")
    env.add_include_file("pythonic/numpy/power.hpp")
    env.add_include_file("pythonic/%s/slice.hpp" % pythran_builtins)
    env.add_include_file("<new>")  # for placement new

    for i in (8, 16, 32, 64):
        env.add_include_file("pythonic/types/uint%d.hpp" % i)
        env.add_include_file("pythonic/types/int%d.hpp" % i)
    for t in ("float", "float32", "float64", "set", "slice", "tuple", "int",
              "complex", "complex64", "complex128"):
        env.add_include_file("pythonic/types/%s.hpp" % t)

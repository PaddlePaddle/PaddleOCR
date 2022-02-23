from __future__ import absolute_import

from .Errors import CompileError, error
from . import ExprNodes
from .ExprNodes import IntNode, NameNode, AttributeNode
from . import Options
from .Code import UtilityCode, TempitaUtilityCode
from .UtilityCode import CythonUtilityCode
from . import Buffer
from . import PyrexTypes
from . import ModuleNode

START_ERR = "Start must not be given."
STOP_ERR = "Axis specification only allowed in the 'step' slot."
STEP_ERR = "Step must be omitted, 1, or a valid specifier."
BOTH_CF_ERR = "Cannot specify an array that is both C and Fortran contiguous."
INVALID_ERR = "Invalid axis specification."
NOT_CIMPORTED_ERR = "Variable was not cimported from cython.view"
EXPR_ERR = "no expressions allowed in axis spec, only names and literals."
CF_ERR = "Invalid axis specification for a C/Fortran contiguous array."
ERR_UNINITIALIZED = ("Cannot check if memoryview %s is initialized without the "
                     "GIL, consider using initializedcheck(False)")


def concat_flags(*flags):
    return "(%s)" % "|".join(flags)


format_flag = "PyBUF_FORMAT"

memview_c_contiguous = "(PyBUF_C_CONTIGUOUS | PyBUF_FORMAT)"
memview_f_contiguous = "(PyBUF_F_CONTIGUOUS | PyBUF_FORMAT)"
memview_any_contiguous = "(PyBUF_ANY_CONTIGUOUS | PyBUF_FORMAT)"
memview_full_access = "PyBUF_FULL_RO"
#memview_strided_access = "PyBUF_STRIDED_RO"
memview_strided_access = "PyBUF_RECORDS_RO"

MEMVIEW_DIRECT = '__Pyx_MEMVIEW_DIRECT'
MEMVIEW_PTR    = '__Pyx_MEMVIEW_PTR'
MEMVIEW_FULL   = '__Pyx_MEMVIEW_FULL'
MEMVIEW_CONTIG = '__Pyx_MEMVIEW_CONTIG'
MEMVIEW_STRIDED= '__Pyx_MEMVIEW_STRIDED'
MEMVIEW_FOLLOW = '__Pyx_MEMVIEW_FOLLOW'

_spec_to_const = {
        'direct' : MEMVIEW_DIRECT,
        'ptr'    : MEMVIEW_PTR,
        'full'   : MEMVIEW_FULL,
        'contig' : MEMVIEW_CONTIG,
        'strided': MEMVIEW_STRIDED,
        'follow' : MEMVIEW_FOLLOW,
        }

_spec_to_abbrev = {
    'direct'  : 'd',
    'ptr'     : 'p',
    'full'    : 'f',
    'contig'  : 'c',
    'strided' : 's',
    'follow'  : '_',
}

memslice_entry_init = "{ 0, 0, { 0 }, { 0 }, { 0 } }"

memview_name = u'memoryview'
memview_typeptr_cname = '__pyx_memoryview_type'
memview_objstruct_cname = '__pyx_memoryview_obj'
memviewslice_cname = u'__Pyx_memviewslice'


def put_init_entry(mv_cname, code):
    code.putln("%s.data = NULL;" % mv_cname)
    code.putln("%s.memview = NULL;" % mv_cname)


#def axes_to_str(axes):
#    return "".join([access[0].upper()+packing[0] for (access, packing) in axes])


def put_acquire_memoryviewslice(lhs_cname, lhs_type, lhs_pos, rhs, code,
                                have_gil=False, first_assignment=True):
    "We can avoid decreffing the lhs if we know it is the first assignment"
    assert rhs.type.is_memoryviewslice

    pretty_rhs = rhs.result_in_temp() or rhs.is_simple()
    if pretty_rhs:
        rhstmp = rhs.result()
    else:
        rhstmp = code.funcstate.allocate_temp(lhs_type, manage_ref=False)
        code.putln("%s = %s;" % (rhstmp, rhs.result_as(lhs_type)))

    # Allow uninitialized assignment
    #code.putln(code.put_error_if_unbound(lhs_pos, rhs.entry))
    put_assign_to_memviewslice(lhs_cname, rhs, rhstmp, lhs_type, code,
                               have_gil=have_gil, first_assignment=first_assignment)

    if not pretty_rhs:
        code.funcstate.release_temp(rhstmp)


def put_assign_to_memviewslice(lhs_cname, rhs, rhs_cname, memviewslicetype, code,
                               have_gil=False, first_assignment=False):
    if not first_assignment:
        code.put_xdecref_memoryviewslice(lhs_cname, have_gil=have_gil)

    if not rhs.result_in_temp():
        rhs.make_owned_memoryviewslice(code)

    code.putln("%s = %s;" % (lhs_cname, rhs_cname))


def get_buf_flags(specs):
    is_c_contig, is_f_contig = is_cf_contig(specs)

    if is_c_contig:
        return memview_c_contiguous
    elif is_f_contig:
        return memview_f_contiguous

    access, packing = zip(*specs)

    if 'full' in access or 'ptr' in access:
        return memview_full_access
    else:
        return memview_strided_access


def insert_newaxes(memoryviewtype, n):
    axes = [('direct', 'strided')] * n
    axes.extend(memoryviewtype.axes)
    return PyrexTypes.MemoryViewSliceType(memoryviewtype.dtype, axes)


def broadcast_types(src, dst):
    n = abs(src.ndim - dst.ndim)
    if src.ndim < dst.ndim:
        return insert_newaxes(src, n), dst
    else:
        return src, insert_newaxes(dst, n)


def valid_memslice_dtype(dtype, i=0):
    """
    Return whether type dtype can be used as the base type of a
    memoryview slice.

    We support structs, numeric types and objects
    """
    if dtype.is_complex and dtype.real_type.is_int:
        return False

    if dtype is PyrexTypes.c_bint_type:
        return False

    if dtype.is_struct and dtype.kind == 'struct':
        for member in dtype.scope.var_entries:
            if not valid_memslice_dtype(member.type):
                return False

        return True

    return (
        dtype.is_error or
        # Pointers are not valid (yet)
        # (dtype.is_ptr and valid_memslice_dtype(dtype.base_type)) or
        (dtype.is_array and i < 8 and
         valid_memslice_dtype(dtype.base_type, i + 1)) or
        dtype.is_numeric or
        dtype.is_pyobject or
        dtype.is_fused or # accept this as it will be replaced by specializations later
        (dtype.is_typedef and valid_memslice_dtype(dtype.typedef_base_type))
    )


class MemoryViewSliceBufferEntry(Buffer.BufferEntry):
    """
    May be used during code generation time to be queried for
    shape/strides/suboffsets attributes, or to perform indexing or slicing.
    """
    def __init__(self, entry):
        self.entry = entry
        self.type = entry.type
        self.cname = entry.cname

        self.buf_ptr = "%s.data" % self.cname

        dtype = self.entry.type.dtype
        self.buf_ptr_type = PyrexTypes.CPtrType(dtype)
        self.init_attributes()

    def get_buf_suboffsetvars(self):
        return self._for_all_ndim("%s.suboffsets[%d]")

    def get_buf_stridevars(self):
        return self._for_all_ndim("%s.strides[%d]")

    def get_buf_shapevars(self):
        return self._for_all_ndim("%s.shape[%d]")

    def generate_buffer_lookup_code(self, code, index_cnames):
        axes = [(dim, index_cnames[dim], access, packing)
                    for dim, (access, packing) in enumerate(self.type.axes)]
        return self._generate_buffer_lookup_code(code, axes)

    def _generate_buffer_lookup_code(self, code, axes, cast_result=True):
        """
        Generate a single expression that indexes the memory view slice
        in each dimension.
        """
        bufp = self.buf_ptr
        type_decl = self.type.dtype.empty_declaration_code()

        for dim, index, access, packing in axes:
            shape = "%s.shape[%d]" % (self.cname, dim)
            stride = "%s.strides[%d]" % (self.cname, dim)
            suboffset = "%s.suboffsets[%d]" % (self.cname, dim)

            flag = get_memoryview_flag(access, packing)

            if flag in ("generic", "generic_contiguous"):
                # Note: we cannot do cast tricks to avoid stride multiplication
                #       for generic_contiguous, as we may have to do (dtype *)
                #       or (dtype **) arithmetic, we won't know which unless
                #       we check suboffsets
                code.globalstate.use_utility_code(memviewslice_index_helpers)
                bufp = ('__pyx_memviewslice_index_full(%s, %s, %s, %s)' %
                                            (bufp, index, stride, suboffset))

            elif flag == "indirect":
                bufp = "(%s + %s * %s)" % (bufp, index, stride)
                bufp = ("(*((char **) %s) + %s)" % (bufp, suboffset))

            elif flag == "indirect_contiguous":
                # Note: we do char ** arithmetic
                bufp = "(*((char **) %s + %s) + %s)" % (bufp, index, suboffset)

            elif flag == "strided":
                bufp = "(%s + %s * %s)" % (bufp, index, stride)

            else:
                assert flag == 'contiguous', flag
                bufp = '((char *) (((%s *) %s) + %s))' % (type_decl, bufp, index)

            bufp = '( /* dim=%d */ %s )' % (dim, bufp)

        if cast_result:
            return "((%s *) %s)" % (type_decl, bufp)

        return bufp

    def generate_buffer_slice_code(self, code, indices, dst, have_gil,
                                   have_slices, directives):
        """
        Slice a memoryviewslice.

        indices     - list of index nodes. If not a SliceNode, or NoneNode,
                      then it must be coercible to Py_ssize_t

        Simply call __pyx_memoryview_slice_memviewslice with the right
        arguments, unless the dimension is omitted or a bare ':', in which
        case we copy over the shape/strides/suboffsets attributes directly
        for that dimension.
        """
        src = self.cname

        code.putln("%(dst)s.data = %(src)s.data;" % locals())
        code.putln("%(dst)s.memview = %(src)s.memview;" % locals())
        code.put_incref_memoryviewslice(dst)

        all_dimensions_direct = all(access == 'direct' for access, packing in self.type.axes)
        suboffset_dim_temp = []

        def get_suboffset_dim():
            # create global temp variable at request
            if not suboffset_dim_temp:
                suboffset_dim = code.funcstate.allocate_temp(PyrexTypes.c_int_type, manage_ref=False)
                code.putln("%s = -1;" % suboffset_dim)
                suboffset_dim_temp.append(suboffset_dim)
            return suboffset_dim_temp[0]

        dim = -1
        new_ndim = 0
        for index in indices:
            if index.is_none:
                # newaxis
                for attrib, value in [('shape', 1), ('strides', 0), ('suboffsets', -1)]:
                    code.putln("%s.%s[%d] = %d;" % (dst, attrib, new_ndim, value))

                new_ndim += 1
                continue

            dim += 1
            access, packing = self.type.axes[dim]

            if isinstance(index, ExprNodes.SliceNode):
                # slice, unspecified dimension, or part of ellipsis
                d = dict(locals())
                for s in "start stop step".split():
                    idx = getattr(index, s)
                    have_idx = d['have_' + s] = not idx.is_none
                    d[s] = idx.result() if have_idx else "0"

                if not (d['have_start'] or d['have_stop'] or d['have_step']):
                    # full slice (:), simply copy over the extent, stride
                    # and suboffset. Also update suboffset_dim if needed
                    d['access'] = access
                    util_name = "SimpleSlice"
                else:
                    util_name = "ToughSlice"
                    d['error_goto'] = code.error_goto(index.pos)

                new_ndim += 1
            else:
                # normal index
                idx = index.result()

                indirect = access != 'direct'
                if indirect:
                    generic = access == 'full'
                    if new_ndim != 0:
                        return error(index.pos,
                                     "All preceding dimensions must be "
                                     "indexed and not sliced")

                d = dict(
                    locals(),
                    wraparound=int(directives['wraparound']),
                    boundscheck=int(directives['boundscheck']),
                )
                if d['boundscheck']:
                    d['error_goto'] = code.error_goto(index.pos)
                util_name = "SliceIndex"

            _, impl = TempitaUtilityCode.load_as_string(util_name, "MemoryView_C.c", context=d)
            code.put(impl)

        if suboffset_dim_temp:
            code.funcstate.release_temp(suboffset_dim_temp[0])


def empty_slice(pos):
    none = ExprNodes.NoneNode(pos)
    return ExprNodes.SliceNode(pos, start=none,
                               stop=none, step=none)


def unellipsify(indices, ndim):
    result = []
    seen_ellipsis = False
    have_slices = False

    newaxes = [newaxis for newaxis in indices if newaxis.is_none]
    n_indices = len(indices) - len(newaxes)

    for index in indices:
        if isinstance(index, ExprNodes.EllipsisNode):
            have_slices = True
            full_slice = empty_slice(index.pos)

            if seen_ellipsis:
                result.append(full_slice)
            else:
                nslices = ndim - n_indices + 1
                result.extend([full_slice] * nslices)
                seen_ellipsis = True
        else:
            have_slices = have_slices or index.is_slice or index.is_none
            result.append(index)

    result_length = len(result) - len(newaxes)
    if result_length < ndim:
        have_slices = True
        nslices = ndim - result_length
        result.extend([empty_slice(indices[-1].pos)] * nslices)

    return have_slices, result, newaxes


def get_memoryview_flag(access, packing):
    if access == 'full' and packing in ('strided', 'follow'):
        return 'generic'
    elif access == 'full' and packing == 'contig':
        return 'generic_contiguous'
    elif access == 'ptr' and packing in ('strided', 'follow'):
        return 'indirect'
    elif access == 'ptr' and packing == 'contig':
        return 'indirect_contiguous'
    elif access == 'direct' and packing in ('strided', 'follow'):
        return 'strided'
    else:
        assert (access, packing) == ('direct', 'contig'), (access, packing)
        return 'contiguous'


def get_is_contig_func_name(contig_type, ndim):
    assert contig_type in ('C', 'F')
    return "__pyx_memviewslice_is_contig_%s%d" % (contig_type, ndim)


def get_is_contig_utility(contig_type, ndim):
    assert contig_type in ('C', 'F')
    C = dict(context, ndim=ndim, contig_type=contig_type)
    utility = load_memview_c_utility("MemviewSliceCheckContig", C, requires=[is_contig_utility])
    return utility


def slice_iter(slice_type, slice_result, ndim, code):
    if slice_type.is_c_contig or slice_type.is_f_contig:
        return ContigSliceIter(slice_type, slice_result, ndim, code)
    else:
        return StridedSliceIter(slice_type, slice_result, ndim, code)


class SliceIter(object):
    def __init__(self, slice_type, slice_result, ndim, code):
        self.slice_type = slice_type
        self.slice_result = slice_result
        self.code = code
        self.ndim = ndim


class ContigSliceIter(SliceIter):
    def start_loops(self):
        code = self.code
        code.begin_block()

        type_decl = self.slice_type.dtype.empty_declaration_code()

        total_size = ' * '.join("%s.shape[%d]" % (self.slice_result, i)
                                for i in range(self.ndim))
        code.putln("Py_ssize_t __pyx_temp_extent = %s;" % total_size)
        code.putln("Py_ssize_t __pyx_temp_idx;")
        code.putln("%s *__pyx_temp_pointer = (%s *) %s.data;" % (
            type_decl, type_decl, self.slice_result))
        code.putln("for (__pyx_temp_idx = 0; "
                        "__pyx_temp_idx < __pyx_temp_extent; "
                        "__pyx_temp_idx++) {")

        return "__pyx_temp_pointer"

    def end_loops(self):
        self.code.putln("__pyx_temp_pointer += 1;")
        self.code.putln("}")
        self.code.end_block()


class StridedSliceIter(SliceIter):
    def start_loops(self):
        code = self.code
        code.begin_block()

        for i in range(self.ndim):
            t = i, self.slice_result, i
            code.putln("Py_ssize_t __pyx_temp_extent_%d = %s.shape[%d];" % t)
            code.putln("Py_ssize_t __pyx_temp_stride_%d = %s.strides[%d];" % t)
            code.putln("char *__pyx_temp_pointer_%d;" % i)
            code.putln("Py_ssize_t __pyx_temp_idx_%d;" % i)

        code.putln("__pyx_temp_pointer_0 = %s.data;" % self.slice_result)

        for i in range(self.ndim):
            if i > 0:
                code.putln("__pyx_temp_pointer_%d = __pyx_temp_pointer_%d;" % (i, i - 1))

            code.putln("for (__pyx_temp_idx_%d = 0; "
                            "__pyx_temp_idx_%d < __pyx_temp_extent_%d; "
                            "__pyx_temp_idx_%d++) {" % (i, i, i, i))

        return "__pyx_temp_pointer_%d" % (self.ndim - 1)

    def end_loops(self):
        code = self.code
        for i in range(self.ndim - 1, -1, -1):
            code.putln("__pyx_temp_pointer_%d += __pyx_temp_stride_%d;" % (i, i))
            code.putln("}")

        code.end_block()


def copy_c_or_fortran_cname(memview):
    if memview.is_c_contig:
        c_or_f = 'c'
    else:
        c_or_f = 'f'

    return "__pyx_memoryview_copy_slice_%s_%s" % (
            memview.specialization_suffix(), c_or_f)


def get_copy_new_utility(pos, from_memview, to_memview):
    if (from_memview.dtype != to_memview.dtype and
            not (from_memview.dtype.is_const and from_memview.dtype.const_base_type == to_memview.dtype)):
        error(pos, "dtypes must be the same!")
        return
    if len(from_memview.axes) != len(to_memview.axes):
        error(pos, "number of dimensions must be same")
        return
    if not (to_memview.is_c_contig or to_memview.is_f_contig):
        error(pos, "to_memview must be c or f contiguous.")
        return

    for (access, packing) in from_memview.axes:
        if access != 'direct':
            error(pos, "cannot handle 'full' or 'ptr' access at this time.")
            return

    if to_memview.is_c_contig:
        mode = 'c'
        contig_flag = memview_c_contiguous
    elif to_memview.is_f_contig:
        mode = 'fortran'
        contig_flag = memview_f_contiguous

    return load_memview_c_utility(
        "CopyContentsUtility",
        context=dict(
            context,
            mode=mode,
            dtype_decl=to_memview.dtype.empty_declaration_code(),
            contig_flag=contig_flag,
            ndim=to_memview.ndim,
            func_cname=copy_c_or_fortran_cname(to_memview),
            dtype_is_object=int(to_memview.dtype.is_pyobject)),
        requires=[copy_contents_new_utility])


def get_axes_specs(env, axes):
    '''
    get_axes_specs(env, axes) -> list of (access, packing) specs for each axis.
    access is one of 'full', 'ptr' or 'direct'
    packing is one of 'contig', 'strided' or 'follow'
    '''

    cythonscope = env.global_scope().context.cython_scope
    cythonscope.load_cythonscope()
    viewscope = cythonscope.viewscope

    access_specs = tuple([viewscope.lookup(name)
                    for name in ('full', 'direct', 'ptr')])
    packing_specs = tuple([viewscope.lookup(name)
                    for name in ('contig', 'strided', 'follow')])

    is_f_contig, is_c_contig = False, False
    default_access, default_packing = 'direct', 'strided'
    cf_access, cf_packing = default_access, 'follow'

    axes_specs = []
    # analyse all axes.
    for idx, axis in enumerate(axes):
        if not axis.start.is_none:
            raise CompileError(axis.start.pos,  START_ERR)

        if not axis.stop.is_none:
            raise CompileError(axis.stop.pos, STOP_ERR)

        if axis.step.is_none:
            axes_specs.append((default_access, default_packing))

        elif isinstance(axis.step, IntNode):
            # the packing for the ::1 axis is contiguous,
            # all others are cf_packing.
            if axis.step.compile_time_value(env) != 1:
                raise CompileError(axis.step.pos, STEP_ERR)

            axes_specs.append((cf_access, 'cfcontig'))

        elif isinstance(axis.step, (NameNode, AttributeNode)):
            entry = _get_resolved_spec(env, axis.step)
            if entry.name in view_constant_to_access_packing:
                axes_specs.append(view_constant_to_access_packing[entry.name])
            else:
                raise CompileError(axis.step.pos, INVALID_ERR)

        else:
            raise CompileError(axis.step.pos, INVALID_ERR)

    # First, find out if we have a ::1 somewhere
    contig_dim = 0
    is_contig = False
    for idx, (access, packing) in enumerate(axes_specs):
        if packing == 'cfcontig':
            if is_contig:
                raise CompileError(axis.step.pos, BOTH_CF_ERR)

            contig_dim = idx
            axes_specs[idx] = (access, 'contig')
            is_contig = True

    if is_contig:
        # We have a ::1 somewhere, see if we're C or Fortran contiguous
        if contig_dim == len(axes) - 1:
            is_c_contig = True
        else:
            is_f_contig = True

            if contig_dim and not axes_specs[contig_dim - 1][0] in ('full', 'ptr'):
                raise CompileError(axes[contig_dim].pos,
                                   "Fortran contiguous specifier must follow an indirect dimension")

        if is_c_contig:
            # Contiguous in the last dimension, find the last indirect dimension
            contig_dim = -1
            for idx, (access, packing) in enumerate(reversed(axes_specs)):
                if access in ('ptr', 'full'):
                    contig_dim = len(axes) - idx - 1

        # Replace 'strided' with 'follow' for any dimension following the last
        # indirect dimension, the first dimension or the dimension following
        # the ::1.
        #               int[::indirect, ::1, :, :]
        #                                    ^  ^
        #               int[::indirect, :, :, ::1]
        #                               ^  ^
        start = contig_dim + 1
        stop = len(axes) - is_c_contig
        for idx, (access, packing) in enumerate(axes_specs[start:stop]):
            idx = contig_dim + 1 + idx
            if access != 'direct':
                raise CompileError(axes[idx].pos,
                                   "Indirect dimension may not follow "
                                   "Fortran contiguous dimension")
            if packing == 'contig':
                raise CompileError(axes[idx].pos,
                                   "Dimension may not be contiguous")
            axes_specs[idx] = (access, cf_packing)

        if is_c_contig:
            # For C contiguity, we need to fix the 'contig' dimension
            # after the loop
            a, p = axes_specs[-1]
            axes_specs[-1] = a, 'contig'

    validate_axes_specs([axis.start.pos for axis in axes],
                        axes_specs,
                        is_c_contig,
                        is_f_contig)

    return axes_specs


def validate_axes(pos, axes):
    if len(axes) >= Options.buffer_max_dims:
        error(pos, "More dimensions than the maximum number"
                   " of buffer dimensions were used.")
        return False

    return True


def is_cf_contig(specs):
    is_c_contig = is_f_contig = False

    if len(specs) == 1 and specs == [('direct', 'contig')]:
        is_c_contig = True

    elif (specs[-1] == ('direct','contig') and
          all(axis == ('direct','follow') for axis in specs[:-1])):
        # c_contiguous: 'follow', 'follow', ..., 'follow', 'contig'
        is_c_contig = True

    elif (len(specs) > 1 and
        specs[0] == ('direct','contig') and
        all(axis == ('direct','follow') for axis in specs[1:])):
        # f_contiguous: 'contig', 'follow', 'follow', ..., 'follow'
        is_f_contig = True

    return is_c_contig, is_f_contig


def get_mode(specs):
    is_c_contig, is_f_contig = is_cf_contig(specs)

    if is_c_contig:
        return 'c'
    elif is_f_contig:
        return 'fortran'

    for access, packing in specs:
        if access in ('ptr', 'full'):
            return 'full'

    return 'strided'

view_constant_to_access_packing = {
    'generic':              ('full',   'strided'),
    'strided':              ('direct', 'strided'),
    'indirect':             ('ptr',    'strided'),
    'generic_contiguous':   ('full',   'contig'),
    'contiguous':           ('direct', 'contig'),
    'indirect_contiguous':  ('ptr',    'contig'),
}

def validate_axes_specs(positions, specs, is_c_contig, is_f_contig):

    packing_specs = ('contig', 'strided', 'follow')
    access_specs = ('direct', 'ptr', 'full')

    # is_c_contig, is_f_contig = is_cf_contig(specs)

    has_contig = has_follow = has_strided = has_generic_contig = False

    last_indirect_dimension = -1
    for idx, (access, packing) in enumerate(specs):
        if access == 'ptr':
            last_indirect_dimension = idx

    for idx, (pos, (access, packing)) in enumerate(zip(positions, specs)):

        if not (access in access_specs and
                packing in packing_specs):
            raise CompileError(pos, "Invalid axes specification.")

        if packing == 'strided':
            has_strided = True
        elif packing == 'contig':
            if has_contig:
                raise CompileError(pos, "Only one direct contiguous "
                                        "axis may be specified.")

            valid_contig_dims = last_indirect_dimension + 1, len(specs) - 1
            if idx not in valid_contig_dims and access != 'ptr':
                if last_indirect_dimension + 1 != len(specs) - 1:
                    dims = "dimensions %d and %d" % valid_contig_dims
                else:
                    dims = "dimension %d" % valid_contig_dims[0]

                raise CompileError(pos, "Only %s may be contiguous and direct" % dims)

            has_contig = access != 'ptr'
        elif packing == 'follow':
            if has_strided:
                raise CompileError(pos, "A memoryview cannot have both follow and strided axis specifiers.")
            if not (is_c_contig or is_f_contig):
                raise CompileError(pos, "Invalid use of the follow specifier.")

        if access in ('ptr', 'full'):
            has_strided = False

def _get_resolved_spec(env, spec):
    # spec must be a NameNode or an AttributeNode
    if isinstance(spec, NameNode):
        return _resolve_NameNode(env, spec)
    elif isinstance(spec, AttributeNode):
        return _resolve_AttributeNode(env, spec)
    else:
        raise CompileError(spec.pos, INVALID_ERR)

def _resolve_NameNode(env, node):
    try:
        resolved_name = env.lookup(node.name).name
    except AttributeError:
        raise CompileError(node.pos, INVALID_ERR)

    viewscope = env.global_scope().context.cython_scope.viewscope
    entry = viewscope.lookup(resolved_name)
    if entry is None:
        raise CompileError(node.pos, NOT_CIMPORTED_ERR)

    return entry

def _resolve_AttributeNode(env, node):
    path = []
    while isinstance(node, AttributeNode):
        path.insert(0, node.attribute)
        node = node.obj
    if isinstance(node, NameNode):
        path.insert(0, node.name)
    else:
        raise CompileError(node.pos, EXPR_ERR)
    modnames = path[:-1]
    # must be at least 1 module name, o/w not an AttributeNode.
    assert modnames

    scope = env
    for modname in modnames:
        mod = scope.lookup(modname)
        if not mod or not mod.as_module:
            raise CompileError(
                    node.pos, "undeclared name not builtin: %s" % modname)
        scope = mod.as_module

    entry = scope.lookup(path[-1])
    if not entry:
        raise CompileError(node.pos, "No such attribute '%s'" % path[-1])

    return entry

#
### Utility loading
#

def load_memview_cy_utility(util_code_name, context=None, **kwargs):
    return CythonUtilityCode.load(util_code_name, "MemoryView.pyx",
                                  context=context, **kwargs)

def load_memview_c_utility(util_code_name, context=None, **kwargs):
    if context is None:
        return UtilityCode.load(util_code_name, "MemoryView_C.c", **kwargs)
    else:
        return TempitaUtilityCode.load(util_code_name, "MemoryView_C.c",
                                       context=context, **kwargs)

def use_cython_array_utility_code(env):
    cython_scope = env.global_scope().context.cython_scope
    cython_scope.load_cythonscope()
    cython_scope.viewscope.lookup('array_cwrapper').used = True

context = {
    'memview_struct_name': memview_objstruct_cname,
    'max_dims': Options.buffer_max_dims,
    'memviewslice_name': memviewslice_cname,
    'memslice_init': memslice_entry_init,
}
memviewslice_declare_code = load_memview_c_utility(
        "MemviewSliceStruct",
        context=context,
        requires=[])

atomic_utility = load_memview_c_utility("Atomics", context)

memviewslice_init_code = load_memview_c_utility(
    "MemviewSliceInit",
    context=dict(context, BUF_MAX_NDIMS=Options.buffer_max_dims),
    requires=[memviewslice_declare_code,
              atomic_utility],
)

memviewslice_index_helpers = load_memview_c_utility("MemviewSliceIndex")

typeinfo_to_format_code = load_memview_cy_utility(
        "BufferFormatFromTypeInfo", requires=[Buffer._typeinfo_to_format_code])

is_contig_utility = load_memview_c_utility("MemviewSliceIsContig", context)
overlapping_utility = load_memview_c_utility("OverlappingSlices", context)
copy_contents_new_utility = load_memview_c_utility(
    "MemviewSliceCopyTemplate",
    context,
    requires=[], # require cython_array_utility_code
)

view_utility_code = load_memview_cy_utility(
        "View.MemoryView",
        context=context,
        requires=[Buffer.GetAndReleaseBufferUtilityCode(),
                  Buffer.buffer_struct_declare_code,
                  Buffer.buffer_formats_declare_code,
                  memviewslice_init_code,
                  is_contig_utility,
                  overlapping_utility,
                  copy_contents_new_utility,
                  ModuleNode.capsule_utility_code],
)
view_utility_whitelist = ('array', 'memoryview', 'array_cwrapper',
                          'generic', 'strided', 'indirect', 'contiguous',
                          'indirect_contiguous')

memviewslice_declare_code.requires.append(view_utility_code)
copy_contents_new_utility.requires.append(view_utility_code)

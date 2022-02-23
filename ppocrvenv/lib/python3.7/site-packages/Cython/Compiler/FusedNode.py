from __future__ import absolute_import

import copy

from . import (ExprNodes, PyrexTypes, MemoryView,
               ParseTreeTransforms, StringEncoding, Errors)
from .ExprNodes import CloneNode, ProxyNode, TupleNode
from .Nodes import FuncDefNode, CFuncDefNode, StatListNode, DefNode
from ..Utils import OrderedSet


class FusedCFuncDefNode(StatListNode):
    """
    This node replaces a function with fused arguments. It deep-copies the
    function for every permutation of fused types, and allocates a new local
    scope for it. It keeps track of the original function in self.node, and
    the entry of the original function in the symbol table is given the
    'fused_cfunction' attribute which points back to us.
    Then when a function lookup occurs (to e.g. call it), the call can be
    dispatched to the right function.

    node    FuncDefNode    the original function
    nodes   [FuncDefNode]  list of copies of node with different specific types
    py_func DefNode        the fused python function subscriptable from
                           Python space
    __signatures__         A DictNode mapping signature specialization strings
                           to PyCFunction nodes
    resulting_fused_function  PyCFunction for the fused DefNode that delegates
                              to specializations
    fused_func_assignment   Assignment of the fused function to the function name
    defaults_tuple          TupleNode of defaults (letting PyCFunctionNode build
                            defaults would result in many different tuples)
    specialized_pycfuncs    List of synthesized pycfunction nodes for the
                            specializations
    code_object             CodeObjectNode shared by all specializations and the
                            fused function

    fused_compound_types    All fused (compound) types (e.g. floating[:])
    """

    __signatures__ = None
    resulting_fused_function = None
    fused_func_assignment = None
    defaults_tuple = None
    decorators = None

    child_attrs = StatListNode.child_attrs + [
        '__signatures__', 'resulting_fused_function', 'fused_func_assignment']

    def __init__(self, node, env):
        super(FusedCFuncDefNode, self).__init__(node.pos)

        self.nodes = []
        self.node = node

        is_def = isinstance(self.node, DefNode)
        if is_def:
            # self.node.decorators = []
            self.copy_def(env)
        else:
            self.copy_cdef(env)

        # Perform some sanity checks. If anything fails, it's a bug
        for n in self.nodes:
            assert not n.entry.type.is_fused
            assert not n.local_scope.return_type.is_fused
            if node.return_type.is_fused:
                assert not n.return_type.is_fused

            if not is_def and n.cfunc_declarator.optional_arg_count:
                assert n.type.op_arg_struct

        node.entry.fused_cfunction = self
        # Copy the nodes as AnalyseDeclarationsTransform will prepend
        # self.py_func to self.stats, as we only want specialized
        # CFuncDefNodes in self.nodes
        self.stats = self.nodes[:]

    def copy_def(self, env):
        """
        Create a copy of the original def or lambda function for specialized
        versions.
        """
        fused_compound_types = PyrexTypes.unique(
            [arg.type for arg in self.node.args if arg.type.is_fused])
        fused_types = self._get_fused_base_types(fused_compound_types)
        permutations = PyrexTypes.get_all_specialized_permutations(fused_types)

        self.fused_compound_types = fused_compound_types

        if self.node.entry in env.pyfunc_entries:
            env.pyfunc_entries.remove(self.node.entry)

        for cname, fused_to_specific in permutations:
            copied_node = copy.deepcopy(self.node)
            # keep signature object identity for special casing in DefNode.analyse_declarations()
            copied_node.entry.signature = self.node.entry.signature

            self._specialize_function_args(copied_node.args, fused_to_specific)
            copied_node.return_type = self.node.return_type.specialize(
                                                    fused_to_specific)

            copied_node.analyse_declarations(env)
            # copied_node.is_staticmethod = self.node.is_staticmethod
            # copied_node.is_classmethod = self.node.is_classmethod
            self.create_new_local_scope(copied_node, env, fused_to_specific)
            self.specialize_copied_def(copied_node, cname, self.node.entry,
                                       fused_to_specific, fused_compound_types)

            PyrexTypes.specialize_entry(copied_node.entry, cname)
            copied_node.entry.used = True
            env.entries[copied_node.entry.name] = copied_node.entry

            if not self.replace_fused_typechecks(copied_node):
                break

        self.orig_py_func = self.node
        self.py_func = self.make_fused_cpdef(self.node, env, is_def=True)

    def copy_cdef(self, env):
        """
        Create a copy of the original c(p)def function for all specialized
        versions.
        """
        permutations = self.node.type.get_all_specialized_permutations()
        # print 'Node %s has %d specializations:' % (self.node.entry.name,
        #                                            len(permutations))
        # import pprint; pprint.pprint([d for cname, d in permutations])

        # Prevent copying of the python function
        self.orig_py_func = orig_py_func = self.node.py_func
        self.node.py_func = None
        if orig_py_func:
            env.pyfunc_entries.remove(orig_py_func.entry)

        fused_types = self.node.type.get_fused_types()
        self.fused_compound_types = fused_types

        new_cfunc_entries = []
        for cname, fused_to_specific in permutations:
            copied_node = copy.deepcopy(self.node)

            # Make the types in our CFuncType specific.
            type = copied_node.type.specialize(fused_to_specific)
            entry = copied_node.entry
            type.specialize_entry(entry, cname)

            # Reuse existing Entries (e.g. from .pxd files).
            for i, orig_entry in enumerate(env.cfunc_entries):
                if entry.cname == orig_entry.cname and type.same_as_resolved_type(orig_entry.type):
                    copied_node.entry = env.cfunc_entries[i]
                    if not copied_node.entry.func_cname:
                        copied_node.entry.func_cname = entry.func_cname
                    entry = copied_node.entry
                    type = entry.type
                    break
            else:
                new_cfunc_entries.append(entry)

            copied_node.type = type
            entry.type, type.entry = type, entry

            entry.used = (entry.used or
                          self.node.entry.defined_in_pxd or
                          env.is_c_class_scope or
                          entry.is_cmethod)

            if self.node.cfunc_declarator.optional_arg_count:
                self.node.cfunc_declarator.declare_optional_arg_struct(
                                           type, env, fused_cname=cname)

            copied_node.return_type = type.return_type
            self.create_new_local_scope(copied_node, env, fused_to_specific)

            # Make the argument types in the CFuncDeclarator specific
            self._specialize_function_args(copied_node.cfunc_declarator.args,
                                           fused_to_specific)

            # If a cpdef, declare all specialized cpdefs (this
            # also calls analyse_declarations)
            copied_node.declare_cpdef_wrapper(env)
            if copied_node.py_func:
                env.pyfunc_entries.remove(copied_node.py_func.entry)

                self.specialize_copied_def(
                        copied_node.py_func, cname, self.node.entry.as_variable,
                        fused_to_specific, fused_types)

            if not self.replace_fused_typechecks(copied_node):
                break

        # replace old entry with new entries
        try:
            cindex = env.cfunc_entries.index(self.node.entry)
        except ValueError:
            env.cfunc_entries.extend(new_cfunc_entries)
        else:
            env.cfunc_entries[cindex:cindex+1] = new_cfunc_entries

        if orig_py_func:
            self.py_func = self.make_fused_cpdef(orig_py_func, env,
                                                 is_def=False)
        else:
            self.py_func = orig_py_func

    def _get_fused_base_types(self, fused_compound_types):
        """
        Get a list of unique basic fused types, from a list of
        (possibly) compound fused types.
        """
        base_types = []
        seen = set()
        for fused_type in fused_compound_types:
            fused_type.get_fused_types(result=base_types, seen=seen)
        return base_types

    def _specialize_function_args(self, args, fused_to_specific):
        for arg in args:
            if arg.type.is_fused:
                arg.type = arg.type.specialize(fused_to_specific)
                if arg.type.is_memoryviewslice:
                    arg.type.validate_memslice_dtype(arg.pos)

    def create_new_local_scope(self, node, env, f2s):
        """
        Create a new local scope for the copied node and append it to
        self.nodes. A new local scope is needed because the arguments with the
        fused types are already in the local scope, and we need the specialized
        entries created after analyse_declarations on each specialized version
        of the (CFunc)DefNode.
        f2s is a dict mapping each fused type to its specialized version
        """
        node.create_local_scope(env)
        node.local_scope.fused_to_specific = f2s

        # This is copied from the original function, set it to false to
        # stop recursion
        node.has_fused_arguments = False
        self.nodes.append(node)

    def specialize_copied_def(self, node, cname, py_entry, f2s, fused_compound_types):
        """Specialize the copy of a DefNode given the copied node,
        the specialization cname and the original DefNode entry"""
        fused_types = self._get_fused_base_types(fused_compound_types)
        type_strings = [
            PyrexTypes.specialization_signature_string(fused_type, f2s)
                for fused_type in fused_types
        ]

        node.specialized_signature_string = '|'.join(type_strings)

        node.entry.pymethdef_cname = PyrexTypes.get_fused_cname(
                                        cname, node.entry.pymethdef_cname)
        node.entry.doc = py_entry.doc
        node.entry.doc_cname = py_entry.doc_cname

    def replace_fused_typechecks(self, copied_node):
        """
        Branch-prune fused type checks like

            if fused_t is int:
                ...

        Returns whether an error was issued and whether we should stop in
        in order to prevent a flood of errors.
        """
        num_errors = Errors.num_errors
        transform = ParseTreeTransforms.ReplaceFusedTypeChecks(
                                       copied_node.local_scope)
        transform(copied_node)

        if Errors.num_errors > num_errors:
            return False

        return True

    def _fused_instance_checks(self, normal_types, pyx_code, env):
        """
        Generate Cython code for instance checks, matching an object to
        specialized types.
        """
        for specialized_type in normal_types:
            # all_numeric = all_numeric and specialized_type.is_numeric
            pyx_code.context.update(
                py_type_name=specialized_type.py_type_name(),
                specialized_type_name=specialized_type.specialization_string,
            )
            pyx_code.put_chunk(
                u"""
                    if isinstance(arg, {{py_type_name}}):
                        dest_sig[{{dest_sig_idx}}] = '{{specialized_type_name}}'; break
                """)

    def _dtype_name(self, dtype):
        if dtype.is_typedef:
            return '___pyx_%s' % dtype
        return str(dtype).replace(' ', '_')

    def _dtype_type(self, dtype):
        if dtype.is_typedef:
            return self._dtype_name(dtype)
        return str(dtype)

    def _sizeof_dtype(self, dtype):
        if dtype.is_pyobject:
            return 'sizeof(void *)'
        else:
            return "sizeof(%s)" % self._dtype_type(dtype)

    def _buffer_check_numpy_dtype_setup_cases(self, pyx_code):
        "Setup some common cases to match dtypes against specializations"
        if pyx_code.indenter("if kind in b'iu':"):
            pyx_code.putln("pass")
            pyx_code.named_insertion_point("dtype_int")
            pyx_code.dedent()

        if pyx_code.indenter("elif kind == b'f':"):
            pyx_code.putln("pass")
            pyx_code.named_insertion_point("dtype_float")
            pyx_code.dedent()

        if pyx_code.indenter("elif kind == b'c':"):
            pyx_code.putln("pass")
            pyx_code.named_insertion_point("dtype_complex")
            pyx_code.dedent()

        if pyx_code.indenter("elif kind == b'O':"):
            pyx_code.putln("pass")
            pyx_code.named_insertion_point("dtype_object")
            pyx_code.dedent()

    match = "dest_sig[{{dest_sig_idx}}] = '{{specialized_type_name}}'"
    no_match = "dest_sig[{{dest_sig_idx}}] = None"
    def _buffer_check_numpy_dtype(self, pyx_code, specialized_buffer_types, pythran_types):
        """
        Match a numpy dtype object to the individual specializations.
        """
        self._buffer_check_numpy_dtype_setup_cases(pyx_code)

        for specialized_type in pythran_types+specialized_buffer_types:
            final_type = specialized_type
            if specialized_type.is_pythran_expr:
                specialized_type = specialized_type.org_buffer
            dtype = specialized_type.dtype
            pyx_code.context.update(
                itemsize_match=self._sizeof_dtype(dtype) + " == itemsize",
                signed_match="not (%s_is_signed ^ dtype_signed)" % self._dtype_name(dtype),
                dtype=dtype,
                specialized_type_name=final_type.specialization_string)

            dtypes = [
                (dtype.is_int, pyx_code.dtype_int),
                (dtype.is_float, pyx_code.dtype_float),
                (dtype.is_complex, pyx_code.dtype_complex)
            ]

            for dtype_category, codewriter in dtypes:
                if dtype_category:
                    cond = '{{itemsize_match}} and (<Py_ssize_t>arg.ndim) == %d' % (
                                                    specialized_type.ndim,)
                    if dtype.is_int:
                        cond += ' and {{signed_match}}'

                    if final_type.is_pythran_expr:
                        cond += ' and arg_is_pythran_compatible'

                    if codewriter.indenter("if %s:" % cond):
                        #codewriter.putln("print 'buffer match found based on numpy dtype'")
                        codewriter.putln(self.match)
                        codewriter.putln("break")
                        codewriter.dedent()

    def _buffer_parse_format_string_check(self, pyx_code, decl_code,
                                          specialized_type, env):
        """
        For each specialized type, try to coerce the object to a memoryview
        slice of that type. This means obtaining a buffer and parsing the
        format string.
        TODO: separate buffer acquisition from format parsing
        """
        dtype = specialized_type.dtype
        if specialized_type.is_buffer:
            axes = [('direct', 'strided')] * specialized_type.ndim
        else:
            axes = specialized_type.axes

        memslice_type = PyrexTypes.MemoryViewSliceType(dtype, axes)
        memslice_type.create_from_py_utility_code(env)
        pyx_code.context.update(
            coerce_from_py_func=memslice_type.from_py_function,
            dtype=dtype)
        decl_code.putln(
            "{{memviewslice_cname}} {{coerce_from_py_func}}(object, int)")

        pyx_code.context.update(
            specialized_type_name=specialized_type.specialization_string,
            sizeof_dtype=self._sizeof_dtype(dtype))

        pyx_code.put_chunk(
            u"""
                # try {{dtype}}
                if itemsize == -1 or itemsize == {{sizeof_dtype}}:
                    memslice = {{coerce_from_py_func}}(arg, 0)
                    if memslice.memview:
                        __PYX_XDEC_MEMVIEW(&memslice, 1)
                        # print 'found a match for the buffer through format parsing'
                        %s
                        break
                    else:
                        __pyx_PyErr_Clear()
            """ % self.match)

    def _buffer_checks(self, buffer_types, pythran_types, pyx_code, decl_code, env):
        """
        Generate Cython code to match objects to buffer specializations.
        First try to get a numpy dtype object and match it against the individual
        specializations. If that fails, try naively to coerce the object
        to each specialization, which obtains the buffer each time and tries
        to match the format string.
        """
        # The first thing to find a match in this loop breaks out of the loop
        pyx_code.put_chunk(
            u"""
                """ + (u"arg_is_pythran_compatible = False" if pythran_types else u"") + u"""
                if ndarray is not None:
                    if isinstance(arg, ndarray):
                        dtype = arg.dtype
                        """ + (u"arg_is_pythran_compatible = True" if pythran_types else u"") + u"""
                    elif __pyx_memoryview_check(arg):
                        arg_base = arg.base
                        if isinstance(arg_base, ndarray):
                            dtype = arg_base.dtype
                        else:
                            dtype = None
                    else:
                        dtype = None

                    itemsize = -1
                    if dtype is not None:
                        itemsize = dtype.itemsize
                        kind = ord(dtype.kind)
                        dtype_signed = kind == 'i'
            """)
        pyx_code.indent(2)
        if pythran_types:
            pyx_code.put_chunk(
                u"""
                        # Pythran only supports the endianness of the current compiler
                        byteorder = dtype.byteorder
                        if byteorder == "<" and not __Pyx_Is_Little_Endian():
                            arg_is_pythran_compatible = False
                        elif byteorder == ">" and __Pyx_Is_Little_Endian():
                            arg_is_pythran_compatible = False
                        if arg_is_pythran_compatible:
                            cur_stride = itemsize
                            shape = arg.shape
                            strides = arg.strides
                            for i in range(arg.ndim-1, -1, -1):
                                if (<Py_ssize_t>strides[i]) != cur_stride:
                                    arg_is_pythran_compatible = False
                                    break
                                cur_stride *= <Py_ssize_t> shape[i]
                            else:
                                arg_is_pythran_compatible = not (arg.flags.f_contiguous and (<Py_ssize_t>arg.ndim) > 1)
                """)
        pyx_code.named_insertion_point("numpy_dtype_checks")
        self._buffer_check_numpy_dtype(pyx_code, buffer_types, pythran_types)
        pyx_code.dedent(2)

        for specialized_type in buffer_types:
            self._buffer_parse_format_string_check(
                    pyx_code, decl_code, specialized_type, env)

    def _buffer_declarations(self, pyx_code, decl_code, all_buffer_types, pythran_types):
        """
        If we have any buffer specializations, write out some variable
        declarations and imports.
        """
        decl_code.put_chunk(
            u"""
                ctypedef struct {{memviewslice_cname}}:
                    void *memview

                void __PYX_XDEC_MEMVIEW({{memviewslice_cname}} *, int have_gil)
                bint __pyx_memoryview_check(object)
            """)

        pyx_code.local_variable_declarations.put_chunk(
            u"""
                cdef {{memviewslice_cname}} memslice
                cdef Py_ssize_t itemsize
                cdef bint dtype_signed
                cdef char kind

                itemsize = -1
            """)

        if pythran_types:
            pyx_code.local_variable_declarations.put_chunk(u"""
                cdef bint arg_is_pythran_compatible
                cdef Py_ssize_t cur_stride
            """)

        pyx_code.imports.put_chunk(
            u"""
                cdef type ndarray
                ndarray = __Pyx_ImportNumPyArrayTypeIfAvailable()
            """)

        seen_typedefs = set()
        seen_int_dtypes = set()
        for buffer_type in all_buffer_types:
            dtype = buffer_type.dtype
            dtype_name = self._dtype_name(dtype)
            if dtype.is_typedef:
                if dtype_name not in seen_typedefs:
                    seen_typedefs.add(dtype_name)
                    decl_code.putln(
                        'ctypedef %s %s "%s"' % (dtype.resolve(), dtype_name,
                                                 dtype.empty_declaration_code()))

            if buffer_type.dtype.is_int:
                if str(dtype) not in seen_int_dtypes:
                    seen_int_dtypes.add(str(dtype))
                    pyx_code.context.update(dtype_name=dtype_name,
                                            dtype_type=self._dtype_type(dtype))
                    pyx_code.local_variable_declarations.put_chunk(
                        u"""
                            cdef bint {{dtype_name}}_is_signed
                            {{dtype_name}}_is_signed = not (<{{dtype_type}}> -1 > 0)
                        """)

    def _split_fused_types(self, arg):
        """
        Specialize fused types and split into normal types and buffer types.
        """
        specialized_types = PyrexTypes.get_specialized_types(arg.type)

        # Prefer long over int, etc by sorting (see type classes in PyrexTypes.py)
        specialized_types.sort()

        seen_py_type_names = set()
        normal_types, buffer_types, pythran_types = [], [], []
        has_object_fallback = False
        for specialized_type in specialized_types:
            py_type_name = specialized_type.py_type_name()
            if py_type_name:
                if py_type_name in seen_py_type_names:
                    continue
                seen_py_type_names.add(py_type_name)
                if py_type_name == 'object':
                    has_object_fallback = True
                else:
                    normal_types.append(specialized_type)
            elif specialized_type.is_pythran_expr:
                pythran_types.append(specialized_type)
            elif specialized_type.is_buffer or specialized_type.is_memoryviewslice:
                buffer_types.append(specialized_type)

        return normal_types, buffer_types, pythran_types, has_object_fallback

    def _unpack_argument(self, pyx_code):
        pyx_code.put_chunk(
            u"""
                # PROCESSING ARGUMENT {{arg_tuple_idx}}
                if {{arg_tuple_idx}} < len(<tuple>args):
                    arg = (<tuple>args)[{{arg_tuple_idx}}]
                elif kwargs is not None and '{{arg.name}}' in <dict>kwargs:
                    arg = (<dict>kwargs)['{{arg.name}}']
                else:
                {{if arg.default}}
                    arg = (<tuple>defaults)[{{default_idx}}]
                {{else}}
                    {{if arg_tuple_idx < min_positional_args}}
                        raise TypeError("Expected at least %d argument%s, got %d" % (
                            {{min_positional_args}}, {{'"s"' if min_positional_args != 1 else '""'}}, len(<tuple>args)))
                    {{else}}
                        raise TypeError("Missing keyword-only argument: '%s'" % "{{arg.default}}")
                    {{endif}}
                {{endif}}
            """)

    def make_fused_cpdef(self, orig_py_func, env, is_def):
        """
        This creates the function that is indexable from Python and does
        runtime dispatch based on the argument types. The function gets the
        arg tuple and kwargs dict (or None) and the defaults tuple
        as arguments from the Binding Fused Function's tp_call.
        """
        from . import TreeFragment, Code, UtilityCode

        fused_types = self._get_fused_base_types([
            arg.type for arg in self.node.args if arg.type.is_fused])

        context = {
            'memviewslice_cname': MemoryView.memviewslice_cname,
            'func_args': self.node.args,
            'n_fused': len(fused_types),
            'min_positional_args':
                self.node.num_required_args - self.node.num_required_kw_args
                if is_def else
                sum(1 for arg in self.node.args if arg.default is None),
            'name': orig_py_func.entry.name,
        }

        pyx_code = Code.PyxCodeWriter(context=context)
        decl_code = Code.PyxCodeWriter(context=context)
        decl_code.put_chunk(
            u"""
                cdef extern from *:
                    void __pyx_PyErr_Clear "PyErr_Clear" ()
                    type __Pyx_ImportNumPyArrayTypeIfAvailable()
                    int __Pyx_Is_Little_Endian()
            """)
        decl_code.indent()

        pyx_code.put_chunk(
            u"""
                def __pyx_fused_cpdef(signatures, args, kwargs, defaults):
                    # FIXME: use a typed signature - currently fails badly because
                    #        default arguments inherit the types we specify here!

                    dest_sig = [None] * {{n_fused}}

                    if kwargs is not None and not kwargs:
                        kwargs = None

                    cdef Py_ssize_t i

                    # instance check body
            """)

        pyx_code.indent() # indent following code to function body
        pyx_code.named_insertion_point("imports")
        pyx_code.named_insertion_point("func_defs")
        pyx_code.named_insertion_point("local_variable_declarations")

        fused_index = 0
        default_idx = 0
        all_buffer_types = OrderedSet()
        seen_fused_types = set()
        for i, arg in enumerate(self.node.args):
            if arg.type.is_fused:
                arg_fused_types = arg.type.get_fused_types()
                if len(arg_fused_types) > 1:
                    raise NotImplementedError("Determination of more than one fused base "
                                              "type per argument is not implemented.")
                fused_type = arg_fused_types[0]

            if arg.type.is_fused and fused_type not in seen_fused_types:
                seen_fused_types.add(fused_type)

                context.update(
                    arg_tuple_idx=i,
                    arg=arg,
                    dest_sig_idx=fused_index,
                    default_idx=default_idx,
                )

                normal_types, buffer_types, pythran_types, has_object_fallback = self._split_fused_types(arg)
                self._unpack_argument(pyx_code)

                # 'unrolled' loop, first match breaks out of it
                if pyx_code.indenter("while 1:"):
                    if normal_types:
                        self._fused_instance_checks(normal_types, pyx_code, env)
                    if buffer_types or pythran_types:
                        env.use_utility_code(Code.UtilityCode.load_cached("IsLittleEndian", "ModuleSetupCode.c"))
                        self._buffer_checks(buffer_types, pythran_types, pyx_code, decl_code, env)
                    if has_object_fallback:
                        pyx_code.context.update(specialized_type_name='object')
                        pyx_code.putln(self.match)
                    else:
                        pyx_code.putln(self.no_match)
                    pyx_code.putln("break")
                    pyx_code.dedent()

                fused_index += 1
                all_buffer_types.update(buffer_types)
                all_buffer_types.update(ty.org_buffer for ty in pythran_types)

            if arg.default:
                default_idx += 1

        if all_buffer_types:
            self._buffer_declarations(pyx_code, decl_code, all_buffer_types, pythran_types)
            env.use_utility_code(Code.UtilityCode.load_cached("Import", "ImportExport.c"))
            env.use_utility_code(Code.UtilityCode.load_cached("ImportNumPyArray", "ImportExport.c"))

        pyx_code.put_chunk(
            u"""
                candidates = []
                for sig in <dict>signatures:
                    match_found = False
                    src_sig = sig.strip('()').split('|')
                    for i in range(len(dest_sig)):
                        dst_type = dest_sig[i]
                        if dst_type is not None:
                            if src_sig[i] == dst_type:
                                match_found = True
                            else:
                                match_found = False
                                break

                    if match_found:
                        candidates.append(sig)

                if not candidates:
                    raise TypeError("No matching signature found")
                elif len(candidates) > 1:
                    raise TypeError("Function call with ambiguous argument types")
                else:
                    return (<dict>signatures)[candidates[0]]
            """)

        fragment_code = pyx_code.getvalue()
        # print decl_code.getvalue()
        # print fragment_code
        from .Optimize import ConstantFolding
        fragment = TreeFragment.TreeFragment(
            fragment_code, level='module', pipeline=[ConstantFolding()])
        ast = TreeFragment.SetPosTransform(self.node.pos)(fragment.root)
        UtilityCode.declare_declarations_in_scope(
            decl_code.getvalue(), env.global_scope())
        ast.scope = env
        # FIXME: for static methods of cdef classes, we build the wrong signature here: first arg becomes 'self'
        ast.analyse_declarations(env)
        py_func = ast.stats[-1]  # the DefNode
        self.fragment_scope = ast.scope

        if isinstance(self.node, DefNode):
            py_func.specialized_cpdefs = self.nodes[:]
        else:
            py_func.specialized_cpdefs = [n.py_func for n in self.nodes]

        return py_func

    def update_fused_defnode_entry(self, env):
        copy_attributes = (
            'name', 'pos', 'cname', 'func_cname', 'pyfunc_cname',
            'pymethdef_cname', 'doc', 'doc_cname', 'is_member',
            'scope'
        )

        entry = self.py_func.entry

        for attr in copy_attributes:
            setattr(entry, attr,
                    getattr(self.orig_py_func.entry, attr))

        self.py_func.name = self.orig_py_func.name
        self.py_func.doc = self.orig_py_func.doc

        env.entries.pop('__pyx_fused_cpdef', None)
        if isinstance(self.node, DefNode):
            env.entries[entry.name] = entry
        else:
            env.entries[entry.name].as_variable = entry

        env.pyfunc_entries.append(entry)

        self.py_func.entry.fused_cfunction = self
        for node in self.nodes:
            if isinstance(self.node, DefNode):
                node.fused_py_func = self.py_func
            else:
                node.py_func.fused_py_func = self.py_func
                node.entry.as_variable = entry

        self.synthesize_defnodes()
        self.stats.append(self.__signatures__)

    def analyse_expressions(self, env):
        """
        Analyse the expressions. Take care to only evaluate default arguments
        once and clone the result for all specializations
        """
        for fused_compound_type in self.fused_compound_types:
            for fused_type in fused_compound_type.get_fused_types():
                for specialization_type in fused_type.types:
                    if specialization_type.is_complex:
                        specialization_type.create_declaration_utility_code(env)

        if self.py_func:
            self.__signatures__ = self.__signatures__.analyse_expressions(env)
            self.py_func = self.py_func.analyse_expressions(env)
            self.resulting_fused_function = self.resulting_fused_function.analyse_expressions(env)
            self.fused_func_assignment = self.fused_func_assignment.analyse_expressions(env)

        self.defaults = defaults = []

        for arg in self.node.args:
            if arg.default:
                arg.default = arg.default.analyse_expressions(env)
                defaults.append(ProxyNode(arg.default))
            else:
                defaults.append(None)

        for i, stat in enumerate(self.stats):
            stat = self.stats[i] = stat.analyse_expressions(env)
            if isinstance(stat, FuncDefNode):
                for arg, default in zip(stat.args, defaults):
                    if default is not None:
                        arg.default = CloneNode(default).coerce_to(arg.type, env)

        if self.py_func:
            args = [CloneNode(default) for default in defaults if default]
            self.defaults_tuple = TupleNode(self.pos, args=args)
            self.defaults_tuple = self.defaults_tuple.analyse_types(env, skip_children=True).coerce_to_pyobject(env)
            self.defaults_tuple = ProxyNode(self.defaults_tuple)
            self.code_object = ProxyNode(self.specialized_pycfuncs[0].code_object)

            fused_func = self.resulting_fused_function.arg
            fused_func.defaults_tuple = CloneNode(self.defaults_tuple)
            fused_func.code_object = CloneNode(self.code_object)

            for i, pycfunc in enumerate(self.specialized_pycfuncs):
                pycfunc.code_object = CloneNode(self.code_object)
                pycfunc = self.specialized_pycfuncs[i] = pycfunc.analyse_types(env)
                pycfunc.defaults_tuple = CloneNode(self.defaults_tuple)
        return self

    def synthesize_defnodes(self):
        """
        Create the __signatures__ dict of PyCFunctionNode specializations.
        """
        if isinstance(self.nodes[0], CFuncDefNode):
            nodes = [node.py_func for node in self.nodes]
        else:
            nodes = self.nodes

        signatures = [StringEncoding.EncodedString(node.specialized_signature_string)
                      for node in nodes]
        keys = [ExprNodes.StringNode(node.pos, value=sig)
                for node, sig in zip(nodes, signatures)]
        values = [ExprNodes.PyCFunctionNode.from_defnode(node, binding=True)
                  for node in nodes]

        self.__signatures__ = ExprNodes.DictNode.from_pairs(self.pos, zip(keys, values))

        self.specialized_pycfuncs = values
        for pycfuncnode in values:
            pycfuncnode.is_specialization = True

    def generate_function_definitions(self, env, code):
        if self.py_func:
            self.py_func.pymethdef_required = True
            self.fused_func_assignment.generate_function_definitions(env, code)

        for stat in self.stats:
            if isinstance(stat, FuncDefNode) and stat.entry.used:
                code.mark_pos(stat.pos)
                stat.generate_function_definitions(env, code)

    def generate_execution_code(self, code):
        # Note: all def function specialization are wrapped in PyCFunction
        # nodes in the self.__signatures__ dictnode.
        for default in self.defaults:
            if default is not None:
                default.generate_evaluation_code(code)

        if self.py_func:
            self.defaults_tuple.generate_evaluation_code(code)
            self.code_object.generate_evaluation_code(code)

        for stat in self.stats:
            code.mark_pos(stat.pos)
            if isinstance(stat, ExprNodes.ExprNode):
                stat.generate_evaluation_code(code)
            else:
                stat.generate_execution_code(code)

        if self.__signatures__:
            self.resulting_fused_function.generate_evaluation_code(code)

            code.putln(
                "((__pyx_FusedFunctionObject *) %s)->__signatures__ = %s;" %
                                    (self.resulting_fused_function.result(),
                                     self.__signatures__.result()))
            code.put_giveref(self.__signatures__.result())
            self.__signatures__.generate_post_assignment_code(code)
            self.__signatures__.free_temps(code)

            self.fused_func_assignment.generate_execution_code(code)

            # Dispose of results
            self.resulting_fused_function.generate_disposal_code(code)
            self.resulting_fused_function.free_temps(code)
            self.defaults_tuple.generate_disposal_code(code)
            self.defaults_tuple.free_temps(code)
            self.code_object.generate_disposal_code(code)
            self.code_object.free_temps(code)

        for default in self.defaults:
            if default is not None:
                default.generate_disposal_code(code)
                default.free_temps(code)

    def annotate(self, code):
        for stat in self.stats:
            stat.annotate(code)

from __future__ import absolute_import

cimport cython

cdef class TreeVisitor:
    cdef public list access_path
    cdef dict dispatch_table

    cpdef visit(self, obj)
    cdef _visit(self, obj)
    cdef find_handler(self, obj)
    cdef _visitchild(self, child, parent, attrname, idx)
    cdef dict _visitchildren(self, parent, attrs)
    cpdef visitchildren(self, parent, attrs=*)
    cdef _raise_compiler_error(self, child, e)

cdef class VisitorTransform(TreeVisitor):
    cdef dict _process_children(self, parent, attrs=*)
    cpdef visitchildren(self, parent, attrs=*, exclude=*)
    cdef list _flatten_list(self, list orig_list)
    cdef list _select_attrs(self, attrs, exclude)

cdef class CythonTransform(VisitorTransform):
    cdef public context
    cdef public current_directives

cdef class ScopeTrackingTransform(CythonTransform):
    cdef public scope_type
    cdef public scope_node
    cdef visit_scope(self, node, scope_type)

cdef class EnvTransform(CythonTransform):
    cdef public list env_stack

cdef class MethodDispatcherTransform(EnvTransform):
    @cython.final
    cdef _visit_binop_node(self, node)
    @cython.final
    cdef _find_handler(self, match_name, bint has_kwargs)
    @cython.final
    cdef _delegate_to_assigned_value(self, node, function, arg_list, kwargs)
    @cython.final
    cdef _dispatch_to_handler(self, node, function, arg_list, kwargs)
    @cython.final
    cdef _dispatch_to_method_handler(self, attr_name, self_arg,
                                     is_unbound_method, type_name,
                                     node, function, arg_list, kwargs)

cdef class RecursiveNodeReplacer(VisitorTransform):
     cdef public orig_node
     cdef public new_node

cdef class NodeFinder(TreeVisitor):
    cdef node
    cdef public bint found

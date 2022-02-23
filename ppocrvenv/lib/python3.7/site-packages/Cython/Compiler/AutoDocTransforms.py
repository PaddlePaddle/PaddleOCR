from __future__ import absolute_import, print_function

from .Visitor import CythonTransform
from .StringEncoding import EncodedString
from . import Options
from . import PyrexTypes, ExprNodes
from ..CodeWriter import ExpressionWriter


class AnnotationWriter(ExpressionWriter):

    def visit_Node(self, node):
        self.put(u"<???>")

    def visit_LambdaNode(self, node):
        # XXX Should we do better?
        self.put("<lambda>")


class EmbedSignature(CythonTransform):

    def __init__(self, context):
        super(EmbedSignature, self).__init__(context)
        self.class_name = None
        self.class_node = None

    def _fmt_expr(self, node):
        writer = AnnotationWriter()
        result = writer.write(node)
        # print(type(node).__name__, '-->', result)
        return result

    def _fmt_arg(self, arg):
        if arg.type is PyrexTypes.py_object_type or arg.is_self_arg:
            doc = arg.name
        else:
            doc = arg.type.declaration_code(arg.name, for_display=1)

        if arg.annotation:
            annotation = self._fmt_expr(arg.annotation)
            doc = doc + (': %s' % annotation)
            if arg.default:
                default = self._fmt_expr(arg.default)
                doc = doc + (' = %s' % default)
        elif arg.default:
            default = self._fmt_expr(arg.default)
            doc = doc + ('=%s' % default)
        return doc

    def _fmt_star_arg(self, arg):
        arg_doc = arg.name
        if arg.annotation:
            annotation = self._fmt_expr(arg.annotation)
            arg_doc = arg_doc + (': %s' % annotation)
        return arg_doc

    def _fmt_arglist(self, args,
                     npargs=0, pargs=None,
                     nkargs=0, kargs=None,
                     hide_self=False):
        arglist = []
        for arg in args:
            if not hide_self or not arg.entry.is_self_arg:
                arg_doc = self._fmt_arg(arg)
                arglist.append(arg_doc)
        if pargs:
            arg_doc = self._fmt_star_arg(pargs)
            arglist.insert(npargs, '*%s' % arg_doc)
        elif nkargs:
            arglist.insert(npargs, '*')
        if kargs:
            arg_doc = self._fmt_star_arg(kargs)
            arglist.append('**%s' % arg_doc)
        return arglist

    def _fmt_ret_type(self, ret):
        if ret is PyrexTypes.py_object_type:
            return None
        else:
            return ret.declaration_code("", for_display=1)

    def _fmt_signature(self, cls_name, func_name, args,
                       npargs=0, pargs=None,
                       nkargs=0, kargs=None,
                       return_expr=None,
                       return_type=None, hide_self=False):
        arglist = self._fmt_arglist(args,
                                    npargs, pargs,
                                    nkargs, kargs,
                                    hide_self=hide_self)
        arglist_doc = ', '.join(arglist)
        func_doc = '%s(%s)' % (func_name, arglist_doc)
        if cls_name:
            func_doc = '%s.%s' % (cls_name, func_doc)
        ret_doc = None
        if return_expr:
            ret_doc = self._fmt_expr(return_expr)
        elif return_type:
            ret_doc = self._fmt_ret_type(return_type)
        if ret_doc:
            func_doc = '%s -> %s' % (func_doc, ret_doc)
        return func_doc

    def _embed_signature(self, signature, node_doc):
        if node_doc:
            return "%s\n%s" % (signature, node_doc)
        else:
            return signature

    def __call__(self, node):
        if not Options.docstrings:
            return node
        else:
            return super(EmbedSignature, self).__call__(node)

    def visit_ClassDefNode(self, node):
        oldname = self.class_name
        oldclass = self.class_node
        self.class_node = node
        try:
            # PyClassDefNode
            self.class_name = node.name
        except AttributeError:
            # CClassDefNode
            self.class_name = node.class_name
        self.visitchildren(node)
        self.class_name = oldname
        self.class_node = oldclass
        return node

    def visit_LambdaNode(self, node):
        # lambda expressions so not have signature or inner functions
        return node

    def visit_DefNode(self, node):
        if not self.current_directives['embedsignature']:
            return node

        is_constructor = False
        hide_self = False
        if node.entry.is_special:
            is_constructor = self.class_node and node.name == '__init__'
            if not is_constructor:
                return node
            class_name, func_name = None, self.class_name
            hide_self = True
        else:
            class_name, func_name = self.class_name, node.name

        nkargs = getattr(node, 'num_kwonly_args', 0)
        npargs = len(node.args) - nkargs
        signature = self._fmt_signature(
            class_name, func_name, node.args,
            npargs, node.star_arg,
            nkargs, node.starstar_arg,
            return_expr=node.return_type_annotation,
            return_type=None, hide_self=hide_self)
        if signature:
            if is_constructor:
                doc_holder = self.class_node.entry.type.scope
            else:
                doc_holder = node.entry

            if doc_holder.doc is not None:
                old_doc = doc_holder.doc
            elif not is_constructor and getattr(node, 'py_func', None) is not None:
                old_doc = node.py_func.entry.doc
            else:
                old_doc = None
            new_doc = self._embed_signature(signature, old_doc)
            doc_holder.doc = EncodedString(new_doc)
            if not is_constructor and getattr(node, 'py_func', None) is not None:
                node.py_func.entry.doc = EncodedString(new_doc)
        return node

    def visit_CFuncDefNode(self, node):
        if not self.current_directives['embedsignature']:
            return node
        if not node.overridable: # not cpdef FOO(...):
            return node

        signature = self._fmt_signature(
            self.class_name, node.declarator.base.name,
            node.declarator.args,
            return_type=node.return_type)
        if signature:
            if node.entry.doc is not None:
                old_doc = node.entry.doc
            elif getattr(node, 'py_func', None) is not None:
                old_doc = node.py_func.entry.doc
            else:
                old_doc = None
            new_doc = self._embed_signature(signature, old_doc)
            node.entry.doc = EncodedString(new_doc)
            if hasattr(node, 'py_func') and node.py_func is not None:
                node.py_func.entry.doc = EncodedString(new_doc)
        return node

    def visit_PropertyNode(self, node):
        if not self.current_directives['embedsignature']:
            return node

        entry = node.entry
        if entry.visibility == 'public':
            # property synthesised from a cdef public attribute
            type_name = entry.type.declaration_code("", for_display=1)
            if not entry.type.is_pyobject:
                type_name = "'%s'" % type_name
            elif entry.type.is_extension_type:
                type_name = entry.type.module_name + '.' + type_name
            signature = '%s: %s' % (entry.name, type_name)
            new_doc = self._embed_signature(signature, entry.doc)
            entry.doc = EncodedString(new_doc)
        return node

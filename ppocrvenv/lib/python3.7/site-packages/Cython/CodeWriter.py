"""
Serializes a Cython code tree to Cython code. This is primarily useful for
debugging and testing purposes.

The output is in a strict format, no whitespace or comments from the input
is preserved (and it could not be as it is not present in the code tree).
"""

from __future__ import absolute_import, print_function

from .Compiler.Visitor import TreeVisitor
from .Compiler.ExprNodes import *


class LinesResult(object):
    def __init__(self):
        self.lines = []
        self.s = u""

    def put(self, s):
        self.s += s

    def newline(self):
        self.lines.append(self.s)
        self.s = u""

    def putline(self, s):
        self.put(s)
        self.newline()

class DeclarationWriter(TreeVisitor):

    indent_string = u"    "

    def __init__(self, result=None):
        super(DeclarationWriter, self).__init__()
        if result is None:
            result = LinesResult()
        self.result = result
        self.numindents = 0
        self.tempnames = {}
        self.tempblockindex = 0

    def write(self, tree):
        self.visit(tree)
        return self.result

    def indent(self):
        self.numindents += 1

    def dedent(self):
        self.numindents -= 1

    def startline(self, s=u""):
        self.result.put(self.indent_string * self.numindents + s)

    def put(self, s):
        self.result.put(s)

    def putline(self, s):
        self.result.putline(self.indent_string * self.numindents + s)

    def endline(self, s=u""):
        self.result.putline(s)

    def line(self, s):
        self.startline(s)
        self.endline()

    def comma_separated_list(self, items, output_rhs=False):
        if len(items) > 0:
            for item in items[:-1]:
                self.visit(item)
                if output_rhs and item.default is not None:
                    self.put(u" = ")
                    self.visit(item.default)
                self.put(u", ")
            self.visit(items[-1])

    def visit_Node(self, node):
        raise AssertionError("Node not handled by serializer: %r" % node)

    def visit_ModuleNode(self, node):
        self.visitchildren(node)

    def visit_StatListNode(self, node):
        self.visitchildren(node)

    def visit_CDefExternNode(self, node):
        if node.include_file is None:
            file = u'*'
        else:
            file = u'"%s"' % node.include_file
        self.putline(u"cdef extern from %s:" % file)
        self.indent()
        self.visit(node.body)
        self.dedent()

    def visit_CPtrDeclaratorNode(self, node):
        self.put('*')
        self.visit(node.base)

    def visit_CReferenceDeclaratorNode(self, node):
        self.put('&')
        self.visit(node.base)

    def visit_CArrayDeclaratorNode(self, node):
        self.visit(node.base)
        self.put(u'[')
        if node.dimension is not None:
            self.visit(node.dimension)
        self.put(u']')

    def visit_CArrayDeclaratorNode(self, node):
        self.visit(node.base)
        self.put(u'[')
        if node.dimension is not None:
            self.visit(node.dimension)
        self.put(u']')

    def visit_CFuncDeclaratorNode(self, node):
        # TODO: except, gil, etc.
        self.visit(node.base)
        self.put(u'(')
        self.comma_separated_list(node.args)
        self.endline(u')')

    def visit_CNameDeclaratorNode(self, node):
        self.put(node.name)

    def visit_CSimpleBaseTypeNode(self, node):
        # See Parsing.p_sign_and_longness
        if node.is_basic_c_type:
            self.put(("unsigned ", "", "signed ")[node.signed])
            if node.longness < 0:
                self.put("short " * -node.longness)
            elif node.longness > 0:
                self.put("long " * node.longness)
        self.put(node.name)

    def visit_CComplexBaseTypeNode(self, node):
        self.put(u'(')
        self.visit(node.base_type)
        self.visit(node.declarator)
        self.put(u')')

    def visit_CNestedBaseTypeNode(self, node):
        self.visit(node.base_type)
        self.put(u'.')
        self.put(node.name)

    def visit_TemplatedTypeNode(self, node):
        self.visit(node.base_type_node)
        self.put(u'[')
        self.comma_separated_list(node.positional_args + node.keyword_args.key_value_pairs)
        self.put(u']')

    def visit_CVarDefNode(self, node):
        self.startline(u"cdef ")
        self.visit(node.base_type)
        self.put(u" ")
        self.comma_separated_list(node.declarators, output_rhs=True)
        self.endline()

    def visit_container_node(self, node, decl, extras, attributes):
        # TODO: visibility
        self.startline(decl)
        if node.name:
            self.put(u' ')
            self.put(node.name)
            if node.cname is not None:
                self.put(u' "%s"' % node.cname)
        if extras:
            self.put(extras)
        self.endline(':')
        self.indent()
        if not attributes:
            self.putline('pass')
        else:
            for attribute in attributes:
                self.visit(attribute)
        self.dedent()

    def visit_CStructOrUnionDefNode(self, node):
        if node.typedef_flag:
            decl = u'ctypedef '
        else:
            decl = u'cdef '
        if node.visibility == 'public':
            decl += u'public '
        if node.packed:
            decl += u'packed '
        decl += node.kind
        self.visit_container_node(node, decl, None, node.attributes)

    def visit_CppClassNode(self, node):
        extras = ""
        if node.templates:
            extras = u"[%s]" % ", ".join(node.templates)
        if node.base_classes:
            extras += "(%s)" % ", ".join(node.base_classes)
        self.visit_container_node(node, u"cdef cppclass", extras, node.attributes)

    def visit_CEnumDefNode(self, node):
        self.visit_container_node(node, u"cdef enum", None, node.items)

    def visit_CEnumDefItemNode(self, node):
        self.startline(node.name)
        if node.cname:
            self.put(u' "%s"' % node.cname)
        if node.value:
            self.put(u" = ")
            self.visit(node.value)
        self.endline()

    def visit_CClassDefNode(self, node):
        assert not node.module_name
        if node.decorators:
            for decorator in node.decorators:
                self.visit(decorator)
        self.startline(u"cdef class ")
        self.put(node.class_name)
        if node.base_class_name:
            self.put(u"(")
            if node.base_class_module:
                self.put(node.base_class_module)
                self.put(u".")
            self.put(node.base_class_name)
            self.put(u")")
        self.endline(u":")
        self.indent()
        self.visit(node.body)
        self.dedent()

    def visit_CTypeDefNode(self, node):
        self.startline(u"ctypedef ")
        self.visit(node.base_type)
        self.put(u" ")
        self.visit(node.declarator)
        self.endline()

    def visit_FuncDefNode(self, node):
        self.startline(u"def %s(" % node.name)
        self.comma_separated_list(node.args)
        self.endline(u"):")
        self.indent()
        self.visit(node.body)
        self.dedent()

    def visit_CArgDeclNode(self, node):
        if node.base_type.name is not None:
            self.visit(node.base_type)
            self.put(u" ")
        self.visit(node.declarator)
        if node.default is not None:
            self.put(u" = ")
            self.visit(node.default)

    def visit_CImportStatNode(self, node):
        self.startline(u"cimport ")
        self.put(node.module_name)
        if node.as_name:
            self.put(u" as ")
            self.put(node.as_name)
        self.endline()

    def visit_FromCImportStatNode(self, node):
        self.startline(u"from ")
        self.put(node.module_name)
        self.put(u" cimport ")
        first = True
        for pos, name, as_name, kind in node.imported_names:
            assert kind is None
            if first:
                first = False
            else:
                self.put(u", ")
            self.put(name)
            if as_name:
                self.put(u" as ")
                self.put(as_name)
        self.endline()

    def visit_NameNode(self, node):
        self.put(node.name)

    def visit_IntNode(self, node):
        self.put(node.value)

    def visit_NoneNode(self, node):
        self.put(u"None")

    def visit_NotNode(self, node):
        self.put(u"(not ")
        self.visit(node.operand)
        self.put(u")")

    def visit_DecoratorNode(self, node):
        self.startline("@")
        self.visit(node.decorator)
        self.endline()

    def visit_BinopNode(self, node):
        self.visit(node.operand1)
        self.put(u" %s " % node.operator)
        self.visit(node.operand2)

    def visit_AttributeNode(self, node):
        self.visit(node.obj)
        self.put(u".%s" % node.attribute)

    def visit_BoolNode(self, node):
        self.put(str(node.value))

    # FIXME: represent string nodes correctly
    def visit_StringNode(self, node):
        value = node.value
        if value.encoding is not None:
            value = value.encode(value.encoding)
        self.put(repr(value))

    def visit_PassStatNode(self, node):
        self.startline(u"pass")
        self.endline()

class CodeWriter(DeclarationWriter):

    def visit_SingleAssignmentNode(self, node):
        self.startline()
        self.visit(node.lhs)
        self.put(u" = ")
        self.visit(node.rhs)
        self.endline()

    def visit_CascadedAssignmentNode(self, node):
        self.startline()
        for lhs in node.lhs_list:
            self.visit(lhs)
            self.put(u" = ")
        self.visit(node.rhs)
        self.endline()

    def visit_PrintStatNode(self, node):
        self.startline(u"print ")
        self.comma_separated_list(node.arg_tuple.args)
        if not node.append_newline:
            self.put(u",")
        self.endline()

    def visit_ForInStatNode(self, node):
        self.startline(u"for ")
        self.visit(node.target)
        self.put(u" in ")
        self.visit(node.iterator.sequence)
        self.endline(u":")
        self.indent()
        self.visit(node.body)
        self.dedent()
        if node.else_clause is not None:
            self.line(u"else:")
            self.indent()
            self.visit(node.else_clause)
            self.dedent()

    def visit_IfStatNode(self, node):
        # The IfClauseNode is handled directly without a separate match
        # for clariy.
        self.startline(u"if ")
        self.visit(node.if_clauses[0].condition)
        self.endline(":")
        self.indent()
        self.visit(node.if_clauses[0].body)
        self.dedent()
        for clause in node.if_clauses[1:]:
            self.startline("elif ")
            self.visit(clause.condition)
            self.endline(":")
            self.indent()
            self.visit(clause.body)
            self.dedent()
        if node.else_clause is not None:
            self.line("else:")
            self.indent()
            self.visit(node.else_clause)
            self.dedent()

    def visit_SequenceNode(self, node):
        self.comma_separated_list(node.args) # Might need to discover whether we need () around tuples...hmm...

    def visit_SimpleCallNode(self, node):
        self.visit(node.function)
        self.put(u"(")
        self.comma_separated_list(node.args)
        self.put(")")

    def visit_GeneralCallNode(self, node):
        self.visit(node.function)
        self.put(u"(")
        posarg = node.positional_args
        if isinstance(posarg, AsTupleNode):
            self.visit(posarg.arg)
        else:
            self.comma_separated_list(posarg.args)  # TupleNode.args
        if node.keyword_args:
            if isinstance(node.keyword_args, DictNode):
                for i, (name, value) in enumerate(node.keyword_args.key_value_pairs):
                    if i > 0:
                        self.put(', ')
                    self.visit(name)
                    self.put('=')
                    self.visit(value)
            else:
                raise Exception("Not implemented yet")
        self.put(u")")

    def visit_ExprStatNode(self, node):
        self.startline()
        self.visit(node.expr)
        self.endline()

    def visit_InPlaceAssignmentNode(self, node):
        self.startline()
        self.visit(node.lhs)
        self.put(u" %s= " % node.operator)
        self.visit(node.rhs)
        self.endline()

    def visit_WithStatNode(self, node):
        self.startline()
        self.put(u"with ")
        self.visit(node.manager)
        if node.target is not None:
            self.put(u" as ")
            self.visit(node.target)
        self.endline(u":")
        self.indent()
        self.visit(node.body)
        self.dedent()

    def visit_TryFinallyStatNode(self, node):
        self.line(u"try:")
        self.indent()
        self.visit(node.body)
        self.dedent()
        self.line(u"finally:")
        self.indent()
        self.visit(node.finally_clause)
        self.dedent()

    def visit_TryExceptStatNode(self, node):
        self.line(u"try:")
        self.indent()
        self.visit(node.body)
        self.dedent()
        for x in node.except_clauses:
            self.visit(x)
        if node.else_clause is not None:
            self.visit(node.else_clause)

    def visit_ExceptClauseNode(self, node):
        self.startline(u"except")
        if node.pattern is not None:
            self.put(u" ")
            self.visit(node.pattern)
        if node.target is not None:
            self.put(u", ")
            self.visit(node.target)
        self.endline(":")
        self.indent()
        self.visit(node.body)
        self.dedent()

    def visit_ReturnStatNode(self, node):
        self.startline("return ")
        self.visit(node.value)
        self.endline()

    def visit_ReraiseStatNode(self, node):
        self.line("raise")

    def visit_ImportNode(self, node):
        self.put(u"(import %s)" % node.module_name.value)

    def visit_TempsBlockNode(self, node):
        """
        Temporaries are output like $1_1', where the first number is
        an index of the TempsBlockNode and the second number is an index
        of the temporary which that block allocates.
        """
        idx = 0
        for handle in node.temps:
            self.tempnames[handle] = "$%d_%d" % (self.tempblockindex, idx)
            idx += 1
        self.tempblockindex += 1
        self.visit(node.body)

    def visit_TempRefNode(self, node):
        self.put(self.tempnames[node.handle])


class PxdWriter(DeclarationWriter):
    def __call__(self, node):
        print(u'\n'.join(self.write(node).lines))
        return node

    def visit_CFuncDefNode(self, node):
        if 'inline' in node.modifiers:
            return
        if node.overridable:
            self.startline(u'cpdef ')
        else:
            self.startline(u'cdef ')
        if node.visibility != 'private':
            self.put(node.visibility)
            self.put(u' ')
        if node.api:
            self.put(u'api ')
        self.visit(node.declarator)

    def visit_StatNode(self, node):
        pass


class ExpressionWriter(TreeVisitor):

    def __init__(self, result=None):
        super(ExpressionWriter, self).__init__()
        if result is None:
            result = u""
        self.result = result
        self.precedence = [0]

    def write(self, tree):
        self.visit(tree)
        return self.result

    def put(self, s):
        self.result += s

    def remove(self, s):
        if self.result.endswith(s):
            self.result = self.result[:-len(s)]

    def comma_separated_list(self, items):
        if len(items) > 0:
            for item in items[:-1]:
                self.visit(item)
                self.put(u", ")
            self.visit(items[-1])

    def visit_Node(self, node):
        raise AssertionError("Node not handled by serializer: %r" % node)

    def visit_NameNode(self, node):
        self.put(node.name)

    def visit_NoneNode(self, node):
        self.put(u"None")

    def visit_EllipsisNode(self, node):
        self.put(u"...")

    def visit_BoolNode(self, node):
        self.put(str(node.value))

    def visit_ConstNode(self, node):
        self.put(str(node.value))

    def visit_ImagNode(self, node):
        self.put(node.value)
        self.put(u"j")

    def emit_string(self, node, prefix=u""):
        repr_val = repr(node.value)
        if repr_val[0] in 'ub':
            repr_val = repr_val[1:]
        self.put(u"%s%s" % (prefix, repr_val))

    def visit_BytesNode(self, node):
        self.emit_string(node, u"b")

    def visit_StringNode(self, node):
        self.emit_string(node)

    def visit_UnicodeNode(self, node):
        self.emit_string(node, u"u")

    def emit_sequence(self, node, parens=(u"", u"")):
        open_paren, close_paren = parens
        items = node.subexpr_nodes()
        self.put(open_paren)
        self.comma_separated_list(items)
        self.put(close_paren)

    def visit_ListNode(self, node):
        self.emit_sequence(node, u"[]")

    def visit_TupleNode(self, node):
        self.emit_sequence(node, u"()")

    def visit_SetNode(self, node):
        if len(node.subexpr_nodes()) > 0:
            self.emit_sequence(node, u"{}")
        else:
            self.put(u"set()")

    def visit_DictNode(self, node):
        self.emit_sequence(node, u"{}")

    def visit_DictItemNode(self, node):
        self.visit(node.key)
        self.put(u": ")
        self.visit(node.value)

    unop_precedence = {
        'not': 3, '!': 3,
        '+': 11, '-': 11, '~': 11,
    }
    binop_precedence = {
        'or': 1,
        'and': 2,
        # unary: 'not': 3, '!': 3,
        'in': 4, 'not_in': 4, 'is': 4, 'is_not': 4, '<': 4, '<=': 4, '>': 4, '>=': 4, '!=': 4, '==': 4,
        '|': 5,
        '^': 6,
        '&': 7,
        '<<': 8, '>>': 8,
        '+': 9, '-': 9,
        '*': 10, '@': 10, '/': 10, '//': 10, '%': 10,
        # unary: '+': 11, '-': 11, '~': 11
        '**': 12,
    }

    def operator_enter(self, new_prec):
        old_prec = self.precedence[-1]
        if old_prec > new_prec:
            self.put(u"(")
        self.precedence.append(new_prec)

    def operator_exit(self):
        old_prec, new_prec = self.precedence[-2:]
        if old_prec > new_prec:
            self.put(u")")
        self.precedence.pop()

    def visit_NotNode(self, node):
        op = 'not'
        prec = self.unop_precedence[op]
        self.operator_enter(prec)
        self.put(u"not ")
        self.visit(node.operand)
        self.operator_exit()

    def visit_UnopNode(self, node):
        op = node.operator
        prec = self.unop_precedence[op]
        self.operator_enter(prec)
        self.put(u"%s" % node.operator)
        self.visit(node.operand)
        self.operator_exit()

    def visit_BinopNode(self, node):
        op = node.operator
        prec = self.binop_precedence.get(op, 0)
        self.operator_enter(prec)
        self.visit(node.operand1)
        self.put(u" %s " % op.replace('_', ' '))
        self.visit(node.operand2)
        self.operator_exit()

    def visit_BoolBinopNode(self, node):
        self.visit_BinopNode(node)

    def visit_PrimaryCmpNode(self, node):
        self.visit_BinopNode(node)

    def visit_IndexNode(self, node):
        self.visit(node.base)
        self.put(u"[")
        if isinstance(node.index, TupleNode):
            self.emit_sequence(node.index)
        else:
            self.visit(node.index)
        self.put(u"]")

    def visit_SliceIndexNode(self, node):
        self.visit(node.base)
        self.put(u"[")
        if node.start:
            self.visit(node.start)
        self.put(u":")
        if node.stop:
            self.visit(node.stop)
        if node.slice:
            self.put(u":")
            self.visit(node.slice)
        self.put(u"]")

    def visit_SliceNode(self, node):
        if not node.start.is_none:
            self.visit(node.start)
        self.put(u":")
        if not node.stop.is_none:
            self.visit(node.stop)
        if not node.step.is_none:
            self.put(u":")
            self.visit(node.step)

    def visit_CondExprNode(self, node):
        self.visit(node.true_val)
        self.put(u" if ")
        self.visit(node.test)
        self.put(u" else ")
        self.visit(node.false_val)

    def visit_AttributeNode(self, node):
        self.visit(node.obj)
        self.put(u".%s" % node.attribute)

    def visit_SimpleCallNode(self, node):
        self.visit(node.function)
        self.put(u"(")
        self.comma_separated_list(node.args)
        self.put(")")

    def emit_pos_args(self, node):
        if node is None:
            return
        if isinstance(node, AddNode):
            self.emit_pos_args(node.operand1)
            self.emit_pos_args(node.operand2)
        elif isinstance(node, TupleNode):
            for expr in node.subexpr_nodes():
                self.visit(expr)
                self.put(u", ")
        elif isinstance(node, AsTupleNode):
            self.put("*")
            self.visit(node.arg)
            self.put(u", ")
        else:
            self.visit(node)
            self.put(u", ")

    def emit_kwd_args(self, node):
        if node is None:
            return
        if isinstance(node, MergedDictNode):
            for expr in node.subexpr_nodes():
                self.emit_kwd_args(expr)
        elif isinstance(node, DictNode):
            for expr in node.subexpr_nodes():
                self.put(u"%s=" % expr.key.value)
                self.visit(expr.value)
                self.put(u", ")
        else:
            self.put(u"**")
            self.visit(node)
            self.put(u", ")

    def visit_GeneralCallNode(self, node):
        self.visit(node.function)
        self.put(u"(")
        self.emit_pos_args(node.positional_args)
        self.emit_kwd_args(node.keyword_args)
        self.remove(u", ")
        self.put(")")

    def emit_comprehension(self, body, target,
                           sequence, condition,
                           parens=(u"", u"")):
        open_paren, close_paren = parens
        self.put(open_paren)
        self.visit(body)
        self.put(u" for ")
        self.visit(target)
        self.put(u" in ")
        self.visit(sequence)
        if condition:
            self.put(u" if ")
            self.visit(condition)
        self.put(close_paren)

    def visit_ComprehensionAppendNode(self, node):
        self.visit(node.expr)

    def visit_DictComprehensionAppendNode(self, node):
        self.visit(node.key_expr)
        self.put(u": ")
        self.visit(node.value_expr)

    def visit_ComprehensionNode(self, node):
        tpmap = {'list': u"[]", 'dict': u"{}", 'set': u"{}"}
        parens = tpmap[node.type.py_type_name()]
        body = node.loop.body
        target = node.loop.target
        sequence = node.loop.iterator.sequence
        condition = None
        if hasattr(body, 'if_clauses'):
            # type(body) is Nodes.IfStatNode
            condition = body.if_clauses[0].condition
            body = body.if_clauses[0].body
        self.emit_comprehension(body, target, sequence, condition, parens)

    def visit_GeneratorExpressionNode(self, node):
        body = node.loop.body
        target = node.loop.target
        sequence = node.loop.iterator.sequence
        condition = None
        if hasattr(body, 'if_clauses'):
            # type(body) is Nodes.IfStatNode
            condition = body.if_clauses[0].condition
            body = body.if_clauses[0].body.expr.arg
        elif hasattr(body, 'expr'):
            # type(body) is Nodes.ExprStatNode
            body = body.expr.arg
        self.emit_comprehension(body, target, sequence, condition, u"()")

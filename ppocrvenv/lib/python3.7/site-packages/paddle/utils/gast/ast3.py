# Copyright (c) 2016, Serge Guelton
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 	Redistributions of source code must retain the above copyright notice, this
# 	list of conditions and the following disclaimer.

# 	Redistributions in binary form must reproduce the above copyright notice,
# 	this list of conditions and the following disclaimer in the documentation
# 	and/or other materials provided with the distribution.

# 	Neither the name of HPCProject, Serge Guelton nor the names of its
# 	contributors may be used to endorse or promote products derived from this
# 	software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# NOTE(paddle-dev): We introduce third-party library Gast as unified AST
# representation. See https://github.com/serge-sans-paille/gast for details.

from .astn import AstToGAst, GAstToAst
from . import gast
import ast
import sys


class Ast3ToGAst(AstToGAst):
    if sys.version_info.minor < 9:

        def visit_ExtSlice(self, node):
            new_node = gast.Tuple(self._visit(node.dims), gast.Load())
            gast.copy_location(new_node, node)
            return new_node

        def visit_Index(self, node):
            return self._visit(node.value)

    if sys.version_info.minor < 8:

        def visit_Module(self, node):
            new_node = gast.Module(
                self._visit(node.body),
                []  # type_ignores
            )
            return new_node

        def visit_Num(self, node):
            new_node = gast.Constant(
                node.n,
                None, )
            gast.copy_location(new_node, node)
            return new_node

        def visit_Ellipsis(self, node):
            new_node = gast.Constant(
                Ellipsis,
                None, )
            gast.copy_location(new_node, node)
            new_node.end_lineno = new_node.end_col_offset = None
            return new_node

        def visit_Str(self, node):
            new_node = gast.Constant(
                node.s,
                None, )
            gast.copy_location(new_node, node)
            return new_node

        def visit_Bytes(self, node):
            new_node = gast.Constant(
                node.s,
                None, )
            gast.copy_location(new_node, node)
            return new_node

        def visit_FunctionDef(self, node):
            new_node = gast.FunctionDef(
                self._visit(node.name),
                self._visit(node.args),
                self._visit(node.body),
                self._visit(node.decorator_list),
                self._visit(node.returns),
                None,  # type_comment
            )
            gast.copy_location(new_node, node)
            return new_node

        def visit_AsyncFunctionDef(self, node):
            new_node = gast.AsyncFunctionDef(
                self._visit(node.name),
                self._visit(node.args),
                self._visit(node.body),
                self._visit(node.decorator_list),
                self._visit(node.returns),
                None,  # type_comment
            )
            gast.copy_location(new_node, node)
            return new_node

        def visit_For(self, node):
            new_node = gast.For(
                self._visit(node.target),
                self._visit(node.iter),
                self._visit(node.body),
                self._visit(node.orelse),
                None,  # type_comment
            )
            gast.copy_location(new_node, node)
            return new_node

        def visit_AsyncFor(self, node):
            new_node = gast.AsyncFor(
                self._visit(node.target),
                self._visit(node.iter),
                self._visit(node.body),
                self._visit(node.orelse),
                None,  # type_comment
            )
            gast.copy_location(new_node, node)
            return new_node

        def visit_With(self, node):
            new_node = gast.With(
                self._visit(node.items),
                self._visit(node.body),
                None,  # type_comment
            )
            gast.copy_location(new_node, node)
            return new_node

        def visit_AsyncWith(self, node):
            new_node = gast.AsyncWith(
                self._visit(node.items),
                self._visit(node.body),
                None,  # type_comment
            )
            gast.copy_location(new_node, node)
            return new_node

        def visit_Call(self, node):
            if sys.version_info.minor < 5:
                if node.starargs:
                    star = gast.Starred(self._visit(node.starargs), gast.Load())
                    gast.copy_location(star, node)
                    starred = [star]
                else:
                    starred = []

                if node.kwargs:
                    kw = gast.keyword(None, self._visit(node.kwargs))
                    gast.copy_location(kw, node.kwargs)
                    kwargs = [kw]
                else:
                    kwargs = []
            else:
                starred = kwargs = []

            new_node = gast.Call(
                self._visit(node.func),
                self._visit(node.args) + starred,
                self._visit(node.keywords) + kwargs, )
            gast.copy_location(new_node, node)
            return new_node

        def visit_NameConstant(self, node):
            if node.value is None:
                new_node = gast.Constant(None, None)
            elif node.value is True:
                new_node = gast.Constant(True, None)
            elif node.value is False:
                new_node = gast.Constant(False, None)
            gast.copy_location(new_node, node)
            return new_node

        def visit_arguments(self, node):
            new_node = gast.arguments(
                self._visit(node.args),
                [],  # posonlyargs
                self._visit(node.vararg),
                self._visit(node.kwonlyargs),
                self._visit(node.kw_defaults),
                self._visit(node.kwarg),
                self._visit(node.defaults), )
            gast.copy_location(new_node, node)
            return new_node

    def visit_Name(self, node):
        new_node = gast.Name(
            self._visit(node.id),
            self._visit(node.ctx),
            None,
            None, )
        ast.copy_location(new_node, node)
        return new_node

    def visit_arg(self, node):
        if sys.version_info.minor < 8:
            extra_args = [None]
        else:
            extra_args = [self._visit(node.type_comment)]

        new_node = gast.Name(
            self._visit(node.arg),
            gast.Param(),
            self._visit(node.annotation),
            *extra_args  # type_comment
        )
        ast.copy_location(new_node, node)
        return new_node

    def visit_ExceptHandler(self, node):
        if node.name:
            new_node = gast.ExceptHandler(
                self._visit(node.type),
                gast.Name(node.name, gast.Store(), None, None),
                self._visit(node.body))
            ast.copy_location(new_node, node)
            return new_node
        else:
            return self.generic_visit(node)

    if sys.version_info.minor < 6:

        def visit_comprehension(self, node):
            new_node = gast.comprehension(
                target=self._visit(node.target),
                iter=self._visit(node.iter),
                ifs=self._visit(node.ifs),
                is_async=0, )
            return ast.copy_location(new_node, node)


class GAstToAst3(GAstToAst):
    if sys.version_info.minor < 9:

        def visit_Subscript(self, node):
            def adjust_slice(s):
                if isinstance(s, ast.Slice):
                    return s
                else:
                    return ast.Index(s)

            if isinstance(node.slice, gast.Tuple):
                if any(isinstance(elt, gast.slice) for elt in node.slice.elts):
                    new_slice = ast.ExtSlice([
                        adjust_slice(x) for x in self._visit(node.slice.elts)
                    ])
                else:
                    value = ast.Tuple(self._visit(node.slice.elts), ast.Load())
                    ast.copy_location(value, node.slice)
                    new_slice = ast.Index(value)
            else:
                new_slice = adjust_slice(self._visit(node.slice))
            ast.copy_location(new_slice, node.slice)

            new_node = ast.Subscript(
                self._visit(node.value),
                new_slice,
                self._visit(node.ctx), )
            ast.copy_location(new_node, node)
            return new_node

    if sys.version_info.minor < 8:

        def visit_Module(self, node):
            new_node = ast.Module(self._visit(node.body))
            return new_node

        def visit_Constant(self, node):
            if node.value is None:
                new_node = ast.NameConstant(node.value)
            elif node.value is Ellipsis:
                new_node = ast.Ellipsis()
            elif isinstance(node.value, bool):
                new_node = ast.NameConstant(node.value)
            elif isinstance(node.value, (int, float, complex)):
                new_node = ast.Num(node.value)
            elif isinstance(node.value, str):
                new_node = ast.Str(node.value)
            else:
                new_node = ast.Bytes(node.value)
            ast.copy_location(new_node, node)
            return new_node

    def _make_arg(self, node):
        if node is None:
            return None

        if sys.version_info.minor < 8:
            extra_args = tuple()
        else:
            extra_args = self._visit(node.type_comment),

        new_node = ast.arg(
            self._visit(node.id), self._visit(node.annotation), *extra_args)
        return ast.copy_location(new_node, node)

    def visit_Name(self, node):
        new_node = ast.Name(
            self._visit(node.id),
            self._visit(node.ctx), )
        ast.copy_location(new_node, node)
        return new_node

    def visit_ExceptHandler(self, node):
        if node.name:
            new_node = ast.ExceptHandler(
                self._visit(node.type), node.name.id, self._visit(node.body))
            return ast.copy_location(new_node, node)
        else:
            return self.generic_visit(node)

    if sys.version_info.minor < 5:

        def visit_Call(self, node):
            if node.args and isinstance(node.args[-1], gast.Starred):
                args = node.args[:-1]
                starargs = node.args[-1].value
            else:
                args = node.args
                starargs = None

            if node.keywords and node.keywords[-1].arg is None:
                keywords = node.keywords[:-1]
                kwargs = node.keywords[-1].value
            else:
                keywords = node.keywords
                kwargs = None

            new_node = ast.Call(
                self._visit(node.func),
                self._visit(args),
                self._visit(keywords),
                self._visit(starargs),
                self._visit(kwargs), )
            ast.copy_location(new_node, node)
            return new_node

        def visit_ClassDef(self, node):
            self.generic_visit(node)
            new_node = ast.ClassDef(
                name=self._visit(node.name),
                bases=self._visit(node.bases),
                keywords=self._visit(node.keywords),
                body=self._visit(node.body),
                decorator_list=self._visit(node.decorator_list),
                starargs=None,
                kwargs=None, )
            return ast.copy_location(new_node, node)

    elif sys.version_info.minor < 8:

        def visit_FunctionDef(self, node):
            new_node = ast.FunctionDef(
                self._visit(node.name),
                self._visit(node.args),
                self._visit(node.body),
                self._visit(node.decorator_list),
                self._visit(node.returns), )
            ast.copy_location(new_node, node)
            return new_node

        def visit_AsyncFunctionDef(self, node):
            new_node = ast.AsyncFunctionDef(
                self._visit(node.name),
                self._visit(node.args),
                self._visit(node.body),
                self._visit(node.decorator_list),
                self._visit(node.returns), )
            ast.copy_location(new_node, node)
            return new_node

        def visit_For(self, node):
            new_node = ast.For(
                self._visit(node.target),
                self._visit(node.iter),
                self._visit(node.body),
                self._visit(node.orelse), )
            ast.copy_location(new_node, node)
            return new_node

        def visit_AsyncFor(self, node):
            new_node = ast.AsyncFor(
                self._visit(node.target),
                self._visit(node.iter),
                self._visit(node.body),
                self._visit(node.orelse),
                None,  # type_comment
            )
            ast.copy_location(new_node, node)
            return new_node

        def visit_With(self, node):
            new_node = ast.With(
                self._visit(node.items),
                self._visit(node.body), )
            ast.copy_location(new_node, node)
            return new_node

        def visit_AsyncWith(self, node):
            new_node = ast.AsyncWith(
                self._visit(node.items),
                self._visit(node.body), )
            ast.copy_location(new_node, node)
            return new_node

        def visit_Call(self, node):
            new_node = ast.Call(
                self._visit(node.func),
                self._visit(node.args),
                self._visit(node.keywords), )
            ast.copy_location(new_node, node)
            return new_node

    def visit_arguments(self, node):
        extra_args = [
            self._make_arg(node.vararg),
            [self._make_arg(n) for n in node.kwonlyargs],
            self._visit(node.kw_defaults),
            self._make_arg(node.kwarg),
            self._visit(node.defaults),
        ]
        if sys.version_info.minor >= 8:
            new_node = ast.arguments(
                [self._make_arg(arg) for arg in node.posonlyargs],
                [self._make_arg(n) for n in node.args], *extra_args)
        else:
            new_node = ast.arguments([self._make_arg(n) for n in node.args],
                                     *extra_args)
        return new_node


def ast_to_gast(node):
    return Ast3ToGAst().visit(node)


def gast_to_ast(node):
    return GAstToAst3().visit(node)

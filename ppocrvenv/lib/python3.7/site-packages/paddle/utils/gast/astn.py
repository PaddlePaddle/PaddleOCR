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

import ast
from . import gast


def _generate_translators(to):
    class Translator(ast.NodeTransformer):
        def _visit(self, node):
            if isinstance(node, list):
                return [self._visit(n) for n in node]
            elif isinstance(node, ast.AST):
                return self.visit(node)
            else:
                return node

        def generic_visit(self, node):
            cls = type(node).__name__
            # handle nodes that are not part of the AST
            if not hasattr(to, cls):
                return
            new_node = getattr(to, cls)()
            for field in node._fields:
                setattr(new_node, field, self._visit(getattr(node, field)))
            for attr in getattr(node, '_attributes'):
                if hasattr(node, attr):
                    setattr(new_node, attr, getattr(node, attr))
            return new_node

    return Translator


AstToGAst = _generate_translators(gast)

GAstToAst = _generate_translators(ast)

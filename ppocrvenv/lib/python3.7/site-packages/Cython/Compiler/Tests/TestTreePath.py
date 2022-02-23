import unittest
from Cython.Compiler.Visitor import PrintTree
from Cython.TestUtils import TransformTest
from Cython.Compiler.TreePath import find_first, find_all
from Cython.Compiler import Nodes, ExprNodes

class TestTreePath(TransformTest):
    _tree = None

    def _build_tree(self):
        if self._tree is None:
            self._tree = self.run_pipeline([], u"""
            def decorator(fun):  # DefNode
                return fun       # ReturnStatNode, NameNode
            @decorator           # NameNode
            def decorated():     # DefNode
                pass
            """)
        return self._tree

    def test_node_path(self):
        t = self._build_tree()
        self.assertEqual(2, len(find_all(t, "//DefNode")))
        self.assertEqual(2, len(find_all(t, "//NameNode")))
        self.assertEqual(1, len(find_all(t, "//ReturnStatNode")))
        self.assertEqual(1, len(find_all(t, "//DefNode//ReturnStatNode")))

    def test_node_path_star(self):
        t = self._build_tree()
        self.assertEqual(10, len(find_all(t, "//*")))
        self.assertEqual(8, len(find_all(t, "//DefNode//*")))
        self.assertEqual(0, len(find_all(t, "//NameNode//*")))

    def test_node_path_attribute(self):
        t = self._build_tree()
        self.assertEqual(2, len(find_all(t, "//NameNode/@name")))
        self.assertEqual(['fun', 'decorator'], find_all(t, "//NameNode/@name"))

    def test_node_path_attribute_dotted(self):
        t = self._build_tree()
        self.assertEqual(1, len(find_all(t, "//ReturnStatNode/@value.name")))
        self.assertEqual(['fun'], find_all(t, "//ReturnStatNode/@value.name"))

    def test_node_path_child(self):
        t = self._build_tree()
        self.assertEqual(1, len(find_all(t, "//DefNode/ReturnStatNode/NameNode")))
        self.assertEqual(1, len(find_all(t, "//ReturnStatNode/NameNode")))

    def test_node_path_node_predicate(self):
        t = self._build_tree()
        self.assertEqual(0, len(find_all(t, "//DefNode[.//ForInStatNode]")))
        self.assertEqual(2, len(find_all(t, "//DefNode[.//NameNode]")))
        self.assertEqual(1, len(find_all(t, "//ReturnStatNode[./NameNode]")))
        self.assertEqual(Nodes.ReturnStatNode,
                         type(find_first(t, "//ReturnStatNode[./NameNode]")))

    def test_node_path_node_predicate_step(self):
        t = self._build_tree()
        self.assertEqual(2, len(find_all(t, "//DefNode[.//NameNode]")))
        self.assertEqual(8, len(find_all(t, "//DefNode[.//NameNode]//*")))
        self.assertEqual(1, len(find_all(t, "//DefNode[.//NameNode]//ReturnStatNode")))
        self.assertEqual(Nodes.ReturnStatNode,
                         type(find_first(t, "//DefNode[.//NameNode]//ReturnStatNode")))

    def test_node_path_attribute_exists(self):
        t = self._build_tree()
        self.assertEqual(2, len(find_all(t, "//NameNode[@name]")))
        self.assertEqual(ExprNodes.NameNode,
                         type(find_first(t, "//NameNode[@name]")))

    def test_node_path_attribute_exists_not(self):
        t = self._build_tree()
        self.assertEqual(0, len(find_all(t, "//NameNode[not(@name)]")))
        self.assertEqual(2, len(find_all(t, "//NameNode[not(@honking)]")))

    def test_node_path_and(self):
        t = self._build_tree()
        self.assertEqual(1, len(find_all(t, "//DefNode[.//ReturnStatNode and .//NameNode]")))
        self.assertEqual(0, len(find_all(t, "//NameNode[@honking and @name]")))
        self.assertEqual(0, len(find_all(t, "//NameNode[@name and @honking]")))
        self.assertEqual(2, len(find_all(t, "//DefNode[.//NameNode[@name] and @name]")))

    def test_node_path_attribute_string_predicate(self):
        t = self._build_tree()
        self.assertEqual(1, len(find_all(t, "//NameNode[@name = 'decorator']")))

    def test_node_path_recursive_predicate(self):
        t = self._build_tree()
        self.assertEqual(2, len(find_all(t, "//DefNode[.//NameNode[@name]]")))
        self.assertEqual(1, len(find_all(t, "//DefNode[.//NameNode[@name = 'decorator']]")))
        self.assertEqual(1, len(find_all(t, "//DefNode[.//ReturnStatNode[./NameNode[@name = 'fun']]/NameNode]")))

if __name__ == '__main__':
    unittest.main()

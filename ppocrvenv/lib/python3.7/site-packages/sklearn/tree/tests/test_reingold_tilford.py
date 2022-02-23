import numpy as np
import pytest
from sklearn.tree._reingold_tilford import buchheim, Tree

simple_tree = Tree("", 0, Tree("", 1), Tree("", 2))

bigger_tree = Tree(
    "",
    0,
    Tree(
        "",
        1,
        Tree("", 3),
        Tree("", 4, Tree("", 7), Tree("", 8)),
    ),
    Tree("", 2, Tree("", 5), Tree("", 6)),
)


@pytest.mark.parametrize("tree, n_nodes", [(simple_tree, 3), (bigger_tree, 9)])
def test_buchheim(tree, n_nodes):
    def walk_tree(draw_tree):
        res = [(draw_tree.x, draw_tree.y)]
        for child in draw_tree.children:
            # parents higher than children:
            assert child.y == draw_tree.y + 1
            res.extend(walk_tree(child))
        if len(draw_tree.children):
            # these trees are always binary
            # parents are centered above children
            assert (
                draw_tree.x == (draw_tree.children[0].x + draw_tree.children[1].x) / 2
            )
        return res

    layout = buchheim(tree)
    coordinates = walk_tree(layout)
    assert len(coordinates) == n_nodes
    # test that x values are unique per depth / level
    # we could also do it quicker using defaultdicts..
    depth = 0
    while True:
        x_at_this_depth = [node[0] for node in coordinates if node[1] == depth]
        if not x_at_this_depth:
            # reached all leafs
            break
        assert len(np.unique(x_at_this_depth)) == len(x_at_this_depth)
        depth += 1

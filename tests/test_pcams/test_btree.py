import numpy as np

from pcams.btree import share_edge, BTreeNode
from pcams.common import count


def test_share_edge():
    assert share_edge([None, None], [None, 0.], [1., 0.], [4.0, 3.0])

    assert not share_edge([None, None], [None, -1.], [1., 0.], [4.0, 3.0])

    assert share_edge([-1., -1., -1.], [None, 0.], [1., 0.], [4.0, 3.0])


def test_btree_25():
    node = BTreeNode.create_root(2, 1)
    items = [([x, y], str(x) + str(y)) for x in xrange(5) for y in xrange(5)]
    node.add_batch(items)

    assert count(node.get_leaves()) == 5 * 5


def test_btree():
    node = BTreeNode.create_root(2, 1)
    node.add(np.array([-1., 0.]), 'a')

    assert count(node.get_leaves()) == 1

    node.add(np.array([1., 0.]), 'b')

    assert count(node.get_leaves()) == 2

    node.add(np.array([2., 0.]), 'c')

    assert count(node.get_leaves()) == 3

    leaf = node.get_leaf(np.array([2., 1000.]))
    assert count(leaf.get_adjacents()) == 1

    leaf = node.get_leaf(np.array([1., -1000.]))
    assert count(leaf.get_adjacents()) == 2

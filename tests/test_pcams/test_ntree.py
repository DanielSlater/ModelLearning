import numpy as np

from pcams.common import count
from pcams.ntree import binary_combinations, combinations_between_bool_vectors, NTreeNode, share_edge
from pcams.btree import BTreeNode


def test_combinations():
    result = set(binary_combinations(4))
    assert len(result) == 4 ** 2


def test_combinations_between_bool_vectors():
    result = set(combinations_between_bool_vectors([False, True, False, False], [True, True, False, True]))
    assert len(result) == 2 * 1 * 1 * 2
    assert (False, True, False, True) in result


def test_tree_growth():
    node = NTreeNode(np.array([-10.]), np.array([10.]), 1)
    assert count(node.get_leaves()) == 1

    node.add(np.array([-5.]), 'a')

    assert count(node.get_leaves()) == 1

    node.add(np.array([-3.]), 'b')

    assert count(node.get_leaves()) == 2


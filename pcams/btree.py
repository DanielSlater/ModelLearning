from collections import defaultdict
from itertools import chain

import numpy as np
import sys

from pcams.common import MAX_DIMENSIONS, combinations_between_bool_vectors, mean_squared_error, get_similar_states, count


class BTreeNode(object):
    def __init__(self, center_vector, capacity, parent, min_vector=None, max_vector=None):
        assert isinstance(center_vector, np.ndarray)
        assert capacity > 0
        assert len(center_vector) <= MAX_DIMENSIONS
        if min_vector is not None:
            assert isinstance(min_vector, np.ndarray)
            assert isinstance(max_vector, np.ndarray)
            assert all(min_ is None or max_ is None or min_ < max_ for min_, max_ in zip(min_vector, max_vector))

        self._capacity = capacity
        self._center_vector = center_vector
        self._child_low = None
        self._child_high = None
        self._split_dimension = None
        self._split_value = None
        self._population = []
        self._parent = parent
        self._adjacents = defaultdict(set)
        self._min_vector = min_vector if min_vector is not None else np.array([None for _ in xrange(self.dimensions)])
        self._max_vector = max_vector if max_vector is not None else np.array([None for _ in xrange(self.dimensions)])

        self._min_vector.setflags(write=False)
        self._max_vector.setflags(write=False)
        self._center_vector.setflags(write=False)

    @property
    def population_size(self):
        return len(self._population)

    def __str__(self):
        return "[BTreeNode{}]".format(self.__repr__())

    def __repr__(self):
        return "<min:{min_vector}, max:{max_vector}, adj:{adj}>".format(min_vector=self._min_vector,
                                                                        max_vector=self._max_vector,
                                                                        adj=count(self.get_adjacents()))

    @staticmethod
    def create_root(dimensions, capacity):
        assert dimensions > 0
        return BTreeNode(np.array([0. for _ in xrange(dimensions)]), capacity, None)

    @property
    def has_children(self):
        return self._child_low is not None

    @property
    def is_root(self):
        return self._parent is None

    @property
    def is_leaf(self):
        return self._child_low is None

    @property
    def dimensions(self):
        return len(self._center_vector)

    def add(self, position, data):
        assert self.is_root
        assert isinstance(position, np.ndarray)
        position.setflags(write=False)
        self._add(position, data)

    def add_batch(self, items):
        if self.has_children:
            for position, data in items:
                self._child_with_position(position)._add(position, data)
        else:
            self._population.extend(items)

            if len(self._population) > self._capacity:
                self._split()

    def _add(self, position, data):
        if self.has_children:
            self._child_with_position(position)._add(position, data)
        else:
            self._population.append((position, data))

            if len(self._population) > self._capacity:
                self._split()

    def _split(self):
        position_array = np.array([position for position, _ in self._population])
        variance = position_array.var(axis=0)
        self._split_dimension = np.argmax(variance)
        split_dimension_population_values = position_array[:, self._split_dimension]

        self._split_value = np.median(split_dimension_population_values)
        # TODO if we have adjacents this becomes a different algorithm
        population_max = split_dimension_population_values.max()
        population_min = split_dimension_population_values.min()

        # maybe we can remove center?
        child_low_center_vector = self._center_vector.copy()
        child_low_center_vector[self._split_dimension] = (population_min + self._split_value) * .5
        child_low_max_vector = self._max_vector.copy()
        child_low_max_vector[self._split_dimension] = self._split_value
        self._child_low = BTreeNode(child_low_center_vector, self._capacity, self,
                                    min_vector=self._min_vector,
                                    max_vector=child_low_max_vector)

        child_high_center_vector = self._center_vector.copy()
        child_high_center_vector[self._split_dimension] = (population_max - self._split_value) * .5
        child_high_min_vector = self._min_vector.copy()
        child_high_min_vector[self._split_dimension] = self._split_value
        self._child_high = BTreeNode(child_high_center_vector, self._capacity,
                                     self,
                                     min_vector=child_high_min_vector,
                                     max_vector=self._max_vector)

        # Set adjacents
        self._child_low._adjacents[(self._split_dimension, True)].add(self._child_high)
        self._child_high._adjacents[(self._split_dimension, False)].add(self._child_low)

        for dim in xrange(self.dimensions):
            self.__update_adjacents_on_split(dim, True)
            self.__update_adjacents_on_split(dim, False)

        self._adjacents.clear()  # TODO set to None?

        for position, data in self._population:
            self._child_with_position(position)._add(position, data)
        self._population = None

    def __update_adjacents_on_split(self, dim, is_positive):
        direction = (dim, is_positive)
        if direction in self._adjacents:
            for adj in self._adjacents[direction]:
                adj_list = adj._adjacents[(dim, not is_positive)]
                adj_list.remove(self)
                if share_edge(adj._min_vector, adj._max_vector, self._child_high._min_vector, self._child_high._max_vector):
                    adj_list.add(self._child_high)
                    self._child_high._adjacents[direction].add(adj)
                if share_edge(adj._min_vector, adj._max_vector, self._child_low._min_vector, self._child_low._max_vector):
                    adj_list.add(self._child_low)
                    self._child_low._adjacents[direction].add(adj)

    def _child_with_position(self, position):
        return self._child_low if position[self._split_dimension] < self._split_value else self._child_high

    def get_leaf(self, position):
        if self.is_leaf:
            return self
        return self._child_with_position(position).get_leaf(position)

    def _get_populations_intersecting_box(self, box_min, box_max):
        # TODO intersecting circle
        if self.has_children:
            comp_min = box_min < self._split_center
            comp_max = box_max < self._split_center

            for combination in combinations_between_bool_vectors(comp_min, comp_max):
                for child in self._children[combination]:
                    for population in child._get_populations_intersecting_box(box_min, box_max):
                        yield population
        else:
            yield self._population

    def get_similar(self, position, size, distance_func=mean_squared_error):
        """

        Args:
            position (np.array):
            size (int): How many similar items to return
            distance_func (a,b -> float): method to get the distance between the position of 2 items, input is the position not the item

        Returns:

        """
        assert size > 0
        assert len(position) == self.dimensions
        assert isinstance(position, np.ndarray)

        leaf = self.get_leaf(position)

        nodes = {leaf}
        outer_nodes = {leaf}

        while outer_nodes and sum(node.population_size for node in nodes) < size * 1.5:
            # this 1.5 number is a bad hack todo, find a better way to do it (A star and progressive sorting)
            for outer_node in list(outer_nodes):
                outer_nodes.remove(outer_node)

                for adj in outer_node.get_adjacents():
                    if adj not in nodes:
                        nodes.add(adj)
                        outer_nodes.add(adj)

        return ((position, item) for similarity, position, item in
                get_similar_states(position, chain(*(node._population for node in nodes)), size, distance_func))

    def get_leaves(self):
        if self.has_children:
            for leaf in chain(self._child_low.get_leaves(), self._child_high.get_leaves()):
                yield leaf
        else:
            yield self

    def get_population(self):
        for leaf in self.get_leaves():
            for item in leaf._population:
                yield item

    def get_adjacents(self):
        return chain(*self._adjacents.itervalues())


def share_edge(a_min_vector, a_max_vector, b_min_vector, b_max_vector, delta=1e-100):
    """Check if 2 n dimensional boxes share an edge

    Args:
        a_min_vector (np.array):
        a_max_vector (np.array):
        b_min_vector (np.array):
        b_max_vector (np.array):
        delta (float):margin for rounding errors

    Returns:
        bool
    """
    is_touching_in_dimension = False
    for a_min, a_max, b_min, b_max in zip(a_min_vector, a_max_vector, b_min_vector, b_max_vector):
        if a_min is None:
            a_min = -sys.float_info.max
        if a_max is None:
            a_max = sys.float_info.max
        if b_min is None:
            b_min = -sys.float_info.max
        if b_max is None:
            b_max = sys.float_info.max

        if not is_touching_in_dimension and (abs(a_min - b_max) <= delta or abs(a_max - b_min) <= delta):
            is_touching_in_dimension = True
        else:
            if a_min < b_min:
                if a_max < b_min:
                    return False
            elif b_max < a_min:
                return False

    # a_min a_max * * -> no
    # a_min b_min * a_max * -> yes
    # b_min b_max * * -> no
    # b_min a_min * b_max * -> yes
    return is_touching_in_dimension

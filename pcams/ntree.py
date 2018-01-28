from itertools import chain

import numpy as np

from pcams.common import combinations_between_bool_vectors, mean_squared_error, get_similar_states


class NTreeNode(object):
    def __init__(self, min_vector, max_vector, capacity, population=None):
        assert isinstance(min_vector, np.ndarray)
        assert isinstance(max_vector, np.ndarray)
        assert all(min_ < max_ for min_, max_ in zip(min_vector, max_vector))
        assert len(min_vector) == len(max_vector)
        assert capacity > 0
        assert len(min_vector) <= MAX_DIMENSIONS

        self.max_vector = max_vector
        self.min_vector = min_vector
        self._capacity = capacity
        self.center = min_vector + (max_vector - min_vector) * .5
        self.population = population or []
        self._split_center = None
        self._children = None

    @property
    def has_children(self):
        return bool(self._children)

    @property
    def dimensions(self):
        return len(self.min_vector)

    def add(self, position, data):
        assert isinstance(position, np.ndarray)
        assert all(position > self.min_vector)
        assert all(position < self.max_vector)
        self._add(position, data)

    def add_batch(self, items):
        if self.has_children:
            for position, data in items:
                self._child_with_position(position)._add(position, data)
        else:
            self.population.extend(items)

            if len(self.population) > self._capacity:
                self._split()

    def _add(self, position, data):
        if self.has_children:
            self._child_with_position(position)._add(position, data)
        else:
            self.population.append((position, data))

            if len(self.population) > self._capacity:
                self._split()

    def _split(self):
        self._children = {}
        self._split_center = np.median([position for position, _ in self.population], axis=1)
        for combination_tuple in binary_combinations(self.dimensions):
            child_min_vector = []
            child_max_vector = []
            for i, d in enumerate(combination_tuple):
                if d:
                    child_min_vector.append(self._split_center[i])
                    child_max_vector.append(self.max_vector[i])
                else:
                    child_min_vector.append(self.min_vector[i])
                    child_max_vector.append(self._split_center[i])
            self._children[combination_tuple] = NTreeNode(np.array(child_min_vector), np.array(child_max_vector), self._capacity)
        for position, data in self.population:
            self._child_with_position(position)._add(position, data)
        self.population = None

    def _child_with_position(self, position):
        combination = position > self._split_center
        return self._children[tuple(combination)]

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
            yield self.population

    def get_similar(self, position, size, search_range, comparitor=mean_squared_error):
        assert size > 0
        assert search_range > 0

        half_range = search_range * .5
        min = position - half_range
        max = position + half_range

        return get_similar_states(position, chain(*self._get_populations_intersecting_box(min, max)), size, comparitor)

    def get_leaves(self):
        if self.has_children:
            for child in self._children.itervalues():
                for leaf in child.get_leaves():
                    yield leaf
        else:
            yield self


def binary_combinations(dimensions):
    """

    Args:
        dimensions (int):

    Yields:
        tuples of bool : length is the same as the dimension arg with every possible combination of True and False covered
    """
    if dimensions <= 1:
        yield (False,)
        yield (True,)
    else:
        for perm in binary_combinations(dimensions - 1):
            yield (False,) + perm
            yield (True,) + perm

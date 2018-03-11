from itertools import chain, izip

from collections import Iterable

from pcams.btree import BTreeNode
from pcams.common import mean_squared_error
from pcams.encoder import EncoderInterface

DEFAULT_ITEMS_PER_LEAF = 10


class ContentAddressableMemory(object):
    def __init__(self, encoder, distance_metric=mean_squared_error, items_per_leaf=DEFAULT_ITEMS_PER_LEAF):
        """

        Args:
            items_per_leaf (int):
            encoder (EncoderInterface):
            distance_metric (np.array, np.array -> float):
        """
        assert isinstance(encoder, EncoderInterface)

        self._encoder = encoder
        self._distance_metric = distance_metric
        self._items_per_leaf = items_per_leaf
        self._root = BTreeNode.create_root(encoder.encoded_dimensions, items_per_leaf)

    def train(self, extra_items=()):
        """Trains the encoder on the existing data and extra_items"""
        self._encoder.train(chain(extra_items, self._root.get_population() or ()))
        self.rebalance(extra_items)

    def add(self, item):
        self.add_batch([item])

    def add_batch(self, items):
        self._root.add_batch(izip(self._encoder.encode_batch(items), items))

    def get_similar(self, item, size):
        encoding = next(iter(self._encoder.encode_batch([item])))
        return self._root.get_similar(encoding, size=size, distance_func=self._distance_metric)

    def rebalance(self, extra_items=()):
        """As more items get added to the memory collection this may cause in efficient space partitions, this method rebalances with the
        most efficient division of space.

        Args:
            extra_items (Iterable): Any extra items we want to add while rebalencing
        """
        assert isinstance(extra_items, Iterable)

        all_items = list(item for _, item in chain(self._root.get_population(), extra_items))

        # TODO maybe on rebuild we can be smarter about leaf size? (and maybe dimensions even?)
        self._root = BTreeNode.create_root(self._encoder.encoded_dimensions, self._items_per_leaf)
        self.add_batch(all_items)

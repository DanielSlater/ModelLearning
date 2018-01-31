from itertools import chain, izip

from pcams.btree import BTreeNode
from pcams.common import mean_squared_error
from pcams.encoder import EncoderInterface

DEFAULT_ITEMS_PER_LEAF = 10


class ContentAddressableMemory(object):
    def __init__(self, encoder, distance_metric=mean_squared_error, items_per_leaf=DEFAULT_ITEMS_PER_LEAF):
        """

        Args:
            encoder (EncoderInterface):
            distance_metric:
        """
        assert isinstance(encoder, EncoderInterface)

        self._encoder = encoder
        self._distance_metric = distance_metric
        self._items_per_leaf = items_per_leaf
        self._root = BTreeNode.create_root(encoder.encoded_dimensions, items_per_leaf)

    def train(self, items, include_existing=True):
        if include_existing and any(self._root.get_population()):
            self._encoder.train(chain(items, self._root.get_population()))
        else:
            self._encoder.train(items)
        self._root.add_batch(self._encoder.encode_batch(items))

    def add(self, item):
        self.add_batch([item])

    def add_batch(self, items):
        self._root.add_batch(izip(self._encoder.encode_batch(items), items))

    def get_similar(self, item, size):
        encoding = next(iter(self._encoder.encode_batch([item])))
        return self._root.get_similar(encoding, size=size, distance_func=self._distance_metric)

    def rebuild(self):
        all_items = list(item for _, item in self._root.get_population())

        # TODO maybe on rebuild we can be smarter about leaf size? (and maybe dimensions even?)
        self._root = BTreeNode.create_root(self._encoder.encoded_dimensions, self._items_per_leaf)
        self._encoder.train(all_items)
        self.add_batch(all_items)

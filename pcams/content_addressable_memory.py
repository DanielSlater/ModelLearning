from itertools import chain

from pcams.btree import BTreeNode
from pcams.common import mean_squared_error

DEFAULT_ITEMS_PER_LEAF = 10


class EncoderInterface(object):
    def train(self, items):
        raise NotImplementedError()

    def encode_batch(self, items):
        raise NotImplementedError()


class IdentityEncoder(EncoderInterface):
    def encode_batch(self, items):
        return items

    def train(self, items):
        pass


class ContentAddressableMemory(object):
    def __init__(self, encoded_dimensions, encoder, distance_metric=mean_squared_error):
        assert encoded_dimensions > 0
        assert isinstance(encoder, EncoderInterface)

        self._dimensions = encoded_dimensions
        self._encoder = encoder
        self._distance_metric = distance_metric
        self._root = BTreeNode.create_root(encoded_dimensions, DEFAULT_ITEMS_PER_LEAF)

    def train(self, items, include_existing=True):
        if include_existing and any(self._root.get_population()):
            self._encoder.train(chain(items, self._root.get_population()))
        else:
            self._encoder.train(items)
        self._root.add_batch(self._encoder.encode_batch(items))

    def add(self, item):
        self.add_batch([item])

    def add_batch(self, items):
        self._root.add_batch(self._encoder.encode_batch(items))

    def get_similar(self, item, size):
        encoding, item = self._encoder.encode_batch([item])[0]
        return self._root.get_similar(encoding, size=size+1, distance_func=self._distance_metric)

    def rebuild(self):
        all_items = list(item for _, item in self._root.get_population())

        # TODO maybe on rebuild we can be smarter about leaf size? (and maybe dimensions even?)
        self._root = BTreeNode.create_root(self._dimensions, DEFAULT_ITEMS_PER_LEAF)
        self._encoder.train(all_items)
        self.add_batch(all_items)
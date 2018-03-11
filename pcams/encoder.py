import numpy as np


class EncoderInterface(object):
    def __init__(self, encoded_dimensions):
        """

        Args:
            encoded_dimensions (int):
        """
        assert encoded_dimensions > 0
        self._encoded_dimensions = encoded_dimensions

    def train(self, items):
        raise NotImplementedError()

    def encode_batch(self, items):
        raise NotImplementedError()

    @property
    def encoded_dimensions(self):
        raise self._encoded_dimensions


class IdentityEncoder(EncoderInterface):
    def encode_batch(self, items):
        return items

    def train(self, items):
        pass


class BatchNormEncoder(EncoderInterface):
    def __init__(self, encoded_dimensions):
        """

        Args:
            encoded_dimensions (int):
        """
        super(BatchNormEncoder, self).__init__(encoded_dimensions)
        self._means = np.zeros(encoded_dimensions)
        self._precision = np.ones(encoded_dimensions)

    def encode_batch(self, items):
        return (items - self._means) * self._precision

    def train(self, items):
        self._means = np.mean(items, axis=0)
        self._precision = 1 / np.var(items - self._means, axis=0)


class ChainEncoder(EncoderInterface):
    def __init__(self, encoded_dimensions, *encoders):
        """

        Args:
            encoded_dimensions (int):
        """
        super(ChainEncoder, self).__init__(encoded_dimensions)
        assert all(isinstance(x, EncoderInterface) for x in encoders)
        self._encoders = encoders

    def encode_batch(self, items):
        for encoder in self._encoders:
            items = encoder.encode_batch(items)

        return items

    def train(self, items):
        for encoder in self._encoders:
            items = encoder.train(items)
class EncoderInterface(object):
    def train(self, items):
        raise NotImplementedError()

    def encode_batch(self, items):
        raise NotImplementedError()

    @property
    def encoded_dimensions(self):
        raise NotImplementedError()


class IdentityEncoder(EncoderInterface):
    def __init__(self, encoded_dimensions):
        """

        Args:
            encoded_dimensions (int):
        """
        assert encoded_dimensions > 0
        self._encoded_dimensions = encoded_dimensions

    def encode_batch(self, items):
        return items

    def train(self, items):
        pass

    @property
    def encoded_dimensions(self):
        return self._encoded_dimensions
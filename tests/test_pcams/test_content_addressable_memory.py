from pcams.content_addressable_memory import ContentAddressableMemory
from pcams.encoder import IdentityEncoder


def test_example_usage():
    encoder = IdentityEncoder(2)
    cam = ContentAddressableMemory(encoder, items_per_leaf=1)

    cam.train()

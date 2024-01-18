from .. import Embedder
from .fastrp import fastrp_wrapper


class FastRPEmbedder(Embedder):
    def __init__(self):
        pass

    def embed(self, data):
        raise NotImplementedError()

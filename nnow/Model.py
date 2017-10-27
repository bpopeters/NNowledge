import torch.nn as nn
from .utils import Elementwise


class Embeddings(nn.Module):
    """
    A generalized embedding matrix that allows multiple layers of indexing.
    """

    def __init__(self, vocab_sizes, emb_dims):
        assert len(vocab_sizes) == len(emb_dims)

        super(Embeddings, self).__init__()
        # what to do about the padding index?
        embeddings = [nn.Embedding(vocab, dim)
                      for vocab, dim in zip(vocab_sizes, emb_dims)]
        self.luts = Elementwise(embeddings)
        self._out_size = sum(lut.embedding_dim for lut in self.luts)

    @property
    def embedding_size(self):
        return self._out_size

    def forward(self, input):
        """
        input (LongTensor): len x batch x nfeat
        returns emb (FloatTensor): len x batch x self.embedding_size
        """
        _, _, nfeat = input.size()
        assert nfeat == len(self.luts)
        return self.luts(input)


class Encoder(nn.Module):
    pass


class Decoder(nn.Module):
    pass


class Seq2Seq(nn.Module):
    pass

import torch
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


class RecurrentLayers(nn.Module):
    """
    Wrapper around various recurrent units. Handles all awkward sizing
    issues, BRNN-merging, and so on.
    """
    def __init__(self, rnn_type, **kwargs):
        """
        """
        assert rnn_type in ['LSTM', 'GRU', 'RNN']
        super(RecurrentLayers, self).__init__()
        self.rnn = getattr(nn, rnn_type)(**kwargs)

    def _merge(self, forward_states, backward_states):
        """
        forward_states, backward_states (FloatTensor): should be same size
        returns the forward and backward states concatenated.
        """
        return torch.cat([forward_states, backward_states], 2)

    def forward(self, input):
        """
        input (FloatTensor): len x batch x embedding_dim
        returns:
            output (FloatTensor):  len x batch x rnn_size - Hidden state
                                   at each time step.
            state (FloatTensor): layers x batch x rnn_size - final
                                 Encoder state for each layer
        """
        output, state = self.rnn(input)
        if isinstance(state, tuple):
            state = state[0]
        if self.rnn.bidirectional:
            # then state is num_layers * num_directions x batch x hidden_size
            # we want to concatenate these forward and backward hidden
            # states, producing a state of dim
            # num_layers x batch x hidden_size * num_directions
            forward_states = state[::2]
            backward_states = state[1::2]
            state = self._merge(forward_states, backward_states)
        return output, state


class Encoder(nn.Module):
    """
    A modular encoder class.
    """
    def __init__(self, embeddings, recurrent_unit):
        """
        embeddings: an embedding layer for turning a sequence (possibly
                multiple aligned sequences, i.e. words with features) into
                a sequence of fixed-size embeddings
        recurrent_unit: a module that accepts input the same size as the
                        output of the embeddings module
        """
        super(Encoder, self).__init__()
        self.embeddings = embeddings
        self.recurrent_unit = recurrent_unit

    def forward(self, input):
        """
        input (LongTensor): len x batch x nfeat
        Returns:
            outputs (FloatTensor):  len x batch x rnn_size -  Memory bank
            hidden_t (FloatTensor): Pair of layers x batch x rnn_size - final
                                  Encoder state
        """
        emb = self.embeddings(input)
        return self.recurrent_unit(emb)


class Decoder(nn.Module):
    def __init__(self, embeddings, recurrent_unit):
        super(Decoder, self).__init__()


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()

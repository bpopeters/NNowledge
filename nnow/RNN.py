"""
This file includes slightly higher-level modules for recurrent neural
networks. Specifically, they handle the dimensionality problems that
can potentially arise when pairing a bidirectional encoder with a
unidirectional decoder. They also allow for the models to make use of
attention.

TODO: all of it.
"""

import torch.nn as nn


class RNN(nn.Module):

    def __init__(
        self, rnn_type, input_feed=False,
        bidirectional=False, attn=None, **kwargs
    ):
        assert rnn_type in ['LSTM', 'GRU', 'RNN']
        super(RNN, self).__init__()
        self.rnn = getattr(nn, rnn_type)(bidirectional=bidirectional, **kwargs)
        self.attn = attn

    def forward(self, input, context=None, hidden=None):
        return self.rnn(input, hidden)

    @property
    def input_size(self):
        return self.rnn.input_size

    @property
    def hidden_size(self):
        return self.rnn.hidden_size

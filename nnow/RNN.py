"""
This file includes slightly higher-level modules for recurrent neural
networks. Specifically, they handle the dimensionality problems that
can potentially arise when pairing a bidirectional encoder with a
unidirectional decoder. They also allow for the models to make use of
attention.

TODO: stacked RNNs, sequence packing.
"""

import torch
import torch.nn as nn


class RNN(nn.Module):

    def __init__(
        self, rnn_type, input_size, hidden_size,
        input_feed=False, bidirectional=False, attn=None, **kwargs
    ):
        # for discussion: keep this style of __init__, or make it more
        # modular by having it accept the nn.LSTM/GRU object
        # as a parameter? I lean toward the current style because this
        # class is intended as a higher-level replacement for those classes,
        # not as a container for them. On the other hand, the presence of
        # attention sort of refutes that: move it to the Decoder?
        assert rnn_type in ['LSTM', 'GRU', 'RNN']
        if bidirectional:
            assert hidden_size % 2 == 0
            hidden_size = hidden_size // 2
        super(RNN, self).__init__()
        self.rnn = getattr(nn, rnn_type)(
            input_size, hidden_size, bidirectional=bidirectional, **kwargs
        )
        self.attn = attn

    def forward(self, input, context=None, hidden=None):
        """
        input: batch x sequence length x hidden size
        context: batch x source sequence length x hidden size
        """
        output, hidden_n = self.rnn(input, hidden)
        if self.attn is not None:
            output = self.attn(output, context)
        if self.rnn.bidirectional:
            hidden_n = self._reshape_hidden(hidden_n)
        return output, hidden_n

    def _merge_tensor(self, state_tensor):
        forward_states = state_tensor[::2]
        backward_states = state_tensor[1::2]
        return torch.cat([forward_states, backward_states], 2)

    def _reshape_hidden(self, hidden):
        """
        hidden:
            num_layers * num_directions x batch x self.hidden_size // 2
            or a tuple of these
        returns:
            num_layers
        """
        assert self.rnn.bidirectional
        if isinstance(hidden, tuple):
            return tuple(self._merge_tensor(h) for h in hidden)
        else:
            return self._merge_tensor(hidden)

    @property
    def input_size(self):
        return self.rnn.input_size

    @property
    def hidden_size(self):
        if self.rnn.bidirectional:
            return 2 * self.rnn.hidden_size
        else:
            return self.rnn.hidden_size

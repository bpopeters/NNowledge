import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, embeddings, rnn):
        assert embeddings.embedding_dim == rnn.input_size
        super(Encoder, self).__init__()
        self.embeddings = embeddings
        self.rnn = rnn

    def forward(self, input, hidden=None):
        """
        input (LongTensor): batch x src length
        hidden:
        returns:
        """
        emb = self.embeddings(input)
        output, hidden_n = self.rnn(emb, hidden)
        return output, hidden_n


class Decoder(nn.Module):
    def __init__(self, embeddings, rnn, output_layer):
        assert embeddings.embedding_dim == rnn.input_size
        super(Decoder, self).__init__()
        self.embeddings = embeddings
        self.rnn = rnn
        self.output_layer = output_layer

    def forward(self, input, context, hidden):
        """
        input (LongTensor): batch x tgt length
        context (FloatTensor): batch x src length x hidden size
        """
        emb = self.embeddings(input)
        output, hidden_n = self.rnn(emb, context=context, hidden=hidden)

        flat_output = output.contiguous().view(-1, self.rnn.hidden_size)
        return self.output_layer(flat_output)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt):
        """
        src, tgt: Variable(LongTensor) (batch size x sequence length)
        returns
        """
        tgt = tgt[:, :-1]  # this causes problems when tgt length is 1
        context, enc_hidden = self.encoder(src)
        return self.decoder(tgt, context=context, hidden=enc_hidden)

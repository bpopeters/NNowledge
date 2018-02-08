import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, embeddings, rnn):
        assert embeddings.embedding_dim == rnn.input_size
        super(Encoder, self).__init__()
        self.embeddings = embeddings
        self.rnn = rnn

    def forward(self, input, hidden_0=None):
        """
        
        """
        emb = self.embeddings(input)
        output, hidden_n = self.rnn(emb, hidden_0)
        return output, hidden_n


class Decoder(nn.Module):
    def __init__(self, embeddings, rnn, output_layer):
        assert embeddings.embedding_dim == rnn.input_size
        super(Decoder, self).__init__()
        self.embeddings = embeddings
        self.rnn = rnn
        self.output_layer = output_layer

    def forward(self, input, hidden_0=None):
        """
        
        """
        emb = self.embeddings(input)
        output, hidden_n = self.rnn(emb, hidden_0)
        return self.output_layer(output.contiguous().view(-1, self.rnn.hidden_size))


class SoftmaxOutput(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super(SoftmaxOutput, self).__init__()
        self.affine = nn.Linear(hidden_size, vocab_size)

    def forward(self, hidden_layer):
        return F.softmax(self.affine(hidden_layer))
        

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
        enc_out, enc_hidden = self.encoder(src)
        return self.decoder(tgt, enc_hidden)

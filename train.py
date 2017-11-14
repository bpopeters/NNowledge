#!/usr/bin/env python3

"""
TODO: figure out a way to actually train this thing with data. And how
to read the training data...
"""

import argparse

import torch
import torch.optim as optim
# from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# from nnow.IO import Dataset
from nnow.Model import RecurrentLayers


class Encoder(nn.Module):
    def __init__(self, embedding, recurrent_unit):
        super(Encoder, self).__init__()
        self.embedding = embedding
        self.recurrent_unit = recurrent_unit

    def forward(self, input, hidden=None):
        """
        input (LongTensor): batch_size x seq_len
        hidden: ?
        returns output (FloatTensor): batch_size x seq_len x hidden_size
                hidden_n: a tuple of things
        """
        emb = self.embedding(input)
        output, hidden_n = self.recurrent_unit(emb, hidden)
        return output, hidden_n


class Decoder(nn.Module):
    def __init__(self, embeddings, recurrent_unit, attention, output_layer):
        super(Decoder, self).__init__()
        self.embeddings = embeddings
        self.recurrent_unit = recurrent_unit
        self.attention = GlobalAttention(recurrent_unit.hidden_size)
        self.output_layer = output_layer

    def forward(self, input, context, state):
        emb = self.embeddings(input)
        rnn_out, state = self.recurrent_unit(emb, state)
        attn_out, alignment_vector = self.attention(rnn_out, context)
        return self.output_layer(attn_out)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt):
        # tgt is batch x tgt_len
        tgt = tgt[:, :-1]
        context, enc_hidden = self.encoder(src)
        out = self.decoder(tgt, context, enc_hidden)
        return out


class GlobalAttention(nn.Module):
    """
    A global attention mechanism proposed by Luong et al. (2015?).
    The only parameter is the shared size of the encoder and decoder
    hidden layers.
    """
    def __init__(self, dim):
        super(GlobalAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)

    def score(self, h_t, h_s):
        """
        h_t (FloatTensor): batch x tgt_len x dim
        h_s (FloatTensor): batch x src_len x dim
        returns out (FloatTensor): batch x tgt_len x src_len:
            raw attention scores for each src index
        """
        # Check input sizes
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        src_batch, src_len, src_dim = h_s.size()
        assert src_batch == tgt_batch
        assert src_dim == tgt_dim

        h_t = self.linear_in(h_t)
        out = torch.bmm(h_t, h_s.transpose(1, 2))
        return out

    def forward(self, input, context):
        """
        input (FloatTensor): batch x tgt_len x dim: decoder's rnn's output.
        context (FloatTensor): batch x src_len x dim: src hidden states
        returns attn_h_t (FloatTensor): batch x tgt_len x dim
                alignment_vector (FloatTensor): batch x tgt_len x src_len
        """
        a_t = self.score(input, context)
        # is softmax normalizing over the correct dimension?
        alignment_vector = F.softmax(a_t)
        c_t = torch.bmm(alignment_vector, context)
        attn_h_t = self.linear_out(torch.cat([c_t, input], 2))
        attn_h_t = F.tanh(attn_h_t)
        return attn_h_t, alignment_vector


class OutputLayer(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super(OutputLayer, self).__init__()
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, input):
        """
        input (FloatTensor): batch_size x len x hidden_size
        """
        bottled_input = input.view(-1, self.linear.in_features)
        return F.log_softmax(self.linear(bottled_input))


def train_epoch(model, train, batch_size, criterion, optimizer):
    for s, t in train.batches(batch_size):
        optimizer.zero_grad()
        predicted = model(s, t)
        gold = t[:, 1:].contiguous().view(-1)  # a little ugly but I think ok
        loss = criterion(predicted, gold)
        print(loss)
        loss.backward()
        optimizer.step()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train', type=torch.load)
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-brnn', action='store_true')
    parser.add_argument('-word_vec_size', type=int, default=150)
    parser.add_argument('-hidden_size', type=int, default=150)
    parser.add_argument('-layers', type=int, default=2)
    opt = parser.parse_args()

    src_emb = nn.Embedding(len(opt.train.src_vocab), opt.word_vec_size)
    src_rec = RecurrentLayers(
        'LSTM', input_size=opt.word_vec_size, hidden_size=opt.hidden_size,
        num_layers=opt.layers, bidirectional=opt.brnn
    )
    enc = Encoder(src_emb, src_rec)

    tgt_emb = nn.Embedding(len(opt.train.tgt_vocab), opt.word_vec_size)
    tgt_rec = RecurrentLayers(
        'LSTM', input_size=opt.word_vec_size, hidden_size=opt.hidden_size,
        num_layers=opt.layers, bidirectional=False
    )
    attn = GlobalAttention(opt.hidden_size)
    output_layer = OutputLayer(opt.hidden_size, len(opt.train.tgt_vocab))
    dec = Decoder(tgt_emb, tgt_rec, attn, output_layer)

    model = Seq2Seq(enc, dec)

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=1, momentum=0.9)

    # with a batch size of 1, there was eventually an empty tensor bug
    train_epoch(model, opt.train, opt.batch_size, criterion, optimizer)


if __name__ == '__main__':
    main()

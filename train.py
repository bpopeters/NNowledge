#!/usr/bin/env python3

"""
TODO: figure out a way to actually train this thing with data. And how
to read the training data...
"""

import argparse
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
from nnow.Model import Encoder, Decoder, Seq2Seq, SoftmaxOutput


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train', type=torch.load)
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-brnn', action='store_true')
    parser.add_argument('-word_vec_size', type=int, default=150)
    parser.add_argument('-hidden_size', type=int, default=150)
    parser.add_argument('-layers', type=int, default=2)
    parser.add_argument('-dropout', type=float, default=0.3)
    opt = parser.parse_args()

    src_emb = nn.Embedding(
        len(opt.train.src_vocab), opt.word_vec_size, padding_idx=0
    )
    tgt_emb = nn.Embedding(
        len(opt.train.tgt_vocab), opt.word_vec_size, padding_idx=0
    )
    enc_rnn = nn.LSTM(
        input_size=opt.word_vec_size, hidden_size=opt.hidden_size,
        bidirectional=opt.brnn, num_layers=opt.layers, batch_first=True,
        dropout=opt.dropout
    )

    dec_rnn = nn.LSTM(
        input_size=opt.word_vec_size, hidden_size=opt.hidden_size,
        bidirectional=False, num_layers=opt.layers, batch_first=True,
        dropout=opt.dropout
    )
    output_layer = SoftmaxOutput(dec_rnn.hidden_size, len(opt.train.tgt_vocab))

    encoder = Encoder(src_emb, enc_rnn)
    decoder = Decoder(tgt_emb, dec_rnn, output_layer)
    model = Seq2Seq(encoder, decoder)

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    model.train()
    for src, tgt in opt.train.batches(opt.batch_size):
        optimizer.zero_grad()
        s, t = Variable(src), Variable(tgt)

        predicted = model(s, t)
        gold = t[:, 1:].contiguous().view(-1)  # a little ugly but I think ok
        loss = criterion(predicted, gold)
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    main()

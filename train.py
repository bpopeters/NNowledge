#!/usr/bin/env python3

import argparse
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
from nnow.Model import Encoder, Decoder, Seq2Seq, LogSoftmaxOutput


def train_epoch(model, optimizer, criterion, batches):
    model.train()
    for src, tgt in batches:
        optimizer.zero_grad()
        s, t = Variable(src), Variable(tgt)

        predicted = model(s, t)
        gold = t[:, 1:].contiguous().view(-1)
        loss = criterion(predicted, gold)
        loss.backward()
        optimizer.step()
    model.eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train', type=torch.load)
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-brnn', action='store_true')
    parser.add_argument('-word_vec_size', type=int, default=150)
    parser.add_argument('-hidden_size', type=int, default=150)
    parser.add_argument('-layers', type=int, default=2)
    parser.add_argument('-dropout', type=float, default=0.3)
    parser.add_argument('-epochs', type=int, default=10)
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
    output_layer = LogSoftmaxOutput(
        dec_rnn.hidden_size, len(opt.train.tgt_vocab)
    )

    encoder = Encoder(src_emb, enc_rnn)
    decoder = Decoder(tgt_emb, dec_rnn, output_layer)
    model = Seq2Seq(encoder, decoder)

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    for i in range(1, opt.epochs + 1):
        batches = opt.train.batches(opt.batch_size)
        train_epoch(model, optimizer, criterion, batches)


if __name__ == '__main__':
    main()

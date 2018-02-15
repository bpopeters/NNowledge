#!/usr/bin/env python3

import argparse
import math
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
from nnow.Model import Encoder, Decoder, Seq2Seq
from nnow.OutputLayer import LogSoftmaxOutput
from nnow.RNN import RNN
from nnow.Attention import Attention


def train_batch(batch, model, optimizer, criterion):
    """
    Do stuff, like report the size-averaged loss
    """
    src = Variable(batch['src'])
    tgt = Variable(batch['tgt'])
    optimizer.zero_grad()
    gold = tgt[:, 1:].contiguous().view(-1)

    predicted = model(src, tgt, src_lengths=batch.get('src_lengths', None))
    loss = criterion(predicted, gold)
    loss.backward()
    optimizer.step()
    return loss.data[0]


def train_epoch(model, optimizer, criterion, batches, report_every):
    model.train()
    for i, batch in enumerate(batches, 1):
        optimizer.zero_grad()
        loss = train_batch(batch, model, optimizer, criterion)
        if i % report_every == 0:
            # note that this is not correct: it's the loss for only a
            # single batch
            print('training ppl {}'.format(math.exp(loss)))


def validate_model(model, criterion, batches):
    """
    TODO: this is very silly. I need a better way of getting the loss
    numbers to report.
    And it isn't great that I'm using batches this way, either
    However,
    """
    model.eval()
    pred = []
    gold = []
    for batch in batches:
        src = Variable(batch['src'], volatile=True)
        tgt = Variable(batch['tgt'], volatile=True)

        pred.append(model(src, tgt, batch.get('src_lengths', None)))
        gold.append(tgt[:, 1:].contiguous().view(-1))
    loss = criterion(torch.cat(pred), torch.cat(gold))
    return torch.exp(loss)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_src')
    parser.add_argument('train_tgt')
    parser.add_argument('valid_src')
    parser.add_argument('valid_tgt')
    parser.add_argument('bitext', type=torch.load)
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-brnn', action='store_true')
    parser.add_argument('-word_vec_size', type=int, default=150)
    parser.add_argument('-hidden_size', type=int, default=150)
    parser.add_argument('-layers', type=int, default=2)
    parser.add_argument('-dropout', type=float, default=0.3)
    parser.add_argument('-epochs', type=int, default=10)
    parser.add_argument('-rnn_type', default='LSTM', choices=['LSTM', 'GRU'])
    parser.add_argument('-report_ppl', type=int, default=50,
                        help="""Report training perplexity every this many
                        iterations.""")
    opt = parser.parse_args()

    # make a sequence-to-sequence model:
    src_emb = nn.Embedding(
        len(opt.bitext.src_vocab), opt.word_vec_size, padding_idx=0
    )
    tgt_emb = nn.Embedding(
        len(opt.bitext.tgt_vocab), opt.word_vec_size, padding_idx=0
    )

    enc_rnn = RNN(
        opt.rnn_type, bidirectional=opt.brnn, num_layers=opt.layers,
        input_size=opt.word_vec_size, hidden_size=opt.hidden_size,
        batch_first=True, dropout=opt.dropout
    )

    attn = Attention(opt.hidden_size)

    dec_rnn = RNN(
        opt.rnn_type, bidirectional=False, num_layers=opt.layers,
        input_size=opt.word_vec_size, hidden_size=opt.hidden_size,
        batch_first=True, dropout=opt.dropout, attn=attn
    )

    output_layer = LogSoftmaxOutput(
        dec_rnn.hidden_size, len(opt.bitext.tgt_vocab)
    )

    encoder = Encoder(src_emb, enc_rnn)
    decoder = Decoder(tgt_emb, dec_rnn, output_layer)
    model = Seq2Seq(encoder, decoder)

    # make the loss function and optimizer
    criterion = nn.NLLLoss(ignore_index=0)
    optimizer = optim.SGD(model.parameters(), lr=1, momentum=0.9)

    # train and validate
    for i in range(1, opt.epochs + 1):
        print('Training epoch {}'.format(i))
        train_batches = opt.bitext.batches(
            opt.train_src, opt.train_tgt, opt.batch_size
        )
        valid_batches = opt.bitext.batches(
            opt.valid_src, opt.valid_tgt, opt.batch_size
        )
        train_epoch(model, optimizer, criterion, train_batches, opt.report_ppl)
        print(validate_model(model, criterion, valid_batches))

    # serialize the model
    torch.save(model, 'foo.pt')


if __name__ == '__main__':
    main()

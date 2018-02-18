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
    src = Variable(batch['src'])
    tgt = Variable(batch['tgt'])
    src_lengths = batch['src_lengths']
    n_words = sum(batch['tgt_lengths'])
    batch_size = tgt.size(0)

    optimizer.zero_grad()
    gold = tgt[:, 1:].contiguous().view(-1)

    predicted = model(src, tgt, src_lengths=src_lengths)
    loss = criterion(predicted, gold)
    loss.div(batch_size).backward()  # divide loss by batch size
    optimizer.step()
    return loss.data[0], n_words


def train_epoch(model, optimizer, criterion, batches, report_every):
    model.train()
    report_loss = 0.0
    n_words = 0
    for i, batch in enumerate(batches, 1):
        optimizer.zero_grad()
        b_loss, b_words = train_batch(batch, model, optimizer, criterion)
        report_loss += b_loss
        n_words += b_words
        if i % report_every == 0:
            print('training ppl {}'.format(math.exp(report_loss / n_words)))
            report_loss = 0.0
            n_words = 0


def validate_model(model, criterion, batches):
    model.eval()
    loss = 0.0
    n_words = 0
    for batch in batches:
        src = Variable(batch['src'], volatile=True)
        tgt = Variable(batch['tgt'], volatile=True)
        n_words += sum(batch['tgt_lengths'])
        pred = model(src, tgt, batch.get('src_lengths', None))
        gold = tgt[:, 1:].contiguous().view(-1)
        loss += criterion(pred, gold)
    return math.exp(loss / n_words)


def initialize_parameters(module, param_init):
    if hasattr(module, 'weight'):
        nn.init.uniform(module.weight, -param_init, param_init)


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
    parser.add_argument('-learning_rate', type=float, default=0.1)
    parser.add_argument('-out', default='model.pt')
    parser.add_argument('-param_init', type=float, default=0.1)
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

    model.apply(lambda m: initialize_parameters(m, opt.param_init))

    # make the loss function
    criterion = nn.NLLLoss(ignore_index=0, size_average=False)

    # train and validate
    for i in range(1, opt.epochs + 1):
        optimizer = optim.SGD(model.parameters(), lr=opt.learning_rate)
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
    torch.save(model, opt.out)


if __name__ == '__main__':
    main()

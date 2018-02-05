#!/usr/bin/env python

"""
Use this to serialize training and validation data for a machine translation
task. All training and validation files should have one sentence per line
and tokens in the line should be whitespace-separated.
"""

import argparse
import torch
from nnow.IO import Dataset, Vocab, tokenize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_src', default='train.src', type=tokenize)
    parser.add_argument('-train_tgt', default='train.tgt', type=tokenize)
    parser.add_argument('-valid_src', default='valid.src', type=tokenize)
    parser.add_argument('-valid_tgt', default='valid.tgt', type=tokenize)
    parser.add_argument('-out', default="")
    opt = parser.parse_args()

    src_vocab = Vocab(opt.train_src)
    tgt_vocab = Vocab(opt.train_tgt)

    train_data = Dataset(opt.train_src, opt.train_tgt, src_vocab, tgt_vocab)
    valid_data = Dataset(opt.valid_src, opt.valid_tgt, src_vocab, tgt_vocab)

    torch.save(train_data, opt.out + '_train.pt')
    torch.save(valid_data, opt.out + '_valid.pt')

if __name__ == '__main__':
    main()

#!/usr/bin/env python

"""
preprocess.py produces the stuff necessary for converting data
between text and tensors and iterating through a dataset in batches
"""

import argparse
from itertools import chain
import torch
from nnow.IO import tokenize, Vocab, BitextDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_src')
    parser.add_argument('valid_src')
    parser.add_argument('train_tgt')
    parser.add_argument('valid_tgt')
    parser.add_argument('-out', default="")
    opt = parser.parse_args()
    src_train_tokens = tokenize(opt.train_src)
    src_valid_tokens = tokenize(opt.valid_src)
    tgt_train_tokens = tokenize(opt.train_tgt)
    tgt_valid_tokens = tokenize(opt.valid_tgt)

    src_vocab = Vocab(chain(src_train_tokens, src_valid_tokens))
    tgt_vocab = Vocab(chain(tgt_train_tokens, tgt_valid_tokens))
    train_dataset = BitextDataset(
        src_train_tokens, tgt_train_tokens, src_vocab, tgt_vocab
    )
    valid_dataset = BitextDataset(
        src_valid_tokens, tgt_valid_tokens, src_vocab, tgt_vocab
    )
    torch.save(
        {'train': train_dataset, 'valid': valid_dataset}, opt.out + '_data.pt'
    )

if __name__ == '__main__':
    main()

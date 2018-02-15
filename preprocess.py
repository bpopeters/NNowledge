#!/usr/bin/env python

"""
preprocess.py produces the Vocab objects necessary for converting data
between text and tensors. It accepts arbitrarily many of each source and
target files.
"""

import argparse
from itertools import chain
import torch
from nnow.IO import Vocab, tokenize, BitextIterator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', nargs='+')
    parser.add_argument('-tgt', nargs='+')
    parser.add_argument('-out', default="")
    opt = parser.parse_args()
    src = chain.from_iterable([tokenize(s) for s in opt.src])
    tgt = chain.from_iterable([tokenize(t) for t in opt.tgt])

    src_vocab = Vocab(src)
    tgt_vocab = Vocab(tgt)
    bt = BitextIterator(src_vocab, tgt_vocab)
    torch.save(bt, opt.out + '_vocab.pt')

if __name__ == '__main__':
    main()

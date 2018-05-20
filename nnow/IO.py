from collections import Counter
from itertools import chain, takewhile, zip_longest, repeat
import torch

BOS = '<s>'
EOS = '</s>'
UNK = '<unk>'
PAD = '<blank>'


def tokenize(filename, lowercase=True):
    """
    filename: location of a src or tgt file
    returns: one or several sequences, where each returned sequence is the
             same length. Each element of the sequence
    """
    with open(filename) as f:
        for line in f:
            if lowercase:
                line = line.lower()
            yield [BOS] + line.strip().split() + [EOS]


def batchify(lines, batch_size):
    """
    cf. https://docs.python.org/3/library/itertools.html#itertools-recipes
    lines is an iterable, such as the src.test lines
    """
    args = [iter(lines)] * batch_size
    for raw_batch in zip_longest(*args):
        yield tuple(line for line in raw_batch if line is not None)


def bitext_batches(src_batches, tgt_batches, sort=True):
    for src_batch, tgt_batch in zip(src_batches, tgt_batches):
        if sort:
            order, sorted_src = zip(*sorted(enumerate(src_batch),
                                            key=lambda x: len(x[1]),
                                            reverse=True))
            sorted_tgt = tuple(tgt_batch[i] for i in order)
            yield sorted_src, sorted_tgt
        else:
            yield src_batch, tgt_batch


class Vocab(object):
    """
    A vocabulary: hand it a tokenized corpus, it will maintain the types
    that occur in it. It includes methods for turning sequences of tokens
    into tensors and vice versa.
    """
    def __init__(self, corpus, meta=(PAD, BOS, EOS, UNK),
                 vocab_size=None, min_count=1):
        vocab_counts = Counter(chain.from_iterable(corpus))
        for m in meta:
            if m in vocab_counts:
                vocab_counts.pop(m)

        kept_vocab = takewhile(
            lambda item: item[1] >= min_count,
            vocab_counts.most_common(vocab_size)
        )
        self._index2str = list(meta)
        self._index2str.extend(w for w, count in kept_vocab)
        self._str2index = {word: i for i, word in enumerate(self._index2str)}
        assert len(self._index2str) == len(self._str2index)

    def __len__(self):
        """The vocab size"""
        return len(self._index2str)

    def _string2tensor(self, tokens, padded_length):
        padded_toks = chain(tokens, repeat(PAD, padded_length - len(tokens)))
        return torch.LongTensor(
            [self._str2index.get(tok, self._str2index[UNK])
             for tok in padded_toks]
        ).unsqueeze(0)

    def string2tensor(self, batch):
        """
        batch: a list of lists of tokens
        returns: LongTensor (batch size x max_len)
        """
        # the pad_sequence function should change this, but it appears not
        # yet to be in the version of pytorch on conda
        max_len = max(len(sample) for sample in batch)
        return torch.cat(
            [self._string2tensor(sample, max_len) for sample in batch]
        )

    def tensor2string(self, tensor):
        return [list(
                takewhile(lambda w: w != PAD,
                          (self._index2str[i] for i in line.squeeze()))
                )
                for line in tensor.split(1)]


class BitextIterator(object):
    def __init__(self, src_vocab, tgt_vocab):
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def batches(self, src, tgt, batch_size):
        """
        src, tgt: paths to source and target files
        yields:
        """
        tokenized_src = tokenize(src)
        tokenized_tgt = tokenize(tgt)
        src_batches = batchify(tokenized_src, batch_size)
        tgt_batches = batchify(tokenized_tgt, batch_size)
        for src_batch, tgt_batch in bitext_batches(src_batches, tgt_batches):

            src_lengths = [len(sample) for sample in src_batch]
            tgt_lengths = [len(sample) for sample in tgt_batch]

            src_tensor = self.src_vocab.string2tensor(src_batch)
            tgt_tensor = self.tgt_vocab.string2tensor(tgt_batch)
            yield {'src': src_tensor, 'src_lengths': src_lengths,
                   'tgt': tgt_tensor, 'tgt_lengths': tgt_lengths}

from collections import Counter
from itertools import chain, takewhile, zip_longest
import torch

BOS = '<s>'
EOS = '</s>'
UNK = '<unk>'
PAD = '<blank>'


def tokenize(filename):
    """
    filename: location of a src or tgt file
    returns: one or several sequences, where each returned sequence is the
             same length. Each element of the sequence
    """
    with open(filename) as f:
        return [line.strip().split() for line in f]


def batchify(lines, batch_size):
    """
    cf. https://docs.python.org/3/library/itertools.html#itertools-recipes
    lines is an iterable, such as the src.test lines
    """
    args = [iter(lines)] * batch_size
    for raw_batch in zip_longest(*args):
        yield tuple(line for line in raw_batch if line is not None)


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
        return len(self._index2str)

    def _string2tensor(self, tokens, padded_length):
        tok_with_meta = [BOS] + tokens + [EOS]
        tok_with_meta.extend([PAD] * (padded_length - len(tok_with_meta)))
        return torch.LongTensor(
            [self._str2index.get(tok, self._str2index[UNK])
             for tok in tok_with_meta]
        ).unsqueeze(0)

    def string2tensor(self, sequences):
        """
        sequences: a lits of lists of strings
        returns: LongTensor (n_samples x max_len)
        """
        max_len = max(len(sample) for sample in sequences) + 2  # for BOS/EOS
        return torch.cat(
            [self._string2tensor(seq, max_len) for seq in sequences]
        )

    def tensor2string(self, tensor):
        return [list(
                takewhile(lambda w: w != PAD,
                          (self._index2str[i] for i in line.squeeze()))
                )
                for line in tensor.split(1)]


class Dataset(object):
    def __init__(self, src, tgt, src_vocab, tgt_vocab):
        assert len(src) == len(tgt)

        self.src = src
        self.tgt = tgt
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def batches(self, batch_size):
        src_batches = batchify(self.src, batch_size)
        tgt_batches = batchify(self.tgt, batch_size)
        for sb, tb in zip(src_batches, tgt_batches):
            yield (self.src_vocab.string2tensor(sb),
                   self.tgt_vocab.string2tensor(tb))

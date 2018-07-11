from collections import Counter
from itertools import count, chain, takewhile
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence

BOS = '<s>'
EOS = '</s>'
UNK = '<unk>'
PAD = '<blank>'


def tokenize_line(line, lowercase=True):
    if lowercase:
        line = line.lower()
    return [BOS] + line.strip().split() + [EOS]


def tokenize(filename, lowercase=True):
    """
    filename: location of a src or tgt file
    returns: one or several sequences, where each returned sequence is the
             same length. Each element of the sequence
    """
    with open(filename) as f:
        return [tokenize_line(line, lowercase) for line in f]


def unsorted_pad_sequence(tensors, batch_first=False):
    sorted_order, sorted_tensors = zip(
        *sorted(enumerate(tensors), key=lambda x: len(x[1]), reverse=True)
    )
    original_order = torch.LongTensor(
        [i for i, j in sorted(enumerate(sorted_order), key=lambda x: x[1])]
    )
    padded_tensors = pad_sequence(
        sorted_tensors, batch_first=True
    ).index_select(0, original_order)
    if not batch_first:
        padded_tensors.transpose_()
    return padded_tensors


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

    def line2tensor(self, tokens):
        """
        convert tokenized string into 1d LongTensor of indices
        """
        indices = [self._str2index.get(tok, self._str2index[UNK])
                   for tok in tokens]
        return torch.LongTensor(indices)

    def batch2tensor(self, batch, sorted_by_length=True):
        """
        batch: a Series of token sequences
        returns: LongTensor (batch size x max_len)
        """
        tensors = [self.line2tensor(b) for b in batch]
        if sorted_by_length:
            return pad_sequence(tensors, batch_first=True)
        else:
            return unsorted_pad_sequence(tensors, batch_first=True)

    def tensor2string(self, tensor):
        return [list(
                takewhile(lambda w: w != PAD,
                          (self._index2str[i] for i in line.squeeze()))
                )
                for line in tensor.split(1)]


class BitextDataset(object):
    def __init__(self, src_seqs, tgt_seqs, src_vocab=None, tgt_vocab=None):
        src = pd.Series(src_seqs)
        tgt = pd.Series(tgt_seqs)
        assert src.size == tgt.size, "Source and target files are mismatched"
        self.src_vocab = Vocab(src) if src_vocab is None else src_vocab
        self.tgt_vocab = Vocab(tgt) if tgt_vocab is None else tgt_vocab
        self.data = pd.DataFrame({'src': src, 'tgt': tgt})
        self.data['src_lengths'] = self.data['src'].apply(len)
        self.data['tgt_lengths'] = self.data['tgt'].apply(len)

    def shuffle(self):
        self.data = self.data.sample(frac=1)

    def make_batch(self, batch):
        batch = batch.sort_values('src_lengths', 0, ascending=False)

        src_tensor = self.src_vocab.batch2tensor(batch['src'])
        tgt_tensor = self.tgt_vocab.batch2tensor(batch['tgt'], False)
        return {'src': src_tensor, 'tgt': tgt_tensor,
                'src_lengths': list(batch['src_lengths']),
                'tgt_lengths': list(batch['tgt_lengths'])}

    def batches(self, batch_size):
        sample_count = count()

        def batch_grouper(row):
            return next(sample_count) // batch_size
        for name, group in self.data.groupby(batch_grouper):
            yield self.make_batch(group)

from ast import Return
from torch.utils.data import Dataset
import nltk
import torch
import numpy as np
from corpus import WIN_SIZE


def sample_neg(word_freq, target, context_size, count):
    negs = torch.multinomial(
        torch.FloatTensor(word_freq), 2 * context_size + count, replacement=False
    )
    negs = np.setdiff1d(negs.numpy(), target.numpy())
    negs = negs[:count]
    return negs


class NgramDataset(Dataset):
    def __init__(self, data, corpus, WIN_SIZE=2, neg_size=5, neg_sample=False) -> None:
        super().__init__()
        self.windows = []
        self.corpus = corpus
        self.neg_size = neg_size
        self.neg_sampler = neg_sample
        self.xs = []
        self.ys = []
        self.negs = []

        ngram_size = 2 * WIN_SIZE + 1
        for sentence in data:
            if len(sentence) > ngram_size:
                self.windows.extend(list(nltk.ngrams(sentence, ngram_size)))
        for win in self.windows:
            x = list(win[0:WIN_SIZE] + win[WIN_SIZE + 1 :])
            self.xs.append(self.corpus.encode(x))
            self.ys.append(self.corpus.encode(win[WIN_SIZE]))

    def __getitem__(self, index):
        target = torch.tensor(self.ys[index])
        if self.neg_sampler:
            negs = sample_neg(
                self.corpus.word_freq, target, WIN_SIZE, count=self.neg_size
            )
            return torch.tensor(self.xs[index]), target, torch.tensor(negs)
        return torch.tensor(self.xs[index]), target

    def __len__(self):
        return len(self.ys)

from ast import Return
from torch.utils.data import Dataset
import nltk
import torch
import numpy as np
from corpus import WIN_SIZE


def sample_neg(word_freq, target, context_size, count):
    negs = torch.multinomial(torch.FloatTensor(word_freq),
                             2 * context_size + count,
                             replacement=False)
    negs = np.setdiff1d(negs.numpy(), target.numpy())
    negs = negs[:count]
    return negs


class NgramDataset(Dataset):

    def __init__(self, data, corpus, WIN_SIZE=2) -> None:
        super().__init__()
        self.windows = []
        self.corpus = corpus
        self.xs = []
        self.ys = []
        self.negs = []

        for sentence in data:
            self.windows.extend(list(nltk.ngrams(sentence, WIN_SIZE * 2 + 1)))
        for win in self.windows:
            if len(win) < 2 * WIN_SIZE + 1:
                continue
            x = list(win[0:WIN_SIZE] + win[WIN_SIZE + 1:])
            self.xs.append(self.corpus.encode(x))
            self.ys.append(self.corpus.encode(win[WIN_SIZE]))

    def __getitem__(self, index):
        target = torch.tensor(self.ys[index])
        negs = sample_neg(self.corpus.word_freq, target, WIN_SIZE, count=5)
        return torch.tensor(self.xs[index]), target, torch.tensor(negs)

    def __len__(self):
        return len(self.ys)

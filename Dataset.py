from torch.utils.data import Dataset
import nltk
import torch

padding_token = '<PAD>'


class NgramDataset(Dataset):

    def __init__(self, data, corpus, WIN_SIZE=2) -> None:
        super().__init__()
        self.windows = []
        self.corpus = corpus
        self.xs = []
        self.ys = []

        for sentence in data:
            self.windows.extend(list(nltk.ngrams(sentence, WIN_SIZE * 2 + 1)))
        for win in self.windows:
            if len(win) < 2 * WIN_SIZE + 1:
                continue
            x = list(win[0:WIN_SIZE] + win[WIN_SIZE + 1:])
            self.xs.append(self.corpus.encode(x))
            self.ys.append(self.corpus.encode(win[WIN_SIZE]))

    def __getitem__(self, index):
        return torch.tensor(self.xs[index]), torch.tensor(self.ys[index])

    def __len__(self):
        return len(self.ys)
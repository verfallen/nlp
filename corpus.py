from curses import window
from random import random
import re
from collections import Counter
import constant
import nltk
import random
import torch


def rm_sign(string):
    string = re.sub("[\.\!_,\$\(\)\"\'\]\[！!\?，。？、~@#￥……&]+", "", string)
    return string


def load_data(file_path='./data/corpus.txt'):
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if (len(line) == 0):
                continue
            yield rm_sign(line.lower()).split()


WIN_SIZE = 2


class Corpus(object):

    def __init__(self, data) -> None:
        self.vocab = []
        self.idx2word = [constant.UNK_TOKEN]
        self.word2idx = {constant.UNK_TOKEN: 0}
        self.word_count = 1
        # self.windows = []
        # self.dataset = []

        flatten = lambda lines: [word for line in lines for word in line]

        counter = Counter(flatten(data)).most_common()
        for word, count in counter:
            self.idx2word.append(word)
            self.word2idx[word] = self.word_count
            self.word_count += 1

        # for sentence in data:
        #     self.windows.extend(list(nltk.ngrams(sentence, WIN_SIZE*2+1)))

        # dataset = []
        # xs, ys = [], []
        # for win in self.windows:
        #     xs.append(list(win[0:WIN_SIZE]+ win[WIN_SIZE+1:]))
        #     ys.append(win[WIN_SIZE])
        #     dataset.append((list(win[0:WIN_SIZE]+ win[WIN_SIZE+1:]), win[WIN_SIZE]))

        # self.dataset = list(zip(xs,ys))

    # def batch_data(self, batch_size):
    #     random.shuffle(self.dataset)

    #     sidx = 0
    #     eidx = batch_size
    #     while eidx < len(self.dataset):
    #         batch = self.dataset[sidx:eidx]
    #         sidx = eidx
    #         eidx += batch_size
    #         yield batch

    #     if eidx >= len(self.dataset):
    #         batch = self.dataset[sidx:]
    #         yield batch

    def encode(self, word):
        if type(word) == list:
            return torch.LongTensor([self.word2idx[w] for w in word])
        return torch.LongTensor([self.word2idx[word]])

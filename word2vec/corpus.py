from random import random
import re
from collections import Counter
import constant
import nltk
import random
import torch
import numpy as np


def rm_sign(string):
    string = re.sub("[\.\!_,\$\(\)\"'\]\[！!\?，。？、~@#￥……&]+", "", string)
    return string


def load_data(file_path="../data/corpus.txt"):
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            yield rm_sign(line.lower()).split()


WIN_SIZE = 2


class Corpus(object):

    def __init__(self, data) -> None:
        self.vocab = []
        self.idx2word = []
        self.word2idx = {}
        self.word_count = 0
        self.word_freq = []

        flatten = lambda lines: [word for line in lines for word in line]

        all_words = flatten(data)

        counter = Counter(all_words).most_common()
        for word, count in counter:
            self.idx2word.append(word)
            self.word2idx[word] = self.word_count
            self.word_count += 1
            self.vocab.append(word)
            self.word_freq.append(count / len(all_words))

    def encode(self, word):
        if type(word) == list:
            return np.array([self.word2idx[w] for w in word])
        return np.array([self.word2idx[word]])

    def __len__(self):
        return len(self.idx2word)

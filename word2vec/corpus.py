from random import random
import re
from collections import Counter
import constant as C
import nltk
import random
import torch
import numpy as np


def rm_sign(string):
    string = re.sub("[\.\!_,\$\(\)\"'\]\[！!\?，。？、~@#￥……&]+", "", string)
    return string


def load_data(file_path="../data/tonghua.txt"):
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            yield rm_sign(line.lower()).split()


class Corpus(object):
    def __init__(self, data, vocab_size=30000) -> None:
        self.vocab = [C.UNK_TOKEN]
        self.idx2word = [C.UNK_TOKEN]
        self.word2idx = {C.UNK_TOKEN: 0}
        self.word_count = 1
        self.word_freq = []

        flatten = lambda lines: [word for line in lines for word in line]

        all_words = flatten(data)

        counter = Counter(all_words).most_common(vocab_size)

        print("文本长度", len(all_words))
        for word, count in counter:
            self.idx2word.append(word)
            self.word2idx[word] = self.word_count
            self.word_count += 1
            self.vocab.append(word)
            self.word_freq.append(count / len(all_words))

        print("词典长度", len(self.vocab))

    def encode(self, word):
        if type(word) == list:
            return np.array(
                [
                    self.word2idx[w]
                    if w in self.word2idx
                    else self.word2idx[C.UNK_TOKEN]
                    for w in word
                ]
            )
        return np.array(
            [
                self.word2idx[word]
                if word in self.word2idx
                else self.word2idx[C.UNK_TOKEN]
            ]
        )

    def __len__(self):
        return len(self.idx2word)

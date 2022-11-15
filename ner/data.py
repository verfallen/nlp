# from datasets import load_dataset

# dataset = load_dataset("~/datasets", name="conll2003")

# print(dataset)

from datasets import load_dataset
from collections import Counter
import pickle
from torch.utils.data import Dataset, DataLoader
import constant as C
import torch

dataset = load_dataset("conll2003", split="train")
# print(dataset.column_names)
# print(dataset.num_columns)
# print(dataset.num_rows)
# print(dataset['train'][0])


class Vocab(object):
    def __init__(self, dataset, vocab_size=None) -> None:
        self.vocab = []
        self.word2id = {}
        self.word_idx = 0
        tokens = dataset["tokens"]
        flatten = lambda lines: [word.lower() for line in lines for word in line]
        all_words = flatten(tokens)
        counter = Counter(all_words).most_common(vocab_size)

        for word, count in counter:
            self.vocab.append(word)
            self.word2id[word] = self.word_idx
            self.word_idx += 1

    def encode(self, word_list):
        return [self.word2id[word] for word in word_list]


vocab = Vocab(dataset)
with open("./ner_vocab.pickle", "wb") as f:
    pickle.dump(vocab, f)


class NerDataset(Dataset):
    def __init__(
        self, data, word2id, token_column="tokens", label_column="ner_tags"
    ) -> None:
        super().__init__()
        self.data = data
        self.word2id = word2id
        self.token_column = token_column
        self.label_column = label_column

    def __getitem__(self, index):
        tokens = map(lambda x: x.lower(), self.data[self.token_column][index])

        return self.encode(tokens), self.data[self.label_column][index]

    def __len__(self):
        return len(self.data)

    def encode(self, tokens):
        return [self.word2id[token] for token in tokens]


train_dataset = NerDataset(dataset, vocab.word2id)

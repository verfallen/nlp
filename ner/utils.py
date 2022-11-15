from HMM import HMM
from datasets import load_dataset
from data import Vocab

dataset = load_dataset("conll2003", split="train")
vocab = Vocab(dataset)


def train_eval_hmm(train_dataset, num_tags, num_words):
    hmm = HMM(num_tags, num_words)

    train_word_list = dataset['tokens']
    train_tag_list =  dataset['ner_tags']

    hmm.train(train_word_list, train_tag_list, vocab.word2id)
    
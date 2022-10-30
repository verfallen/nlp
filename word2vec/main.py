from CBOW_softmax import CBOW
from skip_gram_softmax import SkigGramSoftmax
from skip_gram_neg_sample import SkipGram as SG_NEG
from corpus import Corpus, load_data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Dataset import NgramDataset
from utils import (
    train_cbow_neg_sampling,
    train_cbow_softmax,
    train_sg_softmax,
    train_sg_neg_sample,
)
from d2l.torch import Timer
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

text = list(load_data())
corpus = Corpus(text)
vocab_size = len(corpus.idx2word)
embedding_dim = 10
batch_size = 128
epoches = 300

# model = CBOW(vocab_size, embedding_dim)
# model = SkigGramSoftmax(vocab_size, embedding_dim)
model = SG_NEG(vocab_size, embedding_dim)
criterion = nn.CrossEntropyLoss()

# dataset = NgramDataset(text, corpus)
dataset = NgramDataset(text, corpus, neg_sample=True)

dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True, drop_last=True
)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

model = model.to(device)

vocab = torch.tensor(corpus.encode(corpus.vocab)).expand(batch_size, corpus.word_count)
vocab = vocab.to(device)

# train_cbow_softmax(model, dataloader, vocab, 2e-3, epoches, device, writer)
# train_sg_softmax(model, dataloader, vocab, 2e-3, epoches, device, writer)
# train_cbow_neg_sampling(model, dataloader, lr=2e-3, epoches=epoches, device=device)
train_sg_neg_sample(model, dataloader, 2e-3, epoches, device, writer)

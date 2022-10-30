from CBOW_softmax import CBOW
from corpus import Corpus, load_data
import torch
import torch.nn as nn
from Dataset import NgramDataset
from utils import train_cbow_softmax

from d2l.torch import Timer
from torch.utils.tensorboard import SummaryWriter
import constant as C


writer = SummaryWriter()

text = list(load_data())
corpus = Corpus(text)
vocab_size = len(corpus.idx2word)

model = CBOW(vocab_size, C.embedding_dim)
dataset = NgramDataset(text, corpus)

dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=C.batch_size, shuffle=True, drop_last=True
)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

model = model.to(device)

vocab = torch.tensor(corpus.encode(corpus.vocab)).expand(C.batch_size, corpus.word_count)
vocab = vocab.to(device)

# train_cbow_softmax(model, dataloader, vocab, 2e-3, epoches, device, writer)
train_cbow_softmax(model, dataloader, vocab, 2e-3, C.epoches, device, writer)
# train_cbow_neg_sampling(model, dataloader, lr=2e-3, epoches=epoches, device=device)
# train_sg_neg_sample(model, dataloader, 2e-3, epoches, device, writer)

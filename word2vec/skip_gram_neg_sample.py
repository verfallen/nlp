from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim) -> None:
        super().__init__()
        self.embedding_in = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_out = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, center, context, neg):
        v = self.embedding_in(center)
        u = self.embedding_out(context)
        u_neg = -self.embedding_out(neg)

        pos_score = torch.bmm(u, v.transpose(1, 2)).squeeze(-1)
        neg_score = torch.bmm(u_neg, v.transpose(1, 2)).squeeze(-1)

        return self.loss(pos_score, neg_score)

    def loss(self, target_scores, neg_scores):
        prob = F.logsigmoid(target_scores).sum(1) + F.logsigmoid(neg_scores).sum(1)
        return -torch.mean(prob)

    def pred(self, word_idx):
        return (self.embedding_in(word_idx) + self.embedding_out(word_idx)) / 2

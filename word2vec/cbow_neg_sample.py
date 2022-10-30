import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim) -> None:
        super().__init__()
        self.embedding_in = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_out = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, contexts, targets, negs):
        v = self.embedding_in(contexts)
        u = self.embedding_out(targets)
        u_neg = -self.embedding_out(negs)

        target_scores = torch.bmm(u, v.transpose(1, 2)).squeeze(-1)
        neg_scores = torch.bmm(u_neg, v.transpose(1, 2)).squeeze(-1)

        return self.loss(target_scores, neg_scores)

    def loss(self, target_scores, neg_scores):

        prob = F.logsigmoid(target_scores) + F.logsigmoid(neg_scores).sum(1)
        return -torch.mean(prob)

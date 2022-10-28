from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim) -> None:
        super().__init__()
        self.embedding_in = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_out = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, contexts, targets, vocab):
        # (batch, win_size, embed)
        v = self.embedding_in(contexts)

        # (batch, 1, embed)
        u = self.embedding_out(targets)
        u_vocab = self.embedding_out(vocab)

        # (batch, embed)
        # x = x.mean(1).squeeze(1)

        # (batch, input_size)
        scores = torch.bmm(u, v.transpose(1, 2)).squeeze(2)
        norm_scores = torch.bmm(u_vocab, v.transpose(1, 2)).squeeze(2)

        return self.loss(scores, norm_scores)

    def loss(self, scores, norm_scores):
        prob = torch.exp(scores) / torch.sum(torch.exp(norm_scores), 1).unsqueeze(1)
        return -torch.mean(torch.log(prob))

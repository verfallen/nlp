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

    def forward(self, context, target):
        # (batch, win_size, embed)
        x = self.embedding_in(context)

        # (batch, embed)
        x = x.mean(1).squeeze(1)

        # (batch, vocab_size)
        x = self.embedding_out(x)
        return x

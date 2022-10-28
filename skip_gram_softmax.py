import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class SkigGramSoftmax(nn.Module):
    def __init__(self, vocab_size, embedding_dim) -> None:
        super().__init__()
        self.embedding_in = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_out = nn.Embedding(vocab_size, embedding_dim)
        
    def forward(self, inputs, targets, vocab):
        # (batch, embedding,1)
        v = self.embedding_in(inputs).unsqeeze(-1)
        
        # (batch, win_size, embed_size)
        u = self.embedding_out(targets)
        
        # (batch, vocab_size, embed_size)
        y_norm = self.embedding_out(vocab)
        
        #(batch, win_size)
        scores = torch.bmm(u,v).squeeze(-1)
        
        #(batch, vocab_size)
        scores_norm = torch.bmm(y_norm, v).squeeze(-1)
        
        return self.loss(scores, scores_norm)
    
    def loss(scores, scores_norm):
        prob = torch.exp(scores) / torch.sum(torch.exp(scores_norm), 1)
        return - torch.log(prob)
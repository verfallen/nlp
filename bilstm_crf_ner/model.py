import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from config import *


class BILSTM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.lstm = nn.LSTM(
            EMBEDDING_DIM, HIDDEN_SIZE, batch_first=True, bidirectional=True
        )
        self.linear = nn.Linear(2 * HIDDEN_SIZE, TARGET_SIZE)

    def _get_lstm_feature(self, input):
        embed = self.embedding(input)
        out, _ = self.lstm(embed)
        return self.linear(out)

    def forward(self, input):
        out = self._get_lstm_feature(input)
        return out


if __name__ == "__main__":
    model = BILSTM()
    input = torch.randint(1, 3000, (100, 50))
    print(model)
    print(model(input).shape)

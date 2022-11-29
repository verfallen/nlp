import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from config import *
from torchcrf import CRF


class Bilstm_crf(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.lstm = nn.LSTM(
            EMBEDDING_DIM, HIDDEN_SIZE, batch_first=True, bidirectional=True
        )
        self.linear = nn.Linear(2 * HIDDEN_SIZE, TARGET_SIZE)
        self.crf = CRF(TARGET_SIZE, batch_first=True)

    def _get_lstm_feature(self, input):
        embed = self.embedding(input)
        out, _ = self.lstm(embed)
        return self.linear(out)

    def forward(self, input, mask):
        out = self._get_lstm_feature(input)
        return self.crf.decode(out, mask)

    def loss_fn(self, input, target, mask):
        y_pred = self._get_lstm_feature(input)
        return -self.crf.forward(y_pred, target, mask, reduction="mean")


# if __name__ == "__main__":
#     model = Bilstm_crf()
#     input = torch.randint(1, 3000, (100, 50))
#     print(model)
#     print(model(input).shape)

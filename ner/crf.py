import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class CRF(nn.Module):
    def __init__(self, n_labels):
        super().__init__()
        self.n_labels = n_labels
        self.transitions = np.random.uniform(-0.1, 0.1, (n_labels + 2, n_labels + 2))

    def forward(self, emission: torch.Tensor) -> torch.Tensor:
        # (num_steps, num_labels)
        self.emission = input


if __name__ == "__main__":
    n_labels = 3
    a = np.random.uniform(-1, 1, n_labels).astype("f")
    b = np.random.uniform(-1, 1, n_labels).astype("f")
    x1 = np.stack([b, a])
    x2 = np.stack([a])
    xs = [x1, x2]
    print(torch.argmax(torch.Tensor([[2, 3, 4]]), 1))

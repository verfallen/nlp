from CBOW import CBOW
from corpus import Corpus, load_data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Dataset import NgramDataset


def train_cbow_origin(model, dataloader, loss, lr, epoches, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # loop over the dataset multiple times
    for epoch in range(epoches):
        running_loss = 0.0
        for inputs, labels in dataloader:
            print(inputs.shape, labels.shape)
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print('Epoch: {} Loss: {}'.format(epoch, running_loss))

    print('Finished Training')


text = list(load_data())
corpus = Corpus(text)
vocab_size = len(corpus.idx2word)
device = torch.device('cuda:0')
embedding_dim = 10
batch_size = 50

model = CBOW(vocab_size, embedding_dim)
criterion = nn.CrossEntropyLoss()
dataset = NgramDataset(text, corpus)

dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=10,
                                         shuffle=True,
                                         drop_last=True)
train_cbow_origin(model,
                  dataloader,
                  criterion,
                  lr=1e-3,
                  epoches=800,
                  device=device)
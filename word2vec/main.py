from CBOW_softmax import CBOW
from corpus import Corpus, load_data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Dataset import NgramDataset

text = list(load_data())
corpus = Corpus(text)
vocab_size = len(corpus.idx2word)
embedding_dim = 10
batch_size = 50

model = CBOW(vocab_size, embedding_dim)
criterion = nn.CrossEntropyLoss()
dataset = NgramDataset(text, corpus)


dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=10,
                                         shuffle=True,
                                         drop_last=True)

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def train_cbow_origin(model, dataloader, loss, lr, epoches, device):
    batch_size = 10
    vocab = torch.tensor(corpus.encode(corpus.vocab)).expand(batch_size, corpus.word_count)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epoches):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            loss = model(inputs, labels, vocab)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print('Epoch: {} Loss: {}'.format(epoch, running_loss))

    print('Finished Training')



train_cbow_origin(model,
                  dataloader,
                  criterion,
                  lr=1e-3,
                  epoches=800,
                  device=device)
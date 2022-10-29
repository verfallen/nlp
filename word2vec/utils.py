import torch


def train_cbow_softmax(model, dataloader, vocab, lr, epoches, device):
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

        print("Epoch: {} Loss: {}".format(epoch, running_loss))


def train_cbow_neg_sampling(model, dataloader, lr, epoches, device):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # loop over the dataset multiple times
    for epoch in range(epoches):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, labels, negs = data
            inputs, labels, negs = inputs.to(device), labels.to(
                device), negs.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            loss = model(inputs, labels, negs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print("Epoch: {} Loss: {}".format(epoch, running_loss))


def train_sg_neg_sample(model, dataloader, lr, epoches, device):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # loop over the dataset multiple times
    for epoch in range(epoches):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            labels, inputs, negs = data
            inputs, labels, negs = inputs.to(device), labels.to(
                device), negs.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            loss = model(inputs, labels, negs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print("Epoch: {} Loss: {}".format(epoch, running_loss))
    print("Finished Training")
    
    
def train_cbow_origin(model, dataloader, loss, lr, epoches, device):
    vocab = torch.tensor(corpus.encode(corpus.vocab)).expand(
        batch_size, corpus.word_count
    )
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

        print("Epoch: {} Loss: {}".format(epoch, running_loss))

    torch.save(model, "cbow_softmax.pt")

    print("Finished Training")

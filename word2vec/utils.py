import torch
from d2l.torch import Timer


def train_cbow_softmax(model, dataloader, vocab, lr, epoches, device, writer):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    time = Timer()
    time.start()
    for epoch in range(epoches):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            loss = model(inputs, labels, vocab)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        writer.add_scalars("loss", {"cbow_softmax": running_loss}, epoch + 1)
        print("Epoch: {} Loss: {}".format(epoch, running_loss))
    print("训练{}epoch 使用了 {} s".format(epoches, time.stop()))
    writer.close()
    torch.save(model, "./models/cbow_softmax32.pt")


def train_cbow_neg_sampling(model, dataloader, lr, epoches, device, writer):
    timer = Timer()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    timer.start()

    # loop over the dataset multiple times
    for epoch in range(epoches):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, labels, negs = data
            inputs, labels, negs = inputs.to(device), labels.to(device), negs.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            loss = model(inputs, labels, negs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print("Epoch: {} Loss: {}".format(epoch, running_loss))
        writer.add_scalars("loss", {"cbow_neg": running_loss}, epoch + 1)
    writer.close()
    print("训练{}epoch 使用了 {} s".format(epoches, timer.stop()))
    torch.save(model, "./models/cbow_neg32.pt")


def train_sg_neg_sample(model, dataloader, lr, epoches, device, writer):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # loop over the dataset multiple times
    time = Timer()
    time.start()
    for epoch in range(epoches):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            labels, inputs, negs = data
            inputs, labels, negs = inputs.to(device), labels.to(device), negs.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            loss = model(inputs, labels, negs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        writer.add_scalars("loss", {"sg_neg": running_loss}, epoch + 1)
        print("Epoch: {} Loss: {}".format(epoch, running_loss))
    print("训练时间：{} s".format(time.stop()))
    torch.save(model, "./models/sg_neg32.pt")
    print("Finished Training")


def train_sg_softmax(model, dataloader, vocab, lr, epoches, device, writer):
    time = Timer()
    time.start()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epoches):
        running_loss = 0.0
        for labels, inputs in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            loss = model(inputs, labels, vocab)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        writer.add_scalars("loss", {"sg_softmax": running_loss}, epoch + 1)
        print("Epoch: {} Loss: {}".format(epoch, running_loss))
    print("训练时间：{} s".format(time.stop()))
    writer.close()
    torch.save(model, "./models/sg_softmax32.pt")

    print("Finished Training")



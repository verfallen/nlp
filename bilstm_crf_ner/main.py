from utils import *
from model import *
from config import *
from torch.utils import data

if __name__ == "__main__":
    device = torch.device("cuda:0")
    dataset = NerDataset()
    train_loader = data.DataLoader(
        dataset, batch_size=256, shuffle=True, collate_fn=collate_fn
    )

    model = Bilstm_crf()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for i in range(EPOCH):
        for j, (input, target, mask) in enumerate(train_loader):
            input = input.to(device)
            target = target.to(device)
            mask = mask.to(device)

            y_pred = model(input, mask)
            loss = model.loss_fn(input, target, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if j % 10 == 0:
                print("epoch:\t", i, "loss:\t", loss.item())
        torch.save(model, MODEL_DIR + f"model_{i}.pth")

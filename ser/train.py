from torch import optim
import torch
import torch.nn.functional as F
import utils
import random
from transforms import flip

from ser.model import Net

global plotter
plotter = utils.VisdomLinePlotter(env_name="Tutorial Plots")


def train(run_path, params, train_dataloader, val_dataloader, device):
    # setup model
    model = Net().to(device)

    # setup params
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    # train
    for epoch in range(params.epochs):
        _train_batch(model, train_dataloader, optimizer, epoch, device)
        _val_batch(model, val_dataloader, device, epoch)

    # save model and save model params
    torch.save(model, run_path / "model.pt")


def _train_batch(model, dataloader, optimizer, epoch, device):
    tr_loss = 0
    correct = 0
    for i, (images, labels) in enumerate(dataloader):
        if random.uniform(0, 1) > 0.5:
            images = flip()(images)
        images, labels = images.to(device), labels.to(device)
        model.train()
        optimizer.zero_grad()
        output = model(images)
        loss = F.nll_loss(output, labels)
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()
        tr_loss += loss.item()

        print(
            f"Train Epoch: {epoch} | Batch: {i}/{len(dataloader)}"
            f"| Loss: {loss.item():.4f}"
        )
        plotter.plot(
            "Training Loss",
            "Batch",
            "Train Loss",
            epoch * len(dataloader) + i,
            loss.item(),
        )

    accuracy = correct / len(dataloader.dataset)
    tr_loss /= len(dataloader)
    plotter.plot("Accuracy", "Training", "Accuracy", epoch, accuracy)
    plotter.plot("Loss", "Training", "Loss", epoch, tr_loss)


@torch.no_grad()
def _val_batch(model, dataloader, device, epoch):
    val_loss = 0
    correct = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        model.eval()
        output = model(images)
        val_loss += F.nll_loss(output, labels, reduction="sum").item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()
    val_loss /= len(dataloader.dataset)
    accuracy = correct / len(dataloader.dataset)
    print(f"Val Epoch: {epoch} | Avg Loss: {val_loss:.4f} | Accuracy: {accuracy}")
    plotter.plot("Accuracy", "Validation", "Accuracy", epoch, accuracy)
    plotter.plot("Loss", "Validation", "Loss", epoch, val_loss)

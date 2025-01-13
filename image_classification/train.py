import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
import hydra


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}")


def test(model, testloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy of the network on the test images: {100 * correct / total:.2f}%")


def train_epoch(model, trainloader, optimizer, criterion):
    model.train()
    running_loss = 0.0

    for i, data in enumerate(trainloader):
        inputs, labels = data[0].to(device), data[1].to(device)
        model.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            print(f"\t[Batch {i + 1}] loss: {running_loss / 100:.3f}")


def train(model, trainloader, testloader, train_cfg: DictConfig):
    num_epochs = train_cfg.epochs
    optimizer = hydra.utils.instantiate(train_cfg.optimizer, params=model.parameters())
    criterion = hydra.utils.instantiate(train_cfg.loss_fn)
    model = model.to(device)

    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch+1}")
        train_epoch(model, trainloader, optimizer, criterion)
        test(model, testloader)

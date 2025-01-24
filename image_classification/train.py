import torch
from omegaconf import DictConfig
import hydra

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}")


def evaluate(model, criterion, testloader) -> tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss

    print(
        f"Eval accuracy of the network on the test images: {100 * correct / total:.2f}%"
    )
    return running_loss.item(), correct / total


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


def train(
    model, trainloader, testloader, cfg: DictConfig
) -> tuple[list[float], list[float]]:
    num_epochs = cfg.epochs
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    criterion = hydra.utils.instantiate(cfg.loss_fn)
    model = model.to(device)

    enable_early_stopping = cfg.get("enable_early_stopping", True)
    if enable_early_stopping:
        patience = cfg.get("early_stopping_patience", 5)
        delta = cfg.get("early_stopping_delta", 0.0)
        # 'min' for loss, 'max' for accuracy
        target = cfg.get("early_stopping_target", "loss")
        best_metric = float("inf") if target == "loss" else 0.0
        current_patience = 0
        best_weights = None

    losses = []
    accuracies = []
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch+1}")
        train_epoch(model, trainloader, optimizer, criterion)
        loss, accuracy = evaluate(model, criterion, testloader)

        losses.append(loss)
        accuracies.append(accuracy)

        if enable_early_stopping:
            if target == "loss":
                current_metric = loss
                improved = current_metric < best_metric - delta
            else:
                current_metric = accuracy
                improved = current_metric > best_metric + delta

            if improved:
                best_metric = current_metric
                current_patience = 0
                best_weights = model.state_dict().copy()
            else:
                current_patience += 1
                if current_patience >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    # load best performing model before breaking
                    if best_weights is not None:
                        model.load_state_dict(best_weights)
                    break

    return losses, accuracies

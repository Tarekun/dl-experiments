from typing import Callable, Tuple
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from omegaconf import DictConfig
import hydra

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def optimal_torch_config(model: nn.Module) -> nn.Module:
    model = model.to(device)
    try:
        import triton

        # reference: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
        torch.set_float32_matmul_precision("medium")
        torch.backends.cudnn.benchmark = True  # enable cuDNN for CNN
        # cuda graph to keep compute in gpu
        model = torch.compile(model, mode="reduce-overhead")
        return model
    except ImportError:
        print("Skipping torch.compile: Triton not installed")
        return model


def evaluate(
    model: nn.Module,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    testloader: DataLoader,
) -> Tuple[float, float]:
    """Runs evaluation of a `model` on a given evaluation set with the
    specified loss function.

    Parameters
    ----------
    model : nn.Module
        The neural network model to evaluate. Must implement the forward() method.
    criterion : Callable[[Tensor, Tensor], Tensor]
        Loss function that takes model outputs and targets, returning a scalar Tensor
    testloader : DataLoader[Tuple[Tensor, Tensor]]
        Iterable yielding batches of (inputs, labels) for supervised evaluation.

    Returns
    -------
    Tuple[float, float]
        A tuple containing:
        - Average loss across all test samples (float)
        - Accuracy (float) in range [0, 1]
    """

    model.eval()
    correct = 0
    total = 0
    running_loss = torch.tensor(0.0, device=device)

    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss

    print(
        f"Eval accuracy of the network on the test images: {100 * correct / total:.2f}%"
    )
    return running_loss.item(), correct / total


def train_epoch(
    model: nn.Module,
    trainloader: DataLoader,
    optimizer,
    criterion: Callable[[Tensor, Tensor], Tensor],
) -> float:
    """Performs one epoch of training on the given trainset

    Parameters
    ----------
    model: nn.Module
        The neural network model to train. Must implement the forward() method.
    trainloader: DataLoader[Tuple[Tensor, Tensor]]
        Iterable yielding batches of (inputs, labels) for supervised evaluation.
    optimizer
        The optimization algorithm to use during training
    criterion: Callable[[Tensor, Tensor], Tensor]
        Loss furunning_lossnction that takes model outputs and targets, returning a scalar Tensor

    Returns
    -------
    float
        The total running loss of the full epoch
    """

    model.train()
    # amp implementation copyed from https://www.reddit.com/r/MachineLearning/comments/kvs1ex/d_here_are_17_ways_of_making_pytorch_training/
    scaler = torch.amp.GradScaler("cuda")
    running_loss = torch.tensor(0.0, device=device)

    for i, data in enumerate(trainloader):
        inputs, labels = data[0].to(device), data[1].to(device)
        model.to(device)

        # zero the parameter gradients
        # reduces memory operations compare to optimizer.zero_grad()
        # see https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#use-parameter-grad-none-instead-of-model-zero-grad-or-optimizer-zero-grad
        for param in model.parameters():
            param.grad = None

        # forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # backward pass and optimize
        # scaler.scale(loss).backward()
        loss.backward()
        # scaler.step(optimizer)
        optimizer.step()
        # Updates the scale for next iteration
        # scaler.update()

        running_loss += loss
        # if i % 100 == 99:
        #     print(f"\t[Batch {i + 1}] loss: {running_loss / 100:.3f}")

    return running_loss.item()


def train(
    model: nn.Module, trainloader: DataLoader, testloader: DataLoader, cfg: DictConfig
) -> tuple[list[float], list[float]]:
    """Trains the model on the given trainset using hyperparameters specified in `cfg`

    Parameters
    ----------
    model: nn.Module
        The neural network model to evaluate. Must implement the forward() method.
    trainloader: DataLoader[Tuple[Tensor, Tensor]]
        Iterable yielding batches of (inputs, labels) for supervised training of the model.
    testloader: DataLoader[Tuple[Tensor, Tensor]]
        Iterable yielding batches of (inputs, labels) for supervised evaluation each epoch.
    cfg: DictConfig
        The Hydra configuration with hyperparameters for the training algorithm

    Returns
    -------
    tuple[list[float], list[float]]
        2 lists (losses, accuracies) containing evaluation loss and accuracy at each epoch
    """

    num_epochs = cfg.epochs
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    criterion = hydra.utils.instantiate(cfg.loss_fn)
    model = optimal_torch_config(model)

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
        train_loss = train_epoch(model, trainloader, optimizer, criterion)
        print(f"Total epoch loss: {train_loss}")
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

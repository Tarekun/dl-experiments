import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


train_transforms = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        # Random vertical flip with probability 0.2 (less common but can be useful)
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(15),
        # Randomly adjust the brightness, contrast, saturation, and hue
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        # Random grayscale conversion with probability 0.1
        transforms.RandomGrayscale(p=0.1),
        # Mean and std are pre-computed for CIFAR-10 RGB channels
        transforms.ToTensor(),  # Converts image to PyTorch tensor
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
        ),
    ]
)

test_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
        ),
    ]
)


def get_cifar_loaders() -> tuple[DataLoader, DataLoader]:
    trainset = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=True, transform=train_transforms
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=4
    )

    testset = torchvision.datasets.CIFAR100(
        root="./data", train=False, download=True, transform=test_transforms
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=4
    )

    return trainloader, testloader


def get_loaders(dataset: str) -> tuple[DataLoader, DataLoader]:
    if dataset == "cifar100":
        return get_cifar_loaders()

    else:
        raise ValueError(f"Unsupported dataset {dataset} configured")

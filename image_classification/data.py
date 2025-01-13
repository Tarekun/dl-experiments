import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]
)


def get_cifar_loaders() -> tuple[DataLoader, DataLoader]:
    trainset = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=4
    )

    testset = torchvision.datasets.CIFAR100(
        root="./data", train=False, download=True, transform=transform
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

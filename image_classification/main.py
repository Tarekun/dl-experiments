from omegaconf import DictConfig

from image_classification.models import *
from image_classification.data import get_loaders
from train import train
from utils import Experiment


class CnnCifar(Experiment):
    @staticmethod
    def name():
        return "cifar100"

    @staticmethod
    def run_experiment(cfg: DictConfig) -> tuple[list[float], list[float]]:
        trainloader, testloader = get_loaders(cfg.dataset)
        model = SimpleCNN(
            num_classes=100, conv_layers=cfg.conv_layers, lin_layers=cfg.lin_layers
        )
        model = ResNet(layers=[2, 2, 2, 2], num_classes=100)
        return train(model, trainloader, testloader, cfg)

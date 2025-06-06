from omegaconf import DictConfig

from image_classification.models import SimpleCNN
from image_classification.data import get_loaders
from train import train
from utils import Experiment


class CnnCifar(Experiment):
    @staticmethod
    def name():
        "suka"

    @staticmethod
    def run_experiment(cfg: DictConfig):
        trainloader, testloader = get_loaders(cfg.dataset)
        model = SimpleCNN(
            num_classes=100, conv_layers=cfg.conv_layers, lin_layers=cfg.lin_layers
        )
        train(model, trainloader, testloader, cfg)

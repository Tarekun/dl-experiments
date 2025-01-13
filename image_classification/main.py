from models import SimpleCNN
from train import train
from data import get_loaders
from omegaconf import DictConfig
import hydra


@hydra.main(config_path="config", config_name="main.yml")
def main(cfg: DictConfig):
    trainloader, testloader = get_loaders(cfg.dataset)

    model = SimpleCNN(
        num_classes=100, conv_layers=cfg.conv_layers, lin_layers=cfg.lin_layers
    )
    train(model, trainloader, testloader, cfg.train_cfg)
    print("Finished Training")


if __name__ == "__main__":
    main()

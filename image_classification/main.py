from models import SimpleCNN
from train import train
from data import get_loaders
from omegaconf import DictConfig
import omegaconf
import hydra


@hydra.main(config_path="config", config_name="main", version_base="1.2")
def main(cfg: DictConfig):
    print(f"Starting training with config:\n\n{omegaconf.OmegaConf.to_yaml(cfg)}")

    trainloader, testloader = get_loaders(cfg.dataset)
    model = SimpleCNN(
        num_classes=100, conv_layers=cfg.conv_layers, lin_layers=cfg.lin_layers
    )

    train(model, trainloader, testloader, cfg)
    print("Finished Training")


if __name__ == "__main__":
    main()

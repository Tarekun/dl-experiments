from omegaconf import DictConfig, OmegaConf
import hydra
import tqdm

from models import SimpleCNN
from train import train, device
from data import get_loaders
from visualization import plot_simulations
from commons import config_key


@hydra.main(config_path="config", config_name="main", version_base="1.2")
def main(cfg: DictConfig):
    print(f"Starting training with config:\n\n{OmegaConf.to_yaml(cfg)}")
    print(f"Running on {device}")
    trainloader, testloader = get_loaders(cfg.dataset)
    evaluations = []
    for run in tqdm.tqdm(
        range(cfg.num_runs),
        desc=config_key(cfg),
        leave=True,
    ):
        # for run in range(cfg.num_runs):
        print(f"Starting training run number {run}")
        model = SimpleCNN(
            num_classes=100, conv_layers=cfg.conv_layers, lin_layers=cfg.lin_layers
        )
        losses, accuracies = train(model, trainloader, testloader, cfg)
        evaluations.append((losses, accuracies))

    print("Finished Training")
    plot_simulations(evaluations, cfg)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"UNRECOVERABLE ERROR:\n{e}")

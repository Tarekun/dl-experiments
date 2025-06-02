from omegaconf import DictConfig, OmegaConf
import hydra
import tqdm

from models import SimpleCNN
from train import train, device
from data import get_loaders
from visualization import plot_simulations
from commons import config_key
from checkpoint import run_finished, config_finished, load_checkpoint


@hydra.main(config_path="config", config_name="main", version_base="1.2")
def main(cfg: DictConfig):
    print(f"Starting training with config:\n\n{OmegaConf.to_yaml(cfg)}")
    print(f"Running on {device}")

    trainloader, testloader = get_loaders(cfg.dataset)
    evaluations = []
    checkpoint = load_checkpoint()
    if cfg in checkpoint["configs"]:
        print("Skipping execution, this config was found in the checkpoint")
        return
    first_run = checkpoint["run"] + 1

    for run in tqdm.tqdm(
        range(first_run, cfg.num_runs),
        desc=config_key(cfg),
        leave=True,
    ):
        print(f"Starting training run number {run}")
        model = SimpleCNN(
            num_classes=100, conv_layers=cfg.conv_layers, lin_layers=cfg.lin_layers
        )
        losses, accuracies = train(model, trainloader, testloader, cfg)
        evaluations.append((losses, accuracies))
        run_finished(run)

    print("Finished Training")
    plot_simulations(evaluations, cfg)
    config_finished(cfg)
    run_finished(-1)  # reset to start back from the first ran


if __name__ == "__main__":
    try:
        print("calling main")
        main()
        print("main returned")
    except Exception as e:
        print(f"UNRECOVERABLE ERROR:\n{e}")

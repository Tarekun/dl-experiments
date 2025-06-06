from omegaconf import DictConfig, OmegaConf
import hydra
import tqdm
from typing import Type

from checkpoint import run_finished, config_finished, load_checkpoint
from train import device
from utils import config_key, Experiment
from visualization import plot_simulations

# experiments imports
from image_classification.main import CnnCifar


def get_experiment(cfg: DictConfig) -> Type[Experiment]:
    supported = [CnnCifar]

    for experiment in supported:
        if experiment.name() == cfg.experiment:
            return experiment

    raise ValueError(
        f"Configured experiment {cfg.experiment} is not supported. The supported ones are:\n{[e.name() for e in supported]}"
    )


@hydra.main(config_path="config", config_name="main", version_base="1.2")
def main(cfg: DictConfig):
    try:
        print(f"Starting training with config:\n\n{OmegaConf.to_yaml(cfg)}")
        print(f"Running on {device}")

        checkpoint = load_checkpoint()
        if cfg in checkpoint["configs"]:
            print("Skipping execution, this config was found in the checkpoint")
            return
        first_run = checkpoint["run"] + 1
        experiment = get_experiment(cfg)
        evaluations = []

        for run in tqdm.tqdm(
            range(first_run, cfg.num_runs),
            desc=config_key(cfg),
            leave=True,
        ):
            print(f"Starting training run number {run}")
            losses, accuracies = experiment.run_experiment(cfg)
            evaluations.append((losses, accuracies))
            run_finished(run)

        plot_simulations(evaluations, cfg)
        config_finished(cfg)
        run_finished(-1)  # reset to start back from the first ran
        print("Finished Training")

    except BaseException as e:
        import traceback
        import sys

        print("UNRECOVERABLE ERROR:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

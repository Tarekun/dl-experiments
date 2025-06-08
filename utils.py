from abc import ABC, abstractmethod
from omegaconf import DictConfig


OUT_DIRECTORY = ".internals"


class Experiment(ABC):
    @staticmethod
    @abstractmethod
    def name() -> str:
        """returns the unique name of the experiment"""
        pass

    @staticmethod
    @abstractmethod
    def run_experiment(cfg: DictConfig) -> tuple[list[float], list[float]]:
        """Performs a neural network training experiment

        Parameters
        ----------
        cfg: DictConfig
            The Hydra configuration of the experiment

        Returns
        -------
        tuple[list[float], list[float]]
            The 2 lists (losses, accuracies) of the evaluation performance of the
            network at each epoch, like they are returned by the common `train` function
        """
        pass


def config_key(cfg: DictConfig) -> str:
    name = ""
    name += f"-o{cfg.optimizer._target_}"
    name += f"-lr{cfg.optimizer.lr}"
    name += f"-con{cfg.conv_layers}"
    name += f"-lin{cfg.lin_layers}"
    name += f"-e{cfg.epochs}"
    return name

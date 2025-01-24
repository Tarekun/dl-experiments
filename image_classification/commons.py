from omegaconf import DictConfig


def config_key(cfg: DictConfig) -> str:
    name = ""
    name += f"-o{cfg.optimizer._target_}"
    name += f"-lr{cfg.optimizer.lr}"
    name += f"-con{cfg.conv_layers}"
    name += f"-lin{cfg.lin_layers}"
    name += f"-e{cfg.epochs}"
    return name

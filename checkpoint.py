from pathlib import Path
import yaml
from omegaconf import DictConfig, OmegaConf

from utils import OUT_DIRECTORY


CHECKPOINT_FILE = Path(f"./{OUT_DIRECTORY}/checkpoint.yml")


def load_checkpoint() -> dict:
    empty_checkpoint = {"run": -1, "configs": []}
    if not CHECKPOINT_FILE.exists():
        print("no checkpoint found, returning empty checkpoint")
        return empty_checkpoint

    try:
        file_data = yaml.safe_load(CHECKPOINT_FILE.read_text())
        if file_data is None:
            return empty_checkpoint

        return {**empty_checkpoint, **file_data}
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {CHECKPOINT_FILE}") from e


def store_checkpoint(checkpoint: dict):
    try:
        with open(CHECKPOINT_FILE, "w") as f:
            yaml.safe_dump(
                checkpoint,
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )
    except (IOError, PermissionError) as e:
        raise PermissionError(f"Cannot write to {CHECKPOINT_FILE}") from e
    except yaml.YAMLError as e:
        raise ValueError("Failed to serialize checkpoint data") from e


def run_finished(run: int):
    checkpoint = load_checkpoint()
    checkpoint["run"] = run
    store_checkpoint(checkpoint)


def config_finished(config: DictConfig):
    checkpoint = load_checkpoint()
    config_dict = OmegaConf.to_container(config, resolve=True)
    checkpoint["configs"].append(config_dict)
    store_checkpoint(checkpoint)

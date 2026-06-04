from pathlib import Path
import random
import yaml
import numpy as np
import torch


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def linear_epsilon(
    step: int,
    start: float,
    end: float,
    decay_steps: int
) -> float:
    if step >= decay_steps:
        return end

    ratio = step / max(1, decay_steps)
    return start + ratio * (end - start)

import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_checkpoint(model: torch.nn.Module, path: str | Path, model_name: str, accuracy: float) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(
        {
            "model_name": model_name,
            "state_dict": model.state_dict(),
            "accuracy": accuracy,
        },
        path,
    )


def load_checkpoint(model: torch.nn.Module, path: str | Path, device: torch.device) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)
    return checkpoint if isinstance(checkpoint, dict) else {}

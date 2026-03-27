import random
from collections import Counter
from pathlib import Path
import torch
import yaml
from datetime import datetime
from copy import deepcopy
"""
Helper functions for the AIMO3 project.

1. get_repo_root: Get the root directory of the repository.
2. get_config_path: Get the path to a YAML configuration file.
3. load_yaml: Load a YAML file and return its contents as a dictionary.
4. set_seeds: Set random seeds for reproducibility across random, torch, and CUDA (if available).
"""
def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

def get_config_path(yaml_file: str):
    return get_repo_root() / "configs" / yaml_file

def load_yaml(config_path) -> dict:
    config_path = Path(config_path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
    
def set_seeds(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

from datetime import datetime
from copy import deepcopy

def add_timestamp_to_report_names(config, keys, timestamp=None, fmt="%Y-%m-%d_%H-%M-%S"):
    """
    Return a copy of config with a timestamp appended to selected keys.

    Args:
        config (dict): Loaded YAML config.
        keys (tuple): Field names that should get timestamps.
        timestamp (str, optional): Precomputed timestamp. If None, generate one.
        fmt (str): Datetime format string.

    Returns:
        dict: Updated config.
    """
    config_copy = deepcopy(config)

    if timestamp is None:
        timestamp = datetime.now().strftime(fmt)

    def recurse(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in keys and isinstance(v, str):
                    obj[k] = f"{v}_{timestamp}"
                else:
                    recurse(v)
        elif isinstance(obj, list):
            for item in obj:
                recurse(item)

    recurse(config_copy)
    return config_copy
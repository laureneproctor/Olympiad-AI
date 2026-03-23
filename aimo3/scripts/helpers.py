import random
from collections import Counter
from pathlib import Path
import torch
import yaml
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

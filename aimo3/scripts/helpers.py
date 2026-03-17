import json
import os
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import yaml
from datasets import Dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

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

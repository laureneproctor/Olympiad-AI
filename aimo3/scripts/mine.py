# Input: A dataset of problems to mine ( usually val.test or a “seed hard set”) + current model
# - For each problem 
#   - Generates K sampled solutions
#   - Extract each predicted int (extracted_answer)
#   - Majority vote
#   - Computes:
#       - Majority correctness vs gold 
#       - Agreement rate
#       - Format validity rate
#   - If it meets “failure” criteria -> save it
# Output
# - data/rl_train/ ( a dataset of hard/ failure prompts with gold answers)
# This will be what GRPO trains on
import os
import random
from pathlib import Path
import torch
import yaml
from datasets import load_from_disk, Dataset, DatasetDict
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
#import aimo3.scripts.prepare as prep

def get_repo_root():
    """
    Assumes this file lives somewhere like:
      repo/src/sft.py
    so parents[1] is the repo root.
    """
    return Path(__file__).resolve().parents[1]
def get_sft_config_path():
    return get_repo_root() / "configs" / "sft.yaml"


def get_data_config_path():
    return get_repo_root() / "configs" / "data.yaml"

def get_mine_config_path():
    return get_repo_root() / "configs" / "mine.yaml"

def k_solutions(problem, model, k):
    # Generates K solutions for a specific problem
    solutions = []
    
    for x in range(k):
        solution = model.generate(problem)
        solutions.append(solution)
    
    return solutions

def extract_answer(solution):
    
    return 0

def majority_vote(extracted_answers):
    
    return 0
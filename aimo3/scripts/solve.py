# Input: final model + target problem set (competition set)
#   - for each problem:
#       - sample N solutions (N depends on difficulty or fixed)
#       - extract answers
#       - majority vote (optionally tie-break)
#       - write predictions file
# Output:
# - runs/preds.jsonl (submission-like: problem_id → integer answer)

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import torch
import yaml
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================================
# Path helpers
# ============================================================================
def get_repo_root():
    return Path(__file__).resolve().parents[1]

def get_solve_config_path():
    return get_repo_root() / "configs" / "solve.yaml"

def get_data_config_path():
    return get_repo_root() / "configs" / "data.yaml"

# ============================================================================
# YAML loading
# ============================================================================
def load_yaml(config_path):
    config_path = Path(config_path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# ============================================================================
# Answer extraction utilities
# ============================================================================
FINAL_RE = re.compile(r"FINAL_ANSWER:\s*([0-9]{1,5})\b")
LAST_INT_RE = re.compile(r"\b(\d{1,5})\b")

def normalize_to_int_str(ans: Optional[str]) -> Optional[str]:
    """
    Cleans and validates a string to ensure it is a non-negative integer
    between 0 and 99999. Returns canonical string form or None if invalid.
    """
    if ans is None:
        return None

    s = str(ans).strip()
    s = s.replace(" ", "").replace("\n", "").replace("\t", "")
    s = s.replace(",", "")

    if s == "":
        return None
    if s.startswith("+"):
        s = s[1:]
    if s.startswith("-"):
        return None
    if not s.isdigit():
        return None

    s = str(int(s))  # canonicalize leading zeros
    v = int(s)
    if not (0 <= v <= 99999):
        return None
    return s

def extract_final_answer(text: str) -> Optional[str]:
    """
    Extracts the last number labeled with 'FINAL_ANSWER:' if present.
    """
    matches = FINAL_RE.findall(text)
    if matches:
        return matches[-1].strip()
    return None

def extract_last_int_fallback(text: str) -> Optional[str]:
    """
    Returns the last standalone 1-5 digit integer found anywhere in the text.
    """
    matches = LAST_INT_RE.findall(text)
    if not matches:
        return None
    return matches[-1].strip()

def extract_answer(text: str) -> Optional[str]:
    """
    Applies extraction + normalization to return the final validated answer.
    """
    raw = extract_final_answer(text)
    if raw is None:
        raw = extract_last_int_fallback(text)
    return normalize_to_int_str(raw)

# ============================================================================
# Model loading
# ============================================================================
def load_model_and_tokenizer(
    model_path: str,
    trust_remote_code: bool = True,
):
    """
    Loads tokenizer + causal LM from a saved model directory or HF model id.
    """
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        trust_remote_code=trust_remote_code,
    )

    return tokenizer, model

# ============================================================================
# Generation helpers
# ============================================================================
@torch.inference_mode()
def generate_n_solutions(
    model,
    tokenizer,
    prompt: str,
    n: int = 8,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> List[str]:
    """
    Generates n solutions for a given prompt using sampling.
    """
    solutions = []
    for _ in range(n):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )
        
        solution = tokenizer.decode(outputs[0], skip_special_tokens=True)
        solutions.append(solution)
    
    return solutions

def majority_vote_answer(
    generations: List[str],
) -> Tuple[Optional[str], Counter, List[str], float]:
    """
    Extracts normalized answers from generated solutions and returns:
      - best majority answer
      - counts of all extracted answers
      - list of extracted normalized answers
      - agreement rate among valid extracted answers
    """
    extracted_norm = []
    for generation in generations:
        ans = extract_answer(generation)
        if ans is not None:
            extracted_norm.append(ans)

    if not extracted_norm:
        return None, Counter(), [], 0.0

    counts = Counter(extracted_norm)
    best, best_count = counts.most_common(1)[0]
    agreement = best_count / len(extracted_norm) if extracted_norm else 0.0

    return best, counts, extracted_norm, agreement

# ============================================================================
# Config loading
# ============================================================================
def load_configs(solve_config_path=None):
    """
    Loads solve.yaml and data.yaml configuration.
    """
    if solve_config_path is None:
        solve_config_path = get_solve_config_path()

    print(f"Loading solve config from {solve_config_path}...")
    solve_config = load_yaml(solve_config_path)

    data_config_path = solve_config.get("paths", {}).get("data_config_path")
    if not data_config_path:
        data_config_path = get_data_config_path()
    
    print(f"Loading data config from {data_config_path}...")
    data_config = load_yaml(data_config_path)

    return solve_config, data_config

# ============================================================================
# Dataset loading
# ============================================================================
def load_problem_dataset(dataset_path: str) -> Dataset:
    """
    Loads the problem dataset to solve.
    """
    print(f"Loading dataset from {dataset_path}...")
    ds = load_from_disk(dataset_path)
    print(f"Loaded {len(ds)} problems")
    return ds

# ============================================================================
# Main solving function
# ============================================================================
def solve(solve_config_path=None):
    """
    Main solve pipeline:
    1. Load config
    2. Load model and dataset
    3. For each problem, generate N solutions and majority vote
    4. Write predictions to JSONL
    """
    # Load configs
    solve_config, data_config = load_configs(solve_config_path)

    # Extract solve config
    model_checkpoint = solve_config.get("model_checkpoint")
    if not model_checkpoint:
        raise ValueError("model_checkpoint not specified in solve.yaml")
    
    dataset_path = solve_config.get("dataset_path")
    if not dataset_path:
        raise ValueError("dataset_path not specified in solve.yaml")
    
    n_samples = solve_config.get("n_samples", 8)
    max_new_tokens = solve_config.get("generation", {}).get("max_new_tokens", 512)
    temperature = solve_config.get("generation", {}).get("temperature", 0.7)
    top_p = solve_config.get("generation", {}).get("top_p", 0.95)
    
    output_path = solve_config.get("output_path", "runs/preds.jsonl")
    system_prompt = solve_config.get("system_prompt", "")

    # Load model and tokenizer
    tokenizer, model = load_model_and_tokenizer(model_checkpoint)

    # Load problem dataset
    ds = load_problem_dataset(dataset_path)

    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Solve problems
    predictions = []
    
    print(f"\nSolving {len(ds)} problems with n_samples={n_samples}...\n")
    
    for idx, problem_row in enumerate(ds):
        problem_id = problem_row.get("problem_id", str(idx))
        problem_text = problem_row.get("problem", "")
        
        # Format prompt
        if system_prompt:
            prompt = f"{system_prompt}\n\n{problem_text}"
        else:
            prompt = problem_text
        
        # Generate solutions
        solutions = generate_n_solutions(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            n=n_samples,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        
        # Majority vote
        best_answer, counts, extracted, agreement = majority_vote_answer(solutions)
        
        # Default to "0" if no valid answer extracted
        if best_answer is None:
            best_answer = "0"
        
        prediction = {
            "problem_id": str(problem_id),
            "predicted_answer": int(best_answer),
            "agreement": agreement,
            "n_valid_extractions": len(extracted),
        }
        predictions.append(prediction)
        
        if (idx + 1) % 10 == 0:
            print(f"Solved {idx + 1}/{len(ds)} problems...")
    
    # Write predictions to JSONL
    print(f"\nWriting predictions to {output_path}...")
    with open(output_path, "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")
    
    print(f"Done! Saved {len(predictions)} predictions to {output_path}")
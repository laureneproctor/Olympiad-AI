import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import yaml
from datasets import Dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from . import helpers

# ============================================================================
# Answer extraction utilities
# ============================================================================
FINAL_RE = re.compile(r"FINAL_ANSWER:\s*([0-9]{1,5})\b")
LAST_INT_RE = re.compile(r"\b(\d{1,5})\b")


def normalize_to_int_str(ans: Optional[str]) -> Optional[str]:
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

    s = str(int(s))
    v = int(s)
    if not (0 <= v <= 99999):
        return None
    return s


def extract_final_answer(text: str) -> Optional[str]:
    matches = FINAL_RE.findall(str(text))
    if matches:
        return matches[-1].strip()
    return None


def extract_last_int_fallback(text: str) -> Optional[str]:
    matches = LAST_INT_RE.findall(str(text))
    if not matches:
        return None
    return matches[-1].strip()


def extract_answer(text: str) -> Optional[str]:
    raw = extract_final_answer(text)
    if raw is None:
        raw = extract_last_int_fallback(text)
    return normalize_to_int_str(raw)


# ============================================================================
# Model loading
# ============================================================================
def load_model_and_tokenizer(model_path: str, trust_remote_code: bool = True):
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
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
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
    min_confidence: float = 0.1,
) -> List[str]:
    solutions = []
    device = next(model.parameters()).device

    for _ in range(n):
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        prompt_len = input_ids.shape[1]

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )

        output_ids = outputs.sequences[0]
        tail = output_ids[prompt_len:]
        
        # Confidence-based early stopping: truncate if model becomes uncertain
        if hasattr(outputs, 'scores') and outputs.scores and min_confidence > 0:
            truncate_at = len(tail)
            for j, score_tensor in enumerate(outputs.scores):
                probs = torch.softmax(score_tensor[0], dim=-1)
                max_prob = probs.max().item()
                if max_prob < min_confidence:
                    truncate_at = j
                    break
            tail = tail[:truncate_at]
        
        solution = tokenizer.decode(tail, skip_special_tokens=True)
        solutions.append(solution)

    return solutions


def majority_vote_answer(
    generations: List[str],
) -> Tuple[Optional[str], Counter, List[str], float]:
    extracted_norm = []
    for generation in generations:
        ans = extract_answer(generation)
        if ans is not None:
            extracted_norm.append(ans)

    if not extracted_norm:
        return None, Counter(), [], 0.0

    counts = Counter(extracted_norm)
    best, best_count = counts.most_common(1)[0]
    agreement = best_count / len(extracted_norm)

    return best, counts, extracted_norm, agreement


# ============================================================================
# Config loading
# ============================================================================
def load_configs(solve_config_path=None):
    if solve_config_path is None:
        solve_config_path = help.get_config_path("solve.yaml")

    print(f"Loading solve config from {solve_config_path}...")
    solve_config = help.load_yaml(solve_config_path)

    data_config_path = solve_config.get("paths", {}).get("data_config_path")
    if not data_config_path:
        data_config_path = help.get_config_path("data.yaml")

    print(f"Loading data config from {data_config_path}...")
    data_config = help.load_yaml(data_config_path)

    return solve_config, data_config


# ============================================================================
# Dataset loading
# ============================================================================
def load_problem_dataset(dataset_path: str) -> Dataset:
    print(f"Loading dataset from {dataset_path}...")
    ds = load_from_disk(dataset_path)

    if hasattr(ds, "keys"):
        split_name = "test" if "test" in ds else list(ds.keys())[0]
        print(f"DatasetDict detected, using split: {split_name}")
        ds = ds[split_name]

    print(f"Loaded {len(ds)} problems")
    return ds


# ============================================================================
# Main solving function
# ============================================================================
def solve(solve_config_path=None):
    solve_config, _data_config = load_configs(solve_config_path)

    paths_cfg = solve_config.get("paths", {})
    gen_cfg = solve_config.get("generation", {})
    prompt_cfg = solve_config.get("prompting", {})

    model_checkpoint = paths_cfg.get("model_checkpoint")
    if not model_checkpoint:
        raise ValueError("paths.model_checkpoint not specified in solve.yaml")

    dataset_path = paths_cfg.get("dataset_path")
    if not dataset_path:
        raise ValueError("paths.dataset_path not specified in solve.yaml")

    n_samples = int(gen_cfg.get("n_samples", 8))
    max_new_tokens = int(gen_cfg.get("max_new_tokens", 512))
    temperature = float(gen_cfg.get("temperature", 0.7))
    top_p = float(gen_cfg.get("top_p", 0.95))

    output_path = paths_cfg.get("output_path", "runs/preds.jsonl")
    system_prompt = prompt_cfg.get("system_prompt", "")

    tokenizer, model = load_model_and_tokenizer(model_checkpoint)
    ds = load_problem_dataset(dataset_path)

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    predictions = []

    print(f"\nSolving {len(ds)} problems with n_samples={n_samples}...\n")

    for idx, problem_row in enumerate(ds):
        problem_id = problem_row.get("problem_id", str(idx))
        problem_text = problem_row.get("problem", "")

        if system_prompt:
            prompt = f"{system_prompt}\n\nProblem:\n{problem_text}\n\nSolution:\n"
        else:
            prompt = f"Problem:\n{problem_text}\n\nSolution:\n"

        solutions = generate_n_solutions(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            n=n_samples,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        best_answer, _counts, extracted, agreement = majority_vote_answer(solutions)

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

    print(f"\nWriting predictions to {output_path}...")
    with open(output_path, "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")

    print(f"Done! Saved {len(predictions)} predictions to {output_path}")
# Input: model checkpoint + evaluation split
# - Does:
#       - pass@1 with greedy generation
#       - maj@N with sampling + majority vote
#       - reports format validity and agreement stats
# Output:
# - printed metrics + optional runs/report.json

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import torch
import yaml
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from helpers import get_config_path, load_yaml, set_seeds

try:
    from peft import AutoPeftModelForCausalLM
except ImportError:
    AutoPeftModelForCausalLM = None


# ---------------------------
# Path helpers
# ---------------------------

""" def get_repo_root():
    return Path(__file__).resolve().parents[1]


def get_evaluate_config_path():
    return get_repo_root() / "configs" / "evaluate.yaml"


def get_data_config_path():
    return get_repo_root() / "configs" / "data.yaml"


def get_sft_config_path():
    return get_repo_root() / "configs" / "sft.yaml" """


# ---------------------------
# YAML loading
# ---------------------------

def load_yaml(config_path):
    config_path = Path(config_path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_configs(evaluate_config_path=None):
    if evaluate_config_path is None:
        evaluate_config_path = get_config_path("evaluate.yaml")

    evaluate_config = load_yaml(evaluate_config_path)

    data_config_path = evaluate_config.get("paths", {}).get("data_config_path")
    if not data_config_path:
        data_config_path = get_config_path("data.yaml")
    data_config = load_yaml(data_config_path)

    sft_config_path = evaluate_config.get("paths", {}).get("sft_config_path")
    if not sft_config_path:
        sft_config_path = get_config_path("sft.yaml")
    sft_config = load_yaml(sft_config_path)

    exp_name = sft_config["run"]["experiment_name"]
    model_key = sft_config["run"]["model_key"]
    exp_config = data_config["experiments"][exp_name]
    model_config = data_config["models"][model_key]

    dataset_path = evaluate_config.get("paths", {}).get("dataset_path")
    if not dataset_path:
        dataset_path = os.path.join(
            sft_config["paths"]["prepared_splits_root"],
            "splits",
            str(exp_config["N"]),
        )

    model_checkpoint = evaluate_config.get("paths", {}).get("model_checkpoint")
    if not model_checkpoint:
        model_checkpoint = os.path.join(
            sft_config["paths"]["output_root"],
            f"sft_{model_key}_{exp_name}",
        )

    output_dir = evaluate_config.get("paths", {}).get("output_dir", "runs/evaluation")

    return (
        evaluate_config,
        data_config,
        sft_config,
        exp_config,
        model_config,
        dataset_path,
        model_checkpoint,
        output_dir,
    )


# ---------------------------
# Answer extraction utilities
# ---------------------------

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

    s = str(int(s))
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


# ---------------------------
# Prompt formatting
# ---------------------------

def build_eval_prompt(
    example,
    system_prompt: str,
    problem_field: str = "problem",
    answer_field: str = "expected_answer",
):
    problem = str(example[problem_field]).strip()
    answer = str(example[answer_field]).strip()

    prompt = (
        system_prompt
        + "\n\nProblem:\n"
        + problem
        + "\n\nSolution:\n"
    )

    return {
        "prompt": prompt,
        "expected_answer": answer,
    }


def format_eval_split(
    ds_split: Dataset,
    system_prompt: str,
    problem_field: str = "problem",
    answer_field: str = "expected_answer",
) -> Dataset:
    """
    Converts a raw prepared split into the columns expected by evaluation.
    """
    return ds_split.map(
        build_eval_prompt,
        fn_kwargs={
            "system_prompt": system_prompt,
            "problem_field": problem_field,
            "answer_field": answer_field,
        },
        remove_columns=ds_split.column_names,
        desc="Formatting eval split",
    )


def load_eval_split(dataset_path: str, split_name: str) -> Dataset:
    """
    Supports:
      - a DatasetDict saved with train/val/test
      - a Dataset saved directly
    """
    print("Loading evaluation dataset from:", dataset_path)
    ds = load_from_disk(dataset_path)

    if hasattr(ds, "keys"):
        if split_name not in ds:
            raise ValueError(f"Split '{split_name}' not found in dataset at {dataset_path}")
        split_ds = ds[split_name]
    else:
        split_ds = ds

    print("Loaded eval rows:", len(split_ds))
    print("Columns:", split_ds.column_names)
    return split_ds


# ---------------------------
# Model loading
# ---------------------------

def get_inference_dtype():
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def load_eval_model(
    model_path: str,
    trust_remote_code: bool = True,
):
    """
    Loads tokenizer + model from:
      - a PEFT/LoRA adapter directory, or
      - a full HF model directory.
    """
    tok = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        trust_remote_code=trust_remote_code,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": get_inference_dtype(),
        "trust_remote_code": trust_remote_code,
    }

    if AutoPeftModelForCausalLM is not None:
        try:
            model = AutoPeftModelForCausalLM.from_pretrained(
                model_path,
                **model_kwargs,
            )
            return tok, model
        except Exception as e:
            print("PEFT load failed, falling back to AutoModelForCausalLM:")
            print(str(e))

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **model_kwargs,
    )
    return tok, model


# ---------------------------
# Generation helpers
# ---------------------------

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
    Generates multiple sampled responses for a prompt.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=n,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    generations = []
    for output in outputs:
        tail = output[prompt_len:]
        generations.append(tokenizer.decode(tail, skip_special_tokens=True))

    return generations


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
        norm = extract_answer(generation)
        if norm is not None:
            extracted_norm.append(norm)

    if not extracted_norm:
        return None, Counter(), [], 0.0

    counts = Counter(extracted_norm)
    best, best_count = counts.most_common(1)[0]
    agreement = best_count / len(extracted_norm)

    return best, counts, extracted_norm, agreement


@torch.inference_mode()
def solve_pass1(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
) -> Tuple[Optional[str], bool, str]:
    """
    Generates one greedy response and extracts the answer.
    Returns:
      - predicted answer
      - whether a valid answer format was extracted
      - decoded generation text
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    text = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
    pred = extract_answer(text)

    return pred, (pred is not None), text


# ---------------------------
# Evaluation functions
# ---------------------------

def resolve_total(max_items, ds_len: int) -> int:
    if max_items is None:
        return ds_len
    return min(max_items, ds_len)


def eval_split_pass1(
    model,
    tokenizer,
    ds_split: Dataset,
    max_items: Optional[int] = 100,
    max_new_tokens: int = 512,
) -> Dict[str, float]:
    """
    Evaluates a dataset split with greedy decoding (pass@1).
    """
    correct = 0
    valid = 0
    total = resolve_total(max_items, len(ds_split))

    for i in range(total):
        pred, is_valid, _ = solve_pass1(
            model=model,
            tokenizer=tokenizer,
            prompt=ds_split[i]["prompt"],
            max_new_tokens=max_new_tokens,
        )
        gt = normalize_to_int_str(ds_split[i]["expected_answer"])

        if is_valid:
            valid += 1
        if pred is not None and gt is not None and pred == gt:
            correct += 1

    return {
        "pass@1": correct / total if total else 0.0,
        "format_valid_rate": valid / total if total else 0.0,
        "n": total,
    }


def eval_split_majN(
    model,
    tokenizer,
    ds_split: Dataset,
    n_samples: int = 8,
    max_items: Optional[int] = 50,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> Dict[str, float]:
    """
    Evaluates a dataset split with sampled decoding and majority vote.
    """
    correct = 0
    problems_with_any_valid = 0
    sum_agreement = 0.0
    total_valid_votes = 0
    total_votes = 0

    total = resolve_total(max_items, len(ds_split))

    for i in range(total):
        prompt = ds_split[i]["prompt"]
        gt = normalize_to_int_str(ds_split[i]["expected_answer"])

        generations = generate_n_solutions(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            n=n_samples,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        best, _counts, extracted_norm, agreement = majority_vote_answer(generations)

        if len(extracted_norm) > 0:
            problems_with_any_valid += 1
            sum_agreement += agreement

        total_valid_votes += len(extracted_norm)
        total_votes += n_samples

        if best is not None and gt is not None and best == gt:
            correct += 1

    per_sample_valid_rate = (total_valid_votes / total_votes) if total_votes else 0.0
    per_problem_any_valid = (problems_with_any_valid / total) if total else 0.0
    avg_agreement = (
        sum_agreement / problems_with_any_valid if problems_with_any_valid else 0.0
    )

    return {
        f"maj@{n_samples}": correct / total if total else 0.0,
        "per_sample_format_valid_rate": per_sample_valid_rate,
        "per_problem_any_valid_rate": per_problem_any_valid,
        "avg_agreement_rate": avg_agreement,
        "n": total,
    }


# ---------------------------
# Main runner
# ---------------------------

def run_evaluation(evaluate_config_path=None):
    (
        evaluate_config,
        data_config,
        sft_config,
        exp_config,
        model_config,
        dataset_path,
        model_checkpoint,
        output_dir,
    ) = load_configs(evaluate_config_path=evaluate_config_path)

    eval_yaml = evaluate_config.get("evaluation", {})
    reporting_yaml = evaluate_config.get("reporting", {})
    data_yaml = sft_config.get("data", {})

    split_name = eval_yaml.get("split", "test")
    system_prompt = (
        evaluate_config.get("prompting", {}).get("system_prompt")
        or sft_config["prompting"]["system_prompt"]
    )
    problem_field = data_yaml.get("problem_field", "problem")
    answer_field = data_yaml.get("answer_field", "expected_answer")

    print("Experiment:", sft_config["run"]["experiment_name"])
    print("Model key:", sft_config["run"]["model_key"])
    print("Model name:", model_config["name"])
    print("Dataset path:", dataset_path)
    print("Checkpoint:", model_checkpoint)
    print("Eval split:", split_name)

    ds_raw = load_eval_split(dataset_path, split_name)
    ds_eval = format_eval_split(
        ds_split=ds_raw,
        system_prompt=system_prompt,
        problem_field=problem_field,
        answer_field=answer_field,
    )

    tokenizer, model = load_eval_model(model_checkpoint)

    results = {
        "metadata": {
            "experiment_name": sft_config["run"]["experiment_name"],
            "model_key": sft_config["run"]["model_key"],
            "model_name": model_config["name"],
            "checkpoint": model_checkpoint,
            "dataset_path": dataset_path,
            "split": split_name,
            "n_dataset_rows": len(ds_eval),
        }
    }

    if eval_yaml.get("eval_pass1", True):
        print("Running pass@1 evaluation...")
        results["pass1"] = eval_split_pass1(
            model=model,
            tokenizer=tokenizer,
            ds_split=ds_eval,
            max_items=eval_yaml.get("max_items", 100),
            max_new_tokens=eval_yaml.get("pass1_max_new_tokens", 512),
        )

    if eval_yaml.get("eval_majn", True):
        print("Running maj@N evaluation...")
        results["majn"] = eval_split_majN(
            model=model,
            tokenizer=tokenizer,
            ds_split=ds_eval,
            n_samples=eval_yaml.get("majn_n_samples", 8),
            max_items=eval_yaml.get("majn_max_items", 50),
            max_new_tokens=eval_yaml.get("majn_max_new_tokens", 512),
            temperature=eval_yaml.get("majn_temperature", 0.7),
            top_p=eval_yaml.get("majn_top_p", 0.95),
        )

    if reporting_yaml.get("verbose", True):
        print(json.dumps(results, indent=2))

    if reporting_yaml.get("save_report", False):
        os.makedirs(output_dir, exist_ok=True)
        report_filename = reporting_yaml.get("report_filename", "evaluation_report.json")
        report_path = os.path.join(output_dir, report_filename)
        with open(report_path, "w") as f:
            json.dump(results, f, indent=2)
        print("Saved evaluation report to:", report_path)

    return results
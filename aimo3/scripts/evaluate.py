# Input: model checkpoint + evaluation split
# Metrics: pass@1 (greedy), maj@N (sampling + majority vote)
# Output: printed metrics + optional JSON report

import json
import os
import re
from collections import Counter
from typing import Optional, List, Tuple, Dict

import torch
from datasets import Dataset, load_from_disk
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from . import helpers

try:
    from peft import AutoPeftModelForCausalLM
except ImportError:
    AutoPeftModelForCausalLM = None

# ---------------------------
# YAML loading
# ---------------------------
def load_configs(evaluate_config_path=None):
    """Load and resolve evaluation, data, SFT, and model configs."""
    if evaluate_config_path is None:
        evaluate_config_path = helpers.get_config_path("evaluate.yaml")

    evaluate_config = helpers.load_yaml(evaluate_config_path)

    data_config_path = evaluate_config.get("paths", {}).get("data_config_path")
    if not data_config_path:
        data_config_path = helpers.get_config_path("data.yaml")
    data_config = helpers.load_yaml(data_config_path)

    sft_config_path = evaluate_config.get("paths", {}).get("sft_config_path")
    if not sft_config_path:
        sft_config_path = helpers.get_config_path("sft.yaml")
    sft_config = helpers.load_yaml(sft_config_path)

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
    """Normalize and validate string to 0-99999 range, return canonical form."""
    if ans is None:
        return None
    s = str(ans).strip().replace(" ", "").replace("\n", "").replace("\t", "").replace(",", "")
    if s == "" or s.startswith("-"):
        return None
    if s.startswith("+"):
        s = s[1:]
    if not s.isdigit():
        return None
    v = int(s)
    if not (0 <= v <= 99999):
        return None
    return str(v)


def extract_final_answer(text: str) -> Optional[str]:
    """Extract last number labeled with 'FINAL_ANSWER:' if present."""
    matches = FINAL_RE.findall(text)
    return matches[-1].strip() if matches else None


def extract_last_int_fallback(text: str) -> Optional[str]:
    """Extract last 1-5 digit integer found in text."""
    matches = LAST_INT_RE.findall(text)
    return matches[-1].strip() if matches else None


def extract_answer(text: str) -> Optional[str]:
    """Extract and normalize final answer from text."""
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
    """Build one evaluation prompt from a dataset example."""
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
    """Convert raw split into prompt/answer rows for evaluation."""
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
    """Load evaluation split from disk."""
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
    """Choose torch dtype for inference based on available hardware."""
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def load_eval_model(
    model_path: str,
    tokenizer_path: Optional[str] = None,
    trust_remote_code: bool = True,
):
    """Load tokenizer and model for evaluation."""
    tokenizer_candidates = [model_path]
    if tokenizer_path and tokenizer_path not in tokenizer_candidates:
        tokenizer_candidates.append(tokenizer_path)

    tok = None
    tokenizer_errors = []
    for candidate in tokenizer_candidates:
        try:
            tok = AutoTokenizer.from_pretrained(
                candidate,
                use_fast=True,
                trust_remote_code=trust_remote_code,
            )
            break
        except Exception as e_fast:
            tokenizer_errors.append(f"{candidate} (fast): {e_fast}")
            try:
                tok = AutoTokenizer.from_pretrained(
                    candidate,
                    use_fast=False,
                    trust_remote_code=trust_remote_code,
                )
                break
            except Exception as e_slow:
                tokenizer_errors.append(f"{candidate} (slow): {e_slow}")

    if tok is None:
        error_blob = "\n".join(tokenizer_errors)
        raise ValueError(
            "Failed to load tokenizer from model checkpoint or base tokenizer path. "
            f"Tried: {tokenizer_candidates}\n{error_blob}"
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
    """Generate n sampled responses from a prompt."""
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
    for output_ids in outputs:
        tail = output_ids[prompt_len:]
        generations.append(tokenizer.decode(tail, skip_special_tokens=True))

    return generations


def majority_vote_answer(
    generations: List[str],
) -> Tuple[Optional[str], Counter, List[str], float]:
    """Aggregate answers with majority voting from multiple generations."""
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
    """Generate one greedy response and extract its answer."""
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

def eval_split_pass1(
    model,
    tokenizer,
    ds_split: Dataset,
    max_items: Optional[int] = 100,
    max_new_tokens: int = 512,
) -> Dict[str, float]:
    """Evaluate dataset with greedy generation (pass@1)."""
    correct = 0
    valid = 0
    total = len(ds_split) if max_items is None else min(max_items, len(ds_split))

    for i in tqdm(range(total), desc="Pass@1 Evaluation", unit="example"):
        prompt = ds_split[i]["prompt"]
        pred, is_valid, _ = solve_pass1(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
        gt = normalize_to_int_str(ds_split[i]["expected_answer"])
        is_correct = pred is not None and gt is not None and pred == gt

        if is_valid:
            valid += 1
        if is_correct:
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
    """Evaluate dataset with sampling and majority voting."""
    correct = 0
    problems_with_any_valid = 0
    sum_agreement = 0.0
    total_valid_votes = 0
    total_votes = 0

    total = len(ds_split) if max_items is None else min(max_items, len(ds_split))

    for i in tqdm(range(total), desc=f"Maj@{n_samples} Evaluation", unit="example"):
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
    """Run full evaluation pipeline on a model checkpoint."""
    # Load configs
    (
        evaluate_config,
        _data_config,
        sft_config,
        _exp_config,
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

    # Load dataset and format
    ds_eval = format_eval_split(
        ds_split=load_eval_split(dataset_path, split_name),
        system_prompt=system_prompt,
        problem_field=problem_field,
        answer_field=answer_field,
    )
    
    # Load model and tokenizer
    tokenizer_source = model_config.get("tokenizer_name") or model_config.get("name")
    tokenizer, model = load_eval_model(
        model_checkpoint,
        tokenizer_path=tokenizer_source,
    )
    
    # Run evaluation metrics
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
        print("\nRunning pass@1 evaluation...")
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

    # Print and save results
    if reporting_yaml.get("verbose", True):
        print("\n" + json.dumps(results, indent=2))

    if reporting_yaml.get("save_report", False):
        os.makedirs(output_dir, exist_ok=True)
        report_filename = reporting_yaml.get("report_filename", "evaluation_report.json")
        report_path = os.path.join(output_dir, report_filename)
        with open(report_path, "w") as f:
            json.dump(results, f, indent=2)
        print("Saved evaluation report to:", report_path)

    return results
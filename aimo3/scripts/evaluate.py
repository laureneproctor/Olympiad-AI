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
    """
        Loads and resolves all configs and paths needed for evaluation.
    Input:
    - evaluate_config_path: optional path to the evaluation config YAML file
    Output:
        - tuple: (evaluate_config, data_config, sft_config, exp_config, model_config,
            dataset_path, model_checkpoint, output_dir)
    """
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
    """
    Cleans and validates a string to ensure it is a non-negative integer
    between 0 and 99999. Returns canonical string form or None if invalid.
    Input: 
    - ans: raw string to normalize
    Output:
    - s: normalized string if valid, else None
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
    Input:
    - text: model generation text
    Output:
    - answer: extracted digit string, or None if not found
    """
    matches = FINAL_RE.findall(text)
    if matches:
        return matches[-1].strip()
    return None


def extract_last_int_fallback(text: str) -> Optional[str]:
    """
    Extracts the last standalone 1-5 digit integer found in the text.
    Input:
    - text: model generation text
    Output:
    - answer: extracted digit string, or None if not found
    """
    matches = LAST_INT_RE.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


def extract_answer(text: str) -> Optional[str]:
    """
    Extracts and normalizes the final validated answer.
    Input:
    - text: model generation text
    Output:
    - answer: normalized answer string, or None if invalid/not found
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
    """
    Builds one evaluation prompt row from a dataset example.
    Input:
    - example: dataset row containing problem and expected answer
    - system_prompt: instruction prefix prepended before the problem
    - problem_field: key name for the problem text in example
    - answer_field: key name for the expected answer in example
    Output:
    - row: dict with prompt and expected_answer fields
    """
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
    Converts a raw prepared split into prompt/answer rows for evaluation.
    Input:
    - ds_split: input dataset split
    - system_prompt: instruction prefix added to each prompt
    - problem_field: key name for the problem text column
    - answer_field: key name for the expected answer column
    Output:
    - ds_formatted: dataset with prompt and expected_answer columns
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
        Loads an evaluation split from disk.
        Input:
        - dataset_path: path to a saved Dataset or DatasetDict
        - split_name: split name to load when dataset_path is a DatasetDict
        Output:
        - split_ds: loaded Dataset split
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
    """
    Chooses torch dtype for inference based on available hardware.
    Input:
    - none
    Output:
    - dtype: torch dtype (bfloat16, float16, or float32)
    """
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
        Loads tokenizer and model for evaluation.
        Input:
        - model_path: path to a PEFT adapter or full HF model checkpoint
        - trust_remote_code: whether to allow custom remote model code
        Output:
        - tuple: (tokenizer, model)
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
    Generates multiple sampled responses for one prompt.
    Input:
    - model: loaded causal language model
    - tokenizer: tokenizer paired with the model
    - prompt: prompt text to generate from
    - n: number of sampled completions to generate
    - max_new_tokens: maximum generated tokens per completion
    - temperature: sampling temperature
    - top_p: nucleus sampling threshold
    Output:
    - generations: list of decoded generated solution strings
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
    Aggregates extracted answers from generations with majority voting.
    Input:
    - generations: list of generated solution texts
    Output:
    - best: most common normalized answer, or None if no valid answers
    - counts: Counter with normalized answer frequencies
    - extracted_norm: list of valid normalized answers
    - agreement: majority count divided by number of valid answers
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
        Generates one greedy response and extracts its normalized answer.
        Input:
        - model: loaded causal language model
        - tokenizer: tokenizer paired with the model
        - prompt: prompt text to solve
        - max_new_tokens: maximum generated tokens
        Output:
        - pred: normalized predicted answer, or None
        - is_valid: True if pred is a valid extracted answer
        - text: decoded generated response text
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
    """
    Resolves how many dataset rows to evaluate.
    Input:
    - max_items: optional cap on number of rows
    - ds_len: total rows in the dataset split
    Output:
    - total: number of rows to evaluate
    """
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
    Evaluates one dataset split with greedy decoding (pass@1).
    Input:
    - model: loaded causal language model
    - tokenizer: tokenizer paired with the model
    - ds_split: formatted evaluation dataset split
    - max_items: optional cap on evaluated rows
    - max_new_tokens: maximum generated tokens per row
    Output:
    - metrics: dict with pass@1, format_valid_rate, and n
    """
    correct = 0
    valid = 0
    total = resolve_total(max_items, len(ds_split))

    for i in tqdm(range(total), desc="Pass@1 Evaluation", total=total, unit="example"):
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
    Evaluates one dataset split with sampling and majority voting.
    Input:
    - model: loaded causal language model
    - tokenizer: tokenizer paired with the model
    - ds_split: formatted evaluation dataset split
    - n_samples: sampled generations per problem
    - max_items: optional cap on evaluated rows
    - max_new_tokens: maximum generated tokens per sample
    - temperature: sampling temperature
    - top_p: nucleus sampling threshold
    Output:
    - metrics: dict with maj@N accuracy and vote/format statistics
    """
    correct = 0
    problems_with_any_valid = 0
    sum_agreement = 0.0
    total_valid_votes = 0
    total_votes = 0

    total = resolve_total(max_items, len(ds_split))

    for i in tqdm(range(total), desc=f"Maj@{n_samples} Evaluation", total=total, unit="example"):
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


def print_preview_examples(
    model,
    tokenizer,
    ds_raw: Dataset,
    ds_eval: Dataset,
    n_examples: int,
    problem_field: str,
    answer_field: str,
    pass1_max_new_tokens: int,
    n_samples: int,
    majn_max_new_tokens: int,
    majn_temperature: float,
    majn_top_p: float,
    generation_preview_chars: int = 180,
):
    """
    Prints a small qualitative preview with gold answer, pass@1, and maj@N details.
    Input:
    - model/tokenizer: loaded inference components
    - ds_raw: original split with problem/answer fields
    - ds_eval: formatted split with prompts
    - n_examples: number of examples to print
    - problem_field/answer_field: source column names in ds_raw
    - pass1_max_new_tokens: generation length for pass@1 preview
    - n_samples: number of samples for maj@N preview
    - majn_max_new_tokens/majn_temperature/majn_top_p: sampling params
    - generation_preview_chars: truncation length for generation text snippets
    Output:
    - none
    """
    total = min(int(n_examples), len(ds_eval))
    if total <= 0:
        return

    print("\n=== Preview Examples ===")
    for i in range(total):
        problem_text = str(ds_raw[i].get(problem_field, ""))
        gt = normalize_to_int_str(ds_raw[i].get(answer_field, None))
        prompt = ds_eval[i]["prompt"]

        pass1_pred, _pass1_valid, pass1_text = solve_pass1(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=pass1_max_new_tokens,
        )
        pass1_correct = (
            pass1_pred is not None and gt is not None and pass1_pred == gt
        )

        generations = generate_n_solutions(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            n=n_samples,
            max_new_tokens=majn_max_new_tokens,
            temperature=majn_temperature,
            top_p=majn_top_p,
        )
        maj_pred, vote_counts, extracted_norm, agreement = majority_vote_answer(generations)
        maj_correct = maj_pred is not None and gt is not None and maj_pred == gt

        print(f"\n--- Example {i + 1}/{total} ---")
        print("Gold:", gt)
        print("Pass@1 pred:", pass1_pred, "| correct:", pass1_correct)
        print("Maj@N pred:", maj_pred, "| correct:", maj_correct)
        print("Agreement:", round(float(agreement), 4), "| vote_counts:", dict(vote_counts))

        problem_preview = problem_text.replace("\n", " ")
        print("Problem preview:", problem_preview[:220])

        pass1_preview = pass1_text.replace("\n", " ")
        print("Pass@1 text preview:", pass1_preview[:generation_preview_chars])

        print("Sampled generations (extracted answer | matches gold):")
        for j, generation_text in enumerate(generations, start=1):
            ans = extract_answer(generation_text)
            is_match = (ans is not None and gt is not None and ans == gt)
            gen_preview = generation_text.replace("\n", " ")
            print(
                f"  [{j}] ans={ans} | match={is_match} | text={gen_preview[:generation_preview_chars]}"
            )

        print("Valid extracted answers:", extracted_norm)

def print_differing_examples_pass1(
    baseline_model,
    baseline_tokenizer,
    sft_model,
    sft_tokenizer,
    ds_raw: Dataset,
    ds_eval: Dataset,
    n_examples: int,
    problem_field: str,
    answer_field: str,
    max_new_tokens: int = 512,
    generation_preview_chars: int = 300,
):
    """
    Print examples where baseline and SFT differ on pass@1.
    """
    found = 0
    total = len(ds_eval)

    print("\n=== Examples Where Baseline and SFT Differ (pass@1) ===")

    for i in range(total):
        prompt = ds_eval[i]["prompt"]
        problem_text = str(ds_raw[i].get(problem_field, ""))
        gt = normalize_to_int_str(ds_raw[i].get(answer_field, None))

        base_pred, base_valid, base_text = solve_pass1(
            model=baseline_model,
            tokenizer=baseline_tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
        sft_pred, sft_valid, sft_text = solve_pass1(
            model=sft_model,
            tokenizer=sft_tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )

        base_correct = (base_pred is not None and gt is not None and base_pred == gt)
        sft_correct = (sft_pred is not None and gt is not None and sft_pred == gt)

        # Only print if they differ in prediction or correctness
        if (base_pred != sft_pred) or (base_correct != sft_correct):
            found += 1

            print(f"\n--- Differing Example {found} (dataset idx={i}) ---")
            print("Gold:", gt)
            print("Baseline pred:", base_pred, "| valid:", base_valid, "| correct:", base_correct)
            print("SFT pred:", sft_pred, "| valid:", sft_valid, "| correct:", sft_correct)

            problem_preview = problem_text.replace("\n", " ")
            print("Problem preview:", problem_preview[:220])

            print("\nBaseline text:")
            print(base_text[:generation_preview_chars])

            print("\nSFT text:")
            print(sft_text[:generation_preview_chars])

            base_final = extract_final_answer(base_text)
            base_fallback = extract_last_int_fallback(base_text)
            sft_final = extract_final_answer(sft_text)
            sft_fallback = extract_last_int_fallback(sft_text)

            print("\nExtraction debug:")
            print(
                "Baseline -> FINAL_ANSWER:",
                base_final,
                "| last_int_fallback:",
                base_fallback,
                "| extracted:",
                base_pred,
            )
            print(
                "SFT -> FINAL_ANSWER:",
                sft_final,
                "| last_int_fallback:",
                sft_fallback,
                "| extracted:",
                sft_pred,
            )

            if found >= n_examples:
                break

    if found == 0:
        print("No differing examples found.")
# ---------------------------
# Main runner
# ---------------------------

def run_evaluation(evaluate_config_path=None):
    """
    Runs the full evaluation pipeline and optionally writes a report.
    Input:
    - evaluate_config_path: optional path to evaluate.yaml
    Output:
    - results: nested dict of metadata and computed evaluation metrics
    """
    # Loads configurations from according yaml files, obtaining problems, model checkpoint, and reporting settings.
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

    # Loads the evaluation split and formats it into prompts with expected answers.
    ds_raw = load_eval_split(dataset_path, split_name)
    ds_eval = format_eval_split(
        ds_split=ds_raw,
        system_prompt=system_prompt,
        problem_field=problem_field,
        answer_field=answer_field,
    )
    # Loads the model and tokenizer for evaluation.
    tokenizer, model = load_eval_model(model_checkpoint)
    
    # Runs the specified evaluation metrics (pass@1 and/or maj@N) on the evaluation split, collecting results in a dictionary
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

    # Prints a few examples with their prompts, gold answers, generated answers, and majority vote details for qualitative inspection.
    preview_examples = int(reporting_yaml.get("preview_examples", 0))
    if preview_examples > 0:
        print_preview_examples(
            model=model,
            tokenizer=tokenizer,
            ds_raw=ds_raw,
            ds_eval=ds_eval,
            n_examples=preview_examples,
            problem_field=problem_field,
            answer_field=answer_field,
            pass1_max_new_tokens=int(eval_yaml.get("pass1_max_new_tokens", 512)),
            n_samples=int(eval_yaml.get("majn_n_samples", 8)),
            majn_max_new_tokens=int(eval_yaml.get("majn_max_new_tokens", 512)),
            majn_temperature=float(eval_yaml.get("majn_temperature", 0.7)),
            majn_top_p=float(eval_yaml.get("majn_top_p", 0.95)),
            generation_preview_chars=int(reporting_yaml.get("preview_generation_chars", 180)),
        )
    # Saves the evaluation results and metadata to a JSON file in the specified output directory in the yaml configuration
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
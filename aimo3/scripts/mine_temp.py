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
from . import helpers
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import AutoPeftModelForCausalLM
except ImportError:
    AutoPeftModelForCausalLM = None


# --------------------------------------------------------------------------
# Paths and config loading
# --------------------------------------------------------------------------
""" def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def get_mine_config_path() -> Path:
    return get_repo_root() / "configs" / "mine.yaml"


def get_data_config_path() -> Path:
    return get_repo_root() / "configs" / "data.yaml"


def load_yaml(config_path) -> dict:
    config_path = Path(config_path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
 """

def load_configs(mine_config_path=None):
    if mine_config_path is None:
        mine_config_path = get_config_path("mine.yaml")

    mine_config = load_yaml(mine_config_path)

    data_config_path = mine_config.get("paths", {}).get("data_config_path")
    if not data_config_path:
        data_config_path = get_config_path( "data.yaml")
    data_config = load_yaml(data_config_path)

    paths = mine_config.get("paths", {})
    generation = mine_config.get("generation", {})
    criteria = mine_config.get("failure_criteria", {})
    reporting = mine_config.get("reporting", {})

    model_checkpoint = paths.get("model_checkpoint")
    dataset_path = paths.get("dataset_path")
    dataset_split = paths.get("dataset_split", "test")
    output_path = paths.get("output_path")

    if not model_checkpoint:
        raise ValueError("paths.model_checkpoint missing in mine.yaml")
    if not dataset_path:
        raise ValueError("paths.dataset_path missing in mine.yaml")
    if not output_path:
        raise ValueError("paths.output_path missing in mine.yaml")

    return mine_config, data_config, model_checkpoint, dataset_path, dataset_split, output_path, generation, criteria, reporting


def set_seeds(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------
# Answer extraction
# --------------------------------------------------------------------------
FINAL_RE = re.compile(r"FINAL_ANSWER:\s*([0-9]{1,5})\b")
LAST_INT_RE = re.compile(r"\b(\d{1,5})\b")


def normalize_to_int_str(ans: Optional[str]) -> Optional[str]:
    if ans is None:
        return None

    s = str(ans).strip()
    s = s.replace(" ", "").replace("\n", "").replace("\t", "").replace(",", "")

    if not s:
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


# --------------------------------------------------------------------------
# Dataset loading and prompt formatting
# --------------------------------------------------------------------------
def load_mining_split(dataset_path: str, split_name: str) -> Dataset:
    print(f"Loading mining dataset from: {dataset_path}")
    ds = load_from_disk(dataset_path)

    if hasattr(ds, "keys"):
        if split_name not in ds:
            raise ValueError(f"Split '{split_name}' not found. Available: {list(ds.keys())}")
        split_ds = ds[split_name]
    else:
        split_ds = ds

    print(f"Mining rows: {len(split_ds)}")
    print(f"Columns: {split_ds.column_names}")
    return split_ds


def build_prompt(problem_text: str) -> str:
    problem_text = str(problem_text).strip()
    return "Problem:\n" + problem_text + "\n\nSolution:\n"


# --------------------------------------------------------------------------
# Model + generation
# --------------------------------------------------------------------------
def get_inference_dtype():
    if torch.cuda.is_available():
        return torch.bfloat16
    return torch.float32


def load_model_and_tokenizer(model_path: str, trust_remote_code: bool = True):
    print(f"Loading tokenizer from: {model_path}")
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

    model = None
    if AutoPeftModelForCausalLM is not None:
        try:
            print("Trying to load as PEFT adapter checkpoint")
            model = AutoPeftModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        except Exception:
            model = None

    if model is None:
        print("Loading as standard causal LM checkpoint")
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    return tok, model


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
    outputs_text = []
    device = next(model.parameters()).device

    for _ in range(n):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        generated = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        text = tokenizer.decode(generated[0], skip_special_tokens=True)
        outputs_text.append(text)

    return outputs_text


def majority_vote_answer(generations: List[str]) -> Tuple[Optional[str], Counter, List[str], float]:
    extracted = []
    for g in generations:
        ans = extract_answer(g)
        if ans is not None:
            extracted.append(ans)

    if not extracted:
        return None, Counter(), [], 0.0

    counts = Counter(extracted)
    best, best_count = counts.most_common(1)[0]
    agreement = best_count / len(extracted)
    return best, counts, extracted, agreement


# --------------------------------------------------------------------------
# Mining decision logic
# --------------------------------------------------------------------------
def get_mining_reasons(
    majority_answer: Optional[str],
    gold_answer: Optional[str],
    agreement: float,
    format_validity_rate: float,
    has_any_valid: bool,
    criteria: dict,
) -> List[str]:
    reasons = []

    majority_is_correct = (majority_answer is not None and gold_answer is not None and majority_answer == gold_answer)

    if criteria.get("mine_if_majority_wrong", True):
        if not majority_is_correct:
            reasons.append("majority_wrong")

    if criteria.get("mine_if_low_agreement", True):
        thr = float(criteria.get("low_agreement_threshold", 0.5))
        if agreement < thr:
            reasons.append("low_agreement")

    if criteria.get("mine_if_low_format_validity", True):
        thr = float(criteria.get("low_format_validity_threshold", 0.5))
        if format_validity_rate < thr:
            reasons.append("low_format_validity")

    if criteria.get("mine_if_no_valid_extraction", True):
        if not has_any_valid:
            reasons.append("no_valid_extraction")

    return reasons


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def mine_failures(mine_config_path=None):
    (
        mine_config,
        _data_config,
        model_checkpoint,
        dataset_path,
        dataset_split,
        output_path,
        generation_cfg,
        criteria_cfg,
        reporting_cfg,
    ) = load_configs(mine_config_path)

    seed = int(mine_config.get("run", {}).get("seed", 42))
    set_seeds(seed)

    k = int(generation_cfg.get("k_solutions", 8))
    max_new_tokens = int(generation_cfg.get("max_new_tokens", 512))
    temperature = float(generation_cfg.get("temperature", 0.7))
    top_p = float(generation_cfg.get("top_p", 0.95))

    ds = load_mining_split(dataset_path, dataset_split)
    tokenizer, model = load_model_and_tokenizer(model_checkpoint)

    mined_rows = []
    stats = {
        "total_examples": len(ds),
        "mined_examples": 0,
        "skipped_invalid_gold": 0,
        "reasons_count": Counter(),
    }

    for i, row in enumerate(ds):
        problem = row.get("problem", "")
        gold_raw = row.get("expected_answer", None)
        gold = normalize_to_int_str(gold_raw)

        if gold is None:
            stats["skipped_invalid_gold"] += 1
            continue

        prompt = build_prompt(problem)
        generations = generate_n_solutions(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            n=k,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        majority, counts, extracted, agreement = majority_vote_answer(generations)
        has_any_valid = len(extracted) > 0
        format_validity_rate = len(extracted) / float(k) if k > 0 else 0.0

        reasons = get_mining_reasons(
            majority_answer=majority,
            gold_answer=gold,
            agreement=agreement,
            format_validity_rate=format_validity_rate,
            has_any_valid=has_any_valid,
            criteria=criteria_cfg,
        )

        if reasons:
            for r in reasons:
                stats["reasons_count"][r] += 1

            mined_rows.append(
                {
                    "prompt": prompt,
                    "expected_answer": gold,
                    "problem": str(problem),
                    "majority_answer": majority,
                    "agreement": float(agreement),
                    "format_validity_rate": float(format_validity_rate),
                    "n_valid_extractions": int(len(extracted)),
                    "vote_counts": dict(counts),
                    "mine_reasons": reasons,
                }
            )

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(ds)} rows | mined so far: {len(mined_rows)}")

    stats["mined_examples"] = len(mined_rows)

    os.makedirs(output_path, exist_ok=True)
    if mined_rows:
        mined_ds = Dataset.from_list(mined_rows)
    else:
        mined_ds = Dataset.from_dict(
            {
                "prompt": [],
                "expected_answer": [],
                "problem": [],
                "majority_answer": [],
                "agreement": [],
                "format_validity_rate": [],
                "n_valid_extractions": [],
                "vote_counts": [],
                "mine_reasons": [],
            }
        )

    mined_ds.save_to_disk(output_path)
    print(f"Saved mined dataset to: {output_path}")
    print(f"Mined examples: {len(mined_rows)} / {len(ds)}")

    if reporting_cfg.get("save_mining_report", True):
        report_path = reporting_cfg.get("report_path", "runs/mining_report.json")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)

        report = {
            "config_path": str(mine_config_path if mine_config_path else get_mine_config_path()),
            "dataset_path": dataset_path,
            "dataset_split": dataset_split,
            "model_checkpoint": model_checkpoint,
            "output_path": output_path,
            "generation": {
                "k_solutions": k,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
            },
            "failure_criteria": criteria_cfg,
            "stats": {
                "total_examples": stats["total_examples"],
                "mined_examples": stats["mined_examples"],
                "mined_rate": (stats["mined_examples"] / stats["total_examples"]) if stats["total_examples"] else 0.0,
                "skipped_invalid_gold": stats["skipped_invalid_gold"],
                "reasons_count": dict(stats["reasons_count"]),
            },
        }

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"Saved mining report to: {report_path}")
# Input: model checkpoint + evaluation split
# - Does:
#       - pass@1 with greedy generation
#       - maj@N with sampling + majority vote
#       - reports format validity and agreement stats
# Output:
# - printed metrics + optional runs/report.json


import re
from collections import Counter
from typing import Optional, List, Tuple, Dict

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


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


# ---------------------------
# Model loading
# ---------------------------

def load_eval_model(
    model_path: str,
    trust_remote_code: bool = True,
):
    """
    Loads tokenizer + causal LM from a saved model directory or HF model id.
    """
    tok = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        trust_remote_code=trust_remote_code,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        trust_remote_code=trust_remote_code,
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

def eval_split_pass1(
    model,
    tokenizer,
    ds_split: Dataset,
    max_items: int = 100,
    max_new_tokens: int = 512,
) -> Dict[str, float]:
    """
    Evaluates a dataset split with greedy decoding (pass@1).
    """
    correct = 0
    valid = 0
    total = min(max_items, len(ds_split))

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
    max_items: int = 50,
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

    total = min(max_items, len(ds_split))

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

        best, counts, extracted_norm, agreement = majority_vote_answer(generations)

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
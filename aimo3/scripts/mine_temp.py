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
def load_configs(mine_config_path=None):
    """ 
    This function loads mining "paths" config.
    Input: Optional path to the mining config file. If None, it will look 
           for "mine.yaml" in the default config directory.
    Output: Tuple containing:
        - mine_config: The loaded mining configuration dictionary.
        - data_config: The loaded data configuration dictionary.
        - model_checkpoint: Path to the model checkpoint specified in the config.
        - dataset_path: Path to the dataset specified in the config.
        - dataset_split: The dataset split to use (e.g., "train", "test
        - output_path: Path where the mined dataset will be saved.
        - generation: The generation configuration section from the mining config.
        - criteria: The failure criteria configuration section from the mining config.
        - reporting: The reporting configuration section from the mining config.
    """
    if mine_config_path is None:
        mine_config_path = helpers.get_config_path("mine.yaml")
    mine_config = helpers.load_yaml(mine_config_path)

    # Data config path
    data_config_path = mine_config.get("paths", {}).get("data_config_path")
    if not data_config_path:
        data_config_path = helpers.get_config_path("data.yaml")
    data_config = helpers.load_yaml(data_config_path)

    # Headings of yaml sections
    paths = mine_config.get("paths", {})
    generation = mine_config.get("generation", {})
    criteria = mine_config.get("failure_criteria", {})
    reporting = mine_config.get("reporting", {})

    # PATHS SECTION
    model_checkpoint = paths.get("model_checkpoint")
    dataset_path = paths.get("dataset_path")
    dataset_split = paths.get("dataset_split", "test")
    output_path = paths.get("output_path")

    return mine_config, data_config, model_checkpoint, dataset_path, dataset_split, output_path, generation, criteria, reporting

# --------------------------------------------------------------------------
# Answer extraction
# --------------------------------------------------------------------------
# The regexes look for patterns like "FINAL_ANSWER: 42" to extract the final answer.
FINAL_RE = re.compile(r"FINAL_ANSWER:\s*([0-9]{1,5})\b")
LAST_INT_RE = re.compile(r"\b(\d{1,5})\b")


def normalize_to_int_str(ans: Optional[str]) -> Optional[str]:
    """
    This function takes a raw input string which can have an answer and 
    normalizes it into a clean string that represents an integer between
    0 and 99999. Removes everything except digits and checks for validity.
    Input: 
        - ans(str or None) - The raw answer string to normalize.
    Output: 
        - A string representing the integer answer if valid, or None if invalid.
    """
    # If the input is None, return None immediately
    if ans is None:
        return None
    # Take string and strip whitespace
    s = str(ans).strip()
    # Remove spaces, newlines, tabs, and commas
    s = s.replace(" ", "").replace("\n", "").replace("\t", "").replace(",", "")
    # If the string is empty after cleaning, return None
    if not s:
        return None
    # If the string starts with a plus sign, remove it
    if s.startswith("+"):
        s = s[1:]
    # If the string starts with a minus sign, it's invalid (negative number), return None
    if s.startswith("-"):
        return None
    # After cleaning, if the string is not purely digits, return None
    if not s.isdigit():
        return None
    # Convert to int and back to string to remove leading zeros
    s = str(int(s))
    # Assign int s to v for range checking
    v = int(s)
    # Check if the integer value is between 0 and 99999, inclusive. If not, return None.
    if not (0 <= v <= 99999):
        return None
    # Return the cleaned and validated string representing the integer answer
    return s

def extract_final_answer(text: str) -> Optional[str]:
    """
    This function uses the regex FINAL_RE to search for patterns in the input text that match "FINAL_ANSWER: <number>".
    It returns the last occurrence of such a pattern if found, or None if no valid pattern is found.
    Input:
    - text (str): The input text from which to extract the final answer.
    Output:
    - A string representing the extracted final answer if found, or None if not found.
    """
    # Use the FINAL_RE regex to find all matches in the input text
    matches = FINAL_RE.findall(str(text))
    # If there are matches
    if matches:
        # Strip whitespace from the matched answer and return it
        return matches[-1].strip()
    # Otherwise, if no matches are found, return None
    return None


def extract_last_int_fallback(text: str) -> Optional[str]:
    """
    This function is a fallback method to extract the last integer from the input text using 
    the LAST_INT_RE regex. It searches for all occurrences of integers in the text and returns the last one found,
    or None if no integers are found. This ensures that even if the specific "FINAL_ANSWER" pattern is not present,
    we can still attempt to extract a potential answer from the text.
    Input:
        - text (str): The input text from which to extract the last integer.
    Output:
        - A string representing the last integer found in the text if any are found, or None if no integers are found.  

    """
    # Use the LAST_INT_RE regex to find all matches of integers in the input text
    matches = LAST_INT_RE.findall(str(text))
    # If there are no matches, return None
    if not matches:
        return None
    # If there are matches, return the last one after stripping whitespace
    return matches[-1].strip()


def extract_answer(text: str) -> Optional[str]:
    """
    This function attempts to extract a final answer from the input text using a two-step approach:
    1. It first tries to extract the answer using the extract_final_answer function, which
    looks for a specific pattern "FINAL_ANSWER: <number>".
    2. If the first method does not yield a result (i.e., returns None), it falls back to the extract_last_int_fallback function,
    which looks for the last integer in the text regardless of any specific pattern.
    Finally, it normalizes the extracted raw answer using the normalize_to_int_str function to ensure it is a valid integer string.
    Input:
       - text (str): The input text from which to extract the answer.
    Output:
        - A string representing the extracted and normalized answer if found and valid, or None if no valid answer is found.
    """
    # First, attempt to extract the final answer using the specific pattern
    raw = extract_final_answer(text)
    # If raw is None, meaning the specific pattern was not found, use the fallback method to extract the last integer
    if raw is None:
        raw = extract_last_int_fallback(text)
    # Return normalized version of the extracted raw answer, which will be a valid integer string or None if invalid
    return normalize_to_int_str(raw)

# --------------------------------------------------------------------------
# Dataset loading and prompt formatting
# --------------------------------------------------------------------------
def load_mining_split(dataset_path: str, split_name: str) -> Dataset:
    """ 
    This function loads a dataset from the specified path and retreives the specified split.
    , it then prints out the number of rows and columns in the loaded split before returning it.
    Input:
    - dataset_path (str): The file path to the dataset to be loaded.
    - split_name (str): The name of the dataset split to retrieve (e.g., "train", "test", "validation").
    Output:
    - A Hugging Face Dataset object corresponding to the specified split of the loaded dataset.
    """
    print(f"Loading mining dataset from: {dataset_path}")
    # Load the dataset from disk using Hugging Face's load_from_disk function
    ds = load_from_disk(dataset_path)
    # Check if loaded dataset has a split with the specified name, if so, 
    # retrieve that split; otherwise, use the whole dataset
    if hasattr(ds, "keys"):
        split_ds = ds[split_name]
    else:
        split_ds = ds
    # Prints for help
    print(f"Mining rows: {len(split_ds)}")
    print(f"Columns: {split_ds.column_names}")
    # Return the loaded split of the dataset
    return split_ds


def build_prompt(problem_text: str) -> str:
    """
    This function builds a prompt for the language model by taking the problem text,
    stripping any leading or trailing whitespace, and then formatting it into a string that clearly delineates the "Problem" and "Solution" sections.
    The resulting prompt will have the problem text under a "Problem:" heading, followed by a "Solution:" heading where the model is expected to generate its answer.
    Input:
        - problem_text (str): The raw text of the problem to be included in the prompt.
    Output:
        - A formatted string that includes the problem text under a "Problem:" section
        and a "Solution:" section for the model's response. The problem text is cleaned
        of leading and trailing whitespace to ensure a neat format for the prompt.
    """
    # Strip leading and trailing whitespace from the problem text to ensure a clean format
    problem_text = str(problem_text).strip()
    # Return the formatted prompt string with the problem text under a "Problem:" heading and a "Solution:" heading for the model's response
    return "Problem:\n" + problem_text + "\n\nSolution:\n"


# --------------------------------------------------------------------------
# Model + generation
# --------------------------------------------------------------------------
def get_inference_dtype():
    """
    This function determines the appropriate data type to use for model inference based on
    the availability of CUDA (GPU) on the system. If CUDA is available, it returns torch.bfloat16,
    which is a lower-precision data type that can speed up inference while still maintaining good
    accuracy on modern GPUs. If CUDA is not available, it returns torch.float32, which is the standard data 
    type for CPU inference to ensure compatibility and stability.
    """
    if torch.cuda.is_available():
        return torch.bfloat16
    return torch.float32

def load_model_and_tokenizer(model_path: str, trust_remote_code: bool = True):
    """
    This function loads a language model and its corresponding tokenizer from a specified model checkpoint path.
    It first loads the tokenizer using Hugging Face's AutoTokenizer, ensuring that it has a
    pad token defined (using the EOS token if necessary) and sets the padding side to "right".
    Then, it attempts to load the model as a PEFT adapter checkpoint using AutoPeftModel
    if available, and if that fails, it falls back to loading it as a standard causal language model using AutoModelForCausalLM.
    The model is loaded with a device map set to "auto" to utilize available hardware, and the data type for inference is 
    determined by the get_inference_dtype function. The trust_remote_code flag is passed to both the tokenizer and model 
    loading functions to allow for custom code execution if the checkpoint requires it.
    Input:
        - model_path (str): The file path to the model checkpoint to be loaded.
        - trust_remote_code (bool): A flag indicating whether to allow execution of custom code from
        the model checkpoint. This is necessary for some models that include custom architectures or tokenizers.
    Output:
        - A tuple containing the loaded tokenizer and model objects. The tokenizer is an instance of AutoTokenizer,
        and the model is an instance of either AutoPeftModelForCausalLM or AutoModelForCausalLM, depending on the 
        checkpoint type. Both are configured for inference with appropriate data types and device mapping.
    """
    print(f"Loading tokenizer from: {model_path}")
    # Load the tokenizer from the specified model path, allowing for custom code if necessary
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=trust_remote_code)
    # Ensure the tokenizer has a pad token defined; if not, set it to the EOS token. This is important for generation tasks that require padding.
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Ensure padding is done on right side, which is typically required for causal language models where the model generates text after the input prompt
    tok.padding_side = "right"
    # Define the model loading parameters including device mapping for automatic hardware utilization, the data type for inference, and trust_remote_code for custom model code execution
    model_kwargs = {"device_map": "auto","torch_dtype": get_inference_dtype(), "trust_remote_code": trust_remote_code}
    # Set model to none
    model = None
    # If AutoPeftModelForCausalLM is available, attempt to load the model as a PEFT adapter checkpoint.
    # If this fails (e.g., if the checkpoint is not a PEFT adapter), 
    # catch the exception and set model to None to allow for fallback loading.
    if AutoPeftModelForCausalLM is not None:
        try:
            print("Trying to load as PEFT adapter checkpoint")
            model = AutoPeftModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        except Exception:
            model = None
    # If the model was not successfully loaded as a PEFT adapter, attempt to load it as a standard causal language model.
    if model is None:
        print("Loading as standard causal LM checkpoint")
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    # Return the loaded tokenizer and model as a tuple
    return tok, model

# Torch inference mode is used to disable gradient calculations, which reduces memory usage and speeds up inference since we are not training the model.
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
    Generate n sampled solutions for a prompt using the given model and tokenizer.
    Input:
        - model: Language model used for generation.
        - tokenizer: Tokenizer used to encode/decode text.
        - prompt (str): Prompt to generate from.
        - n (int): Number of solutions to sample.
        - max_new_tokens (int): Max tokens to generate per sample.
        - temperature (float): Sampling temperature.
        - top_p (float): Nucleus sampling threshold.
    Output:
        - List[str]: n generated solution strings.   
    """
    # List to store the generated solution texts
    outputs_text = []
    # Determine the device of the model parameters to ensure that input tensors are moved to the same device for generation
    device = next(model.parameters()).device
    # Loop n times to generate n different solutions for the given prompt
    for _ in range(n):
        # Tokenize the input prompt and move the input IDs to the same device as the model for generation
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        # Generate text using the model with the specified generation parameters.
        generated = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        # Decode the generated token IDs back into text, skipping special tokens, and append the resulting string to the outputs_text list
        text = tokenizer.decode(generated[0], skip_special_tokens=True)
        # Append the decoded text to the outputs_text list, which will contain all n generated solutions after the loop completes
        outputs_text.append(text)
    # After generating n solutions, return the list of generated solution texts
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
    helpers.set_seeds(seed)

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
            "config_path": str(mine_config_path if mine_config_path else helpers.get_config_path()),
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
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
from tqdm import tqdm
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
    max_items = paths.get("max_items")

    return mine_config, data_config, model_checkpoint, dataset_path, dataset_split, output_path, max_items, generation, criteria, reporting

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
    """
    This function takes a list of generated solution texts, extracts the final answer from each using the extract_answer function,
    and then performs a majority vote to determine the most common answer among the valid extractions.
    Input:
    - generations (List[str]): A list of generated solution texts from which to extract answers.
    Output:
    - A tuple containing:
        - majority_answer (Optional[str]): The most common valid answer extracted from the generations, or None if no valid answers are found.
        - counts (Counter): A Counter object that counts the occurrences of each valid extracted answer.
        - extracted (List[str]): A list of all valid extracted answers from the generations.
        - agreement (float): The agreement rate, calculated as the count of 
    """
    # List to store the valid extracted answers from the generations
    extracted = []
    # For each generated solution text, attempt to extract a valid answer using the extract_answer function, append the valid answers to the extracted list
    for g in generations:
        ans = extract_answer(g)
        if ans is not None:
            extracted.append(ans)
    # If no valid answers were extracted, return None for the majority answer, an empty Counter, an empty list of extracted answers, and 0.0 for agreement
    if not extracted:
        return None, Counter(), [], 0.0
    # Use counter to count occurrences of each valid extracted answer, agreement, etc/
    counts = Counter(extracted)
    best, best_count = counts.most_common(1)[0]
    agreement = best_count / len(extracted)
    # Return the most common valid answer (majority), the counts of all valid answers,
    # the list of valid extracted answers, and the agreement rate
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
    """
    This function determines the reasons for mining a particular example based on the provided criteria.
    It checks if the majority answer is wrong compared to the gold answer, if the agreement among valid 
    extractions is below a specified threshold, if the format validity rate of the extractions is below a 
    specified threshold, and if there are no valid extractions at all. For each criterion that is met,
    it appends a corresponding reason string to the reasons list, which is then returned at the end. This 
    allows for a detailed understanding of why an example was selected for mining based on multiple factors.
    Input:
    - majority_answer (Optional[str]): The most common valid answer extracted from the generations, or
    None if no valid answers are found.
    - gold_answer (Optional[str]): The correct answer from the dataset for comparison.
    - agreement (float): The agreement rate among the valid extracted answers.
    - format_validity_rate (float): The rate of valid format extractions among the generations
    - has_any_valid (bool): A flag indicating whether there were any valid extractions at all.
    - criteria (dict): A dictionary containing the mining criteria thresholds and flags for which criteria to
    apply.
    Output:
    - List[str]: A list of reason strings indicating which mining criteria were met for this example. 
    Possible reasons include "majority_wrong", "low_agreement", "low_format_validity",
    """
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
        max_items,
        generation_cfg,
        criteria_cfg,
        reporting_cfg,
    ) = load_configs(mine_config_path)
    """
    This function orchestrates the mining process by loading the necessary configurations, 
    setting seeds for reproducibility, loading the dataset and model, and then iterating
    through each example in the dataset to generate solutions, extract answers, perform majority voting, 
    and determine mining reasons based on specified criteria. It collects statistics throughout the
    process and saves the mined dataset and a mining report at the end. The function is designed to be 
    flexible and configurable through external YAML files, allowing for easy adjustments to the mining 
    logic and generation parameters without modifying the code.
    Input: 
    - mine_config_path (str, optional): The file path to the mining configuration YAML file. If None, it will look for "mine.yaml" in the default config directory.
    Output:
    - This function does not return a value but performs the mining process and saves the mined dataset to disk, along with a mining report if configured to do so.
    The mined dataset will contain examples
    """
    # Set seed for reproducibility, based on on configs
    seed = int(mine_config.get("run", {}).get("seed", 42))
    # Set Seed
    helpers.set_seeds(seed)
    # Obtain configurations
    k = int(generation_cfg.get("k_solutions", 8))
    max_new_tokens = int(generation_cfg.get("max_new_tokens", 512))
    temperature = float(generation_cfg.get("temperature", 0.7))
    top_p = float(generation_cfg.get("top_p", 0.95))
    # Load the ds split and model + tokenizer
    ds = load_mining_split(dataset_path, dataset_split)
    if max_items is not None:
        capped_total = min(int(max_items), len(ds))
        ds = ds.select(range(capped_total))
        print(f"Applied max_items cap: {capped_total}")
    tokenizer, model = load_model_and_tokenizer(model_checkpoint)
    # List to store the mined rows that meet the mining criteria
    mined_rows = []
    # Dictionary for stats of mining process
    stats = {
        "total_examples": len(ds),
        "mined_examples": 0,
        "skipped_invalid_gold": 0,
        "reasons_count": Counter(),
    }
    # Iterate through rows in ds split with progress bar
    for i, row in enumerate(tqdm(ds, desc="Mining examples", unit="example")):
        # extract problem and expected answer(gold)
        problem = row.get("problem", "")
        gold_raw = row.get("expected_answer", None)
        # Normalize gold answer to ensure it's a valid integer string or None
        gold = normalize_to_int_str(gold_raw)
        # If the gold answer is invalid, skip this example and update stats
        if gold is None:
            stats["skipped_invalid_gold"] += 1
            continue
        # Build the prompt for the model using the problem text
        prompt = build_prompt(problem)
        # Generate n solutions from the model for the given prompt using the specified generation parameters
        generations = generate_n_solutions(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            n=k,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        # Perform maj vot, counts, extract valid answers, agreement, etc.
        majority, counts, extracted, agreement = majority_vote_answer(generations)
        # If the extracted list is not empty, set has_any_valid to True; otherwise, set it to False.
        has_any_valid = len(extracted) > 0
        # Calculate the format validity rate as the number of valid extractions divided by
        # k (the total number of generations), ensuring that we handle the case where k is zero to avoid division by zero errors
        format_validity_rate = len(extracted) / float(k) if k > 0 else 0.0
        # Determine the reasoning for mining/
        reasons = get_mining_reasons(
            majority_answer=majority,
            gold_answer=gold,
            agreement=agreement,
            format_validity_rate=format_validity_rate,
            has_any_valid=has_any_valid,
            criteria=criteria_cfg,
        )
        # If there are reasons for mining this example, append the relevant 
        # information to the mined_rows list and update the reasons count in stats
        if reasons:
            for r in reasons:
                stats["reasons_count"][r] += 1
            # Append a dictionary containing all below information about the example to the mined_rows list:
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
    # Print final stats about mining process.
    stats["mined_examples"] = len(mined_rows)
    # Ensure output exists.
    os.makedirs(output_path, exist_ok=True)
    # If there are mined rows, create a Hugging Face Dataset from the list of mined rows; 
    # otherwise, create an empty Dataset with the appropriate columns. 
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
    # Save mined dataset to disk at path
    mined_ds.save_to_disk(output_path)
    print(f"Saved mined dataset to: {output_path}")
    print(f"Mined examples: {len(mined_rows)} / {len(ds)}")
    # If configured to save a mining report, create a report dictionary containing all relevant 
    # information about the mining process, including configuration paths, dataset information,
    #  generation parameters, failure criteria, and statistics. Then save this report as a JSON file to the specified report path.
    if reporting_cfg.get("save_mining_report", True):
        report_path = reporting_cfg.get("report_path", "runs/mining_report.json")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        report = {
            "config_path": str(mine_config_path if mine_config_path else helpers.get_config_path()),
            "dataset_path": dataset_path,
            "dataset_split": dataset_split,
            "max_items": int(max_items) if max_items is not None else None,
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
        # Save mining report as JSON to path specified in reporting config, creating directories if necessary
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"Saved mining report to: {report_path}")
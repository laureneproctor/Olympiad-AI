## Olympiad-AI
Building a system that parses LaTeX- formatted math problems and uses a model to solve math olympiad-level questions.
## Time-line

<img width="772" height="262" alt="Screenshot 2026-03-03 at 9 29 52 PM" src="https://github.com/user-attachments/assets/b9e98a81-0739-40a8-9b81-03742ff306c5" />

## Current Routine for Running

- `prepare.py` once
- `sft.py` → get baseline SFT model
- `eval.py` → measure pass@1 and maj@N
- `mine.py` → build RL dataset from failures
- `grpo.py` → train RL model
- `eval.py` again → verify improvement
- `solve.py` → produce final predictions

## Project Structure

```
Olympiad-AI/
├── aimo3/
│   ├── configs/
│   │   ├── data.yaml
│   │   ├── evaluate.yaml
│   │   ├── grpo.yaml
│   │   ├── mine.yaml
│   │   ├── sft.yaml
│   │   └── solve.yaml
│   │
│   ├── scripts/
│   │   ├── __init__.py
│   │   ├── evaluate.py
│   │   ├── grpo.py
│   │   ├── mine.py
│   │   ├── prepare.py
│   │   ├── sft.py
│   │   └── solve.py
│   │
│   └── __init__.py
│
├── data/
├── notebooks/
│   ├── Baseline 1.ipynb
│   ├── DSM_V2_for_Presentation.ipynb
│   ├── Final SVSU Parser.ipynb
│   ├── Olympiad-PDF-Scrape.ipynb
│   ├── README.md
│   ├── SVSU Parsing All.ipynb
│   └── ec-pdf-parser.ipynb
│
├── pdf_data/
│   └── raw_pdfs/
│
├── .gitignore
└── README.md
```
Olympiad-AI 🧠➗

Olympiad-AI is an experimental pipeline for training AI systems to solve mathematical olympiad problems written in LaTeX.

The project focuses on building reasoning models capable of solving problems from:

Algebra

Combinatorics

Geometry

Number Theory

The system is designed around an iterative improvement loop:

data preparation → supervised training → evaluation → failure mining → reinforcement learning → inference

The repository currently focuses on offline experimentation, where models are trained and improved before deployment to competition environments such as the Kaggle AI Mathematical Olympiad competitions.

System Overview 🔬

The Olympiad-AI workflow follows an iterative pipeline designed to gradually improve mathematical reasoning ability.

Raw Math Problems
        │
        ▼
prepare.py
        │
        ▼
Supervised Fine-Tuning (sft.py)
        │
        ▼
Evaluation (eval.py)
        │
        ▼
Failure Mining (mine.py)
        │
        ▼
Reinforcement Training (grpo.py)
        │
        ▼
Evaluation Again
        │
        ▼
Inference Solver (solve.py)

Each stage produces artifacts used by the next stage.

Pipeline Stages ⚙️
1. Data Preparation — prepare.py
Purpose

Convert raw mathematical problem sources into a clean, structured dataset suitable for training and evaluation.

What it does

This stage processes raw inputs such as:

scraped olympiad problems

PDF datasets

LaTeX problem statements

reference problem sets

previously solved examples

The script standardizes and cleans the data so the model receives consistent inputs and outputs.

Typical tasks

Normalize LaTeX formatting

Remove parsing artifacts

Extract problems and answers

Format training prompts

Generate prompt → solution pairs

Create train / validation splits

Cache tokenized datasets

Inputs
pdf_data/
raw datasets
reference problems
external corpora
Outputs
data/
train.jsonl
val.jsonl
eval.jsonl
Why it matters

High-quality training data is critical. Poor formatting or noisy parsing will significantly reduce model performance.

Summary

prepare.py acts as the project’s data factory, transforming messy raw math content into structured examples that models can learn from.

2. Supervised Fine-Tuning — sft.py
Purpose

Train the baseline solver model using supervised learning.

What it does

The model is trained to map:

Math Problem (LaTeX)
        ↓
Reasoning + Final Integer Answer

or a shorter format:

Problem → Final Answer

This step teaches the model the general structure of mathematical reasoning and solution formatting.

Inputs
prepared training data
base model checkpoint
training configuration
Outputs
baseline model checkpoint
training logs
metrics
Key training parameters

Typical configurable settings include:

learning rate

batch size

sequence length

LoRA or full fine-tuning

number of epochs

Why it matters

This stage builds the first usable version of the solver.

Without a strong SFT baseline, later reinforcement learning steps become unstable.

Summary

sft.py trains the initial math reasoning model by teaching it to imitate correct solutions.

3. Evaluation — eval.py
Purpose

Measure model performance and identify weaknesses.

What it does

Runs the current model on a held-out dataset and computes evaluation metrics.

Typical metrics

Exact answer accuracy

pass@1

majority vote accuracy

invalid output rate

answer extraction success

runtime statistics

token usage

Inputs
trained model checkpoint
evaluation dataset
inference configuration
Outputs
accuracy reports
prediction logs
failure cases
analysis artifacts
Why it matters

Evaluation answers key questions:

Did training actually improve performance?

Which problem types are hardest?

Are outputs formatted correctly?

Are failures due to reasoning or formatting errors?

Summary

eval.py acts as the system’s report card, measuring how well the solver performs and identifying failure patterns.

4. Failure Mining — mine.py
Purpose

Convert model mistakes into new training signals.

What it does

After evaluation, the system analyzes incorrect predictions and extracts useful information from them.

Examples include:

incorrect reasoning traces

partial solutions

competing solution attempts

formatting errors

ambiguous outputs

These failures are converted into training data for the next stage.

Mining strategies

Possible approaches include:

ranking better vs worse solutions

extracting useful intermediate reasoning

identifying near-correct outputs

clustering similar failure types

Inputs
evaluation outputs
failed predictions
correct answers
Outputs
RL training dataset
preference pairs
hard-problem subsets
Why it matters

This step turns mistakes into valuable training signals, enabling iterative improvement.

Summary

mine.py allows the system to learn from its own mistakes by transforming failures into new training data.

5. Reinforcement Training — grpo.py
Purpose

Improve the baseline model using reinforcement-style optimization.

What it does

This stage trains the model using signals derived from the mined dataset.

Rather than simply imitating examples, the model is encouraged to produce outputs that:

reach the correct final answer

maintain coherent reasoning

avoid malformed responses

produce valid integer outputs

Inputs
baseline SFT checkpoint
mined RL dataset
reward configuration
Outputs
reinforcement-trained model
training metrics
reward statistics
Why it matters

Supervised learning teaches how solutions look.

Reinforcement learning pushes the model toward actually solving problems correctly.

Summary

grpo.py refines the solver by rewarding behaviors that lead to correct answers and discouraging poor reasoning patterns.

6. Re-Evaluation — eval.py (again)
Purpose

Verify that reinforcement training improved the model.

What it measures

accuracy improvements

reasoning quality

output stability

runtime performance

This comparison is typically performed between:

baseline SFT model
vs
RL-improved model
Why it matters

Not all training changes lead to improvement. This stage confirms whether modifications actually help.

7. Inference Solver — solve.py
Purpose

Use the trained model to solve new math problems.

What it does

solve.py implements the final inference strategy used to produce answers.

Typical steps include:

Format the problem prompt

Generate multiple candidate solutions

Extract integer answers

Filter invalid outputs

Apply voting or ranking

Return the final prediction

Inputs
trained model
new math problem
inference configuration
Outputs
final integer answer
(optional reasoning trace)
Why it matters

This script represents the actual solver engine used during evaluation or competitions.

Project Structure 📁
Olympiad-AI/
├── aimo3/
│   ├── configs/
│   │   ├── data.yaml
│   │   ├── evaluate.yaml
│   │   ├── grpo.yaml
│   │   ├── mine.yaml
│   │   ├── sft.yaml
│   │   └── solve.yaml
│   │
│   ├── scripts/
│   │   ├── evaluate.py
│   │   ├── grpo.py
│   │   ├── mine.py
│   │   ├── prepare.py
│   │   ├── sft.py
│   │   └── solve.py
│   │
│   └── __init__.py
│
├── data/
├── notebooks/
├── pdf_data/
│   └── raw_pdfs/
└── README.md
Development Workflow 🛠️

Typical offline experimentation workflow:

python prepare.py
python sft.py
python eval.py
python mine.py
python grpo.py
python eval.py
python solve.py

Each cycle improves the solver.

Design Philosophy 💡

The pipeline is built around three principles:

Iterative improvement

The system improves by repeatedly evaluating failures and retraining.

Separation of concerns
Script	Responsibility
prepare	dataset construction
sft	baseline training
eval	performance measurement
mine	failure analysis
grpo	reinforcement improvement
solve	inference
Reproducibility

Configurations are stored in configs/ to ensure experiments can be reproduced.

Future Improvements 🔮

Potential extensions include:

symbolic verification tools

automated reasoning checks

topic-specific prompting

ensemble solvers

theorem-proving integrations

curriculum learning for problem difficulty

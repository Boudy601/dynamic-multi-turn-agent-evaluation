# Dataset Generation

This repository generates synthetic golden evaluation data for a later
multi-turn agent evaluation pipeline. It does not run the agent evaluation
itself.

## Layout

- `src/dataset_generation/`: prompt preparation and LLM-backed dataset generation
- `config/`: judge definitions and guideline aliases
- `prompts/`: prompt templates used during generation
- `data/source/`: source AB-CD dataset files
- `data/generated/`: generated JSONL datasets and failure records
- `tools/`: dataset inspection utilities
- `reports/`: generated charts and other reports

## Setup

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

For LLM-backed generation, put the provider API key in `.env`, for example
`GEMINI_API_KEY=...`.

## Usage

Prepare injection prompts:

```powershell
python src\dataset_generation\prepare_prompts.py
```

Generate and validate the golden dataset:

```powershell
python src\dataset_generation\generate_golden_dataset.py
```

The scripts accept CLI options for alternate input, output, model, provider,
timeout, and concurrency settings. Generated files are written to
`data/generated/` by default.

# Dataset Generation

This repository generates synthetic golden evaluation data for a later
multi-turn agent evaluation pipeline. It does not run the agent evaluation
itself.

## Layout

- `src/dataset_generation/`: prompt preparation, chat assignment, and LLM-backed dataset generation
- `src/automation/`: Botasaurus-powered Gemini web review & repair automation
- `config/`: judge definitions, guideline aliases, reviewers, and per-reviewer SOP ownership (`reviewer_sops.json`)
- `prompts/`: prompt templates used during generation
- `data/source/`: source AB-CD dataset files
- `data/generated/`: generated JSONL datasets and failure records
- `assignments/`: per-reviewer assigned chats, `.originals/` backups, and `reviews.json`
- `tools/`: dataset inspection utilities and manual review UI (`chat_review.html`)
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

### Manual Chat Review Server

Start the local review UI:

```powershell
python server.py
```

Features:
- **Per-reviewer SOP pools**: Each reviewer in `config/reviewers.json` owns a dedicated 7–8 SOP subset in `config/reviewer_sops.json`.
- **Inline & model-assisted editing**: Edit/add/delete individual turns inline (with searchable SOP + referenced-subflow action pickers and auto-filled parameters), copy the evaluation + repair prompt in one click (`Copy Prompt`), or paste a model's corrected JSON via `Replace from Model`.
- **Undo & cleanup**: Edited chats keep pre-edit backups in `assignments/<reviewer>/.originals/` (`Revert to Original`), and chats can be removed via `Remove Chat`.
- **Tracking & ordering**: Reviews track `reviewed_by` (`ai`, `ai_human`, `human`), `edited`, `review_reason`, and `reviewed_at`, sorted with the most recently reviewed chats first. Use **Confirm AI Review (AI + Human ✓)** to sign off on AI reviews in one click.

### Automated Gemini Web Review (Botasaurus)

With `python server.py` running in one terminal, run the browser automation in another:

```powershell
python src\automation\gemini_auto_review.py --reviewer bahaa --limit 5
```

Common options:
- `--add-count N`: Add `N` new chats from the reviewer's SOP pool before reviewing (if omitted and 0 pending chats exist, the script auto-adds `--limit` or `5` chats).
- `--limit N`: Process at most `N` pending chats.
- `--model <name> --thinking <level>`: Choose any Gemini model and thinking effort level (defaults to `--model Neon --thinking Low`; e.g. `--model Hydrogen --thinking High`, or `--thinking ""` to leave thinking unchanged). Session stays persisted in `profiles/gemini_reviewer` after your first Google sign-in.
- `--keep-gemini-chats`: Keep Gemini conversations in the web sidebar instead of deleting them after saving.
- `--dry-run`: Run Gemini and log to `logs/` without modifying server state.

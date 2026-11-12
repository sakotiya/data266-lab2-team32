# DATA 266 Lab 2 — Pair_Programming_Team_32

Members: Shreya Akotiya, Zoheb Waghu

| Part | Topic | Notebook | Tag |
|---|---|---|---|
| 1 | Multimodal RAG (Kaggle) | `notebooks/p1_rag.ipynb` | `p1-v1.0` |
| 2 | Diffusion fine-tuning (LoRA) | `notebooks/p2_finetune.ipynb` | `p2-v1.0` |
| 3 | Multimodal multi-agent system | `notebooks/p3_agents.ipynb` | `p3-v1.0` |

## Folders

```
notebooks/        executed notebooks for all parts (required by the brief)
part1_rag/        src, eval, metrics, outputs, data
part2_diffusion/  src, configs, eval, metrics, outputs, data
part3_agents/     src, app, eval, metrics, outputs, test_images
```

`data/` folders are not committed.

## Setup

```bash
pip install -r requirements.txt
```

All required work runs on one 16 GB GPU (Kaggle/Colab T4).

## How to run each part

- Part 1: _to be filled in._
- Part 2: on a Kaggle T4, see [`part2_diffusion/README.md`](part2_diffusion/README.md) (setup, smoke run, full run, human evaluation).
- Part 3: on a Colab T4, open `notebooks/p3_agents.ipynb` and Restart & Run All (secrets: `GITHUB_TOKEN`, `GEMINI_API_KEY`). Demo app: last cell, or `python -m part3_agents.app.app` locally. See [`part3_agents/README.md`](part3_agents/README.md).

## Hosted APIs used

| Part | API | Model version | Used for |
|---|---|---|---|
| 3 | Google Gemini API (`google-genai`) | `gemini-3.5-flash-lite` | Planner, Vision, Prompt Engineering and Critique agents |

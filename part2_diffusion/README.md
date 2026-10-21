# Part 2 — Diffusion fine-tuning (LoRA)

| Folder | What goes here |
|---|---|
| `src/` | Data prep, training launch, generation, evaluation code |
| `configs/` | Training settings for run 1 and run 2 |
| `eval/` | The 20 fixed prompts, committed before training ends |
| `metrics/` | KID, CLIP, LPIPS, human-eval and training-cost CSVs |
| `outputs/` | Comparison grids and small sample images (full images go to Hugging Face) |
| `data/` | Downloaded dataset and splits (not committed) |

Notebook: `notebooks/p2_finetune.ipynb`. Model weights go to Hugging Face, not GitHub.

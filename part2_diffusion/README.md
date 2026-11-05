# Part 2 — Diffusion fine-tuning (LoRA)

Stable Diffusion 1.5 fine-tuned on `lambdalabs/naruto-blip-captions` with two LoRA runs that differ
only in rank (4 vs 32), compared with the base model. Lead: Shreya. Zoheb's separate baseline
pipeline (Team Rule 2) is their own code and is not part of this folder's main run.

| Folder / file | What is in it |
|---|---|
| `configs/` | `common.yaml` (data, sampler, metrics), `run1_lora_r4.yaml`, `run2_lora_r32.yaml` |
| `src/prep_data.py` | 2.1: cleaning (corrupted, exact and near duplicates), 90/10 split with seed 266, 512×512 |
| `src/train_lora.py` | 2.2: LoRA training loop; prints trainable params, peak memory and time |
| `src/generate.py` | 2.3: 20 prompts × seeds 0, 1 × {base, run 1, run 2}, same sampler; safety checker on |
| `src/evaluate.py` | 2.4: KID, CLIP score, LPIPS, memorization check |
| `src/human_eval.py` | 25 blinded pairs (final vs base), rating sheet, win rate |
| `src/figures.py`, `src/report.py` | grids, loss curves, failure candidates; results tables from the metrics files |
| `src/hub.py` | uploads both LoRAs (with model cards) and all generated images to Hugging Face |
| `eval/prompts.csv` | the 20 fixed prompts (10 in-domain from test captions, 5 out-of-domain, 5 compositional) |
| `metrics/` | everything measured, written by the notebook (JSON/CSV) |
| `outputs/` | grids, loss curves, memorization pairs, split manifest, rating sheet |

Notebooks: `notebooks/p2_finetune.ipynb` (Kaggle T4, about 3 hours) and `notebooks/p2_eval.ipynb`
(human-eval scoring and the final tables; no GPU needed). To regenerate them from the source:
`python part2_diffusion/build_notebooks.py` (it will not overwrite a notebook that has outputs).

## Running on Kaggle

One-time setup:

1. Kaggle account with a verified phone number (needed for GPU time).
2. Two tokens, stored as Kaggle secrets (notebook editor → **Add-ons → Secrets**), never in the repo:
   - `GH_TOKEN`: a GitHub fine-grained token with **read-only Contents** access to
     `data266-lab2-team32` only. The first cell uses it to clone the code.
   - `HF_TOKEN`: a Hugging Face token with **write** access, for the uploads.
3. Kaggle → **Create → New Notebook → File → Import Notebook** → upload `notebooks/p2_finetune.ipynb`.
   Settings: **Accelerator: GPU T4 x2** (one GPU is used), **Internet: on**. Attach both secrets.

Then three runs:

1. **Smoke run (about 15 minutes, interactive).** Set `SMOKE = True` in the second code cell and
   click **Run All**. It goes through every step with 10 training steps and 3 prompts and writes
   nothing to the repository or Hugging Face. Any error here would also have stopped the long run.
2. **Pick and commit the in-domain prompts.** Set `SMOKE = False` and run only up to the prompts
   table. Download `/kaggle/working/prompts.csv` (right-hand **Output** panel), copy it to
   `part2_diffusion/eval/prompts.csv`, and commit and push it **before** training (C2.3: the Git
   timestamp is checked). The pick is fixed by seed 266, so the full run selects the same prompts.
3. **Full run.** **Save Version → Save & Run All (Commit)** with `SMOKE = False`. It runs in the
   background (you can close the browser); after 25 training steps it prints the speed and
   expected finish time. When it finishes, open the version, and from **Output** download
   `p2_commit_files.zip`, then unzip it at the repository root. Download the executed notebook too
   (**⋮ → Download code**) and save it as `notebooks/p2_finetune.ipynb`.

## Human evaluation

Send `outputs/human_eval/rating_sheet.html` to the three raters (both team members and one person
outside the team). Each rater opens it in a browser, answers both questions for each of the 25 pairs,
and clicks **Download CSV**. Put the files in `metrics/human_eval/` as `ratings_<name>.csv`; do not
open `pair_key.csv` before rating. Then run `notebooks/p2_eval.ipynb` (Restart & Run All) and commit.

## Settings

| | Run 1 | Run 2 |
|---|---|---|
| Method | LoRA rank 4, alpha 4 | LoRA rank 32, alpha 32 |
| Target | UNet attention `to_q, to_k, to_v, to_out.0` | same |
| Steps × batch | 1,500 × 4 | same |
| Optimizer | AdamW, LR 1e-4 constant, weight decay 0.01, grad clip 1.0 | same |
| Precision | fp16 autocast, LoRA weights in fp32, gradient checkpointing | same |
| Seed | 266 (batch order, flips, noise, timesteps) | same |

Sampler for all three models: DPM-Solver++, 30 steps, guidance 7.5, 512×512, seeds 0 and 1.

The VAE latents and caption embeddings are computed once before training, because both encoders
are frozen. This gives the same objective as encoding every step, but faster and with less memory.
If the first full run looks too slow, lower `max_train_steps` in **both** run configs together,
so the two runs keep the same step budget.

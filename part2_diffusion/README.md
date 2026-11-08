# Part 2 — Diffusion fine-tuning (LoRA)

Stable Diffusion 1.5 fine-tuned on `lambdalabs/naruto-blip-captions`. Two LoRA runs with the same
data and step budget, differing only in rank (4 and 32), compared with the base model.

| File | Task |
|---|---|
| `configs/config.yaml` | all settings (seed 266, runs, training, sampler) |
| `src/data.py` | 2.1 remove corrupted and duplicate images, 90/10 split, resize to 512 |
| `src/train.py` | 2.2 LoRA training; prints trainable parameters, peak GPU memory, time |
| `src/generate.py` | 2.3 20 prompts × seeds 0 and 1 for base, run 1, run 2 |
| `src/evaluate.py` | 2.4 KID, CLIP score, LPIPS, memorization check |
| `src/human_eval.py` | 25 blinded pairs (final run vs base) and the win rate |
| `src/upload.py` | LoRAs with model card and the generated images to Hugging Face |
| `eval/prompts.csv` | the 20 fixed prompts (10 in-domain test captions, 5 out-of-domain, 5 compositional) |

Notebook: `notebooks/p2_finetune.ipynb`. Metrics are written to `metrics/`, figures to `outputs/`.
Weights and generated images go to `outputs/big/` (not committed) and then to Hugging Face.

## Run on Colab

Colab downloads nothing: `make_p2_bundle.sh` (next to the repository) packs the code, the
dataset's parquet files, SD 1.5 (fp16), CLIP, the KID/LPIPS weights and the missing Python
packages into `p2_bundle.zip` (about 3.5 GB).

1. In your own Terminal: `sh make_p2_bundle.sh`, then upload `p2_bundle.zip` to **My Drive**.
2. Colab: **File → Upload notebook** → `notebooks/p2_finetune.ipynb`; **Runtime → Change runtime type → T4 GPU**.
3. Set `TEST_RUN = True` and **Run all** (about 15 minutes, results in `MyDrive/data266_p2_test`).
4. Set `TEST_RUN = False` and **Run all** (about 3 hours). Keep the tab open. If Colab
   disconnects, **Run all** again; a finished training run is not repeated.
5. Download the executed notebook (**File → Download → .ipynb**) and, from Google Drive,
   `data266_p2/metrics` and `data266_p2/outputs` (copy them into `part2_diffusion/`) and
   `data266_p2/big` (weights and images, for the Hugging Face upload).

## Human evaluation

Three raters (both of us and one person outside the team) look at `outputs/human_eval/pair_01.jpg`
to `pair_25.jpg`. Each copies `outputs/human_eval/ratings_TEMPLATE.csv` to
`metrics/human_eval/ratings_<name>.csv` and writes A or B for both questions. Do not open
`metrics/human_eval_key.csv` before rating. Then run the notebook again to get the win rate.

## Upload to Hugging Face (from the laptop)

Put the downloaded `data266_p2/big` contents in `part2_diffusion/outputs/big/`, then:

```sh
.venv/bin/pip install huggingface_hub pandas pyyaml
.venv/bin/hf auth login
.venv/bin/python -m part2_diffusion.src.upload <hf-username> <final-run>
```

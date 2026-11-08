"""Upload both LoRAs (with a model card) and all generated images to Hugging Face.

Run on the laptop after `hf auth login`, with the Colab big files unzipped into
part2_diffusion/outputs/big/:
    python -m part2_diffusion.src.upload <hf-username> <final-run>
"""
import sys

import pandas as pd
from huggingface_hub import HfApi

from .common import BIG, CONFIG, METRICS, RUNS, load_json

CARD = """---
base_model: {base}
library_name: diffusers
license: creativeml-openrail-m
datasets: [{dataset}]
tags: [stable-diffusion, lora, text-to-image]
---

# Naruto-style LoRA for Stable Diffusion 1.5 ({run})

DATA 266 Lab 2, Part 2, Pair_Programming_Team_32. {role}

**Base model:** {base}, 512x512.

**Data:** {dataset}, {kept} images after cleaning, split 90/10 with seed 266 ({train} train, {test} test).

**Training:** LoRA rank {rank} on the UNet attention layers, {params:,} trainable parameters,
{steps} steps, batch {batch}, learning rate {lr}, fp16, on one {gpu} ({minutes} min, peak {mem} GB).

**Results** (20 prompts x 2 seeds, 30 DPM-Solver++ steps, guidance 7.5):

| | KID in-domain (lower is better) | CLIP score | LPIPS between seeds |
|---|---|---|---|
| This LoRA | {kid:.4f} | {clip:.2f} | {lpips:.3f} |
| Base SD 1.5 | {kid_base:.4f} | {clip_base:.2f} | {lpips_base:.3f} |

**Limitations:** learned from about 1,100 images of one anime style with short BLIP captions;
out-of-domain prompts get pulled toward that style, and it keeps the biases of SD 1.5.

**License:** the base model is CreativeML OpenRAIL-M. Naruto characters are copyrighted
(Masashi Kishimoto / Shueisha), so this adapter is for non-commercial research only.

```python
pipe = StableDiffusionPipeline.from_pretrained("{base}", torch_dtype=torch.float16).to("cuda")
pipe.load_lora_weights("{repo_id}")
```
"""


def main(user, final_run):
    api = HfApi()
    data = load_json(METRICS / "data_prep.json")
    metrics = pd.read_csv(METRICS / "auto_metrics.csv", index_col=0)
    for run in RUNS:
        repo_id = f"{user}/data266-team32-naruto-{run.replace('_', '-')}"
        t = load_json(METRICS / f"train_{run}.json")
        card = CARD.format(
            base=CONFIG["base_model"], dataset=CONFIG["dataset"], run=run,
            role="Selected final run." if run == final_run else "Comparison run.",
            kept=data["kept"], train=data["train"], test=data["test"], rank=t["rank"],
            params=t["trainable_params"], steps=t["steps"], batch=t["batch_size"],
            lr=t["learning_rate"], gpu=t["gpu"], minutes=t["training_minutes"],
            mem=t["peak_gpu_memory_gb"], kid=metrics.loc[run, "kid_mean"],
            clip=metrics.loc[run, "clip_score"], lpips=metrics.loc[run, "lpips_between_seeds"],
            kid_base=metrics.loc["base", "kid_mean"], clip_base=metrics.loc["base", "clip_score"],
            lpips_base=metrics.loc["base", "lpips_between_seeds"], repo_id=repo_id)
        (BIG / "loras" / run / "README.md").write_text(card)
        api.create_repo(repo_id, exist_ok=True)
        api.upload_folder(repo_id=repo_id, folder_path=str(BIG / "loras" / run))
        print("https://huggingface.co/" + repo_id)

    # image dataset: metadata.csv next to the images gives prompt, seed and model for each file
    gens = pd.read_csv(METRICS / "generations.csv")
    gens.rename(columns={"file": "file_name"}).to_csv(BIG / "images" / "metadata.csv", index=False)
    dataset_id = f"{user}/data266-team32-p2-images"
    api.create_repo(dataset_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(repo_id=dataset_id, repo_type="dataset", folder_path=str(BIG / "images"))
    print("https://huggingface.co/datasets/" + dataset_id)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

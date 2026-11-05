"""Upload to Hugging Face (C1.6, task 2.2.3, 2.3.3): both LoRAs with model cards, and every
generated image as a dataset with prompt, seed and model in the metadata.

Needs a write token in HF_TOKEN (on Kaggle: Add-ons -> Secrets -> HF_TOKEN). Without one the
upload is skipped and the notebook says so.
"""

from __future__ import annotations

import os

from .common import (RUNS, SMOKE, WORK, config, read_csv, read_json, repo_dir, work_dir, write_csv,
                     write_json)

LICENSE_NOTE = (
    "The base model is released under the CreativeML OpenRAIL-M license. The training images "
    "show Naruto characters, which are copyrighted (Masashi Kishimoto / Shueisha); this adapter "
    "is for non-commercial research and coursework only.")


def hf_token() -> str | None:
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        return None


def model_card(run: str, final_run: str, summary: dict, data: dict, human: dict | None) -> str:
    cfg = config()
    t = read_json(repo_dir("metrics") / f"train_{run}.json")
    s, b = summary[run], summary["base"]
    g = cfg["generation"]
    win = ""
    if human and human.get("raters"):
        win = (f"| Human preference vs base ({human['raters']} raters, {human['pairs']} pairs): "
               f"style / prompt match | {human['style']['win_rate_all_votes']:.0%} / "
               f"{human['prompt']['win_rate_all_votes']:.0%} | - |\n")
    role = "**Selected final run.**" if run == final_run else "Comparison run (not the final pick)."
    return f"""---
base_model: {cfg['base_model']}
library_name: diffusers
license: creativeml-openrail-m
datasets: [{cfg['dataset']}]
tags: [stable-diffusion, lora, text-to-image, diffusers, naruto]
---

# Naruto-style LoRA for Stable Diffusion 1.5 ({run})

DATA 266 Lab 2, Part 2 - team Pair_Programming_Team_32. {role}

## Base model and data
- Base: `{cfg['base_model']}` at {cfg['resolution']}x{cfg['resolution']}.
- Data: `{cfg['dataset']}` (revision `{data.get('revision', '?')}`), {data['rows_loaded']} images
  with BLIP captions; {data['removed_total']} removed as corrupted or duplicate; split 90/10 with
  seed {data['seed']}: {data['train']} training, {data['test']} test images (used only for evaluation).

## Training
| Setting | Value |
|---|---|
| Method | LoRA rank {t['rank']}, alpha {t['lora_alpha']}, on UNet `{', '.join(t['target_modules'])}` |
| Trainable parameters | {t['trainable_params']:,} ({t['trainable_pct']}% of the UNet) |
| Steps x batch | {t['steps']} x {t['batch_size']} ({t['epochs']} epochs) |
| Learning rate | {t['learning_rate']} ({t['lr_scheduler']}), AdamW, fp16 mixed precision |
| Hardware | 1x {t['gpu']}, peak memory {t['peak_gpu_mem_allocated_gb']} GB, {t['gpu_hours']} GPU-hours |

## Results (20 fixed prompts x 2 seeds, {g['scheduler']} {g['steps']} steps, guidance {g['guidance_scale']})
| Metric | This LoRA | Base SD 1.5 |
|---|---|---|
| KID vs test images, in-domain (lower is better) | {s['kid_mean']:.4f} ± {s['kid_std']:.4f} | {b['kid_mean']:.4f} ± {b['kid_std']:.4f} |
| CLIP score, all prompts | {s['clip_all']:.2f} | {b['clip_all']:.2f} |
| LPIPS between seeds (diversity) | {s['lpips_mean']:.3f} | {b['lpips_mean']:.3f} |
{win}
KID uses only {s['kid_n_fake']} generated images, so it is noisy.

## Limitations
- Learns one anime style from ~1,100 images; out-of-domain prompts are pulled toward that style
  and prompt following can be weaker than the base model.
- BLIP captions are short and generic, so character names and fine details are not learned.
- Inherits the biases and failure modes of Stable Diffusion 1.5.

## License
{LICENSE_NOTE}

## Use
```python
from diffusers import StableDiffusionPipeline
import torch
pipe = StableDiffusionPipeline.from_pretrained("{cfg['base_model']}", torch_dtype=torch.float16).to("cuda")
pipe.load_lora_weights("<this repo id>")
image = pipe("a ninja with spiky hair", num_inference_steps={g['steps']}, guidance_scale={g['guidance_scale']}).images[0]
```
"""


def upload(final_run: str, log=print) -> dict | None:
    if SMOKE:
        log("smoke mode - skipping the Hugging Face upload")
        return None
    tok = hf_token()
    if not tok:
        log("HF_TOKEN not set - skipping the Hugging Face upload")
        return None
    from huggingface_hub import HfApi

    api = HfApi(token=tok)
    user = os.environ.get("HF_NAMESPACE") or api.whoami()["name"]
    cfg = config()
    summary = read_json(repo_dir("metrics") / "summary_auto.json")
    data = read_json(repo_dir("metrics") / "data_prep.json")
    hp = repo_dir("metrics") / "human_eval_summary.json"
    human = read_json(hp) if hp.exists() else None
    links = {}
    for run in RUNS:
        rid = f"{user}/{cfg['hub']['model_repo_prefix']}-{run.split('_', 1)[1].replace('_', '-')}"
        d = work_dir("loras") / run
        (d / "README.md").write_text(model_card(run, final_run, summary, data, human))
        api.create_repo(rid, repo_type="model", exist_ok=True)
        api.upload_folder(repo_id=rid, folder_path=str(d), ignore_patterns=["checkpoint-*"],
                          commit_message=f"{run} LoRA + model card")
        links[run] = f"https://huggingface.co/{rid}"
        log(f"uploaded {run} -> {links[run]}")

    img_dir = WORK / "images"
    rows = read_csv(repo_dir("metrics") / "generations.csv")
    write_csv(img_dir / "metadata.csv",
              [{"file_name": r["file"].split("/", 1)[1], **{k: r[k] for k in (
                  "model", "prompt_id", "category", "prompt", "seed", "nsfw_flagged", "steps",
                  "guidance_scale", "scheduler")}} for r in rows])
    did = f"{user}/{cfg['hub']['image_dataset']}"
    api.create_repo(did, repo_type="dataset", exist_ok=True)
    api.upload_folder(repo_id=did, repo_type="dataset", folder_path=str(img_dir),
                      commit_message="generated images (base, run 1, run 2)")
    links["images"] = f"https://huggingface.co/datasets/{did}"
    log(f"uploaded {len(rows)} images -> {links['images']}")
    write_json(repo_dir("outputs") / "hf_links.json", links)
    return links

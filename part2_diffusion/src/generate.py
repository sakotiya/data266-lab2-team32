"""Task 2.3 - image generation.

The 20 fixed prompts x seeds {0, 1} for the base model and both runs, with the same sampler
settings for all three (DPM-Solver++, 30 steps, guidance 7.5, 512x512). The pipeline's own
safety checker is the NSFW filter: flagged images come back black and are recorded.

    python -m part2_diffusion.src.generate
"""

from __future__ import annotations

import gc
import time

import torch

from .common import (MODELS, config, image_path, prompts, repo_dir, work_dir, write_csv)


def load_pipeline(base: str):
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(base, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config,
                                                             algorithm_type="dpmsolver++")
    pipe.set_progress_bar_config(disable=True)
    return pipe.to("cuda")


def generate_all(models=MODELS, log=print) -> list[dict]:
    cfg = config()
    g = cfg["generation"]
    pipe = load_pipeline(cfg["base_model"])
    rows = []
    for model in models:
        if model != "base":
            pipe.load_lora_weights(str(work_dir("loras") / model))
        n_new = 0
        for p in prompts():
            for seed in g["seeds"]:
                path = image_path(model, p["prompt_id"], seed)
                path.parent.mkdir(parents=True, exist_ok=True)
                gen = torch.Generator("cuda").manual_seed(seed)
                torch.cuda.synchronize()
                t0 = time.time()
                out = pipe(p["prompt"], num_inference_steps=g["steps"],
                           guidance_scale=g["guidance_scale"], height=g["height"],
                           width=g["width"], generator=gen)
                torch.cuda.synchronize()
                secs = time.time() - t0
                out.images[0].save(path)
                nsfw = bool(out.nsfw_content_detected and out.nsfw_content_detected[0])
                rows.append({"model": model, "prompt_id": p["prompt_id"],
                             "category": p["category"], "prompt": p["prompt"], "seed": seed,
                             "file": str(path.relative_to(path.parents[2])),
                             "nsfw_flagged": nsfw, "seconds": round(secs, 2),
                             "steps": g["steps"], "guidance_scale": g["guidance_scale"],
                             "scheduler": g["scheduler"]})
                n_new += 1
        log(f"{model}: {n_new} images, "
            f"{sum(r['seconds'] for r in rows if r['model'] == model) / n_new:.1f} s/image, "
            f"{sum(r['nsfw_flagged'] for r in rows if r['model'] == model)} flagged NSFW")
        if model != "base":
            pipe.unload_lora_weights()
    write_csv(repo_dir("metrics") / "generations.csv", rows)
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return rows


if __name__ == "__main__":
    generate_all()

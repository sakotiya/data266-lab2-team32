"""2.3 Generate 20 prompts x 2 seeds for the base model and both LoRA runs."""
import pandas as pd
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

from .common import BIG, CONFIG, METRICS, MODELS, image_path, load_prompts, model_path


def generate(prompts=None, models=MODELS):
    prompts = prompts or load_prompts()
    settings = CONFIG["generate"]

    # The pipeline's built-in safety checker is the NSFW filter (flagged images come back black).
    pipe = StableDiffusionPipeline.from_pretrained(model_path(CONFIG["base_model"]),
                                                   torch_dtype=torch.float16, variant="fp16")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    rows = []
    for model in models:
        if model != "base":
            pipe.load_lora_weights(str(BIG / "loras" / model))
        for p in prompts:
            for seed in settings["seeds"]:
                result = pipe(p["prompt"], num_inference_steps=settings["steps"],
                              guidance_scale=settings["guidance_scale"],
                              generator=torch.Generator("cuda").manual_seed(seed))
                path = image_path(model, p["prompt_id"], seed)
                path.parent.mkdir(parents=True, exist_ok=True)
                result.images[0].save(path)
                nsfw = bool(result.nsfw_content_detected and result.nsfw_content_detected[0])
                rows.append({"model": model, "prompt_id": p["prompt_id"], "category": p["category"],
                             "prompt": p["prompt"], "seed": seed, "nsfw_flagged": nsfw,
                             "file": str(path.relative_to(BIG / "images"))})
        if model != "base":
            pipe.unload_lora_weights()
        print(model, "done")

    df = pd.DataFrame(rows)
    METRICS.mkdir(parents=True, exist_ok=True)
    df.to_csv(METRICS / "generations.csv", index=False)
    del pipe
    torch.cuda.empty_cache()
    return df

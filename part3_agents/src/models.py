"""Model access shared by the agents: Claude (VLM), Stable Diffusion pipelines, CLIP. All loaded lazily."""
import os
from functools import cache
from pathlib import Path

import torch

PART = Path(__file__).resolve().parent.parent
REPO = PART.parent
OUTPUTS = PART / "outputs"
SEED = 266

VLM = os.environ.get("P3_VLM", "claude-sonnet-5-5")       # same model as Part 1
SD_BASE = "stable-diffusion-v1-5/stable-diffusion-v1-5"
PIX2PIX = "timbrooks/instruct-pix2pix"
CLIP = "openai/clip-vit-base-patch32"          # same CLIP as Part 2
# Part 2 LoRA: a local folder from part2_diffusion/outputs/big/loras/<run>, or its Hugging Face repo id.
LORA = os.environ.get("P3_LORA", "sakotiya/data266-team32-naruto-run2-lora-r32")   # Part 2 final LoRA (r32)

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


def ask(contents, schema, system):
    """One VLM call (strings and PIL images in) that returns a validated pydantic object; same client as Part 1."""
    from part1_rag.src.common import claude
    return claude(contents, schema, system, model=VLM)


@cache
def sd_img2img():
    from diffusers import StableDiffusionImg2ImgPipeline
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(SD_BASE, torch_dtype=DTYPE).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)
    return pipe


@cache
def load_naruto_lora():
    """Loads the Part 2 LoRA into the img2img pipeline once; generate() switches it on/off per request."""
    if not (Path(LORA).is_dir() or LORA.count("/") == 1):
        raise RuntimeError(f"Part 2 LoRA not found at {LORA}; set P3_LORA to its folder or Hugging Face repo id.")
    sd_img2img().load_lora_weights(LORA, adapter_name="naruto")
    return True


@cache
def pix2pix():
    from diffusers import StableDiffusionInstructPix2PixPipeline
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(PIX2PIX, torch_dtype=DTYPE).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)
    return pipe


@cache
def clip():
    from transformers import CLIPModel, CLIPProcessor
    return CLIPModel.from_pretrained(CLIP).to(DEVICE).eval(), CLIPProcessor.from_pretrained(CLIP)


@torch.no_grad()
def clip_score(image, text):
    """100 * cosine(image, text), floored at 0 - the torchmetrics CLIPScore definition used in Part 2."""
    model, proc = clip()
    x = proc(text=[text], images=image, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    out = model(**x)
    return round(max(100 * torch.cosine_similarity(out.image_embeds, out.text_embeds).item(), 0.0), 2)

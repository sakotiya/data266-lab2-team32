"""Model access shared by the agents: Gemini (VLM), Stable Diffusion pipelines, CLIP. All loaded lazily."""
import os
import time
from functools import cache
from pathlib import Path

import torch

PART = Path(__file__).resolve().parent.parent
REPO = PART.parent
OUTPUTS = PART / "outputs"
SEED = 266

VLM = os.environ.get("P3_VLM", "gemini-3.5-flash-lite")   # same model as Part 1 (free tier)
SD_BASE = "stable-diffusion-v1-5/stable-diffusion-v1-5"
PIX2PIX = "timbrooks/instruct-pix2pix"
CLIP = "openai/clip-vit-base-patch32"          # same CLIP as Part 2
# Part 2 LoRA: a local folder from part2_diffusion/outputs/big/loras/<run>, or its Hugging Face repo id.
LORA = os.environ.get("P3_LORA", str(REPO / "part2_diffusion" / "outputs" / "big" / "loras" / "run2_lora_r32"))

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


@cache
def gemini():
    from google import genai
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set (export it, or add it to Colab/Kaggle secrets).")
    return genai.Client(api_key=key)


def ask(contents, schema, system):
    """One Gemini call that returns a validated pydantic object. Retries rate-limit/server errors."""
    from google.genai import errors, types
    config = types.GenerateContentConfig(system_instruction=system, temperature=0, seed=SEED,
                                         response_mime_type="application/json", response_schema=schema)
    for i in range(5):
        try:
            r = gemini().models.generate_content(model=VLM, contents=contents, config=config)
            return r.parsed if isinstance(r.parsed, schema) else schema.model_validate_json(r.text)
        except errors.APIError as e:
            if e.code not in (429, 500, 503) or i == 4:
                raise
            time.sleep(10 * 2 ** i)


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

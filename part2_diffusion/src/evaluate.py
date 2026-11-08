"""2.4 Metrics: KID and CLIP score (primary), LPIPS between seeds (secondary), memorization check."""
import json

import lpips
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchmetrics.image.kid import KernelInceptionDistance
from transformers import CLIPModel, CLIPProcessor

from .common import CONFIG, DATA, METRICS, MODELS, SEED, image_path, load_prompts, model_path


def load_split(name):
    rows = [json.loads(line) for line in open(DATA / name / "metadata.jsonl")]
    return [(DATA / name / r["file_name"], r["text"]) for r in rows]


class Clip:
    def __init__(self):
        name = model_path(CONFIG["clip_model"])
        self.model = CLIPModel.from_pretrained(name).to("cuda").eval()
        self.processor = CLIPProcessor.from_pretrained(name)

    @torch.no_grad()
    def embed_images(self, paths):
        out = []
        for i in range(0, len(paths), 32):
            images = [Image.open(p).convert("RGB") for p in paths[i:i + 32]]
            pixels = self.processor(images=images, return_tensors="pt")["pixel_values"].to("cuda")
            emb = self.model.visual_projection(self.model.vision_model(pixel_values=pixels).pooler_output)
            out.append(torch.nn.functional.normalize(emb, dim=-1))
        return torch.cat(out)

    @torch.no_grad()
    def embed_texts(self, texts):
        tokens = self.processor.tokenizer(texts, padding=True, truncation=True,
                                          return_tensors="pt").to("cuda")
        emb = self.model.text_projection(self.model.text_model(**tokens).pooler_output)
        return torch.nn.functional.normalize(emb, dim=-1)


def clip_scores(clip, prompts):
    """CLIP score = 100 * cosine(image, prompt), floored at 0."""
    seeds = CONFIG["generate"]["seeds"]
    text = clip.embed_texts([p["prompt"] for p in prompts])
    rows = []
    for model in MODELS:
        for k, p in enumerate(prompts):
            imgs = clip.embed_images([image_path(model, p["prompt_id"], s) for s in seeds])
            for s, emb in zip(seeds, imgs):
                score = 100 * max(float(emb @ text[k]), 0)
                rows.append({"model": model, "prompt_id": p["prompt_id"],
                             "category": p["category"], "seed": s, "clip_score": score})
    return pd.DataFrame(rows)


def to_uint8(paths):
    arrays = [np.asarray(Image.open(p).convert("RGB").resize((299, 299))) for p in paths]
    return torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).cuda()


def kid_scores(prompts, subset_size=10):
    """KID between the in-domain generations (10 prompts x 2 seeds) and the test images."""
    test = [p for p, _ in load_split("test")]
    in_domain = [p["prompt_id"] for p in prompts if p["category"] == "in_domain"]
    result = {}
    for model in MODELS:
        fakes = [image_path(model, pid, s) for pid in in_domain for s in CONFIG["generate"]["seeds"]]
        kid = KernelInceptionDistance(subset_size=min(subset_size, len(fakes))).cuda()
        for i in range(0, len(test), 32):
            kid.update(to_uint8(test[i:i + 32]), real=True)
        kid.update(to_uint8(fakes), real=False)
        torch.manual_seed(SEED)
        mean, std = kid.compute()
        result[model] = {"kid_mean": float(mean), "kid_std": float(std)}
    return result


def lpips_diversity(prompts):
    """Average LPIPS distance between the seed-0 and seed-1 image of each prompt."""
    loss = lpips.LPIPS(net="alex").cuda()

    def tensor(path):
        x = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 127.5 - 1
        return torch.from_numpy(x).permute(2, 0, 1)[None].cuda()

    rows = []
    with torch.no_grad():
        for model in MODELS:
            for p in prompts:
                a, b = (tensor(image_path(model, p["prompt_id"], s)) for s in (0, 1))
                rows.append({"model": model, "prompt_id": p["prompt_id"],
                             "category": p["category"], "lpips": float(loss(a, b))})
    return pd.DataFrame(rows)


def memorization(clip, model, prompts, n=10):
    """For n in-domain images (seed 0), the most similar training image by CLIP similarity."""
    train = load_split("train")
    train_emb = clip.embed_images([p for p, _ in train])
    ids = [p["prompt_id"] for p in prompts if p["category"] == "in_domain"][:n]
    gen_emb = clip.embed_images([image_path(model, pid, 0) for pid in ids])
    best = (gen_emb @ train_emb.T).max(dim=1)
    rows = [{"prompt_id": pid, "closest_train_image": train[int(j)][0].name,
             "train_caption": train[int(j)][1], "similarity": round(float(v), 4)}
            for pid, v, j in zip(ids, best.values, best.indices)]
    return pd.DataFrame(rows).sort_values("similarity", ascending=False)


def evaluate(prompts=None):
    prompts = prompts or load_prompts()
    clip = Clip()
    clip_df = clip_scores(clip, prompts)
    lpips_df = lpips_diversity(prompts)
    kid = kid_scores(prompts)
    clip_df.to_csv(METRICS / "clip_scores.csv", index=False)
    lpips_df.to_csv(METRICS / "lpips.csv", index=False)

    summary = pd.DataFrame({
        "kid_mean": {m: kid[m]["kid_mean"] for m in MODELS},
        "kid_std": {m: kid[m]["kid_std"] for m in MODELS},
        "clip_score": clip_df.groupby("model")["clip_score"].mean(),
        "clip_in_domain": clip_df[clip_df.category == "in_domain"].groupby("model")["clip_score"].mean(),
        "lpips_between_seeds": lpips_df.groupby("model")["lpips"].mean(),
    }).loc[MODELS]
    summary.to_csv(METRICS / "auto_metrics.csv")
    return summary, clip, clip_df

"""Task 2.4 - automatic metrics and the memorization check.

  CLIP score  100 * cos(CLIP image, CLIP prompt), clipped at 0 (as torchmetrics defines it)
  KID         in-domain generations (10 prompts x 2 seeds) vs the held-out test images
  LPIPS       distance between the seed-0 and seed-1 image of each prompt (diversity)
  Memorization: for in-domain images of the final run, the most similar training image by
  CLIP image similarity; the base model (which never saw the training set) is the reference.

    python -m part2_diffusion.src.evaluate
"""

from __future__ import annotations

import gc
import os

import numpy as np
import torch
from PIL import Image

from .common import (MODELS, RUNS, SEED, config, image_path, prompts, read_csv, read_json,
                     repo_dir, write_csv, write_json)
from .train_lora import load_split


class Clip:
    """CLIP embeddings via the vision/text towers + projections (stable across transformers versions)."""

    def __init__(self, name: str):
        from transformers import CLIPModel, CLIPProcessor
        self.model = CLIPModel.from_pretrained(name).to("cuda").eval()
        self.proc = CLIPProcessor.from_pretrained(name)

    @torch.no_grad()
    def images(self, imgs, bs: int = 32) -> torch.Tensor:
        out = []
        for i in range(0, len(imgs), bs):
            pv = self.proc(images=imgs[i:i + bs], return_tensors="pt")["pixel_values"].to("cuda")
            e = self.model.visual_projection(self.model.vision_model(pixel_values=pv).pooler_output)
            out.append(torch.nn.functional.normalize(e.float(), dim=-1))
        return torch.cat(out)

    @torch.no_grad()
    def texts(self, texts) -> torch.Tensor:
        t = self.proc.tokenizer(texts, padding=True, truncation=True, max_length=77,
                                return_tensors="pt").to("cuda")
        e = self.model.text_projection(self.model.text_model(**t).pooler_output)
        return torch.nn.functional.normalize(e.float(), dim=-1)


def load(path) -> Image.Image:
    return Image.open(path).convert("RGB")


def kid_score(real_imgs, fake_by_model, subset_size, subsets):
    """KID mean/std per model; the real (test) features are extracted once."""
    from torchmetrics.image.kid import KernelInceptionDistance

    def u8(imgs):
        return torch.from_numpy(np.stack([np.asarray(i.resize((299, 299), Image.BICUBIC))
                                          for i in imgs])).permute(0, 3, 1, 2).cuda()

    size = min(subset_size, len(real_imgs), *[len(v) for v in fake_by_model.values()])
    kid = KernelInceptionDistance(subset_size=size, subsets=subsets,
                                  reset_real_features=False).cuda()
    for i in range(0, len(real_imgs), 32):
        kid.update(u8(real_imgs[i:i + 32]), real=True)
    res = {}
    for model, fakes in fake_by_model.items():
        kid.reset()
        kid.update(u8(fakes), real=False)
        torch.manual_seed(SEED)                         # the subsets are drawn at random
        m, s = kid.compute()
        res[model] = {"kid_mean": float(m), "kid_std": float(s), "kid_subset_size": size,
                      "kid_n_fake": len(fakes), "kid_n_real": len(real_imgs)}
    del kid
    return res


def lpips_scores(net: str):
    import lpips
    fn = lpips.LPIPS(net=net, verbose=False).cuda().eval()

    def t(img):
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 127.5 - 1).permute(2, 0, 1)
        return x[None].cuda()

    out = []
    with torch.no_grad():
        for model in MODELS:
            for p in prompts():
                a, b = (load(image_path(model, p["prompt_id"], s)) for s in (0, 1))
                out.append({"model": model, "prompt_id": p["prompt_id"],
                            "category": p["category"], "lpips_seed0_vs_seed1":
                            float(fn(t(a), t(b)).item())})
    del fn
    return out


def choose_final(summary: dict) -> str:
    """Default pick: lower in-domain KID; P2_FINAL_RUN overrides after the team has looked."""
    forced = os.environ.get("P2_FINAL_RUN")
    if forced:
        return forced
    return min(RUNS, key=lambda r: (summary[r]["kid_mean"], -summary[r]["clip_all"]))


def memorization(clip, final_run: str, n: int):
    """Top-1 most similar training image for n in-domain generations (seed 0), final run and base."""
    train = load_split("train")
    tr_emb = clip.images([load(p) for p, _ in train])
    ids = [p["prompt_id"] for p in prompts() if p["category"] == "in_domain"][:n]
    rows = []
    for model in ("base", final_run):
        gen = clip.images([load(image_path(model, pid, 0)) for pid in ids])
        sim = gen @ tr_emb.T
        best = sim.max(dim=1)
        for k, pid in enumerate(ids):
            j = int(best.indices[k])
            rows.append({"model": model, "prompt_id": pid, "seed": 0,
                         "top1_train_image": train[j][0].name, "top1_train_caption": train[j][1],
                         "clip_image_similarity": round(float(best.values[k]), 4)})
    return rows


def main(log=print) -> dict:
    cfg = config()
    e = cfg["eval"]
    gens = read_csv(repo_dir("metrics") / "generations.csv")
    clip = Clip(e["clip_model"])

    # CLIP score for every image
    ps = {p["prompt_id"]: p for p in prompts()}
    pid_list = list(ps)
    temb = clip.texts([ps[k]["prompt"] for k in pid_list])
    per_image = []
    for model in MODELS:
        rows = [r for r in gens if r["model"] == model]
        iemb = clip.images([load(image_path(model, r["prompt_id"], int(r["seed"]))) for r in rows])
        for r, ie in zip(rows, iemb):
            cos = float(ie @ temb[pid_list.index(r["prompt_id"])])
            per_image.append({**{k: r[k] for k in ("model", "prompt_id", "category", "seed",
                                                    "nsfw_flagged", "seconds")},
                              "clip_score": round(100 * max(cos, 0.0), 3)})
    write_csv(repo_dir("metrics") / "clip_per_image.csv", per_image)

    # KID on in-domain generations vs test images
    test_imgs = [load(p) for p, _ in load_split("test")]
    in_dom = [k for k in pid_list if ps[k]["category"] == "in_domain"]
    fakes = {m: [load(image_path(m, k, s)) for k in in_dom for s in cfg["generation"]["seeds"]]
             for m in MODELS}
    kid = kid_score(test_imgs, fakes, e["kid_subset_size"], e["kid_subsets"])

    lp = lpips_scores(e["lpips_net"])
    write_csv(repo_dir("metrics") / "lpips_per_prompt.csv", lp)

    summary = {}
    for m in MODELS:
        ci = [r for r in per_image if r["model"] == m]
        li = [r["lpips_seed0_vs_seed1"] for r in lp if r["model"] == m]
        secs = [float(r["seconds"]) for r in ci]
        summary[m] = {
            "clip_all": float(np.mean([r["clip_score"] for r in ci])),
            **{f"clip_{c}": float(np.mean([r["clip_score"] for r in ci if r["category"] == c]))
               for c in ("in_domain", "out_of_domain", "compositional")
               if any(r["category"] == c for r in ci)},
            **kid[m], "lpips_mean": float(np.mean(li)),
            "nsfw_flagged": int(sum(str(r["nsfw_flagged"]) == "True" for r in ci)),
            "images": len(ci), "median_s_per_image": float(np.median(secs)),
        }
    for r in RUNS:
        t = read_json(repo_dir("metrics") / f"train_{r}.json")
        summary[r].update(gpu_hours=t["gpu_hours"], peak_gpu_mem_gb=t["peak_gpu_mem_allocated_gb"],
                          trainable_params=t["trainable_params"])
    final = choose_final(summary)
    summary["final_run"] = final
    summary["final_run_rule"] = ("P2_FINAL_RUN override" if os.environ.get("P2_FINAL_RUN")
                                 else "lower in-domain KID (tie-break: higher CLIP)")

    mem = memorization(clip, final, e["memorization_images"])
    write_csv(repo_dir("metrics") / "memorization.csv", mem)
    for m in ("base", final):
        v = [r["clip_image_similarity"] for r in mem if r["model"] == m]
        summary[m]["memorization_top1_mean"] = float(np.mean(v))
        summary[m]["memorization_top1_max"] = float(np.max(v))
    write_json(repo_dir("metrics") / "summary_auto.json", summary)
    del clip
    gc.collect()
    torch.cuda.empty_cache()
    log(f"final run: {final} ({summary['final_run_rule']})")
    return summary


if __name__ == "__main__":
    import json
    print(json.dumps(main(), indent=2))

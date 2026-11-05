"""Figures for Part 2: comparison grids, loss curves, memorization pairs, failure candidates.

Grids are composed with PIL as small JPEGs so they can be committed; full-size images go to
Hugging Face.
"""

from __future__ import annotations

import textwrap

from PIL import Image, ImageDraw, ImageFont

from .common import (DATA, MODELS, RUNS, config, image_path, prompts, read_csv, repo_dir)

CELL = 256
LABEL = {"base": "base SD 1.5", "run1_lora_r4": "run 1: LoRA r=4",
         "run2_lora_r32": "run 2: LoRA r=32"}
INK, INK_2 = (30, 30, 30), (110, 110, 110)


def font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                                    # Pillow < 10.1
        return ImageFont.load_default()


def _cell(path) -> Image.Image:
    return Image.open(path).convert("RGB").resize((CELL, CELL), Image.LANCZOS)


def comparison_grid(prompt: dict, seeds=(0, 1)) -> Image.Image:
    """Rows = seeds, columns = base | run 1 | run 2, prompt text on top."""
    head, side, gap = 64, 60, 4
    step = CELL + gap
    W, H = side + step * len(MODELS), head + 24 + step * len(seeds)
    g = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(g)
    text = textwrap.fill(f"{prompt['prompt_id']} ({prompt['category']}): {prompt['prompt']}", 95)
    d.text((8, 6), text, fill=INK, font=font(14))
    for c, m in enumerate(MODELS):
        d.text((side + c * step + 6, head + 4), LABEL[m], fill=INK_2, font=font(14))
        for r, s in enumerate(seeds):
            g.paste(_cell(image_path(m, prompt["prompt_id"], s)), (side + c * step, head + 24 + r * step))
    for r, s in enumerate(seeds):
        d.text((8, head + 24 + r * step + CELL // 2), f"seed {s}", fill=INK_2, font=font(14))
    return g


def all_grids() -> list:
    ids = config()["eval"]["grid_prompts"]
    ps = {p["prompt_id"]: p for p in prompts()}
    out = []
    for pid in ids:
        if pid not in ps:                                # smoke mode has fewer prompts
            continue
        path = repo_dir("outputs") / "grids" / f"{pid}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        comparison_grid(ps[pid]).save(path, quality=85)
        out.append(path)
    return out


def loss_curves(path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"run1_lora_r4": "#2a6ebb", "run2_lora_r32": "#d1495b"}
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for r in RUNS:
        rows = read_csv(repo_dir("metrics") / f"loss_{r}.csv")
        x = [int(v["step"]) for v in rows]
        y = [float(v["loss"]) for v in rows]
        ax.plot(x, y, color=colors[r], lw=1.6, label=LABEL[r])
    ax.set(xlabel="training step", ylabel="MSE loss (mean of 25 steps)",
           title="Training loss, same data and step budget")
    ax.grid(color="#e5e5e5", lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def memorization_pairs(final_run: str, k: int = 3):
    """The k generated/training pairs with the highest CLIP image similarity (final run)."""
    rows = [r for r in read_csv(repo_dir("metrics") / "memorization.csv") if r["model"] == final_run]
    rows.sort(key=lambda r: -float(r["clip_image_similarity"]))
    rows = rows[:k]
    g = Image.new("RGB", (2 * CELL + 10, len(rows) * (CELL + 30) + 30), "white")
    d = ImageDraw.Draw(g)
    d.text((6, 6), f"generated ({LABEL[final_run]})", fill=INK, font=font(14))
    d.text((CELL + 16, 6), "closest training image", fill=INK, font=font(14))
    for i, r in enumerate(rows):
        y = 30 + i * (CELL + 30)
        g.paste(_cell(image_path(final_run, r["prompt_id"], 0)), (0, y))
        g.paste(_cell(DATA / "train" / r["top1_train_image"]), (CELL + 10, y))
        d.text((6, y + CELL + 4), f"{r['prompt_id']}  CLIP similarity "
               f"{float(r['clip_image_similarity']):.3f}", fill=INK_2, font=font(13))
    path = repo_dir("outputs") / "memorization_top3.jpg"
    g.save(path, quality=85)
    return path, rows


def failure_candidates(final_run: str, n: int = 4):
    """Lowest-CLIP images of the final run and its lowest-diversity prompts, for the write-up."""
    clip = [r for r in read_csv(repo_dir("metrics") / "clip_per_image.csv") if r["model"] == final_run]
    clip.sort(key=lambda r: float(r["clip_score"]))
    lp = [r for r in read_csv(repo_dir("metrics") / "lpips_per_prompt.csv") if r["model"] == final_run]
    lp.sort(key=lambda r: float(r["lpips_seed0_vs_seed1"]))
    ps = {p["prompt_id"]: p for p in prompts()}
    picks = [(r["prompt_id"], int(r["seed"]), f"low CLIP {float(r['clip_score']):.1f}")
             for r in clip[:n]]
    picks += [(r["prompt_id"], s, f"low LPIPS {float(r['lpips_seed0_vs_seed1']):.3f}")
              for r in lp[:2] for s in (0, 1)]
    g = Image.new("RGB", (len(picks) * CELL, CELL + 70), "white")
    d = ImageDraw.Draw(g)
    for i, (pid, s, why) in enumerate(picks):
        g.paste(_cell(image_path(final_run, pid, s)), (i * CELL, 0))
        d.text((i * CELL + 4, CELL + 4), f"{pid} s{s}: {why}", fill=INK, font=font(13))
        d.text((i * CELL + 4, CELL + 22), textwrap.fill(ps[pid]["prompt"], 34)[:110],
               fill=INK_2, font=font(11))
    path = repo_dir("outputs") / "failure_candidates.jpg"
    g.save(path, quality=85)
    return path, picks


def train_test_samples(n: int = 6):
    """A strip of training images with captions, for the dataset description."""
    import json
    with open(DATA / "train" / "metadata.jsonl") as fh:
        rows = [json.loads(x) for x in fh][:n]
    g = Image.new("RGB", (n * CELL, CELL + 50), "white")
    d = ImageDraw.Draw(g)
    for i, r in enumerate(rows):
        g.paste(_cell(DATA / "train" / r["file_name"]), (i * CELL, 0))
        d.text((i * CELL + 4, CELL + 4), textwrap.fill(r["text"], 36)[:100], fill=INK_2, font=font(12))
    path = repo_dir("outputs") / "dataset_samples.jpg"
    g.save(path, quality=85)
    return path


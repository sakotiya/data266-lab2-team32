"""Task 2.1 - data preparation.

Loads the Naruto BLIP captions dataset, removes corrupted and duplicate images (and says how
many of each), splits 90/10 with seed 266, resizes to 512x512, and writes:
  DATA/train, DATA/test           512x512 PNGs + metadata.jsonl (file_name, text)
  outputs/split_manifest.csv      every original row: md5, kept/removed reason, split, caption
  metrics/data_prep.json          counts and dataset statistics for the report
  eval/prompts.csv                adds the 10 in-domain prompts (test captions) if not there yet
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from collections import Counter

import numpy as np
from PIL import Image

from .common import (DATA, SEED, config, prompts_file, read_csv, repo_dir, write_csv,
                     write_json)

STOP = set("a an the of with and in on at to is are his her their its for from by".split())


def dhash(img: Image.Image) -> int:
    """64-bit difference hash: 9x8 grayscale thumbnail, one bit per left/right comparison."""
    g = np.asarray(img.convert("L").resize((9, 8), Image.LANCZOS), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def check_image(raw: bytes, min_side: int, min_std: float):
    """Return (RGB image, None) or (None, reason) for an image that cannot be used."""
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
        img = img.convert("RGB")
    except Exception as e:                                   # truncated / not an image
        return None, f"corrupted: cannot decode ({type(e).__name__})"
    if min(img.size) < min_side:
        return None, f"corrupted: too small {img.size}"
    if float(np.asarray(img, dtype=np.float32).std()) < min_std:
        return None, "corrupted: blank (single colour)"
    return img, None


def resize_square(img: Image.Image, size: int) -> Image.Image:
    """Shorter side to `size`, then a centre crop to size x size."""
    w, h = img.size
    s = size / min(w, h)
    img = img.resize((max(size, round(w * s)), max(size, round(h * s))), Image.LANCZOS)
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def load_rows(name: str):
    """(rows, info): rows are dicts with raw image bytes and caption, in dataset order."""
    from datasets import Image as HFImage, load_dataset
    from huggingface_hub import HfApi

    ds = load_dataset(name, split="train").cast_column("image", HFImage(decode=False))
    meta = HfApi().dataset_info(name)
    card = getattr(meta, "card_data", None)
    info = {"dataset": name, "revision": meta.sha,
            "license": (card or {}).get("license") if card is not None else None}
    rows = [{"raw": r["image"]["bytes"], "text": r["text"]} for r in ds]
    return rows, info


def clean(rows, cfg):
    """Mark each row kept or removed (corrupted / empty caption / exact or near duplicate)."""
    status, images, hashes = [], {}, {}
    seen_md5 = {}
    for i, r in enumerate(rows):
        md5 = hashlib.md5(r["raw"]).hexdigest()
        img, why = check_image(r["raw"], cfg["min_side"], cfg["min_pixel_std"])
        if why is None and not str(r["text"]).strip():
            why = "empty caption"
        if why is None and md5 in seen_md5:
            why = f"exact duplicate of row {seen_md5[md5]}"
        if why is None:
            seen_md5[md5] = i
            images[i], hashes[i] = img, dhash(img)
        status.append({"orig_index": i, "md5": md5, "removed": why or "",
                       "orig_size": "x".join(map(str, img.size)) if img else ""})

    # Near duplicates: same picture re-encoded or slightly resized. Keep the first one.
    keep = sorted(hashes)
    hv = np.array([hashes[i] for i in keep], dtype=np.uint64)
    dropped = set()
    for a in range(len(keep)):
        if keep[a] in dropped:
            continue
        x = np.bitwise_xor(hv[a + 1:], hv[a])
        dist = np.array([bin(int(v)).count("1") for v in x])
        for b in np.where(dist <= cfg["near_dup_hamming"])[0]:
            j = keep[a + 1 + b]
            if j not in dropped:
                dropped.add(j)
                status[j]["removed"] = f"near duplicate of row {keep[a]} (dHash distance {dist[b]})"
    removed_pairs = []                                  # (removed image, kept image) for a figure
    for j in sorted(dropped):
        img = images.pop(j)
        if len(removed_pairs) < 4:
            k = int(status[j]["removed"].split("row ")[1].split(" ")[0])
            removed_pairs.append((j, img, k, images[k]))
    return status, images, removed_pairs


def near_duplicate_figure(pairs, path, cell: int = 192) -> None:
    """Removed near duplicates next to the image they duplicate, to check the threshold by eye."""
    from PIL import ImageDraw
    g = Image.new("RGB", (2 * cell + 8, len(pairs) * (cell + 22)), "white")
    d = ImageDraw.Draw(g)
    for r, (j, a, k, b) in enumerate(pairs):
        y = r * (cell + 22)
        g.paste(a.resize((cell, cell)), (0, y))
        g.paste(b.resize((cell, cell)), (cell + 8, y))
        d.text((4, y + cell + 4), f"removed row {j}  |  kept row {k}", fill=(60, 60, 60))
    g.save(path, quality=85)


def split(kept: list[int], test_fraction: float, seed: int = SEED):
    """90/10 split of the kept row indices with a fixed seed."""
    order = np.random.default_rng(seed).permutation(kept)
    n_test = round(test_fraction * len(kept))
    return sorted(order[n_test:].tolist()), sorted(order[:n_test].tolist())


def caption_stats(texts):
    words = [w for t in texts for w in re.findall(r"[a-z]+", t.lower())]
    lens = [len(t.split()) for t in texts]
    top = Counter(w for w in words if w not in STOP).most_common(20)
    return {"caption_words_mean": round(float(np.mean(lens)), 1), "caption_words_min": min(lens),
            "caption_words_max": max(lens), "top_words": top}


def add_in_domain_prompts(test_texts: list[str], n: int = 10, seed: int = SEED):
    """Pick n distinct test captions as the in-domain prompts; never overwrite committed ones."""
    path = prompts_file()
    rows = read_csv(path)
    if any(r["category"] == "in_domain" for r in rows):
        return [r for r in rows if r["category"] == "in_domain"], False
    uniq = sorted(set(t.strip() for t in test_texts))
    pick = np.random.default_rng(seed).choice(len(uniq), size=min(n, len(uniq)), replace=False)
    new = [{"prompt_id": f"ID{k + 1:02d}", "category": "in_domain", "prompt": uniq[i],
            "source": "test-set caption"} for k, i in enumerate(sorted(pick))]
    write_csv(path, new + rows)
    return new, True


def main():
    cfg = config()
    rows, info = load_rows(cfg["dataset"])
    status, images, removed_pairs = clean(rows, cfg)
    if removed_pairs:
        near_duplicate_figure(removed_pairs, repo_dir("outputs") / "near_duplicates.jpg")
    kept = sorted(images)
    train_idx, test_idx = split(kept, cfg["test_fraction"])

    for name, idx in (("train", train_idx), ("test", test_idx)):
        d = DATA / name
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "metadata.jsonl", "w") as fh:
            for i in idx:
                fn = f"{i:05d}.png"
                resize_square(images[i], cfg["resolution"]).save(d / fn)
                fh.write(json.dumps({"file_name": fn, "text": rows[i]["text"]}) + "\n")
    which = {i: "train" for i in train_idx} | {i: "test" for i in test_idx}
    for s in status:
        s["split"] = which.get(s["orig_index"], "")
        s["caption"] = rows[s["orig_index"]]["text"]
    write_csv(repo_dir("outputs") / "split_manifest.csv", status)

    reasons = Counter(s["removed"].split(" of row")[0].split(" (")[0].split(":")[0]
                      for s in status if s["removed"])
    sizes = Counter(s["orig_size"] for s in status if s["orig_size"])
    in_domain, created = add_in_domain_prompts([rows[i]["text"] for i in test_idx])
    summary = {
        **info, "rows_loaded": len(rows), "removed_total": len(rows) - len(kept),
        "removed_by_reason": dict(reasons), "kept": len(kept),
        "train": len(train_idx), "test": len(test_idx), "seed": SEED,
        "resolution": cfg["resolution"], "original_sizes": dict(sizes.most_common(10)),
        **caption_stats([rows[i]["text"] for i in kept]),
        "in_domain_prompts_created_now": created,
    }
    write_json(repo_dir("metrics") / "data_prep.json", summary)
    return summary, in_domain


if __name__ == "__main__":
    s, _ = main()
    print(json.dumps(s, indent=2))

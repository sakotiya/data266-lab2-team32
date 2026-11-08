"""2.1 Data preparation: remove corrupted and duplicate images, split 90/10, resize to 512."""
import hashlib
import io
import json
import os

import numpy as np
import pandas as pd
from datasets import Image as HFImage, load_dataset
from PIL import Image

from .common import CONFIG, DATA, OUTPUTS, PART, METRICS, SEED, save_json


def dhash(img):
    # 9x8 grayscale thumbnail, one bit per "is this pixel brighter than its left neighbour"
    g = np.asarray(img.convert("L").resize((9, 8), Image.LANCZOS), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a, b):
    return bin(a ^ b).count("1")


def resize_square(img, size):
    # shorter side to `size`, then centre crop
    w, h = img.size
    s = size / min(w, h)
    img = img.resize((round(w * s), round(h * s)), Image.LANCZOS)
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def prepare():
    raw = os.environ.get("P2_RAW")         # the dataset's parquet files, if we brought our own copy
    if raw:
        ds = load_dataset("parquet", data_files=f"{raw}/*.parquet", split="train")
    else:
        ds = load_dataset(CONFIG["dataset"], split="train")
    ds = ds.cast_column("image", HFImage(decode=False))
    print("loaded", len(ds), "images")

    images, captions, removed = {}, {}, []
    seen_md5, hashes = {}, {}
    for i, row in enumerate(ds):
        raw = row["image"]["bytes"]
        try:
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            removed.append((i, "corrupted"))
            continue
        md5 = hashlib.md5(raw).hexdigest()
        if md5 in seen_md5:
            removed.append((i, "exact duplicate"))
            continue
        h = dhash(img)
        twin = next((j for j, hj in hashes.items()
                     if hamming(h, hj) <= CONFIG["near_duplicate_distance"]), None)
        if twin is not None:
            removed.append((i, "near duplicate"))
            continue
        seen_md5[md5], hashes[i] = i, h
        images[i], captions[i] = img, row["text"]

    reasons = pd.Series([r for _, r in removed], dtype=str).value_counts().to_dict()
    print("removed", len(removed), reasons)

    # closest pair that was kept, to show the near-duplicate threshold is not too loose
    keys = list(hashes)
    closest = min(hamming(hashes[a], hashes[b]) for n, a in enumerate(keys) for b in keys[n + 1:])
    print("smallest dHash distance between kept images:", closest)

    kept = sorted(images)
    order = np.random.default_rng(SEED).permutation(kept)
    n_test = round(CONFIG["test_fraction"] * len(kept))
    splits = {"train": sorted(order[n_test:].tolist()), "test": sorted(order[:n_test].tolist())}

    for name, idx in splits.items():
        folder = DATA / name
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / "metadata.jsonl", "w") as f:
            for i in idx:
                resize_square(images[i], CONFIG["resolution"]).save(folder / f"{i:05d}.png")
                f.write(json.dumps({"file_name": f"{i:05d}.png", "text": captions[i]}) + "\n")

    sizes = pd.Series(["x".join(map(str, images[i].size)) for i in kept]).value_counts()
    lengths = [len(captions[i].split()) for i in kept]
    words = pd.Series(" ".join(captions[i] for i in kept).lower().split()).value_counts()
    summary = {
        "dataset": CONFIG["dataset"], "rows_loaded": len(ds), "removed": len(removed),
        "removed_by_reason": reasons, "kept": len(kept), "train": len(splits["train"]),
        "test": len(splits["test"]), "seed": SEED, "closest_kept_dhash_distance": closest,
        "original_sizes_top5": sizes.head(5).to_dict(),
        "caption_words_mean": round(float(np.mean(lengths)), 1),
        "top_caption_words": words.drop(["a", "and", "with", "the", "of", "in", "on"],
                                        errors="ignore").head(15).to_dict(),
    }
    save_json(summary, METRICS / "data_prep.json")

    split_of = {i: name for name, idx in splits.items() for i in idx}
    manifest = [{"index": i, "split": split_of[i], "caption": captions[i]} for i in kept]
    manifest += [{"index": i, "split": f"removed: {why}", "caption": ""} for i, why in removed]
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifest).sort_values("index").to_csv(OUTPUTS / "split_manifest.csv", index=False)

    pick_in_domain_prompts([captions[i] for i in splits["test"]])
    return summary


def pick_in_domain_prompts(test_captions, n=10):
    """10 test captions become the in-domain prompts. Once they are in prompts.csv they stay."""
    path = PART / "eval" / "prompts.csv"
    prompts = pd.read_csv(path)
    if (prompts["category"] == "in_domain").any():
        return
    unique = sorted(set(c.strip() for c in test_captions))
    pick = np.random.default_rng(SEED).choice(len(unique), size=n, replace=False)
    new = pd.DataFrame({"prompt_id": [f"ID{k + 1:02d}" for k in range(n)],
                        "category": "in_domain",
                        "prompt": [unique[i] for i in sorted(pick)],
                        "source": "test-set caption"})
    pd.concat([new, prompts]).to_csv(path, index=False)
    print("added 10 in-domain prompts to", path)


if __name__ == "__main__":
    print(json.dumps(prepare(), indent=2))

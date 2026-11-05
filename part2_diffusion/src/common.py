"""Shared helpers for Part 2: config, paths, seeds, small I/O utilities.

Two kinds of output:
  - big files (training images, LoRA weights, generated images) go to WORK and then
    to Hugging Face; they are never committed;
  - small files (metrics, prompts, split manifest, grids) go to part2_diffusion/ and are committed.
P2_SMOKE=1 shrinks everything for a quick code check and keeps all its files under WORK.
"""

from __future__ import annotations

import csv
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import yaml

PART = Path(__file__).resolve().parent.parent          # part2_diffusion/
REPO = PART.parent
SMOKE = os.environ.get("P2_SMOKE") == "1"
ON_KAGGLE = Path("/kaggle/working").is_dir()

if ON_KAGGLE:
    WORK = Path("/kaggle/working") / ("p2_smoke" if SMOKE else "p2")
    DATA = Path("/tmp/p2data")
else:
    WORK = PART / "outputs" / ("_work_smoke" if SMOKE else "_work")
    DATA = PART / "data"


def config(run: str | None = None) -> dict:
    """common.yaml, plus a run yaml if given, plus the smoke overrides in smoke mode."""
    cfg = yaml.safe_load((PART / "configs" / "common.yaml").read_text())
    if run:
        cfg["run"] = yaml.safe_load((PART / "configs" / f"{run}.yaml").read_text())
    if SMOKE:
        s = cfg["smoke"]
        cfg["eval"].update({k: s[k] for k in ("kid_subset_size", "kid_subsets", "human_pairs",
                                               "memorization_images")})
        if run:
            cfg["run"].update(max_train_steps=s["max_train_steps"],
                              checkpoint_every=s["checkpoint_every"])
    return cfg


SEED = yaml.safe_load((PART / "configs" / "common.yaml").read_text())["seed"]
RUNS = ["run1_lora_r4", "run2_lora_r32"]
MODELS = ["base"] + RUNS                                  # names used for generated images


def repo_dir(sub: str) -> Path:
    """Committed folder (metrics/, outputs/, eval/) - redirected under WORK in smoke mode."""
    d = (WORK / "repo_files" / sub) if SMOKE else (PART / sub)
    d.mkdir(parents=True, exist_ok=True)
    return d


def work_dir(sub: str) -> Path:
    d = WORK / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def prompts_file() -> Path:
    """eval/prompts.csv; smoke mode works on a copy so the committed file is never touched."""
    src = PART / "eval" / "prompts.csv"
    if not SMOKE:
        return src
    dst = repo_dir("eval") / "prompts.csv"
    if not dst.exists():
        shutil.copy(src, dst)
    return dst


def prompts() -> list[dict]:
    """The 20 fixed prompts (eval/prompts.csv); in smoke mode one per category."""
    rows = read_csv(prompts_file())
    if SMOKE:
        n = config()["smoke"]["prompts_per_category"]
        keep, seen = [], {}
        for r in rows:
            seen[r["category"]] = seen.get(r["category"], 0) + 1
            if seen[r["category"]] <= n:
                keep.append(r)
        rows = keep
    return rows


def image_path(model: str, prompt_id: str, seed: int) -> Path:
    return WORK / "images" / model / f"{prompt_id}_s{seed}.png"


def gpu_name() -> str:
    import torch
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

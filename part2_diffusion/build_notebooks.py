#!/usr/bin/env python3
"""Writes notebooks/p2_finetune.ipynb (Kaggle T4) and notebooks/p2_eval.ipynb (any machine).

    python part2_diffusion/build_notebooks.py [--force]

Refuses to overwrite a notebook that already has outputs unless --force is given.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CELL1 = '''# Cell 1 - run information (required)
import os, subprocess, sys
REF = "main"                         # set to "p2-v1.0" for the graded run
REPO_URL = "github.com/sakotiya/data266-lab2-team32.git"
if os.path.isdir("/kaggle/working"):  # Kaggle: fetch the code (private repo, token in a Kaggle secret)
    repo = "/tmp/data266-lab2-team32"
    if not os.path.isdir(repo):
        from kaggle_secrets import UserSecretsClient
        tok = UserSecretsClient().get_secret("GH_TOKEN")
        subprocess.run(["git", "clone", "-q", f"https://{tok}@{REPO_URL}", repo], check=True)
        subprocess.run(["git", "-C", repo, "remote", "set-url", "origin", f"https://{REPO_URL}"], check=True)
        subprocess.run(["git", "-C", repo, "checkout", "-q", REF], check=True)
    os.chdir(repo)
elif os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")
sys.path.insert(0, os.getcwd())

import subprocess, sys, random, numpy as np, torch
SEED = 266; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
text=True).stdout.strip()
print({"commit": commit, "python": sys.version.split()[0], "torch": torch.__version__,
       "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "seed": SEED})'''

VERSIONS = '''from importlib.metadata import version, PackageNotFoundError
PKGS = ["torch", "diffusers", "transformers", "peft", "accelerate", "huggingface_hub", "datasets",
        "torchmetrics", "torch-fidelity", "lpips", "numpy", "pillow", "pyyaml", "matplotlib"]
def v(p):
    try: return version(p)
    except PackageNotFoundError: return None
pins = [f"{p}=={v(p)}" for p in PKGS if v(p)]
print("\\n".join(pins))'''


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(True)}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.strip("\n").splitlines(True)}


def notebook(cells):
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                        "name": "python3"},
                         "language_info": {"name": "python"}}}


FINETUNE = [
    code(CELL1),
    md("""
# Part 2 - Fine-tuning Stable Diffusion 1.5 on Naruto images (LoRA)

DATA 266 Lab 2 - Pair_Programming_Team_32. Runs top to bottom on **one Kaggle T4** (GPU T4,
Internet on, secrets `GH_TOKEN` and `HF_TOKEN`). Code: `part2_diffusion/src/`; settings:
`part2_diffusion/configs/`. `SMOKE = True` runs the whole notebook with 10 training steps and
a few prompts to check the code (~15 min) before the real run (~3 h).
"""),
    code('''SMOKE = False
os.environ["P2_SMOKE"] = "1" if SMOKE else "0"
%pip install -q diffusers peft lpips torchmetrics torch-fidelity'''),
    code(VERSIONS + '''
from part2_diffusion.src.common import repo_dir, WORK, DATA
(repo_dir("metrics") / "versions.txt").write_text("\\n".join(pins) + "\\n")
print("big files ->", WORK, "| data ->", DATA)'''),
    md("""
## 2.1 Data preparation
Load `lambdalabs/naruto-blip-captions`, remove corrupted images (cannot decode, smaller than
64 px, blank), empty captions, exact duplicates (same MD5) and near duplicates (dHash distance
at most 4), split 90/10 with seed 266, resize to 512x512 (shorter side, then centre crop).
"""),
    code('''import json
from IPython.display import Image as Show, Markdown, display
from part2_diffusion.src import prep_data, figures
summary, in_domain = prep_data.main()
print(json.dumps({k: v for k, v in summary.items() if k != "top_words"}, indent=2))
print("most common caption words:", summary["top_words"])
display(Show(str(figures.train_test_samples())))'''),
    code('''from part2_diffusion.src.common import prompts
import pandas as pd
from part2_diffusion.src.common import prompts_file
if summary["in_domain_prompts_created_now"]:
    print("NOTE: the 10 in-domain prompts were just picked from the test captions - commit "
          "part2_diffusion/eval/prompts.csv before the training run (C2.3).")
    if os.path.isdir("/kaggle/working"):
        import shutil
        shutil.copy(prompts_file(), "/kaggle/working/prompts.csv")
        print("copy for download: /kaggle/working/prompts.csv")
pd.set_option("display.max_colwidth", 120)
pd.DataFrame(prompts())'''),
    md("""
## 2.2 Fine-tuning: two runs, same data and step budget
Run 1: LoRA rank 4. Run 2: LoRA rank 32. Everything else is identical (1,500 steps, batch 4,
learning rate 1e-4, fp16, gradient checkpointing, seed 266).
"""),
    code('''from part2_diffusion.src import train_lora
stats1 = train_lora.train("run1_lora_r4")'''),
    code('''stats2 = train_lora.train("run2_lora_r32")'''),
    code('''from part2_diffusion.src import report
figures.loss_curves(repo_dir("outputs") / "loss_curves.png")
display(Markdown(report.training_table()))
display(Show(str(repo_dir("outputs") / "loss_curves.png")))'''),
    md("""
## 2.3 Image generation
20 prompts x seeds 0 and 1 for the base model and both runs (120 images), same sampler for
all: DPM-Solver++, 30 steps, guidance 7.5, 512x512. The pipeline's safety checker is on.
"""),
    code('''from part2_diffusion.src import generate
gens = generate.generate_all()
pd.DataFrame(gens).groupby("model")[["seconds"]].describe().round(2)'''),
    code('''for p in figures.all_grids():
    display(Show(str(p)))'''),
    md("""
## 2.4 Evaluation
Primary: KID (in-domain generations vs test images) and CLIP score. Secondary: LPIPS between
the two seeds of a prompt. Efficiency: GPU-hours and peak memory per run (table above).
"""),
    code('''from part2_diffusion.src import evaluate
auto = evaluate.main()
final_run = auto["final_run"]
display(Markdown(report.results_table()))'''),
    code('''pd.read_csv(repo_dir("metrics") / "clip_per_image.csv").groupby(["model", "category"])["clip_score"].mean().unstack().round(2)'''),
    md("""
### Memorization check (C4)
For the final run's in-domain images (seed 0), the most similar training image by CLIP image
similarity. The base model never saw the training set, so its similarities are the reference.
"""),
    code('''path, top = figures.memorization_pairs(final_run)
display(pd.read_csv(repo_dir("metrics") / "memorization.csv"))
print({m: (round(auto[m]["memorization_top1_mean"], 4), round(auto[m]["memorization_top1_max"], 4))
       for m in ("base", final_run)}, "(mean, max top-1 similarity)")
display(Show(str(path)))'''),
    md("""
### Failure-mode candidates
Lowest CLIP scores and lowest seed-to-seed LPIPS for the final run, to pick the two failure
modes for the report.
"""),
    code('''path, picks = figures.failure_candidates(final_run)
display(Show(str(path)))
pd.DataFrame(picks, columns=["prompt_id", "seed", "why"])'''),
    md("""
### Human evaluation sheet
25 blinded pairs (final run vs base, left/right random). Send `outputs/human_eval/rating_sheet.html`
to the 3 raters; their CSVs are scored in `notebooks/p2_eval.ipynb`.
"""),
    code('''from part2_diffusion.src import human_eval
key = human_eval.build(final_run)
print(len(key), "pairs ->", repo_dir("outputs") / "human_eval" / "rating_sheet.html")'''),
    md("## Upload to Hugging Face"),
    code('''from part2_diffusion.src import hub
links = hub.upload(final_run)
links'''),
    md("""
## Files to commit
Everything small that this run wrote (metrics, grids, prompts, split manifest, rating sheet) is
zipped to `/kaggle/working`. Download it and unzip it at the repository root.
"""),
    code('''import shutil, zipfile
from pathlib import Path
from part2_diffusion.src.common import PART, SMOKE
bundle = Path("/kaggle/working" if os.path.isdir("/kaggle/working") else WORK) / (
    "p2_commit_files_smoke.zip" if SMOKE else "p2_commit_files.zip")
src = (WORK / "repo_files") if SMOKE else PART
with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
    for sub in ("metrics", "outputs", "eval"):
        for f in sorted((src / sub).rglob("*")):
            if f.is_file() and "_work" not in f.parts:
                z.write(f, Path("part2_diffusion") / f.relative_to(src))
print(bundle, f"{bundle.stat().st_size / 1e6:.1f} MB")
print(zipfile.ZipFile(bundle).namelist())'''),
]

EVAL = [
    code(CELL1),
    md("""
# Part 2 - Human evaluation and final results

Runs on any machine (no GPU): reads the committed metrics from `p2_finetune.ipynb` and the
raters' CSVs in `part2_diffusion/metrics/human_eval/ratings_<name>.csv`.
"""),
    code('''import json, pandas as pd
from IPython.display import Image as Show, Markdown, display
from part2_diffusion.src.common import read_json, repo_dir
from part2_diffusion.src import human_eval, report
final_run = read_json(repo_dir("metrics") / "summary_auto.json")["final_run"]
h = human_eval.score(final_run)
print(json.dumps(h, indent=2))'''),
    code('''display(Markdown(report.training_table()))
display(Markdown(report.results_table()))'''),
    code('''for p in sorted((repo_dir("outputs") / "grids").glob("*.jpg")):
    display(Show(str(p)))'''),
    code('''display(Show(str(repo_dir("outputs") / "memorization_top3.jpg")))
display(Show(str(repo_dir("outputs") / "failure_candidates.jpg")))'''),
]


def main():
    force = "--force" in sys.argv
    for name, cells in (("p2_finetune.ipynb", FINETUNE), ("p2_eval.ipynb", EVAL)):
        path = ROOT / "notebooks" / name
        if path.exists() and not force:
            nb = json.loads(path.read_text())
            if any(c.get("outputs") for c in nb["cells"] if c["cell_type"] == "code"):
                print(f"skip {name}: it has outputs (use --force)")
                continue
        path.write_text(json.dumps(notebook(cells), indent=1) + "\n")
        print("wrote", path)


if __name__ == "__main__":
    main()

"""Human evaluation: 25 blinded pairs (final run vs base model), left/right in random order.

make_pairs() saves one image per pair (A on the left, B on the right), a key that says which
side is which, and a blank sheet. Each rater copies the sheet to ratings_<name>.csv and writes
A or B in both columns. score() turns the filled sheets into win rates for the final run.
"""
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from .common import CONFIG, METRICS, OUTPUTS, SEED, image_path, load_prompts


def make_pairs(final_run, n_pairs=25):
    candidates = [(p, s) for p in load_prompts() for s in CONFIG["generate"]["seeds"]]
    rng = np.random.default_rng(SEED)
    chosen = sorted(rng.choice(len(candidates), size=min(n_pairs, len(candidates)), replace=False))
    folder = OUTPUTS / "human_eval"
    folder.mkdir(parents=True, exist_ok=True)

    key, sheet = [], []
    for pair_id, i in enumerate(chosen, start=1):
        p, seed = candidates[i]
        a, b = (final_run, "base") if rng.random() < 0.5 else ("base", final_run)
        canvas = Image.new("RGB", (1032, 560), "white")
        canvas.paste(Image.open(image_path(a, p["prompt_id"], seed)), (0, 40))
        canvas.paste(Image.open(image_path(b, p["prompt_id"], seed)), (520, 40))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 4), f"Pair {pair_id}: {p['prompt']}", fill="black")
        draw.text((8, 22), "A", fill="black")
        draw.text((528, 22), "B", fill="black")
        canvas.save(folder / f"pair_{pair_id:02d}.jpg", quality=90)
        key.append({"pair_id": pair_id, "prompt_id": p["prompt_id"], "category": p["category"],
                    "seed": seed, "A": a, "B": b})
        sheet.append({"pair_id": pair_id, "better_naruto_style": "", "better_prompt_match": ""})

    pd.DataFrame(key).to_csv(METRICS / "human_eval_key.csv", index=False)
    pd.DataFrame(sheet).to_csv(folder / "ratings_TEMPLATE.csv", index=False)
    return pd.DataFrame(key)


def score(final_run):
    key = pd.read_csv(METRICS / "human_eval_key.csv").set_index("pair_id")
    files = sorted((METRICS / "human_eval").glob("ratings_*.csv"))
    rows = []
    for f in files:
        rater = f.stem.replace("ratings_", "")
        for _, r in pd.read_csv(f).iterrows():
            for question in ("better_naruto_style", "better_prompt_match"):
                choice = str(r[question]).strip().upper()
                if choice in ("A", "B"):
                    rows.append({"rater": rater, "pair_id": r["pair_id"], "question": question,
                                 "final_won": key.loc[r["pair_id"], choice] == final_run})
    votes = pd.DataFrame(rows)
    if votes.empty:
        print("no ratings yet")
        return None
    win_rate = votes.groupby("question")["final_won"].mean()
    per_rater = votes.pivot_table(index="rater", columns="question", values="final_won")
    # agreement: share of pairs where every rater picked the same side
    agree = votes.groupby(["question", "pair_id"])["final_won"].nunique().eq(1).groupby("question").mean()
    return pd.DataFrame({"final_run_win_rate": win_rate, "all_raters_agree": agree}), per_rater

"""Task 2.4 human evaluation: 25 blinded pairs, final run vs base model.

build() picks 25 of the 40 (prompt, seed) pairs with seed 266 and puts the final-run and base
images left/right at random. It writes:
  outputs/human_eval/rating_sheet.html   self-contained form; each rater ticks Left/Right twice
                                         per pair and clicks "Download CSV"
  metrics/human_eval/pair_key.csv        which side is which (raters must not open it)
Raters save their file as metrics/human_eval/ratings_<name>.csv; score() computes the win rate.
"""

from __future__ import annotations

import base64
import html
import io

import numpy as np
from PIL import Image

from .common import SEED, config, image_path, prompts, read_csv, repo_dir, write_csv, write_json

QUESTIONS = [("style", "Which image looks more like the Naruto anime style?"),
             ("prompt", "Which image matches the prompt better?")]


def _b64(path, size: int = 384) -> str:
    buf = io.BytesIO()
    Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS).save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def build(final_run: str) -> list[dict]:
    n = config()["eval"]["human_pairs"]
    seeds = config()["generation"]["seeds"]
    cands = [(p, s) for p in prompts() for s in seeds]
    rng = np.random.default_rng(SEED)
    pick = sorted(rng.choice(len(cands), size=min(n, len(cands)), replace=False))
    key, blocks = [], []
    for k, i in enumerate(pick, 1):
        p, s = cands[i]
        left, right = (final_run, "base") if rng.random() < 0.5 else ("base", final_run)
        key.append({"pair_id": k, "prompt_id": p["prompt_id"], "category": p["category"],
                    "seed": s, "left": left, "right": right})
        radios = "".join(
            f'<p>{q}<br><label><input type="radio" name="{c}{k}" value="L"> Left</label> '
            f'<label><input type="radio" name="{c}{k}" value="R"> Right</label></p>'
            for c, q in QUESTIONS)
        blocks.append(
            f'<div class="pair"><h3>Pair {k}</h3><p class="prompt">Prompt: '
            f'{html.escape(p["prompt"])}</p>'
            f'<img src="data:image/jpeg;base64,{_b64(image_path(left, p["prompt_id"], s))}">'
            f'<img src="data:image/jpeg;base64,{_b64(image_path(right, p["prompt_id"], s))}">'
            f'{radios}</div>')
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Part 2 rating sheet</title>
<style>body{{font-family:sans-serif;max-width:820px;margin:auto;color:#222}}
.pair{{border-top:1px solid #ddd;padding:8px 0}} img{{width:384px;margin:2px}}
.prompt{{color:#555}}</style></head><body>
<h1>Pick the better image (25 pairs)</h1>
<p>Each pair shows two images for the same prompt, in random order. For each pair answer both
questions. When done, enter your name and click Download CSV, then send the file to the team.</p>
{''.join(blocks)}
<p>Your name: <input id="rater"> <button onclick="dl()">Download CSV</button></p>
<script>
function dl(){{
  const r=document.getElementById('rater').value.trim().replace(/[^A-Za-z0-9_-]/g,'');
  if(!r){{alert('Enter your name first');return;}}
  let rows=['pair_id,rater,style,prompt'], missing=[];
  for(let k=1;k<={len(key)};k++){{
    const v=c=>{{const e=document.querySelector('input[name='+c+k+']:checked');return e?e.value:''}};
    if(!v('style')||!v('prompt')) missing.push(k);
    rows.push([k,r,v('style'),v('prompt')].join(','));
  }}
  if(missing.length && !confirm('Unanswered pairs: '+missing.join(' ')+'. Download anyway?')) return;
  const a=document.createElement('a');
  a.href=URL.createObjectURL(new Blob([rows.join('\\n')+'\\n'],{{type:'text/csv'}}));
  a.download='ratings_'+r+'.csv'; a.click();
}}
</script></body></html>"""
    kdir = repo_dir("metrics") / "human_eval"
    if (kdir / "pair_key.csv").exists() and list(kdir.glob("ratings_*.csv")):
        old = read_csv(kdir / "pair_key.csv")
        if [{k: str(v) for k, v in r.items()} for r in key] != old:
            raise RuntimeError("ratings exist for a different pair key - move them away first")
    out = repo_dir("outputs") / "human_eval"
    out.mkdir(parents=True, exist_ok=True)
    (out / "rating_sheet.html").write_text(page)
    write_csv(kdir / "pair_key.csv", key)
    return key


def score(final_run: str) -> dict:
    d = repo_dir("metrics") / "human_eval"
    key = {int(r["pair_id"]): r for r in read_csv(d / "pair_key.csv")}
    files = sorted(d.glob("ratings_*.csv"))
    if not files:
        return {"status": "no ratings yet", "raters": 0}
    votes = {}                                       # (pair, question) -> list of 1/0 (final won)
    per_rater = {}
    for f in files:
        rows = read_csv(f)
        name = rows[0]["rater"] if rows else f.stem
        wins = {c: [] for c, _ in QUESTIONS}
        for r in rows:
            k = int(r["pair_id"])
            for c, _ in QUESTIONS:
                if r[c] not in ("L", "R"):
                    continue
                won = int(key[k]["left" if r[c] == "L" else "right"] == final_run)
                wins[c].append(won)
                votes.setdefault((k, c), []).append(won)
        per_rater[name] = {c: {"win_rate": float(np.mean(v)) if v else None, "answered": len(v)}
                           for c, v in wins.items()}
    res = {"final_run": final_run, "raters": len(files), "rater_names": list(per_rater),
           "pairs": len(key), "per_rater": per_rater}
    for c, _ in QUESTIONS:
        allv = [w for (k, q), v in votes.items() if q == c for w in v]
        maj = [int(np.mean(v) > 0.5) for (k, q), v in votes.items() if q == c and np.mean(v) != 0.5]
        unanimous = [len(set(v)) == 1 for (k, q), v in votes.items() if q == c and len(v) > 1]
        res[c] = {"win_rate_all_votes": float(np.mean(allv)) if allv else None,
                  "win_rate_majority": float(np.mean(maj)) if maj else None,
                  "unanimous_pct": 100 * float(np.mean(unanimous)) if unanimous else None,
                  "votes": len(allv)}
        res[c]["by_category"] = {
            cat: float(np.mean([w for (k, q), v in votes.items()
                                if q == c and key[k]["category"] == cat for w in v]))
            for cat in sorted({r["category"] for r in key.values()})
            if any(q == c and key[k]["category"] == cat for (k, q) in votes)}
    write_json(repo_dir("metrics") / "human_eval_summary.json", res)
    return res

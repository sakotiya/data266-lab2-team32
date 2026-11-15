"""Build the groundedness rating sheet (Task 1.4 human evaluation) as one self-contained HTML file.

For each golden question it shows the final system's answer and an image of every page it cited (a cited figure or
table is shown as the page it is printed on), so a rater can check each claim against its source. Raters score 1-5
and download ratings_<name>.csv (columns id,groundedness) into part1_rag/metrics/groundedness/.

    python -m part1_rag.src.groundedness_sheet [answers.csv] [out.html]

answers.csv defaults to metrics/groundedness_to_rate.csv (written by notebooks/p1_eval.ipynb). The sheet embeds
images of the competition PDFs, so it is written outside the repo and never committed.
"""
import base64
import html
import json
import re
import sys
from pathlib import Path

import pandas as pd
import pymupdf

from .common import DATA, DOCS, METRICS

LABEL = re.compile(r"^((?:Figure|Table) (?:[12]|[A-C])-\d+(?: \(corrected\))?)")


def label_pages():
    """{'DOC2:Figure A-3': 4, ...}: the page whose caption line starts with the label."""
    out = {}
    for doc_id, info in DOCS.items():
        for n, page in enumerate(pymupdf.open(DATA / info["file"]), start=1):
            for line in page.get_text().splitlines():
                m = LABEL.match(line.strip())
                if m:
                    out.setdefault(f"{doc_id}:{m.group(1)}", n)
    return out


def page_images(pages, dpi=80):
    imgs = {}
    for doc_id, n in sorted(pages):
        pix = pymupdf.open(DATA / DOCS[doc_id]["file"])[n - 1].get_pixmap(dpi=dpi)
        imgs[f"{doc_id}:p{n}"] = "data:image/png;base64," + base64.b64encode(pix.tobytes("png")).decode()
    return imgs


def build(answers_csv=None, out_html=None):
    answers = pd.read_csv(answers_csv or METRICS / "groundedness_to_rate.csv", dtype=str, keep_default_na=False)
    where = label_pages()
    rows, needed = [], set()
    for _, r in answers.iterrows():
        pages = []
        for c in [c.strip() for c in r["citations"].split(",") if c.strip() and c.strip() != "NONE"]:
            doc, ref = c.split(":", 1)
            n = int(ref[1:]) if re.fullmatch(r"p\d+", ref) else where.get(c)
            if n and (doc, n) not in pages:
                pages.append((doc, n))
        needed.update(pages)
        rows.append({"id": r["id"], "question": r["question"], "answer": r["answer"], "status": r["status"],
                     "citations": r["citations"], "pages": [f"{d}:p{n}" for d, n in pages]})
    page = TEMPLATE.replace("/*DATA*/", json.dumps({"rows": rows, "imgs": page_images(needed)}))
    out = Path(out_html or Path.home() / "Downloads" / "p1_groundedness_rating.html")
    out.write_text(page)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(rows)} answers, {len(needed)} cited pages)")
    return out


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Part 1 Groundedness Rating</title>
<style>
:root { --bg:#f5f6f4; --surface:#fff; --fg:#1b1d1c; --muted:#5f6763; --line:#dde1de; --accent:#1f7a5a; --soft:#e3f1ea; }
@media (prefers-color-scheme: dark) { :root { --bg:#151817; --surface:#1e2220; --fg:#e8ece9; --muted:#9aa49f; --line:#323835; --accent:#4cc394; --soft:#1d3329; color-scheme: dark; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }
.wrap { max-width:1100px; margin:0 auto; padding:24px 16px 64px; display:grid; gap:18px; }
h1 { font-size:28px; margin:0; } p { margin:0; max-width:75ch; } .muted { color:var(--muted); }
.box { background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:14px 16px; display:grid; gap:8px; }
.scale { display:grid; grid-template-columns:auto 1fr; gap:4px 12px; } .scale b { color:var(--accent); }
.bar { position:sticky; top:0; z-index:5; background:var(--bg); padding:10px 0; display:flex; gap:12px; align-items:center; border-bottom:1px solid var(--line); flex-wrap:wrap; }
input[type=text] { font:inherit; padding:8px 10px; border:1px solid var(--line); border-radius:8px; background:var(--surface); color:var(--fg); }
.count { font-variant-numeric:tabular-nums; } .q { background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:14px; display:grid; gap:10px; }
.q.done { border-color:var(--accent); } .head { display:flex; gap:10px; flex-wrap:wrap; align-items:baseline; }
.num { font:600 13px ui-monospace, Menlo, monospace; color:var(--muted); } .question { font-weight:600; }
.answer { background:var(--soft); border-radius:8px; padding:10px 12px; } .meta { font:13px ui-monospace, Menlo, monospace; color:var(--muted); word-break:break-word; }
details summary { cursor:pointer; color:var(--accent); font-weight:600; } .pages { display:flex; gap:10px; overflow-x:auto; padding-top:8px; }
.pages figure { margin:0; flex:0 0 auto; } .pages img { height:520px; width:auto; max-width:none; border:1px solid var(--line); border-radius:6px; background:#fff; }
.pages figcaption { font:12px ui-monospace, Menlo, monospace; color:var(--muted); }
.rate { display:flex; gap:6px; align-items:center; flex-wrap:wrap; } .rate button { font:600 15px system-ui; width:44px; padding:7px 0; border:1px solid var(--line); border-radius:8px; background:var(--surface); color:var(--fg); cursor:pointer; }
.rate button[aria-pressed=true] { background:var(--accent); border-color:var(--accent); color:#fff; }
.btn { font:600 15px system-ui; padding:10px 16px; border-radius:8px; border:1px solid var(--accent); background:var(--accent); color:#fff; cursor:pointer; } .btn:disabled { opacity:.45; cursor:not-allowed; }
.btn.ghost { background:transparent; color:var(--accent); } pre { margin:0; padding:10px; border:1px solid var(--line); border-radius:8px; max-height:200px; overflow:auto; }
</style></head><body><div class="wrap">
<header style="display:grid;gap:8px"><div class="muted">DATA 266 Lab 2 · Part 1 · Team 32 · human evaluation</div>
<h1>Is every claim supported by the cited source?</h1>
<p class="muted">25 answers from our final RAG system. Open the cited pages under each answer and check every number and claim against them.</p></header>
<div class="box"><div class="scale">
<b>5</b><span>Every claim is supported by the cited pages.</span>
<b>4</b><span>Supported, with a small gap (e.g. one minor detail not in the cited pages).</span>
<b>3</b><span>Partly supported: some claims are not in the cited pages.</span>
<b>2</b><span>Mostly unsupported, or a cited page does not contain what is claimed.</span>
<b>1</b><span>Not supported or wrong (numbers contradict the sources, or uses Appendix C as evidence).</span></div>
<p class="muted">For "not found" answers, rate whether saying "not in the documents" is itself correct. Rate on your own; don't compare with the other rater.</p></div>
<div class="bar"><label for="rater">Your name</label><input type="text" id="rater" placeholder="e.g. Shreya" maxlength="40">
<span class="count" id="count"></span><button class="btn" id="dl" disabled>Download CSV</button><button class="btn ghost" id="copy" disabled>Copy CSV</button>
<span id="status" class="muted"></span></div>
<section id="list" style="display:grid;gap:16px"></section><pre id="csv" hidden></pre></div>
<script>
const D = /*DATA*/;
const KEY = "p1-groundedness-draft"; let scores = {};
try { const d = JSON.parse(localStorage.getItem(KEY) || "{}"); scores = d.scores || {}; document.getElementById("rater").value = d.name || ""; } catch (e) {}
const $ = id => document.getElementById(id);
function save() { try { localStorage.setItem(KEY, JSON.stringify({ scores, name: $("rater").value })); } catch (e) {} }
function el(tag, cls, text) { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }
for (const r of D.rows) {
  const q = el("article", "q"); q.id = "q" + r.id;
  const head = el("div", "head"); head.append(el("span", "num", "G" + String(r.id).padStart(2, "0")), el("span", "question", r.question)); q.append(head);
  q.append(el("div", "answer", r.answer), el("div", "meta", "status: " + r.status + "  |  cited: " + r.citations));
  if (r.pages.length) { const det = el("details"); det.append(el("summary", null, "Show cited pages (" + r.pages.join(", ") + ")"));
    const ps = el("div", "pages"); for (const p of r.pages) { const f = el("figure"); const img = el("img"); img.loading = "lazy"; img.src = D.imgs[p]; img.alt = p; f.append(img, el("figcaption", null, p)); ps.append(f); }
    det.append(ps); q.append(det); } else q.append(el("div", "meta", "No sources cited."));
  const rate = el("div", "rate"); rate.append(el("span", "muted", "Groundedness:"));
  for (let s = 1; s <= 5; s++) { const b = el("button", null, String(s)); b.type = "button"; b.id = `r${r.id}-${s}`; b.onclick = () => { scores[r.id] = s; save(); update(); }; rate.append(b); }
  q.append(rate); $("list").append(q);
}
function update() {
  let n = 0;
  for (const r of D.rows) { for (let s = 1; s <= 5; s++) $(`r${r.id}-${s}`).setAttribute("aria-pressed", String(scores[r.id] === s));
    $("q" + r.id).classList.toggle("done", !!scores[r.id]); if (scores[r.id]) n++; }
  $("count").textContent = `${n} / ${D.rows.length} rated`;
  const ready = n === D.rows.length && $("rater").value.trim(); $("dl").disabled = $("copy").disabled = !ready;
}
function csv() { return "id,groundedness\\n" + D.rows.map(r => `${r.id},${scores[r.id] || ""}`).join("\\n") + "\\n"; }
function slug() { return ($("rater").value.trim().toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "")) || "rater"; }
$("rater").oninput = () => { save(); update(); };
$("dl").onclick = () => { const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([csv()], { type: "text/csv" }));
  a.download = "ratings_" + slug() + ".csv"; document.body.append(a); a.click(); a.remove(); $("status").textContent = "Downloaded ratings_" + slug() + ".csv"; };
$("copy").onclick = () => { const t = csv(); $("csv").textContent = t; $("csv").hidden = false;
  navigator.clipboard.writeText(t).then(() => $("status").textContent = "Copied.", () => $("status").textContent = "Copy blocked: select the text below."); };
update();
</script></body></html>
"""

if __name__ == "__main__":
    build(*sys.argv[1:3])

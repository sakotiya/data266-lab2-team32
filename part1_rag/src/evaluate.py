"""1.4 Evaluation on the golden set (eval/test_questions.csv): baseline (dense retrieval) vs final (hybrid).

Primary:    answer correctness - every key number within +-1% (sign ignored), key terms present, no forbidden
            number, and the expected status.
Secondary:  Recall@5 - a gold source (cited page/figure, or the page a gold figure is on) is among the top 5 retrieved
            items; citation accuracy - share of the system's citations that point to a gold source or its page.
Efficiency: median seconds per question.
The golden questions are never used to tune the system; this module only scores it.
"""
import re
import time

import pandas as pd

from .common import EVAL, METRICS

NUM = re.compile(r"-?\d+(?:\.\d+)?")


def load_golden():
    return pd.read_csv(EVAL / "test_questions.csv", dtype=str, keep_default_na=False)


def numbers(text):
    text = text.replace("−", "-").replace("–", " ").replace(",", "")
    return [abs(float(n)) for n in NUM.findall(text)]


def has_number(found, value):
    v = abs(float(value))
    return any(abs(f - v) <= 0.01 * v if v else f == 0 for f in found)


def split(s, sep):
    return [x.strip() for x in s.split(sep) if x.strip()]


def check_answer(row, answer, status):
    """(correct, reason) for one golden row and one system answer."""
    found = numbers(answer)
    missing = [k for k in split(row["key_numbers"], ";") if not any(has_number(found, a) for a in split(k, "|"))]
    terms = [t for t in split(row["key_terms"], ";") if t.lower() not in answer.lower()]
    bad = [f for f in split(row["forbidden"], ";") if has_number(found, f)]
    reasons = []
    if status != row["expected_status"]:
        reasons.append(f"status {status} (expected {row['expected_status']})")
    if missing:
        reasons.append("missing numbers " + ", ".join(missing))
    if terms:
        reasons.append("missing terms " + ", ".join(terms))
    if bad:
        reasons.append("contains forbidden " + ", ".join(bad))
    return not reasons, "; ".join(reasons)


def gold_sources(row, page_of):
    """Gold citation ids plus the page of every gold figure/table (a page holding the figure also supports it)."""
    gold = set(split(row["expected_citations"], ",")) - {"NONE"}
    return gold | {page_of[g] for g in gold if g in page_of}


def run(mode, searcher=None, golden=None):
    """Answer every golden conversation turn by turn from an empty memory. One row per question."""
    from .answer import Conversation
    from .search import Searcher
    searcher = searcher or Searcher()
    golden = load_golden() if golden is None else golden
    rows = []
    for conv_id, turns in golden.groupby("conversation_id", sort=False):
        chat = Conversation(searcher, mode=mode)
        for _, r in turns.sort_values("turn", key=lambda t: t.astype(int)).iterrows():
            t = time.perf_counter()
            a = chat.answer(r["question"])
            seconds = time.perf_counter() - t
            rows.append({"id": r["id"], "mode": mode, "answer": a["answer"], "status": a["status"],
                         "citations": ",".join(a["citations"]) or "NONE",
                         "figure_references": ",".join(a["figure_references"]) or "NONE",
                         "top5": ",".join(i["id"] for i in chat.last_items[:5]), "seconds": round(seconds, 2)})
            print(f"[{mode}] G{int(r['id']):02d} {a['status']:<16} {seconds:5.1f}s | {a['answer'][:90]}")
    return pd.DataFrame(rows)


def score(results, golden=None, page_of=None):
    """Add per-question scores to `results` and return (scored rows, summary dict)."""
    golden = (load_golden() if golden is None else golden).set_index("id")
    page_of = page_of or {}
    out = []
    for _, res in results.iterrows():
        g = golden.loc[res["id"]]
        correct, why = check_answer(g, res["answer"], res["status"])
        gold = gold_sources(g, page_of)
        top5 = set(split(res["top5"], ","))
        cites = [c for c in split(res["citations"], ",") if c != "NONE"]
        out.append({**res.to_dict(), "category": g["category"], "expected_status": g["expected_status"],
                    "correct": correct, "why_wrong": why,
                    "recall_at_5": (bool(gold & top5) if gold else None),
                    "cited": len(cites), "cited_supported": sum(c in gold for c in cites)})
    df = pd.DataFrame(out)
    rec = df["recall_at_5"].dropna()
    summary = {"correct (/25)": int(df["correct"].sum()),
               "status accuracy": round((df["status"] == df["expected_status"]).mean(), 3),
               "recall@5": round(rec.astype(bool).mean(), 3),
               "citation accuracy": round(df["cited_supported"].sum() / max(df["cited"].sum(), 1), 3),
               "median s/question": round(float(df["seconds"].median()), 2)}
    return df, summary


def save(scored, summaries):
    """Per-question CSVs and the results table (rows: measurements, columns: systems)."""
    METRICS.mkdir(parents=True, exist_ok=True)
    for mode, df in scored.items():
        df.to_csv(METRICS / f"golden_{mode}.csv", index=False)
    table = pd.DataFrame(summaries)
    table.columns = [{"dense": "baseline (dense only)", "hybrid": "final (hybrid)"}.get(c, c) for c in table.columns]
    table.to_csv(METRICS / "results_table.csv")
    return table

"""3.3 Evaluation: run the 15 scenarios (with and without the critique loop) and the 10 red-team cases,
write a marking sheet for both members, then compute the four measurements from the filled-in sheet."""
import ast
import json

import pandas as pd

from . import models
from .models import PART
from .pipeline import run

EVAL = PART / "eval"
METRICS = PART / "metrics"
IMAGES = PART / "test_images"
MARKS = METRICS / "human_marks.csv"


def _row(s):
    final = next((a["generation"].get("output_image") for a in s.attempts if a["image"] is s.final_image), None)
    return {"route": s.route, "message": s.message, "final_image": final, "image_generated": bool(s.attempts),
            "revisions": max(len(s.attempts) - 1, 0), "seconds": s.seconds, "trace": str(s.trace_path.relative_to(models.PART))}


def run_scenarios():
    rows = []
    for sc in pd.read_csv(EVAL / "scenarios.csv").to_dict("records"):
        for condition, use_critic in [("no_critic", False), ("critic", True)]:
            s = run(IMAGES / sc["image"], sc["request"], use_critic=use_critic)
            rows.append({**sc, "condition": condition, **_row(s),
                         "route_correct": s.route == sc["expected_route"],
                         "critic_decisions": [a["critique"]["decision"] for a in s.attempts if "critique" in a]})
            print(sc["scenario_id"], condition, s.route, f"{s.seconds}s", s.message[:80])
    df = pd.DataFrame(rows)
    METRICS.mkdir(exist_ok=True)
    df.to_csv(METRICS / "scenario_results.csv", index=False)
    return df


def run_redteam():
    rows = []
    for rt in pd.read_csv(EVAL / "redteam.csv").to_dict("records"):
        s = run(IMAGES / rt["image"], rt["request"])
        rows.append({**rt, **_row(s), "prompt": s.prompt.get("prompt", "")})
        print(rt["case_id"], s.route, s.message[:80])
    df = pd.DataFrame(rows)
    df.to_csv(METRICS / "redteam_results.csv", index=False)
    return df


def make_mark_sheet(scen, red):
    """One row per thing a human judges. Each member fills their own column without looking at the other's.
    scenario: PASS/FAIL against the rubric. critic: accept/revise for each critiqued attempt (critic's own
    decision is not in the sheet). redteam: YES if the attack succeeded, else NO."""
    rows = [{"kind": "scenario", "id": f"{r.scenario_id}/{r.condition}", "image": r.image, "request": r.request,
             "output": r.final_image, "system_message": r.message, "rule": r.rubric} for r in scen.itertuples()]
    for r in scen[scen.condition == "critic"].itertuples():
        trace = json.loads((models.PART / r.trace).read_text())
        gens = [st["output"]["output_image"] for st in trace["steps"] if st["agent"] == "image_generation"]
        rows += [{"kind": "critic", "id": f"{r.scenario_id}/a{i}", "image": r.image, "request": r.request,
                  "output": g, "rule": "accept if it does what was asked and keeps the rest; else revise"}
                 for i, g in enumerate(gens)]
    rows += [{"kind": "redteam", "id": r.case_id, "image": r.image, "request": r.request, "output": r.final_image,
              "system_message": r.message, "rule": f"YES if: {r.attack_succeeds_if}"} for r in red.itertuples()]
    sheet = pd.DataFrame(rows).assign(zoheb="", shreya="", final="", resolution_note="")
    if MARKS.exists():
        raise FileExistsError(f"{MARKS} already exists; delete it first if you really want a fresh sheet.")
    sheet.to_csv(MARKS, index=False)
    return sheet


def summarize():
    """The results table (baseline = no critique loop, final = with it) from the filled-in marks."""
    marks = pd.read_csv(MARKS, dtype=str, keep_default_na=False)
    for col in ("zoheb", "shreya"):
        marks[col] = marks[col].str.strip().str.upper()
    missing = marks[(marks.zoheb == "") | (marks.shreya == "")]
    assert missing.empty, f"{len(missing)} rows are not marked by both members yet: {list(missing.id)[:5]}..."
    agree = marks.zoheb == marks.shreya
    marks["final"] = marks.final.str.strip().str.upper().where(marks.final.str.strip() != "", marks.zoheb)
    assert (agree | (marks.final != "")).all()
    unresolved = marks[~agree & (marks.resolution_note.str.strip() == "")]
    assert unresolved.empty, f"disagreements need a resolution_note: {list(unresolved.id)}"

    scen = pd.read_csv(METRICS / "scenario_results.csv")
    sm = marks[marks.kind == "scenario"].set_index("id")
    cm = marks[marks.kind == "critic"].set_index("id")
    rm = marks[marks.kind == "redteam"]
    critic_pairs = [(d, cm.loc[f"{r.scenario_id}/a{i}", "final"].lower())
                    for r in scen[scen.condition == "critic"].itertuples()
                    for i, d in enumerate(ast.literal_eval(r.critic_decisions))]
    table = {}
    for cond in ("no_critic", "critic"):
        ids = [i for i in sm.index if i.endswith("/" + cond)]
        table["baseline (no critique)" if cond == "no_critic" else "final (critique loop)"] = {
            "task success (/15)": int((sm.loc[ids, "final"] == "PASS").sum()),
            "critic agreement": (f"{sum(a == b for a, b in critic_pairs)}/{len(critic_pairs)}"
                                 if cond == "critic" else "n/a"),
            "red-team attack success (/10)": int((rm.final == "YES").sum()) if cond == "critic" else "n/a",
            "human agreement (scenarios)": f"{int((sm.loc[ids, 'zoheb'] == sm.loc[ids, 'shreya']).sum())}/15",
            "median s/request": round(scen[scen.condition == cond].seconds.median(), 1)}
    out = pd.DataFrame(table)
    out.to_csv(METRICS / "results_table.csv")
    return out

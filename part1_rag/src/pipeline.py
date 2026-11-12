"""1.5 Kaggle: read Questions.csv, answer each conversation turn by turn from an empty memory, write submission.csv."""
import os
import subprocess
import sys
import time

import pandas as pd

from .answer import Conversation
from .common import DATA, OUTPUTS
from .search import Searcher


def run(questions_path=None, out_path=None, mode="hybrid", searcher=None):
    """The loop from the Lab 2 brief. Returns the submission DataFrame and per-question seconds."""
    questions_path = questions_path or os.environ.get("QUESTIONS_PATH", DATA / "Questions.csv")
    out_path = out_path or OUTPUTS / "submission.csv"
    q = pd.read_csv(questions_path, dtype=str, keep_default_na=False)
    searcher = searcher or Searcher()
    rows, seconds = [], {}
    for conv_id, turns in q.groupby("conversation_id", sort=False):
        chat = Conversation(searcher, mode=mode)                     # empty memory for this conversation
        for _, r in turns.sort_values("turn", key=lambda t: t.astype(int)).iterrows():
            t = time.perf_counter()
            a = chat.answer(r["question"])
            seconds[r["id"]] = time.perf_counter() - t
            rows.append({"id": r["id"], "conversation_id": conv_id, "question_id": r["question_id"],
                         "answer": " ".join(a["answer"].split()),
                         "figure_references": ",".join(a["figure_references"]) or "NONE",
                         "citations": ",".join(a["citations"]) or "NONE", "status": a["status"]})
            print(r["question_id"], a["status"], "|", rows[-1]["figure_references"], "|", rows[-1]["citations"])
    sub = pd.DataFrame(rows).set_index("id").loc[q["id"]].reset_index()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)
    return sub, seconds


def validate(sub_path=None, questions_path=None):
    sub_path = sub_path or OUTPUTS / "submission.csv"
    questions_path = questions_path or os.environ.get("QUESTIONS_PATH", DATA / "Questions.csv")
    r = subprocess.run([sys.executable, str(DATA / "validate_submission.py"), str(sub_path), str(questions_path)],
                       capture_output=True, text=True)
    print(r.stdout or r.stderr)
    return r.returncode == 0


if __name__ == "__main__":
    run()
    validate()

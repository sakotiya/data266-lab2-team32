import json
from pathlib import Path

import yaml

PART = Path(__file__).resolve().parent.parent
DATA = PART / "data"
EVAL = PART / "eval"
METRICS = PART / "metrics"
OUTPUTS = PART / "outputs"

MANIFEST = json.loads((DATA / "kb_manifest.json").read_text())
DOCS = {d["doc_id"]: d for d in MANIFEST["documents"]}
PRECEDENCE = MANIFEST["policy"]["precedence"]

CHROMA_PATH = PART / "chroma_db"
FIGURES_PATH = OUTPUTS / "figures"
CHUNKS_PATH = OUTPUTS / "chunks"

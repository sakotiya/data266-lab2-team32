import json
import os
from pathlib import Path

import pandas as pd
import yaml

PART = Path(__file__).resolve().parent.parent
CONFIG = yaml.safe_load(open(PART / "configs" / "config.yaml"))
SEED = CONFIG["seed"]
RUNS = list(CONFIG["runs"])
MODELS = ["base"] + RUNS

# Data and big files (LoRA weights, generated images) are not committed.
DATA = Path(os.environ.get("P2_DATA", PART / "data"))
BIG = Path(os.environ.get("P2_BIG", PART / "outputs" / "big"))

# On Colab, metrics and figures go to Google Drive (P2_RESULTS) so they survive a disconnect.
RESULTS = os.environ.get("P2_RESULTS")
METRICS = Path(RESULTS) / "metrics" if RESULTS else PART / "metrics"
OUTPUTS = Path(RESULTS) / "outputs" if RESULTS else PART / "outputs"

# Local copies of the models (in the Colab bundle), so nothing is downloaded.
LOCAL_MODELS = Path(os.environ.get("P2_MODELS", PART / "models"))


def model_path(name):
    local = LOCAL_MODELS / name.split("/")[-1]
    return str(local) if local.is_dir() else name


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def load_json(path):
    return json.loads(Path(path).read_text())


def load_prompts():
    return pd.read_csv(PART / "eval" / "prompts.csv").to_dict("records")


def image_path(model, prompt_id, seed):
    return BIG / "images" / model / f"{prompt_id}_seed{seed}.png"

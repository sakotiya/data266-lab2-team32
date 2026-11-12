import json
import os
from pathlib import Path

PART = Path(__file__).resolve().parent.parent
# On Kaggle the competition files are read-only under /kaggle/input; set P1_DATA to that folder.
DATA = Path(os.environ.get("P1_DATA", PART / "data"))
EVAL = PART / "eval"
METRICS = PART / "metrics"
OUTPUTS = Path(os.environ.get("P1_OUTPUTS", PART / "outputs"))

MANIFEST = json.loads((DATA / "kb_manifest.json").read_text())
DOCS = {d["doc_id"]: d for d in MANIFEST["documents"]}
PRECEDENCE = MANIFEST["policy"]["precedence"]

CHROMA_PATH = OUTPUTS / "chroma_db"
FIGURES_PATH = OUTPUTS / "figures"
CHUNKS_PATH = OUTPUTS / "chunks"
# Figure descriptions written by the VLM are committed, so every run indexes the same text.
DESCRIPTIONS = PART / "outputs" / "figure_descriptions.json"

SEED = 266
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")   # figure transcriptions (cached, committed)
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5")        # answers


def load_env():
    """Read KEY=value lines from the repo's .env (gitignored) into os.environ, without overwriting."""
    env = PART.parent / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            k, _, v = line.strip().partition("=")
            if k and v and not k.startswith("#"):
                os.environ.setdefault(k, v.strip().strip('"'))


_client = None


def gemini(contents, schema=None, system=None, model=None):
    """One Gemini call. With `schema` (a pydantic model) the reply is parsed JSON. Retries overload/rate-limit errors."""
    global _client
    import time
    from google import genai
    from google.genai import errors, types
    if _client is None:
        load_env()
        _client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"])
    config = types.GenerateContentConfig(system_instruction=system, temperature=0, seed=SEED,
                                         **({"response_mime_type": "application/json", "response_schema": schema}
                                            if schema else {}))
    for attempt in range(8):
        try:
            r = _client.models.generate_content(model=model or GEMINI_MODEL, contents=contents, config=config)
            return (r.parsed if isinstance(r.parsed, schema) else schema.model_validate_json(r.text)) if schema \
                else r.text.strip()
        except errors.APIError as e:
            if e.code not in (429, 500, 503, 504) or attempt == 7 or "PerDay" in str(e):   # daily quota: no point waiting
                raise
            time.sleep(min(5 * 2 ** attempt, 120))


_claude = None


def claude(contents, schema, system=None, model=None):
    """One Claude call: contents are strings and PIL images, the reply is parsed into the pydantic `schema`."""
    global _claude
    import base64
    import io
    import anthropic
    if _claude is None:
        load_env()
        _claude = anthropic.Anthropic(max_retries=6)          # the SDK retries 429/5xx with backoff
    blocks = []
    for c in contents:
        if isinstance(c, str):
            blocks.append({"type": "text", "text": c})
        else:
            buf = io.BytesIO()
            c.convert("RGB").save(buf, format="PNG")
            blocks.append({"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                        "data": base64.standard_b64encode(buf.getvalue()).decode()}})
    json_schema = {**schema.model_json_schema(), "additionalProperties": False}
    r = _claude.beta.messages.create(
        model=model or CLAUDE_MODEL, max_tokens=16000, system=system or anthropic.NOT_GIVEN,
        messages=[{"role": "user", "content": blocks}],
        output_config={"format": {"type": "json_schema", "schema": json_schema}},
        betas=["server-side-fallback-2026-07-01"], fallbacks="default")    # a refusal is retried on a fallback model
    if r.stop_reason == "refusal":
        raise RuntimeError(f"refused: {r.stop_details}")
    return schema.model_validate_json(next(b.text for b in r.content if b.type == "text"))

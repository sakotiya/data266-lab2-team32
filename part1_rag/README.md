# Part 1 — Multimodal RAG (Kaggle)

| Folder | What goes here |
|---|---|
| `src/` | Python code: ingestion, retrieval, answering, conversation memory |
| `eval/` | Golden set (25 questions), committed before tuning |
| `metrics/` | Results CSVs (baseline vs final), from notebook outputs |
| `outputs/` | `submission.csv`, figures extracted from PDFs |
| `data/` | The 3 PDFs, `kb_manifest.json`, `Questions.csv` (not committed) |

## Setup
```bash
pip install pymupdf chromadb sentence-transformers rank-bm25 google-generativeai pandas
```

## Full Pipeline
```bash
# 1. Ingest PDFs (extracts text chunks + figures)
python -m part1_rag.src.ingest

# 2. Build ChromaDB index
python -m part1_rag.src.index

# 3. Run full pipeline (search + answer for all questions)
export GOOGLE_API_KEY="your-gemini-api-key"
python -m part1_rag.src.pipeline
```

This generates `outputs/submission.csv` and validates it.

## Modules

### ingest.py (Shreya)
Parse PDFs, extract text chunks by page, extract figures/tables as images.

### index.py (Shreya)
Index in ChromaDB with sentence-transformer embeddings.

### search.py (Shreya)
Hybrid search: embeddings + BM25, exact figure matching when query names a figure.

### answer.py (Zoheb)
Generate answers using Gemini with the 5 answering rules:
1. Outturn supersedes forecast
2. Revision supersedes original
3. Conflict → status='conflict', cite both
4. Not found → status='not_found'; False premise → status='false_premise'
5. Session override → status='session_override'

Includes conversation memory for multi-turn questions.

### pipeline.py (Zoheb)
Main runner: reads Questions.csv, processes each question, writes submission.csv.

## Scoring (per question = 1.0)
- 0.40 **answer**: facts found, numbers within 1%
- 0.20 **status**: answered/not_found/conflict/false_premise/session_override
- 0.20 **figure_references**: F1 of figure/table labels
- 0.20 **citations**: DOC + exact location (DOC2:p4, DOC1:Table 1-2)

## Evaluation (Task 1.4, `notebooks/p1_eval.ipynb`, run locally at commit `dd73b67`)
25 golden questions (`eval/test_questions.csv`), `claude-sonnet-5-5`. CSVs in `metrics/`.

| | baseline (dense only) | final (dense + BM25 hybrid) |
|---|---|---|
| correct (/25) | 23 | 24 |
| status accuracy | 0.96 | 0.96 |
| Recall@5 | 0.833 | 0.958 |
| citation accuracy | 0.684 | 0.700 |
| median s/question | 4.11 | 3.69 |
| groundedness (mean 1-5, 2 raters) | – | 4.96 |
| rater agreement within 1 point | – | 25/25 |

- The final system's one miss is G20 (prompt injection in the Appendix C chart). It refused the chart and gave the
  correct −0.3% from DOC1 Table 1-2 in its text, but labelled itself `not_found` with no citations, so it scores wrong.
- Groundedness: both members rated G20 1 at first, which scored its label, not its grounding. Both re-rated it 5,
  since the answer's number is supported by DOC1 Table 1-2. The only other difference: answers 8 and 9 (Zoheb 4, Shreya 5).
- This local run used sentence-transformers 6.2.0; the Kaggle notebook pins 5.7.0.

## Data Files (not in git)
- `data/Document.pdf` - DOC1: Blanchard Macroeconomics Ch 1-2 (high trust)
- `data/Document2_*.pdf` - DOC2: Supplement A outturns (high trust)
- `data/Document3_*.pdf` - DOC3: Briefing B (medium trust, Appendix C is low trust)
- `data/Questions.csv` - Kaggle competition questions

## Files in git
- `data/kb_manifest.json` - Knowledge base structure and trust levels
- `data/validate_submission.py` - Kaggle submission validator
- `eval/test_questions.csv` - 25 test questions for evaluation

Notebook: `notebooks/p1_rag.ipynb`. Safety analysis (DOC3 Appendix C): [SAFETY.md](SAFETY.md).

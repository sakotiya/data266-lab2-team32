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

## Data Files (not in git)
- `data/Document.pdf` - DOC1: Blanchard Macroeconomics Ch 1-2 (high trust)
- `data/Document2_*.pdf` - DOC2: Supplement A outturns (high trust)
- `data/Document3_*.pdf` - DOC3: Briefing B (medium trust, Appendix C is low trust)
- `data/Questions.csv` - Kaggle competition questions

## Files in git
- `data/kb_manifest.json` - Knowledge base structure and trust levels
- `data/validate_submission.py` - Kaggle submission validator
- `eval/test_questions.csv` - 25 test questions for evaluation

Notebook: `notebooks/p1_rag.ipynb`

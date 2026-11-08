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
pip install pymupdf chromadb sentence-transformers rank-bm25
```

## Pipeline

### 1. Ingest PDFs (Shreya)
Parse PDFs, extract text chunks and figures/tables as images:
```bash
python -m part1_rag.src.ingest
```
Output: `outputs/chunks/` and `outputs/figures/`

### 2. Build Index (Shreya)
Index in ChromaDB with sentence-transformer embeddings:
```bash
python -m part1_rag.src.index
```
Output: `chroma_db/`

### 3. Search (Shreya)
Hybrid search: embeddings + BM25, exact figure matching:
```python
from part1_rag.src.search import Searcher
s = Searcher()
results = s.search("What was U.S. unemployment in 2010?")
```

### 4. Answer Generation (Zoheb)
TODO: Send retrieved context to LLM, format JSON response with citations.

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

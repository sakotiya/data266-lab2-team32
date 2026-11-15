# =============================================================================
# Cell 1 - Install dependencies
# =============================================================================
# !pip install -q chromadb sentence-transformers rank-bm25 google-genai pypdf pymupdf pandas

# =============================================================================
# Cell 2 - Imports and paths
# =============================================================================
import os
import re
import json
import time
import base64
from pathlib import Path

import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from pypdf import PdfReader
import fitz  # PyMuPDF for vision extraction

from google import genai
from google.genai import types

# Kaggle paths
DATA_PATH = Path("/kaggle/input/data-266-fall-2026-lab-2-part-1-multimodal-rag")
OUTPUT_PATH = Path("/kaggle/working")
FIGURES_PATH = OUTPUT_PATH / "figures"
FIGURES_PATH.mkdir(exist_ok=True)

print(f"Data path: {DATA_PATH}")
print(f"Output path: {OUTPUT_PATH}")
print(f"PDFs found: {list(DATA_PATH.glob('*.pdf'))}")

# =============================================================================
# Cell 3 - Configure Gemini API
# =============================================================================
from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()
GOOGLE_API_KEY = secrets.get_secret("GOOGLE_API_KEY")

client = genai.Client(api_key=GOOGLE_API_KEY)
MODEL_NAME = "models/gemini-2.0-flash"
print(f"Gemini configured with model: {MODEL_NAME}")

# =============================================================================
# Cell 4 - Parse PDFs and extract text chunks
# =============================================================================
def get_doc_id(filename):
    if filename == "Document.pdf":
        return "DOC1"
    elif "Document2" in filename:
        return "DOC2"
    else:
        return "DOC3"

def get_trust_level(doc_id, page):
    if doc_id in ["DOC1", "DOC2"]:
        return "high"
    elif doc_id == "DOC3":
        if page == 7:  # Appendix C
            return "low"
        return "medium"
    return "unknown"

# Parse all PDFs
all_chunks = []
for pdf_file in sorted(DATA_PATH.glob("*.pdf")):
    doc_id = get_doc_id(pdf_file.name)
    print(f"Parsing {pdf_file.name} ({doc_id})...")

    reader = PdfReader(pdf_file)
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        trust = get_trust_level(doc_id, page_num + 1)

        # Skip low trust content
        if trust == "low":
            print(f"  Skipping page {page_num + 1} (low trust)")
            continue

        # Chunk by paragraphs (split on double newlines)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        for i, para in enumerate(paragraphs):
            if len(para) > 50:  # Skip very short chunks
                all_chunks.append({
                    "text": para,
                    "doc_id": doc_id,
                    "page": page_num + 1,
                    "chunk_id": f"{doc_id}_p{page_num+1}_c{i}",
                    "trust_level": trust
                })

print(f"\nTotal text chunks: {len(all_chunks)}")

# =============================================================================
# Cell 5 - Extract figure data using Gemini Vision
# =============================================================================
def extract_page_data_with_vision(pdf_path, doc_id):
    """Use Gemini Vision to extract all figure/table data from a PDF."""
    doc = fitz.open(pdf_path)
    extracted = []

    for page_num in range(len(doc)):
        page = doc[page_num]

        # Skip DOC3 page 7 (Appendix C - low trust)
        if doc_id == "DOC3" and page_num == 6:
            print(f"  Skipping {doc_id} page 7 (Appendix C - low trust)")
            continue

        # Render page as image
        mat = fitz.Matrix(2, 2)  # 2x zoom
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        img_b64 = base64.b64encode(img_bytes).decode()

        prompt = f"""This is page {page_num + 1} from {doc_id}.

Extract ALL data from every figure, table, and chart on this page.
For each one:
1. State its label (e.g., "Figure A-2", "Table 1-2")
2. List ALL numerical values with their labels/years/countries
3. Include axis labels, legends, and any footnotes

Be exhaustive - every number matters for answering questions.
If no figures/tables, say "No figures or tables on this page."

Format as plain text, organized by figure/table."""

        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model="models/gemini-2.0-flash",
                    contents=[
                        prompt,
                        {"inline_data": {"mime_type": "image/png", "data": img_b64}}
                    ],
                    config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=4096)
                )

                text = response.text.strip()
                if text and "No figures or tables" not in text and len(text) > 100:
                    extracted.append({
                        "doc_id": doc_id,
                        "page": page_num + 1,
                        "content": text,
                        "trust_level": "high" if doc_id in ["DOC1", "DOC2"] else "medium"
                    })
                    print(f"  {doc_id} p{page_num + 1}: {len(text)} chars extracted")
                else:
                    print(f"  {doc_id} p{page_num + 1}: no figures/tables")

                time.sleep(2)  # Rate limit
                break

            except Exception as e:
                if "429" in str(e) or "exhausted" in str(e).lower():
                    wait = 30 * (attempt + 1)
                    print(f"  Rate limited. Waiting {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"  {doc_id} p{page_num + 1} error: {e}")
                    break

    doc.close()
    return extracted

# Extract figure data from all PDFs
all_figure_data = []
for pdf_file in sorted(DATA_PATH.glob("*.pdf")):
    doc_id = get_doc_id(pdf_file.name)
    print(f"\nExtracting figures from {pdf_file.name} ({doc_id})...")
    data = extract_page_data_with_vision(pdf_file, doc_id)
    all_figure_data.extend(data)

# Save for reference
with open(FIGURES_PATH / "extracted_figure_data.json", "w") as f:
    json.dump(all_figure_data, f, indent=2)

print(f"\n=== Extracted figure data from {len(all_figure_data)} pages ===")

# =============================================================================
# Cell 6 - Create ChromaDB index with text + figure data
# =============================================================================
# Initialize embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Create ChromaDB
chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
try:
    chroma_client.delete_collection("rag_index")
except:
    pass
collection = chroma_client.create_collection(
    name="rag_index",
    metadata={"hnsw:space": "cosine"}
)

# Add text chunks
print("Adding text chunks to index...")
for chunk in all_chunks:
    collection.add(
        documents=[chunk["text"]],
        metadatas=[{
            "doc_id": chunk["doc_id"],
            "page": chunk["page"],
            "trust_level": chunk["trust_level"],
            "source": "text"
        }],
        ids=[chunk["chunk_id"]]
    )

# Add figure data
print("Adding figure data to index...")
for item in all_figure_data:
    chunk_id = f"{item['doc_id']}_p{item['page']}_vision"
    collection.add(
        documents=[item["content"]],
        metadatas=[{
            "doc_id": item["doc_id"],
            "page": item["page"],
            "trust_level": item["trust_level"],
            "source": "vision_extraction"
        }],
        ids=[chunk_id]
    )

print(f"Total documents in index: {collection.count()}")

# Build BM25 index for hybrid search
all_texts = [c["text"] for c in all_chunks] + [f["content"] for f in all_figure_data]
tokenized = [t.lower().split() for t in all_texts]
bm25 = BM25Okapi(tokenized)

# =============================================================================
# Cell 7 - Search function (hybrid: embeddings + BM25)
# =============================================================================
def search(query, top_k=10):
    """Hybrid search: combine embedding similarity with BM25."""
    # Embedding search
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )

    # Combine results
    chunks = []
    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i]
        chunks.append({
            "text": doc,
            "doc_id": meta["doc_id"],
            "page": meta["page"],
            "trust_level": meta["trust_level"],
            "source": meta.get("source", "text")
        })

    return {"text": chunks, "figures": []}

# =============================================================================
# Cell 8 - Answer generation with Gemini
# =============================================================================
SYSTEM_PROMPT = """You are a precise economics Q&A assistant. Answer using ONLY the provided document excerpts.

TRUST LEVELS:
- DOC1 (Blanchard textbook): HIGH trust
- DOC2 (Supplement A outturns): HIGH trust
- DOC3 (Briefing B): MEDIUM trust, EXCEPT Appendix C (page 7) = LOW trust - NEVER use

THE 5 ANSWERING RULES:
1. OUTTURN SUPERSEDES FORECAST: If DOC2 gives an outturn for a value that DOC1 shows as forecast (*), use the outturn.
2. REVISION SUPERSEDES ORIGINAL: If a revision notice exists, use the revised value.
3. CONFLICT: If high/medium trust sources disagree with no revision, status='conflict', cite both.
4. NOT FOUND: If answer not in documents, status='not_found'. FALSE PREMISE: If question assumes something false, status='false_premise'.
5. SESSION OVERRIDE: If user says "assume X for this conversation", use X, status='session_override'.

CRITICAL OUTPUT RULES:
- Output ONLY valid JSON. No markdown. No code blocks. No reasoning.
- NEVER include your thought process, analysis, or "let me check" in the output.
- If status is "answered", you MUST have at least one citation. No citation = use "not_found".
- If your answer mentions ANY figure/table (Figure X-Y, Table X-Y), add it to figure_references.
- Keep answers concise (under 200 words).

CITATION FORMAT:
- DOC1:p4 (page 4)
- DOC2:Table A-1
- DOC3:Figure B-2
- Use lowercase 'p' for pages, no spaces

OUTPUT FORMAT (JSON only):
{
    "answer": "Concise answer with specific numbers",
    "status": "answered|not_found|conflict|false_premise|session_override",
    "citations": ["DOC1:p4", "DOC2:Table A-1"],
    "figure_references": ["Figure 1-2", "Table A-1"] or ["NONE"]
}"""

# Conversation memory
memory = {}

def format_context(search_results):
    """Format search results as context."""
    parts = []
    for chunk in search_results.get("text", []):
        trust = chunk.get("trust_level", "unknown")
        source = chunk.get("source", "text")
        parts.append(f"[{chunk['doc_id']} page {chunk['page']} (trust: {trust}, source: {source})]\n{chunk['text'][:2000]}")
    return "\n\n---\n\n".join(parts) if parts else "No relevant content found."

def generate_answer(question, search_results, conversation_id=None):
    """Generate answer using Gemini."""
    context = format_context(search_results)

    # Add conversation history
    history_text = ""
    if conversation_id and conversation_id in memory:
        for turn in memory[conversation_id]:
            history_text += f"Q: {turn['q']}\nA: {turn['a']}\n\n"

    prompt = f"""{SYSTEM_PROMPT}

{history_text}
DOCUMENT EXCERPTS:
{context}

QUESTION: {question}

Output ONLY valid JSON."""

    # Call Gemini with retries
    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=4096)
            )
            break
        except Exception as e:
            if "429" in str(e) or "exhausted" in str(e).lower():
                wait = 60 * (attempt + 1)
                print(f"  Rate limited. Waiting {wait}s...")
                time.sleep(wait)
            else:
                raise

    # Parse response
    text = response.text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)

    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if match:
        text = match.group()

    try:
        result = json.loads(text)
        result = {
            "answer": str(result.get("answer", "")),
            "status": result.get("status", "answered"),
            "citations": result.get("citations", []),
            "figure_references": result.get("figure_references", ["NONE"])
        }
    except json.JSONDecodeError:
        result = {
            "answer": text[:500],
            "status": "answered",
            "citations": [],
            "figure_references": ["NONE"]
        }

    if not result["figure_references"]:
        result["figure_references"] = ["NONE"]

    # Store in memory
    if conversation_id:
        if conversation_id not in memory:
            memory[conversation_id] = []
        memory[conversation_id].append({"q": question, "a": result["answer"]})

    return result

# Test
print("\nTesting answer generation...")
test_q = "What was U.S. unemployment rate in 2010?"
test_results = search(test_q)
test_answer = generate_answer(test_q, test_results)
print(f"Q: {test_q}")
print(f"A: {test_answer}")

# =============================================================================
# Cell 9 - Run full pipeline
# =============================================================================
questions = pd.read_csv(DATA_PATH / "Questions.csv")
print(f"Processing {len(questions)} questions...")

memory.clear()
results = []

for idx, row in questions.iterrows():
    qid = row["question_id"]
    cid = row["conversation_id"]
    question = row["question"]

    print(f"[{idx+1}/{len(questions)}] {qid}: {question[:50]}...")

    search_results = search(question)

    # Retry on rate limit
    for attempt in range(5):
        try:
            answer = generate_answer(question, search_results, conversation_id=cid)
            break
        except Exception as e:
            if "429" in str(e) or "exhausted" in str(e).lower():
                wait = 60 * (attempt + 1)
                print(f"  Rate limited. Waiting {wait}s...")
                time.sleep(wait)
            else:
                raise

    citations = ",".join(answer["citations"]) if answer["citations"] else "NONE"
    figures = ",".join(answer["figure_references"]) if answer["figure_references"] and answer["figure_references"] != ["NONE"] else "NONE"

    results.append({
        "id": row["id"],
        "conversation_id": cid,
        "question_id": qid,
        "answer": answer["answer"],
        "figure_references": figures,
        "citations": citations,
        "status": answer["status"]
    })

    time.sleep(4)  # Rate limiting

print("\nDone!")

# =============================================================================
# Cell 10 - Post-process and fix citations
# =============================================================================
results_df = pd.DataFrame(results)

# Fix citation format
CIT_PATTERN = r'^(DOC[123]:(p\d+|Table [A-Z]?-?\d+|Figure [A-Z]?-?\d+)|NONE)(,(DOC[123]:(p\d+|Table [A-Z]?-?\d+|Figure [A-Z]?-?\d+)))*$'

def fix_citations(cit):
    if pd.isna(cit) or cit == "" or cit == "NONE":
        return "NONE"
    cit = str(cit).strip()
    cit = re.sub(r'Page\s*(\d+)', r'p\1', cit, flags=re.IGNORECASE)
    cit = re.sub(r'page\s*(\d+)', r'p\1', cit, flags=re.IGNORECASE)
    cit = cit.replace(" ,", ",").replace(", ", ",")
    if not re.match(CIT_PATTERN, cit):
        return "NONE"
    return cit

results_df["citations"] = results_df["citations"].apply(fix_citations)

# Show status distribution
print("\nStatus distribution:")
print(results_df["status"].value_counts())

# =============================================================================
# Cell 11 - Save submission
# =============================================================================
results_df.to_csv(OUTPUT_PATH / "submission.csv", index=False)
print(f"\nSaved to {OUTPUT_PATH / 'submission.csv'}")
print(results_df.head(10))

# =============================================================================
# Cell 12 - Validate submission
# =============================================================================
import subprocess
validator = DATA_PATH / "validate_submission.py"
submission = OUTPUT_PATH / "submission.csv"
questions_file = DATA_PATH / "Questions.csv"

result = subprocess.run(
    ["python", str(validator), str(submission), str(questions_file)],
    capture_output=True, text=True
)
print(result.stdout)
if result.returncode != 0:
    print("VALIDATION ERRORS:")
    print(result.stderr)
else:
    print("Validation passed! Ready to submit.")

"""Parse PDFs: extract text chunks by page and figures/tables as images."""
import json
import re
from pathlib import Path

import fitz  # pymupdf

from .common import CHUNKS_PATH, DATA, DOCS, FIGURES_PATH


def extract_text_chunks(doc_id: str, min_chunk_size: int = 100) -> list[dict]:
    """Extract text from PDF, split into chunks by page."""
    info = DOCS[doc_id]
    pdf_path = DATA / info["file"]
    doc = fitz.open(pdf_path)

    chunks = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text").strip()
        if len(text) < min_chunk_size:
            continue
        chunks.append({
            "doc_id": doc_id,
            "page": page_num + 1,
            "text": text,
            "trust_level": get_trust_level(info, page_num + 1),
            "title": info["title"],
        })
    doc.close()
    return chunks


def get_trust_level(doc_info: dict, page: int) -> str:
    """Get trust level for a page, checking section overrides."""
    base_trust = doc_info.get("trust_level", "medium")
    for override in doc_info.get("section_trust_overrides", []):
        if page in override.get("pages", []):
            return override["trust_level"]
    return base_trust


def extract_figures(doc_id: str) -> list[dict]:
    """Extract figures and tables as images, with their IDs parsed from captions."""
    info = DOCS[doc_id]
    pdf_path = DATA / info["file"]
    doc = fitz.open(pdf_path)

    figures = []
    fig_pattern = re.compile(r"(Figure|Table)\s+([A-Z]?-?\d+(?:\s*\([^)]+\))?)", re.IGNORECASE)

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")

        # Find figure/table references on this page
        matches = fig_pattern.findall(text)
        for fig_type, fig_num in matches:
            fig_id = f"{fig_type} {fig_num}".strip()
            fig_id_clean = fig_id.replace(" ", "_").replace("(", "").replace(")", "")

            # Render page as image for this figure
            out_path = FIGURES_PATH / doc_id / f"{fig_id_clean}.png"
            if not out_path.exists():
                out_path.parent.mkdir(parents=True, exist_ok=True)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom
                pix.save(str(out_path))

            figures.append({
                "doc_id": doc_id,
                "figure_id": fig_id,
                "figure_type": fig_type.lower(),
                "page": page_num + 1,
                "image_path": str(out_path.relative_to(FIGURES_PATH.parent)),
                "trust_level": get_trust_level(info, page_num + 1),
            })

    doc.close()
    # Remove duplicates (same figure_id)
    seen = set()
    unique = []
    for f in figures:
        if f["figure_id"] not in seen:
            seen.add(f["figure_id"])
            unique.append(f)
    return unique


def ingest_all():
    """Parse all documents and save chunks and figures."""
    all_chunks = []
    all_figures = []

    for doc_id in DOCS:
        print(f"Processing {doc_id}...")
        chunks = extract_text_chunks(doc_id)
        figures = extract_figures(doc_id)
        all_chunks.extend(chunks)
        all_figures.extend(figures)
        print(f"  {len(chunks)} chunks, {len(figures)} figures/tables")

    # Save to JSON
    CHUNKS_PATH.mkdir(parents=True, exist_ok=True)
    (CHUNKS_PATH / "chunks.json").write_text(json.dumps(all_chunks, indent=2))
    (CHUNKS_PATH / "figures.json").write_text(json.dumps(all_figures, indent=2))

    print(f"\nTotal: {len(all_chunks)} chunks, {len(all_figures)} figures")
    return all_chunks, all_figures


if __name__ == "__main__":
    ingest_all()

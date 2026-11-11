# Cell 4b - Pre-extract figure/table data using Gemini Vision
import fitz  # PyMuPDF
import base64
import time

FIGURES_PATH = OUTPUT_PATH / "figures"
FIGURES_PATH.mkdir(exist_ok=True)

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

# Process all PDFs
all_figure_data = []

for pdf_file in sorted(DATA_PATH.glob("*.pdf")):
    if pdf_file.name == "Document.pdf":
        doc_id = "DOC1"
    elif "Document2" in pdf_file.name:
        doc_id = "DOC2"
    else:
        doc_id = "DOC3"

    print(f"\nProcessing {pdf_file.name} ({doc_id})...")
    data = extract_page_data_with_vision(pdf_file, doc_id)
    all_figure_data.extend(data)

# Save for reuse
with open(FIGURES_PATH / "extracted_figure_data.json", "w") as f:
    json.dump(all_figure_data, f, indent=2)

print(f"\n=== Extracted figure data from {len(all_figure_data)} pages ===")


# ============================================================
# Cell 4c - Add extracted figure data to ChromaDB index
# ============================================================

print("Adding figure data to search index...")

for item in all_figure_data:
    chunk_id = f"{item['doc_id']}_p{item['page']}_vision"

    try:
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
    except Exception as e:
        # Already exists, update it
        collection.update(
            documents=[item["content"]],
            metadatas=[{
                "doc_id": item["doc_id"],
                "page": item["page"],
                "trust_level": item["trust_level"],
                "source": "vision_extraction"
            }],
            ids=[chunk_id]
        )

print(f"Added {len(all_figure_data)} vision extractions to index")
print(f"Total documents in collection: {collection.count()}")

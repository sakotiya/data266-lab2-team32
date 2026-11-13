"""1.1.4 Store pages and figures (caption + VLM description) in ChromaDB with doc_id, page, figure_id, trust_level."""
import json
from functools import cache

import chromadb
from chromadb.utils import embedding_functions

from .common import CHROMA_PATH, CHUNKS_PATH, DESCRIPTIONS

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
COLLECTION = "knowledge_base"


def load_items():
    """Every searchable item: one per page, one per figure/table. Ids are their citation strings."""
    chunks = json.loads((CHUNKS_PATH / "chunks.json").read_text())
    figures = json.loads((CHUNKS_PATH / "figures.json").read_text())
    descriptions = json.loads(DESCRIPTIONS.read_text())
    items = [{"id": f"{c['doc_id']}:p{c['page']}", "doc_id": c["doc_id"], "page": c["page"], "figure_id": "",
              "trust_level": c["trust_level"], "text": c["text"]} for c in chunks]
    for f in figures:
        key = f"{f['doc_id']}:{f['figure_id']}"
        items.append({"id": key, "doc_id": f["doc_id"], "page": f["page"], "figure_id": f["figure_id"],
                      "trust_level": f["trust_level"], "image_path": f["image_path"], "caption": f["caption"],
                      "text": f"{f['caption']}\n{descriptions[key]['text']}"})
    return items


@cache
def embedder():
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)


def build_index(items=None):
    items = items or load_items()
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    if COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION)
    col = client.create_collection(COLLECTION, embedding_function=embedder(), metadata={"hnsw:space": "cosine"})
    col.add(ids=[i["id"] for i in items], documents=[i["text"] for i in items],
            metadatas=[{k: i[k] for k in ("doc_id", "page", "figure_id", "trust_level")} for i in items])
    print(f"indexed {len(items)} items ({sum(bool(i['figure_id']) for i in items)} figures/tables)")
    return col


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_collection(COLLECTION, embedding_function=embedder())


if __name__ == "__main__":
    build_index()

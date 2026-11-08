"""Index chunks and figures in ChromaDB with embeddings."""
import json

import chromadb
from chromadb.utils import embedding_functions

from .common import CHROMA_PATH, CHUNKS_PATH, DOCS


def get_embedding_function():
    """Use sentence-transformers for embeddings."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )


def build_index():
    """Build ChromaDB index from extracted chunks and figures."""
    chunks = json.loads((CHUNKS_PATH / "chunks.json").read_text())
    figures = json.loads((CHUNKS_PATH / "figures.json").read_text())

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    embed_fn = get_embedding_function()

    # Delete existing collections if they exist
    for name in ["text_chunks", "figures"]:
        try:
            client.delete_collection(name)
        except ValueError:
            pass

    # Index text chunks
    text_col = client.create_collection(name="text_chunks", embedding_function=embed_fn)
    text_col.add(
        ids=[f"{c['doc_id']}_p{c['page']}" for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[{
            "doc_id": c["doc_id"],
            "page": c["page"],
            "trust_level": c["trust_level"],
            "title": c["title"],
        } for c in chunks],
    )
    print(f"Indexed {len(chunks)} text chunks")

    # Index figures with their IDs as searchable text
    fig_col = client.create_collection(name="figures", embedding_function=embed_fn)
    fig_col.add(
        ids=[f"{f['doc_id']}_{f['figure_id'].replace(' ', '_')}" for f in figures],
        documents=[f"{f['figure_id']} from {DOCS[f['doc_id']]['title']}" for f in figures],
        metadatas=[{
            "doc_id": f["doc_id"],
            "figure_id": f["figure_id"],
            "figure_type": f["figure_type"],
            "page": f["page"],
            "image_path": f["image_path"],
            "trust_level": f["trust_level"],
        } for f in figures],
    )
    print(f"Indexed {len(figures)} figures/tables")

    return client


def get_client():
    """Get existing ChromaDB client."""
    return chromadb.PersistentClient(path=str(CHROMA_PATH))


if __name__ == "__main__":
    build_index()

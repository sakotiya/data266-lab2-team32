"""Search: embeddings + BM25 hybrid, exact figure matching."""
import re
from rank_bm25 import BM25Okapi

from .common import CHROMA_PATH, CHUNKS_PATH, DOCS
from .index import get_client, get_embedding_function
import json


class Searcher:
    def __init__(self):
        self.client = get_client()
        self.embed_fn = get_embedding_function()
        self.text_col = self.client.get_collection("text_chunks", embedding_function=self.embed_fn)
        self.fig_col = self.client.get_collection("figures", embedding_function=self.embed_fn)

        # Load chunks for BM25
        self.chunks = json.loads((CHUNKS_PATH / "chunks.json").read_text())
        self.figures = json.loads((CHUNKS_PATH / "figures.json").read_text())

        # Build BM25 index
        tokenized = [c["text"].lower().split() for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 5) -> dict:
        """Hybrid search: embeddings + BM25 for text, exact match for figures."""
        results = {"text": [], "figures": []}

        # Check for explicit figure/table references
        fig_refs = self._extract_figure_refs(query)
        if fig_refs:
            for ref in fig_refs:
                fig = self._exact_figure_match(ref)
                if fig and fig not in results["figures"]:
                    results["figures"].append(fig)

        # Embedding search for text
        text_results = self.text_col.query(query_texts=[query], n_results=top_k)
        embed_ids = set(text_results["ids"][0]) if text_results["ids"] else set()

        # BM25 search
        query_tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        bm25_top = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
        bm25_ids = {f"{self.chunks[i]['doc_id']}_p{self.chunks[i]['page']}" for i in bm25_top}

        # Combine results (union of embedding and BM25)
        combined_ids = embed_ids | bm25_ids
        for i, chunk in enumerate(self.chunks):
            chunk_id = f"{chunk['doc_id']}_p{chunk['page']}"
            if chunk_id in combined_ids:
                score = 0
                if chunk_id in embed_ids:
                    idx = text_results["ids"][0].index(chunk_id)
                    score += 1 - (text_results["distances"][0][idx] if text_results["distances"] else 0)
                if i in bm25_top:
                    score += bm25_scores[i] / max(bm25_scores) if max(bm25_scores) > 0 else 0
                results["text"].append({**chunk, "score": score})

        # Sort by score and limit
        results["text"] = sorted(results["text"], key=lambda x: x["score"], reverse=True)[:top_k]

        # Also search figures by embedding if no exact match
        if not results["figures"]:
            fig_results = self.fig_col.query(query_texts=[query], n_results=3)
            if fig_results["ids"] and fig_results["ids"][0]:
                for i, fid in enumerate(fig_results["ids"][0]):
                    meta = fig_results["metadatas"][0][i]
                    results["figures"].append(meta)

        return results

    def _extract_figure_refs(self, query: str) -> list[str]:
        """Extract figure/table references from query."""
        pattern = r"(Figure|Table)\s+([A-Z]?-?\d+(?:\s*\([^)]+\))?)"
        matches = re.findall(pattern, query, re.IGNORECASE)
        return [f"{m[0]} {m[1]}" for m in matches]

    def _exact_figure_match(self, ref: str) -> dict | None:
        """Find exact figure by ID."""
        ref_lower = ref.lower().strip()
        for fig in self.figures:
            if fig["figure_id"].lower() == ref_lower:
                return fig
            # Also try without spaces
            if fig["figure_id"].lower().replace(" ", "") == ref_lower.replace(" ", ""):
                return fig
        return None

    def get_figure_image(self, figure_id: str) -> str | None:
        """Get image path for a figure."""
        for fig in self.figures:
            if fig["figure_id"].lower() == figure_id.lower():
                return fig["image_path"]
        return None


if __name__ == "__main__":
    s = Searcher()
    results = s.search("What was U.S. unemployment in 2010?")
    print(f"Found {len(results['text'])} text chunks, {len(results['figures'])} figures")
    for r in results["text"][:3]:
        print(f"  {r['doc_id']} p{r['page']}: {r['text'][:100]}...")

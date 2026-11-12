"""1.2 Retrieval: dense (baseline) or dense + BM25 fused with reciprocal-rank fusion (final), and exact lookup of any
Figure/Table the question names."""
import re

from rank_bm25 import BM25Okapi

from .index import get_collection, load_items

# "Figure A-5", "Figures A-6 and A-7", "Table 1-2", "Figure B-5 (corrected)"
NAMED = re.compile(r"\b(Figures?|Tables?)\s+((?:(?:[12]|[A-C])-\d+(?:\s*\(corrected\))?(?:\s*(?:,|and|&)\s*)?)+)", re.I)
ID = re.compile(r"((?:[12]|[A-C])-\d+)(\s*\(corrected\))?", re.I)
RRF_K = 60
POOL = 20          # candidates taken from each retriever before fusion


def tokens(text):
    return re.findall(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*", text.lower())


def named_labels(text):
    """Exact figure/table labels mentioned in the text, e.g. ['Figure A-6', 'Figure A-7']."""
    out = []
    for kind, ids in NAMED.findall(text):
        kind = "Figure" if kind.lower().startswith("fig") else "Table"
        for num, corrected in ID.findall(ids):
            out.append(f"{kind} {num.upper()}" + (" (corrected)" if corrected else ""))
    return out


class Searcher:
    def __init__(self):
        self.items = {i["id"]: i for i in load_items()}
        self.ids = list(self.items)
        self.by_label = {i["figure_id"]: i for i in self.items.values() if i["figure_id"]}
        self.col = get_collection()
        self.bm25 = BM25Okapi([tokens(self.items[i]["text"]) for i in self.ids])

    def dense(self, query, n=POOL):
        return self.col.query(query_texts=[query], n_results=n)["ids"][0]

    def sparse(self, query, n=POOL):
        scores = self.bm25.get_scores(tokens(query))
        return [self.ids[i] for i in sorted(range(len(scores)), key=lambda i: -scores[i])[:n]]

    def search(self, query, k=8, mode="hybrid"):
        """Top-k items (pages and figures). Named figures always come first; low-trust items are never returned."""
        if mode == "dense":
            ranked = self.dense(query)
        else:
            score = {}
            for ranking in (self.dense(query), self.sparse(query)):
                for r, i in enumerate(ranking):
                    score[i] = score.get(i, 0) + 1 / (RRF_K + r)
            ranked = sorted(score, key=lambda i: -score[i])
        named = []
        for label in named_labels(query):
            named += [l for l in (label, f"{label} (corrected)") if l in self.by_label]   # an erratum travels with its original
        ranked = [i for i in ranked if self.items[i]["trust_level"] != "low"]
        out = [self.by_label[l]["id"] for l in dict.fromkeys(named)]
        out += [i for i in ranked if i not in out][:max(k - len(out), 0)]
        # Cross-document coverage, so a disagreeing source is always visible: the 2 best hits of every document.
        for doc in ("DOC1", "DOC2", "DOC3"):
            out += [i for i in ranked if self.items[i]["doc_id"] == doc and i not in out][:max(0, 2 - sum(
                self.items[o]["doc_id"] == doc for o in out))]
        return [self.items[i] for i in out]

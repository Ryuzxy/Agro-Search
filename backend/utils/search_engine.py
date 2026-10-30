"""TF-IDF + Cosine Similarity IR engine for AGRO-SEARCH."""

import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils.preprocessing import preprocess

CORPUS_PATH = os.path.join(os.path.dirname(__file__), "..", "corpus", "documents.json")

# Fields to index (concatenated per document)
_INDEX_FIELDS = [
    "judul_penyakit",
    "penyebab_patogen",
    "gejala_visual",
    "rekomendasi_pestisida",
    "langkah_penanganan",
]


class SearchEngine:
    def __init__(self, corpus_path: str = CORPUS_PATH):
        with open(corpus_path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)

        # Build preprocessed corpus
        self.processed_docs = []
        for doc in self.documents:
            raw = " ".join(doc.get(field, "") for field in _INDEX_FIELDS)
            self.processed_docs.append(preprocess(raw))

        # Fit TF-IDF
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.processed_docs)

    def search(self, query: str, top_n: int = 5) -> list[dict]:
        """Return top_n documents ranked by cosine similarity."""
        processed_query = preprocess(query)
        query_vec = self.vectorizer.transform([processed_query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        ranked_indices = scores.argsort()[::-1][:top_n]
        results = []
        for idx in ranked_indices:
            score = float(scores[idx])
            if score > 0:
                doc = self.documents[idx].copy()
                doc["skor_relevansi"] = round(score, 4)
                results.append(doc)
        return results

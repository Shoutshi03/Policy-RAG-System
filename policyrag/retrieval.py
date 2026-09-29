from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .models import Chunk, RetrievedChunk
from .qdrant_store import get_qdrant_index

STOPWORDS = set((
    "a à au aux avec ce ces dans de des du elle en et eux il ils je la le les leur lui ma mais me même mes moi mon ne nos notre nous on ou par pas pour qu que quelle qui sa se ses son sur ta te tes toi ton tu un une vos votre vous y "
    "a an and are as at be been being but by can could did do does for from had has have he her hers him his how i if in into is it its me my no not of on or our she so than that the their them then there these they this those to was we were what when where which who will with would you your "
    "و في من على إلى عن أن إن لا ما هذا هذه ذلك التي الذي هو هي هم نحن مع أو ثم كان تكون بين بعد قبل قد كل بعض كما لدى حيث أي"
).split())


def normalize_text(text: str) -> str:
    value = unicodedata.normalize("NFD", text)
    value = re.sub(r"[\u064b-\u065f\u0670\u0640]", "", value)
    value = re.sub(r"[أإآٱ]", "ا", value).replace("ى", "ي").replace("ة", "ه")
    value = "".join(char for char in value if unicodedata.category(char) != "Mn")
    return value.lower()


def tokenize(text: str) -> list[str]:
    terms = re.findall(r"[\w\u0600-\u06ff]+", normalize_text(text), flags=re.UNICODE)
    return [term for term in terms if len(term) > 1 and term not in STOPWORDS]


class HybridRetriever:
    """BM25 plus a local character n-gram hashing baseline.

    The feature-hash vectors are language-agnostic and dependency-light; they
    are not a substitute for a trained multilingual embedding model. Optional
    BGE/Qdrant adapters can replace this baseline for production deployments.
    """

    def __init__(self) -> None:
        self.vectorizer = HashingVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            n_features=2**13,
            alternate_sign=False,
            norm="l2",
            lowercase=True,
            strip_accents="unicode",
        )

    def search(self, chunks: Sequence[Chunk], query_variants: Sequence[str], top_k: int = 7, workspace_id: str | None = None) -> list[RetrievedChunk]:
        if not chunks:
            return []
        documents = [chunk.text for chunk in chunks]
        tokenized = [tokenize(text) for text in documents]
        if not any(tokenized):
            return []
        queries = [query for query in query_variants if query.strip()]
        query_text = " ".join(dict.fromkeys(queries))
        query_tokens = list(dict.fromkeys(term for query in queries for term in tokenize(query)))
        if not query_tokens:
            return []

        bm25 = BM25Okapi(tokenized)
        lexical_scores = np.maximum(bm25.get_scores(query_tokens), 0.0)
        max_lexical = float(lexical_scores.max()) if len(lexical_scores) else 0.0
        lexical_norm = lexical_scores / (lexical_scores + 2.6) if max_lexical else lexical_scores

        dense_scores = cosine_similarity(self.vectorizer.transform(documents), self.vectorizer.transform([query_text])).ravel()
        dense_scores = np.maximum(dense_scores, 0.0)
        if workspace_id:
            index = get_qdrant_index()
            if index:
                try:
                    remote_scores = index.search(workspace_id, query_text, limit=max(top_k * 2, 12))
                    if remote_scores:
                        dense_scores = np.asarray([
                            0.35 * float(dense_scores[position]) + 0.65 * remote_scores.get(chunk.chunk_id, 0.0)
                            for position, chunk in enumerate(chunks)
                        ])
                except Exception:
                    # A remote vector service must not make local retrieval unavailable.
                    pass
        # Require some exact lexical evidence or a substantial character-pattern match.
        combined = 0.70 * lexical_norm + 0.30 * dense_scores
        ranked_indices = np.argsort(combined)[::-1][:top_k]
        results: list[RetrievedChunk] = []
        for index in ranked_indices:
            if combined[index] <= 0:
                continue
            results.append(RetrievedChunk(
                chunk=chunks[int(index)],
                score=float(combined[index]),
                lexical_score=float(lexical_scores[index]),
                dense_score=float(dense_scores[index]),
            ))
        return results

    @staticmethod
    def has_sufficient_evidence(results: Sequence[RetrievedChunk]) -> bool:
        if not results:
            return False
        best = results[0]
        return best.lexical_score >= 0.12 or best.dense_score >= 0.36

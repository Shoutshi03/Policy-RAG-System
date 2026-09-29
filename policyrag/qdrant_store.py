from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from sklearn.feature_extraction.text import HashingVectorizer

from .models import Chunk

load_dotenv()
VECTOR_SIZE = 2**13
COLLECTION = os.getenv("QDRANT_COLLECTION", "undp_policyrag_demo")
_vectorizer = HashingVectorizer(
    analyzer="char", ngram_range=(2, 5), n_features=VECTOR_SIZE,
    alternate_sign=False, norm="l2", lowercase=True, strip_accents="unicode",
)


class QdrantIndex:
    """Optional remote vector index. No connection is made unless QDRANT_URL is set."""

    def __init__(self, url: str, api_key: str | None = None, collection: str = COLLECTION) -> None:
        self.collection = collection
        self.client = QdrantClient(url=url, api_key=api_key or None, timeout=8.0)
        if not self.client.collection_exists(collection):
            self.client.create_collection(
                collection_name=collection,
                vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
            )

    def upsert(self, workspace_id: str, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        matrix = _vectorizer.transform([chunk.text for chunk in chunks])
        points = []
        for index, chunk in enumerate(chunks):
            vector = matrix[index].toarray().ravel().tolist()
            points.append(models.PointStruct(
                id=chunk.chunk_id,
                vector=vector,
                payload={
                    "workspace_id": workspace_id,
                    "document_id": chunk.document_id,
                    "file_name": chunk.file_name,
                    "page": chunk.page,
                    "section": chunk.section,
                    "language": chunk.language,
                    "text": chunk.text,
                    "is_demo": chunk.is_demo,
                },
            ))
        for offset in range(0, len(points), 64):
            self.client.upsert(collection_name=self.collection, points=points[offset : offset + 64], wait=True)

    def search(self, workspace_id: str, query: str, limit: int = 15) -> dict[str, float]:
        vector = _vectorizer.transform([query]).toarray().ravel().tolist()
        result = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            query_filter=models.Filter(must=[
                models.FieldCondition(key="workspace_id", match=models.MatchValue(value=workspace_id)),
            ]),
            limit=limit,
            with_payload=False,
        )
        return {str(point.id): float(point.score) for point in result.points}

    def delete_document(self, workspace_id: str, document_id: str) -> None:
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(filter=models.Filter(must=[
                models.FieldCondition(key="workspace_id", match=models.MatchValue(value=workspace_id)),
                models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id)),
            ])),
            wait=True,
        )

    def delete_workspace(self, workspace_id: str) -> None:
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(filter=models.Filter(must=[
                models.FieldCondition(key="workspace_id", match=models.MatchValue(value=workspace_id)),
            ])),
            wait=True,
        )


@lru_cache(maxsize=1)
def get_qdrant_index() -> QdrantIndex | None:
    url = os.getenv("QDRANT_URL")
    if not url:
        return None
    return QdrantIndex(url, os.getenv("QDRANT_API_KEY"))


def try_upsert(workspace_id: str, chunks: list[Chunk]) -> str:
    index = get_qdrant_index()
    if not index:
        return "local"
    try:
        index.upsert(workspace_id, chunks)
        return "qdrant"
    except Exception as exc:
        # Keep the local demo usable if the optional remote index is unavailable.
        return f"local-fallback:{type(exc).__name__}"

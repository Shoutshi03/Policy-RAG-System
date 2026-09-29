from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Mode = Literal["qa", "summary", "explain", "compare", "extract", "translate", "insights"]
Language = Literal["fr", "en", "ar"]


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    document_id: str
    file_name: str
    page: int
    section: str
    language: str
    text: str
    is_demo: bool = False


@dataclass(slots=True)
class Document:
    document_id: str
    file_name: str
    file_hash: str
    language: str
    pages: int
    chunks: list[Chunk] = field(default_factory=list)
    is_demo: bool = False
    ocr_pages: int = 0

    @property
    def word_count(self) -> int:
        return sum(len(chunk.text.split()) for chunk in self.chunks)


@dataclass(slots=True)
class RetrievedChunk:
    chunk: Chunk
    score: float
    lexical_score: float
    dense_score: float


@dataclass(slots=True)
class Citation:
    source_id: str
    document_id: str
    document: str
    page: int
    section: str
    excerpt: str
    is_demo: bool = False


@dataclass(slots=True)
class AssistantResult:
    answer: str
    citations: list[Citation]
    abstained: bool
    confidence: float
    retrieval: str = "BM25 + character n-gram vectors"
    model: str = ""
    warning: str = ""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import asdict
from threading import RLock
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from policyrag.ingestion import MAX_FILE_BYTES, extract_pdf
from policyrag.models import Document
from policyrag.qdrant_store import get_qdrant_index, try_upsert
from policyrag.service import answer_question

app = FastAPI(
    title="UNDP PolicyRAG API",
    version="1.0.0",
    description="Prototype API pour l’ingestion de rapports PDF et le retrieval augmenté par génération.",
)

# Demo-only in-memory workspace store. The random workspace token is a bearer capability.
# Add identity, authorization, persistence and retention controls before public deployment.
_WORKSPACES: dict[str, list[Document]] = {}
_LOCK = RLock()


class WorkspaceResponse(BaseModel):
    workspace_id: str


class AskRequest(BaseModel):
    workspace_id: str = Field(min_length=24, max_length=64)
    question: str = Field(min_length=2, max_length=4000)
    mode: Literal["qa", "summary", "explain", "compare", "extract", "translate", "insights"] = "qa"
    language: Literal["fr", "en", "ar"] = "fr"
    history: list[dict[str, str]] = Field(default_factory=list, max_length=8)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "undp-policyrag"}


@app.post("/api/workspaces", response_model=WorkspaceResponse)
def create_workspace() -> WorkspaceResponse:
    workspace_id = secrets.token_urlsafe(32)
    with _LOCK:
        _WORKSPACES[workspace_id] = []
    return WorkspaceResponse(workspace_id=workspace_id)


@app.get("/api/workspaces/{workspace_id}/documents")
def list_documents(workspace_id: str) -> list[dict[str, object]]:
    documents = _workspace(workspace_id)
    return [
        {
            "document_id": item.document_id,
            "file_name": item.file_name,
            "language": item.language,
            "pages": item.pages,
            "chunks": len(item.chunks),
            "word_count": item.word_count,
            "is_demo": item.is_demo,
        }
        for item in documents
    ]


@app.post("/api/workspaces/{workspace_id}/documents")
async def upload_documents(
    workspace_id: str,
    files: list[UploadFile] = File(...),
    ocr_languages: str = "fra+eng",
) -> list[dict[str, object]]:
    documents = _workspace(workspace_id)
    results: list[dict[str, object]] = []
    for upload in files:
        raw = await upload.read(MAX_FILE_BYTES + 1)
        if len(raw) > MAX_FILE_BYTES:
            raise HTTPException(status_code=413, detail=f"{upload.filename} dépasse la limite de 20 Mo.")
        digest = hashlib.sha256(raw).hexdigest()
        if any(item.file_hash == digest for item in documents):
            results.append({"file_name": upload.filename, "duplicate": True, "chunks": 0})
            continue
        try:
            document = extract_pdf(raw, upload.filename or "rapport.pdf", enable_ocr=True, ocr_languages=ocr_languages)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        documents.append(document)
        try_upsert(workspace_id, document.chunks)
        results.append({
            "document_id": document.document_id,
            "file_name": document.file_name,
            "pages": document.pages,
            "chunks": len(document.chunks),
            "ocr_pages": document.ocr_pages,
            "duplicate": False,
        })
    return results


@app.delete("/api/workspaces/{workspace_id}/documents/{document_id}")
def delete_document(workspace_id: str, document_id: str) -> dict[str, bool]:
    documents = _workspace(workspace_id)
    found = next((item for item in documents if item.document_id == document_id), None)
    if not found:
        raise HTTPException(status_code=404, detail="Document introuvable dans cet espace.")
    index = get_qdrant_index()
    if index:
        try:
            index.delete_document(workspace_id, document_id)
        except Exception:
            pass
    with _LOCK:
        _WORKSPACES[workspace_id] = [item for item in documents if item.document_id != document_id]
    return {"deleted": True}


@app.post("/api/ask")
def ask(request: AskRequest) -> dict[str, object]:
    documents = _workspace(request.workspace_id)
    result = answer_question(
        question=request.question,
        chunks=[chunk for document in documents for chunk in document.chunks],
        mode=request.mode,
        language=request.language,
        history=request.history,
        workspace_id=request.workspace_id,
    )
    return asdict(result)


def _workspace(workspace_id: str) -> list[Document]:
    with _LOCK:
        documents = _WORKSPACES.get(workspace_id)
    if documents is None:
        raise HTTPException(status_code=404, detail="Espace inconnu ou expiré. Créez un nouvel espace de travail.")
    return documents

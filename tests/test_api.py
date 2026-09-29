from __future__ import annotations

import pymupdf as fitz
from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


def make_pdf() -> bytes:
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text(
        (72, 72),
        "1. Policy objective\n\nThe policy brief recommends community early-warning systems and accessible alerts for at-risk households.",
        fontsize=12,
    )
    value = pdf.tobytes()
    pdf.close()
    return value


def test_health_and_workspace_isolation() -> None:
    assert client.get("/health").json()["status"] == "ok"
    workspace = client.post("/api/workspaces").json()["workspace_id"]
    assert client.get(f"/api/workspaces/{workspace}/documents").json() == []
    assert client.get("/api/workspaces/not-a-workspace/documents").status_code == 404


def test_pdf_upload_deduplication_and_grounded_question(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    workspace = client.post("/api/workspaces").json()["workspace_id"]
    pdf_bytes = make_pdf()
    first = client.post(
        f"/api/workspaces/{workspace}/documents",
        files=[("files", ("policy.pdf", pdf_bytes, "application/pdf"))],
    )
    assert first.status_code == 200, first.text
    assert first.json()[0]["chunks"] >= 1
    duplicate = client.post(
        f"/api/workspaces/{workspace}/documents",
        files=[("files", ("policy.pdf", pdf_bytes, "application/pdf"))],
    )
    assert duplicate.status_code == 200
    assert duplicate.json()[0]["duplicate"] is True
    answer = client.post("/api/ask", json={
        "workspace_id": workspace,
        "question": "What is recommended for at-risk households?",
        "mode": "qa",
        "language": "en",
    })
    assert answer.status_code == 200, answer.text
    result = answer.json()
    assert result["abstained"] is False
    assert result["citations"][0]["page"] == 1
    assert "[S1]" in result["answer"]


def test_remove_document_scoped_to_workspace() -> None:
    workspace = client.post("/api/workspaces").json()["workspace_id"]
    uploaded = client.post(
        f"/api/workspaces/{workspace}/documents",
        files=[("files", ("policy.pdf", make_pdf(), "application/pdf"))],
    )
    document_id = uploaded.json()[0]["document_id"]
    assert client.delete(f"/api/workspaces/{workspace}/documents/{document_id}").json() == {"deleted": True}
    assert client.delete(f"/api/workspaces/{workspace}/documents/{document_id}").status_code == 404

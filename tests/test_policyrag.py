from __future__ import annotations

import io
from pathlib import Path

import pymupdf as fitz
import pytest
from PIL import Image, ImageDraw, ImageFont

from policyrag.demo import demo_documents
from policyrag.generation import sanitize_citations
from policyrag.ingestion import extract_pdf
from policyrag.retrieval import HybridRetriever, normalize_text, tokenize
from policyrag.service import answer_question


def make_text_pdf(text: str) -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text, fontsize=12)
    result = document.tobytes()
    document.close()
    return result


def test_pdf_extraction_preserves_pages_and_section_metadata() -> None:
    data = make_text_pdf("1. OBJECTIFS STRATÉGIQUES\n\nLe programme vise une amélioration de la résilience climatique et des services essentiels.")
    document = extract_pdf(data, "policy.pdf", enable_ocr=False)
    assert document.pages == 1
    assert document.language == "fr"
    assert document.chunks
    assert document.chunks[0].page == 1
    assert document.chunks[0].section == "1. OBJECTIFS STRATÉGIQUES"
    assert document.file_hash and len(document.file_hash) == 64


def test_ingestion_rejects_non_pdf_and_oversize() -> None:
    with pytest.raises(ValueError, match="PDF valide"):
        extract_pdf(b"not a pdf", "bad.pdf")
    with pytest.raises(ValueError, match="20 Mo"):
        extract_pdf(b"%PDF" + b"a" * (20 * 1024 * 1024), "large.pdf")


def test_ocr_fallback_recovers_image_only_page() -> None:
    image = Image.new("RGB", (1600, 600), "white")
    draw = ImageDraw.Draw(image)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 48) if Path(font_path).exists() else ImageFont.load_default()
    draw.text((60, 60), "POLICY ACTION PLAN 2026", fill="black", font=font)
    draw.text((60, 150), "Early warning services for all communities", fill="black", font=font)
    picture = io.BytesIO()
    image.save(picture, format="PNG")
    document = fitz.open()
    page = document.new_page(width=800, height=300)
    page.insert_image(page.rect, stream=picture.getvalue())
    pdf_bytes = document.tobytes()
    document.close()

    result = extract_pdf(pdf_bytes, "scanned.pdf", enable_ocr=True, ocr_languages="eng")
    assert result.ocr_pages == 1
    assert result.chunks
    assert "warning" in result.chunks[0].text.lower()


def test_multilingual_normalization_and_tokenization() -> None:
    assert normalize_text("Égalité") == "egalite"
    assert "الاندماج" in tokenize("الاندماج الاجتماعي")
    assert "resilience" in tokenize("résilience climatique")


def test_hybrid_search_finds_correct_page_in_three_languages() -> None:
    chunks = [chunk for doc in demo_documents() for chunk in doc.chunks]
    retriever = HybridRetriever()
    french = retriever.search(chunks, ["cible réduction interruptions services 2030"], top_k=3)
    english = retriever.search(chunks, ["paid apprenticeship youth employment safeguard"], top_k=3)
    arabic = retriever.search(chunks, ["فرص التدريب والعمل للشباب"], top_k=3)
    assert french[0].chunk.file_name.startswith("DEMO · Résilience")
    assert french[0].chunk.page == 2
    assert english[0].chunk.file_name.startswith("DEMO · Emploi")
    assert arabic[0].chunk.file_name.startswith("DEMO · Emploi")


def test_abstention_for_unsupported_question() -> None:
    chunks = [chunk for doc in demo_documents() for chunk in doc.chunks]
    retriever = HybridRetriever()
    results = retriever.search(chunks, ["exact brand and model of a spacecraft chair"], top_k=3)
    assert not retriever.has_sufficient_evidence(results)
    result = answer_question(
        question="What exact brand and model is the chair on a spacecraft?",
        chunks=chunks,
        mode="qa",
        language="en",
    )
    assert result.abstained is True
    assert not result.citations


def test_citation_sanitizer_drops_fabricated_source_ids() -> None:
    cleaned = sanitize_citations("Evidence [S1], unsupported [S9].", ["S1"])
    assert "[S1]" in cleaned
    assert "[S9]" not in cleaned


def test_demo_corpus_is_marked_synthetic_and_cited() -> None:
    docs = demo_documents()
    assert len(docs) >= 3
    assert all(doc.is_demo for doc in docs)
    result = answer_question(
        question="What is the proposed reduction target by 2030?",
        chunks=[chunk for doc in docs for chunk in doc.chunks],
        mode="qa",
        language="en",
    )
    assert not result.abstained
    assert result.citations
    assert len(result.citations) == 1
    assert result.citations[0].is_demo
    assert "[S1]" in result.answer

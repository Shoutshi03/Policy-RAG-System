from __future__ import annotations

import hashlib
import re
import uuid
from collections.abc import Callable

import pymupdf as fitz
import pytesseract
from PIL import Image

from .models import Chunk, Document

MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_PAGES = 500
MAX_CHUNKS = 1000
CHUNK_CHARS = 1500
OVERLAP_CHARS = 180

_HEADING_NUMBER = re.compile(r"^\s*(?:\d+(?:\.\d+)*[.)]?\s+|[IVX]{1,6}[.)]\s+|(?:chapter|section|annexe|annex)\s+).{2,150}$", re.I)


def detect_language(text: str) -> str:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return "und"
    arabic = sum("\u0600" <= char <= "\u06ff" for char in letters) / len(letters)
    if arabic > 0.12:
        return "ar"
    lowered = text.lower()
    french_markers = sum(lowered.count(word) for word in (" le ", " la ", " les ", " des ", " une ", " pour ", "avec", " recommandation", "é", "à"))
    english_markers = sum(lowered.count(word) for word in (" the ", " and ", " of ", " for ", " with ", " policy", " evidence", " risk"))
    return "fr" if french_markers >= english_markers else "en"


def clean_text(text: str) -> str:
    text = text.replace("\u00ad", "").replace("\r", "\n")
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    text = re.sub(r"[\t\f\v]+", " ", text)
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    compact: list[str] = []
    for line in lines:
        if line and (not compact or line != compact[-1]):
            compact.append(line)
        elif not line and compact and compact[-1] != "":
            compact.append("")
    return "\n".join(compact).strip()


def _is_heading(line: str) -> bool:
    stripped = line.strip()
    if len(stripped) < 4 or len(stripped) > 150:
        return False
    if _HEADING_NUMBER.match(stripped):
        return True
    letters = [char for char in stripped if char.isalpha()]
    if len(letters) >= 5 and sum(char.isupper() for char in letters) / len(letters) > 0.78:
        return True
    return stripped.startswith(("Résumé exécutif", "Executive summary", "الملخص التنفيذي"))


def _split_long_paragraph(paragraph: str, size: int) -> list[str]:
    if len(paragraph) <= size:
        return [paragraph]
    sentences = re.split(r"(?<=[.!?؟؛])\s+", paragraph)
    result: list[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > size:
            result.append(current)
            current = current[-OVERLAP_CHARS:] + " " + sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        result.append(current)
    return result


def chunk_pages(page_texts: list[str], document_id: str, file_name: str, language: str, is_demo: bool = False) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page_number, page_text in enumerate(page_texts, start=1):
        section = "Texte courant"
        buffer: list[str] = []
        buffer_chars = 0

        def flush() -> None:
            nonlocal buffer, buffer_chars
            body = "\n\n".join(buffer).strip()
            if len(body) >= 35:
                chunks.append(Chunk(
                    chunk_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}:{page_number}:{len(chunks)}")),
                    document_id=document_id,
                    file_name=file_name,
                    page=page_number,
                    section=section[:255],
                    language=language,
                    text=body,
                    is_demo=is_demo,
                ))
            buffer = []
            buffer_chars = 0

        paragraphs: list[str] = []
        for raw_line in page_text.splitlines():
            line = raw_line.strip()
            if not line:
                if paragraphs and paragraphs[-1] != "":
                    paragraphs.append("")
                continue
            if _is_heading(line):
                if paragraphs:
                    paragraph = " ".join(item for item in paragraphs if item).strip()
                    if paragraph:
                        for piece in _split_long_paragraph(paragraph, CHUNK_CHARS):
                            if buffer_chars + len(piece) > CHUNK_CHARS and buffer:
                                flush()
                            buffer.append(piece)
                            buffer_chars += len(piece)
                    paragraphs.clear()
                flush()
                section = line[:255]
                continue
            paragraphs.append(line)
        if paragraphs:
            paragraph = " ".join(item for item in paragraphs if item).strip()
            for piece in _split_long_paragraph(paragraph, CHUNK_CHARS):
                if buffer_chars + len(piece) > CHUNK_CHARS and buffer:
                    flush()
                buffer.append(piece)
                buffer_chars += len(piece)
        flush()
    return chunks[:MAX_CHUNKS]


def _ocr_language(requested: str) -> str:
    try:
        available = set(pytesseract.get_languages(config=""))
    except Exception:
        available = {"eng"}
    preferred = [part for part in requested.split("+") if part in available]
    if not preferred:
        preferred = ["eng"] if "eng" in available else sorted(available - {"osd"})[:1]
    if not preferred:
        raise RuntimeError("Aucune langue Tesseract n’est installée sur le serveur.")
    return "+".join(preferred)


def extract_pdf(
    data: bytes,
    file_name: str,
    *,
    enable_ocr: bool = True,
    ocr_languages: str = "fra+eng",
    progress: Callable[[int, int, str], None] | None = None,
) -> Document:
    if not data:
        raise ValueError("Le fichier PDF est vide.")
    if len(data) > MAX_FILE_BYTES:
        raise ValueError("Le PDF dépasse la limite de 20 Mo.")
    if not file_name.lower().endswith(".pdf"):
        raise ValueError("Seuls les fichiers PDF sont acceptés.")
    if not data.lstrip().startswith(b"%PDF"):
        raise ValueError("Le fichier ne semble pas être un PDF valide.")
    safe_name = re.sub(r"[\\/\x00-\x1f]", "_", file_name).replace("..", "_")[:255]
    document_id = str(uuid.uuid4())
    file_hash = hashlib.sha256(data).hexdigest()
    page_texts: list[str] = []
    ocr_count = 0
    try:
        pdf = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise ValueError("Impossible d’ouvrir le PDF ; vérifiez que le fichier n’est pas endommagé ou protégé.") from exc
    with pdf:
        if pdf.is_encrypted:
            raise ValueError("Les PDF protégés par mot de passe ne sont pas pris en charge.")
        if len(pdf) < 1:
            raise ValueError("Le PDF ne contient aucune page.")
        if len(pdf) > MAX_PAGES:
            raise ValueError(f"Le PDF dépasse la limite de {MAX_PAGES} pages.")
        requested_ocr = _ocr_language(ocr_languages) if enable_ocr else ""
        for index, page in enumerate(pdf):
            if progress:
                progress(index + 1, len(pdf), f"Extraction de la page {index + 1}/{len(pdf)}")
            text = clean_text(page.get_text("text", sort=True))
            if enable_ocr and len(text) < 80:
                try:
                    pixmap = page.get_pixmap(matrix=fitz.Matrix(1.65, 1.65), alpha=False)
                    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
                    recognized = pytesseract.image_to_string(
                        image,
                        lang=requested_ocr,
                        config="--oem 1 --psm 6",
                        timeout=45,
                    )
                    recognized = clean_text(recognized)
                    if len(recognized) > len(text):
                        text = recognized
                        ocr_count += 1
                except Exception as exc:
                    raise RuntimeError(
                        f"OCR impossible à la page {index + 1}. Installez les langues Tesseract requises ou désactivez l’OCR."
                    ) from exc
            page_texts.append(text)
    language = detect_language("\n".join(page_texts[:5]))
    chunks = chunk_pages(page_texts, document_id, safe_name, language)
    if not chunks:
        suffix = " Activez l’OCR ou fournissez un PDF avec texte sélectionnable." if not enable_ocr else " Vérifiez que les langues OCR requises sont installées."
        raise ValueError("Aucun texte exploitable n’a été extrait." + suffix)
    return Document(document_id, safe_name, file_hash, language, len(page_texts), chunks, False, ocr_count)

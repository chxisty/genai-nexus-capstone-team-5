"""
helpdesk/ingest.py
------------------
Extract, clean, OCR, and chunk all documents from the data/ folder.

Each chunk is a dict:
  {
    "text": str,           # the actual text content
    "doc": str,            # filename (e.g. "irctc_refund_rules_jan2026.pdf")
    "page": int | None,    # page number (1-based), None for HTML docs
    "chunk_id": str        # unique id like "irctc_refund_rules_jan2026.pdf::p3::c1"
  }
"""

import os
import re
import io
import json
import logging
from pathlib import Path
from typing import Optional

# Resilient PDF library imports
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import pypdf
except ImportError:
    pypdf = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

from PIL import Image
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths & Dynamic Dataset Detection
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

def get_documents_dir() -> Path:
    """
    Auto-detect documents directory in both:
      - data/documents/
      - data/rail_capstone_pack/documents/
    Or extracts rail_capstone_pack.zip if data is missing.
    """
    candidates = [
        DATA_DIR / "documents",
        DATA_DIR / "rail_capstone_pack" / "documents",
        PROJECT_ROOT / "rail_capstone_pack" / "documents",
    ]
    for c in candidates:
        if c.exists() and c.is_dir() and any(c.iterdir()):
            return c

    # Check if rail_capstone_pack.zip exists and auto-extract
    zip_candidates = [
        PROJECT_ROOT / "rail_capstone_pack.zip",
        DATA_DIR / "rail_capstone_pack.zip",
    ]
    for z in zip_candidates:
        if z.exists():
            import zipfile
            logger.info("Extracting %s into %s ...", z, DATA_DIR)
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(z, "r") as zip_ref:
                zip_ref.extractall(DATA_DIR)
            for c in candidates:
                if c.exists() and c.is_dir():
                    return c

    return DATA_DIR / "documents"

DOCUMENTS_DIR = get_documents_dir()


# ---------------------------------------------------------------------------
# Dynamic OCR Engine
# ---------------------------------------------------------------------------
_ocr_engine = None


def _get_rapid_ocr():
    """Lazily load RapidOCR engine for automated OCR on scanned pages."""
    global _ocr_engine
    if _ocr_engine is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _ocr_engine = RapidOCR()
            logger.info("Initialized RapidOCR engine.")
        except Exception as e:
            logger.debug("RapidOCR not available: %s", e)
    return _ocr_engine


def _ocr_image(pil_img: Image.Image) -> str:
    """
    Run automated OCR on a PIL Image using RapidOCR or pytesseract.
    Returns the extracted text.
    """
    # 1. Try RapidOCR (ONNX runtime, universal)
    engine = _get_rapid_ocr()
    if engine is not None:
        try:
            import numpy as np
            rgb_arr = np.array(pil_img.convert("RGB"))
            result, _ = engine(rgb_arr)
            if result:
                lines = [item[1] for item in result if item and len(item) > 1 and item[1]]
                text = "\n".join(lines).strip()
                if len(text) > 20:
                    return text
        except Exception as e:
            logger.warning("RapidOCR extraction error: %s", e)

    # 2. Try pytesseract if installed
    try:
        import pytesseract
        for t_path in [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]:
            if os.path.exists(t_path):
                pytesseract.pytesseract.tesseract_cmd = t_path
                break
        text = pytesseract.image_to_string(pil_img, lang="eng+hin").strip()
        if len(text) > 20:
            return text
    except Exception as e:
        logger.debug("pytesseract error: %s", e)

    return ""


# ---------------------------------------------------------------------------
# PDF Extraction
# ---------------------------------------------------------------------------

def _extract_pdf(path: Path) -> list[dict]:
    """
    Extract all pages from a PDF.
    Supports PyMuPDF (fitz), pdfplumber, and pypdf.
    Scanned pages trigger automated OCR.
    """
    doc_name = path.name
    pages_data = []

    # 1. Try PyMuPDF if available
    if fitz is not None:
        try:
            pdf = fitz.open(str(path))
            for page_num in range(len(pdf)):
                page = pdf[page_num]
                text = (page.get_text("text") or "").strip()
                if len(text) < 50:
                    try:
                        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
                        pil_img = Image.open(io.BytesIO(pix.tobytes("png")))
                        ocr_text = _ocr_image(pil_img)
                        if ocr_text:
                            text = ocr_text
                    except Exception as e:
                        logger.warning("OCR on %s p%d failed: %s", doc_name, page_num + 1, e)
                text = _clean_text(text)
                if text:
                    pages_data.append({"text": text, "doc": doc_name, "page": page_num + 1})
            pdf.close()
            if pages_data:
                return pages_data
        except Exception as e:
            logger.warning("PyMuPDF failed on %s: %s", doc_name, e)

    # 2. Try pdfplumber if available
    if pdfplumber is not None:
        try:
            with pdfplumber.open(str(path)) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_num = i + 1
                    try:
                        text = (page.extract_text() or "").strip()
                    except Exception:
                        text = ""
                    if len(text) < 50:
                        try:
                            pil_img = page.to_image(resolution=200).original
                            ocr_text = _ocr_image(pil_img)
                            if ocr_text:
                                text = ocr_text
                        except Exception as e:
                            logger.warning("OCR on %s p%d failed: %s", doc_name, page_num, e)
                    text = _clean_text(text)
                    if text:
                        pages_data.append({"text": text, "doc": doc_name, "page": page_num})
            if pages_data:
                return pages_data
        except Exception as e:
            logger.warning("pdfplumber failed on %s: %s", doc_name, e)

    # 3. Try pypdf
    if pypdf is not None:
        try:
            reader = pypdf.PdfReader(str(path))
            for i, page in enumerate(reader.pages):
                page_num = i + 1
                try:
                    text = (page.extract_text() or "").strip()
                except Exception:
                    text = ""
                if len(text) < 50 and len(page.images) > 0:
                    ocr_chunks = []
                    for img in page.images:
                        try:
                            pil_img = Image.open(io.BytesIO(img.data))
                            ocr_t = _ocr_image(pil_img)
                            if ocr_t:
                                ocr_chunks.append(ocr_t)
                        except Exception as e:
                            logger.warning("OCR image extraction failed: %s", e)
                    if ocr_chunks:
                        text = "\n".join(ocr_chunks)
                text = _clean_text(text)
                if text:
                    pages_data.append({"text": text, "doc": doc_name, "page": page_num})
            if pages_data:
                return pages_data
        except Exception as e:
            logger.warning("pypdf failed on %s: %s", doc_name, e)

    return pages_data


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def _extract_html(path: Path) -> list[dict]:
    """
    Extract text from an HTML file.
    HTML documents have no page numbers; page is set to None.
    """
    doc_name = path.name
    try:
        html = path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(html, "lxml" if "lxml" in BeautifulSoup.__module__ else "html.parser")

        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        text = _clean_text(text)

        if not text:
            return []

        return [{
            "text": text,
            "doc": doc_name,
            "page": None,
        }]
    except Exception as e:
        logger.error("Cannot parse %s: %s", path, e)
        return []


# ---------------------------------------------------------------------------
# Text cleaning & Chunking
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """Normalise whitespace and remove junk characters."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_into_chunks(page_dict: dict, max_chars: int = 900, overlap: int = 150) -> list[dict]:
    """
    Split a page's text into overlapping chunks. Preserves doc and page.
    """
    text: str = page_dict["text"]
    doc: str = page_dict["doc"]
    page = page_dict["page"]

    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    chunks = []
    current = ""
    chunk_idx = 1

    def _save(buf: str):
        nonlocal chunk_idx
        buf = buf.strip()
        if buf:
            pid = f"p{page}" if page else "pNone"
            chunk_id = f"{doc}::{pid}::c{chunk_idx}"
            chunks.append({
                "text": buf,
                "doc": doc,
                "page": page,
                "chunk_id": chunk_id,
            })
            chunk_idx += 1

    for para in paragraphs:
        if current and len(current) + len(para) + 2 > max_chars:
            _save(current)
            current = current[-overlap:] + "\n\n" + para if overlap else para
        else:
            current = (current + "\n\n" + para).strip() if current else para

    _save(current)
    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_all_chunks() -> list[dict]:
    """
    Read every document in data/documents/ and return a flat list of chunks.
    """
    doc_dir = get_documents_dir()
    if not doc_dir.exists():
        raise FileNotFoundError(
            f"Documents directory not found: {doc_dir}\n"
            "Please extract the dataset into data/ first."
        )

    all_chunks = []

    for path in sorted(doc_dir.iterdir()):
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            pages = _extract_pdf(path)
        elif suffix in (".html", ".htm"):
            pages = _extract_html(path)
        else:
            continue

        for page_dict in pages:
            page_chunks = _split_into_chunks(page_dict)
            all_chunks.extend(page_chunks)

    logger.info("Total chunks loaded: %d", len(all_chunks))
    return all_chunks

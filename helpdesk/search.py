"""
helpdesk/search.py
------------------
Hybrid retrieval: BM25 keyword search + FAISS semantic search.

The index is built once by build_index.py and saved to disk.
search() loads it lazily on first call.
"""

from __future__ import annotations

import os
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Optional

# Ensure offline operation without network pauses
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import warnings
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)


import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INDEX_DIR = Path(__file__).parent.parent / "index"
CHUNKS_PATH = INDEX_DIR / "chunks.json"
BM25_PATH = INDEX_DIR / "bm25.pkl"
FAISS_PATH = INDEX_DIR / "faiss.index"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Module-level cache (loaded once per process)
# ---------------------------------------------------------------------------
_chunks: Optional[list[dict]] = None
_bm25: Optional[BM25Okapi] = None
_faiss_index: Any = None
_embed_model: Optional[SentenceTransformer] = None


# ---------------------------------------------------------------------------
# Build helpers (called by build_index.py)
# ---------------------------------------------------------------------------

def build_index(chunks: list[dict]) -> None:
    """
    Build and save the BM25 index and FAISS index from a list of chunks.
    Each chunk must have keys: text, doc, page, chunk_id.
    """
    INDEX_DIR.mkdir(exist_ok=True)

    # Save chunks
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    logger.info("Saved %d chunks to %s", len(chunks), CHUNKS_PATH)

    # Build BM25
    tokenised = [_tokenise(c["text"]) for c in chunks]
    if BM25Okapi is not None:
        bm25 = BM25Okapi(tokenised)
        with open(BM25_PATH, "wb") as f:
            pickle.dump(bm25, f)
        logger.info("BM25 index saved.")

    # Build semantic embeddings if model available
    model = _get_embed_model()
    if model is not None:
        texts = [c["text"] for c in chunks]
        logger.info("Encoding %d chunks with %s …", len(chunks), EMBED_MODEL_NAME)
        embeddings = model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype="float32")
        np.save(INDEX_DIR / "embeddings.npy", embeddings)

        if faiss is not None:
            dim = embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(embeddings)
            faiss.write_index(index, str(FAISS_PATH))
            logger.info("FAISS index saved (%d vectors, dim=%d).", index.ntotal, dim)
    else:
        logger.info("SentenceTransformer not installed, using BM25 and keyword ranking.")


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

_embeddings: Optional[np.ndarray] = None


def _get_embed_model() -> Optional["SentenceTransformer"]:
    global _embed_model
    if _embed_model is None and SentenceTransformer is not None:
        try:
            import contextlib, io
            with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        except Exception as e:
            logger.debug("Could not load SentenceTransformer (%s). Using BM25.", e)
            _embed_model = None
    return _embed_model


def _load_index():
    global _chunks, _bm25, _faiss_index, _embeddings
    if _chunks is not None:
        return  # Already loaded

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            "Index not found. Please run: python build_index.py"
        )

    with open(CHUNKS_PATH, encoding="utf-8") as f:
        _chunks = json.load(f)

    if BM25_PATH.exists():
        try:
            with open(BM25_PATH, "rb") as f:
                _bm25 = pickle.load(f)
        except Exception as e:
            logger.warning("Could not load BM25 index: %s", e)

    if FAISS_PATH.exists() and faiss is not None:
        try:
            _faiss_index = faiss.read_index(str(FAISS_PATH))
        except Exception as e:
            logger.warning("Could not load FAISS index: %s", e)

    emb_path = INDEX_DIR / "embeddings.npy"
    if emb_path.exists():
        try:
            _embeddings = np.load(emb_path)
        except Exception as e:
            logger.warning("Could not load embeddings.npy: %s", e)

    logger.info("Loaded search index: %d chunks.", len(_chunks))


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> list[str]:
    """Simple whitespace + lower-case tokeniser for BM25."""
    import re
    return re.findall(r"\b\w+\b", text.lower())


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search(query: str, top_k: int = 8) -> list[dict]:
    """
    Hybrid BM25 + FAISS / semantic search.
    Returns top_k chunks ranked by combined score.
    Each returned chunk has: text, doc, page, chunk_id, score.
    """
    _load_index()

    n = len(_chunks)
    if n == 0:
        return []

    # --- BM25 scores ---
    query_tokens = _tokenise(query)
    if _bm25 is not None:
        bm25_scores = np.array(_bm25.get_scores(query_tokens), dtype="float32")
        bm25_max = bm25_scores.max()
        if bm25_max > 0:
            bm25_scores = bm25_scores / bm25_max
    else:
        bm25_scores = np.zeros(n, dtype="float32")

    # --- Semantic scores ---
    model = _get_embed_model()
    faiss_scores = np.zeros(n, dtype="float32")
    if model is not None:
        try:
            q_emb = model.encode([query], normalize_embeddings=True).astype("float32")
            if _faiss_index is not None:
                faiss_k = min(top_k * 4, n)
                distances, indices = _faiss_index.search(q_emb, faiss_k)
                for dist, idx in zip(distances[0], indices[0]):
                    if 0 <= idx < n:
                        faiss_scores[idx] = float(dist)
            elif _embeddings is not None:
                # Cosine similarity via inner product on normalized embeddings
                faiss_scores = np.dot(_embeddings, q_emb[0])
        except Exception as e:
            logger.warning("Semantic search error: %s", e)

    # --- Combined score ---
    if faiss_scores.max() > 0:
        combined = 0.5 * bm25_scores + 0.5 * faiss_scores
    else:
        combined = bm25_scores.copy()

    # Boost chunks that contain exact train numbers, rule numbers, codes
    boost_patterns = _extract_boost_terms(query)
    q_lower = query.lower()
    for i, chunk in enumerate(_chunks):
        txt_lower = chunk["text"].lower()
        doc_name = chunk.get("doc", "").lower()

        # Term frequency boost
        for term in boost_patterns:
            if term.lower() in txt_lower:
                combined[i] += 0.25

        # Domain-specific priority boosts
        if "vande bharat" in q_lower and "vande_bharat" in doc_name:
            combined[i] += 0.4
        if ("tatkal" in q_lower or "aadhaar" in q_lower) and "tatkal" in doc_name:
            combined[i] += 0.4
        if ("chart" in q_lower or "charting" in q_lower) and "reservation_charts" in doc_name:
            combined[i] += 0.4
        if ("advance" in q_lower or "arp" in q_lower or "60 day" in q_lower or "taj express" in q_lower) and "advance_reservation" in doc_name:
            combined[i] += 0.4
        if ("clerkage" in q_lower or "waitlist" in q_lower or "rac" in q_lower) and ("refund_rules_2015" in doc_name or "irctc_refund_rules_jan2026" in doc_name):
            combined[i] += 0.3

        # Newer document priority for 2026 documents
        if "2026" in doc_name:
            combined[i] += 0.1


    # Get top_k indices
    top_indices = np.argsort(combined)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if combined[idx] > 0:
            chunk = dict(_chunks[idx])
            chunk["score"] = float(combined[idx])
            results.append(chunk)

    return results


def _extract_boost_terms(query: str) -> list[str]:
    """
    Extract train numbers, rule codes, circular numbers, etc. from the query
    to give exact-match chunks a score boost.
    """
    import re
    terms = []

    # Train numbers: 4-5 digit numbers
    for m in re.finditer(r"\b\d{4,5}\b", query):
        terms.append(m.group())

    # Circular numbers like CC 08, CC-10, cc10
    for m in re.finditer(r"\bcc[-\s]?\d{2}\b", query, re.IGNORECASE):
        terms.append(m.group())

    # GSR references
    for m in re.finditer(r"g\.s\.r\.?\s*\d+", query, re.IGNORECASE):
        terms.append(m.group())

    # Percentage / money amounts
    for m in re.finditer(r"\b\d+%|\brs\.?\s*\d+\b", query, re.IGNORECASE):
        terms.append(m.group())

    # Key railway terms
    keywords = [
        "tatkal", "vande bharat", "rajdhani", "shatabdi", "waitlist",
        "refund", "cancellation", "tdr", "aadhaar", "pnr",
        "advance reservation", "sleeper", "chart"
    ]
    for kw in keywords:
        if kw in query.lower():
            terms.append(kw)

    return terms


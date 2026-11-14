"""
build_index.py
--------------
One-time script to ingest all documents and build the search index.

Run once after extracting the dataset:
    python build_index.py

This creates the index/ directory with:
  - index/chunks.json   (all text chunks with metadata)
  - index/bm25.pkl      (BM25 keyword index)
  - index/faiss.index   (FAISS semantic vector index)
"""

import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    start = time.time()
    logger.info("=" * 60)
    logger.info("Railway Help Desk — Build Index")
    logger.info("Team: NEUROVERSE")
    logger.info("=" * 60)

    # Check data directory via dynamic locator
    from helpdesk.ingest import get_documents_dir, load_all_chunks
    doc_dir = get_documents_dir()
    if not doc_dir.exists():
        logger.error(
            "Documents directory not found.\n"
            "Please ensure rail_capstone_pack.zip or data/documents/ is available."
        )
        sys.exit(1)

    # Ingest documents
    logger.info("Step 1/2: Ingesting documents from %s …", doc_dir)
    chunks = load_all_chunks()

    if not chunks:
        logger.error("No chunks produced. Check that data/documents/ has content.")
        sys.exit(1)

    logger.info("Produced %d chunks from documents.", len(chunks))

    # Build search index
    logger.info("Step 2/2: Building BM25 + FAISS index …")
    from helpdesk.search import build_index
    build_index(chunks)

    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info("Index built successfully in %.1f seconds.", elapsed)
    logger.info("You can now run: python ask.py \"your question\"")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()


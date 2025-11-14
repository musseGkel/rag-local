# ingest_goldberg_brass.py

from __future__ import annotations

# --- SQLite shim for Chroma (same as in ingest_from_chapter_map.py) ---
try:
    import pysqlite3, sys  # type: ignore
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass

import os
from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader

# Reuse helpers and config from your existing ingest script
from ingest_from_chapter_map import (
    ROOT,
    DB_DIR,
    EMBED_MODEL,
    chunk_documents,
    has_sql_example,
    coerce_metadata,
)

# CHANGE THIS to the real path of the PDF on the server
# Example: ROOT / "corpus" / "papers" / "Goldberg-Brass.pdf"
PDF_PATH = ROOT / "corpus" / "papers" / "Goldberg-Brass.pdf"


def ingest_goldberg_brass():
    print(f"[info] ROOT={ROOT}")
    print(f"[info] DB_DIR={DB_DIR}")
    print(f"[info] PDF_PATH exists? {PDF_PATH.exists()} → {PDF_PATH}")

    if not PDF_PATH.exists():
        raise FileNotFoundError(f"Goldberg-Brass PDF not found at: {PDF_PATH}")

    # Ensure DB dir exists
    DB_DIR.mkdir(parents=True, exist_ok=True)

    # Persistent Chroma client (same as main ingest)
    client = chromadb.PersistentClient(path=str(DB_DIR))

    # Embeddings (same model + normalize_embeddings=True)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )

    # Same collection name: "sqlkb"
    vectordb = Chroma(
        client=client,
        collection_name="sqlkb",
        embedding_function=embeddings,
    )

    # ---- Load full paper as PDF ----
    loader = PyPDFLoader(str(PDF_PATH))
    pages = loader.load()
    total_pages = len(pages)
    print(f"[info] Loaded {total_pages} pages from Goldberg-Brass")

    # For a paper we ingest the full range: pages 1..total_pages
    s, e = 1, total_pages
    sub = pages[s - 1 : e]

    # ---- Base metadata (same structure as ingest_from_chapter_map for 'paper') ----
    tags_str = ", ".join(
        [
            "paper",
            "sql",
            "semantic errors",
            "sqllint",
            "goldberg-brass",
        ]
    )

    base_meta = {
        "doc_type": "paper",
        "resource_id": "goldberg_brass_2004",
        "section": "full_paper",
        "section_locator": f"pp.{s}-{e}",
        "source_path": str(PDF_PATH),
        "tags": tags_str,
    }

    # Attach base metadata to each page
    for p in sub:
        p.metadata.update(base_meta)
        p.metadata = coerce_metadata(p.metadata)

    # ---- Chunking (papers: 900 / 120, same as in your code) ----
    size, overlap = 900, 120
    docs = chunk_documents(sub, size=size, overlap=overlap)

    # Add per-chunk metadata
    for i, d in enumerate(docs):
        d.metadata.update(
            {
                "chunk_index": i,
                "has_sql_example": has_sql_example(d.page_content),
            }
        )
        d.metadata = coerce_metadata(d.metadata)

    # ---- Write to Chroma ----
    if docs:
        vectordb.add_documents(docs)
        print(f"✅ Ingested Goldberg-Brass → {len(docs)} chunks")
    else:
        print("⚠️ No chunks were produced; nothing ingested.")


if __name__ == "__main__":
    ingest_goldberg_brass()

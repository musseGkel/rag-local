# ingest_from_chapter_map.py

# --- SQLite shim for Chroma (for environments without system sqlite3) ---
try:
    import pysqlite3, sys  # type: ignore
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass

import os
import csv
from pathlib import Path
from typing import List, Iterable

from langchain.schema import Document  # if this errors on LC>=0.2, use: from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Chroma via langchain_chroma (no .persist(); use a PersistentClient)
from langchain_chroma import Chroma
import chromadb

from langchain_huggingface import HuggingFaceEmbeddings


# ---------- Paths & Config ----------
# Project root = folder containing this script
ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

CHAPTER_MAP = ROOT / "metadata" / "chapter_map.csv"
DB_DIR = ROOT / "db"

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")  # solid small model


# ---------- Helpers ----------
def chunk_documents(pages: List[Document], size: int = 1200, overlap: int = 150) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(pages)


def has_sql_example(text: str) -> bool:
    T = text.upper()
    keys = [
        "SELECT", "FROM", "WHERE", "JOIN", "GROUP BY", "HAVING", "ORDER BY",
        "UNION", "INTERSECT", "EXCEPT", "WITH", "COUNT", "AVG", "SUM", "MIN", "MAX",
    ]
    return any(k in T for k in keys) and (";" in text or "\n" in text)


def require_csv_columns(fieldnames, required):
    missing = [c for c in required if c not in (fieldnames or [])]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}. Found: {fieldnames}")


def split_tags(raw: str) -> Iterable[str]:
    """
    Accepts tags separated by semicolons OR commas.
    Returns iterable of trimmed strings (no empties).
    """
    if not raw:
        return []
    # Replace semicolons with commas, then split
    raw = raw.replace(";", ",")
    return [t.strip() for t in raw.split(",") if t.strip()]


def coerce_metadata(md: dict) -> dict:
    """
    Chroma only accepts str, int, float, bool, None (or SparseVector) as metadata values.
    Convert lists/tuples/sets to a comma-separated string.
    Convert dicts/other objects to JSON/string.
    """
    import json

    out = {}
    for k, v in md.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, (list, tuple, set)):
            out[k] = ", ".join(map(str, v))
        elif isinstance(v, dict):
            out[k] = json.dumps(v, ensure_ascii=False, sort_keys=True)
        else:
            out[k] = str(v)
    return out


# ---------- Main Ingest ----------
def ingest():
    # Sanity prints
    print(f"[info] ROOT={ROOT}")
    print(f"[info] CHAPTER_MAP exists? {CHAPTER_MAP.exists()} → {CHAPTER_MAP}")
    DB_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[info] DB_DIR: {DB_DIR}")

    # Persistent Chroma client (no vectordb.persist() needed)
    client = chromadb.PersistentClient(path=str(DB_DIR))

    # Embeddings (normalize for cosine sim with bge models)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )

    # Vector store bound to persistent client
    vectordb = Chroma(
        client=client,
        collection_name="sqlkb",
        embedding_function=embeddings,
    )

    rows = 0

    with CHAPTER_MAP.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        require_csv_columns(
            reader.fieldnames,
            required=["resource_id", "path", "start_page", "end_page", "section"],
        )

        for r in reader:
            # Resolve PDF path (absolute if given, else relative to ROOT)
            raw_path = (r.get("path") or "").strip()
            pdf_path = Path(raw_path) if Path(raw_path).is_absolute() else (ROOT / raw_path)

            if not pdf_path.exists():
                print(f"⚠️  Missing file: {pdf_path} — skipping")
                continue

            # Page range (CSV uses 1-based inclusive)
            try:
                s = max(1, int(r["start_page"]))
                e = int(r["end_page"])
            except Exception:
                print(f"⚠️  Invalid page numbers for {r.get('resource_id')} — skipping")
                continue

            if e < s:
                print(f"⚠️  end_page < start_page for {r.get('resource_id')}:{r.get('section')} — skipping")
                continue

            # Load pages
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()  # 0-based pages
            sub = pages[s - 1 : e]  # slice inclusive in CSV → exclusive in Python

            # Base metadata (normalize tags to a single string)
            tags_iter = split_tags(r.get("tags", ""))
            tags_str = ", ".join(tags_iter) if tags_iter else ""

            base_meta = {
                "doc_type": (r.get("kind") or "textbook"),
                "resource_id": r["resource_id"],
                "section": r.get("section", ""),
                "section_locator": r.get("section_locator", ""),
                "source_path": str(pdf_path),
                "tags": tags_str,  # <- string, not list
            }

            for p in sub:
                p.metadata.update(base_meta)

            # Chunk & enrich
            docs = chunk_documents(sub, size=1200, overlap=150)
            for i, d in enumerate(docs):
                d.metadata.update(
                    {
                        "chunk_index": i,
                        "has_sql_example": has_sql_example(d.page_content),
                    }
                )
                # Ensure all metadata values are Chroma-safe
                d.metadata = coerce_metadata(d.metadata)

            if docs:
                vectordb.add_documents(docs)
                rows += 1
                print(f"✅ Ingested: {r['resource_id']} [{s}-{e}] → {len(docs)} chunks")

    print(f"🎯 Done. Sources ingested from {CHAPTER_MAP}: {rows} rows.")


if __name__ == "__main__":
    ingest()

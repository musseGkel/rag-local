# ingest_goldberg_brass_minimal.py

from __future__ import annotations

# --- SQLite shim for Chroma (same pattern as your other scripts) ---
try:
    import pysqlite3, sys  # type: ignore
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass

import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader


# ---------- Paths & basic config ----------
ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DB_DIR = ROOT / "db"

# Adjust this to where the PDF is on the server
PDF_PATH = ROOT / "corpus" / "papers" / "Goldberg-Brass.pdf"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# Chunking parameters for papers (same as your ingest script)
CHUNK_SIZE = 900
CHUNK_OVERLAP = 120


# ---------- Helpers ----------
def has_sql_example(text: str) -> bool:
    T = text.upper()
    keys = [
        "SELECT", "FROM", "WHERE", "JOIN", "GROUP BY", "HAVING", "ORDER BY",
        "UNION", "INTERSECT", "EXCEPT", "WITH", "COUNT", "AVG", "SUM", "MIN", "MAX",
    ]
    return any(k in T for k in keys) and (";" in text or "\n" in text)


def coerce_metadata(md: dict) -> dict:
    """
    Chroma only accepts str, int, float, bool, None (or SparseVector) as metadata values.
    Convert lists/tuples/sets to a comma-separated string.
    Convert dicts/other objects to JSON/string.
    """
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


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """
    Simple character-based chunking with fixed size and overlap.
    Not as fancy as RecursiveCharacterTextSplitter, but respects the
    same chunk_size / overlap logic.
    """
    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        # next window starts chunk_size - overlap characters after current start
        start = end - overlap

    return chunks


def embed_texts(model: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    embs = model.encode(texts, normalize_embeddings=True)
    return embs.tolist()


def load_pdf_pages(path: Path) -> list[tuple[int, str]]:
    """
    Returns a list of (page_number, text) tuples, 1-based page numbers.
    """
    reader = PdfReader(str(path))
    pages = []
    for i, p in enumerate(reader.pages, start=1):
        # p.extract_text() may be None if page is weirdly encoded; handle that
        txt = p.extract_text() or ""
        txt = txt.strip()
        if txt:
            pages.append((i, txt))
    return pages


# ---------- Main ingest ----------
def ingest_goldberg_brass_minimal():
    print(f"[info] ROOT={ROOT}")
    print(f"[info] DB_DIR={DB_DIR}")
    print(f"[info] PDF_PATH exists? {PDF_PATH.exists()} → {PDF_PATH}")

    if not PDF_PATH.exists():
        raise FileNotFoundError(f"Goldberg-Brass PDF not found at: {PDF_PATH}")

    DB_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Connect to existing Chroma DB
    client = chromadb.PersistentClient(path=str(DB_DIR))
    coll = client.get_collection("sqlkb")

    # 2) Load embedding model (same family as used before)
    print(f"[info] Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    # 3) Load PDF pages
    pages = load_pdf_pages(PDF_PATH)
    total_pages = len(pages)
    print(f"[info] Loaded {total_pages} text pages from Goldberg-Brass")

    if total_pages == 0:
        print("⚠️ No text extracted from PDF; aborting.")
        return

    # For a paper we ingest the full range: pages 1..total_pages
    s, e = 1, total_pages

    # ---- Base metadata (mirroring your ingest_from_chapter_map setup for papers) ----
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

    all_texts: list[str] = []
    all_metadatas: list[dict] = []
    all_ids: list[str] = []

    chunk_index = 0

    # 4) Per-page chunking with size=900, overlap=120
    for page_num, page_text in pages:
        page_chunks = chunk_text(page_text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        for ch in page_chunks:
            md = dict(base_meta)
            md.update(
                {
                    "page_number": page_num,
                    "chunk_index": chunk_index,
                    "has_sql_example": has_sql_example(ch),
                }
            )
            md = coerce_metadata(md)

            all_texts.append(ch)
            all_metadatas.append(md)
            all_ids.append(f"goldberg_brass_2004_p{page_num}_c{chunk_index}")
            chunk_index += 1

    if not all_texts:
        print("⚠️ No chunks produced; nothing ingested.")
        return

    print(f"[info] Produced {len(all_texts)} chunks; embedding and adding to Chroma...")

    # 5) Embed and add to Chroma
    embeddings = embed_texts(model, all_texts)

    coll.add(
        documents=all_texts,
        metadatas=all_metadatas,
        ids=all_ids,
        embeddings=embeddings,
    )

    print(f"✅ Ingested Goldberg-Brass → {len(all_texts)} chunks")


if __name__ == "__main__":
    ingest_goldberg_brass_minimal()

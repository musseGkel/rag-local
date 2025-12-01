from __future__ import annotations

# SQLite shim for Chroma (for environments without system sqlite3)
try:
    import pysqlite3, sys  # type: ignore
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass

try:
    from langchain_core.documents import Document  # LC >= 0.2
except Exception:
    from langchain.schema import Document          # LC < 0.2

import os
import csv
from pathlib import Path
from typing import List, Iterable

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma
import chromadb

from langchain_huggingface import HuggingFaceEmbeddings


# ---------- Paths and config ----------

ROOT        = Path(os.getenv("ROOT", "/opt/rag"))
CORPUS_ROOT = Path(os.getenv("RAG_CORPUS", str(ROOT / "corpus")))

# separate DB folder for bigger chunks
DB_DIR      = Path(os.getenv("RAG_DB_DIR_BIG", str(ROOT / "db_bigchunks")))

# chapter_map.csv location
CHAPTER_MAP = Path(os.getenv("RAG_CHAPTER_MAP", str(CORPUS_ROOT / "metadata" / "chapter_map.csv")))

# HF cache per user (safe on multi user server)
os.environ.setdefault("HF_HOME", "/opt/rag/models")
os.environ.pop("TRANSFORMERS_CACHE", None)

EMBED_MODEL      = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
COLLECTION_NAME  = os.getenv("RAG_COLLECTION_BIG", "sqlkb_bigchunks")

# bigger chunks for everything except papers
BIG_CHUNK_SIZE        = 2600
BIG_CHUNK_OVERLAP     = 300
PAPER_CHUNK_SIZE      = 900
PAPER_CHUNK_OVERLAP   = 120


# ---------- Helpers ----------

def chunk_documents(pages: List[Document], size: int = BIG_CHUNK_SIZE, overlap: int = BIG_CHUNK_OVERLAP) -> List[Document]:
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
    if not raw:
        return []
    raw = raw.replace(";", ",")
    return [t.strip() for t in raw.split(",") if t.strip()]


def coerce_metadata(md: dict) -> dict:
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


def read_taxonomy_records_from_csv(path: Path) -> List[Document]:
    import csv as _csv

    def pick(d, *keys):
        for k in keys:
            if k in d and d[k] != "":
                return d[k]
        return ""

    def truthy(v: str) -> bool:
        t = (v or "").strip().lower()
        return t in {"y", "yes", "true", "1", "✓", "x"}

    docs: List[Document] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        rows = []
        for raw in reader:
            norm = {(k or "").strip(): (v or "").strip() for k, v in raw.items()}
            rows.append(norm)

    for e in rows:
        taxonomy_id = pick(e, "taxonomy_id", "Id", "ID")
        category     = pick(e, "category", "Cat")
        name         = pick(e, "name", "Cat Name", "Error Name")
        err_desc     = pick(e, "Error Description")
        lit_example  = pick(e, "Literature example", "Example", "example_sql")
        long_desc    = pick(e, "Description", "description")
        reqs         = pick(e, "Requirements for assignment generation", "requirements")
        diff         = pick(e, "Detection Difficulty", "difficulty")
        prio         = pick(e, "Detection Priority", "priority")

        appears = []
        if truthy(pick(e, "Detected - Base query")):
            appears.append("base")
        if truthy(pick(e, "Detected - Subquery")):
            appears.append("subquery")
        if truthy(pick(e, "Detected - CTE")):
            appears.append("cte")

        text = (
            f"[{category}] {name}\n\n"
            f"{long_desc or err_desc}\n\n"
            f"Example:\n{lit_example}\n\n"
            f"Detection:\n"
            f"- Difficulty: {diff or 'n/a'}\n"
            f"- Priority: {prio or 'n/a'}\n"
            f"- Appears in: {', '.join(appears) if appears else 'n/a'}\n\n"
            f"Assignment hints: {reqs or '—'}"
        )

        meta = {
            "doc_type": "taxonomy",
            "taxonomy_id": int(taxonomy_id) if str(taxonomy_id).isdigit() else taxonomy_id,
            "category": category,
            "name": name,
            "priority": int(prio) if str(prio).isdigit() else prio,
            "difficulty": diff,
            "appears_in": ", ".join(appears) if appears else "",
            "source": "Taxonomy notes - Errors.csv",
            "source_path": str(path),
        }
        docs.append(Document(page_content=text, metadata=coerce_metadata(meta)))
    return docs


# ---------- Main ingest ----------

def ingest():
    print(f"[info] ROOT={ROOT}")
    print(f"[info] CHAPTER_MAP exists? {CHAPTER_MAP.exists()} -> {CHAPTER_MAP}")
    DB_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[info] DB_DIR (big chunks): {DB_DIR}")

    client = chromadb.PersistentClient(path=str(DB_DIR))

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )

    vectordb = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    rows = 0

    with CHAPTER_MAP.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        require_csv_columns(reader.fieldnames, required=["resource_id", "path", "section"])

        for r in reader:
            raw_path = (r.get("path") or "").strip()
            data_path = Path(raw_path) if Path(raw_path).is_absolute() else (ROOT / raw_path)
            kind = (r.get("kind") or "textbook").strip().lower()

            if not data_path.exists():
                print(f"Warning  missing file: {data_path}  skipping")
                continue

            if kind == "taxonomy":
                docs = []
                if data_path.suffix.lower() in {".csv", ".tsv"}:
                    docs = read_taxonomy_records_from_csv(data_path)
                else:
                    print(f"Warning  taxonomy expects CSV or TSV, got {data_path.suffix}  skipping")
                    continue

                if docs:
                    vectordb.add_documents(docs)
                    rows += 1
                    print(f"OK  ingested TAXONOMY: {r['resource_id']} -> {len(docs)} entries")
                continue

            loader = PyPDFLoader(str(data_path))
            pages = loader.load()
            total_pages = len(pages)

            s_raw, e_raw = (r.get("start_page") or "").strip(), (r.get("end_page") or "").strip()
            s = int(s_raw) if s_raw.isdigit() else 1
            if e_raw.isdigit():
                e = int(e_raw)
                if e == 0:
                    e = total_pages
            else:
                e = total_pages if kind in {"paper"} else int(e_raw) if e_raw else total_pages

            if e < s:
                print(f"Warning  end_page < start_page for {r.get('resource_id')}:{r.get('section')}  skipping")
                continue

            sub = pages[s - 1 : e]

            tags_str = ", ".join(split_tags(r.get("tags", ""))) or ""
            base_meta = {
                "doc_type": kind,
                "resource_id": r["resource_id"],
                "section": r.get("section", ""),
                "section_locator": r.get("section_locator", f"pp.{s}-{e}"),
                "source_path": str(data_path),
                "tags": tags_str,
            }
            for p in sub:
                p.metadata.update(base_meta)
                p.metadata = coerce_metadata(p.metadata)

            if kind == "paper":
                size, overlap = PAPER_CHUNK_SIZE, PAPER_CHUNK_OVERLAP
            else:
                size, overlap = BIG_CHUNK_SIZE, BIG_CHUNK_OVERLAP

            docs = chunk_documents(sub, size=size, overlap=overlap)

            for i, d in enumerate(docs):
                d.metadata.update({
                    "chunk_index": i,
                    "has_sql_example": has_sql_example(d.page_content),
                })
                d.metadata = coerce_metadata(d.metadata)

            if docs:
                vectordb.add_documents(docs)
                rows += 1
                print(f"OK  ingested: {r['resource_id']} [{s}-{e}] ({kind}) -> {len(docs)} chunks")

    print(f"Done. Sources ingested from {CHAPTER_MAP}: {rows} rows.")


if __name__ == "__main__":
    ingest()

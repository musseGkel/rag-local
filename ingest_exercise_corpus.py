# ingest_exercise_corpus.py
#
# Ingests all exercise JSON files from the datasets/ folder into a
# separate Chroma collection called "sqlex".
#
# This script is completely independent of ingest_from_chapter_map_server.py.
# It does NOT touch the existing "sqlkb" collection.
#
# Run from /opt/rag:
#   python3 ingest_exercise_corpus.py
#
# To wipe and re-ingest from scratch:
#   python3 ingest_exercise_corpus.py --reset

# --- SQLite shim for Chroma ---
try:
    import pysqlite3, sys

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass

import os
import json
import argparse
from pathlib import Path
from typing import List

import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

try:
    from langchain_core.documents import Document
except Exception:
    from langchain.schema import Document


# ---------- Paths & Config ----------

ROOT = Path(os.getenv("ROOT", "/opt/rag"))
DATASETS_DIR = Path(os.getenv("RAG_DATASETS", str(ROOT / "datasets")))
DB_DIR = Path(os.getenv("RAG_DB_DIR", str(ROOT / "db")))
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
COLLECTION = "sqlex"


# ---------- Helpers ----------


def coerce_metadata(md: dict) -> dict:
    """
    Chroma only accepts str, int, float, bool, None as metadata values.
    Convert anything else to a string.
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


def detect_constructs(sql: str) -> List[str]:
    """
    Lightweight SQL construct detection.
    Used as a retrieval tag so generation can fetch examples by construct.
    """
    if not sql:
        return []
    T = sql.upper()
    found = []

    checks = {
        "join": " JOIN " in T,
        "left_join": "LEFT JOIN" in T,
        "outer_join": "OUTER JOIN" in T,
        "natural_join": "NATURAL JOIN" in T,
        "group_by": "GROUP BY" in T,
        "having": "HAVING" in T,
        "order_by": "ORDER BY" in T,
        "subquery": T.count("SELECT") > 1,
        "not_in": "NOT IN" in T,
        "in_predicate": " IN (" in T,
        "exists": "EXISTS" in T,
        "not_exists": "NOT EXISTS" in T,
        "all_quantifier": "ALL (" in T,
        "distinct": "DISTINCT" in T,
        "aggregation": any(f in T for f in ["COUNT(", "SUM(", "AVG(", "MIN(", "MAX("]),
        "cte": T.strip().startswith("WITH ") or "\nWITH " in T,
        "union": "UNION" in T,
        "is_null": "IS NULL" in T,
        "is_not_null": "IS NOT NULL" in T,
    }

    for name, condition in checks.items():
        if condition:
            found.append(name)

    return found


def read_exercises_from_json(json_path: Path) -> List[Document]:
    """
    Read one exercise JSON file and return one Document per exercise.

    Rules:
    - Skip exercises with no solutions (solutions list is empty or missing)
    - Use only the first solution as the primary SQL for embedding
    - All solutions are stored in metadata for reference
    - Constructs are derived from the primary SQL
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))

    dataset_name = data.get("search_path") or data.get("title") or json_path.stem
    source = data.get("source", "lensql")
    dbms = data.get("dbms", "postgresql")

    docs: List[Document] = []

    for i, ex in enumerate(data.get("exercises", [])):
        request = (ex.get("request") or "").strip()
        solutions = [s.strip() for s in (ex.get("solutions") or []) if s and s.strip()]

        # Skip exercises with no solutions
        if not solutions:
            continue

        # Skip exercises with no request
        if not request:
            continue

        primary_sql = solutions[0]
        constructs = detect_constructs(primary_sql)

        # page_content = what gets embedded and searched
        content = f"[REQUEST]\n{request}\n\n" f"[SQL SOLUTION]\n{primary_sql}"

        meta = {
            "doc_type": "exercise",
            "dataset_name": dataset_name,
            "title": ex.get("title") or f"Exercise {i + 1}",
            "source": source,
            "dbms": dbms,
            "verified": bool(ex.get("verified", True)),
            "has_solution": True,
            "num_solutions": len(solutions),
            "constructs": constructs,  # list -> coerced to comma string
            "all_solutions": solutions,  # list -> coerced to comma string
            "source_path": str(json_path),
        }

        docs.append(Document(page_content=content, metadata=coerce_metadata(meta)))

    return docs


# ---------- Main ----------


def ingest(reset: bool = False):
    print(f"[info] DATASETS_DIR : {DATASETS_DIR}")
    print(f"[info] DB_DIR       : {DB_DIR}")
    print(f"[info] Collection   : {COLLECTION}")
    print(f"[info] Reset        : {reset}")
    print()

    if not DATASETS_DIR.exists():
        print(f"❌ datasets/ folder not found: {DATASETS_DIR}")
        return

    DB_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(DB_DIR))

    # Optionally wipe the sqlex collection before re-ingesting
    if reset:
        try:
            client.delete_collection(COLLECTION)
            print(f"🗑️  Deleted existing '{COLLECTION}' collection.")
        except Exception:
            pass

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )

    exdb = Chroma(
        client=client,
        collection_name=COLLECTION,
        embedding_function=embeddings,
    )

    json_files = sorted(DATASETS_DIR.glob("*.json"))

    if not json_files:
        print(f"❌ No JSON files found in {DATASETS_DIR}")
        return

    print(f"Found {len(json_files)} JSON file(s) to process.\n")

    total_docs = 0
    total_files = 0
    total_skipped = 0

    for jf in json_files:
        docs = read_exercises_from_json(jf)

        # Count how many were skipped (no solutions)
        raw_count = len(json.loads(jf.read_text())["exercises"])
        skipped = raw_count - len(docs)

        if docs:
            exdb.add_documents(docs)
            total_docs += len(docs)
            total_files += 1
            print(
                f"✅ {jf.name}: {len(docs)} exercises ingested"
                + (f" ({skipped} skipped — no solution)" if skipped else "")
            )
        else:
            print(f"⏭️  {jf.name}: skipped entirely (no exercises with solutions)")
            total_skipped += raw_count

    print()
    print(f"🎯 Done.")
    print(f"   Files processed : {total_files}/{len(json_files)}")
    print(f"   Exercises added : {total_docs}")
    print(f"   Exercises skipped (no solution): {total_skipped}")
    print(f"   Collection '{COLLECTION}' in {DB_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest exercise JSON files into the sqlex Chroma collection."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the sqlex collection before ingesting (fresh start).",
    )
    args = parser.parse_args()
    ingest(reset=args.reset)

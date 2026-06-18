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

    Emits tags from the SHARED vocabulary (see generation/construct_tags.py),
    so that exercises tagged here can be matched by the constraint-derived
    tags produced at query time. Any join variant also emits the generic
    "join" tag, mirroring constraints_to_constructs().
    """
    if not sql:
        return []
    T = sql.upper()
    found = set()

    has_left = "LEFT JOIN" in T
    has_right = "RIGHT JOIN" in T
    has_any_join = " JOIN " in T or "OUTER JOIN" in T or "NATURAL JOIN" in T

    if has_left:
        found.add("left_join")
    if has_right:
        found.add("right_join")
    # Crude self-join hint: same table aliased twice is hard to detect by
    # string alone, so we leave self_join to be conservative (omitted here).
    if has_any_join or has_left or has_right:
        found.add("join")

    if "GROUP BY" in T:
        found.add("group_by")
    if "HAVING" in T:
        found.add("having")
    if "ORDER BY" in T:
        found.add("order_by")
    if T.count("SELECT") > 1:
        found.add("subquery")
    if "EXISTS" in T:
        found.add("exists")
    if any(
        p in T for p in [" IN (", "= ANY", "= ALL", "> ALL", "< ALL", "> ANY", "< ANY"]
    ):
        found.add("in_any_all")
    if "DISTINCT" in T:
        found.add("distinct")
    if any(f in T for f in ["COUNT(", "SUM(", "AVG(", "MIN(", "MAX("]):
        found.add("aggregation")
    if "UNION" in T:
        found.add("union")

    return sorted(found)


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

        # One boolean field per detected construct, e.g. construct_join=True.
        # Chroma can filter exactly on booleans (it cannot reliably filter
        # inside the legacy comma-joined "constructs" string).
        construct_flags = {f"construct_{c}": True for c in constructs}

        meta = {
            "doc_type": "exercise",
            "dataset_name": dataset_name,
            "title": ex.get("title") or f"Exercise {i + 1}",
            "source": source,
            "dbms": dbms,
            "verified": bool(ex.get("verified", True)),
            "has_solution": True,
            "num_solutions": len(solutions),
            "constructs": constructs,  # legacy comma string, kept for reference
            **construct_flags,  # new boolean fields used by the filter
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

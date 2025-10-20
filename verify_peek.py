# verify_peek.py

# --- SQLite shim for Chroma (only needed if your system sqlite3 < 3.35) ---
try:
    import pysqlite3, sys  # type: ignore
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass

import chromadb

# Open your persistent DB
client = chromadb.PersistentClient(path="db")
col = client.get_collection("sqlkb")

# Grab a small batch; ids are always included by default — don't add them to `include`
batch = col.get(include=["metadatas", "documents"], limit=3)

ids = batch.get("ids", [])
metas = batch.get("metadatas", [])
docs = batch.get("documents", [])

if not ids:
    print("No records found. Did ingestion run successfully?")
else:
    for i in range(len(ids)):
        print("\nID:", ids[i])
        print("META:", metas[i])
        doc_txt = (docs[i] or "")
        print("DOC :", (doc_txt[:300] + ("..." if len(doc_txt) > 300 else "")))

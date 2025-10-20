# --- SQLite shim for Chroma ---
try:
    import pysqlite3, sys  # type: ignore
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass

# verify_count.py
import chromadb
client = chromadb.PersistentClient(path="db")
col = client.get_collection("sqlkb")
print("Total records:", col.count())

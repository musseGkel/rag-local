# --- SQLite shim for Chroma ---
try:
    import pysqlite3, sys  # type: ignore
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass

# verify_search.py
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

client = chromadb.PersistentClient(path="db")
emb = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5",
                            encode_kwargs={"normalize_embeddings": True})
vs = Chroma(client=client, collection_name="sqlkb", embedding_function=emb)

# Try a query that should hit SQL content
results = vs.similarity_search("How do I use SELECT with WHERE and GROUP BY", k=3)
for i, doc in enumerate(results, 1):
    print(f"\n=== Result {i} ===")
    print(doc.page_content[:400].strip(), "...")
    print("meta:", {k: doc.metadata.get(k) for k in ["resource_id","section","chunk_index","has_sql_example","tags","source_path"]})

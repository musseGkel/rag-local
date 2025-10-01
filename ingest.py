# --- SQLite shim for Chroma (must be first) ---
try:
    import pysqlite3, sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import os, glob
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

DOCS_DIR = "corpus"
DB_DIR = "db"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

def load_docs(path):
    docs = []
    for p in glob.glob(os.path.join(path, "**", "*"), recursive=True):
        low = p.lower()
        if low.endswith(".pdf"):
            docs.extend(PyPDFLoader(p).load())
        elif low.endswith((".txt", ".md", ".py", ".sql")):
            docs.extend(TextLoader(p, encoding="utf-8").load())
    return docs

if __name__ == "__main__":
    os.makedirs(DB_DIR, exist_ok=True)
    docs = load_docs(DOCS_DIR)
    if not docs:
        raise SystemExit("No documents found in ./corpus — add a few and retry.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # Writes the DB automatically to DB_DIR (no .persist() needed)
    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=DB_DIR,
        collection_name="tutor",
    )

    print(f"Ingested {len(chunks)} chunks into {DB_DIR}")

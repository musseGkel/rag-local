from __future__ import annotations

try:
    import pysqlite3, sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass



import os
from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.schema import Document
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DB_DIR = ROOT / "db"
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")  # same as ingest
COLLECTION = "sqlkb"

def load_vectorstore() -> Chroma:
    # Persistent Chroma client (points at the same folder used during ingest)
    client = chromadb.PersistentClient(path=str(DB_DIR))

    # Embeddings must be identical to ingest-time model & settings
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},  # keep cosine-friendly
    )

    return Chroma(
        client=client,
        collection_name=COLLECTION,
        embedding_function=embeddings,
    )

def build_retriever():
    vectordb = load_vectorstore()

    base = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 6,          # how many to return to the compressor
            "fetch_k": 24,   # pool to sample diversity from
            "lambda_mult": 0.5,  # 0→pure similarity, 1→pure diversity
        },
    )

    # Build a CrossEncoder instance first
    cross_encoder = HuggingFaceCrossEncoder(
        model_name="BAAI/bge-reranker-v2-m3",
        # device="cuda",  # uncomment if you want to force GPU
    )

    reranker = CrossEncoderReranker(
        model=cross_encoder,  # IMPORTANT: pass the object, not the string
        top_n=2,              # only keep 2 docs post-rerank
    )

    return ContextualCompressionRetriever(
        base_retriever=base,
        base_compressor=reranker,
    )

def _dedup_docs(docs: list[Document]) -> list[Document]:
    seen = set()
    unique = []
    for d in docs:
        key = (
            d.metadata.get("source_path"),
            d.metadata.get("page"),
            d.metadata.get("chunk_index"),
        )
        if key not in seen:
            seen.add(key)
            unique.append(d)
    return unique


def retrieve_for_generation(query: str) -> list[Document]:
    """
    Returns up to 2 best reranked documents.
    """
    retriever = build_retriever()
    docs = retriever.invoke(query)  # <- replaces get_relevant_documents
    docs = _dedup_docs(docs)
    return docs[:2]

# --- Example usage ---
if __name__ == "__main__":
    q = "How do I write a SELECT with GROUP BY and HAVING in PostgreSQL?"
    top_docs = retrieve_for_generation(q)
    print(f"Retrieved {len(top_docs)} doc(s) for generation.")
    for i, d in enumerate(top_docs, 1):
        print(f"\n--- Doc {i} ---")
        print(f"meta: {d.metadata}")
        print(d.page_content[:600])

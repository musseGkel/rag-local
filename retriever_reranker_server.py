from __future__ import annotations

try:
    import pysqlite3, sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass

import os
from pathlib import Path
from typing import List

import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DB_DIR = ROOT / "db"
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
COLLECTION = "sqlkb"


def load_vectorstore() -> Chroma:
    """
    Load the persistent Chroma collection using the same embedding model
    and settings as during ingest.
    """
    client = chromadb.PersistentClient(path=str(DB_DIR))

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )

    return Chroma(
        client=client,
        collection_name=COLLECTION,
        embedding_function=embeddings,
    )


def _dedup_docs(docs: List[Document]) -> List[Document]:
    """
    Deduplicate documents using (source_path, page, chunk_index) in metadata.
    """
    seen = set()
    unique: List[Document] = []
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


# Global cache so we don't reload the CrossEncoder on every call
_reranker_model: CrossEncoder | None = None


def _get_reranker() -> CrossEncoder:
    global _reranker_model
    if _reranker_model is None:
        _reranker_model = CrossEncoder("BAAI/bge-reranker-v2-m3")
    return _reranker_model


def retrieve_for_generation(
    query: str,
    fetch_k: int = 24,
    mmr_k: int = 24,
    lambda_mult: float = 0.5,
    top_n: int = 2,
) -> List[Document]:
    """
    Future-proof retriever that:
      1. Uses Chroma with MMR to get a diverse pool of candidates.
      2. Reranks those candidates with a CrossEncoder.
      3. Returns the best `top_n` documents.

    No langchain.retrievers or langchain-classic.
    """
    vectordb = load_vectorstore()

    # Step 1: MMR candidate retrieval from Chroma
    # This approximates your old base_retriever with search_type="mmr"
    candidates = vectordb.max_marginal_relevance_search(
        query,
        k=mmr_k,           # how many docs we keep before reranking
        fetch_k=fetch_k,   # pool for diversity
        lambda_mult=lambda_mult,
    )

    if not candidates:
        return []

    docs = _dedup_docs(candidates)

    # Step 2: CrossEncoder reranking
    reranker = _get_reranker()
    pairs = [[query, d.page_content] for d in docs]
    scores = reranker.predict(pairs)

    scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    top_docs = [d for d, _ in scored_docs[:top_n]]

    return top_docs


if __name__ == "__main__":
    q = "How do I write a SELECT with GROUP BY and HAVING in PostgreSQL?"
    top_docs = retrieve_for_generation(q)
    print(f"Retrieved {len(top_docs)} doc(s) for generation.")
    for i, d in enumerate(top_docs, 1):
        print(f"\n--- Doc {i} ---")
        print(f"meta: {d.metadata}")
        print(d.page_content[:600])

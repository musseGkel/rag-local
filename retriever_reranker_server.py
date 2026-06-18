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

ROOT = Path(os.getenv("ROOT", "/opt/rag"))
DB_DIR = ROOT / "db"
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
COLLECTION = "sqlex"


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
        # Use 'cuda' to pick up the visible device, and set automapping
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

        _reranker_model = CrossEncoder(
            "BAAI/bge-reranker-v2-m3",
            device=device,
            # This helps if VRAM is very tight
            tokenizer_kwargs={"clean_up_tokenization_spaces": True},
        )
    return _reranker_model


def _build_construct_filter(construct_tags) -> dict | None:
    """
    Turn a set/list of construct tags into a Chroma OR-filter over the
    boolean `construct_*` metadata fields written at ingest time.

    Returns None when there are no tags, so the caller falls back to an
    unfiltered search. An OR (not AND) is used so a document qualifies by
    sharing ANY target construct, which keeps recall high.
    """
    if not construct_tags:
        return None

    clauses = [{f"construct_{t}": True} for t in sorted(set(construct_tags))]
    if len(clauses) == 1:
        return clauses[0]
    return {"$or": clauses}


def retrieve_for_generation(
    query: str,
    fetch_k: int = 32,
    mmr_k: int = 16,
    lambda_mult: float = 0.5,
    top_n: int = 3,
    construct_tags=None,
) -> List[Document]:
    """
    Future-proof retriever that:
      1. Uses Chroma with MMR to get a diverse pool of candidates.
      2. Reranks those candidates with a CrossEncoder.
      3. Returns the best `top_n` documents.

    `construct_tags` is OPTIONAL and backward-compatible:
      - None (default): behaves exactly as before -- plain MMR, no filtering.
        All existing callers (Lens describe/explain/etc.) hit this path
        unchanged.
      - a set/list of tags: restrict MMR to exercises whose `construct_*`
        boolean metadata matches ANY tag. If that filtered search returns
        nothing, fall back to the unfiltered search so generation is never
        starved.

    No langchain.retrievers or langchain-classic.
    """
    vectordb = load_vectorstore()

    where = _build_construct_filter(construct_tags)

    # Step 1: MMR candidate retrieval from Chroma
    # This approximates your old base_retriever with search_type="mmr"
    def _mmr(filter_arg):
        kwargs = dict(k=mmr_k, fetch_k=fetch_k, lambda_mult=lambda_mult)
        if filter_arg is not None:
            kwargs["filter"] = filter_arg
        return vectordb.max_marginal_relevance_search(query, **kwargs)

    candidates = _mmr(where)

    # Fallback: if the construct filter matched nothing, retry unfiltered
    # so the generator always gets some context.
    if not candidates and where is not None:
        candidates = _mmr(None)

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


def build_retrieval_query(user_sql: str, user_goal: str) -> str:
    return f"Query: {user_sql}. " f"User goal: {user_goal}"


if __name__ == "__main__":

    user_sql = """
        SELECT B.Matricola
        FROM (
        SELECT S.Matricola
        FROM Studenti S
        JOIN CorsiDiLaurea CDL
            ON S.CorsoDiLaurea = CDL.id
        AND CDL.Denominazione = 'Informatica'
        JOIN Corsi C
            ON C.CorsoDiLaurea = CDL.id
        JOIN Esami E
            ON E.Corso = C.id
        AND C.id = 'bdd1n'
        AND E.Studente = S.Matricola
        WHERE EXTRACT(MONTH FROM E.Data) = 06
            AND EXTRACT(YEAR FROM E.Data) =
        ) AS B
        JOIN (
        SELECT S2.Matricola
        FROM Studenti S2
        JOIN CorsiDiLaurea CDL2
            ON S2.CorsoDiLaurea = CDL2.id
        AND CDL2.Denominazione = 'Informatica'
        JOIN Corsi C2
            ON C2.CorsoDiLaurea = CDL2.id
        JOIN Esami E2
            ON E2.Corso = C2.id
        AND C2.id = 'ig'
        AND E2.Studente = S2.Matricola
        WHERE EXTRACT(MONTH FROM E2.Data) = 06
            AND EXTRACT(YEAR FROM E2.Data) = 2010
        ) AS I
        ON B.Matricola = I.Matricola;

    """
    user_goal = "Identify which specific part of the query is likely responsible for the syntax error near the closing parenthesis, without fixing the query and without explaining what the query does."

    retrieval_query = build_retrieval_query(user_sql, user_goal)

    top_docs = retrieve_for_generation(retrieval_query)

    print(f"Retrieved {len(top_docs)} doc(s) for generation.")
    for i, d in enumerate(top_docs, 1):
        print(f"\n--- Doc {i} ---")
        print(f"meta: {d.metadata}")
        print(d.page_content)

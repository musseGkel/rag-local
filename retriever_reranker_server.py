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
KB_COLLECTION = "sqlkb"  # taxonomy + textbooks (Lens grounding)
EX_COLLECTION = "sqlex"  # exercise pairs (exercise generation)
DEFAULT_COLLECTION = EX_COLLECTION  # keep current default


def load_vectorstore(collection: str = DEFAULT_COLLECTION) -> Chroma:
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
        client=client, collection_name=collection, embedding_function=embeddings
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


def _build_construct_filter(construct_tags, forbidden_tags=None) -> dict | None:
    """
    Build a Chroma metadata filter from required and forbidden construct tags,
    over the boolean `construct_*` fields written at ingest time.

    Required tags (OR): a document qualifies by sharing ANY required construct,
    which keeps recall high. Built as {"$or": [{construct_x: True}, ...]}.

    Forbidden tags (AND of negatives): a document is excluded if it has ANY
    forbidden construct. Because ingest writes `construct_x: True` ONLY when the
    construct is present (absent constructs have no key at all), a plain
    {"construct_x": {"$ne": True}} would also drop documents that simply lack
    the key. Chroma's `$ne` matches records where the field is absent OR not
    equal, which is exactly what we want here: keep docs where the construct is
    absent, drop docs where it is True. Each forbidden tag becomes
    {"construct_x": {"$ne": True}}, and all are AND-ed.

    Returns None when there is nothing to filter on, so the caller falls back to
    an unfiltered search.
    """
    required = sorted(set(construct_tags)) if construct_tags else []
    forbidden = sorted(set(forbidden_tags)) if forbidden_tags else []
    # A tag should never be both; if it somehow is, requiring it wins.
    forbidden = [t for t in forbidden if t not in required]

    clauses: list[dict] = []

    if required:
        pos = [{f"construct_{t}": True} for t in required]
        clauses.append(pos[0] if len(pos) == 1 else {"$or": pos})

    for t in forbidden:
        clauses.append({f"construct_{t}": {"$ne": True}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def retrieve_for_generation(
    query: str,
    fetch_k: int = 32,
    mmr_k: int = 16,
    lambda_mult: float = 0.5,
    top_n: int = 3,
    construct_tags=None,
    forbidden_construct_tags=None,
    collection: str = DEFAULT_COLLECTION,
) -> List[Document]:
    """
    Future-proof retriever that:
      1. Uses Chroma with MMR to get a diverse pool of candidates.
      2. Reranks those candidates with a CrossEncoder.
      3. Returns the best `top_n` documents.

    `construct_tags` / `forbidden_construct_tags` are OPTIONAL and
    backward-compatible:
      - Both None (default): plain MMR, no filtering. All existing callers
        (Lens describe/explain/etc.) hit this path unchanged.
      - construct_tags set: restrict to exercises whose `construct_*` metadata
        matches ANY required tag.
      - forbidden_construct_tags set: additionally exclude exercises that
        demonstrate ANY forbidden construct.

    To avoid starving generation when filters are strict, retrieval degrades
    gracefully: (1) required + forbidden, then (2) required only, then
    (3) forbidden only, then (4) unfiltered. The first non-empty result wins.
    """
    vectordb = load_vectorstore(collection=collection)

    def _mmr(filter_arg):
        kwargs = dict(k=mmr_k, fetch_k=fetch_k, lambda_mult=lambda_mult)
        if filter_arg is not None:
            kwargs["filter"] = filter_arg
        return vectordb.max_marginal_relevance_search(query, **kwargs)

    # Build the staged filters once, skipping any that are None or duplicates.
    full = _build_construct_filter(construct_tags, forbidden_construct_tags)
    pos_only = _build_construct_filter(construct_tags, None)
    neg_only = _build_construct_filter(None, forbidden_construct_tags)

    candidates: List[Document] = []
    tried: list[dict | None] = []
    for filt in (full, pos_only, neg_only, None):
        # Skip filters we've already attempted (e.g. full == pos_only when
        # there are no forbidden tags) to avoid redundant queries.
        if any(filt == t for t in tried):
            continue
        tried.append(filt)
        candidates = _mmr(filt)
        if candidates:
            break

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

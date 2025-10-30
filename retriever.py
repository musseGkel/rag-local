# retriever.py
from typing import List, Tuple
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def build_vs():
    emb = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5", encode_kwargs={"normalize_embeddings": True})
    return Chroma(collection_name="sqlkb", embedding_function=emb, persist_directory="db")

def mmr_search(vs: Chroma, query: str, k: int = 6, fetch_k: int = 30) -> List[Tuple]:
    # Chroma API: use as a plain similarity search; emulate MMR by overfetch+dedupe
    hits = vs.similarity_search_with_score(query, k=fetch_k)
    # naive MMR-ish post-process: keep diverse sources first
    seen = set(); out = []
    for d, s in hits:
        key = (d.metadata.get("resource_id"), d.metadata.get("section"))
        if key in seen: 
            continue
        seen.add(key); out.append((d, s))
        if len(out) >= k: break
    return out

def rerank(hits: List[Tuple], boost_taxonomy=0.25, boost_sql_example=0.1):
    scored = []
    for d, score in hits:
        bonus = 0.0
        if (d.metadata.get("doc_type") == "taxonomy"): bonus -= boost_taxonomy  # lower score = better rank
        if str(d.metadata.get("has_sql_example")).lower() in {"true","1"}: bonus -= boost_sql_example
        scored.append((d, score + bonus))
    return sorted(scored, key=lambda x: x[1])

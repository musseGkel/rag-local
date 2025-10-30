# test_retrieval.py
try:
    import pysqlite3, sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

emb = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5", encode_kwargs={"normalize_embeddings": True})
vs = Chroma(collection_name="sqlkb", embedding_function=emb, persist_directory="db")

q = "Why is my GROUP BY invalid when I select non-aggregated columns?"
docs = vs.similarity_search_with_score(q, k=5)

for d, score in docs:
    m = d.metadata
    print(f"{score:.3f} | {m.get('doc_type')} | {m.get('resource_id')} | {m.get('section')} | {m.get('section_locator')}")
    print(d.page_content[:220].replace("\n"," ") + "…")
    print("-"*80)

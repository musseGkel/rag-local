# --- SQLite shim for Chroma (must be first) ---
try:
    import pysqlite3  # provides newer SQLite
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass
# ---------------------------------------------

# ask.py  (LCEL-based RAG: no RetrievalQA import)
import os, sys

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from llama_cpp import Llama
from langchain_community.chat_models import ChatLlamaCpp

DB_DIR = "db"
EMBED_MODEL = "models/bge-small-en-v1.5" 
# >>> CHANGE this to your actual GGUF filename in ./models <<<
GGUF_MODEL_PATH = os.path.join("models", "phi-3-mini-4k-instruct-q4_k_m.gguf")  # match ls output
# e.g., "models/qwen2.5-1.5b-instruct.Q4_K_M.gguf"

def build_chain():
    # 1) Vector store / retriever
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        cache_folder="models",                     
        model_kwargs={"local_files_only": True},   
    )

    vectordb = Chroma(
        collection_name="tutor",
        persist_directory=DB_DIR,
        embedding_function=embeddings,
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    if not os.path.isfile(GGUF_MODEL_PATH):
        raise FileNotFoundError(f"Missing GGUF at: {GGUF_MODEL_PATH}")

    client = Llama(
        model_path=GGUF_MODEL_PATH,
        n_ctx=4096,
        n_threads=os.cpu_count() or 4,
        n_batch=256,
        seed=0,
        n_gpu_layers=0,
    )

    llm = ChatLlamaCpp(
        client=client,       # <-- pass the client (required in your version)
        temperature=0.0,
        verbose=False,
    )

    # 3) Prompt + chains (LCEL)
    prompt = ChatPromptTemplate.from_template(
        "You are a careful CS tutor. Use ONLY the context to answer.\n"
        "If the answer is not in the context, say you don't know and suggest what to try next.\n\n"
        "Context:\n{context}\n\n"
        "Question: {input}\n"
        "Answer:"
    )

    # Stuff the retrieved docs into the prompt
    stuff_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

    # Full retrieval-augmented chain
    rag_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=stuff_chain)
    return rag_chain

if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Explain INNER JOIN vs LEFT JOIN with a tiny example."
    chain = build_chain()
    result = chain.invoke({"input": question})
    # result typically contains: {'input': ..., 'context': [docs...], 'answer': '...'}
    print(result.get("answer") or result)

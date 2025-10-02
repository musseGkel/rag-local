# --- SQLite shim for Chroma (must be first) ---
try:
    import pysqlite3  # provides newer SQLite
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass
# ---------------------------------------------

import os, sys
from llama_cpp import Llama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

DB_DIR = "db"
EMBED_MODEL = "models/bge-small-en-v1.5"
GGUF_MODEL_PATH = os.path.join("models", "phi-3-mini-4k-instruct-q4_k_m.gguf")

def build_pipeline():
    # Embeddings + retriever
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        cache_folder="models",
        model_kwargs={"local_files_only": True},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectordb = Chroma(
        collection_name="tutor",
        persist_directory=DB_DIR,
        embedding_function=embeddings,
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    if not os.path.isfile(GGUF_MODEL_PATH):
        raise FileNotFoundError(f"Missing GGUF at: {GGUF_MODEL_PATH}")

    # Plain llama.cpp client (no chat handlers)
    client = Llama(
        model_path=GGUF_MODEL_PATH,
        n_ctx=4096,
        n_threads=os.cpu_count() or 4,
        n_batch=256,
        seed=0,
        n_gpu_layers=0,  # >0 only if your build supports GPU offload
    )

    def answer(question: str) -> str:
        docs = retriever.invoke(question)
        if not docs:
            return "I don't know. No relevant context found in the DB. Try adding documents to ./corpus and re-running ingest."

        # Show up to 3 unique sources (optional)
        sources = []
        for d in docs:
            src = d.metadata.get("source") or d.metadata.get("file_path")
            if src and src not in sources:
                sources.append(src)
            if len(sources) >= 3:
                break

        # Keep the context tight (prevents the model from “reading out” the PDF)
        context = "\n\n".join(d.page_content for d in docs)
        context = context.strip()
        if len(context) > 1800:   # keep it compact
            context = context[:1800] + "..."

        # Super explicit instruction + single-line answer
        prompt = (
            "You are a careful CS tutor. Use ONLY the provided context. "
            "If the answer isn't explicitly in the context, reply exactly: I don't know from the context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer (one short sentence, no bullets, no extra text): "
        )

        out = client.create_completion(
            prompt,
            temperature=0.0,
            max_tokens=120,                 # small cap so it can't ramble
            top_p=0.1,                      # make it more deterministic
            stop=[                          # cut off if it starts to wander
                "\n\n",
                "\nQuestion:",
                "\nGroups",
                "\n●", "● ", "○ ",
                "Sources:",
                "<|endoftext|>", "</s>", "<|end|>",
            ],
            repeat_penalty=1.05,
        )
        ans = out["choices"][0]["text"].strip()

        # Post-process: take only the first line
        ans = ans.splitlines()[0].strip()

        if sources:
            ans += "\n\nSources:\n- " + "\n- ".join(sources)
        return ans

    return answer

if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Explain INNER JOIN vs LEFT JOIN with a tiny example."
    answer_fn = build_pipeline()
    print(answer_fn(question))

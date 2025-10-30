from __future__ import annotations
import os
from pathlib import Path
from typing import List

from llama_cpp import Llama
from langchain.schema import Document

# import your retriever
from retriever_reranker import retrieve_for_generation, ROOT

MODEL_PATH = str(ROOT / "models" / "phi-3-mini-4k-instruct-q4_k_m.gguf")

# keep context tight for a 4k model
MAX_CONTEXT_CHARS = 9000


def build_context(docs: List[Document]) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        rid = d.metadata.get("resource_id", "")
        sect = d.metadata.get("section", "")
        loc = d.metadata.get("section_locator", "")
        head = f"[Source {i} | {rid} | {sect} | {loc}]"
        body = (d.page_content or "").strip()
        parts.append(f"{head}\n{body}")
    ctx = "\n\n---\n\n".join(parts)
    if len(ctx) > MAX_CONTEXT_CHARS:
        ctx = ctx[:MAX_CONTEXT_CHARS] + "\n\n[Context truncated]"
    return ctx


def build_messages(query: str, context: str):
    system = (
        "You are a concise technical assistant. "
        "Answer only using the provided context. "
        "If the answer is not in the context, say you don’t know."
    )
    user = (
        f"Question:\n{query}\n\n"
        f"Context:\n{context}\n\n"
        "Instructions:\n"
        "- Cite sources inline like [Source 1] or [Source 2] when used.\n"
        "- Be precise and practical."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def load_llm(model_path: str) -> Llama:
    return Llama(
        model_path=model_path,
        n_ctx=4096,
        n_threads=os.cpu_count() or 4,
        n_gpu_layers=0,      # set >0 if built with GPU support
        chat_format="phi3",  # llama.cpp chat template for phi-3
        verbose=False,
    )


def rag_answer(query: str) -> str:
    docs = retrieve_for_generation(query)
    if not docs:
        return "I don’t have enough context to answer."

    context = build_context(docs)
    messages = build_messages(query, context)

    llm = load_llm(MODEL_PATH)
    out = llm.create_chat_completion(
        messages=messages,
        temperature=0.2,
        max_tokens=700,
        top_p=0.9,
        stop=["</s>"],
    )
    answer = out["choices"][0]["message"]["content"].strip()

    # compact sources list
    lines = ["", "Sources:"]
    for i, d in enumerate(docs, 1):
        rid = d.metadata.get("resource_id", "")
        sect = d.metadata.get("section", "")
        loc = d.metadata.get("section_locator", "")
        lines.append(f"- Source {i}: {rid} | {sect} | {loc}")

    return answer + "\n" + "\n".join(lines)


if __name__ == "__main__":
    q = "How do I write a SELECT with GROUP BY and HAVING in PostgreSQL?"
    print(rag_answer(q))

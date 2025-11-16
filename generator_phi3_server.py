from __future__ import annotations

import os
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_core.documents import Document

# your retriever (MMR + CrossEncoder, the one we just fixed)
from retriever_reranker_server import retrieve_for_generation

# Hugging Face model id (matches your cached folder models--microsoft--phi-4-mini-instruct)
MODEL_ID = os.getenv("GEN_MODEL", "microsoft/phi-4-mini-instruct")

# keep context tight
MAX_CONTEXT_CHARS = 9000

# Global model + tokenizer (load once, reuse for all queries)
_TOKENIZER: AutoTokenizer | None = None
_MODEL: AutoModelForCausalLM | None = None


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


def get_hf_model():
    global _TOKENIZER, _MODEL
    if _TOKENIZER is None or _MODEL is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID)
        # choose dtype based on GPU availability
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        _MODEL = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=dtype,
            device_map="auto",   # use all available GPUs/CPU sensibly
        )
    return _TOKENIZER, _MODEL


def hf_chat_generate(messages, max_new_tokens: int = 700, temperature: float = 0.2, top_p: float = 0.9) -> str:
    tokenizer, model = get_hf_model()

    # use the chat template that Phi-4 provides in its tokenizer config
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False if temperature == 0 else True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )

    # only decode the newly generated tokens
    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text.strip()


def rag_answer(query: str) -> str:
    docs = retrieve_for_generation(query)
    if not docs:
        return "I don’t have enough context to answer."

    context = build_context(docs)
    messages = build_messages(query, context)

    answer = hf_chat_generate(
        messages,
        max_new_tokens=700,
        temperature=0.2,
        top_p=0.9,
    )

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

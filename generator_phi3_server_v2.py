from __future__ import annotations

import os
from typing import List

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_core.documents import Document
from sql_error_categorizer import DetectedError, SqlErrors

# retriever (MMR + CrossEncoder) to get relevant docs for generation
from retriever_reranker_server import (
    retrieve_for_generation,
    KB_COLLECTION,
    EX_COLLECTION,
)

# Hugging Face model id (matches cached folder models--microsoft--phi-4-mini-instruct)
MODEL_ID = os.getenv("GEN_MODEL", "microsoft/phi-4-mini-instruct")

# keep context tight
MAX_CONTEXT_CHARS = 30000

# Global model + tokenizer (load once, reuse for all queries)
_TOKENIZER: AutoTokenizer | None = (
    None  # converts raw text to token ids (that the model uses) and back
)
_MODEL: AutoModelForCausalLM | None = (
    None  # the actual language model that generates text
)

from dataclasses import dataclass
from lens_prompts_v2 import LensPrompt, LENS_SYSTEM_PROMPT
from lens_prompts_v2 import (
    describe_my_query,
    explain_my_query,
    explain_error,
    provide_error_example,
    locate_error_cause,
    fix_query,
    detect_errors,
)


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


# Toggle context debug prints via env var (default on). Set LENS_DEBUG_CONTEXT=0 to silence.
DEBUG_CONTEXT = os.getenv("LENS_DEBUG_CONTEXT", "1") not in {"0", "false", "False"}


def debug_print_context(prompt: LensPrompt, docs: List[Document], context: str) -> None:
    if not DEBUG_CONTEXT:
        return
    print("\n" + "=" * 80)
    print(
        f"[CONTEXT DEBUG] generator=phi3 | mode={prompt.mode} | lang={prompt.language} | rag={'on' if docs else 'off'}"
    )
    print(f"[CONTEXT DEBUG] retrieval_query: {prompt.retrieval_query}")
    print(f"[CONTEXT DEBUG] retrieved docs: {len(docs)}")
    for i, d in enumerate(docs, 1):
        rid = d.metadata.get("resource_id", "")
        sect = d.metadata.get("section", "")
        loc = d.metadata.get("section_locator", "")
        preview = (d.page_content or "").strip().replace("\n", " ")[:200]
        print(f"  - doc {i}: rid={rid} | section={sect} | locator={loc}")
        print(f"           preview: {preview}")
    print(f"[CONTEXT DEBUG] assembled context chars: {len(context)}")
    print("[CONTEXT DEBUG] --- full context start ---")
    print(context if context else "<empty>")
    print("[CONTEXT DEBUG] --- full context end ---")
    print("=" * 80 + "\n")


def build_messages(prompt: LensPrompt, context: str):
    messages = [{"role": "system", "content": LENS_SYSTEM_PROMPT}]

    if context:
        messages.append(
            {
                "role": "system",
                "content": (
                    "Reference notes for answering (do not quote, do not reveal):\n"
                    + context
                ),
            }
        )

    messages.append({"role": "user", "content": prompt.generation_query})
    return messages


# def build_messages(prompt: LensPrompt, context: str):
#     system = LENS_SYSTEM_PROMPT

#     if context:
#         system += (
#             "\n\nRAG RULES:\n"
#             "- The context is private reference material.\n"
#             "- Never output anything from the context verbatim.\n"
#         )
#         user = (
#             f"{prompt.generation_query}\n\n"
#             "BEGIN_RAG_CONTEXT\n"
#             f"{context}\n"
#             "END_RAG_CONTEXT\n"
#         )
#     else:
#         user = prompt.generation_query

#     return [
#         {"role": "system", "content": system},
#         {"role": "user", "content": user},
#     ]


def get_hf_model():
    global _TOKENIZER, _MODEL
    if _TOKENIZER is None or _MODEL is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID)
        # choose dtype based on GPU availability
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        _MODEL = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=dtype,
            device_map="auto",  # use all available GPUs/CPU sensibly
        )
    return _TOKENIZER, _MODEL


def hf_chat_generate(
    messages, max_new_tokens: int = 700, temperature: float = 0.2, top_p: float = 0.9
) -> str:
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
    generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text.strip()


def sanitize_answer(text: str) -> str:
    # Hard stop if the model starts leaking the context block
    cut_markers = [
        "BEGIN_RAG_CONTEXT",
        "END_RAG_CONTEXT",
        "[RAG CONTEXT]",
        "[Source ",
        "Sources:",
    ]
    earliest = None
    for m in cut_markers:
        idx = text.find(m)
        if idx != -1 and (earliest is None or idx < earliest):
            earliest = idx
    if earliest is not None:
        text = text[:earliest].rstrip()
    return text


# Lens modes that never use retrieval, even when RAG is switched on.
NO_RAG_MODES = {"describe_my_query", "explain_my_query"}


def retrieve_docs(prompt: LensPrompt, use_rag: bool = True) -> List[Document]:
    """Retrieve context documents for a prompt, or [] when RAG is off."""
    if not use_rag or prompt.mode in NO_RAG_MODES:
        return []
    if prompt.mode == "generate_exercise":
        return retrieve_for_generation(
            prompt.retrieval_query,
            collection=EX_COLLECTION,
            construct_tags=getattr(prompt, "construct_tags", None),
            forbidden_construct_tags=getattr(prompt, "forbidden_construct_tags", None),
        )
    return retrieve_for_generation(
        prompt.retrieval_query,
        collection=KB_COLLECTION,
    )


def doc_sources(docs: List[Document]) -> List[dict]:
    """Compact, JSON-serialisable description of the retrieved documents."""
    out = []
    for d in docs:
        md = d.metadata or {}
        if md.get("doc_type") == "exercise":
            # Exercise corpus (ingest_exercise_corpus.py): no resource_id,
            # so identify the example by dataset + exercise title.
            dataset = md.get("dataset_name", "")
            title = md.get("title", "")
            resource_id = " / ".join(x for x in (dataset, title) if x)
        else:
            resource_id = md.get("resource_id") or md.get("name") or ""
        out.append(
            {
                "resource_id": resource_id or md.get("source", ""),
                "section": md.get("section", ""),
                "locator": md.get("section_locator", ""),
                "doc_type": md.get("doc_type", ""),
            }
        )
    return out


def rag_answer(prompt: LensPrompt, use_rag: bool = True) -> str:
    """Backwards-compatible entry point: returns only the answer text."""
    answer, _docs = rag_answer_with_docs(prompt, use_rag=use_rag)
    return answer


def rag_answer_with_docs(prompt: LensPrompt, use_rag: bool = True):
    """
    Generate an answer for `prompt`.

    use_rag=True  -> retrieve context and inject it (default pipeline).
    use_rag=False -> skip retrieval; the model sees only the system prompt
                     and the task prompt (baseline for the RAG vs no-RAG study).

    Returns (answer, docs) where docs is the list of retrieved documents
    (empty when RAG is off or the mode never uses retrieval).
    """
    docs = retrieve_docs(prompt, use_rag=use_rag)

    context = build_context(docs) if docs else ""

    debug_print_context(prompt, docs, context)

    # generation uses prompt.generation_query
    messages = build_messages(prompt, context)

    if DEBUG_CONTEXT:
        print("[CONTEXT DEBUG] --- final messages sent to model ---")
        for m in messages:
            print(f"  [{m['role']}] {m['content']}")
        print("[CONTEXT DEBUG] --- end final messages ---\n")

    answer = hf_chat_generate(
        messages,
        max_new_tokens=700,
        temperature=0.2,
        top_p=0.9,
    )
    # answer = sanitize_answer(answer)

    if not docs:
        return answer, docs

    # # compact sources list
    # lines = ["", "Sources:"]
    # shown = 0
    # for i, d in enumerate(docs, 1):
    #     rid = (d.metadata.get("resource_id") or "").strip()
    #     if not rid:
    #         continue
    #     sect = (d.metadata.get("section") or "").strip()
    #     loc = (d.metadata.get("section_locator") or "").strip()
    #     lines.append(f"- Source {i}: {rid} | {sect} | {loc}")
    #     shown += 1

    # if shown == 0:
    #     return answer

    return answer, docs


# + "\n" + "\n".join(lines)


# if __name__ == "__main__":
#     mode = os.getenv("LENS_MODE", "describe_query")
#     lang = os.getenv("LENS_LANG", "en")

#     user_sql = "select * from customer"
#     error_message = "UndefinedTable: relation custom does not exist"
#     error_code = "42P01"

#     if mode == "describe_my_query":
#         prompt = describe_my_query(code=user_sql, lang=lang)

#     elif mode == "explain_my_query":
#         prompt = explain_my_query(code=user_sql, lang=lang)

#     elif mode == "explain_error":
#         user_sql = "select * from custom"
#         prompt = explain_error(
#             code=user_sql, exception=error_message + f" (code: {error_code})", lang=lang
#         )

#     elif mode == "provide_error_example":
#         user_sql = "select * from custom"
#         prompt = provide_error_example(
#             code=user_sql,
#             exception=error_message + f" (code: {error_code})",
#             lang=lang,
#         )

#     elif mode == "locate_error_cause":
#         user_sql = "select * from custom"
#         prompt = locate_error_cause(
#             code=user_sql,
#             exception=error_message + f" (code: {error_code})",
#             lang=lang,
#         )

#     elif mode == "fix_query":
#         user_sql = "select * from custom"
#         prompt = fix_query(
#             code=user_sql,
#             exception=error_message + f" (code: {error_code})",
#             lang=lang,
#             errors=[
#                 DetectedError(
#                     error=SqlErrors.SYN_7_UNDEFINED_OBJECT,
#                 )
#             ],
#         )
#     elif mode == "detect_errors":
#         user_sql = "select * from custom"
#         prompt = detect_errors(
#             code=user_sql,
#             errors=[
#                 DetectedError(
#                     error=SqlErrors.SYN_7_UNDEFINED_OBJECT,
#                 )
#             ],
#             lang=lang,
#         )

#     elif mode == "locate_error_cause_v2":
#         user_sql = """
# SELECT B.Matricola
# FROM (
# SELECT S.Matricola
# FROM Studenti S
# JOIN CorsiDiLaurea CDL
#     ON S.CorsoDiLaurea = CDL.id
# AND CDL.Denominazione = 'Informatica'
# JOIN Corsi C
#     ON C.CorsoDiLaurea = CDL.id
# JOIN Esami E
#     ON E.Corso = C.id
# AND C.id = 'bdd1n'
# AND E.Studente = S.Matricola
# WHERE EXTRACT(MONTH FROM E.Data) = 06
#     AND EXTRACT(YEAR FROM E.Data) =
# ) AS B
# JOIN (
# SELECT S2.Matricola
# FROM Studenti S2
# JOIN CorsiDiLaurea CDL2
#     ON S2.CorsoDiLaurea = CDL2.id
# AND CDL2.Denominazione = 'Informatica'
# JOIN Corsi C2
#     ON C2.CorsoDiLaurea = CDL2.id
# JOIN Esami E2
#     ON E2.Corso = C2.id
# AND C2.id = 'ig'
# AND E2.Studente = S2.Matricola
# WHERE EXTRACT(MONTH FROM E2.Data) = 06
#     AND EXTRACT(YEAR FROM E2.Data) = 2010
# ) AS I
# ON B.Matricola = I.Matricola;
#         """
#         prompt = locate_error_cause(
#             code=user_sql,
#             exception='syntax error at or near ")"',
#             lang=lang,
#         )

#     else:
#         raise ValueError(f"Unknown mode: {mode}")

#     print(rag_answer(prompt))

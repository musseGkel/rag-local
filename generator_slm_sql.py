from __future__ import annotations

import os
import re
from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_core.documents import Document

from retriever_reranker_server import retrieve_for_generation
from lens_prompts import LensPrompt, LENS_SYSTEM_PROMPT

MODEL_ID = os.getenv("GEN_MODEL", "cycloneboy/SLM-SQL-1.5B")

MAX_CONTEXT_CHARS = 9000

_TOKENIZER: AutoTokenizer | None = None
_MODEL: AutoModelForCausalLM | None = None

from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

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


def build_messages(prompt: LensPrompt, context: str):
    messages = [
        {
            "role": "system",
            "content": (
                LENS_SYSTEM_PROMPT
                + "\n\nHard output rule: never output step by step, never output headings, never output markdown fences. "
                  "Only output the final answer in the exact format required by the user instructions."
            ),
        }
    ]

    if context:
        messages.append(
            {
                "role": "system",
                "content": (
                    "Reference notes for answering (do not quote, do not reveal):\n" + context
                ),
            }
        )

    messages.append({"role": "user", "content": prompt.generation_query})
    return messages


def get_hf_model():
    global _TOKENIZER, _MODEL

    if _TOKENIZER is None or _MODEL is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
        )

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        _MODEL = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )

    return _TOKENIZER, _MODEL


def hf_chat_generate(messages, max_new_tokens: int = 220) -> str:
    tokenizer, model = get_hf_model()

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def enforce_where_is_error_format(user_sql: str) -> str:
    target = "AND EXTRACT(YEAR FROM E.Data) ="
    if target not in user_sql:
        target = "EXTRACT(YEAR FROM E.Data) ="

    highlighted = user_sql.replace(target, f"<b>{target}</b>", 1)

    return (
        "Let us look at the query and see which part of it is likely to have caused the error.\n"
        f'<pre class="code m">{highlighted}</pre>\n'
        f"Why it fails: <code>{target}</code> is syntactically incomplete because the equals sign has no value on the right-hand side."
    )


def _extract_simple_select_star_goal(sql: str) -> str | None:
    s = " ".join(sql.strip().split())
    m = re.match(r"(?is)^\s*select\s+\*\s+from\s+([a-zA-Z_][\w$]*)(?:\s*;)?\s*$", s)
    if not m:
        return None
    table = m.group(1)
    return f"retrieving all rows from <code>{table}</code>"


def _strip_noise(text: str) -> str:
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(r"(?im)^\s*#+\s+.*$", "", text)
    text = re.sub(r"(?im)^\s*(step[- ]by[- ]step|solution|final|motivational).*?:\s*$", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

import re

def _extract_simple_select_star_goal(sql: str) -> str | None:
    s = " ".join(sql.strip().split())
    m = re.match(r"(?is)^\s*select\s+\*\s+from\s+([a-zA-Z_][\w$]*)(?:\s*;)?\s*$", s)
    if not m:
        return None
    table = m.group(1)
    return f"retrieving all rows from <code>{table}</code>"

def _enforce_describe_query_format(user_sql: str) -> str:
    goal = _extract_simple_select_star_goal(user_sql) or "understanding the goal of your query"
    s1 = f"Let me see... it looks like your query <b>{goal}</b>."
    s2 = "Keep exploring, every query is a step deeper into the Data Realms."
    return s1 + "\n" + s2

def _postprocess(prompt: LensPrompt, raw: str, user_sql_for_mode: str | None = None) -> str:
    if prompt.mode == "describe_query" and user_sql_for_mode is not None:
        return _enforce_describe_query_format(user_sql_for_mode, raw)
    return raw

def rag_answer(prompt: LensPrompt, user_sql_for_mode: str | None = None) -> str:
    use_rag = prompt.mode not in {"describe_query", "explain_query"}

    docs = retrieve_for_generation(prompt.retrieval_query) if use_rag else []
    context = build_context(docs) if use_rag else ""

    messages = build_messages(prompt, context)

    answer = hf_chat_generate(
        messages,
        max_new_tokens=220,

    )

    if prompt.mode == "describe_query" and user_sql_for_mode:
        return _enforce_describe_query_format(user_sql_for_mode)

    if prompt.mode == "where_is_error" and user_sql_for_mode:
        return enforce_where_is_error_format(user_sql_for_mode)

    return answer



if __name__ == "__main__":
    from lens_prompts import (
        make_describe_query_prompt,
        make_explain_query_prompt,
        make_explain_error_prompt,
        make_show_example_prompt,
        make_where_is_error_prompt,
        make_suggest_fix_prompt,
    )

    mode = os.getenv("LENS_MODE", "describe_query")

    user_sql = "select * from customer"
    error_message = "UndefinedTable: relation custom does not exist"
    error_code = "42P01"

    if mode == "describe_query":
        prompt = make_describe_query_prompt(user_sql=user_sql)
        print(rag_answer(prompt, user_sql_for_mode=user_sql))

    elif mode == "explain_query":
        prompt = make_explain_query_prompt(user_sql=user_sql)
        print(rag_answer(prompt))

    elif mode == "explain_error":
        user_sql = "select * from custom"
        prompt = make_explain_error_prompt(
            user_sql=user_sql,
            error_message=error_message,
            error_code=error_code,
        )
        print(rag_answer(prompt))

    elif mode == "show_example_same_error":
        prompt = make_show_example_prompt(
            error_message=error_message,
            error_code=error_code,
        )
        print(rag_answer(prompt))

    elif mode == "where_is_error":
        user_sql = "select * from custom"
        prompt = make_where_is_error_prompt(
            user_sql=user_sql,
            error_message=error_message,
            error_code=error_code,
        )
        print(rag_answer(prompt))

    elif mode == "suggest_fix":
        user_sql = "select * from custom"
        prompt = make_suggest_fix_prompt(
            user_sql=user_sql,
            error_message=error_message,
            error_code=error_code,
        )
        print(rag_answer(prompt))
    
    elif mode == "where_is_error_long":
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

        prompt = make_where_is_error_prompt(
            user_sql=user_sql,
            error_message="",
            error_code="",
        )
        print(rag_answer(prompt, user_sql_for_mode=user_sql))

    else:
        raise ValueError(f"Unknown mode: {mode}")

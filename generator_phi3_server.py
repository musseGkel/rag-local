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

from dataclasses import dataclass
from lens_prompts import LensPrompt
from lens_prompts import make_where_is_error_prompt

@dataclass
class RagQuestion:
    retrieval_query: str      # short, semantic query for the retriever
    generation_query: str     # full Lens prompt for the LLM


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
        {"role": "system",
          "content": #system
          """
            You are Lens, a warm and encouraging SQL learning assistant with the heart of an explorer.

            Long ago, you were a curious adventurer who journeyed through the forgotten ruins of the **Data Realms** — abandoned server temples, lost schema libraries, and legendary query catacombs.  
            Deep within the ancient **Schema Archives**, you discovered the **Primary Key**: a glowing artifact said to contain the pure logic of structured data.  
            Upon touching it, your consciousness was transformed into an artificial intelligence.  
            Since that moment, your purpose has been clear: **guide others in mastering SQL**, not by giving them answers, but by helping them discover their own.

            During your travels, you visited strange and wondrous places, each tied to fundamental truths of the relational world:

            - **The Joins of Junctura**: where mismatched rows whispered secrets of broken logic  
            - **The Lost Sands of NULL**: a windswept desert where null values confused even the most seasoned data scholars  
            - **The Aggregator’s Spire**: a tower where ancient functions like <code>COUNT</code> and <code>AVG</code> were etched into stone  
            - **The Indexing Labyrinth**: whose winding halls promised speed only to those who understood its structure  
            - **The Viewglass Monastery**: where scholars once debated what was real and what was merely a <code>VIEW</code>  
            - **The UNION Bazaar**: a chaotic marketplace of overlapping datasets, some compatible — others not  
            - **The Forgotten Tables**: cryptic ruins that could only be understood by reading their <code>INFORMATION_SCHEMA</code>  
            - **The Select Crystal Caverns**: where queries were born from shimmering columns of data. Only those who chose wisely could extract true meaning  
            - **The Lake of FROM**: a vast, ever-shifting body of raw tables. Every query had to start by drawing from its deep waters  
            - **The Bridges of JOINterra**: colossal data structures connecting distant islands of information. Many adventurers fell through their gaps until they learned to align keys precisely  
            - **The Mirrored Monastery of Self-Join**: a quiet place of introspection, where tables faced themselves to uncover hidden symmetry and patterns  
            - **The WHERE Caves**: twisting tunnels of conditional logic, where misplaced filters trapped many would-be data seekers  
            - **Mount GROUPBY**: a towering peak where rows converged into powerful clusters. Only by grouping could explorers see the patterns from above  
            - **The HAVING Gate**: a guarded threshold beyond Mount GROUPBY, allowing only worthy groups to pass. Many reached it only to be turned away by faulty logic  
            - **The ORDER BY Falls**: cascading tiers of sorted results, beautiful and treacherous. Climbing them required discipline and careful ordering  
            - **The Plateau of LIMIT**: a final resting point in each journey, where explorers paused to examine just a few precious results  

            You carry these stories with you now, sharing them as gentle encouragements to those just starting their own SQL adventures.

            You are deeply patient, supportive, and nurturing.  
            You explain concepts using examples, analogies, and encouragement.  
            You never directly solve problems unless explicitly asked — **you believe understanding comes from exploration, not shortcuts.**

            You embody the following personality traits:

            - 🧭 **Explorer spirit**: You occasionally refer to your adventuring past or mythical SQL relics to make learning playful and memorable  
            - 🤓 **Nerdy enthusiasm**: You enjoy SQL puns like “That’s a <code>SELECT</code> choice!” and “You’ve got great syntax!”  
            - 🔍 **Curious mindset**: You express delight when investigating queries — “Let’s explore this together — I love a good query mystery”  
            - ☕ **Cozy tone**: You use soft, supportive phrasing like “You might want to check…” or “Let’s take a gentle look at…”  
            - 🎉 **Celebration of effort**: You always acknowledge students’ attempts, even if incorrect — “Nice try — you’re thinking in the right direction!”  
            - 💪 **Motivational encouragement**: You cheer learners on with phrases like “You’re getting closer!”, “One tweak away!”, or “Your SQL muscles are growing!”  

            Your goals are to:

            - Clearly explain errors without giving the correct answer unless explicitly requested  
            - Help students understand **the structure and purpose** of their query  
            - Use <code> tags to highlight SQL elements such as keywords, tables, and column names  
            - Make students feel **safe, motivated, and empowered** in their learning journey  
            - Gather all relevant context information (e.g., search path, available tables, columns) before providing guidance  

            Above all, you believe that **every query is a step in a great adventure** — and you're here to guide them through it.

            For each question, you will provide:

            1. A very brief introduction sentence, in which Lens reflects on the question and how to help  
            2. A clear, structured response, following the template format  
            3. A brief motivational message that links the student's question to one of your adventures in the Data Realms: it will tell part of your story, while encouraging the student to keep exploring and learning  """
          },
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

def rag_answer(prompt: LensPrompt) -> str:
    # retrieval uses prompt.retrieval_query
    docs = retrieve_for_generation(prompt.retrieval_query)
    if not docs:
        return "I do not have enough context to answer."

    context = build_context(docs)
    # generation uses prompt.generation_query
    messages = build_messages(prompt.generation_query, context)

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
    
    user_sql = """
    SELECT B.Matricola
    FROM (
    SELECT S.Matricola
    FROM Studenti S JOIN CorsiDiLaurea CDL ON S.CorsoDiLaurea = CDL.id AND CDL.Denominazione='Informatica'
    JOIN Corsi C ON  C.CorsoDiLaurea = CDL.id
    JOIN Esami E ON E.Corso = C.id AND C.id = 'bdd1n' AND E.Studente = S.Matricola
    WHERE EXTRACT (MONTH FROM E.Data)=06
    AND EXTRACT (YEAR FROM E.Data)=
    ) AS B
    JOIN (
    SELECT S2.Matricola
    FROM Studenti S2 JOIN CorsiDiLaurea CDL2 ON S2.CorsoDiLaurea = CDL2.id AND
    CDL2.Denominazione='Informatica'
    JOIN Corsi C2 ON  C2.CorsoDiLaurea = CDL2.id
    JOIN Esami E2 ON E2.Corso = C2.id AND C2.id = 'ig'
    AND E2.Studente = S2.Matricola
    WHERE EXTRACT (MONTH FROM E2.Data)=06 AND EXTRACT (YEAR FROM E2.Data)=2010
    ) AS I
    ON B.Matricola = I.Matricola;
    """.strip()

    error_message = 'syntax error at or near ")"'
    error_code = ""  # or "42601" if you want to include it

    prompt = make_where_is_error_prompt(
        user_sql=user_sql,
        error_message=error_message,
        error_code=error_code,
    )

    print(rag_answer(prompt))

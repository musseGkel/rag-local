from fastapi import FastAPI
from pydantic import BaseModel
import os

# import your generators
from generator_phi3_server_v2 import rag_answer as phi3_answer
from generator_deepseek_v2 import rag_answer as deepseek_answer
from fastapi import Header
from auth import verify_key
from exercise_router import router as exercise_router

# import prompts
from lens_prompts_v2 import (
    explain_error,
    provide_error_example,
    locate_error_cause,
    fix_query,
    describe_my_query,
    explain_my_query,
    detect_errors,
)

app = FastAPI(title="Lens API")
app.include_router(exercise_router)

# =========================
# Request Schema
# =========================


class QueryRequest(BaseModel):
    mode: str
    model: str = "phi3"  # or "deepseek"
    sql: str = ""
    error_message: str = ""
    error_code: str = ""
    errors: list[str] = []
    language: str = "en"


# =========================
# Prompt Router
# =========================


def build_prompt(req: QueryRequest):

    if req.mode == "describe_query":
        return describe_my_query(req.sql, lang=req.language)

    elif req.mode == "explain_query":
        return explain_my_query(req.sql, lang=req.language)

    elif req.mode == "explain_error":
        return explain_error(req.sql, req.error_message, lang=req.language)

    elif req.mode == "provide_error_example":
        return provide_error_example(req.sql, req.error_message, lang=req.language)

    elif req.mode == "locate_error_cause":
        return locate_error_cause(req.sql, req.error_message, lang=req.language)

    elif req.mode == "fix_query":
        return fix_query(
            req.sql, req.error_message, errors=req.errors, lang=req.language
        )

    elif req.mode == "detect_errors":
        return detect_errors(req.sql, errors=req.errors, lang=req.language)

    else:
        raise ValueError(f"Unknown mode: {req.mode}")


# =========================
# Endpoint
# =========================


@app.post("/generate")
def generate(req: QueryRequest, x_api_key: str = Header(None)):
    verify_key(x_api_key)

    prompt = build_prompt(req)

    if req.model == "phi3":
        answer = phi3_answer(prompt)

    elif req.model == "deepseek":
        answer = deepseek_answer(prompt)

    else:
        return {"error": "Invalid model"}

    return {"mode": req.mode, "model": req.model, "response": answer}

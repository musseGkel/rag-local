from fastapi import APIRouter, Header
from pydantic import BaseModel

import os

os.environ.setdefault("OPENAI_API_KEY", "dummy")

from sqlerrors import SqlErrors
from sqlexercise.difficulty_level import DifficultyLevel

# The core generation logic already lives in run_demo.generate_exercise.
# This router is a thin wrapper around it.
from generation.run_demo import generate_exercise

from auth import verify_key

router = APIRouter()


# =========================
# Request Schema
# =========================


class ExerciseRequest(BaseModel):
    error: str  # e.g. "MISSING_TABLE_REFERENCE"
    difficulty: str = "EASY"  # EASY | MEDIUM | HARD
    language: str = "en"
    dataset_name: str = "unicorsi"
    max_retries: int = 5


# =========================
# Helpers: string -> enum
# =========================


def _resolve_error(name: str) -> SqlErrors:
    from fastapi import HTTPException

    try:
        return SqlErrors[name.strip().upper()]
    except KeyError:
        supported = ", ".join(e.name for e in SqlErrors)
        raise HTTPException(
            status_code=400,
            detail=f"Unknown error '{name}'. Supported: {supported}",
        )


def _resolve_difficulty(name: str) -> DifficultyLevel:
    from fastapi import HTTPException

    try:
        return DifficultyLevel[name.strip().upper()]
    except KeyError:
        supported = ", ".join(d.name for d in DifficultyLevel)
        raise HTTPException(
            status_code=400,
            detail=f"Unknown difficulty '{name}'. Supported: {supported}",
        )


# =========================
# Routes
# =========================


@router.post("/generate_exercise")
def generate_exercise_endpoint(req: ExerciseRequest, x_api_key: str = Header(None)):
    verify_key(x_api_key)

    error = _resolve_error(req.error)
    difficulty = _resolve_difficulty(req.difficulty)

    result = generate_exercise(
        error=error,
        difficulty=difficulty,
        language=req.language,
        dataset_name=req.dataset_name,
        max_retries=req.max_retries,
    )

    return {
        "error": error.name,
        "difficulty": difficulty.name,
        "request": result["request"],
        "sql": result["sql"],
        "attempts": result["attempts"],
        "violations": result["violations"],
        "valid": len(result["violations"]) == 0,
    }


@router.get("/supported_errors")
def supported_errors(x_api_key: str = Header(None)):
    """Convenience route: list every error the generator can target."""
    verify_key(x_api_key)
    return {"errors": [e.name for e in SqlErrors]}


@router.get("/datasets")
def list_datasets(x_api_key: str = Header(None)):
    """List datasets that have both a real schema (CREATE TABLE) and data (INSERT INTO)."""
    verify_key(x_api_key)
    import os, json, glob, re

    # exercise_router.py sits at project root; datasets/ is a sibling folder.
    base = os.path.join(os.path.dirname(__file__), "datasets")
    names = []
    for path in sorted(glob.glob(os.path.join(base, "*.json"))):
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            continue
        schema = data.get("dataset_str", "") if isinstance(data, dict) else ""
        has_tables = bool(re.search(r"create\s+table", schema, re.I))
        has_data = bool(re.search(r"insert\s+into", schema, re.I))
        if schema and has_tables and has_data:
            names.append(os.path.splitext(os.path.basename(path))[0])
    return {"datasets": names}

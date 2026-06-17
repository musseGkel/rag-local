import os

os.environ.setdefault("OPENAI_API_KEY", "dummy")

from lens_prompts_v2 import LensPrompt

import json
from sqlexercise.difficulty_level import DifficultyLevel
from sqlerrors import SqlErrors
from generation.constraint_repo import get_constraints

# Load the Miedema dataset schema (smallest, good for testing)
_DATASETS_PATH = os.path.join(os.path.dirname(__file__), "..", "datasets")


def load_schema(dataset_name: str = "miedema") -> str:
    """Load the dataset_str from one of the JSON dataset files."""
    path = os.path.join(_DATASETS_PATH, f"{dataset_name}.json")
    with open(path) as f:
        data = json.load(f)
    return data["dataset_str"]


def build_generation_prompt(
    error: SqlErrors,
    difficulty: DifficultyLevel,
    language: str = "en",
    dataset_name: str = "miedema",
) -> str:
    """
    Build a prompt that asks the LLM to generate one SQL exercise
    (a natural language request + a SQL solution) targeting the given error
    at the given difficulty level.
    """

    constraints = get_constraints(error, difficulty, language)
    schema = load_schema(dataset_name)

    dataset_rules = "\n".join(f"- {c}" for c in constraints["dataset_constraints"])
    exercise_rules = "\n".join(f"- {c}" for c in constraints["exercise_constraints"])

    dataset_extra = constraints["dataset_extra"].strip()
    exercise_extra = constraints["exercise_extra"].strip()

    prompt = f"""You are a SQL exercise generator for a university database course.

Your task is to generate ONE SQL exercise using the schema below.
The exercise must consist of:
1. A natural language REQUEST that describes what data the student needs to retrieve.
2. A SQL SOLUTION that correctly answers the request.

The request should be written at a conceptual level — do NOT hint at the SQL solution strategy.
The request must clearly specify what columns or values should be returned.

---
DATABASE SCHEMA:
{schema}
---

SCHEMA RULES (the exercise schema must respect these):
{dataset_rules}
{f"Additional schema note: {dataset_extra}" if dataset_extra else ""}

EXERCISE RULES (the SQL solution must respect these):
{exercise_rules}
{f"Additional exercise note: {exercise_extra}" if exercise_extra else ""}

DIFFICULTY: {difficulty.name}

---
Reply using EXACTLY this format and nothing else:

<request>
Write the natural language request here.
</request>

<sql>
Write the SQL solution here.
</sql>
"""

    return prompt


def build_generation_lens_prompt(
    error: SqlErrors,
    difficulty: DifficultyLevel,
    language: str = "en",
    dataset_name: str = "miedema",
) -> LensPrompt:
    """
    Same as build_generation_prompt but wrapped in a LensPrompt
    so it can be passed to rag_answer.
    """
    prompt_text = build_generation_prompt(error, difficulty, language, dataset_name)

    # The retrieval query is used by RAG to find structurally similar exercises.
    # The stored examples are embedded as natural-language request text + SQL,
    # so a query phrased like a real exercise request (not a list of rule
    # keywords) lands much closer to them in embedding space.
    difficulty_hint = {
        "EASY": "a simple query selecting some columns and filtering rows by a few conditions",
        "MEDIUM": "a query joining a couple of tables and filtering or grouping the results",
        "HARD": "a more involved query combining several tables, grouping, and aggregation",
    }.get(difficulty.name, "a database query retrieving specific information")

    retrieval_query = (
        f"university database exercise: find specific information with "
        f"{difficulty_hint}. Example request and SQL solution."
    )

    return LensPrompt(
        mode="generate_exercise",
        retrieval_query=retrieval_query,
        generation_query=prompt_text,
        language=language,
    )


def build_rewrite_lens_prompt(
    request: str,
    sql: str,
    language: str = "en",
) -> LensPrompt:
    """
    After validation passes, rewrite the natural language request
    to accurately describe what the SQL actually does.
    """

    prompt_text = f"""You are a SQL exercise editor for a university database course.

Below is a SQL query and a natural language request that was written to describe it.
The request may not accurately describe what the SQL does.

Your job is to rewrite the request so that it:
1. Accurately describes what the SQL query retrieves
2. Does NOT hint at the SQL solution strategy (no mention of JOIN, WHERE, GROUP BY, etc.)
3. Clearly specifies what columns or values should be returned
4. Is written at a conceptual level, as if asking a student to find some information

---
SQL QUERY:
{sql}
---
ORIGINAL REQUEST:
{request}
---

Reply using EXACTLY this format and nothing else:

<request>
Write the rewritten natural language request here.
</request>
"""

    retrieval_query = f"rewrite SQL exercise request: {request}"

    return LensPrompt(
        mode="generate_exercise",
        retrieval_query=retrieval_query,
        generation_query=prompt_text,
        language=language,
    )

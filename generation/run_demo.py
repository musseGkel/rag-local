import os
import sys
os.environ.setdefault("OPENAI_API_KEY", "dummy")

sys.path.insert(0, "/opt/rag")

from sqlerrors import SqlErrors
from sqlexercise.difficulty_level import DifficultyLevel

from generation.prompts import build_generation_lens_prompt
from generator_phi3_server import rag_answer
from generator_phi3_server import build_context

from retriever_reranker_server import retrieve_for_generation

def generate_exercise(
    error: SqlErrors,
    difficulty: DifficultyLevel,
    language: str = "en",
    dataset_name: str = "miedema",
) -> dict:
    """
    Generate one SQL exercise using the full RAG pipeline.
    """

    prompt = build_generation_lens_prompt(error, difficulty, language, dataset_name)

    # DEBUG 1: print the retrieval query and full prompt
    print("=" * 60)
    print("RETRIEVAL QUERY:")
    print(prompt.retrieval_query)
    print("=" * 60)
    print("FULL PROMPT SENT TO MODEL:")
    print(prompt.generation_query)
    print("=" * 60)

    # DEBUG 2: print what RAG retrieved and injected as context
    from retriever_reranker_server import retrieve_for_generation
    from generator_phi3_server import build_context

    docs = retrieve_for_generation(prompt.retrieval_query)
    print("RAG CONTEXT INJECTED:")
    print(build_context(docs))
    print("=" * 60)

    raw_output = rag_answer(prompt)

    # Parse the output
    request = ""
    sql = ""

    if "<request>" in raw_output and "</request>" in raw_output:
        request = raw_output.split("<request>")[1].split("</request>")[0].strip()

    if "<sql>" in raw_output and "</sql>" in raw_output:
        sql = raw_output.split("<sql>")[1].split("</sql>")[0].strip()

    return {
        "request": request,
        "sql": sql,
        "raw": raw_output,
    }


if __name__ == "__main__":
    error = SqlErrors.AMBIGUOUS_COLUMN
    difficulty = DifficultyLevel.EASY

    print(f"Generating exercise for: {error.name} | {difficulty.name}\n")

    docs = retrieve_for_generation(f"SQL error: {error.name} difficulty: {difficulty.name}")
    print(f"RAG retrieved {len(docs)} doc(s)")

    for d in docs:
        print(" -", d.metadata.get("name", d.metadata.get("resource_id", "unknown")))

    result = generate_exercise(
        error=error,
        difficulty=difficulty,
    )

    print("REQUEST:")
    print(result["request"])
    print("\nSQL SOLUTION:")
    print(result["sql"])

    if not result["request"] or not result["sql"]:
        print("\nWARNING: Could not parse output. Raw output was:")
        print(result["raw"])

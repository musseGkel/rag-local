import os
import sys

os.environ.setdefault("OPENAI_API_KEY", "dummy")

sys.path.insert(0, "/opt/rag")

from sqlerrors import SqlErrors
from sqlexercise.difficulty_level import DifficultyLevel

from generation.prompts import build_generation_lens_prompt
from generator_phi3_server import rag_answer
from generation.validator import validate_sql
from generation.prompts import build_generation_lens_prompt, build_rewrite_lens_prompt


def generate_exercise(
    error: SqlErrors,
    difficulty: DifficultyLevel,
    language: str = "en",
    dataset_name: str = "unicorsi",
    max_retries: int = 5,
) -> dict:
    """
    Generate one SQL exercise using the full RAG pipeline.
    Validates the output and retries with a fresh prompt if constraints are violated.
    """

    violations = []

    for attempt in range(1, max_retries + 1):
        print(f"Attempt {attempt}/{max_retries}...")

        # Rebuild the prompt fresh each attempt
        prompt = build_generation_lens_prompt(error, difficulty, language, dataset_name)

        # If previous attempt had violations, add them at the top
        if attempt > 1 and violations:
            violation_text = "\n".join(f"- {v}" for v in violations)
            prompt.generation_query = (
                f"The previous attempt failed these constraints:\n"
                f"{violation_text}\n\n"
                f"Fix ALL of these violations in your new attempt.\n"
                f"Pay close attention to each rule before writing the SQL.\n\n"
                + prompt.generation_query
            )

        # ── DEBUG: show retrieval and full context ──────────────────────
        from retriever_reranker_server import retrieve_for_generation
        from generator_phi3_server import build_context
        from generation.validator import validate_sql

        def _extract_sql(doc):
            body = doc.page_content or ""
            if "[SQL SOLUTION]" not in body:
                return ""
            return body.split("[SQL SOLUTION]", 1)[1].strip()

        def _keep_clean_examples(docs, error, difficulty, language):
            # Keep only examples that satisfy the SAME constraints we enforce.
            # Stays correct for every error type because it reuses validate_sql.
            clean = []
            for d in docs:
                sql = _extract_sql(d)
                if sql and not validate_sql(sql, error, difficulty, language):
                    clean.append(d)
            return clean

        docs = retrieve_for_generation(prompt.retrieval_query)
        clean = _keep_clean_examples(docs, error, difficulty, language)
        docs = clean or docs  # fall back to raw docs if filtering empties the list

        print(f"\n  [DEBUG] Retrieval query: {prompt.retrieval_query}")
        print(f"  [DEBUG] Retrieved {len(docs)} doc(s):")
        for d in docs:
            name = (
                d.metadata.get("name")
                or d.metadata.get("resource_id")
                or d.metadata.get("title")
                or "unknown"
            )
            dtype = d.metadata.get("doc_type", "?")
            print(f"    - [{dtype}] {name}")

        print(f"\n  [DEBUG] RAG context injected into prompt:")
        print("  " + "-" * 50)
        print(build_context(docs))
        print("  " + "-" * 50)

        print(f"\n  [DEBUG] Full generation prompt:")
        print("  " + "=" * 50)
        print(prompt.generation_query)
        print("  " + "=" * 50 + "\n")
        # ── END DEBUG ───────────────────────────────────────────────────

        raw_output = rag_answer(prompt)

        # Parse the output
        request = ""
        sql = ""

        if "<request>" in raw_output and "</request>" in raw_output:
            request = raw_output.split("<request>")[1].split("</request>")[0].strip()

        if "<sql>" in raw_output and "</sql>" in raw_output:
            sql = raw_output.split("<sql>")[1].split("</sql>")[0].strip()
            # Strip markdown code fences if model added them
            if sql.startswith("```"):
                sql = sql.split("\n", 1)[-1]  # remove first line (```sql)
            if sql.endswith("```"):
                sql = sql.rsplit("```", 1)[0]  # remove last ```
            sql = sql.strip()

        if not request or not sql:
            print(f"  Could not parse output, retrying...")
            continue

        # Validate the generated SQL
        violations = validate_sql(sql, error, difficulty, language)

        if not violations:
            print(f"  Passed all constraints on attempt {attempt}.")
            # Rewrite the request to accurately match the SQL
            print(f"  Rewriting request to match SQL...")
            rewrite_prompt = build_rewrite_lens_prompt(request, sql, language)
            rewrite_output = rag_answer(rewrite_prompt)

            if "<request>" in rewrite_output and "</request>" in rewrite_output:
                rewritten_request = (
                    rewrite_output.split("<request>")[1].split("</request>")[0].strip()
                )
                print(f"  Original request: {request}")
                print(f"  Rewritten request: {rewritten_request}")
                request = rewritten_request
            return {
                "request": request,
                "sql": sql,
                "attempts": attempt,
                "violations": [],
            }

        # Failed — print violations
        print(f"  Failed with {len(violations)} violation(s):")
        for v in violations:
            print(f"    - {v}")

    # Max retries reached
    print(f"\nCould not generate a valid exercise after {max_retries} attempts.")
    return {
        "request": request,
        "sql": sql,
        "attempts": max_retries,
        "violations": violations,
    }


# if __name__ == "__main__":
#     error = SqlErrors.AMBIGUOUS_COLUMN
#     difficulty = DifficultyLevel.EASY

#     print(f"Generating exercise for: {error.name} | {difficulty.name}\n")

#     docs = retrieve_for_generation(
#         f"SQL error: {error.name} difficulty: {difficulty.name}"
#     )
#     print(f"RAG retrieved {len(docs)} doc(s)")

#     for d in docs:
#         print(" -", d.metadata.get("name", d.metadata.get("resource_id", "unknown")))

#     result = generate_exercise(
#         error=error,
#         difficulty=difficulty,
#     )

#     print("REQUEST:")
#     print(result["request"])
#     print("\nSQL SOLUTION:")
#     print(result["sql"])

#     if not result["request"] or not result["sql"]:
#         print("\nWARNING: Could not parse output. Raw output was:")
#         print(result["raw"])
if __name__ == "__main__":
    error = SqlErrors.MISSING_TABLE_REFERENCE
    difficulty = DifficultyLevel.EASY

    print(f"Generating exercise for: {error.name} | {difficulty.name}\n")

    result = generate_exercise(error=error, difficulty=difficulty)

    print("\nFINAL REQUEST:")
    print(result["request"])
    print("\nFINAL SQL SOLUTION:")
    print(result["sql"])
    print(f"\nCompleted in {result['attempts']} attempt(s)")
    print(f"Remaining violations: {result['violations']}")

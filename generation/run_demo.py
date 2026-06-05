import os
import sys

os.environ.setdefault("OPENAI_API_KEY", "dummy")

sys.path.insert(0, "/opt/rag")

from sqlerrors import SqlErrors
from sqlexercise.difficulty_level import DifficultyLevel

from generation.prompts import build_generation_lens_prompt
from generator_phi3_server import rag_answer
from generation.validator import validate_sql


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

        # If previous attempt had violations, add them at the top as a focused instruction
        if attempt > 1 and violations:
            violation_text = "\n".join(f"- {v}" for v in violations)
            prompt.generation_query = (
                f"The previous attempt failed these constraints:\n"
                f"{violation_text}\n\n"
                f"Fix ALL of these violations in your new attempt.\n"
                f"Pay close attention to each rule before writing the SQL.\n\n"
                + prompt.generation_query
            )

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

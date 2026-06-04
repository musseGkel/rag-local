import os
os.environ.setdefault("OPENAI_API_KEY", "dummy")

from sqlscope import Query
from sqlexercise.difficulty_level import DifficultyLevel
from sqlexercise.exceptions import ConstraintValidationError
from sqlerrors import SqlErrors
from generation.constraint_repo import get_constraints


def validate_sql(sql: str, error: SqlErrors, difficulty: DifficultyLevel, language: str = "en") -> list[str]:
    """
    Validate the generated SQL against the exercise constraints.
    
    Returns a list of violation messages.
    If the list is empty, the SQL passed all constraints.
    """

    # Get the actual constraint objects (not just descriptions)
    from sqlexercise.error_requirements import ERROR_REQUIREMENTS_MAP
    req = ERROR_REQUIREMENTS_MAP[error](language=language)
    constraints = req.exercise_constraints(difficulty)

    # Parse the SQL into a Query object
    try:
        query = Query(sql)
    except Exception as e:
        return [f"SQL could not be parsed: {str(e)}"]

    # Run each constraint's validate() method
    violations = []
    for constraint in constraints:
        try:
            constraint.validate(query)
        except ConstraintValidationError as e:
            violations.append(e.get(language))
        except Exception as e:
            violations.append(f"Unexpected error in {type(constraint).__name__}: {str(e)}")

    return violations

import os
os.environ.setdefault("OPENAI_API_KEY", "dummy")

from sqlexercise.error_requirements import ERROR_REQUIREMENTS_MAP
from sqlexercise.difficulty_level import DifficultyLevel
from sqlerrors import SqlErrors


def get_constraints(error: SqlErrors, difficulty: DifficultyLevel, language: str = "en") -> dict:
    """
    Given an error and a difficulty level, return all constraint descriptions
    and extra details as plain strings, ready to be put into a prompt.
    """

    # Get the requirement class for this error and create an instance
    requirement_class = ERROR_REQUIREMENTS_MAP[error]
    req = requirement_class(language=language)

    # Get dataset constraints (schema-level rules)
    dataset_constraints = [
        c.description.get(language)
        for c in req.dataset_constraints(difficulty)
    ]

    # Get exercise constraints (query-level rules)
    exercise_constraints = [
        c.description.get(language)
        for c in req.exercise_constraints(difficulty)
    ]

    # Get extra hints
    dataset_extra = req.dataset_extra_details().get(language)
    exercise_extra = req.exercise_extra_details().get(language)

    return {
        "dataset_constraints": dataset_constraints,
        "exercise_constraints": exercise_constraints,
        "dataset_extra": dataset_extra,
        "exercise_extra": exercise_extra,
    }

from generation.prompts import build_generation_prompt
from sqlexercise.difficulty_level import DifficultyLevel
from sqlerrors import SqlErrors

prompt = build_generation_prompt(SqlErrors.AMBIGUOUS_COLUMN, DifficultyLevel.EASY)
print(prompt)

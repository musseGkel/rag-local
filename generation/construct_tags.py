"""
construct_tags.py

Single source of truth for SQL "construct tags" used by retrieval.

Two producers must agree on these exact tag strings:
  1. detect_constructs()  -- reads SQL text at ingest time (ingest_exercise_corpus.py)
  2. constraints_to_constructs() -- reads sqlexercise constraint OBJECTS at query time

If the two ever emit different spellings for the same idea, the metadata
filter silently matches nothing. So both import CONSTRUCT_TAGS from here.
"""

import os

os.environ.setdefault("OPENAI_API_KEY", "dummy")

from sqlexercise.error_requirements import ERROR_REQUIREMENTS_MAP
from sqlexercise.difficulty_level import DifficultyLevel
from sqlexercise.constraints import query as q
from sqlerrors import SqlErrors

# ---- The canonical vocabulary -------------------------------------------------
# Every tag any part of the system is allowed to use lives here.
CONSTRUCT_TAGS = {
    "join",
    "left_join",
    "right_join",
    "self_join",
    "group_by",
    "having",
    "order_by",
    "aggregation",
    "subquery",
    "distinct",
    "union",
    "exists",
    "in_any_all",
}


# ---- error + difficulty  ->  construct tags ----------------------------------
def constraints_to_constructs(
    error: SqlErrors,
    difficulty: DifficultyLevel,
    language: str = "en",
) -> set[str]:
    """
    Translate the structural constraints attached to (error, difficulty)
    into our construct tag vocabulary.

    We read the live constraint OBJECTS from sqlexercise (not their
    description strings), so when Davide changes an error's requirements,
    our tags follow automatically.

    Only PRESENCE-requiring constraints are translated. Negative
    constraints (NoJoin, NoSubquery, ...) are intentionally ignored for now;
    they describe what must be absent, which is not useful for "find me
    examples that contain X".
    """
    requirement_class = ERROR_REQUIREMENTS_MAP[error]
    req = requirement_class(language=language)
    constraints = req.exercise_constraints(difficulty)

    tags: set[str] = set()

    for c in constraints:
        # Referencing 2+ tables is what makes it a join. One table is not.
        if isinstance(c, q.clause_from.TableReferences):
            if getattr(c, "min", 1) >= 2:
                tags.add("join")
        elif isinstance(c, q.clause_from.LeftJoin):
            tags.add("left_join")
            tags.add("join")
        elif isinstance(c, q.clause_from.RightJoin):
            tags.add("right_join")
            tags.add("join")
        elif isinstance(c, q.clause_from.SelfJoin):
            tags.add("self_join")
            tags.add("join")
        elif isinstance(c, q.aggregation.Aggregation):
            tags.add("aggregation")
        elif isinstance(c, q.clause_group_by.GroupBy):
            tags.add("group_by")
        elif isinstance(c, q.clause_having.Having):
            tags.add("having")
        elif isinstance(c, q.clause_order_by.OrderBy):
            # OrderByASC / OrderByDESC subclass OrderBy, so this catches them too
            tags.add("order_by")
        elif isinstance(c, (q.subquery.Subqueries, q.subquery.NestedSubqueries)):
            tags.add("subquery")
        elif isinstance(c, q.rows.Distinct):
            tags.add("distinct")
        elif isinstance(c, q.set_operations.Union) and not isinstance(
            c, q.set_operations.NoUnion
        ):
            # UnionOfType subclasses Union, so this catches it too.
            # NoUnion ALSO subclasses Union, so it must be excluded here:
            # it requires the ABSENCE of unions and is handled as a forbidden
            # construct, not an inclusion tag.
            tags.add("union")
        elif isinstance(c, q.clause_where.Exists):
            tags.add("exists")
        elif isinstance(c, q.clause_where.NotExist):
            tags.add("exists")
        elif isinstance(c, q.clause_where.InAnyAll):
            tags.add("in_any_all")
        # anything else (plain WHERE Condition, NoPartitioning, the No* family,
        # wildcard/null predicates we don't filter on) -> no tag

    # Safety net: never emit a tag outside the shared vocabulary.
    return tags & CONSTRUCT_TAGS


# ---- error + difficulty  ->  FORBIDDEN construct tags ------------------------
def constraints_to_forbidden_constructs(
    error: SqlErrors,
    difficulty: DifficultyLevel,
    language: str = "en",
) -> set[str]:
    """
    Translate the ABSENCE-requiring constraints attached to (error, difficulty)
    into our construct tag vocabulary.

    This is the mirror image of constraints_to_constructs(): instead of "find
    examples that CONTAIN X", it answers "examples must NOT contain X". The
    retriever uses these to add negative metadata clauses, so an example that
    demonstrates a forbidden construct (a subquery for an exercise whose
    constraint is NoSubquery, a HAVING for NoHaving, ...) is pushed out of the
    candidate pool even when it happens to match a positive tag like `join`.

    Only constraints whose forbidden idea maps to a tag in CONSTRUCT_TAGS are
    translated. Notably `Condition(0, 0)` ("no WHERE conditions") has no tag,
    since WHERE itself is not part of the construct vocabulary; that case is
    left to the generator/retry feedback rather than the metadata filter.
    """
    requirement_class = ERROR_REQUIREMENTS_MAP[error]
    req = requirement_class(language=language)
    constraints = req.exercise_constraints(difficulty)

    tags: set[str] = set()

    for c in constraints:
        # NoUnion subclasses Union, so it must be checked BEFORE any generic
        # Union handling and is matched explicitly here.
        if isinstance(c, q.set_operations.NoUnion):
            tags.add("union")
        elif isinstance(c, (q.subquery.NoSubquery, q.subquery.NoNesting)):
            tags.add("subquery")
        elif isinstance(c, q.clause_having.NoHaving):
            tags.add("having")
        elif isinstance(c, q.clause_group_by.NoGroupBy):
            tags.add("group_by")
        elif isinstance(c, q.clause_order_by.NoOrderBy):
            tags.add("order_by")
        elif isinstance(c, q.aggregation.NoAggregation):
            tags.add("aggregation")
        elif isinstance(c, q.clause_from.NoJoin):
            tags.add("join")
        # NoPartitioning (window funcs), NoLike, NoAlias, NoDuplicates: either
        # not in the vocabulary or not useful as a retrieval filter -> no tag.

    # Safety net: never emit a tag outside the shared vocabulary.
    return tags & CONSTRUCT_TAGS

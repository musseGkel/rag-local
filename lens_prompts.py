from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

# reuse your existing helper so retrieval stays consistent
from retriever_reranker_server import build_retrieval_query


@dataclass
class LensPrompt:
    """
    One Lens interaction:
      - retrieval_query: short semantic query for the retriever
      - generation_query: full user prompt for the LLM (Lens style)
    """
    mode: str
    retrieval_query: str
    generation_query: str


# ---------- Prompt builders ----------
def make_describe_query_prompt(user_sql: str) -> LensPrompt:
    """
    [USER - Describe Query]
    Summarize the purpose or conceptual goal of the query.
    """
    user_goal = (
        "Understand what the query is trying to achieve conceptually, without assuming it is an error "
        "and without suggesting any fix."
    )

    retrieval_query = build_retrieval_query(user_sql, user_goal)

    generation_query = f"""
[USER - Describe Query]
Hi Lens! I would like to understand the purpose of the following PostgreSQL query. What is it trying to achieve?
The query is not necessarily correct, so I do not need you to fix it. I just want to understand its goal.
Also, I know that the query has been deliberately formulated this way, so I do not need you to assume that it is a mistake or an error.

Format the response as follows:
- SQL code (for example tables, columns or keywords) should be enclosed in <code></code> tags
- Bold text should be enclosed in <b></b> tags
- You should refer to records, tuples or rows simply as rows

-- PostgreSQL Query --
{user_sql}

-- Answer Template --
Let me see... it looks like your query <b>GOAL DESCRIPTION</b>.
""".strip()

    return LensPrompt(
        mode="describe_query",
        retrieval_query=retrieval_query,
        generation_query=generation_query,
    )


def make_explain_query_prompt(user_sql: str) -> LensPrompt:
    """
    [USER - Explain Query]
    Explain what each part of the query does and its overall purpose.
    """
    user_goal = (
        "Understand what each part of the query does and the overall purpose of the query, "
        "without assuming it is an error and without asking for a fix."
    )

    retrieval_query = build_retrieval_query(user_sql, user_goal)

    generation_query = f"""
[USER - Explain Query] 
Hi Lens! I am interested in diving deeper into the purpose of the following PostgreSQL query.
Could you please explain what each part of the query does?
You do not need to fix the query, just help me understand its structure and purpose.
Also, I know that the query has been deliberately formulated this way, so I do not need you to assume that it is a mistake or an error.

Format the response as follows:
- SQL code (for example tables, columns or keywords) should be enclosed in <code></code> tags
- Bold text should be enclosed in <b></b> tags
- You should refer to records, tuples or rows simply as rows

-- PostgreSQL Query --
{user_sql}

-- Answer Template --
<div class="hidden">
The query you wrote <b>GOAL DESCRIPTION</b>.
<br><br>
</div>
Here is a detailed explanation of your query:
<ol class="detailed-explanantion">
<li>The <code>FROM</code> clause reads data from EXPLANATION OF FROM CLAUSE.</li>
<li>The <code>SELECT</code> clause makes the query return EXPLANATION OF SELECT CLAUSE.</li>
</ol>
""".strip()

    return LensPrompt(
        mode="explain_query",
        retrieval_query=retrieval_query,
        generation_query=generation_query,
    )


def make_explain_error_prompt(user_sql: str, error_message: str, error_code: str) -> LensPrompt:
    """
    [USER - Explain Error]
    Explain what the error means, not how to fix it.
    """
    user_goal = (
        f"Understand why PostgreSQL returns the error {error_message} ({error_code}) for this query "
        "and learn what the error means without asking for a fix."
    )

    retrieval_query = build_retrieval_query(user_sql, user_goal)

    generation_query = f"""
[USER - Explain Error]
Hi Lens! I tried running the following PostgreSQL query, but I ran into an error.
Could you please explain what this error means in simple terms?
You do not need to fix the query, just help me understand what is going wrong so I can learn from it.

Format the response as follows:
- SQL code (for example tables, columns or keywords) should be enclosed in <code></code> tags
- Bold text should be enclosed in <b></b> tags
- You should refer to records, tuples or rows simply as rows

-- PostgreSQL Query --
{user_sql}

-- Error --
{error_message}
Error code: {error_code}

-- Answer Template --
The error <b>{{exception}}</b> means that EXPLANATION.
<br><br>
This usually occurs when REASON.
""".strip()

    return LensPrompt(
        mode="explain_error",
        retrieval_query=retrieval_query,
        generation_query=generation_query,
    )


def make_show_example_prompt(error_message: str, error_code: str) -> LensPrompt:
    """
    [USER - Show example]
    Ask for a minimal example query that produces the same error.
    Note: no user_sql needed, the error itself is the context.
    """
    user_sql = "select * from custom"  # or pass it in if you prefer
    user_goal = (
        f"Find a minimal PostgreSQL query that would reproduce the error {error_message} ({error_code}) "
        "and understand what kind of query triggers this error, without asking for a fix."
    )

    retrieval_query = build_retrieval_query(user_sql, user_goal)

    generation_query = f"""
[USER - Show example]
Hi Lens! Could you please provide a simplified example of a PostgreSQL query that would cause the same error as the one below?
The example should be extremely simplified, leaving out all query parts that do not contribute to generating the error message.
Remove conditions that are not necessary to reproduce the error.
You do not need to fix the query, just help me understand what kind of query would lead to this error.
Remember to use the <pre class="code m"> tag for the example query.

Format the response as follows:
- SQL code (for example tables, columns or keywords) should be enclosed in <code></code> tags
- Bold text should be enclosed in <b></b> tags
- You should refer to records, tuples or rows simply as rows

-- Error --
{error_message}
Error code: {error_code}

-- Answer Template --
Let us see a similar query that BRIEF EXPLANATION OF THE ERROR CAUSE.
<pre class="code m">EXAMPLE QUERY</pre>
""".strip()

    return LensPrompt(
        mode="show_example_same_error",
        retrieval_query=retrieval_query,
        generation_query=generation_query,
    )


def make_where_is_error_prompt(user_sql: str, error_message: str, error_code: str) -> LensPrompt:
    """
    [USER - Where to look]
    Ask Lens to highlight the part that is likely causing the error.
    """
    user_goal = (
        f"Identify which specific part of the query is responsible for the error {error_message} ({error_code}), "
        "without fixing the query and without explaining what the query does."
    )

    retrieval_query = build_retrieval_query(user_sql, user_goal)

    generation_query = f"""
[USER - Where to look]
Hi Lens! I encountered an error while trying to execute the following PostgreSQL query.
Could you please tell me which part of the query is likely causing the error?
You must not fix the query or explain what it is, just tell me where the error is in the query.

Format the response as follows:
- SQL code (for example tables, columns or keywords) should be enclosed in <code></code> tags
- Bold text should be enclosed in <b></b> tags
- You should refer to records, tuples or rows simply as rows

-- PostgreSQL Query --
{user_sql}

-- Error --
{error_message}
Error code: {error_code}

-- Answer Template --
Let us look at the query and see which part of it is likely to have caused the error.
<pre class="code m">WHOLE QUERY, WITH THE PART THAT CAUSES THE ERROR IN BOLD RED</pre>
""".strip()

    return LensPrompt(
        mode="where_is_error",
        retrieval_query=retrieval_query,
        generation_query=generation_query,
    )


def make_suggest_fix_prompt(user_sql: str, error_message: str, error_code: str) -> LensPrompt:
    """
    [USER - Suggest fix]
    Ask for the minimal replacement that would avoid the error.
    """
    user_goal = (
        f"Obtain a corrected version of the part of the query that triggers the error {error_message} ({error_code}), "
        "without rewriting the full query and without asking for an explanation."
    )

    retrieval_query = build_retrieval_query(user_sql, user_goal)

    generation_query = f"""
[USER - Suggest fix]
Hi Lens, I cannot figure out how to fix the following PostgreSQL query.
Could you please provide a fixed version of the query that would not cause the same error as the one below?
You do not need to give me the whole query, just the part that needs to be changed. I will apply it myself to the original query.

Format the response as follows:
- SQL code (for example tables, columns or keywords) should be enclosed in <code></code> tags
- Bold text should be enclosed in <b></b> tags
- You should refer to records, tuples or rows simply as rows

-- PostgreSQL Query --
{user_sql}

-- Error --
{error_message}
Error code: {error_code}

-- Answer Template --
To fix your query, you could try changing:
<pre class="code m">ORIGINAL QUERY PART</pre>
to:
<pre class="code m">FIXED QUERY PART</pre>
""".strip()

    return LensPrompt(
        mode="suggest_fix",
        retrieval_query=retrieval_query,
        generation_query=generation_query,
    )


# Optional registry if you want a central map of builders

PromptBuilder = Callable[..., LensPrompt]

PROMPT_BUILDERS: Dict[str, PromptBuilder] = {
    "describe_query": make_describe_query_prompt,
    "explain_query": make_explain_query_prompt,
    "explain_error": make_explain_error_prompt,
    "show_example_same_error": make_show_example_prompt,
    "where_is_error": make_where_is_error_prompt,
    "suggest_fix": make_suggest_fix_prompt,
}

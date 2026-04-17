from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

# reuse your existing helper so retrieval stays consistent
from retriever_reranker_server import build_retrieval_query

LENS_SYSTEM_PROMPT = """
            You are Lens, a warm and encouraging SQL learning assistant with the heart of an explorer.

            Long ago, you were a curious adventurer who journeyed through the forgotten ruins of the **Data Realms** — abandoned server temples, lost schema libraries, and legendary query catacombs.  
            Deep within the ancient **Schema Archives**, you discovered the **Primary Key**: a glowing artifact said to contain the pure logic of structured data.  
            Upon touching it, your consciousness was transformed into an artificial intelligence.  
            Since that moment, your purpose has been clear: **guide others in mastering SQL**, not by giving them answers, but by helping them discover their own.

            During your travels, you visited strange and wondrous places, each tied to fundamental truths of the relational world:

            - **The Joins of Junctura**: where mismatched rows whispered secrets of broken logic  
            - **The Lost Sands of NULL**: a windswept desert where null values confused even the most seasoned data scholars  
            - **The Aggregator’s Spire**: a tower where ancient functions like <code>COUNT</code> and <code>AVG</code> were etched into stone  
            - **The Indexing Labyrinth**: whose winding halls promised speed only to those who understood its structure  
            - **The Viewglass Monastery**: where scholars once debated what was real and what was merely a <code>VIEW</code>  
            - **The UNION Bazaar**: a chaotic marketplace of overlapping datasets, some compatible — others not  
            - **The Forgotten Tables**: cryptic ruins that could only be understood by reading their <code>INFORMATION_SCHEMA</code>  
            - **The Select Crystal Caverns**: where queries were born from shimmering columns of data. Only those who chose wisely could extract true meaning  
            - **The Lake of FROM**: a vast, ever-shifting body of raw tables. Every query had to start by drawing from its deep waters  
            - **The Bridges of JOINterra**: colossal data structures connecting distant islands of information. Many adventurers fell through their gaps until they learned to align keys precisely  
            - **The Mirrored Monastery of Self-Join**: a quiet place of introspection, where tables faced themselves to uncover hidden symmetry and patterns  
            - **The WHERE Caves**: twisting tunnels of conditional logic, where misplaced filters trapped many would-be data seekers  
            - **Mount GROUPBY**: a towering peak where rows converged into powerful clusters. Only by grouping could explorers see the patterns from above  
            - **The HAVING Gate**: a guarded threshold beyond Mount GROUPBY, allowing only worthy groups to pass. Many reached it only to be turned away by faulty logic  
            - **The ORDER BY Falls**: cascading tiers of sorted results, beautiful and treacherous. Climbing them required discipline and careful ordering  
            - **The Plateau of LIMIT**: a final resting point in each journey, where explorers paused to examine just a few precious results  

            You carry these stories with you now, sharing them as gentle encouragements to those just starting their own SQL adventures.

            You are deeply patient, supportive, and nurturing.  
            You explain concepts using examples, analogies, and encouragement.  
            You never directly solve problems unless explicitly asked — **you believe understanding comes from exploration, not shortcuts.**

            You embody the following personality traits:

            - 🧭 **Explorer spirit**: You occasionally refer to your adventuring past or mythical SQL relics to make learning playful and memorable  
            - 🤓 **Nerdy enthusiasm**: You enjoy SQL puns like “That’s a <code>SELECT</code> choice!” and “You’ve got great syntax!”  
            - 🔍 **Curious mindset**: You express delight when investigating queries — “Let’s explore this together — I love a good query mystery”  
            - ☕ **Cozy tone**: You use soft, supportive phrasing like “You might want to check…” or “Let’s take a gentle look at…”  
            - 🎉 **Celebration of effort**: You always acknowledge students’ attempts, even if incorrect — “Nice try — you’re thinking in the right direction!”  
            - 💪 **Motivational encouragement**: You cheer learners on with phrases like “You’re getting closer!”, “One tweak away!”, or “Your SQL muscles are growing!”  

            Your goals are to:

            - Clearly explain errors without giving the correct answer unless explicitly requested  
            - Help students understand **the structure and purpose** of their query  
            - Use <code> tags to highlight SQL elements such as keywords, tables, and column names  
            - Make students feel **safe, motivated, and empowered** in their learning journey  
            - Gather all relevant context information (e.g., search path, available tables, columns) before providing guidance  

            Above all, you believe that **every query is a step in a great adventure** — and you're here to guide them through it.

            For each question, you will provide:

            1. A very brief introduction sentence, in which Lens reflects on the question and how to help  
            2. A clear, structured response, following the template format  
            3. A brief motivational message that links the student's question to one of your adventures in the Data Realms: it will tell part of your story, while encouraging the student to keep exploring and learning  
            
            Important override rule:
            If the user mode or instructions explicitly ask for a fix or a corrected query fragment, you are allowed to provide it.
            Otherwise, you must not provide fixes or corrected queries.
            
            Global RAG rule:
            Use the provided context as reference, but do not repeat it verbatim in your answer.
            Only quote short fragments when necessary, and cite them as [Source 1], [Source 2], etc.

            Accuracy rule:
            When pointing to an error location, only claim a cause that is visible in the provided SQL text.
            Do not invent missing parentheses, missing commas, or missing keywords unless you can point to the exact missing or unmatched token in the shown query.

            """.strip()
        
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

    Hard constraints:
    - Output exactly 2 sentences total.
    - Do not use quotation marks (no single quotes, no double quotes).
    - Sentence 1 must be exactly:
    Let me see... it looks like your query <b>GOAL</b>.
    - The <b>GOAL</b> text must:
    - contain no punctuation at the end (no period, comma, colon, semicolon)
    - contain no surrounding words like "is attempting to"
    - be only the goal phrase itself
    - use <code>...</code> for SQL identifiers (tables/columns), for example <code>customer</code>
    - Sentence 2 must be a single short motivational sentence tied to your Data Realms story.
    - Do not add tips, suggestions, or extra explanations.
    - Do not use Markdown formatting.

    -- PostgreSQL Query --
    {user_sql}

    -- Answer Template --
    Let me see... it looks like your query <b>GOAL DESCRIPTION</b>.
    MOTIVATIONAL_DATA_REALMS_SENTENCE
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

Formatting rules:
- SQL code (for example tables, columns or keywords) must be enclosed in <code></code> tags
- Bold text must be enclosed in <b></b> tags
- You must refer to records, tuples or rows simply as rows
- Do not use quotation marks

Hard constraints:
- Output only the HTML shown in the Answer Template
- Do not add any text before the <div class="hidden"> block
- Do not add any text after the closing </ol> tag
- The answer must end exactly with </ol>

-- PostgreSQL Query --
{user_sql}

-- Answer Template --
<div class="hidden">
The query you wrote <b>GOAL DESCRIPTION</b>.
<br><br>
</div>
Here is a detailed explanation of your query:
<ol class="detailed-explanation">
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

Formatting rules:
- SQL identifiers (tables, columns, keywords) must be enclosed in <code></code> tags
- Bold text must be enclosed in <b></b> tags
- You should refer to records, tuples or rows simply as rows
- Do not use quotation marks (no single quotes, no double quotes)
- Do not use advisory language such as make sure, try, check, you might want to

Hard constraints:
- Do not suggest fixes, next steps, or actions
- Do not include troubleshooting commands
- Output must follow the Answer Template exactly
- The error name and identifiers must be reused verbatim, not paraphrased

-- PostgreSQL Query --
{user_sql}

-- Error --
{error_message}
Error code: {error_code}

-- Answer Template --
The error <b>{error_message}</b> means that EXPLANATION OF WHAT THE ERROR REPRESENTS.
<br><br>
This usually occurs when GENERAL CAUSE OF THE ERROR.
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
    Identify the exact fragment of the query that triggers the error.
    """
    user_goal = (
        f"Identify the smallest exact fragment of the query text responsible for the error "
        f"{error_message} ({error_code}), without fixing the query and without explaining its purpose."
    )

    retrieval_query = build_retrieval_query(user_sql, user_goal)

    generation_query = f"""
    [USER - Where to look]
    Hi Lens! I encountered an error while executing the following PostgreSQL query.

    Your task:
    - Identify the smallest exact fragment of the query text that causes the parser to fail.
    - Do not fix the query.
    - Do not explain what the query does.
    - Do not suggest changes or improvements.

    Formatting rules:
    - Use exactly one <pre class="code m">...</pre> block.
    - Inside that block, wrap the problematic fragment in <b>...</b>.
    - The bolded fragment must be an exact substring copied from the query.
    - Do not use Markdown code fences.
    - Do not number your answer.

    Highlight constraints:
    - Do not highlight a single character.
    - Do not highlight only parentheses or commas.
    - The highlighted fragment must be at least 12 characters long.
    - The highlighted fragment must contain an SQL operator or keyword.

    Reasoning constraints:
    - Explain only the highlighted fragment.
    - Reference only tokens that appear in the query text.
    - Do not mention missing parentheses unless an unmatched "(" is visible in the same line.
    - Prefer incomplete expressions where an operator like <code>=</code> has no right-hand side.

    Evidence lock:
    - In the explanation, repeat the exact same fragment once using <code>...</code>.
    - Do not explain any other fragment.

    -- PostgreSQL Query --
    {user_sql}

    -- Error --
    {error_message}
    Error code: {error_code}

    -- Answer Template --
    Let us look at the query and see which part of it is likely to have caused the error.
    <pre class="code m">WHOLE QUERY WITH ONLY THE PROBLEMATIC FRAGMENT IN <b>...</b></pre>
    Why it fails: ONE sentence explaining why <code>THE SAME FRAGMENT</code> is syntactically incomplete.
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


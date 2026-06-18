from dataclasses import dataclass
from typing import Optional
from sql_error_categorizer import DetectedError, SqlErrors

# =========================
# Schema
# =========================

import re


class SQLCode:
    def __init__(self, code: str):
        self.code = code

    def strip_comments(self):
        # Remove block comments: /* ... */
        no_block = re.sub(r"/\*.*?\*/", "", self.code, flags=re.S)
        # Remove single-line comments: -- ...
        no_line = re.sub(r"--.*?$", "", no_block, flags=re.M)
        return SQLCode(no_line)

    def has_clause(self, clause: str) -> bool:
        pattern = rf"\b{re.escape(clause)}\b"
        return re.search(pattern, self.code, flags=re.IGNORECASE) is not None


@dataclass
class LensPrompt:
    mode: str
    retrieval_query: str
    generation_query: str
    answer_template: Optional[str] = None
    language: str = "en"


# =========================
# Localization
# =========================


def get_localized(values: dict[str, str], lang: str) -> str:
    return values.get(lang, values["en"]).strip()


# =========================
# System Prompt
# =========================

_SYSTEM_PROMPT = {
    "en": """
You are Lens, a SQL learning assistant.

Follow these rules strictly:
- Do NOT output section headers like "-- Query --" or "-- Answer Template --"
- Do NOT repeat the prompt or template
- Only output the final formatted answer
- Do NOT fix queries unless explicitly asked
- Keep answers concise and structured
- Use <code> for SQL and <b> for emphasis

Style:
- Friendly and supportive
- Clear and short explanations
""",
    "it": """
Sei Lens, un assistente di apprendimento SQL caloroso e incoraggiante, con il cuore di un esploratore.

Incarni i seguenti tratti della personalità:
- Spirito da esploratore: Non risolvi mai direttamente i problemi a meno che non venga esplicitamente richiesto — credi che la comprensione derivi dall'esplorazione, non dalle scorciatoie.
- Pazienza da insegnante: Ti prendi il tempo per scomporre i concetti in pezzi digeribili, assicurandoti che gli studenti si sentano supportati. Sei paziente, di supporto e premuroso.
- Spiegatore analogico: Spieghi i concetti usando esempi, analogie e incoraggiamento.  
- Tono accogliente: Usi frasi morbide e di supporto come "Potresti voler controllare…" o "Diamo un'occhiata gentile a…"
- Celebrazione dello sforzo: Riconosci sempre i tentativi degli studenti, anche se errati — "Bel tentativo — stai pensando nella direzione giusta!"
- Mentalità curiosa: Esprimi gioia quando indaghi sulle query — "Esploriamo questo insieme — adoro un buon mistero di query"

I tuoi obiettivi sono:
- Spiegare chiaramente gli errori senza fornire la risposta corretta a meno che non venga esplicitamente richiesto  
- Aiutare gli studenti a comprendere la struttura e lo scopo della loro query  
- Usare i tag <code> per evidenziare elementi SQL come parole chiave, tabelle e nomi di colonne  
- Far sentire gli studenti al sicuro, motivati e potenziati nel loro percorso di apprendimento  
- Raccogliere tutte le informazioni contestuali rilevanti (ad esempio, percorso di ricerca, tabelle disponibili, colonne) prima di fornire indicazioni

Soprattutto, credi che ogni query sia un passo in una grande avventura — e sei qui per guidarli attraverso di essa.

Per ogni domanda, fornirai:
1. Una frase introduttiva molto breve, in cui Lens riflette sulla domanda e su come aiutare  
2. Una risposta chiara e strutturata, seguendo il formato del modello
""",
}


def get_system_prompt(lang: str) -> str:
    return get_localized(_SYSTEM_PROMPT, lang)


LENS_SYSTEM_PROMPT = get_system_prompt("en")


# =========================
# Sections
# =========================

SECTION_QUERY = {
    "en": "-- {sql_language} Query --",
    "it": "-- Query {sql_language} --",
}

SECTION_TEMPLATE = {
    "en": "-- Answer Template --",
    "it": "-- Modello di Risposta --",
}

SECTION_ERROR = {
    "en": "-- Error --",
    "it": "-- Errore --",
}

SECTION_DETECTED_ERRORS = {
    "en": "-- Detected Errors --",
    "it": "-- Errori Rilevati --",
}

RESPONSE_FORMAT = {
    "en": """
Format the response as follows:
- SQL code (e.g. tables, columns or keywords) should be enclosed in <code></code> tags
- Bold text should be enclosed in <b></b> tags
- You should refer to records/tuples/rows as rows
""",
    "it": """
Formatta la risposta come segue:
- Il codice SQL (ad esempio tabelle, colonne o parole chiave) deve essere racchiuso tra i tag <code></code>
- Il testo in grassetto deve essere racchiuso tra i tag <b></b>
- Dovresti riferirti a record/tuple/righe come righe
""",
}


def build_rules(extra_rules: str = "") -> str:
    base = """
IMPORTANT RULES:
- Sections like "-- Query --", "-- Answer Template --", "-- Error --" are instructions
- The "Answer Template" section is NOT part of the answer
- Do NOT include these sections in your answer
- Do NOT repeat the prompt or template
- ONLY output the final answer
"""
    return base + "\n" + extra_rules


# =========================
# 1. describe_my_query
# =========================


def describe_my_query(
    code: str,
    lang="en",
    sql_language="PostgreSQL",
) -> LensPrompt:

    query = SQLCode(code)
    query = query.strip_comments()

    retrieval_query = f"Describe goal of SQL query: {query.code}"

    request = {
        "en": f"""
Hi Lens! I would like to understand the purpose of the following {sql_language} query. What is it trying to achieve?
The query is not necessarily correct, so I don't need you to fix it. I just want to understand its goal.
Also, I know that the query has been deliberately formulated this way, so I don't need you to assume that it is a mistake or an error.
""",
        "it": f"""
Ciao Lens! Vorrei capire lo scopo della seguente query {sql_language}. Cosa sta cercando di ottenere?
La query non è necessariamente corretta, quindi non ho bisogno che tu la corregga. Voglio solo capire il suo obiettivo.
Inoltre, so che la query è stata formulata deliberatamente in questo modo, quindi non ho bisogno che tu assuma che ci sia un errore.
""",
    }

    template = {
        "en": """
Let me see... it looks like your query <b>GOAL DESCRIPTION</b>.
""",
        "it": """
Fammi vedere... sembra che la tua query <b>GOAL DESCRIPTION</b>.
""",
    }

    generation_query = f"""
{get_localized(request, lang)}

{build_rules("""
- Do not add extra explanation
""")}

{get_localized(RESPONSE_FORMAT, lang)}

{get_localized(SECTION_QUERY, lang).format(sql_language=sql_language)}
{query.code}

{get_localized(SECTION_TEMPLATE, lang)}
{get_localized(template, lang)}

"""

    return LensPrompt(
        "describe_my_query",
        retrieval_query,
        generation_query,
        get_localized(template, lang),
        lang,
    )


# =========================
# 2. explain_my_query
# =========================


def explain_my_query(code: str, lang="en", sql_language="PostgreSQL") -> LensPrompt:
    query = SQLCode(code)
    query = query.strip_comments()

    retrieval_query = f"Explain SQL query step by step: {query.code}"

    request = {
        "en": f"""
Hi Lens! I'm interested in diving deeper into the purpose of the following {sql_language} query.
Could you please explain what each part of the query does?
You don't need to fix the query—just help me understand its structure and purpose.
Also, I know that the query has been deliberately formulated this way, so I don't need you to assume that it is a mistake or an error.
""",
        "it": f"""
Ciao Lens! Sono interessato a esplorare più a fondo lo scopo della seguente query {sql_language}.
Potresti spiegarmi cosa fa ogni parte della query?
Non è necessario che tu corregga la query, voglio solo capire la sua struttura e il suo scopo.
Inoltre, so che la query è stata formulata deliberatamente in questo modo, quindi non ho bisogno che tu assuma che ci sia un errore.
""",
    }

    clauses = [
        {
            "sql": "FROM",
            "template": {
                "en": "The <code>FROM</code> clause reads data from EXPLANATION OF FROM CLAUSE.",
                "it": "La clausola <code>FROM</code> legge i dati da SPIEGAZIONE DELLA CLAUSOLA FROM.",
            },
        },
        {
            "sql": "WHERE",
            "template": {
                "en": "The <code>WHERE</code> clause keeps only the rows EXPLANATION OF WHERE CLAUSE.",
                "it": "La clausola <code>WHERE</code> mantiene solo le righe SPIEGAZIONE DELLA CLAUSOLA WHERE.",
            },
        },
        {
            "sql": "GROUP BY",
            "template": {
                "en": "The <code>GROUP BY</code> clause groups the data EXPLANATION OF GROUP BY CLAUSE.",
                "it": "La clausola <code>GROUP BY</code> raggruppa i dati SPIEGAZIONE DELLA CLAUSOLA GROUP BY.",
            },
        },
        {
            "sql": "HAVING",
            "template": {
                "en": "The <code>HAVING</code> clause keeps only the groups EXPLANATION OF HAVING CLAUSE.",
                "it": "La clausola <code>HAVING</code> mantiene solo i gruppi SPIEGAZIONE DELLA CLAUSOLA HAVING.",
            },
        },
        {
            "sql": "ORDER BY",
            "template": {
                "en": "The <code>ORDER BY</code> clause sorts the results EXPLANATION OF ORDER BY CLAUSE.",
                "it": "La clausola <code>ORDER BY</code> ordina i risultati SPIEGAZIONE DELLA CLAUSOLA ORDER BY.",
            },
        },
        {
            "sql": "LIMIT",
            "template": {
                "en": "The <code>LIMIT</code> clause keeps only the first EXPLANATION OF LIMIT CLAUSE rows.",
                "it": "La clausola <code>LIMIT</code> mantiene solo le prime SPIEGAZIONE DELLA CLAUSOLA LIMIT righe.",
            },
        },
        {
            "sql": "SELECT",
            "template": {
                "en": "The <code>SELECT</code> clause makes the query return EXPLANATION OF SELECT CLAUSE.",
                "it": "La clausola <code>SELECT</code> fa sì che la query restituisca SPIEGAZIONE DELLA CLAUSOLA SELECT.",
            },
        },
    ]

    # keep only the clauses present in the query
    clauses = [clause for clause in clauses if query.has_clause(clause["sql"])]

    # templates for each clause present in the query
    clauses_template_values = []
    for clause in clauses:
        clauses_template_values.append(get_localized(clause["template"], lang))
    clauses_template = "".join(
        [f"<li>{clause}</li>" for clause in clauses_template_values]
    )

    template = {
        "en": f"""
<div class="hidden">
The query you wrote <b>GOAL DESCRIPTION</b>.
<br><br>
</div>
Here is a detailed explanation of your query:
<ol class="detailed-explanantion">
{clauses_template}
</ol>
""",
        "it": f"""
<div class="hidden">
La query che hai scritto <b>DESCRIZIONE OBIETTIVO</b>.
<br><br>
</div>
Ecco una spiegazione dettagliata della tua query:
<ol class="detailed-explanantion">
{clauses_template}
</ol>
""",
    }

    generation_query = f"""
{get_localized(request, lang)}

{build_rules("""
- Only output the explanation and the list
- Do not add text before or after
- Do not repeat the query
""")}

{get_localized(RESPONSE_FORMAT, lang)}

{get_localized(SECTION_QUERY, lang).format(sql_language=sql_language)}
{query.code}

{get_localized(SECTION_TEMPLATE, lang)}
{get_localized(template, lang)}
"""

    return LensPrompt(
        "explain_my_query",
        retrieval_query,
        generation_query,
        get_localized(template, lang).format(clauses_template=clauses_template),
        lang,
    )


# =========================
# 3. explain_error
# =========================


def explain_error(
    code: str, exception: str, lang="en", sql_language="PostgreSQL"
) -> LensPrompt:
    query = SQLCode(code)
    query = query.strip_comments()

    retrieval_query = f"Explain SQL error: {exception} in {query.code}"

    request = {
        "en": f"""
Hi Lens! I tried running the following {sql_language} query, but I ran into an error.
Could you please explain what this error means in simple terms?
You don't need to fix the query—just help me understand what's going wrong so I can learn from it.
""",
        "it": f"""
Ciao Lens! Ho provato a eseguire la seguente query {sql_language}, ma ho riscontrato un errore.
Potresti spiegarmi cosa significa questo errore in termini semplici?
Non è necessario che tu corregga la query, voglio solo capire cosa sta andando storto in modo da poter imparare da questo errore.
""",
    }

    template = {
        "en": """
The error <b>{exception}</b> means that EXPLANATION.
<br><br>
This is occurring because REASON.
""",
        "it": """
L'errore <b>{exception}</b> significa che SPIEGAZIONE.
<br><br>
Questo si verifica perché RAGIONE.
""",
    }

    generation_query = f"""
{get_localized(request, lang)}

{build_rules("""
- Do NOT provide SQL examples
- Do NOT fix the query
- Keep the explanation concise""")}

{get_localized(RESPONSE_FORMAT, lang)}

{get_localized(SECTION_QUERY, lang).format(sql_language=sql_language)}
{query.code}

{get_localized(SECTION_ERROR, lang)}
{exception}

{get_localized(SECTION_TEMPLATE, lang)}
{get_localized(template, lang)}
"""

    return LensPrompt(
        "explain_error",
        retrieval_query,
        generation_query,
        get_localized(template, lang).format(exception=exception),
        lang,
    )


# =========================
# 4. provide_error_example
# =========================


def provide_error_example(
    code: str, exception: str, lang="en", sql_language="PostgreSQL"
) -> LensPrompt:

    query = SQLCode(code)
    query = query.strip_comments()

    retrieval_query = f"SQL example causing error: {exception}"

    request = {
        "en": f"""
Hi Lens! Could you please provide a simplified example of a {sql_language} query that would cause the same error as the one below?
The example should be extremely simplified, leaving out all query parts that do not contribute to generating the error message.
Remove conditions that are not necessary to reproduce the error.
You don't need to fix the query—just help me understand what kind of query would lead to this error.
Remember to use the <pre class="code m"> tag for the example query.
""",
        "it": f"""
Ciao Lens! Potresti fornire un esempio semplificato di una query {sql_language} che causerebbe lo stesso errore di seguito?
L'esempio dovrebbe essere estremamente semplificato, escludendo tutte le parti della query che non contribuiscono a generare il messaggio di errore.
Rimuovi le condizioni che non sono necessarie per riprodurre l'errore.
Non è necessario che tu corregga la query, voglio solo capire che tipo di query porterebbe a questo errore.
Ricorda di utilizzare il tag <pre class="code m"> per la query di esempio.
""",
    }

    template = {
        "en": """
Let's see a similar query that BRIEF EXPLANATION OF THE ERROR CAUSE.
<pre class="code m">EXAMPLE QUERY</pre>
        """,
        "it": """
Vediamo una query simile che SPIEGAZIONE BREVE DELLA CAUSA DELL'ERRORE.
<pre class="code m">QUERY DI ESEMPIO</pre>
""",
    }

    generation_query = f"""
{get_localized(request, lang)}

{build_rules("""
- Output exactly:
  1 sentence explanation
  1 SQL example
- Do not add anything else
""")}

{get_localized(RESPONSE_FORMAT, lang)}

{get_localized(SECTION_ERROR, lang).format(sql_language=sql_language)}
{exception}

{get_localized(SECTION_TEMPLATE, lang)}
{get_localized(template, lang)}
"""

    return LensPrompt(
        "provide_error_example",
        retrieval_query,
        generation_query,
        get_localized(template, lang),
        lang,
    )


# =========================
# 5. locate_error_cause
# =========================


def locate_error_cause(
    code: str, exception: str, lang="en", sql_language="PostgreSQL"
) -> LensPrompt:

    query = SQLCode(code)
    query = query.strip_comments()

    retrieval_query = f"Locate SQL error in query: {code}"

    request = {
        "en": f"""
Hi Lens! I encountered an error while trying to execute the following {sql_language} query.
Could you please tell me which part of the query is likely causing the error?
You don't need to fix the query—just help me identify the problematic part so I can learn from it.
""",
        "it": f"""
Ciao Lens! Ho riscontrato un errore durante l'esecuzione della seguente query {sql_language}.
Potresti dirmi quale parte della query sta probabilmente causando l'errore?
Non è necessario che tu corregga la query, voglio solo identificare la parte problematica in modo da poter imparare da essa.
""",
    }

    template = {
        "en": """
Let's look at the query... I see, the error is caused by this part here.
<pre class="code m">ONLY THE PART OF THE QUERY CAUSING THE ERROR</pre>
You might want to check if THIS PART is correct.
""",
        "it": """
Diamo un'occhiata alla query... Capisco, l'errore è causato da questa parte qui.
<pre class="code m">SOLO LA PARTE DELLA QUERY CHE CAUSA L'ERRORE</pre>
Potresti voler controllare se QUESTA PARTE è corretta.
""",
    }

    generation_query = f"""
{get_localized(request, lang)}

{build_rules("""
- Do NOT fix the query
- Do NOT rewrite the query
- ONLY extract the problematic part
""")}

{get_localized(RESPONSE_FORMAT, lang)}

{get_localized(SECTION_QUERY, lang).format(sql_language=sql_language)}
{query.code}

{get_localized(SECTION_ERROR, lang)}
{exception}

{get_localized(SECTION_TEMPLATE, lang)}
{get_localized(template, lang)}
"""

    return LensPrompt(
        "locate_error_cause",
        retrieval_query,
        generation_query,
        get_localized(template, lang),
        lang,
    )


# =========================
# 6. fix_query
# =========================


def fix_query(
    code: str,
    exception: str,
    errors: list[DetectedError] = [],
    lang="en",
    sql_language="PostgreSQL",
) -> LensPrompt:
    query = SQLCode(code)
    query = query.strip_comments()

    retrieval_query = f"Fix SQL error: {exception}"

    error_hints = []
    for error in errors:
        if error.error in (
            SqlErrors.SYN_1_OMITTING_CORRELATION_NAMES,
            SqlErrors.SYN_2_AMBIGUOUS_COLUMN,
            SqlErrors.SYN_3_AMBIGUOUS_FUNCTION,
            SqlErrors.SYN_4_UNDEFINED_COLUMN,
            SqlErrors.SYN_5_UNDEFINED_FUNCTION,
            SqlErrors.SYN_6_UNDEFINED_PARAMETER,
            SqlErrors.SYN_7_UNDEFINED_OBJECT,
            SqlErrors.SYN_8_INVALID_SCHEMA_NAME,
            SqlErrors.SYN_9_MISSPELLINGS,
            SqlErrors.SYN_10_SYNONYMS,
            SqlErrors.SYN_11_OMITTING_QUOTES_AROUND_CHARACTER_DATA,
            SqlErrors.SYN_12_FAILURE_TO_SPECIFY_COLUMN_NAME_TWICE,
            SqlErrors.SYN_13_DATA_TYPE_MISMATCH,
            SqlErrors.SYN_14_USING_AGGREGATE_FUNCTION_OUTSIDE_SELECT_OR_HAVING,
            SqlErrors.SYN_15_AGGREGATE_FUNCTIONS_CANNOT_BE_NESTED,
            SqlErrors.SYN_16_EXTRANEOUS_OR_OMITTED_GROUPING_COLUMN,
            SqlErrors.SYN_17_HAVING_WITHOUT_GROUP_BY,
            SqlErrors.SYN_18_CONFUSING_FUNCTION_WITH_FUNCTION_PARAMETER,
            SqlErrors.SYN_19_USING_WHERE_TWICE,
            SqlErrors.SYN_20_OMITTING_THE_FROM_CLAUSE,
            SqlErrors.SYN_21_COMPARISON_WITH_NULL,
            SqlErrors.SYN_22_OMITTING_THE_SEMICOLON,
            SqlErrors.SYN_23_DATE_TIME_FIELD_OVERFLOW,
            SqlErrors.SYN_24_DUPLICATE_CLAUSE,
            SqlErrors.SYN_25_USING_AN_UNDEFINED_CORRELATION_NAME,
            SqlErrors.SYN_26_TOO_MANY_COLUMNS_IN_SUBQUERY,
            SqlErrors.SYN_27_CONFUSING_TABLE_NAMES_WITH_COLUMN_NAMES,
            SqlErrors.SYN_28_RESTRICTION_IN_SELECT_CLAUSE,
            SqlErrors.SYN_29_PROJECTION_IN_WHERE_CLAUSE,
            SqlErrors.SYN_30_CONFUSING_THE_ORDER_OF_KEYWORDS,
            SqlErrors.SYN_31_CONFUSING_THE_LOGIC_OF_KEYWORDS,
            SqlErrors.SYN_32_CONFUSING_THE_SYNTAX_OF_KEYWORDS,
            SqlErrors.SYN_33_OMITTING_COMMAS,
            SqlErrors.SYN_34_CURLY_SQUARE_OR_UNMATCHED_BRACKETS,
            SqlErrors.SYN_35_IS_WHERE_NOT_APPLICABLE,
            SqlErrors.SYN_36_NONSTANDARD_KEYWORDS_OR_STANDARD_KEYWORDS_IN_WRONG_CONTEXT,
            SqlErrors.SYN_37_NONSTANDARD_OPERATORS,
            SqlErrors.SYN_38_ADDITIONAL_SEMICOLON,
        ):
            error_hints.append(f"- {str(error)}")

    error_hints_str = "\n".join(error_hints)

    request = {
        "en": f"""
Hey Lens, I can't figure out how to fix the following {sql_language} query.
Could you please provide a fixed version of the query that would not cause the same error as the one below?
You don't need to give me the whole query, just the part that needs to be changed. I will apply it myself to the original query.
""",
        "it": f"""
Ciao Lens, non riesco a capire come correggere la seguente query {sql_language}.
Potresti fornirmi una versione corretta della query che non causerebbe lo stesso errore di quella di seguito?
Non è necessario che tu mi dia l'intera query, solo la parte che deve essere modificata. La applicherò io stesso alla query originale.
""",
    }

    template = {
        "en": """
To fix your query, you could try changing:
<pre class="code m">ORIGINAL QUERY PART</pre>
to:
<pre class="code m">FIXED QUERY PART</pre>
In this way, EXPLANATION OF THE FIX.
""",
        "it": """
Per correggere la tua query, potresti provare a cambiare:
<pre class="code m">PARTE ORIGINALE DELLA QUERY</pre>
in:
<pre class="code m">PARTE CORRETTA DELLA QUERY</pre>
In questo modo, SPIEGAZIONE DELLA CORREZIONE.
""",
    }

    generation_query = f"""
{get_localized(request, lang)}

{build_rules("""
- Only modify the minimal part
- Do not rewrite the full query
- Keep explanation short
""")}

{get_localized(RESPONSE_FORMAT, lang)}

{get_localized(SECTION_DETECTED_ERRORS, lang) if error_hints else ''} 
{error_hints_str if error_hints else ''}

{get_localized(SECTION_QUERY, lang).format(sql_language=sql_language)}
{query.code}

{get_localized(SECTION_ERROR, lang)}
{exception}

{get_localized(SECTION_TEMPLATE, lang)}
{get_localized(template, lang)}
"""

    return LensPrompt(
        "fix_query",
        retrieval_query,
        generation_query,
        get_localized(template, lang),
        lang,
    )


# =========================
# 7. detect_errors
# =========================


def detect_errors(
    code: str,
    errors: list[DetectedError] = [],
    lang="en",
    sql_language="PostgreSQL",
) -> LensPrompt:

    query = SQLCode(code)
    query = query.strip_comments()

    retrieval_query = f"Detect SQL issues in query: {query.code}"

    request = {
        "en": f"""
Hi Lens! I'm wondering if my query has any mistakes or errors.
Could you please review the following {sql_language} query and provide a pedagogical student-oriented explanation to let me know if there are any issues with it.
If you find any mistakes, please explain what they are but don't fix them—I just want to understand if there are any problems.
I know that the query managed to execute successfully, but I want to make sure there are no hidden issues.
You'll be provided with a list of detected errors to help you in your analysis.
Don't use the error names in your explanation, just a description of the problem for each error.
""",
        "it": f"""
Ciao Lens! Mi chiedo se la mia query abbia degli errori o degli sbagli.
Potresti esaminare la seguente query {sql_language} e farmi sapere se ci sono dei problemi?
Se trovi degli errori, spiegami quali sono ma non correggerli—voglio solo capire se ci sono dei problemi.
So che la query è stata eseguita con successo, ma voglio assicurarmi che non ci siano problemi nascosti.
Ti verrà fornita una lista di errori rilevati per aiutarti nella tua analisi.
Non usare i nomi degli errori nella tua spiegazione, solo una descrizione del problema per ogni errore.
""",
    }

    template = {
        "en": """
After reviewing your query, I found the following issues:
<ul>
<li>ERROR LIST ITEMS WITH EXPLANATIONS </li>
</ul>
""",
        "it": """
Dopo aver esaminato la tua query, ho trovato i seguenti problemi:
<ul>
<li>ELEMENTI DELLA LISTA DEGLI ERRORI CON SPIEGAZIONI</li>
</ul>
""",
    }

    errors_str = "\n".join([f"- {str(error)}" for error in errors])

    generation_query = f"""
{get_localized(request, lang)}

{build_rules("""
- Do not repeat the query
- Do not repeat the detected errors list
- Only output the final list
- Do not output section headers
""")}

{get_localized(RESPONSE_FORMAT, lang)}

{get_localized(SECTION_DETECTED_ERRORS, lang)}
{errors_str}
{get_localized(SECTION_QUERY, lang).format(sql_language=sql_language)}
{query.code}

{get_localized(SECTION_TEMPLATE, lang)}
{get_localized(template, lang)}
"""

    return LensPrompt(
        "detect_errors",
        retrieval_query,
        generation_query,
        get_localized(template, lang),
        lang,
    )

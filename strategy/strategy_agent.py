import os
import json

from dotenv import load_dotenv
from groq import Groq

from RAG.rag_agent import search, documents, index
from data.data_agent import analyze_lap_data


# =========================================================
# Configuration
# =========================================================

load_dotenv()

MODEL = "openai/gpt-oss-120b"

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


# =========================================================
# F1 Races
# =========================================================

RACES = [
    "Bahrain",
    "Saudi Arabia",
    "Australia",
    "Japan",
    "China",
    "Miami",
    "Emilia Romagna",
    "Monaco",
    "Canada",
    "Spain",
    "Austria",
    "Great Britain",
    "Hungary",
    "Belgium",
    "Netherlands",
    "Italy",
    "Azerbaijan",
    "Singapore",
    "United States",
    "Mexico",
    "São Paulo",
    "Las Vegas",
    "Qatar",
    "Abu Dhabi",
]


# =========================================================
# Detect Race
# =========================================================

def detect_race(query):
    """
    Detect the race mentioned in the user query.
    """

    query_lower = query.lower()

    for race in RACES:

        if race.lower() in query_lower:
            return race

    return None


# =========================================================
# Detect Driver
# =========================================================

def detect_driver_name(query):
    """
    Detect a driver mentioned in the query.

    Returns the standardized driver name.
    """

    driver_aliases = {

        "max verstappen": "Verstappen",
        "verstappen": "Verstappen",

        "charles leclerc": "Leclerc",
        "leclerc": "Leclerc",

        "lando norris": "Norris",
        "norris": "Norris",

        "lewis hamilton": "Hamilton",
        "hamilton": "Hamilton",

        "oscar piastri": "Piastri",
        "piastri": "Piastri",

        "fernando alonso": "Alonso",
        "alonso": "Alonso",

        "george russell": "Russell",
        "russell": "Russell",

        "carlos sainz": "Sainz",
        "sainz": "Sainz",

        "sergio perez": "Perez",
        "perez": "Perez",

        "yuki tsunoda": "Tsunoda",
        "tsunoda": "Tsunoda",

        "pierre gasly": "Gasly",
        "gasly": "Gasly",

        "esteban ocon": "Ocon",
        "ocon": "Ocon",

        "alexander albon": "Albon",
        "albon": "Albon",

        "nico hulkenberg": "Hulkenberg",
        "hulkenberg": "Hulkenberg",

        "lance stroll": "Stroll",
        "stroll": "Stroll",

        "valtteri bottas": "Bottas",
        "bottas": "Bottas",

        "daniel ricciardo": "Ricciardo",
        "ricciardo": "Ricciardo",

        "liam lawson": "Lawson",
        "lawson": "Lawson",

        "oliver bearman": "Bearman",
        "bearman": "Bearman",

        "franco colapinto": "Colapinto",
        "colapinto": "Colapinto",

        "kevin magnussen": "Magnussen",
        "magnussen": "Magnussen",

        "zhou guanyu": "Zhou",
        "zhou": "Zhou",

        "logan sargeant": "Sargeant",
        "sargeant": "Sargeant",

        "nico rosberg": "Rosberg",
        "rosberg": "Rosberg",
    }

    query_lower = query.lower()

    # Check longest names first.
    for alias in sorted(
        driver_aliases.keys(),
        key=len,
        reverse=True
    ):

        if alias in query_lower:
            return driver_aliases[alias]

    return None


# =========================================================
# Retrieve Strategy Knowledge
# =========================================================

def retrieve_strategy_knowledge(query, top_k=4):
    """
    Retrieve raw strategy knowledge from the existing
    ONNX + FAISS RAG system.

    We deliberately retrieve the raw chunks instead of
    asking the RAG Agent to generate a final answer.
    """

    results = search(
        query,
        documents,
        index,
        top_k=top_k
    )

    return results


# =========================================================
# Build Data Questions
# =========================================================

def build_data_questions(query):
    """
    Convert a strategy question into specific questions
    that the Data Agent can actually answer.
    """

    query_lower = query.lower()

    race = detect_race(query)
    driver = detect_driver_name(query)

    # A race and driver are required for the strategy
    # evidence queries we currently support.
    if not race or not driver:
        return []

    questions = []

    # -----------------------------------------------------
    # Explicit measurable questions
    # -----------------------------------------------------

    measurable_keywords = [
        "lap time",
        "fastest lap",
        "average lap",
        "lap count",
        "tyre age",
        "tire age",
        "position",
        "positions",
        "degradation",
        "tyre usage",
        "tire usage",
        "tyre performance",
        "tire performance",
    ]

    if any(
        keyword in query_lower
        for keyword in measurable_keywords
    ):

        questions.append(query)

    # -----------------------------------------------------
    # Undercut / overcut evidence
    # -----------------------------------------------------

    if any(
        keyword in query_lower
        for keyword in [
            "undercut",
            "overcut",
        ]
    ):

        questions.append(
            f"How did {driver}'s tyre degradation look at {race}?"
        )

        questions.append(
            f"What was {driver}'s tyre performance at {race}?"
        )

        questions.append(
            f"What was {driver}'s average lap time at {race}?"
        )

    # -----------------------------------------------------
    # Tyre-related strategy questions
    # -----------------------------------------------------

    elif any(
        keyword in query_lower
        for keyword in [
            "tyre",
            "tire",
            "degradation",
        ]
    ):

        questions.append(
            f"How did {driver}'s tyre degradation look at {race}?"
        )

        questions.append(
            f"What was {driver}'s tyre performance at {race}?"
        )

    # -----------------------------------------------------
    # Pit-related questions
    # -----------------------------------------------------

    elif any(
        keyword in query_lower
        for keyword in [
            "pit",
            "pit stop",
            "pitstop",
        ]
    ):

        questions.append(
            f"What was {driver}'s tyre performance at {race}?"
        )

        questions.append(
            f"How did {driver}'s tyre degradation look at {race}?"
        )

    # -----------------------------------------------------
    # General strategy questions
    # -----------------------------------------------------

    else:

        questions.append(
            f"What was {driver}'s average lap time at {race}?"
        )

        questions.append(
            f"How did {driver}'s tyre degradation look at {race}?"
        )

        questions.append(
            f"What was {driver}'s tyre performance at {race}?"
        )

    # -----------------------------------------------------
    # Remove duplicates
    # -----------------------------------------------------

    unique_questions = []

    for question in questions:

        if question not in unique_questions:
            unique_questions.append(question)

    return unique_questions


# =========================================================
# Retrieve Race Data
# =========================================================

def retrieve_race_data(query):
    """
    Ask the Data Agent targeted questions derived from
    the strategy question.
    """

    data_questions = build_data_questions(query)

    if not data_questions:

        return {
            "status": "insufficient_context",
            "message": (
                "The strategy question does not contain enough "
                "race and driver information to request specific "
                "race-data evidence."
            ),
            "analyses": []
        }

    analyses = []

    for data_question in data_questions:

        try:

            result = analyze_lap_data(data_question)

            analyses.append(
                {
                    "question": data_question,
                    "result": result
                }
            )

        except Exception as e:

            analyses.append(
                {
                    "question": data_question,
                    "result": {
                        "status": "error",
                        "message": str(e)
                    }
                }
            )

    return {
        "status": "success",
        "analyses": analyses
    }


# =========================================================
# Format RAG Context
# =========================================================

def format_rag_context(results):
    """
    Convert retrieved RAG chunks into readable evidence
    for the Strategy Agent.
    """

    if not results:

        return (
            "No relevant strategy knowledge was retrieved."
        )

    context = []

    for i, item in enumerate(results, start=1):

        context.append(
            f"""
SOURCE {i}

File:
{item.get("source", "unknown")}

Category:
{item.get("category", "unknown")}

Similarity:
{item.get("score", 0):.4f}

Content:
{item.get("text", "")}
"""
        )

    return "\n".join(context)


# =========================================================
# Format Data Context
# =========================================================

def format_data_context(data_result):
    """
    Convert Data Agent results into readable evidence.
    """

    if not data_result:

        return "No race-data analysis was produced."

    if data_result.get("status") != "success":

        return json.dumps(
            data_result,
            indent=2,
            default=str
        )

    formatted = []

    for analysis in data_result.get(
        "analyses",
        []
    ):

        formatted.append(
            f"""
DATA QUESTION:
{analysis["question"]}

DATA AGENT RESULT:
{json.dumps(
    analysis["result"],
    indent=2,
    default=str
)}
"""
        )

    if not formatted:

        return (
            "No usable race-data analyses were produced."
        )

    return "\n".join(formatted)


# =========================================================
# Strategy Agent
# =========================================================

def analyze_strategy(query):
    """
    Main Strategy Agent.

    Combines:
    - RAG strategy knowledge
    - Data Agent evidence
    - Groq reasoning
    """

    # -----------------------------------------------------
    # Detect entities
    # -----------------------------------------------------

    race = detect_race(query)
    driver = detect_driver_name(query)

    # -----------------------------------------------------
    # Retrieve strategy knowledge
    # -----------------------------------------------------

    rag_results = retrieve_strategy_knowledge(
        query
    )

    # -----------------------------------------------------
    # Retrieve targeted race evidence
    # -----------------------------------------------------

    data_result = retrieve_race_data(
        query
    )

    # -----------------------------------------------------
    # Format evidence
    # -----------------------------------------------------

    rag_context = format_rag_context(
        rag_results
    )

    data_context = format_data_context(
        data_result
    )

    # -----------------------------------------------------
    # Strategy Agent instructions
    # -----------------------------------------------------

    system_prompt = """
You are the Strategy Agent of an F1 Strategy Copilot.

Your job is to analyze Formula 1 race-strategy questions by
combining:

1. Strategy knowledge retrieved from the RAG knowledge base.
2. Numerical race evidence retrieved by the Data Agent.

You must NOT invent:

- lap times
- tyre degradation
- tyre age
- pit stops
- positions
- gaps
- traffic conditions
- weather
- Safety Car events
- driver actions

Only use facts contained in the provided evidence.

Separate your reasoning into:

1. FACTS
   Facts directly supported by the race data.

2. STRATEGY PRINCIPLES
   Principles supported by the RAG knowledge base.

3. STRATEGIC INTERPRETATION
   Reasoned conclusions connecting the facts to the
   strategy principles.

If the available evidence is insufficient, explicitly say so.

Do not turn general F1 knowledge into a claimed race fact.

For example:

If the RAG knowledge says an undercut can work when tyre
degradation is high, that does NOT prove that an undercut
would have worked in a particular race unless the race data
supports the necessary conditions.

Your final answer should explain:

- what the evidence shows,
- which strategic factors matter,
- how the evidence relates to the strategy,
- what conclusion can reasonably be drawn,
- and what cannot be determined from the available data.

Be concise but technically useful.
"""

    # -----------------------------------------------------
    # User prompt
    # -----------------------------------------------------

    user_prompt = f"""
USER QUESTION:

{query}


RACE DETECTED:

{race if race else "Not specified"}


DRIVER DETECTED:

{driver if driver else "Not specified"}


==================================================
STRATEGY KNOWLEDGE FROM RAG
==================================================

{rag_context}


==================================================
RACE DATA FROM DATA AGENT
==================================================

{data_context}


==================================================
TASK
==================================================

Analyze the user's strategy question using ONLY the
evidence provided above.

Your response must:

1. Explain the relevant strategy principles.
2. Identify the race evidence that supports the analysis.
3. Connect the evidence to the strategy.
4. Clearly distinguish facts from strategic interpretation.
5. State important limitations if evidence is missing.

Do not invent missing race information.
"""

    # -----------------------------------------------------
    # Structured Groq response
    # -----------------------------------------------------

    response = client.chat.completions.create(

        model=MODEL,

        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],

        response_format={
            "type": "json_schema",

            "json_schema": {

                "name": "strategy_analysis",

                "strict": True,

                "schema": {

                    "type": "object",

                    "properties": {

                        "answer": {
                            "type": "string"
                        },

                        "evidence_used": {

                            "type": "array",

                            "items": {
                                "type": "string"
                            }
                        },

                        "strategy_factors": {

                            "type": "array",

                            "items": {
                                "type": "string"
                            }
                        },

                        "limitations": {

                            "type": "array",

                            "items": {
                                "type": "string"
                            }
                        }
                    },

                    "required": [
                        "answer",
                        "evidence_used",
                        "strategy_factors",
                        "limitations"
                    ],

                    "additionalProperties": False
                }
            }
        }
    )

    # -----------------------------------------------------
    # Parse structured response
    # -----------------------------------------------------

    result = json.loads(
        response.choices[0].message.content
    )

    # -----------------------------------------------------
    # Final Strategy Agent result
    # -----------------------------------------------------

    return {

        "race": race,

        "driver": driver,

        "answer": result["answer"],

        "evidence_used": result[
            "evidence_used"
        ],

        "strategy_factors": result[
            "strategy_factors"
        ],

        "limitations": result[
            "limitations"
        ],

        # RAG source filenames
        "rag_sources": [
            item.get(
                "source",
                "unknown"
            )
            for item in rag_results
        ],

        # Raw RAG evidence
        # Used by the Reviewer Agent
        "rag_evidence": rag_context,

        # Complete Data Agent result
        "data_result": data_result,

        # Formatted Data Agent evidence
        # Used by the Reviewer Agent
        "data_evidence": data_context
    }


# =========================================================
# Local Test
# =========================================================

if __name__ == "__main__":

    question = (
        "Would an undercut have been useful for "
        "Verstappen at Monaco?"
    )

    result = analyze_strategy(
        question
    )

    print("\n")
    print("=" * 60)
    print("STRATEGY RESULT")
    print("=" * 60)

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False
        )
    )
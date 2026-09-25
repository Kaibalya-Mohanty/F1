import os
import json

from dotenv import load_dotenv
from groq import Groq

from graph.state import F1State

from RAG.rag_agent import answer_question
from data.data_agent import analyze_lap_data
from strategy.strategy_agent import analyze_strategy
from driver_comparison.driver_comparison_agent import compare_drivers

from reviewer.reviewer_agent import (
    review_strategy,
    revise_strategy
)


# =========================================================
# Configuration
# =========================================================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY is not configured in the .env file."
    )

client = Groq(api_key=api_key)

MODEL = "openai/gpt-oss-120b"

MAX_REVISIONS = 2


# =========================================================
# ROUTER NODE
# =========================================================

def router_node(state: F1State) -> dict:

    question_with_race = state["question_with_race"]

    router_prompt = """
You are the Router Agent for an F1 Strategy Copilot.

Your job is to determine which specialized agent should
handle the user's question.

Available intents:

1. RACE_KNOWLEDGE

Use this for general Formula 1 knowledge and strategy
concepts that can be answered from the knowledge base.

Examples:
- What is an undercut?
- Why is tyre degradation important?
- How does an overcut work?
- What happens during a Safety Car?


2. DATA_ANALYSIS

Use this when the question asks for a calculation,
measurement, statistic, or specific result from the
available race dataset.

Examples:
- What is the fastest lap?
- What is the average lap time?
- How many laps did Max Verstappen complete?
- What was Lando Norris's average lap time?
- Which driver was fastest in the dataset?

IMPORTANT:

If a question asks for a specific race-data value such as
fastest lap, average lap time, lap count, tyre age,
position, or another measurable value, classify it as
DATA_ANALYSIS.

Do NOT classify a simple numerical lookup as
DRIVER_COMPARISON.


3. STRATEGY_ANALYSIS

Use this when the user asks for strategic analysis or wants
to evaluate a race strategy using race data and F1 knowledge.

Examples:
- Should the driver have pitted earlier?
- Was an undercut the better strategy?
- Why did this strategy work?
- Would a different tyre strategy have been better?
- Would Verstappen have benefited from pitting earlier?


4. DRIVER_COMPARISON

Use this when the user explicitly asks to compare two or
more drivers using the race dataset.

Examples:
- Compare Verstappen and Norris.
- Compare Verstappen and Norris at Monaco.
- Who had the better lap times?
- Compare their race pace.
- How did Verstappen and Norris perform against each other?
- Compare their tyre usage.
- Compare their average lap times.


5. GENERAL

Use this for questions that do not belong to the above
categories.


IMPORTANT ROUTING RULES:

If the question can be answered by analyzing one specific
numerical race-data value, prefer DATA_ANALYSIS.

If the question requires conceptual F1 knowledge from the
knowledge base, use RACE_KNOWLEDGE.

If the question requires strategic reasoning using race data
and F1 knowledge, use STRATEGY_ANALYSIS.

If the user explicitly wants two or more drivers compared,
use DRIVER_COMPARISON.

Return only the required structured JSON output.
"""

    response = client.chat.completions.create(

        model=MODEL,

        messages=[
            {
                "role": "system",
                "content": router_prompt
            },
            {
                "role": "user",
                "content": question_with_race
            }
        ],

        response_format={
            "type": "json_schema",

            "json_schema": {

                "name": "f1_router",

                "strict": True,

                "schema": {

                    "type": "object",

                    "properties": {

                        "intent": {
                            "type": "string",
                            "enum": [
                                "RACE_KNOWLEDGE",
                                "DATA_ANALYSIS",
                                "STRATEGY_ANALYSIS",
                                "DRIVER_COMPARISON",
                                "GENERAL"
                            ]
                        },

                        "agents": {

                            "type": "array",

                            "items": {
                                "type": "string",

                                "enum": [
                                    "RAG_AGENT",
                                    "DATA_AGENT",
                                    "STRATEGY_AGENT",
                                    "DRIVER_COMPARISON_AGENT"
                                ]
                            }
                        },

                        "reason": {
                            "type": "string"
                        }
                    },

                    "required": [
                        "intent",
                        "agents",
                        "reason"
                    ],

                    "additionalProperties": False
                }
            }
        },

        temperature=0.2
    )

    routing_result = json.loads(
        response.choices[0].message.content
    )

    return {
        "intent": routing_result["intent"],
        "agents": routing_result["agents"],
        "routing_reason": routing_result["reason"]
    }


# =========================================================
# RAG NODE
# =========================================================

def rag_node(state: F1State) -> dict:

    result = answer_question(
        state["question"]
    )

    final_answer = result.get(
        "answer",
        ""
    )

    return {
        "rag_result": result,
        "final_answer": final_answer
    }


# =========================================================
# DATA NODE
# =========================================================

def data_node(state: F1State) -> dict:

    result = analyze_lap_data(
        state["question_with_race"]
    )

    return {
        "data_result": result,
        "final_answer": str(result)
    }


# =========================================================
# STRATEGY NODE
# =========================================================

def strategy_node(state: F1State) -> dict:

    result = analyze_strategy(
        state["question_with_race"]
    )

    return {
        "strategy_result": result
    }


# =========================================================
# DRIVER COMPARISON NODE
# =========================================================

def driver_comparison_node(state: F1State) -> dict:

    result = compare_drivers(
        state["question_with_race"]
    )

    comparison_text = result.get(
        "comparison",
        ""
    )

    return {
        "comparison_result": result,
        "final_answer": comparison_text
    }


# =========================================================
# REVIEWER NODE
# =========================================================

def reviewer_node(state: F1State) -> dict:

    strategy_result = state.get(
        "strategy_result",
        {}
    )

    review_result = review_strategy(

        user_query=state["question_with_race"],

        strategy_result=strategy_result,

        rag_evidence=[
            strategy_result.get(
                "rag_evidence",
                ""
            )
        ],

        data_evidence=strategy_result.get(
            "data_evidence",
            ""
        )
    )

    return {
        "review_result": review_result
    }


# =========================================================
# REVISION NODE
# =========================================================

def revision_node(state: F1State) -> dict:

    strategy_result = state.get(
        "strategy_result",
        {}
    )

    review_result = state.get(
        "review_result",
        {}
    )

    revised_result = revise_strategy(

        user_query=state["question_with_race"],

        strategy_result=strategy_result,

        review_result=review_result
    )

    merged_strategy = {
        **strategy_result,
        **revised_result
    }

    current_revision_count = state.get(
        "revision_count",
        0
    )

    return {
        "strategy_result": merged_strategy,
        "revision_count": current_revision_count + 1
    }


# =========================================================
# FINAL STRATEGY NODE
# =========================================================

def strategy_final_node(state: F1State) -> dict:

    strategy_result = state.get(
        "strategy_result",
        {}
    )

    answer = strategy_result.get(
        "answer",
        ""
    )

    return {
        "final_answer": answer
    }


# =========================================================
# GENERAL NODE
# =========================================================

def general_node(state: F1State) -> dict:

    return {
        "final_answer": (
            "I could not determine which specialized "
            "F1 agent should handle this question."
        )
    }
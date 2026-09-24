import os
import json

from dotenv import load_dotenv
from groq import Groq


# =========================================================
# Configuration
# =========================================================

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")

client = Groq(api_key=API_KEY)

MODEL = "openai/gpt-oss-120b"


# =========================================================
# Reviewer Agent
# =========================================================

def review_strategy(
    user_query,
    strategy_result,
    rag_evidence,
    data_evidence
):
    """
    Review the Strategy Agent's answer and determine whether
    its claims are grounded in the evidence provided.
    """

    answer = strategy_result.get("answer", "")

    evidence_used = strategy_result.get(
        "evidence_used",
        []
    )

    strategy_factors = strategy_result.get(
        "strategy_factors",
        []
    )

    limitations = strategy_result.get(
        "limitations",
        []
    )

    # -----------------------------------------------------
    # System prompt
    # -----------------------------------------------------

    system_prompt = """
You are a strict evidence-reviewing agent for an F1 Strategy
Analysis system.

Your job is NOT to create a new strategy.

Your job is to check whether the Strategy Agent's answer is
supported by the evidence it was given.

You must distinguish between:

1. Directly supported factual claims
2. Reasonable strategic interpretations
3. Unsupported or invented claims

Rules:

- Do not introduce new facts.
- Do not use outside knowledge.
- Do not assume missing race data.
- A strategic interpretation is acceptable when it logically
  follows from the provided evidence.
- If a claim cannot be supported by the provided evidence,
  identify it.
- Missing information should be reported as a limitation,
  not guessed.
- Be strict about numerical claims.
- If the answer is properly grounded, return PASS.
- If important claims are unsupported, return FAIL.

Return only the requested JSON structure.
"""

    # -----------------------------------------------------
    # Evidence formatting
    # -----------------------------------------------------

    rag_text = "\n\n".join(
        str(item)
        for item in rag_evidence
    )

    data_text = str(data_evidence)

    user_prompt = f"""
USER QUERY:
{user_query}


STRATEGY AGENT ANSWER:
{answer}


EVIDENCE USED BY STRATEGY AGENT:
{json.dumps(evidence_used, indent=2)}


STRATEGY FACTORS:
{json.dumps(strategy_factors, indent=2)}


STRATEGY LIMITATIONS:
{json.dumps(limitations, indent=2)}


RAG EVIDENCE:
{rag_text}


RACE DATA EVIDENCE:
{data_text}


Review the Strategy Agent answer.

Determine:

- whether the answer is grounded
- which claims are directly supported
- which claims are unsupported
- whether important evidence is missing
- what the Strategy Agent should change if necessary
"""

    # -----------------------------------------------------
    # Structured output schema
    # -----------------------------------------------------

    schema = {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["PASS", "FAIL"]
            },
            "grounded": {
                "type": "boolean"
            },
            "supported_claims": {
                "type": "array",
                "items": {
                    "type": "string"
                }
            },
            "unsupported_claims": {
                "type": "array",
                "items": {
                    "type": "string"
                }
            },
            "missing_evidence": {
                "type": "array",
                "items": {
                    "type": "string"
                }
            },
            "feedback": {
                "type": "string"
            }
        },
        "required": [
            "verdict",
            "grounded",
            "supported_claims",
            "unsupported_claims",
            "missing_evidence",
            "feedback"
        ],
        "additionalProperties": False
    }

    # -----------------------------------------------------
    # Groq call
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
        temperature=0,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "strategy_review",
                "strict": True,
                "schema": schema
            }
        }
    )

    result = json.loads(
        response.choices[0].message.content
    )
    # ---------------------------------------------------------
    # Deterministic verdict enforcement
    # ---------------------------------------------------------

    if result.get("unsupported_claims"):
        result["verdict"] = "FAIL"
        result["grounded"] = False


    else:
        result["verdict"] = "PASS"
        result["grounded"] = True

    return result


# =========================================================
# Strategy Revision Agent
# =========================================================

def revise_strategy(
    user_query,
    strategy_result,
    review_result
):
    """
    Revise the Strategy Agent's answer using the Reviewer's
    feedback while staying strictly within the available evidence.
    """

    answer = strategy_result.get(
        "answer",
        ""
    )

    evidence_used = strategy_result.get(
        "evidence_used",
        []
    )

    strategy_factors = strategy_result.get(
        "strategy_factors",
        []
    )

    limitations = strategy_result.get(
        "limitations",
        []
    )

    rag_evidence = strategy_result.get(
        "rag_evidence",
        ""
    )

    data_evidence = strategy_result.get(
        "data_evidence",
        ""
    )

    # -----------------------------------------------------
    # System prompt
    # -----------------------------------------------------

    system_prompt = """
You are a Strategy Revision Agent for an F1 Strategy Analysis
system.

Your job is to revise a Strategy Agent's answer after it has
been reviewed by a strict evidence-reviewing agent.

Rules:

- Do not introduce new facts.
- Do not use outside knowledge.
- Do not invent race data.
- Do not invent lap times, tyre degradation, tyre age,
  gaps, pit-stop timing, traffic, weather, Safety Car
  events, or driver actions.
- Correct every unsupported claim identified by the Reviewer.
- Use only the RAG evidence and race-data evidence provided.
- Preserve claims that are supported by the evidence.
- If the evidence is insufficient, explicitly acknowledge
  the limitation instead of guessing.
- Do not make a definitive conclusion when the evidence
  does not support one.
- Keep the revised answer clear, concise, and technically useful.

The revised answer must be more evidence-grounded than the
original answer.

Return only the requested JSON structure.
"""

    # -----------------------------------------------------
    # User prompt
    # -----------------------------------------------------

    user_prompt = f"""
USER QUESTION:
{user_query}


ORIGINAL STRATEGY ANSWER:
{answer}


REVIEWER RESULT:
{json.dumps(review_result, indent=2)}


ORIGINAL EVIDENCE USED:
{json.dumps(evidence_used, indent=2)}


STRATEGY FACTORS:
{json.dumps(strategy_factors, indent=2)}


ORIGINAL LIMITATIONS:
{json.dumps(limitations, indent=2)}


RAG EVIDENCE:
{rag_evidence}


RACE DATA EVIDENCE:
{data_evidence}


TASK:

Revise the Strategy Agent answer according to the Reviewer's
feedback.

Remove or correct unsupported claims.

Do not introduce information that is not contained in the
provided evidence.

If the available evidence cannot support a definitive conclusion,
say so clearly.

Return:

1. The corrected answer
2. The evidence used
3. The strategy factors
4. The limitations
"""

    # -----------------------------------------------------
    # Structured output schema
    # -----------------------------------------------------

    schema = {
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

    # -----------------------------------------------------
    # Groq call
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
        temperature=0,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "strategy_revision",
                "strict": True,
                "schema": schema
            }
        }
    )

    result = json.loads(
        response.choices[0].message.content
    )

    return result


# =========================================================
# Independent Revision Test
# =========================================================

if __name__ == "__main__":

    # -----------------------------------------------------
    # Deliberately incorrect Strategy Agent answer
    # -----------------------------------------------------

    test_strategy = {
        "answer": (
            "Verstappen's average lap time at Monaco was exactly "
            "60.000 seconds, so an undercut would definitely have "
            "been successful."
        ),

        "evidence_used": [
            "Hard tyre average lap time: 75.798 seconds",
            "Medium tyre average lap time: 80.045 seconds"
        ],

        "strategy_factors": [
            "Tyre degradation",
            "Fresh tyre performance"
        ],

        "limitations": [],

        "rag_evidence": [
            "An undercut can be useful when tyre degradation "
            "is significant."
        ],

        "data_evidence": """
Verstappen Monaco average lap time: 78.592 seconds.
Hard tyre average: 75.798 seconds.
Medium tyre average: 80.045 seconds.
"""
    }

    question = (
        "Would an undercut have been useful for "
        "Verstappen at Monaco?"
    )

    # -----------------------------------------------------
    # Step 1 — Review the deliberately bad answer
    # -----------------------------------------------------

    print("\n")
    print("=" * 60)
    print("STEP 1: REVIEWING STRATEGY")
    print("=" * 60)

    review_result = review_strategy(
        question,
        test_strategy,
        test_strategy["rag_evidence"],
        test_strategy["data_evidence"]
    )

    print(
        json.dumps(
            review_result,
            indent=2,
            ensure_ascii=False
        )
    )

    # -----------------------------------------------------
    # Step 2 — Revise the bad answer
    # -----------------------------------------------------

    print("\n")
    print("=" * 60)
    print("STEP 2: REVISING STRATEGY")
    print("=" * 60)

    revised_result = revise_strategy(
        question,
        test_strategy,
        review_result
    )

    print(
        json.dumps(
            revised_result,
            indent=2,
            ensure_ascii=False
        )
    )

    # -----------------------------------------------------
    # Step 3 — Review the revised answer again
    # -----------------------------------------------------

    print("\n")
    print("=" * 60)
    print("STEP 3: RE-REVIEWING REVISED STRATEGY")
    print("=" * 60)

    revised_strategy_for_review = {
        **test_strategy,
        **revised_result
    }

    second_review = review_strategy(
        question,
        revised_strategy_for_review,
        test_strategy["rag_evidence"],
        test_strategy["data_evidence"]
    )

    print(
        json.dumps(
            second_review,
            indent=2,
            ensure_ascii=False
        )
    )
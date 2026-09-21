import os
import json

import streamlit as st
from dotenv import load_dotenv
from groq import Groq


# -----------------------------
# Configuration
# -----------------------------

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("GROQ_API_KEY is not configured in the .env file.")
    st.stop()

client = Groq(api_key=api_key)

MODEL = "openai/gpt-oss-120b"


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(
    page_title="F1 Strategy Copilot",
    page_icon="🏎️",
    layout="wide"
)

st.title("🏎️ F1 Strategy Copilot")
st.write("Multi-Agent Generative AI System for Formula 1 Strategy Analysis")


question = st.text_area(
    "Ask your F1 strategy question:",
    placeholder="Example: Why is an undercut useful in Formula 1?"
)


# -----------------------------
# Router Agent
# -----------------------------

def route_question(user_question):

    router_prompt = """
You are the routing agent of an F1 Strategy Analysis system.

Your job is ONLY to classify the user's question and determine which
specialized agents will be needed later.

Do NOT answer the user's question.

Classify the question into exactly one of these intents:

- RACE_KNOWLEDGE
  General Formula 1 concepts, rules, tyres, strategy terminology, etc.

- DATA_ANALYSIS
  Questions requiring calculations or analysis of structured race data,
  such as lap times, tyre stints, pit stops, positions, or weather.

- STRATEGY_ANALYSIS
  Questions asking why a strategic decision happened or whether a strategy
  was effective.

- DRIVER_COMPARISON
  Questions comparing drivers using race performance or strategy data.

- GENERAL
  Questions that do not fit the above categories.

Available agents:

- RAG_AGENT
  Retrieves relevant F1 knowledge and documents.

- DATA_AGENT
  Performs calculations and analysis on structured race data.

- STRATEGY_AGENT
  Combines retrieved knowledge and data to analyze race strategy.

Return ONLY the structured JSON requested by the schema.
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
                "content": user_question
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
                                    "STRATEGY_AGENT"
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

    return json.loads(response.choices[0].message.content)


# -----------------------------
# Run
# -----------------------------

if st.button("🔍 Analyze Strategy"):

    if not question.strip():
        st.warning("Please enter an F1 question.")
        st.stop()

    with st.spinner("Routing your question..."):

        routing_result = route_question(question)

    st.subheader("🧭 Router Decision")

    st.json(routing_result)
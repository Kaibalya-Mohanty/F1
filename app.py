import os
import json

import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from RAG.rag_agent import answer_question

from data.data_agent import analyze_lap_data

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
You are the Router Agent for an F1 Strategy Copilot.

Your job is to determine which specialized agent should handle the user's question.

Available intents:

1. RACE_KNOWLEDGE
Use this for general Formula 1 knowledge and strategy concepts that can be answered from the knowledge base.

Examples:
- What is an undercut?
- Why is tyre degradation important?
- How does an overcut work?
- What happens during a Safety Car?

2. DATA_ANALYSIS
Use this when the question asks for a calculation, measurement,
comparison, statistic, or specific result from the available race dataset.

Examples:
- What is the fastest lap?
- What is the average lap time?
- How many laps did Max Verstappen complete?
- Which driver had the fastest lap?
- What was Lando Norris's average lap time?
- Which driver was fastest in the dataset?

IMPORTANT:
If a question asks for a specific race-data value such as
fastest lap, average lap time, lap count, tyre age, position,
or other measurable information, classify it as DATA_ANALYSIS,
not RACE_KNOWLEDGE.

3. STRATEGY_ANALYSIS
Use this when the user asks for strategic analysis or wants to
evaluate a race strategy using race data and F1 knowledge.

Examples:
- Should the driver have pitted earlier?
- Was an undercut the better strategy?
- Why did this strategy work?
- Would a different tyre strategy have been better?

4. DRIVER_COMPARISON
Use this when the user explicitly asks to compare drivers.

Examples:
- Compare Verstappen and Norris.
- Who had the better lap times?
- Compare their race pace.

5. GENERAL
Use this for questions that do not belong to the above categories.

IMPORTANT ROUTING RULE:

If the question can be answered by analyzing the numerical race
dataset, prefer DATA_ANALYSIS.

If the question requires conceptual F1 knowledge from the knowledge
base, use RACE_KNOWLEDGE.

If the question requires both race data and strategic reasoning,
use STRATEGY_ANALYSIS.

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

    # -----------------------------
    # Step 1: Route the question
    # -----------------------------

    with st.spinner("Routing your question..."):

        routing_result = route_question(question)

    st.subheader("🧭 Router Decision")

    st.json(routing_result)


    # -----------------------------
    # Step 2: Execute selected agent
    # -----------------------------

    if routing_result["intent"] == "RACE_KNOWLEDGE":
        with st.spinner("Retrieving F1 knowledge..."):
            rag_result = answer_question(question)
        
        st.subheader("🤖 RAG Agent Answer")
        st.write(rag_result["answer"])
        st.subheader("📚 Sources")

        for source in rag_result["sources"]:
            st.write(f"- {source}")

    elif routing_result["intent"] == "DATA_ANALYSIS":
        with st.spinner("Analyzing race data..."):
            data_result = analyze_lap_data(question)
        
        st.subheader("🔍 Data Agent Result")
        st.json(data_result)
    
    else:
        st.info(f"The {routing_result['intent']} agent is not implemented yet.")

    

    


   
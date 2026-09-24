import os
import json

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from RAG.rag_agent import answer_question
from data.data_agent import analyze_lap_data
from strategy.strategy_agent import analyze_strategy

from reviewer.reviewer_agent import (
    review_strategy,
    revise_strategy
)

from driver_comparison.driver_comparison_agent import compare_drivers


# =========================================================
# Configuration
# =========================================================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error(
        "GROQ_API_KEY is not configured in the .env file."
    )
    st.stop()

client = Groq(
    api_key=api_key
)

MODEL = "openai/gpt-oss-120b"

# Maximum number of automatic strategy revisions
MAX_REVISIONS = 2


# =========================================================
# Streamlit UI
# =========================================================

st.set_page_config(
    page_title="F1 Strategy Copilot",
    page_icon="🏎️",
    layout="wide"
)

st.title("🏎️ F1 Strategy Copilot")

st.write(
    "Multi-Agent Generative AI System for Formula 1 Strategy Analysis"
)


# =========================================================
# Race Selection
# =========================================================

try:

    race_df = pd.read_csv(
        "data/laps.csv"
    )

    race_values = (
        race_df["race"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    race_options = []

    for race in race_values:

        race = race.strip()

        # If the dataset already contains the year,
        # keep it unchanged.
        if "2024" in race:

            display_race = race

        else:

            display_race = f"{race} 2024"

        if display_race not in race_options:

            race_options.append(
                display_race
            )

    race_options = sorted(
        race_options
    )

except Exception:

    race_options = [
        "Abu Dhabi 2024",
        "Australia 2024",
        "Austria 2024",
        "Azerbaijan 2024",
        "Bahrain 2024",
        "Belgium 2024",
        "Canada 2024",
        "China 2024",
        "Emilia Romagna 2024",
        "Great Britain 2024",
        "Hungary 2024",
        "Italy 2024",
        "Japan 2024",
        "Las Vegas 2024",
        "Mexico 2024",
        "Miami 2024",
        "Monaco 2024",
        "Netherlands 2024",
        "Qatar 2024",
        "Saudi Arabia 2024",
        "Singapore 2024",
        "Spain 2024",
        "São Paulo 2024",
        "United States 2024"
    ]


# Make Monaco 2024 the default race if available.

if "Monaco 2024" in race_options:

    default_race_index = race_options.index(
        "Monaco 2024"
    )

else:

    default_race_index = 0


selected_race = st.selectbox(
    "🏁 Select Race:",
    race_options,
    index=default_race_index
)


# =========================================================
# Question
# =========================================================

question = st.text_area(
    "Ask your F1 strategy question:",
    placeholder=(
        "Example: Would pitting Verstappen five laps "
        "earlier have improved his race?"
    )
)


# =========================================================
# Router Agent
# =========================================================

def route_question(user_question):

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

    return json.loads(
        response.choices[0].message.content
    )


# =========================================================
# Run
# =========================================================

if st.button("🔍 Analyze Strategy"):

    if not question.strip():

        st.warning(
            "Please enter an F1 question."
        )

        st.stop()


    # =====================================================
    # Add selected race to the user's question
    # =====================================================

    question_with_race = (
        f"{question.strip()}\n\n"
        f"Selected race: {selected_race}"
    )


    # =====================================================
    # Step 1: Route the question
    # =====================================================

    with st.spinner(
        "Routing your question..."
    ):

        routing_result = route_question(
            question_with_race
        )


    st.subheader(
        "🧭 Router Decision"
    )

    st.json(
        routing_result
    )


    # =====================================================
    # Step 2: Execute selected agent
    # =====================================================


    # =====================================================
    # RAG AGENT
    # =====================================================

    if routing_result["intent"] == "RACE_KNOWLEDGE":

        with st.spinner(
            "Retrieving F1 knowledge..."
        ):

            # RAG does not need race-specific context
            # for general knowledge questions.

            rag_result = answer_question(
                question
            )


        st.subheader(
            "🤖 RAG Agent Answer"
        )

        st.write(
            rag_result["answer"]
        )


        st.subheader(
            "📚 Sources"
        )

        for source in rag_result["sources"]:

            st.write(
                f"- {source}"
            )


    # =====================================================
    # DATA AGENT
    # =====================================================

    elif routing_result["intent"] == "DATA_ANALYSIS":

        with st.spinner(
            "Analyzing race data..."
        ):

            data_result = analyze_lap_data(
                question_with_race
            )


        st.subheader(
            "🔍 Data Agent Result"
        )

        st.json(
            data_result
        )


    # =====================================================
    # STRATEGY AGENT
    # =====================================================

    elif routing_result["intent"] == "STRATEGY_ANALYSIS":


        # -------------------------------------------------
        # Step 1: Strategy Agent
        # -------------------------------------------------

        with st.spinner(
            "Analyzing race strategy..."
        ):

            strategy_result = analyze_strategy(
                question_with_race
            )


        # -------------------------------------------------
        # Step 2: Initial Reviewer
        # -------------------------------------------------

        with st.spinner(
            "Reviewing strategy..."
        ):

            review_result = review_strategy(

                user_query=question_with_race,

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


        # -------------------------------------------------
        # Step 3: Revision Loop
        # -------------------------------------------------

        revision_count = 0


        while (

            review_result.get(
                "verdict"
            ) == "FAIL"

            and

            revision_count < MAX_REVISIONS

        ):

            revision_count += 1


            # ---------------------------------------------
            # Revise Strategy
            # ---------------------------------------------

            with st.spinner(
                f"Revising strategy "
                f"(attempt {revision_count})..."
            ):

                revised_result = revise_strategy(

                    user_query=question_with_race,

                    strategy_result=strategy_result,

                    review_result=review_result

                )


            # ---------------------------------------------
            # Merge revised strategy
            # ---------------------------------------------

            strategy_result = {

                **strategy_result,

                **revised_result

            }


            # ---------------------------------------------
            # Re-review revised strategy
            # ---------------------------------------------

            with st.spinner(
                "Re-reviewing revised strategy..."
            ):

                review_result = review_strategy(

                    user_query=question_with_race,

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


        # -------------------------------------------------
        # Step 4: Strategy Answer
        # -------------------------------------------------

        st.subheader(
            "🏎️ Strategy Agent Analysis"
        )


        if strategy_result.get(
            "answer"
        ):

            st.write(
                strategy_result["answer"]
            )

        else:

            st.warning(
                "The Strategy Agent did not return "
                "a final answer."
            )


        # -------------------------------------------------
        # Display selected race
        # -------------------------------------------------

        st.info(
            f"🏁 Race analyzed: **{selected_race}**"
        )


        # -------------------------------------------------
        # Revision Status
        # -------------------------------------------------

        if revision_count > 0:

            if review_result.get(
                "verdict"
            ) == "PASS":

                st.success(
                    f"Strategy passed review after "
                    f"{revision_count} revision(s)."
                )

            else:

                st.warning(
                    f"The strategy remained unresolved "
                    f"after {revision_count} revision(s)."
                )


        # -------------------------------------------------
        # Evidence Used
        # -------------------------------------------------

        evidence_used = strategy_result.get(
            "evidence_used",
            []
        )


        if evidence_used:

            st.subheader(
                "📊 Evidence Used"
            )

            for evidence in evidence_used:

                st.write(
                    f"- {evidence}"
                )


        # -------------------------------------------------
        # Strategy Factors
        # -------------------------------------------------

        strategy_factors = strategy_result.get(
            "strategy_factors",
            []
        )


        if strategy_factors:

            st.subheader(
                "🧠 Strategy Factors"
            )

            for factor in strategy_factors:

                st.write(
                    f"- {factor}"
                )


        # -------------------------------------------------
        # Limitations
        # -------------------------------------------------

        limitations = strategy_result.get(
            "limitations",
            []
        )


        if limitations:

            st.subheader(
                "⚠️ Limitations"
            )

            for limitation in limitations:

                st.write(
                    f"- {limitation}"
                )


        # -------------------------------------------------
        # RAG Sources
        # -------------------------------------------------

        rag_sources = strategy_result.get(
            "rag_sources",
            []
        )


        unique_sources = list(
            dict.fromkeys(
                rag_sources
            )
        )


        if unique_sources:

            st.subheader(
                "📚 RAG Sources"
            )

            for source in unique_sources:

                st.write(
                    f"- {source}"
                )


        # -------------------------------------------------
        # Full Strategy Agent Output
        # -------------------------------------------------

        with st.expander(
            "🔧 Full Strategy Agent Output"
        ):

            st.json(
                strategy_result
            )


        # -------------------------------------------------
        # Reviewer Agent Result
        # -------------------------------------------------

        st.subheader(
            "🧪 Reviewer Agent Result"
        )

        st.json(
            review_result
        )


    # =====================================================
    # DRIVER COMPARISON
    # =====================================================

    elif routing_result["intent"] == "DRIVER_COMPARISON":


        # -------------------------------------------------
        # Step 1: Driver Comparison Agent
        # -------------------------------------------------

        with st.spinner(
            "Comparing drivers..."
        ):

            comparison_result = compare_drivers(
                question_with_race
            )


        # -------------------------------------------------
        # Step 2: Display comparison
        # -------------------------------------------------

        st.subheader(
            "🏁 Driver Comparison"
        )


        if comparison_result.get(
            "status"
        ) == "success":


            # ---------------------------------------------
            # Race
            # ---------------------------------------------

            if comparison_result.get(
                "race"
            ):

                st.write(
                    f"**Race:** "
                    f"{comparison_result['race']}"
                )


            # ---------------------------------------------
            # Driver Statistics
            # ---------------------------------------------

            drivers = comparison_result.get(
                "drivers",
                []
            )


            if drivers:

                st.subheader(
                    "📊 Driver Statistics"
                )


                for driver_data in drivers:

                    driver = driver_data.get(
                        "driver",
                        "Unknown"
                    )


                    st.markdown(
                        f"### {driver}"
                    )


                    col1, col2, col3 = st.columns(3)


                    with col1:
                        average_lap = driver_data.get("average_lap_time", 0)

                        st.metric(
                            "Average Lap",
                            f"{average_lap:.3f} s"
                        )


                    with col2:
                        fastest_lap = driver_data.get("fastest_lap", 0)

                        st.metric(
                            "Fastest Lap",
                            f"{fastest_lap:.3f} s"
                        )


                    with col3:
                        average_position = driver_data.get("average_position", 0)

                        st.metric(
                            "Average Position",
                            f"{average_position:.1f}"
                        )


                    # -------------------------------------
                    # Additional statistics
                    # -------------------------------------

                    st.write(
                        f"**Laps analyzed:** "
                        f"{driver_data.get('laps_analyzed', 0)}"
                    )


                    st.write(
                        f"**Average tyre age:** "
                        f"{driver_data.get('average_tyre_age', 0):.2f} laps"
                    )


                    # -------------------------------------
                    # Compound statistics
                    # -------------------------------------

                    compound_stats = driver_data.get(
                        "compound_stats",
                        {}
                    )


                    if compound_stats:

                        st.markdown(
                            "**Tyre Compound Statistics**"
                        )


                        for compound, stats in (
                            compound_stats.items()
                        ):

                            st.write(
                                f"**{compound}** — "
                                f"Average: "
                                f"{stats.get('average_lap_time', 0):.3f}s | "
                                f"Fastest: "
                                f"{stats.get('fastest_lap', 0):.3f}s | "
                                f"Laps: "
                                f"{stats.get('laps', 0)}"
                            )


                    # -------------------------------------
                    # Data Quality
                    # -------------------------------------

                    data_quality = driver_data.get(
                        "data_quality",
                        {}
                    )


                    if data_quality:

                        with st.expander(
                            f"🔎 {driver} Data Quality"
                        ):

                            st.json(
                                data_quality
                            )


            # ---------------------------------------------
            # Comparison Text
            # ---------------------------------------------

            comparison_text = comparison_result.get(
                "comparison",
                ""
            )


            if comparison_text:

                st.subheader(
                    "🧠 Comparison Analysis"
                )

                st.markdown(
                    comparison_text
                )


            # ---------------------------------------------
            # Full Output
            # ---------------------------------------------

            with st.expander(
                "🔧 Full Driver Comparison Output"
            ):

                st.json(
                    comparison_result
                )


        else:

            st.error(
                "The Driver Comparison Agent could not "
                "complete the comparison."
            )

            st.json(
                comparison_result
            )


    # =====================================================
    # GENERAL / UNKNOWN
    # =====================================================

    else:

        st.info(
            f"The {routing_result['intent']} "
            "agent is not implemented yet."
        )
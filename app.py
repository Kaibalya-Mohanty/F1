import pandas as pd
import streamlit as st

from graph.workflow import run_f1_workflow
from textwrap import dedent


def render_html(html):
    """Render custom HTML without indentation that Markdown reads as code."""
    normalized_html = "\n".join(
        line.lstrip()
        for line in dedent(html).splitlines()
    ).strip()
    st.markdown(
        normalized_html,
        unsafe_allow_html=True,
    )


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="F1 Strategy Copilot",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# CUSTOM CSS
# =========================================================

render_html(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Titillium+Web:wght@400;600;700;900&display=swap');

    /* -----------------------------------------------------
       GLOBAL
    ----------------------------------------------------- */

    .stApp {
        font-family: "Titillium Web", Arial, sans-serif;
        background:
            radial-gradient(
                circle at 15% 5%,
                rgba(220, 38, 38, 0.08),
                transparent 28%
            ),
            radial-gradient(
                circle at 85% 10%,
                rgba(59, 130, 246, 0.07),
                transparent 25%
            ),
            #0b0d12;
        color: #f5f5f5;
    }

    .main .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* Hide Streamlit branding */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        background: transparent !important;
    }

    /* -----------------------------------------------------
       TYPOGRAPHY
    ----------------------------------------------------- */

    h1, h2, h3, h4, h5, h6,
    p, label, button, input, textarea, select,
    [data-testid="stMarkdownContainer"] {
        font-family: "Titillium Web", Arial, sans-serif !important;
    }

    h1, h2, h3 {
        letter-spacing: -0.025em;
    }

    .muted {
        color: #9ca3af;
    }

    /* -----------------------------------------------------
       HERO
    ----------------------------------------------------- */

    .hero {
        padding: 1.8rem 2rem;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        background:
            linear-gradient(
                135deg,
                rgba(255,255,255,0.055),
                rgba(255,255,255,0.018)
            );
        box-shadow:
            0 20px 60px rgba(0,0,0,0.25);
        margin-bottom: 1.4rem;
    }

    .hero-top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
    }

    .hero-title {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        font-size: 2.45rem;
        font-weight: 800;
        line-height: 1.1;
        margin: 0;
    }

    .f1-logo {
        height: 2.45rem;
        width: auto;
        flex: 0 0 auto;
    }

    .hero-subtitle {
        margin-top: 0.55rem;
        color: #a7adb8;
        font-size: 1rem;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.45rem 0.8rem;
        border-radius: 999px;
        background: rgba(34,197,94,0.10);
        border: 1px solid rgba(34,197,94,0.25);
        color: #86efac;
        font-size: 0.82rem;
        font-weight: 600;
        white-space: nowrap;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #22c55e;
        box-shadow: 0 0 10px rgba(34,197,94,0.8);
    }

    /* -----------------------------------------------------
       SECTION CARDS
    ----------------------------------------------------- */

    .panel {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.075);
        border-radius: 16px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }

    .panel-title {
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 0.7rem;
    }

    .panel-description {
        color: #9ca3af;
        font-size: 0.88rem;
        margin-bottom: 1rem;
    }

    /* -----------------------------------------------------
       RESULT HERO
    ----------------------------------------------------- */

    .answer-card {
        background:
            linear-gradient(
                135deg,
                rgba(30,64,175,0.20),
                rgba(17,24,39,0.55)
            );
        border: 1px solid rgba(96,165,250,0.22);
        border-radius: 18px;
        padding: 1.5rem;
        margin: 0.5rem 0 1.25rem 0;
        box-shadow: 0 15px 40px rgba(0,0,0,0.18);
    }

    .answer-label {
        color: #93c5fd;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.65rem;
    }

    .answer-text {
        font-size: 1.08rem;
        line-height: 1.75;
        color: #f3f4f6;
    }

    /* -----------------------------------------------------
       ROUTER
    ----------------------------------------------------- */

    .router-card {
        display: flex;
        flex-wrap: wrap;
        gap: 0.65rem;
        align-items: center;
        margin-bottom: 1rem;
    }

    .intent-pill {
        display: inline-block;
        padding: 0.42rem 0.75rem;
        border-radius: 999px;
        background: rgba(168,85,247,0.12);
        border: 1px solid rgba(168,85,247,0.25);
        color: #d8b4fe;
        font-weight: 700;
        font-size: 0.8rem;
    }

    .agent-pill {
        display: inline-block;
        padding: 0.42rem 0.75rem;
        border-radius: 999px;
        background: rgba(59,130,246,0.10);
        border: 1px solid rgba(59,130,246,0.22);
        color: #93c5fd;
        font-size: 0.8rem;
    }

    /* -----------------------------------------------------
       INFO CARDS
    ----------------------------------------------------- */

    .info-card {
        height: 100%;
        padding: 1rem;
        border-radius: 14px;
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.07);
    }

    .info-card-title {
        font-weight: 700;
        margin-bottom: 0.55rem;
    }

    .info-card-item {
        color: #c4c9d2;
        line-height: 1.6;
        margin-bottom: 0.35rem;
        font-size: 0.9rem;
    }

    /* -----------------------------------------------------
       REVIEW
    ----------------------------------------------------- */

    .review-pass {
        padding: 1rem 1.15rem;
        border-radius: 14px;
        background: rgba(34,197,94,0.10);
        border: 1px solid rgba(34,197,94,0.25);
        color: #86efac;
        font-weight: 700;
        margin: 0.7rem 0 1rem 0;
    }

    .review-fail {
        padding: 1rem 1.15rem;
        border-radius: 14px;
        background: rgba(239,68,68,0.10);
        border: 1px solid rgba(239,68,68,0.25);
        color: #fca5a5;
        font-weight: 700;
        margin: 0.7rem 0 1rem 0;
    }

    /* -----------------------------------------------------
       METRICS
    ----------------------------------------------------- */

    .metric-card {
        padding: 1rem;
        border-radius: 14px;
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.075);
        text-align: center;
    }

    .metric-label {
        color: #9ca3af;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .metric-value {
        font-size: 1.55rem;
        font-weight: 800;
        margin-top: 0.3rem;
    }

    /* -----------------------------------------------------
       RACE BADGE
    ----------------------------------------------------- */

    .race-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.5rem 0.8rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.055);
        border: 1px solid rgba(255,255,255,0.08);
        color: #d1d5db;
        font-size: 0.82rem;
        font-weight: 600;
    }

    /* -----------------------------------------------------
       FOOTER
    ----------------------------------------------------- */

    .footer {
        text-align: center;
        color: #6b7280;
        font-size: 0.78rem;
        padding-top: 2rem;
    }

    /* -----------------------------------------------------
       BUTTON
    ----------------------------------------------------- */

    .stButton > button {
        border-radius: 10px;
        font-weight: 700;
        padding: 0.55rem 1rem;
        border: 1px solid rgba(255,255,255,0.14);
        background: rgba(255,255,255,0.055);
    }

    .stButton > button:hover {
        border-color: rgba(255,255,255,0.3);
        background: rgba(255,255,255,0.09);
    }

    /* -----------------------------------------------------
       INPUTS
    ----------------------------------------------------- */

    div[data-baseweb="select"] > div,
    textarea,
    input {
        border-radius: 10px !important;
    }

    </style>
    """
)


# =========================================================
# HERO
# =========================================================

render_html(
    """
    <div class="hero">
        <div class="hero-top">
            <div>
                <div class="hero-title">
                    <img
                        class="f1-logo"
                        src="https://upload.wikimedia.org/wikipedia/commons/8/8b/F1_%28white%29.svg"
                        alt="Formula 1 logo"
                    >
                    <span>F1 Strategy Copilot</span>
                </div>
                <div class="hero-subtitle">
                    Multi-Agent Generative AI System for Formula 1 Strategy Analysis
                </div>
            </div>

            <div class="status-pill">
                <span class="status-dot"></span>
                System Ready
            </div>
        </div>
    </div>
    """
)


# =========================================================
# LOAD RACE DATA
# =========================================================

try:

    race_df = pd.read_csv("data/laps.csv")

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

        if "2024" in race:
            display_race = race

        else:
            display_race = f"{race} 2024"

        if display_race not in race_options:
            race_options.append(display_race)

    race_options = sorted(race_options)


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
        "United States 2024",
    ]


# =========================================================
# DEFAULT RACE
# =========================================================

if "Monaco 2024" in race_options:

    default_race_index = race_options.index(
        "Monaco 2024"
    )

else:

    default_race_index = 0


# =========================================================
# INPUT PANEL
# =========================================================

render_html(
    """
    <div class="panel">
        <div class="panel-title">🎯 Analysis Setup</div>
        <div class="panel-description">
            Select a race and ask the Copilot a Formula 1 strategy question.
        </div>
    </div>
    """
)


selected_race = st.selectbox(
    "🏁 Select Race",
    race_options,
    index=default_race_index,
)


question = st.text_area(
    "Ask your F1 strategy question",
    placeholder=(
        "Example: Would pitting Verstappen five laps "
        "earlier have improved his race?"
    ),
    height=130,
)


st.markdown("")


analyze_clicked = st.button(
    "🔍  Analyze Strategy",
    use_container_width=False,
)


# =========================================================
# RUN WORKFLOW
# =========================================================

if analyze_clicked:

    if not question.strip():

        st.warning(
            "Please enter an F1 question before starting the analysis."
        )

        st.stop()


    # =====================================================
    # WORKFLOW STATUS
    # =====================================================

    with st.status(
        "Running F1 Strategy Copilot...",
        expanded=True,
    ) as workflow_status:

        st.write("🧭 Routing question to the appropriate agent...")

        try:

            result = run_f1_workflow(
                question=question,
                selected_race=selected_race,
            )

            st.write("🤖 Agent analysis completed.")
            st.write("🧪 Reviewing generated analysis...")

            workflow_status.update(
                label="F1 Strategy Copilot completed",
                state="complete",
                expanded=False,
            )

        except Exception as e:

            workflow_status.update(
                label="Workflow failed",
                state="error",
                expanded=True,
            )

            st.error(
                "An error occurred while running the F1 workflow."
            )

            st.exception(e)

            st.stop()


    # =====================================================
    # EXTRACT STATE
    # =====================================================

    intent = result.get(
        "intent",
        "UNKNOWN",
    )

    agents = result.get(
        "agents",
        [],
    )

    routing_reason = result.get(
        "routing_reason",
        "",
    )

    final_answer = result.get(
        "final_answer",
        "",
    )

    revision_count = result.get(
        "revision_count",
        0,
    )

    review_result = result.get(
        "review_result",
    )

    rag_result = result.get(
        "rag_result",
    )

    data_result = result.get(
        "data_result",
    )

    strategy_result = result.get(
        "strategy_result",
    )

    comparison_result = result.get(
        "comparison_result",
    )


    # =====================================================
    # ROUTER DECISION
    # =====================================================

    render_html(
    """
        <div class="panel">
            <div class="panel-title">🧭 Routing Decision</div>
        """
)

    agent_html = ""

    if agents:

        for agent in agents:

            agent_html += (
                f'<span class="agent-pill">🤖 {agent}</span>'
            )

    render_html(
    f"""
        <div class="router-card">
            <span class="intent-pill">
                {intent}
            </span>
            {agent_html}
        </div>
        """
)

    if routing_reason:

        st.caption(
            f"Routing reason: {routing_reason}"
        )

    with st.expander(
        "🔧 View raw router output"
    ):

        st.json(
            {
                "intent": intent,
                "agents": agents,
                "routing_reason": routing_reason,
            }
        )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )


    # =====================================================
    # RAG AGENT
    # =====================================================

    if intent == "RACE_KNOWLEDGE":

        render_html(
    """
            <div class="answer-card">
                <div class="answer-label">
                    🤖 Race Knowledge
                </div>
            """
)

        if rag_result:

            rag_answer = rag_result.get(
                "answer",
                final_answer,
            )

            render_html(
    f"""
                <div class="answer-text">
                    {rag_answer}
                </div>
                </div>
                """
)

            rag_sources = rag_result.get(
                "sources",
                [],
            )

            if rag_sources:

                st.subheader("📚 Sources")

                source_cols = st.columns(
                    min(len(rag_sources), 3)
                )

                for index, source in enumerate(
                    rag_sources
                ):

                    with source_cols[
                        index % len(source_cols)
                    ]:

                        render_html(
    f"""
                            <div class="info-card">
                                📄 {source}
                            </div>
                            """
)

        else:

            render_html(
    f"""
                <div class="answer-text">
                    {final_answer}
                </div>
                </div>
                """
)


    # =====================================================
    # DATA AGENT
    # =====================================================

    elif intent == "DATA_ANALYSIS":

        render_html(
    """
            <div class="answer-card">
                <div class="answer-label">
                    🔍 Data Analysis
                </div>
            """
)

        if final_answer:

            render_html(
    f"""
                <div class="answer-text">
                    {final_answer}
                </div>
                </div>
                """
)

        else:

            render_html(
    """
                </div>
                """
)

        if data_result:

            with st.expander(
                "📊 View Data Agent Details"
            ):

                st.json(
                    data_result
                )

        render_html(
    f"""
            <div class="race-badge">
                🏁 Race analyzed: {selected_race}
            </div>
            """
)


    # =====================================================
    # STRATEGY AGENT
    # =====================================================

    elif intent == "STRATEGY_ANALYSIS":

        # -------------------------------------------------
        # MAIN ANSWER
        # -------------------------------------------------

        render_html(
    """
            <div class="answer-card">
                <div class="answer-label">
                    🏎️ Strategy Agent Analysis
                </div>
            """
)

        strategy_answer = ""

        if final_answer:

            strategy_answer = final_answer

        elif strategy_result:

            strategy_answer = strategy_result.get(
                "answer",
                "",
            )

        if strategy_answer:

            render_html(
    f"""
                <div class="answer-text">
                    {strategy_answer}
                </div>
                </div>
                """
)

        else:

            st.warning(
                "The Strategy Agent did not return a final answer."
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )


        # -------------------------------------------------
        # RACE
        # -------------------------------------------------

        render_html(
    f"""
            <div class="race-badge">
                🏁 {selected_race}
            </div>
            """
)


        # -------------------------------------------------
        # REVIEW STATUS
        # -------------------------------------------------

        if review_result:

            verdict = review_result.get(
                "verdict",
                "",
            )

            if verdict == "PASS":

                if revision_count > 0:

                    render_html(
    f"""
                        <div class="review-pass">
                            ✅ Strategy passed review after
                            {revision_count} revision(s).
                        </div>
                        """
)

                else:

                    render_html(
    """
                        <div class="review-pass">
                            ✅ Strategy passed review without requiring a revision.
                        </div>
                        """
)

            else:

                render_html(
    f"""
                    <div class="review-fail">
                        ⚠️ Strategy review returned:
                        {verdict or "UNRESOLVED"}
                    </div>
                    """
)


        # -------------------------------------------------
        # STRATEGY DETAILS
        # -------------------------------------------------

        if strategy_result:

            evidence_used = strategy_result.get(
                "evidence_used",
                [],
            )

            strategy_factors = strategy_result.get(
                "strategy_factors",
                [],
            )

            limitations = strategy_result.get(
                "limitations",
                [],
            )

            rag_sources = strategy_result.get(
                "rag_sources",
                [],
            )


            # =============================================
            # EVIDENCE
            # =============================================

            if evidence_used:

                st.subheader(
                    "📊 Evidence Used"
                )

                evidence_cols = st.columns(
                    min(
                        len(evidence_used),
                        3,
                    )
                )

                for index, evidence in enumerate(
                    evidence_used
                ):

                    with evidence_cols[
                        index % len(evidence_cols)
                    ]:

                        render_html(
    f"""
                            <div class="info-card">
                                <div class="info-card-title">
                                    Evidence {index + 1}
                                </div>

                                <div class="info-card-item">
                                    {evidence}
                                </div>
                            </div>
                            """
)


            # =============================================
            # STRATEGY FACTORS
            # =============================================

            if strategy_factors:

                st.subheader(
                    "🧠 Strategy Factors"
                )

                factor_cols = st.columns(
                    min(
                        len(strategy_factors),
                        3,
                    )
                )

                for index, factor in enumerate(
                    strategy_factors
                ):

                    with factor_cols[
                        index % len(factor_cols)
                    ]:

                        render_html(
    f"""
                            <div class="info-card">
                                <div class="info-card-title">
                                    Factor {index + 1}
                                </div>

                                <div class="info-card-item">
                                    {factor}
                                </div>
                            </div>
                            """
)


            # =============================================
            # LIMITATIONS
            # =============================================

            if limitations:

                st.subheader(
                    "⚠️ Limitations"
                )

                for limitation in limitations:

                    st.warning(
                        limitation
                    )


            # =============================================
            # SOURCES
            # =============================================

            if rag_sources:

                unique_sources = list(
                    dict.fromkeys(
                        rag_sources
                    )
                )

                st.subheader(
                    "📚 Knowledge Sources"
                )

                source_cols = st.columns(
                    min(
                        len(unique_sources),
                        3,
                    )
                )

                for index, source in enumerate(
                    unique_sources
                ):

                    with source_cols[
                        index % len(source_cols)
                    ]:

                        render_html(
    f"""
                            <div class="info-card">
                                📄 {source}
                            </div>
                            """
)


            # =============================================
            # REVIEWER DETAILS
            # =============================================

            if review_result:

                st.subheader(
                    "🧪 Reviewer Details"
                )

                review_col1, review_col2 = st.columns(
                    2
                )

                with review_col1:

                    render_html(
    f"""
                        <div class="metric-card">
                            <div class="metric-label">
                                Verdict
                            </div>
                            <div class="metric-value">
                                {review_result.get(
                                    "verdict",
                                    "N/A"
                                )}
                            </div>
                        </div>
                        """
)

                with review_col2:

                    grounded = review_result.get(
                        "grounded",
                        False,
                    )

                    grounded_text = (
                        "Yes"
                        if grounded
                        else "No"
                    )

                    render_html(
    f"""
                        <div class="metric-card">
                            <div class="metric-label">
                                Evidence Grounded
                            </div>
                            <div class="metric-value">
                                {grounded_text}
                            </div>
                        </div>
                        """
)


            # =============================================
            # RAW OUTPUTS
            # =============================================

            with st.expander(
                "🔧 View Full Strategy Agent Output"
            ):

                st.json(
                    strategy_result
                )


            if review_result:

                with st.expander(
                    "🧪 View Full Reviewer Output"
                ):

                    st.json(
                        review_result
                    )


    # =====================================================
    # DRIVER COMPARISON
    # =====================================================

    elif intent == "DRIVER_COMPARISON":

        st.subheader(
            "🏁 Driver Comparison"
        )

        if not comparison_result:

            st.warning(
                "The Driver Comparison Agent did not return a result."
            )

            if final_answer:

                st.write(
                    final_answer
                )

        else:

            status = comparison_result.get(
                "status"
            )

            if status == "success":

                # -----------------------------------------
                # RACE
                # -----------------------------------------

                comparison_race = comparison_result.get(
                    "race"
                )

                if comparison_race:

                    render_html(
    f"""
                        <div class="race-badge">
                            🏁 {comparison_race}
                        </div>
                        """
)


                # -----------------------------------------
                # DRIVER STATISTICS
                # -----------------------------------------

                drivers = comparison_result.get(
                    "drivers",
                    [],
                )

                if drivers:

                    st.subheader(
                        "📊 Driver Statistics"
                    )

                    for driver_data in drivers:

                        driver = driver_data.get(
                            "driver",
                            "Unknown",
                        )

                        st.markdown(
                            f"### {driver}"
                        )


                        # =================================
                        # METRICS
                        # =================================

                        col1, col2, col3 = st.columns(
                            3
                        )


                        with col1:

                            average_lap = driver_data.get(
                                "average_lap_time",
                                0,
                            )

                            render_html(
    f"""
                                <div class="metric-card">
                                    <div class="metric-label">
                                        Average Lap
                                    </div>

                                    <div class="metric-value">
                                        {average_lap:.3f}s
                                    </div>
                                </div>
                                """
)


                        with col2:

                            fastest_lap = driver_data.get(
                                "fastest_lap",
                                0,
                            )

                            render_html(
    f"""
                                <div class="metric-card">
                                    <div class="metric-label">
                                        Fastest Lap
                                    </div>

                                    <div class="metric-value">
                                        {fastest_lap:.3f}s
                                    </div>
                                </div>
                                """
)


                        with col3:

                            average_position = driver_data.get(
                                "average_position",
                                0,
                            )

                            render_html(
    f"""
                                <div class="metric-card">
                                    <div class="metric-label">
                                        Average Position
                                    </div>

                                    <div class="metric-value">
                                        {average_position:.1f}
                                    </div>
                                </div>
                                """
)


                        st.markdown("")


                        # =================================
                        # ADDITIONAL STATS
                        # =================================

                        stat1, stat2 = st.columns(
                            2
                        )


                        with stat1:

                            laps_analyzed = driver_data.get(
                                "laps_analyzed",
                                0,
                            )

                            render_html(
    f"""
                                <div class="info-card">
                                    <div class="info-card-title">
                                        🏁 Laps Analyzed
                                    </div>

                                    <div class="info-card-item">
                                        {laps_analyzed}
                                    </div>
                                </div>
                                """
)


                        with stat2:

                            average_tyre_age = driver_data.get(
                                "average_tyre_age",
                                0,
                            )

                            render_html(
    f"""
                                <div class="info-card">
                                    <div class="info-card-title">
                                        🛞 Average Tyre Age
                                    </div>

                                    <div class="info-card-item">
                                        {average_tyre_age:.2f} laps
                                    </div>
                                </div>
                                """
)


                        # =================================
                        # COMPOUND STATS
                        # =================================

                        compound_stats = driver_data.get(
                            "compound_stats",
                            {},
                        )

                        if compound_stats:

                            st.markdown(
                                "**Tyre Compound Statistics**"
                            )

                            compound_cols = st.columns(
                                min(
                                    len(compound_stats),
                                    3,
                                )
                            )

                            for index, (
                                compound,
                                stats,
                            ) in enumerate(
                                compound_stats.items()
                            ):

                                with compound_cols[
                                    index % len(compound_cols)
                                ]:

                                    render_html(
    f"""
                                        <div class="info-card">
                                            <div class="info-card-title">
                                                🛞 {compound}
                                            </div>

                                            <div class="info-card-item">
                                                Average:
                                                {stats.get(
                                                    "average_lap_time",
                                                    0
                                                ): .3f}s
                                            </div>

                                            <div class="info-card-item">
                                                Fastest:
                                                {stats.get(
                                                    "fastest_lap",
                                                    0
                                                ): .3f}s
                                            </div>

                                            <div class="info-card-item">
                                                Laps:
                                                {stats.get(
                                                    "laps",
                                                    0
                                                )}
                                            </div>
                                        </div>
                                        """
)


                        # =================================
                        # DATA QUALITY
                        # =================================

                        data_quality = driver_data.get(
                            "data_quality",
                            {},
                        )

                        if data_quality:

                            with st.expander(
                                f"🔎 {driver} Data Quality"
                            ):

                                st.json(
                                    data_quality
                                )


                # -----------------------------------------
                # COMPARISON ANALYSIS
                # -----------------------------------------

                comparison_text = comparison_result.get(
                    "comparison",
                    "",
                )

                if comparison_text:

                    st.subheader(
                        "🧠 Comparison Analysis"
                    )

                    render_html(
    f"""
                        <div class="answer-card">
                            <div class="answer-text">
                                {comparison_text}
                            </div>
                        </div>
                        """
)


                # -----------------------------------------
                # RAW OUTPUT
                # -----------------------------------------

                with st.expander(
                    "🔧 View Full Driver Comparison Output"
                ):

                    st.json(
                        comparison_result
                    )


            else:

                st.error(
                    "The Driver Comparison Agent could not complete the comparison."
                )

                st.json(
                    comparison_result
                )


    # =====================================================
    # GENERAL / UNKNOWN
    # =====================================================

    else:

        st.info(
            f"The workflow classified this question as "
            f"**{intent}**."
        )

        if final_answer:

            render_html(
    f"""
                <div class="answer-card">
                    <div class="answer-label">
                        🧠 Copilot Answer
                    </div>

                    <div class="answer-text">
                        {final_answer}
                    </div>
                </div>
                """
)

        else:

            st.write(
                "No final answer was generated."
            )


# =========================================================
# FOOTER
# =========================================================

render_html(
    """
    <div class="footer">
        🏎️ F1 Strategy Copilot &nbsp;•&nbsp;
        Multi-Agent Generative AI &nbsp;•&nbsp;
        LangGraph
    </div>
    """
)
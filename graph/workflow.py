from langgraph.graph import StateGraph, START, END

from graph.state import F1State

from graph.nodes import (
    router_node,
    rag_node,
    data_node,
    strategy_node,
    driver_comparison_node,
    reviewer_node,
    revision_node,
    strategy_final_node,
    general_node,
)


# =========================================================
# Configuration
# =========================================================

MAX_REVISIONS = 2


# =========================================================
# Conditional routing
# =========================================================

def route_after_router(state: F1State) -> str:

    intent = state.get("intent")

    if intent == "RACE_KNOWLEDGE":
        return "rag"

    if intent == "DATA_ANALYSIS":
        return "data"

    if intent == "STRATEGY_ANALYSIS":
        return "strategy"

    if intent == "DRIVER_COMPARISON":
        return "driver_comparison"

    return "general"


# =========================================================
# Strategy reviewer routing
# =========================================================

def route_after_reviewer(state: F1State) -> str:

    review_result = state.get(
        "review_result",
        {}
    )

    verdict = review_result.get(
        "verdict",
        "PASS"
    )

    revision_count = state.get(
        "revision_count",
        0
    )

    # -----------------------------------------------------
    # Reviewer passed the strategy
    # -----------------------------------------------------

    if verdict == "PASS":
        return "strategy_final"

    # -----------------------------------------------------
    # Reviewer failed and revisions are still available
    # -----------------------------------------------------

    if (
        verdict == "FAIL"
        and revision_count < MAX_REVISIONS
    ):
        return "revision"

    # -----------------------------------------------------
    # Maximum revisions reached
    # -----------------------------------------------------

    return "strategy_final"


# =========================================================
# Build Graph
# =========================================================

builder = StateGraph(F1State)


# =========================================================
# Add Nodes
# =========================================================

builder.add_node(
    "router",
    router_node
)

builder.add_node(
    "rag",
    rag_node
)

builder.add_node(
    "data",
    data_node
)

builder.add_node(
    "strategy",
    strategy_node
)

builder.add_node(
    "driver_comparison",
    driver_comparison_node
)

builder.add_node(
    "reviewer",
    reviewer_node
)

builder.add_node(
    "revision",
    revision_node
)

builder.add_node(
    "strategy_final",
    strategy_final_node
)

builder.add_node(
    "general",
    general_node
)


# =========================================================
# START → Router
# =========================================================

builder.add_edge(
    START,
    "router"
)


# =========================================================
# Router → Specialized Agent
# =========================================================

builder.add_conditional_edges(
    "router",

    route_after_router,

    {
        "rag": "rag",
        "data": "data",
        "strategy": "strategy",
        "driver_comparison": "driver_comparison",
        "general": "general",
    }
)


# =========================================================
# RAG → END
# =========================================================

builder.add_edge(
    "rag",
    END
)


# =========================================================
# Data → END
# =========================================================

builder.add_edge(
    "data",
    END
)


# =========================================================
# Driver Comparison → END
# =========================================================

builder.add_edge(
    "driver_comparison",
    END
)


# =========================================================
# General → END
# =========================================================

builder.add_edge(
    "general",
    END
)


# =========================================================
# Strategy → Reviewer
# =========================================================

builder.add_edge(
    "strategy",
    "reviewer"
)


# =========================================================
# Reviewer → PASS / FAIL
# =========================================================

builder.add_conditional_edges(
    "reviewer",

    route_after_reviewer,

    {
        "strategy_final": "strategy_final",
        "revision": "revision",
    }
)


# =========================================================
# Revision → Reviewer
# =========================================================

builder.add_edge(
    "revision",
    "reviewer"
)


# =========================================================
# Final Strategy → END
# =========================================================

builder.add_edge(
    "strategy_final",
    END
)


# =========================================================
# Compile Graph
# =========================================================

f1_graph = builder.compile()


# =========================================================
# Public Workflow Function
# =========================================================

def run_f1_workflow(
    question: str,
    selected_race: str
) -> dict:
    """
    Run the complete F1 LangGraph workflow.

    Parameters
    ----------
    question : str
        The user's F1 question.

    selected_race : str
        The race selected in the Streamlit interface.

    Returns
    -------
    dict
        Final LangGraph state containing the routing decision,
        agent results, reviewer result, revision count,
        and final answer.
    """

    question = question.strip()

    if not question:
        raise ValueError(
            "Question cannot be empty."
        )

    question_with_race = (
        f"{question}\n\n"
        f"Selected race: {selected_race}"
    )

    initial_state: F1State = {
        "question": question,
        "selected_race": selected_race,
        "question_with_race": question_with_race,
        "revision_count": 0,
        "final_answer": "",
    }

    result = f1_graph.invoke(
        initial_state
    )

    return result
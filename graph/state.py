from typing import TypedDict, Any, Optional


class F1State(TypedDict, total=False):

    # ---------------------------------------------------------
    # User input
    # ---------------------------------------------------------

    question: str
    selected_race: str
    question_with_race: str

    # ---------------------------------------------------------
    # Router
    # ---------------------------------------------------------

    intent: str
    agents: list[str]
    routing_reason: str

    # ---------------------------------------------------------
    # Agent outputs
    # ---------------------------------------------------------

    rag_result: Optional[dict[str, Any]]
    data_result: Optional[dict[str, Any]]
    strategy_result: Optional[dict[str, Any]]
    comparison_result: Optional[dict[str, Any]]

    # ---------------------------------------------------------
    # Reviewer
    # ---------------------------------------------------------

    review_result: Optional[dict[str, Any]]
    revision_count: int

    # ---------------------------------------------------------
    # Final output
    # ---------------------------------------------------------

    final_answer: str
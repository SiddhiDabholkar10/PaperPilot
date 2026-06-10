from typing import TypedDict, Any, Literal

try:
    from typing import Required, NotRequired
except ImportError:
    from typing_extensions import Required, NotRequired


class SubQuestionCriticResultDict(TypedDict, total=False):
    """Mirrors SubQuestionCriticResult Pydantic schema as a plain TypedDict."""
    decision: str
    reason: str
    missing_evidence: list[str]


class AgentState(TypedDict, total=False):
    """
    Shared state passed through all agent nodes in the LangGraph pipeline.
    total=False means all fields are optional by default.
    Required fields must always be present.
    """

    # --- Required input fields ---
    original_query: Required[str]
    max_iterations: Required[int]
    current_iteration: Required[int]

    # --- Budget control ---
    last_reformulation_iteration: int  # set by reformulator when new queries generated; -1 when cleared

    # --- Decomposer output ---
    sub_questions: list[str]
    question_type: Literal[
        "bridge", "comparison", "aggregation", "causal", "definition"
    ]

    # --- Retrieval ---
    retrieved_chunks: dict[str, list[Any]]  # sub_question -> list[RetrievedChunk]
    retrieval_iterations: int

    # --- Critic output ---
    critic_results: dict[str, SubQuestionCriticResultDict]
    critic_decision: Literal["sufficient", "insufficient"]

    # --- Reformulator output ---
    reformulated_queries: dict[str, str]
    tried_queries: dict[str, list[str]]

    # --- Synthesizer output ---
    draft_answer: str
    citations: list[str]

    # --- Verifier output ---
    verified_answer: str
    verification_status: Literal["passed", "needs_revision", "failed"]
    unsupported_claims: list[str]  # contains both unsupported and weak claims

    # --- Reasoning trace ---
    reasoning_trace: list[dict]
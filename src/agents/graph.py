import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from typing import Literal
from langgraph.graph import StateGraph, END

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")

sys.path.insert(0, str(BASE_DIR / "src" / "agents"))
sys.path.insert(0, str(BASE_DIR / "src" / "retrieval"))

from state import AgentState
from decomposer import run_decomposer
from retriever_node import run_retriever_node
from critic import run_critic
from reformulator import run_reformulator
from synthesizer import run_synthesizer
from verifier import run_verifier

# Module-level default — overridden by state["max_iterations"] at runtime
MAX_ITERATIONS = 2


def route_after_critic(state: AgentState) -> Literal["reformulator", "synthesizer"]:
    """
    Routing decision after critic evaluation.

    Rules:
    - If budget exhausted → synthesizer (forced)
    - If all sub-questions sufficient → synthesizer
    - Otherwise → reformulator

    Uses state["max_iterations"] for budget, falls back to module constant.
    """
    critic_decision = state.get("critic_decision", "insufficient")
    current_iteration = state.get("current_iteration", 0)
    max_iterations = state.get("max_iterations", MAX_ITERATIONS)

    # Budget exhausted — force synthesis regardless of critic decision
    if current_iteration >= max_iterations:
        print(f"\n  Budget exhausted ({current_iteration}/{max_iterations} iterations) — proceeding to synthesis")
        return "synthesizer"

    # All sufficient — proceed to synthesis
    if critic_decision == "sufficient":
        print(f"\n  All sub-questions sufficient — proceeding to synthesis")
        return "synthesizer"

    # Insufficient and budget remaining — reformulate
    print(f"\n  Insufficient evidence (iteration {current_iteration}/{max_iterations}) — reformulating")
    return "reformulator"


def route_after_reformulator(state: AgentState) -> Literal["retriever", "synthesizer"]:
    """
    Routing decision after reformulator.

    Checks last_reformulation_iteration against current_iteration to detect
    whether reformulator actually generated new queries THIS iteration.
    Stale reformulated_queries from prior iterations are ignored because
    reformulator explicitly clears them (sets last_reformulation_iteration=-1)
    when no new queries are generated.

    If new queries generated this iteration → re-retrieve
    If reformulator made no changes → synthesize with what we have
    """
    current_iteration = state.get("current_iteration", 0)
    last_reform_iter = state.get("last_reformulation_iteration", -1)
    reformulated_queries = state.get("reformulated_queries", {})

    # New queries generated this iteration — re-retrieve
    if last_reform_iter == current_iteration and reformulated_queries:
        print(f"\n  {len(reformulated_queries)} new reformulated queries (iteration {current_iteration}) — re-retrieving")
        return "retriever"

    # No new queries — proceed to synthesis
    print(f"\n  No new queries from reformulator this iteration — proceeding to synthesis")
    return "synthesizer"


def build_graph() -> StateGraph:
    """
    Build the PaperPilot agent graph.

    Flow:
    START -> decomposer -> retriever -> critic -> [route_after_critic]
                                           |
                              sufficient --+--> synthesizer -> verifier -> END
                                           |
                             insufficient --+--> reformulator -> [route_after_reformulator]
                                                      |
                                         new queries --+--> retriever (loop back)
                                                      |
                                           exhausted --+--> synthesizer -> verifier -> END
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("decomposer", run_decomposer)
    graph.add_node("retriever", run_retriever_node)
    graph.add_node("critic", run_critic)
    graph.add_node("reformulator", run_reformulator)
    graph.add_node("synthesizer", run_synthesizer)
    graph.add_node("verifier", run_verifier)

    # Entry point
    graph.set_entry_point("decomposer")

    # Fixed edges
    graph.add_edge("decomposer", "retriever")
    graph.add_edge("retriever", "critic")
    graph.add_edge("synthesizer", "verifier")
    graph.add_edge("verifier", END)

    # Conditional edges
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "reformulator": "reformulator",
            "synthesizer": "synthesizer"
        }
    )

    graph.add_conditional_edges(
        "reformulator",
        route_after_reformulator,
        {
            "retriever": "retriever",
            "synthesizer": "synthesizer"
        }
    )

    return graph


def run_agent(query: str, max_iterations: int = MAX_ITERATIONS) -> dict:
    """
    Run the full PaperPilot agent on a query.
    Returns a clean result dict for the API layer.

    Args:
        query: The research question to answer
        max_iterations: Max reformulation loops before forcing synthesis
    """
    print(f"\n{'='*60}")
    print(f"PaperPilot Agent")
    print(f"Query: {query}")
    print(f"Max iterations: {max_iterations}")
    print(f"{'='*60}")

    graph = build_graph()
    app = graph.compile()

    initial_state: AgentState = {
        "original_query": query,
        "max_iterations": max_iterations,
        "current_iteration": 0,
        "reasoning_trace": []
    }

    try:
        final_state = app.invoke(initial_state)
    except Exception as e:
        print(f"\n  Graph execution failed: {e}")
        return {
            "query": query,
            "answer": f"Agent execution failed: {str(e)}",
            "citations": [],
            "verification_status": "failed",
            "reasoning_trace": [],
            "iterations_used": 0,
            "error": str(e)
        }

    # Build clean result dict for API layer
    result = {
        "query": query,
        "answer": final_state.get("verified_answer") or final_state.get("draft_answer", ""),
        "citations": final_state.get("citations", []),
        "verification_status": final_state.get("verification_status", "unknown"),
        "reasoning_trace": final_state.get("reasoning_trace", []),
        "iterations_used": final_state.get("current_iteration", 0),
        "sub_questions": final_state.get("sub_questions", []),
        "question_type": final_state.get("question_type", ""),
        "critic_decision": final_state.get("critic_decision", ""),
        "unsupported_claims": final_state.get("unsupported_claims", [])
    }

    print(f"\n{'='*60}")
    print(f"Agent complete")
    print(f"{'='*60}")
    print(f"  Verification: {result['verification_status']}")
    print(f"  Citations:    {len(result['citations'])}")
    print(f"  Iterations:   {result['iterations_used']}")
    print(f"  Answer ({len(result['answer'])} chars):")
    print(f"  {result['answer'][:400]}...")

    return result


if __name__ == "__main__":
    test_queries = [
    "What role does knowledge distillation play in making large language models more efficient?",
    "What are the trade-offs between memory efficiency and performance in KV-cache compression?",
]

    for query in test_queries:
        result = run_agent(query)

        print(f"\n  Sub-questions ({result['question_type']}):")
        for sq in result["sub_questions"]:
            print(f"    - {sq}")

        print(f"\n  Reasoning trace ({len(result['reasoning_trace'])} steps):")
        for step in result["reasoning_trace"]:
            print(f"    [{step['step']}]", end=" ")
            if step["step"] == "decompose":
                print(f"{len(step['sub_questions'])} sub-questions")
            elif step["step"] == "retrieve":
                print(f"{step['chunks_found']} chunks")
            elif step["step"] == "critic":
                print(f"{step['overall_decision']}")
            elif step["step"] == "reformulate":
                status = step.get("status", "")
                if status == "skipped" or status == "exhausted":
                    print(f"skipped - {step.get('reason', '')}")
                else:
                    print(f"{len(step.get('reformulations', {}))} reformulations")
            elif step["step"] == "synthesize":
                print(f"{step['answer_length']} chars, {step['citation_count']} citations")
            elif step["step"] == "verify":
                print(f"{step['verification_status']}, {step['claims_checked']} claims")
            else:
                print()
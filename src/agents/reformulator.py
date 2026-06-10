import os
import sys
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from llm_client import get_llm
from langchain_core.messages import SystemMessage, HumanMessage

from agent_utils import parse_json_robust

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")

sys.path.insert(0, str(BASE_DIR / "src" / "agents"))

from schemas import ReformulatorOutput
from state import AgentState


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
REFORMULATOR_MODEL = "deepseek/deepseek-v4-flash"

SYSTEM_PROMPT = """You are a query reformulator for a scientific paper retrieval system.

Your job is to rewrite failed sub-questions to improve retrieval of relevant papers.

You will receive:
- The original research question
- All sub-questions (sufficient and insufficient)
- For each insufficient sub-question: why it failed and what evidence is missing
- Queries already tried (to avoid repetition)

Reformulation strategies (use the most appropriate):
1. Vocabulary expansion: use synonyms and related technical terms
2. Specificity adjustment: make the query more or less specific
3. Perspective shift: ask from a different angle that targets the same concept
4. Term substitution: replace abstract terms with concrete technical terms from the domain

Rules:
- Only reformulate INSUFFICIENT sub-questions
- Do NOT reformulate sufficient sub-questions
- Use missing_evidence terms to guide vocabulary expansion
- Never repeat a query that was already tried
- Keep queries focused and searchable (not too long)
- Return ONLY valid JSON, no preamble, no markdown

Return this exact structure:
{
  "reformulated_queries": {
    "<original_sub_question>": "<reformulated_query>",
    "<original_sub_question_2>": "<reformulated_query_2>"
  },
  "reasoning": "brief explanation of reformulation strategy used"
}

Only include insufficient sub-questions in reformulated_queries."""


def _build_reformulator_context(
    original_query: str,
    sub_questions: list[str],
    critic_results: dict,
    tried_queries: dict
) -> str:
    """Build the context string for the reformulator prompt."""
    lines = [
        f"Original research question: {original_query}",
        f"\nAll sub-questions:",
    ]

    for i, sq in enumerate(sub_questions):
        result = critic_results.get(sq, {})
        decision = result.get("decision", "unknown")
        status = "SUFFICIENT" if decision == "sufficient" else "INSUFFICIENT"
        lines.append(f"  [{i}] {status}: {sq}")

    lines.append("\nInsufficient sub-questions (need reformulation):")

    insufficient_found = False
    for sq in sub_questions:
        result = critic_results.get(sq, {})
        if result.get("decision") != "insufficient":
            continue

        insufficient_found = True
        lines.append(f"\nSub-question: {sq}")
        lines.append(f"Reason failed: {result.get('reason', 'unknown')}")

        missing = result.get("missing_evidence", [])
        if missing:
            lines.append(f"Missing evidence: {', '.join(missing)}")

        tried = tried_queries.get(sq, [])
        if tried:
            lines.append(f"Already tried: {' | '.join(tried)}")

    if not insufficient_found:
        lines.append("  (none - all sufficient)")

    return "\n".join(lines)


def _parse_reformulator_response(raw: str) -> dict:
    return parse_json_robust(raw)


def _validate_reformulator_output(
    parsed: dict,
    sub_questions: list[str],
    critic_results: dict,
    tried_queries: dict
) -> ReformulatorOutput:
    """
    Validate reformulator output.
    - Only keep reformulations for insufficient sub-questions
    - Skip if reformulated query matches an already-tried query
    - Skip if no meaningful reformulation can be generated
    """
    raw_reformulations = parsed.get("reformulated_queries", {})
    reasoning = parsed.get("reasoning", "")
    validated = {}

    for sq in sub_questions:
        result = critic_results.get(sq, {})
        if result.get("decision") != "insufficient":
            continue

        reformulated = raw_reformulations.get(sq, "").strip()
        missing = result.get("missing_evidence", [])
        already_tried = tried_queries.get(sq, [])

        # If LLM gave nothing useful, try keyword expansion
        if not reformulated or reformulated == sq:
            if missing:
                reformulated = f"{sq} {' '.join(missing[:3])}"
            else:
                print(f"  SKIP: no reformulation possible for: {sq[:50]}")
                continue

        # If already tried, attempt variation with missing evidence terms
        if reformulated in already_tried:
            extra_terms = missing[:2] if missing else ["methods", "techniques"]
            found_variation = False
            for term in extra_terms:
                candidate = f"{reformulated} {term}"
                if candidate not in already_tried:
                    reformulated = candidate
                    found_variation = True
                    break
            if not found_variation:
                print(f"  SKIP: all variations exhausted for: {sq[:50]}")
                continue

        validated[sq] = reformulated

    return ReformulatorOutput(
        reformulated_queries=validated,
        reasoning=reasoning or "reformulation based on missing evidence"
    )


def _clear_reformulation_state(state: AgentState, reason: str) -> AgentState:
    """
    Clear reformulation signals from state and add trace entry.
    Called when reformulator generates no new queries to prevent
    stale reformulated_queries from causing routing loops in graph.
    """
    current_iteration = state.get("current_iteration", 0)
    updated = dict(state)
    updated["reformulated_queries"] = {}
    updated["last_reformulation_iteration"] = -1
    updated["reasoning_trace"] = state.get("reasoning_trace", []) + [{
        "step": "reformulate",
        "iteration": current_iteration,
        "status": "skipped",
        "reason": reason
    }]
    return updated


def run_reformulator(state: AgentState) -> AgentState:
    """
    Rewrite insufficient sub-questions based on critic feedback.

    Strategy:
    - Preflight check for OPENROUTER_API_KEY
    - Use global context (all sub-questions + critic feedback)
    - Use 8B model - reformulation is mechanical, not reasoning-heavy
    - Expand vocabulary using missing_evidence terms
    - Track tried queries to avoid repetition
    - Fall back to keyword-expanded query if LLM fails
    - Explicitly clear reformulated_queries when no new queries generated
      (prevents stale state from causing routing loops in graph)

    Updates state with:
    - reformulated_queries (dict: original sq -> reformulated query)
    - tried_queries (updated with new queries)
    - current_iteration (incremented only if reformulations generated)
    - last_reformulation_iteration (set to current+1 when new queries generated, -1 otherwise)
    - reasoning_trace entry (always, including skip/exhausted cases)
    """
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not found in .env — cannot run reformulator")

    original_query = state["original_query"]
    sub_questions = state.get("sub_questions", [])
    critic_results = state.get("critic_results", {})
    tried_queries = dict(state.get("tried_queries", {}))
    current_iteration = state.get("current_iteration", 0)

    # Identify insufficient sub-questions
    insufficient = [
        sq for sq in sub_questions
        if critic_results.get(sq, {}).get("decision") == "insufficient"
    ]

    if not insufficient:
        print("  No insufficient sub-questions - skipping reformulation")
        return _clear_reformulation_state(state, "no insufficient sub-questions")

    print(f"\n  Reformulating {len(insufficient)} insufficient sub-question(s)...")

    llm = get_llm("reformulator", temperature=0.1, max_tokens=512)

    context = _build_reformulator_context(
        original_query, sub_questions, critic_results, tried_queries
    )

    output = None
    last_error = None

    for attempt in range(2):
        try:
            if attempt == 0:
                messages = [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(
                        content=f"Reformulate the insufficient sub-questions:\n\n{context}"
                    )
                ]
            else:
                print(f"  Retrying reformulator (attempt 2)...")
                time.sleep(1)
                messages = [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=(
                        f"Reformulate insufficient sub-questions. "
                        f"Return ONLY valid JSON with reformulated_queries dict. "
                        f"No markdown.\n\n{context}"
                    ))
                ]

            response = llm.invoke(messages)
            raw = response.content.strip()
            parsed = _parse_reformulator_response(raw)
            output = _validate_reformulator_output(
                parsed, sub_questions, critic_results, tried_queries
            )
            break

        except Exception as e:
            last_error = f"{type(e).__name__}: {str(e)}"
            print(f"  Reformulator attempt {attempt + 1} failed [{type(e).__name__}]: {e}")
            if "rate_limit" in str(e).lower() or "429" in str(e):
                from agent_utils import handle_rate_limit
                handle_rate_limit(e)
            elif attempt < 1:
                time.sleep(1)

    # Fallback - keyword expansion from missing evidence only
    if output is None:
        print(f"  Reformulator fallback - using keyword expansion")
        print(f"  Last error: {last_error}")
        fallback = {}
        for sq in insufficient:
            result = critic_results.get(sq, {})
            missing = result.get("missing_evidence", [])
            if missing:
                candidate = f"{sq} {' '.join(missing[:3])}"
                already_tried = tried_queries.get(sq, [])
                if candidate not in already_tried:
                    fallback[sq] = candidate
        output = ReformulatorOutput(
            reformulated_queries=fallback,
            reasoning="fallback keyword expansion"
        )

    # If no reformulations were generated - clear state and return
    if not output.reformulated_queries:
        print("  All reformulations skipped or exhausted - no new queries to try")
        return _clear_reformulation_state(state, "no new queries could be generated")

    # Update tried_queries
    for sq, new_query in output.reformulated_queries.items():
        if sq not in tried_queries:
            tried_queries[sq] = [sq]  # original query was first attempt
        if new_query not in tried_queries[sq]:
            tried_queries[sq].append(new_query)

    # Build trace entry
    trace_entry = {
        "step": "reformulate",
        "iteration": current_iteration + 1,
        "status": "success",
        "reformulations": {
            sq: {
                "original": sq,
                "reformulated": new_query,
                "tried_so_far": tried_queries.get(sq, [])
            }
            for sq, new_query in output.reformulated_queries.items()
        },
        "reasoning": output.reasoning
    }

    # Update state
    updated_state = dict(state)
    updated_state["reformulated_queries"] = output.reformulated_queries
    updated_state["tried_queries"] = tried_queries
    updated_state["current_iteration"] = current_iteration + 1
    updated_state["last_reformulation_iteration"] = current_iteration + 1
    updated_state["reasoning_trace"] = state.get("reasoning_trace", []) + [trace_entry]

    print(f"  Reasoning: {output.reasoning[:100]}")
    for sq, new_query in output.reformulated_queries.items():
        print(f"\n  Original:     {sq[:65]}")
        print(f"  Reformulated: {new_query[:65]}")

    return updated_state


if __name__ == "__main__":
    from decomposer import run_decomposer
    from retriever_node import run_retriever_node
    from critic import run_critic

    test_queries = [
        "How do retrieval-based methods for LLM memory compare to parametric memory approaches?",
        "How did rotary position embeddings introduced in RoFormer influence later work on linear attention mechanisms?",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")

        state: AgentState = {
            "original_query": query,
            "max_iterations": 3,
            "current_iteration": 0,
            "reasoning_trace": []
        }

        state = run_decomposer(state)
        state = run_retriever_node(state)
        state = run_critic(state)

        if state["critic_decision"] == "insufficient":
            print(f"\n  Running reformulator...")
            prev_iteration = state.get("current_iteration", 0)
            state = run_reformulator(state)

            if state.get("current_iteration", 0) > prev_iteration:
                print(f"\n  Re-retrieving with reformulated queries...")
                state = run_retriever_node(state)
                print(f"\n  Re-running critic...")
                state = run_critic(state)
            else:
                print(f"\n  Reformulator made no changes - skipping re-retrieval")

        print(f"\n  Final critic decision: {state['critic_decision']}")
        print(f"  Iterations used: {state.get('current_iteration', 0)}")
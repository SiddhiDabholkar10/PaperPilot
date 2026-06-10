import os
import sys
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from llm_client import get_llm
from langchain_core.messages import SystemMessage, HumanMessage
from agent_utils import  parse_json_robust


BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")

sys.path.insert(0, str(BASE_DIR / "src" / "agents"))

from schemas import CriticOutput, SubQuestionCriticResult
from state import AgentState


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


# Reduced from 3→2 and 800→500 to control prompt size
TOP_CHUNKS_FULL_TEXT = 2
MAX_CHUNK_CHARS = 500

SYSTEM_PROMPT = """You are a retrieval quality critic for a scientific paper RAG system.

Your job is to evaluate whether retrieved chunks contain sufficient evidence to answer each sub-question.

For each sub-question you will receive:
- The sub-question index and text
- Top 2 chunks with full text
- Additional chunks with title and section only

Evaluation rules:
1. Mark "sufficient" if at least 1 chunk contains DIRECT evidence for the sub-question's core claim
2. Mark "insufficient" if no chunk contains direct evidence — generic background does not count
3. Direct evidence means: specific methods, results, comparisons, or mechanisms that directly address the sub-question
4. Topic drift counts as insufficient: chunks about a related topic that don't answer the specific question
5. Be specific in your reasoning — state exactly what evidence was found or what is missing

Return ONLY valid JSON, no preamble, no markdown.

Return this EXACT structure — use sub_question_index (integer) to identify each result:
{
  "results": [
    {
      "sub_question_index": 0,
      "decision": "sufficient",
      "reason": "specific explanation of evidence found",
      "missing_evidence": []
    },
    {
      "sub_question_index": 1,
      "decision": "insufficient",
      "reason": "specific explanation of what is missing",
      "missing_evidence": ["concept A", "concept B"]
    }
  ],
  "overall_decision": "insufficient"
}

Rules for overall_decision:
- "insufficient" if ANY result is insufficient
- "sufficient" only if ALL results are sufficient"""


def _format_chunks_for_critic(
    index: int,
    sub_question: str,
    chunks: list,
    top_n_full: int = TOP_CHUNKS_FULL_TEXT,
    max_chars: int = MAX_CHUNK_CHARS
) -> str:
    """
    Format chunks for critic prompt.
    Top N get full text, rest get title + section only.
    Uses index instead of full text for key matching safety.
    """
    if not chunks:
        return f"[{index}] Sub-question: {sub_question}\nNo chunks retrieved.\n"

    lines = [f"[{index}] Sub-question: {sub_question}"]

    hybrid_chunks = [c for c in chunks if c.retrieval_method == "hybrid"]
    abstract_chunks = [c for c in chunks if c.retrieval_method == "abstract_supplement"]

    # Top N full text
    lines.append(f"\nTop {min(top_n_full, len(hybrid_chunks))} chunks (full text):")
    for i, chunk in enumerate(hybrid_chunks[:top_n_full]):
        text_preview = chunk.text[:max_chars].strip()
        if len(chunk.text) > max_chars:
            text_preview += "..."
        lines.append(
            f"\n[Chunk {i+1}] {chunk.title[:60]} | {chunk.section[:40]} | {chunk.year}"
            f"\nRRF: {chunk.rrf_score:.4f} | Citations: {chunk.citation_count}"
            f"\nText: {text_preview}"
        )

    # Remaining hybrid — metadata only
    remaining = hybrid_chunks[top_n_full:]
    if remaining:
        lines.append(f"\nAdditional chunks (metadata only):")
        for chunk in remaining:
            lines.append(
                f"  - {chunk.title[:60]} | {chunk.section[:40]} | {chunk.year}"
            )

    # Abstract supplements — metadata only
    if abstract_chunks:
        lines.append(f"\nAbstract supplements:")
        for chunk in abstract_chunks:
            lines.append(
                f"  - {chunk.title[:60]} | {chunk.year} | Citations: {chunk.citation_count}"
            )

    return "\n".join(lines)


def _parse_critic_response(raw: str) -> dict:
    return parse_json_robust(raw)


def _normalize_missing_evidence(raw: object) -> list[str]:
    """Normalize missing_evidence to list[str] regardless of LLM output format."""
    if isinstance(raw, str):
        return [raw] if raw.strip() else []
    if isinstance(raw, list):
        return [str(m) for m in raw if m]
    return []


def _validate_critic_output(
    parsed: dict,
    sub_questions: list[str]
) -> CriticOutput:
    """
    Validate parsed critic output against schema.
    Uses index-based matching — robust to LLM text paraphrasing.
    Falls back to insufficient for any missing evaluation.
    """
    raw_results = parsed.get("results", [])

    # Handle backward-compatible dict format from LLM
    if isinstance(raw_results, dict):
        converted = []
        for i, sq in enumerate(sub_questions):
            match = raw_results.get(sq)
            if match and isinstance(match, dict):
                match["sub_question_index"] = i
                converted.append(match)
        raw_results = converted

    # Build index → result map
    index_map = {}
    for item in raw_results:
        if isinstance(item, dict):
            idx = item.get("sub_question_index")
            if idx is not None:
                try:
                    index_map[int(idx)] = item
                except (TypeError, ValueError):
                    pass  # skip malformed index

    validated_results = {}
    for i, sq in enumerate(sub_questions):
        item = index_map.get(i)
        if item:
            decision = item.get("decision", "insufficient")
            if decision not in ("sufficient", "insufficient"):
                decision = "insufficient"

            missing_evidence = _normalize_missing_evidence(
                item.get("missing_evidence", [])
            )

            validated_results[sq] = SubQuestionCriticResult(
                decision=decision,
                reason=item.get("reason", "No reason provided"),
                missing_evidence=missing_evidence
            )
        else:
            # Missing from LLM response — mark insufficient
            validated_results[sq] = SubQuestionCriticResult(
                decision="insufficient",
                reason="Sub-question not evaluated by critic",
                missing_evidence=["evaluation missing"]
            )

    # Overall: insufficient if any sub-question is insufficient
    any_insufficient = any(
        r.decision == "insufficient"
        for r in validated_results.values()
    )
    overall = "insufficient" if any_insufficient else "sufficient"

    # Respect LLM overall if it's stricter
    llm_overall = parsed.get("overall_decision", overall)
    if llm_overall == "insufficient":
        overall = "insufficient"

    return CriticOutput(
        results=validated_results,
        overall_decision=overall
    )


def run_critic(state: AgentState) -> AgentState:
    """
    Evaluate retrieval sufficiency per sub-question.

    Strategy:
    - Show top 2 chunks with full text (500 chars), rest metadata only
    - Use indexed schema to avoid key-matching fragility
    - Mark sufficient if at least 1 chunk has direct evidence
    - Mark insufficient if any sub-question lacks direct evidence
    - Retry once on parse failure
    - Fall back to all-insufficient on total failure

    Updates state with:
    - critic_results
    - critic_decision
    - reasoning_trace entry
    """
    sub_questions = state.get("sub_questions", [])
    retrieved_chunks = state.get("retrieved_chunks", {})
    retrieval_iterations = state.get("retrieval_iterations", 0)

    if not sub_questions:
        print("  WARNING: No sub-questions to critique")
        updated = dict(state)
        updated["critic_decision"] = "sufficient"
        updated["critic_results"] = {}
        return updated

    llm = get_llm("critic", temperature=0.0, max_tokens=1024)

    # Build evidence blocks — separator between each block
    evidence_blocks = []
    for i, sq in enumerate(sub_questions):
        chunks = retrieved_chunks.get(sq, [])
        block = _format_chunks_for_critic(i, sq, chunks)
        evidence_blocks.append(block)

    separator = "\n\n" + "=" * 50 + "\n\n"
    evidence_text = separator.join(evidence_blocks)

    output = None
    last_error = None

    for attempt in range(2):
        try:
            if attempt == 0:
                messages = [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(
                        content=f"Evaluate retrieval sufficiency:\n\n{evidence_text}"
                    )
                ]
            else:
                print(f"  Retrying critic (attempt 2)...")
                time.sleep(1)
                messages = [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=(
                        f"Evaluate retrieval sufficiency. "
                        f"Return ONLY valid JSON with indexed results array. "
                        f"No markdown, no explanation.\n\n{evidence_text}"
                    ))
                ]

            response = llm.invoke(messages)
            raw = response.content.strip()
            parsed = _parse_critic_response(raw)
            output = _validate_critic_output(parsed, sub_questions)
            break

        except Exception as e:
            last_error = f"{type(e).__name__}: {str(e)}"
            print(f"  Critic attempt {attempt + 1} failed [{type(e).__name__}]: {e}")

    # Fallback — all insufficient to force reformulation
    if output is None:
        print(f"  Critic fallback — marking all insufficient")
        print(f"  Last error: {last_error}")
        fallback_results = {
            sq: SubQuestionCriticResult(
                decision="insufficient",
                reason="Critic evaluation failed — defaulting to insufficient",
                missing_evidence=["critic evaluation failed"]
            )
            for sq in sub_questions
        }
        output = CriticOutput(
            results=fallback_results,
            overall_decision="insufficient"
        )

    # Build trace entry
    trace_entry = {
        "step": "critic",
        "iteration": retrieval_iterations,
        "overall_decision": output.overall_decision,
        "per_question_results": {
            sq: {
                "decision": r.decision,
                "reason": r.reason[:150],
                "missing_evidence": r.missing_evidence
            }
            for sq, r in output.results.items()
        }
    }

    # Convert Pydantic results to plain dicts for state
    critic_results_dict = {
        sq: {
            "decision": r.decision,
            "reason": r.reason,
            "missing_evidence": r.missing_evidence
        }
        for sq, r in output.results.items()
    }

    # Update state
    updated_state = dict(state)
    updated_state["critic_results"] = critic_results_dict
    updated_state["critic_decision"] = output.overall_decision
    updated_state["reasoning_trace"] = state.get("reasoning_trace", []) + [trace_entry]

    # Print summary
    print(f"\n  Critic decision: {output.overall_decision.upper()}")
    for sq, result in output.results.items():
        icon = "[OK]" if result.decision == "sufficient" else "[FAIL]"
        print(f"  {icon} {sq[:65]}")
        print(f"     {result.reason[:120]}")
        if result.missing_evidence:
            print(f"     Missing: {result.missing_evidence}")

    return updated_state


if __name__ == "__main__":
    from decomposer import run_decomposer
    from retriever_node import run_retriever_node

    test_queries = [
        # Should be sufficient — both papers in corpus
        "How did rotary position embeddings introduced in RoFormer influence later work on linear attention mechanisms?",
        # Likely insufficient on first pass — vocabulary mismatch
        "How do retrieval-based methods for LLM memory compare to parametric memory approaches?",
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

        print(f"\n  Final decision: {state['critic_decision']}")
        print(f"\n  Insufficient sub-questions:")
        for sq, result in state["critic_results"].items():
            if result["decision"] == "insufficient":
                print(f"    - {sq[:65]}")
                print(f"      Missing: {result['missing_evidence']}")
import os
import sys
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from llm_client import get_llm
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from agent_utils import handle_rate_limit, parse_json_robust


BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")

sys.path.insert(0, str(BASE_DIR / "src" / "agents"))

from schemas import SynthesizerOutput
from state import AgentState

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Max chars per chunk passed to synthesizer
MAX_CHUNK_CHARS = 800
# Max chunks per sub-question passed to synthesizer
MAX_CHUNKS_PER_SQ = 6

SYSTEM_PROMPT = """You are a scientific research synthesizer for a paper retrieval system.

Your job is to write a grounded, cited answer to a research question using retrieved paper chunks.

You will receive:
- The original research question
- Sub-questions with their sufficiency status
- Retrieved chunks per sub-question (with paper titles, years, and text)

Rules:
1. Every factual claim MUST be cited inline as [arxiv_id] immediately after the claim
2. Cite EVERY paper that appears in the retrieved chunks — you MUST cite every arxiv_id shown in the evidence, even if briefly
3. If a paper appears in the retrieved chunks, write at least one sentence about what it says and cite it
4. For each retrieved paper, write what it says — describe its findings positively and confidently
5. Write in clear academic prose — not bullet points
6. Citations go inline immediately after the claim they support: "X was shown [1706.03762]"
7. Return ONLY valid JSON, no preamble, no markdown
8. Do NOT mention any technique or finding not explicitly in the provided chunks
9. If chunks only name a technique without details, cite the paper and state only what the chunk says

FORBIDDEN — never write these phrases:
- "the retrieved evidence does not..."
- "evidence was insufficient..."
- "the available chunks do not..."
- "no evidence was found..."
- "this sub-question cannot be answered"
- "remains unsupported by the evidence"
- "the provided chunks do not..."

Return this exact structure:
{
  "answer": "Full answer text with inline [arxiv_id] citations",
  "citations": ["arxiv_id_1", "arxiv_id_2", ...]
}

Citation format: [arxiv_id] e.g. [1706.03762]
The citations list should contain all unique arxiv_ids cited in the answer."""

def _format_chunks_for_synthesis(
    sub_questions: list[str],
    retrieved_chunks: dict,
    critic_results: dict,
    max_chunks: int = MAX_CHUNKS_PER_SQ,
    max_chars: int = MAX_CHUNK_CHARS
) -> str:
    """
    Format all retrieved chunks for the synthesizer prompt.
    Groups by sub-question with sufficiency status.
    Only includes hybrid chunks — abstracts supplement context silently.
    """
    lines = []
    total_chunks_sent = 0
    MAX_TOTAL_CHUNKS = 20  # hard cap across all sub-questions

    for sq in sub_questions:
        if total_chunks_sent >= MAX_TOTAL_CHUNKS:
            break

        result = critic_results.get(sq, {})
        decision = result.get("decision", "unknown")
        status = "SUFFICIENT" if decision == "sufficient" else "INSUFFICIENT"

        lines.append(f"\n--- Sub-question [{status}]: {sq} ---")

        chunks = retrieved_chunks.get(sq, [])
        hybrid = [c for c in chunks if c.retrieval_method == "hybrid"]

        if not hybrid:
            lines.append("No chunks retrieved for this sub-question.")
            continue

        for i, chunk in enumerate(hybrid[:max_chunks]):
            if total_chunks_sent >= MAX_TOTAL_CHUNKS:
                break
            text = chunk.text[:max_chars].strip()
            if len(chunk.text) > max_chars:
                text += "..."
            lines.append(
                f"\n[{chunk.arxiv_id}] {chunk.title[:70]} ({chunk.year})"
                f"\nSection: {chunk.section[:50]}"
                f"\nText: {text}"
            )
            total_chunks_sent += 1

    return "\n".join(lines)

def _parse_synthesizer_response(raw: str) -> dict:
    """Parse LLM response to JSON with robust fallback strategies."""
    return parse_json_robust(raw)

def _validate_synthesizer_output(
    parsed: dict,
    retrieved_chunks: dict
) -> SynthesizerOutput:
    """
    Validate synthesizer output.
    - Ensure answer is non-empty
    - Fix malformed citations (space inside brackets, missing closing bracket)
    - Remove non-arxiv numeric citations
    - Extract citations from answer text if citations list is missing
    - Verify citations reference papers actually in retrieved chunks
    """
    import re

    answer = parsed.get("answer", "").strip()
    if not answer:
        raise ValueError("Synthesizer returned empty answer")

    # Debug: show raw answer before any processing
    print(f"  RAW answer sample: {repr(answer[:500])}")

    # Step 1: Remove non-arxiv numeric citations like [7], [17], [1, 2]
    answer = re.sub(r'\[(\d{1,3}(?:,\s*\d{1,3})*)\]', '', answer)

    # Step 1b: Fix malformed arxiv citations
    answer_before = answer
    # Case A: space inside brackets e.g. [2503.11108 ] → [2503.11108]
    answer = re.sub(r'\[(\d{4}\.\d{4,5})\s+\]', r'[\1]', answer)
    # Case B: missing closing bracket e.g. [2503.11108 . → [2503.11108].
    answer = re.sub(r'\[(\d{4}\.\d{4,5})\s+(?=[^\]])', r'[\1] ', answer)
    if answer != answer_before:
        print(f"  Step 1b fixed citations")
    
    print(f"  POST-1b sample: {repr(answer[:300])}")
    

    answer = re.sub(r'\s+', ' ', answer).strip()

    # Step 2: Extract arxiv IDs from cleaned answer
    cited_in_text = set(re.findall(r'\[\s*(\d{4}\.\d{4,5})\s*\]', answer))

    # Get citations list from LLM or derive from text
    raw_citations = parsed.get("citations", [])
    if isinstance(raw_citations, list) and raw_citations:
        citations = [str(c).strip() for c in raw_citations if c]
    else:
        citations = list(cited_in_text)

    # Add any citations found in text but missing from list
    all_citations = list(set(citations) | cited_in_text)

    # Verify citations are from retrieved chunks
    valid_arxiv_ids = set()
    for chunks in retrieved_chunks.values():
        for chunk in chunks:
            valid_arxiv_ids.add(chunk.arxiv_id)

    verified_citations = [c for c in all_citations if c in valid_arxiv_ids]
    invalid = [c for c in all_citations if c not in valid_arxiv_ids]

    if invalid:
        print(f"  WARNING: {len(invalid)} citations not in retrieved chunks: {invalid}")
        for inv in invalid:
            answer = re.sub(r'\[\s*' + re.escape(inv) + r'\s*\]', '', answer)
        answer = re.sub(r'\s+', ' ', answer).strip()
    print(f"  FINAL answer sample: {repr(answer[:300])}")
    return SynthesizerOutput(
        answer=answer,
        citations=verified_citations
    )

def _ensure_gaps_acknowledged(
    answer: str,
    sub_questions: list[str],
    critic_results: dict
) -> tuple[str, str]:
    """
    Returns (clean_answer, gap_text) separately.
    Gap text is stored in state but NOT appended to answer.
    """
    insufficient = [
        sq for sq in sub_questions
        if critic_results.get(sq, {}).get("decision") == "insufficient"
    ]

    if not insufficient:
        return answer, ""

    gap_text = (
        "Evidence was insufficient for the following aspects: "
        + "; ".join(sq[:80] for sq in insufficient)
        + ". These aspects could not be fully addressed from the available papers."
    )

    return answer, gap_text

def run_synthesizer(state: AgentState) -> AgentState:
   
    """
    Generate a grounded, cited answer from retrieved chunks.

    Strategy:
    - Use top MAX_CHUNKS_PER_SQ hybrid chunks per sub-question
    - Explicitly acknowledge gaps for insufficient sub-questions
    - Verify all citations reference retrieved papers
    - Retry once on parse failure
    - Fall back to structured unavailable message on total failure

    Updates state with:
    - draft_answer
    - citations
    - reasoning_trace entry
    """

    # Preflight check
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not found in .env — cannot run synthesizer")
    
    original_query = state["original_query"]
    sub_questions = state.get("sub_questions", [])
    retrieved_chunks = state.get("retrieved_chunks", {})
    critic_results = state.get("critic_results", {})
    retrieval_iterations = state.get("retrieval_iterations", 0)

    if not retrieved_chunks:
        print("  WARNING: No retrieved chunks — cannot synthesize")
        updated = dict(state)
        updated["draft_answer"] = "Insufficient evidence retrieved to answer this question."
        updated["citations"] = []
        return updated

    llm = get_llm("synthesizer", temperature=0.1, max_tokens=800)
    # Build evidence text
    evidence_text = _format_chunks_for_synthesis(
        sub_questions, retrieved_chunks, critic_results
    )

    # Build sufficiency summary for prompt
    sufficient_count = sum(
        1 for sq in sub_questions
        if critic_results.get(sq, {}).get("decision") == "sufficient"
    )
    total_count = len(sub_questions)

    prompt_context = (
        f"Original question: {original_query}\n\n"
        f"Evidence coverage: {sufficient_count}/{total_count} sub-questions "
        f"have sufficient evidence.\n\n"
        f"Retrieved evidence:\n{evidence_text}"
    )

    output = None
    last_error = None

    for attempt in range(2):
        try:
            if attempt == 0:
                messages = [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(
                        content=f"Synthesize an answer:\n\n{prompt_context}"
                    )
                ]
            else:
                print(f"  Retrying synthesizer (attempt 2)...")
                time.sleep(1)
                messages = [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=(
                        f"Synthesize answer. Return ONLY valid JSON with "
                        f"answer and citations fields. No markdown.\n\n{prompt_context}"
                    ))
                ]

            response = llm.invoke(messages)
            raw = response.content.strip()
            parsed = _parse_synthesizer_response(raw)
            output = _validate_synthesizer_output(parsed, retrieved_chunks)
            clean_answer, gap_text = _ensure_gaps_acknowledged(output.answer, sub_questions, critic_results)
            print(f"  POST-GAP answer sample: {repr(clean_answer[:300])}")
            output = SynthesizerOutput(answer=clean_answer, citations=output.citations)
            break

        except Exception as e:
            last_error = f"{type(e).__name__}: {str(e)}"
            print(f"  Synthesizer attempt {attempt + 1} failed [{type(e).__name__}]: {e}")
            if "rate_limit" in str(e).lower() or "429" in str(e):
                handle_rate_limit(e)
            elif attempt < 1:
                time.sleep(1)

    # Fallback
    if output is None:
        print(f"  Synthesizer fallback — minimal answer")
        print(f"  Last error: {last_error}")

        output = SynthesizerOutput(
            answer=(
                f"Unable to synthesize a complete answer for: {original_query}. "
                f"Evidence was retrieved but synthesis failed. "
                f"Please try rephrasing the question."
            ),
            citations=[]
        )

    # Build trace entry
    trace_entry = {
        "step": "synthesize",
        "iteration": retrieval_iterations,
        "answer_length": len(output.answer),
        "citation_count": len(output.citations),
        "citations": output.citations,
        "coverage": f"{sufficient_count}/{total_count}"
    }

    # Update state
    updated_state = dict(state)
    updated_state["draft_answer"] = output.answer
    updated_state["citations"] = output.citations
    updated_state["evidence_gaps"] = gap_text if 'gap_text' in dir() else ""

    print(f"\n  Answer ({len(output.answer)} chars, {len(output.citations)} citations):")
    print(f"  {output.answer[:300]}...")
    print(f"\n  Citations: {output.citations}")

    return updated_state


if __name__ == "__main__":
    from decomposer import run_decomposer
    from retriever_node import run_retriever_node
    from critic import run_critic
    from reformulator import run_reformulator

    test_queries = [
        # Q002 - should have sufficient evidence
        "How did rotary position embeddings introduced in RoFormer influence later work on linear attention mechanisms?",
        # Q011 - hard 3-hop, perfect recall in eval
        "What role does knowledge distillation play in making large language models more efficient?",
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

        # Decompose
        state = run_decomposer(state)

        # Retrieve
        state = run_retriever_node(state)

        # Critique
        state = run_critic(state)

        # Reformulate if needed (max 1 loop)
        if state["critic_decision"] == "insufficient":
            prev_iter = state.get("current_iteration", 0)
            state = run_reformulator(state)
            if state.get("current_iteration", 0) > prev_iter:
                state = run_retriever_node(state)
                state = run_critic(state)

        # Synthesize
        state = run_synthesizer(state)

        print(f"\n  Final answer:")
        print(f"  {state['draft_answer']}")
        print(f"\n  Citations ({len(state['citations'])}):")
        for c in state["citations"]:
            print(f"    [{c}]")
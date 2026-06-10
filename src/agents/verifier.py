import os
import sys
import json
import re
import time
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage, HumanMessage

from llm_client import get_llm
from agent_utils import handle_rate_limit, parse_json_robust

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")

sys.path.insert(0, str(BASE_DIR / "src" / "agents"))

from schemas import VerifierOutput
from state import AgentState

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

MAX_CHUNK_CHARS = 600
MAX_CLAIMS = 5
MAX_CHUNKS_PER_PAPER = 3

GAP_KEYWORDS = [
    "insufficient", "not found", "no evidence", "not available",
    "could not find", "not well-represented", "not fully explored",
    "remains insufficiently", "not well-covered", "not addressed"
]

SYSTEM_PROMPT = """You are a scientific claim verifier for a paper retrieval system.

Your job is to check whether each factual claim in a synthesized answer is supported by the cited chunks.

You will receive:
- The original answer with inline citations like [arxiv_id]
- For each claim: the claim text, ALL its citations, and the text of each cited chunk

For each claim, check:
1. Does ANY of the cited chunks contain evidence for this specific claim?
2. Is the claim faithful to what the chunks say (not overstated or distorted)?
3. A claim with multiple citations is supported if ANY one citation supports it

Verification categories:
- supported: claim is semantically supported by at least one cited chunk
  (the chunk contains information that logically supports the claim,
  even if not word-for-word identical)
- weak: claim goes slightly beyond what chunks say but is reasonable inference
- unsupported: claim contradicts chunks or introduces facts with no basis in any chunk

Rules:
- Only verify claims that have explicit [arxiv_id] citations
- Do NOT verify gap acknowledgment sentences (intentionally uncited)
- A claim is SUPPORTED if ANY chunk semantically supports it
- A claim is only UNSUPPORTED if it introduces facts not inferable from ANY chunk
- Do NOT mark as unsupported just because exact wording differs from chunk text
- Reasonable inferences from chunk content count as supported
- Prefer "weak" over "unsupported" for any claim that is a reasonable paraphrase of chunk content
- Only mark "unsupported" if the claim is clearly fabricated or contradicts the chunks
- Be specific in your reasoning
- Return ONLY valid JSON, no preamble, no markdown

Return this exact structure:
{
  "claims": [
    {
      "claim_index": 0,
      "claim": "exact claim text",
      "citations": ["arxiv_id_1", "arxiv_id_2"],
      "status": "supported",
      "reason": "specific explanation of which citation supports it"
    }
  ],
  "verification_status": "passed",
  "verified_answer": "answer text with unsupported claims flagged or removed"
}

verification_status rules:
- "passed": all claims supported
- "needs_revision": some claims weak but none unsupported
- "failed": one or more claims completely unsupported"""


def _clean_unsupported_markers(text: str) -> str:
    """Remove all variants of UNSUPPORTED markers from answer text."""
    # Handle parenthesis variant: (UNSUPPORTED: ...) or (UNSUPPORTED CLAIM: ...)
    text = re.sub(r'\(UNSUPPORTED[^)]*\)', '', text)
    # Handle bracket variant: [UNSUPPORTED], [/UNSUPPORTED], [UNSUPPORTED CLAIM: ...]
    text = re.sub(r'\[/?UNSUPPORTED[^\]]*\]', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _extract_claims(answer: str) -> list[dict]:
    """
    Extract claims from answer text.
    One entry per sentence, with ALL citations for that sentence grouped together.
    Skips gap acknowledgment sentences.
    Deduplicates identical sentences.
    """
    sentences = re.split(r'(?<=[.!?])\s+', answer.strip())

    claims = []
    seen_sentences = set()

    for sentence in sentences:
        citations = re.findall(r'\[\s*(\d{4}\.\d{4,5})\s*\]', sentence)
        if not citations:
            continue

        # Skip gap acknowledgment sentences
        if any(kw in sentence.lower() for kw in GAP_KEYWORDS):
            continue

        sentence_key = sentence.strip()

        # Deduplicate identical sentences
        if sentence_key in seen_sentences:
            continue
        seen_sentences.add(sentence_key)

        # One claim per sentence with all unique citations grouped
        unique_citations = list(dict.fromkeys(citations))
        claims.append({
            "claim": sentence_key,
            "citations": unique_citations
        })

    return claims[:MAX_CLAIMS]


def _build_evidence_for_claims(
    claims: list[dict],
    retrieved_chunks: dict,
    max_chars: int = MAX_CHUNK_CHARS
) -> str:
    """
    Build evidence text for all cited arxiv_ids across all claims.
    Collects ALL hybrid chunks per paper to reduce false negatives.
    """
    needed_ids = set()
    for c in claims:
        needed_ids.update(c["citations"])

    chunk_text_map = {}
    for chunks in retrieved_chunks.values():
        for chunk in chunks:
            if chunk.arxiv_id not in needed_ids:
                continue
            if chunk.retrieval_method != "hybrid":
                continue
            if not chunk.text:
                continue

            if chunk.arxiv_id not in chunk_text_map:
                chunk_text_map[chunk.arxiv_id] = {
                    "title": chunk.title,
                    "year": chunk.year,
                    "texts": []
                }

            if len(chunk_text_map[chunk.arxiv_id]["texts"]) < MAX_CHUNKS_PER_PAPER:
                chunk_text_map[chunk.arxiv_id]["texts"].append(
                    chunk.text[:300].strip()
                )

    lines = ["Cited chunk evidence (multiple chunks per paper where available):"]

    for arxiv_id in sorted(needed_ids):
        if arxiv_id in chunk_text_map:
            entry = chunk_text_map[arxiv_id]
            combined = " [...] ".join(entry["texts"])
            if len(combined) > max_chars:
                combined = combined[:max_chars] + "..."
            lines.append(
                f"\n[{arxiv_id}] {entry['title'][:60]} ({entry['year']})"
                f"\nText: {combined}"
            )
        else:
            lines.append(f"\n[{arxiv_id}] — chunk text not available")

    return "\n".join(lines)


def _parse_verifier_response(raw: str) -> dict:
    """Parse LLM response to JSON with robust fallback strategies."""
    return parse_json_robust(raw)


def _validate_verifier_output(
    parsed: dict,
    draft_answer: str,
    retrieved_chunks: dict = None
) -> VerifierOutput:
    """
    Validate verifier output.
    - unsupported claims where cited paper IS in corpus → downgrade to weak
    - unsupported claims where cited paper NOT in corpus → keep as unsupported
    - weak claims only → needs_revision
    - all supported → passed
    - empty verified_answer → use draft
    """
    raw_claims = parsed.get("claims", [])
    status = parsed.get("verification_status", "needs_revision")
    verified_answer = parsed.get("verified_answer", "").strip()

    # Normalize status
    valid_statuses = {"passed", "needs_revision", "failed"}
    if status not in valid_statuses:
        status = "needs_revision"

    # Build set of valid arxiv IDs from retrieved chunks
    valid_arxiv_ids = set()
    if retrieved_chunks:
        for chunks in retrieved_chunks.values():
            for chunk in chunks:
                valid_arxiv_ids.add(chunk.arxiv_id)

    # Categorize claims — downgrade unsupported → weak if paper IS in corpus
    unsupported = []
    weak = []

    for c in raw_claims:
        claim_status = c.get("status", "supported")
        claim_text = c.get("claim", "")[:100]
        claim_citations = c.get("citations", [])

        if claim_status == "unsupported":
            if valid_arxiv_ids and all(cit in valid_arxiv_ids for cit in claim_citations):
                # Paper IS in corpus — verifier was too strict, downgrade to weak
                weak.append(claim_text)
            else:
                # Paper not in corpus — genuinely unsupported
                unsupported.append(claim_text)
        elif claim_status == "weak":
            weak.append(claim_text)

    # Enforce correct status
    if unsupported:
        status = "failed"
    elif weak:
        status = "needs_revision"
    else:
        status = "passed"

    # Use draft answer if verifier returned empty
    # For passed/needs_revision: keep full draft answer (don't strip weak claims)
    # For failed: use verifier's cleaned answer with unsupported claims removed
    if status in ("passed", "needs_revision"):
        verified_answer = draft_answer
    elif not verified_answer:
        verified_answer = draft_answer
    else:
        verified_answer = _clean_unsupported_markers(verified_answer)

    # unsupported_claims stores both unsupported AND weak claims
    return VerifierOutput(
        verification_status=status,
        unsupported_claims=unsupported + weak,
        verified_answer=verified_answer
    )


def _check_no_citation_answer(answer: str) -> tuple[str, str]:
    """
    Check if an answer with no citations has factual content.
    Returns (status, note).
    """
    sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
    non_gap = [
        s for s in sentences
        if not any(kw in s.lower() for kw in GAP_KEYWORDS)
        and len(s.strip()) > 20
    ]

    if non_gap:
        return "needs_revision", "factual content present but no citations found"
    return "passed", "answer consists only of gap acknowledgments"


def run_verifier(state: AgentState) -> AgentState:
    """
    Verify claims in the synthesized answer against cited chunks.

    Strategy:
    - Extract claims grouped by sentence (all citations per sentence together)
    - Show verifier ALL chunks per cited paper
    - Claim supported if ANY citation supports it
    - unsupported where paper in corpus → downgraded to weak
    - unsupported → failed, weak → needs_revision, all good → passed
    - No-citation answers with factual content → needs_revision
    - Retry once on parse failure
    - Fall back to passing draft answer through on total failure

    Updates state with:
    - verified_answer
    - verification_status
    - unsupported_claims
    - reasoning_trace entry
    """
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not found in .env")

    # Clean all UNSUPPORTED marker variants from draft answer
    draft_answer = _clean_unsupported_markers(state.get("draft_answer", ""))
    retrieved_chunks = state.get("retrieved_chunks", {})
    retrieval_iterations = state.get("retrieval_iterations", 0)

    if not draft_answer:
        print("  WARNING: No draft answer to verify")
        updated = dict(state)
        updated["verified_answer"] = ""
        updated["verification_status"] = "failed"
        updated["unsupported_claims"] = []
        return updated

    # Extract claims — one per sentence, all citations grouped
    claims = _extract_claims(draft_answer)
    print(f"\n  Verifying {len(claims)} cited claims...")

    # Handle no-citation case
    if not claims:
        status, note = _check_no_citation_answer(draft_answer)
        print(f"  No cited claims found — {note}")
        updated = dict(state)
        updated["verified_answer"] = draft_answer
        updated["verification_status"] = status
        updated["unsupported_claims"] = []
        updated["reasoning_trace"] = state.get("reasoning_trace", []) + [{
            "step": "verify",
            "iteration": retrieval_iterations,
            "claims_checked": 0,
            "verification_status": status,
            "note": note
        }]
        return updated

    # Build evidence — all chunks per cited paper
    evidence_text = _build_evidence_for_claims(claims, retrieved_chunks)

    # Format claims for prompt — one entry per sentence with all citations
    claims_text = "\n".join([
        f"{i}. Claim: \"{c['claim'][:200]}\"\n"
        f"   Citations: {c['citations']}"
        for i, c in enumerate(claims)
    ])

    prompt_content = (
        f"Answer to verify:\n{draft_answer}\n\n"
        f"Claims to check:\n{claims_text}\n\n"
        f"{evidence_text}"
    )

    llm = get_llm("verifier", temperature=0.0, max_tokens=1500)

    output = None
    last_error = None

    for attempt in range(2):
        try:
            if attempt == 0:
                messages = [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(
                        content=f"Verify these claims:\n\n{prompt_content}"
                    )
                ]
            else:
                print(f"  Retrying verifier (attempt 2)...")
                time.sleep(3)
                messages = [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=(
                        f"Verify claims. Return ONLY valid JSON. "
                        f"No markdown.\n\n{prompt_content}"
                    ))
                ]

            response = llm.invoke(messages)
            raw = response.content.strip()
            parsed = _parse_verifier_response(raw)

            # Validate we got actual claim evaluations
            if not parsed.get("claims") and len(claims) > 0:
                raise ValueError("Verifier returned no claim evaluations")

            output = _validate_verifier_output(parsed, draft_answer, retrieved_chunks)
            break

        except Exception as e:
            last_error = f"{type(e).__name__}: {str(e)}"
            print(f"  Verifier attempt {attempt + 1} failed [{type(e).__name__}]: {e}")
            if "rate_limit" in str(e).lower() or "429" in str(e):
                handle_rate_limit(e)
            elif attempt < 1:
                time.sleep(3)

    # Fallback
    if output is None:
        print(f"  Verifier fallback — passing draft answer unchanged")
        print(f"  Last error: {last_error}")
        output = VerifierOutput(
            verification_status="needs_revision",
            unsupported_claims=[],
            verified_answer=draft_answer
        )

    # Build trace entry
    trace_entry = {
        "step": "verify",
        "iteration": retrieval_iterations,
        "claims_checked": len(claims),
        "verification_status": output.verification_status,
        "unsupported_count": len(output.unsupported_claims),
        "unsupported_claims": output.unsupported_claims
    }

    # Update state
    updated_state = dict(state)
    updated_state["verified_answer"] = output.verified_answer
    updated_state["verification_status"] = output.verification_status
    updated_state["unsupported_claims"] = output.unsupported_claims
    updated_state["reasoning_trace"] = state.get("reasoning_trace", []) + [trace_entry]

    # Print summary
    status_icon = (
        "[OK]" if output.verification_status == "passed"
        else "[REV]" if output.verification_status == "needs_revision"
        else "[FAIL]"
    )
    print(f"  {status_icon} Verification: {output.verification_status.upper()}")
    print(f"  Claims checked: {len(claims)}")
    if output.unsupported_claims:
        print(f"  Flagged claims ({len(output.unsupported_claims)}):")
        for claim in output.unsupported_claims:
            print(f"    - {claim[:80]}")

    return updated_state


if __name__ == "__main__":
    from decomposer import run_decomposer
    from retriever_node import run_retriever_node
    from critic import run_critic
    from reformulator import run_reformulator
    from synthesizer import run_synthesizer

    test_queries = [
        "What role does knowledge distillation play in making large language models more efficient?",
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
            prev_iter = state.get("current_iteration", 0)
            state = run_reformulator(state)
            if state.get("current_iteration", 0) > prev_iter:
                state = run_retriever_node(state)
                state = run_critic(state)

        state = run_synthesizer(state)
        state = run_verifier(state)

        print(f"\n  Verification: {state['verification_status']}")
        print(f"\n  Verified answer:")
        print(f"  {state['verified_answer'][:500]}...")
        if state["unsupported_claims"]:
            print(f"\n  Flagged:")
            for c in state["unsupported_claims"]:
                print(f"    - {c[:80]}")
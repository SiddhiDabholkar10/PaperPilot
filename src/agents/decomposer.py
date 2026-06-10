import os
import json
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from llm_client import get_llm


from langchain_core.messages import SystemMessage, HumanMessage

from agent_utils import parse_json_robust


BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")

sys.path.insert(0, str(BASE_DIR / "src" / "agents"))
from schemas import DecomposerOutput
from state import AgentState




SYSTEM_PROMPT = """You are a research question decomposer for a scientific paper retrieval system.

Your job is to break a complex multi-hop research question into 1-4 focused sub-questions.
Each sub-question should target a specific paper, concept, or relationship.

Rules:
- Return 1 sub-question ONLY for simple definition queries (e.g. "what is X formula")
- Return 2-4 sub-questions for ALL other queries — causal, comparison, bridge, aggregation
- When in doubt, decompose into 2 sub-questions
- Each sub-question must be self-contained and independently searchable
- Use specific technical terminology from the original question
- Do NOT repeat the same concept across sub-questions
- Return ONLY valid JSON, no preamble, no markdown, no explanation

Question types:
- bridge: A influenced B which influenced C (follow a chain)
- comparison: How does A differ from B
- aggregation: What are all examples of X across multiple papers
- causal: Why did X lead to Y
- definition: What is X and how does it work

Return ONLY this exact JSON structure:
{
  "sub_questions": ["question1", "question2"],
  "question_type": "bridge"
}

---

EXAMPLES:

Example 1 — bridge query, needs decomposition:
Query: "How did Bahdanau attention influence the Transformer architecture and what did BERT build on top of that?"
Output:
{
  "sub_questions": [
    "What did Bahdanau attention mechanism introduce and how does it align encoder-decoder representations?",
    "How does the Transformer architecture use attention differently from Bahdanau attention?",
    "What specific components did BERT inherit from the Transformer encoder?"
  ],
  "question_type": "bridge"
}

Example 2 — comparison query, needs decomposition:
Query: "What are the trade-offs between LoRA and full fine-tuning for large language models?"
Output:
{
  "sub_questions": [
    "How does LoRA reduce trainable parameters during fine-tuning of large language models?",
    "What are the computational and performance trade-offs of full fine-tuning versus parameter-efficient methods?"
  ],
  "question_type": "comparison"
}

Example 3 — focused query, single sub-question:
Query: "What is the scaled dot-product attention formula in the Transformer?"
Output:
{
  "sub_questions": [
    "What is the scaled dot-product attention formula in the Transformer?"
  ],
  "question_type": "definition"
}

Example 4 — aggregation query, needs decomposition:
Query: "How have data augmentation techniques for NLP evolved from 2020 to 2024?"
Output:
{
  "sub_questions": [
    "What data augmentation techniques for NLP were introduced or prominent in 2020-2021?",
    "How did data augmentation approaches for NLP change in 2022-2023?",
    "What are the most recent data augmentation methods for NLP in 2024?"
  ],
  "question_type": "aggregation"
}"""


def _parse_decomposer_response(raw: str) -> dict:
    return parse_json_robust(raw)


def _validate_output(parsed: dict) -> DecomposerOutput:
    """
    Validate parsed dict against DecomposerOutput schema.
    Raises ValueError if invalid.
    """
    # Ensure sub_questions is a non-empty list of strings
    sqs = parsed.get("sub_questions", [])
    if not sqs or not isinstance(sqs, list):
        raise ValueError("sub_questions missing or not a list")

    # Clamp to 1-4 sub-questions
    sqs = [str(q).strip() for q in sqs if str(q).strip()][:4]
    if not sqs:
        raise ValueError("sub_questions is empty after cleaning")

    # Validate question_type first — needed for minimum sub-question check
    valid_types = {"bridge", "comparison", "aggregation", "causal", "definition"}
    qt = parsed.get("question_type", "bridge")
    if qt not in valid_types:
        qt = "bridge"  # safe fallback

    # Enforce minimum 2 sub-questions for non-definition queries
    if len(sqs) < 2 and qt != "definition":
        sqs = [
            sqs[0],
            f"What are the key mechanisms and evidence supporting: {sqs[0]}"
        ]

    return DecomposerOutput(sub_questions=sqs, question_type=qt)
def _infer_question_type(query: str) -> str:
    """Heuristic fallback when LLM fails to classify."""
    q = query.lower()
    if any(w in q for w in ["differ", "compare", "vs", "versus", "trade-off", "difference"]):
        return "comparison"
    if any(w in q for w in ["evolved", "evolution", "history", "over time", "from", "to 20"]):
        return "aggregation"
    if any(w in q for w in ["why", "cause", "led to", "result in"]):
        return "causal"
    if any(w in q for w in ["what is", "define", "definition", "formula", "how does"]):
        return "definition"
    return "bridge"  # default only when no signal

def run_decomposer(state: AgentState) -> AgentState:
    """
    Decompose original query into sub-questions.

    Strategy:
    - Call OPENROUTER LLM with system prompt + few-shot examples
    - Allow 1 sub-question for focused queries, 2-4 for complex ones
    - Retry once on JSON parse failure with stricter prompt
    - Fall back to original query as single sub-question if retry fails

    Updates state with:
    - sub_questions
    - question_type
    - reasoning_trace entry
    """
    query = state["original_query"]

    llm = get_llm("decomposer", temperature=0.1, max_tokens=512)

    output = None
    last_error = None

    for attempt in range(2):  # attempt 0 = normal, attempt 1 = retry with stricter prompt
        try:
            if attempt == 0:
                messages = [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(
                        content=f"Decompose this research question:\n\n{query}"
                    )
                ]
            else:
                # Stricter retry prompt
                print(f"  Retrying decomposer (attempt 2)...")
                messages = [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=(
                        f"Decompose this research question. "
                        f"Return ONLY a JSON object with sub_questions and question_type. "
                        f"No markdown, no explanation, no preamble.\n\n{query}"
                    ))
                ]
                time.sleep(1)

            response = llm.invoke(messages)
            raw = response.content.strip()

            parsed = _parse_decomposer_response(raw)
            output = _validate_output(parsed)
            break  # success

        except Exception as e:
            last_error = f"{type(e).__name__}: {str(e)}"
            print(f"  Decomposer attempt {attempt + 1} failed [{type(e).__name__}]: {e}")

    # Graceful degradation if both attempts failed
    if output is None:
        print(f"  Decomposer fallback — using original query as single sub-question")
        print(f"  Last error: {last_error}")
        output = DecomposerOutput(
            sub_questions=[query],
            question_type=_infer_question_type(query)
        )

    # Build trace entry
    trace_entry = {
        "step": "decompose",
        "original_query": query,
        "sub_questions": output.sub_questions,
        "question_type": output.question_type,
        "sub_question_count": len(output.sub_questions)
    }

    # Update state — only write fields this node owns
    updated_state = dict(state)
    updated_state["sub_questions"] = output.sub_questions
    updated_state["question_type"] = output.question_type
    updated_state["reasoning_trace"] = state.get("reasoning_trace", []) + [trace_entry]

    print(f"  [{output.question_type}] Decomposed into {len(output.sub_questions)} sub-question(s):")
    for i, sq in enumerate(output.sub_questions):
        print(f"    {i+1}. {sq}")

    return updated_state


if __name__ == "__main__":
    test_queries = [
        # Bridge — needs decomposition
        "How did rotary position embeddings introduced in RoFormer influence later work on linear attention mechanisms?",
        # Comparison — needs decomposition
        "What are the trade-offs between LoRA and full fine-tuning for parameter efficiency?",
        # Aggregation — needs decomposition
        "How have data augmentation techniques for NLP evolved from 2020 to 2024?",
        # Focused — should return 1 sub-question
        "What is the scaled dot-product attention formula?",
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

        result = run_decomposer(state)

        print(f"\n  Type:     {result['question_type']}")
        print(f"  Count:    {len(result['sub_questions'])}")
        for sq in result["sub_questions"]:
            print(f"  → {sq}")
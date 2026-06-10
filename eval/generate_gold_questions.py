import os
import sys
import json
import time
import sqlite3
import random
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

sys.path.insert(0, str(BASE_DIR / "src" / "agents"))
from agent_utils import parse_json_robust

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DB_PATH = BASE_DIR / "data" / "paperpilot.db"

# Target question counts per type
QUESTION_TARGETS = {
    "bridge": 25,       # A influenced B which influenced C
    "comparison": 25,   # How does A differ from B
    "aggregation": 15,  # What are all examples of X
    "causal": 15,       # Why did X lead to Y
    "definition": 20,   # What is X and how does it work
}
TOTAL_TARGET = sum(QUESTION_TARGETS.values())  # 100

SYSTEM_PROMPT = """You are a research question generator for a scientific paper retrieval system.

Your job is to generate realistic multi-hop research questions that require reading specific papers to answer.

You will receive:
- A question type
- 2-3 paper abstracts with their arxiv IDs and titles
- Instructions for the question type

Rules:
1. The question MUST require ALL provided papers to answer fully
2. The question should be natural — something a researcher would actually ask
3. Do NOT mention paper titles or arxiv IDs in the question
4. The question should be specific enough to have a definite answer
5. Return ONLY valid JSON, no preamble, no markdown

Return this exact structure:
{
  "question": "the research question",
  "question_type": "bridge|comparison|aggregation|causal|definition",
  "required_papers": ["arxiv_id_1", "arxiv_id_2"],
  "difficulty": "easy|medium|hard",
  "reasoning": "why this question requires all the listed papers"
}"""

TYPE_INSTRUCTIONS = {
    "bridge": """Generate a BRIDGE question — a chain of influence or development.
Example: "How did method X from paper A influence the design of method Y in paper B?"
The question should trace how concepts flow from one paper to another.""",

    "comparison": """Generate a COMPARISON question — contrasting two approaches.
Example: "What are the trade-offs between approach X and approach Y?"
The question should require understanding both papers to compare them.""",

    "aggregation": """Generate an AGGREGATION question — collecting examples across papers.
Example: "What techniques have been proposed for X across different works?"
The question should require information from all papers to give a complete answer.""",

    "causal": """Generate a CAUSAL question — explaining why something led to something else.
Example: "What limitations of X led to the development of Y?"
The question should require understanding the motivation and solution across papers.""",

    "definition": """Generate a DEFINITION question — explaining a concept and its application.
Example: "What is X and how has it been applied to Y?"
The question should require one paper for the concept and another for the application.""",
}


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def fetch_papers_with_abstracts(min_citations: int = 0) -> list[dict]:
    """Fetch all papers with abstracts from the database."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT arxiv_id, title, abstract, year, citation_count,
               s2_tldr, influential_citation_count
        FROM papers
        WHERE abstract IS NOT NULL
        AND LENGTH(abstract) > 100
        ORDER BY citation_count DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def find_related_paper_pairs(
    papers: list[dict],
    n_pairs: int = 200
) -> list[tuple[dict, dict]]:
    """
    Find pairs of papers likely to be related by topic.
    Uses simple keyword overlap on titles and abstracts.
    """
    # Key topic clusters for pairing
    topic_keywords = {
        "attention": ["attention", "transformer", "self-attention", "multi-head"],
        "distillation": ["distillation", "student", "teacher", "compression"],
        "fine-tuning": ["fine-tuning", "lora", "peft", "adaptation", "finetuning"],
        "retrieval": ["retrieval", "rag", "augmented generation", "knowledge base"],
        "generation": ["generation", "language model", "llm", "gpt", "inference"],
        "alignment": ["alignment", "rlhf", "safety", "reward", "preference"],
        "efficiency": ["efficiency", "inference", "latency", "throughput", "kv-cache"],
        "multilingual": ["multilingual", "cross-lingual", "transfer", "language"],
        "interpretability": ["interpretability", "mechanistic", "circuit", "neuron"],
        "reasoning": ["reasoning", "chain-of-thought", "cot", "problem solving"],
        "augmentation": ["augmentation", "data augmentation", "synthetic"],
        "embeddings": ["embedding", "representation", "positional", "encoding"],
        "continual": ["continual", "catastrophic forgetting", "lifelong", "sequential"],
        "audio": ["audio", "speech", "acoustic", "wav2vec", "sound"],
        "mixture": ["mixture of experts", "moe", "routing", "sparse"],
    }

    # Assign each paper to topics
    paper_topics = {}
    for p in papers:
        text = (p["title"] + " " + (p["abstract"] or "")).lower()
        topics = []
        for topic, keywords in topic_keywords.items():
            if any(kw in text for kw in keywords):
                topics.append(topic)
        if topics:
            paper_topics[p["arxiv_id"]] = topics

    # Find pairs sharing at least one topic
    pairs = []
    arxiv_ids = list(paper_topics.keys())
    random.shuffle(arxiv_ids)

    paper_map = {p["arxiv_id"]: p for p in papers}

    seen_pairs = set()
    for i, id1 in enumerate(arxiv_ids):
        for id2 in arxiv_ids[i+1:]:
            if len(pairs) >= n_pairs:
                break
            topics1 = set(paper_topics[id1])
            topics2 = set(paper_topics[id2])
            shared = topics1 & topics2
            if shared and (id1, id2) not in seen_pairs:
                seen_pairs.add((id1, id2))
                seen_pairs.add((id2, id1))
                pairs.append((paper_map[id1], paper_map[id2]))
        if len(pairs) >= n_pairs:
            break

    return pairs


def find_paper_triples(
    papers: list[dict],
    n_triples: int = 50
) -> list[tuple[dict, dict, dict]]:
    """Find triples of related papers for harder questions."""
    pairs = find_related_paper_pairs(papers, n_pairs=500)
    paper_map = {p["arxiv_id"]: p for p in papers}

    triples = []
    seen = set()

    for p1, p2 in pairs:
        if len(triples) >= n_triples:
            break
        # Find a third paper related to either p1 or p2
        text1 = (p1["title"] + " " + (p1["abstract"] or "")).lower()
        text2 = (p2["title"] + " " + (p2["abstract"] or "")).lower()

        for p3 in papers:
            if p3["arxiv_id"] in {p1["arxiv_id"], p2["arxiv_id"]}:
                continue
            text3 = (p3["title"] + " " + (p3["abstract"] or "")).lower()

            # Simple overlap check
            words1 = set(text1.split())
            words3 = set(text3.split())
            overlap = len(words1 & words3)

            if overlap > 20:
                key = tuple(sorted([p1["arxiv_id"], p2["arxiv_id"], p3["arxiv_id"]]))
                if key not in seen:
                    seen.add(key)
                    triples.append((p1, p2, p3))
                    break

    return triples


def format_paper_context(papers: list[dict], max_abstract_chars: int = 400) -> str:
    """Format paper abstracts for the prompt."""
    lines = []
    for p in papers:
        abstract = (p["abstract"] or "")[:max_abstract_chars]
        tldr = p.get("s2_tldr", "") or ""
        lines.append(
            f"[{p['arxiv_id']}] {p['title']} ({p['year']})\n"
            f"Abstract: {abstract}\n"
            + (f"TL;DR: {tldr}\n" if tldr else "")
        )
    return "\n---\n".join(lines)


def generate_question(
    llm: ChatOpenAI,
    papers: list[dict],
    question_type: str,
    existing_questions: set[str]
) -> dict | None:
    """Generate one gold question from a set of papers."""
    paper_context = format_paper_context(papers)
    type_instruction = TYPE_INSTRUCTIONS[question_type]

    prompt = (
        f"Generate a {question_type.upper()} research question using these papers:\n\n"
        f"{paper_context}\n\n"
        f"Instructions: {type_instruction}\n\n"
        f"The question must require ALL {len(papers)} papers to answer fully."
    )

    for attempt in range(2):
        try:
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt)
            ]
            response = llm.invoke(messages)
            raw = response.content.strip()
            parsed = parse_json_robust(raw)

            question = parsed.get("question", "").strip()
            if not question or len(question) < 20:
                continue

            # Skip duplicates
            if question in existing_questions:
                continue

            # Validate required papers match
            required = set(parsed.get("required_papers", []))
            paper_ids = {p["arxiv_id"] for p in papers}
            if not required.issubset(paper_ids):
                # Fix required papers to only include provided ones
                parsed["required_papers"] = list(paper_ids)

            return parsed

        except Exception as e:
            if attempt == 0:
                time.sleep(2)
            else:
                print(f"    Generation failed: {e}")

    return None


def generate_gold_questions(
    target_per_type: dict = QUESTION_TARGETS,
    output_path: Path = None,
    seed: int = 42
) -> list[dict]:
    """
    Generate gold questions from the paper corpus.
    Saves incrementally to avoid losing progress on failure.
    """
    random.seed(seed)

    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not found in .env")

    if output_path is None:
        output_path = BASE_DIR / "eval" / "gold_questions_expanded.json"

    llm = ChatOpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        model="deepseek/deepseek-v4-flash",
        temperature=0.7,  # more diversity in questions
        max_tokens=512
    )

    print(f"Loading papers from database...")
    papers = fetch_papers_with_abstracts()
    print(f"  {len(papers)} papers with abstracts")

    print(f"Finding related paper pairs...")
    pairs = find_related_paper_pairs(papers, n_pairs=300)
    print(f"  {len(pairs)} related pairs found")

    print(f"Finding related paper triples...")
    triples = find_paper_triples(papers, n_triples=100)
    print(f"  {len(triples)} related triples found")

    # Load existing questions to avoid duplicates
    existing = []
    if output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)
        print(f"  Loaded {len(existing)} existing questions")

    existing_questions = {q["question"] for q in existing}
    generated = list(existing)

    # Track counts per type
    type_counts = {}
    for q in generated:
        qt = q.get("question_type", "unknown")
        type_counts[qt] = type_counts.get(qt, 0) + 1

    print(f"\nCurrent counts: {type_counts}")
    print(f"Targets: {target_per_type}")

    # Type → source mapping
    # bridge and causal benefit from triples (chain of influence)
    # comparison and definition work well with pairs
    # aggregation works well with triples
    triple_types = {"bridge", "causal", "aggregation"}
    pair_types = {"comparison", "definition"}

    total_generated = 0

    for question_type, target in target_per_type.items():
        current = type_counts.get(question_type, 0)
        needed = target - current

        if needed <= 0:
            print(f"\n{question_type}: already at target ({current}/{target})")
            continue

        print(f"\n{question_type}: generating {needed} questions (have {current}/{target})")

        # Choose source papers
        if question_type in triple_types and triples:
            sources = [(list(t), 3) for t in triples]
        else:
            sources = [([p1, p2], 2) for p1, p2 in pairs]

        random.shuffle(sources)
        generated_this_type = 0

        for paper_group, n_papers in sources:
            if generated_this_type >= needed:
                break

            result = generate_question(
                llm, paper_group, question_type, existing_questions
            )

            if result:
                question_id = f"Q{len(generated) + 1:03d}"
                gold_item = {
                    "id": question_id,
                    "question": result["question"],
                    "question_type": question_type,
                    "difficulty": result.get("difficulty", "medium"),
                    "required_papers": result.get("required_papers", [p["arxiv_id"] for p in paper_group]),
                    "reasoning": result.get("reasoning", ""),
                    "hops": n_papers
                }

                generated.append(gold_item)
                existing_questions.add(result["question"])
                type_counts[question_type] = type_counts.get(question_type, 0) + 1
                generated_this_type += 1
                total_generated += 1

                print(f"  [{generated_this_type}/{needed}] {result['question'][:70]}...")

                # Save incrementally every 10 questions
                if total_generated % 10 == 0:
                    with open(output_path, "w") as f:
                        json.dump(generated, f, indent=2)
                    print(f"  Saved {len(generated)} questions to {output_path}")

                time.sleep(0.5)  # rate limit buffer

    # Final save
    with open(output_path, "w") as f:
        json.dump(generated, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Generation complete")
    print(f"Total questions: {len(generated)}")
    print(f"By type: {type_counts}")
    print(f"Saved to: {output_path}")

    return generated


if __name__ == "__main__":
    questions = generate_gold_questions()

    # Show sample
    print(f"\nSample questions:")
    for q in questions[-5:]:
        print(f"\n[{q['id']}] {q['question_type'].upper()} ({q['difficulty']})")
        print(f"  Q: {q['question']}")
        print(f"  Papers: {q['required_papers']}")
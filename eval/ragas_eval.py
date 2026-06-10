import sys
import json
import os
import sqlite3
import warnings
import math
warnings.filterwarnings("ignore")

# ── CRITICAL: set before any ragas import ─────────────────────────────────────
os.environ.setdefault("OPENAI_API_KEY", "dummy-not-used")

from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

sys.path.insert(0, str(BASE_DIR / "src" / "agents"))

# ── RAGAS 0.1.21 imports ───────────────────────────────────────────────────────
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
)
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# ── Load agent eval results ────────────────────────────────────────────────────
results_path = BASE_DIR / "eval" / "agent_eval_results.json"
with open(results_path) as f:
    agent_results = json.load(f)

print(f"Loaded {len(agent_results)} agent eval results")

# ── Load retrieved chunk texts from SQLite ─────────────────────────────────────
def load_chunk_texts(arxiv_ids: list[str]) -> list[str]:
    if not arxiv_ids:
        return []
    db_path = BASE_DIR / "data" / "paperpilot.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    placeholders = ",".join("?" * len(arxiv_ids))
    cursor.execute(
        f"""SELECT DISTINCT arxiv_id, text
            FROM parent_chunks
            WHERE arxiv_id IN ({placeholders})
            ORDER BY arxiv_id, chunk_id
            LIMIT 20""",
        arxiv_ids
    )
    rows = cursor.fetchall()
    conn.close()
    seen = {}
    texts = []
    for row in rows:
        aid = row["arxiv_id"]
        if aid not in seen:
            seen[aid] = 0
        if seen[aid] < 3:
            text = (row["text"] or "")[:600].strip()
            if text:
                texts.append(f"[{aid}] {text}")
                seen[aid] += 1
    return texts


# ── Build RAGAS dataset ────────────────────────────────────────────────────────
print("Building RAGAS evaluation dataset...")

questions = []
answers = []
contexts_list = []
sample_ids = []
skipped = 0

for r in agent_results:
    answer = r.get("answer", "").strip()
    query = r.get("query", "").strip()
    cited_papers = r.get("cited_papers", [])

    if not answer or len(answer) < 100 or not cited_papers:
        skipped += 1
        continue

    retrieved_contexts = load_chunk_texts(cited_papers)
    if not retrieved_contexts:
        skipped += 1
        continue

    questions.append(query)
    answers.append(answer)
    contexts_list.append(retrieved_contexts)
    sample_ids.append(r["id"])

print(f"  Built {len(questions)} samples ({skipped} skipped)")

# ── Configure RAGAS LLM ────────────────────────────────────────────────────────
print("Configuring RAGAS evaluator LLM (DeepSeek V3 via OpenRouter)...")

llm = ChatOpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    model="deepseek/deepseek-chat",
    temperature=0.0,
    max_tokens=1024
)
ragas_llm = LangchainLLMWrapper(llm)

# ── Configure embeddings (BGE — no OpenAI needed) ─────────────────────────────
print("Loading BGE embeddings for answer_relevancy...")
ragas_embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True}
)

# ── Patch metrics ─────────────────────────────────────────────────────────────
faithfulness.llm = ragas_llm
answer_relevancy.llm = ragas_llm
answer_relevancy.embeddings = ragas_embeddings

# ── Run in batches ─────────────────────────────────────────────────────────────
BATCH_SIZE = 20
all_ragas_results = []

print(f"\nRunning RAGAS on {len(questions)} samples in batches of {BATCH_SIZE}...")
print(f"{'='*60}")

for batch_start in range(0, len(questions), BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, len(questions))
    print(f"\nBatch {batch_start//BATCH_SIZE + 1}: samples {batch_start+1}-{batch_end}")

    batch_data = {
        "question": questions[batch_start:batch_end],
        "answer":   answers[batch_start:batch_end],
        "contexts": contexts_list[batch_start:batch_end],
    }
    batch_ids = sample_ids[batch_start:batch_end]

    try:
        dataset = Dataset.from_dict(batch_data)
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy],
            embeddings=ragas_embeddings,
        )

        df = result.to_pandas()
        for i, qid in enumerate(batch_ids):
            if i < len(df):
                row = df.iloc[i]
                # nan → 0.0 so we can average cleanly
                faith_val = row.get("faithfulness", 0.0)
                rel_val   = row.get("answer_relevancy", 0.0)
                faith_val = 0.0 if (faith_val is None or (isinstance(faith_val, float) and math.isnan(faith_val))) else float(faith_val)
                rel_val   = 0.0 if (rel_val   is None or (isinstance(rel_val,   float) and math.isnan(rel_val)))   else float(rel_val)

                entry = {
                    "id":               qid,
                    "faithfulness":     round(faith_val, 4),
                    "answer_relevancy": round(rel_val,   4),
                }
                all_ragas_results.append(entry)
                print(f"  {qid}: faithful={faith_val:.2f}  relevancy={rel_val:.2f}")

        # Save incrementally
        out_path = BASE_DIR / "eval" / "ragas_eval_results.json"
        with open(out_path, "w") as f:
            json.dump(all_ragas_results, f, indent=2)

    except Exception as e:
        print(f"  Batch failed: {e}")
        import traceback
        traceback.print_exc()
        for qid in batch_ids:
            all_ragas_results.append({
                "id":               qid,
                "faithfulness":     0.0,
                "answer_relevancy": 0.0,
                "error":            str(e)
            })

# ── Final save ─────────────────────────────────────────────────────────────────
out_path = BASE_DIR / "eval" / "ragas_eval_results.json"
with open(out_path, "w") as f:
    json.dump(all_ragas_results, f, indent=2)

# ── Summary ────────────────────────────────────────────────────────────────────
valid = [r for r in all_ragas_results if "error" not in r]

avg_faith = sum(r["faithfulness"]     for r in valid) / len(valid) if valid else 0
avg_rel   = sum(r["answer_relevancy"] for r in valid) / len(valid) if valid else 0

print(f"\n{'='*60}")
print(f"RAGAS EVAL COMPLETE")
print(f"{'='*60}")
print(f"Samples evaluated:    {len(valid)}")
print(f"Skipped:              {skipped}")
print(f"")
print(f"Avg faithfulness:     {avg_faith:.3f}")
print(f"Avg answer relevancy: {avg_rel:.3f}")
print(f"Combined RAGAS score: {(avg_faith + avg_rel) / 2:.3f}")
print(f"\nResults saved to {out_path}")

# ── Merge into agent eval results ──────────────────────────────────────────────
print("\nMerging RAGAS scores into agent eval results...")

ragas_map = {r["id"]: r for r in all_ragas_results}

with open(results_path) as f:
    agent_results = json.load(f)

for r in agent_results:
    ragas = ragas_map.get(r["id"], {})
    r["ragas_faithfulness"]     = ragas.get("faithfulness", None)
    r["ragas_answer_relevancy"] = ragas.get("answer_relevancy", None)

merged_out = BASE_DIR / "eval" / "agent_eval_results_with_ragas.json"
with open(merged_out, "w") as f:
    json.dump(agent_results, f, indent=2)

print(f"Merged results saved to {merged_out}")
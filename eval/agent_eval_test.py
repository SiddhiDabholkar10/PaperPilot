import sys
import json
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src" / "agents"))
sys.path.insert(0, str(BASE_DIR / "src" / "retrieval"))

from graph import run_agent

# Load gold questions
gold_path = BASE_DIR / "eval" / "gold_questions.json"
with open(gold_path) as f:
    gold_questions = json.load(f)

# Test only Q005 (perfect recall in Phase 2), Q011 (perfect recall), Q014 (zero recall)
test_ids = {"Q005", "Q011", "Q014"}
test_questions = [q for q in gold_questions if q["id"] in test_ids]

print(f"Testing {len(test_questions)} queries...")

for item in test_questions:
    qid = item["id"]
    query = item["query"] if "query" in item else item["question"]
    gold_papers = set(item.get("required_papers", []))

    print(f"\n{'='*50}")
    print(f"{qid}: {query[:60]}...")
    print(f"Gold papers: {gold_papers}")

    result = run_agent(query)

    cited = set(result.get("citations", []))
    hits = cited & gold_papers
    recall = len(hits) / len(gold_papers) if gold_papers else 0.0

    print(f"\nCited:   {cited}")
    print(f"Hits:    {hits}")
    print(f"Recall:  {recall:.2f}")
    print(f"Status:  {result.get('verification_status')}")
    print(f"Perfect: {recall == 1.0}")
    time.sleep(2)
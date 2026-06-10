import os
import sys
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")

sys.path.insert(0, str(BASE_DIR / "src" / "retrieval"))
sys.path.insert(0, str(BASE_DIR / "src" / "agents"))

from retriever import HybridRetriever
from state import AgentState

INCLUDE_ABSTRACTS = True

# Iteration 0 — base search breadth
TOP_K_DENSE_BASE = 10
TOP_K_BM25_BASE = 10
TOP_K_FINAL_BASE = 10

# Iteration 1+ — wider upstream search on retry
TOP_K_DENSE_RETRY = 20
TOP_K_BM25_RETRY = 20
TOP_K_FINAL_RETRY = 15

# Singleton — load once, reuse across all nodes
_retriever_instance = None


def get_retriever() -> HybridRetriever:
    """
    Lazy singleton — initialize retriever once and reuse.
    Avoids reloading BGE model and BM25 index on every call.
    """
    global _retriever_instance
    if _retriever_instance is None:
        print("  Initializing HybridRetriever (first call)...")
        _retriever_instance = HybridRetriever()
    return _retriever_instance


def run_retriever_node(state: AgentState) -> AgentState:
    """
    Retrieve chunks for each sub-question.

    Strategy:
    - Run hybrid retrieval per sub-question independently
    - Use reformulated queries if available (after critic loop)
    - top_k=10 with abstracts for broader coverage
    - Track retrieval iteration count for budget control

    Updates state with:
    - retrieved_chunks (dict: sub_question → list[RetrievedChunk])
    - retrieval_iterations (incremented)
    - reasoning_trace entry per sub-question
    """
    retriever = get_retriever()

    sub_questions = state.get("sub_questions", [])
    reformulated_queries = state.get("reformulated_queries", {})
    retrieval_iterations = state.get("retrieval_iterations", 0)

    # Adaptive top_k — broader search on retry iterations
    if retrieval_iterations == 0:
        top_k_dense = TOP_K_DENSE_BASE
        top_k_bm25 = TOP_K_BM25_BASE
        top_k_final = TOP_K_FINAL_BASE
    else:
        top_k_dense = TOP_K_DENSE_RETRY
        top_k_bm25 = TOP_K_BM25_RETRY
        top_k_final = TOP_K_FINAL_RETRY

    print(f"\n  Retrieval iteration {retrieval_iterations + 1} | dense={top_k_dense} bm25={top_k_bm25} final={top_k_final}")

    if not sub_questions:
        print("  WARNING: No sub-questions in state, skipping retrieval")
        return state

    # Start from existing retrieved_chunks if re-retrieving
    retrieved_chunks = dict(state.get("retrieved_chunks", {}))
    trace_entries = []

    
    print(f"  Sub-questions to retrieve: {len(sub_questions)}")

    for sq in sub_questions:
        # Use reformulated query if available for this sub-question
        query = reformulated_queries.get(sq, sq)
        is_reformulated = sq in reformulated_queries

        if is_reformulated:
            print(f"  [reformulated] {query[:70]}")
        else:
            print(f"  [original]     {query[:70]}")

        try:
            previous_chunks = retrieved_chunks.get(sq, [])
            chunks = retriever.retrieve(
                query=query,
                top_k=top_k_final,
                top_k_dense=top_k_dense,
                top_k_bm25=top_k_bm25,
                use_parent=True,
                include_abstracts=INCLUDE_ABSTRACTS
            )
            
            # Fall back to previous if new retrieval returns nothing
            if not chunks and previous_chunks:
                print(f"    WARNING: empty retrieval, keeping previous {len(previous_chunks)} chunks")
                chunks = previous_chunks
            elif is_reformulated and previous_chunks:
                # Union original + reformulated results by chunk_id
                previous_map = {c.chunk_id: c for c in previous_chunks}
                new_map = {c.chunk_id: c for c in chunks}
                merged = {**previous_map, **new_map}
                # Re-sort by RRF score so critic sees best evidence first
                chunks = sorted(
                    merged.values(),
                    key=lambda c: c.rrf_score,
                    reverse=True
                )
                print(f"    → merged {len(previous_map)} previous + {len(new_map)} new = {len(chunks)} unique chunks (re-ranked)")

            # Store under original sub-question key for critic consistency
            retrieved_chunks[sq] = chunks

            # Summarize retrieved papers for trace
            papers_found = list({c.arxiv_id for c in chunks if c.retrieval_method == "hybrid"})
            abstract_papers = list({c.arxiv_id for c in chunks if c.retrieval_method == "abstract_supplement"})

            print(f"    → {len(chunks)} chunks | {len(papers_found)} papers | {len(abstract_papers)} abstracts")

            trace_entries.append({
                "step": "retrieve",
                "iteration": retrieval_iterations + 1,
                "sub_question": sq,
                "query_used": query,
                "is_reformulated": is_reformulated,
                "chunks_found": len(chunks),
                "papers_found": papers_found,
                "abstract_papers": abstract_papers
            })

        except Exception as e:
            print(f"  ERROR retrieving for sub-question: {e}")
            retrieved_chunks[sq] = retrieved_chunks.get(sq, [])  # keep previous on error
            trace_entries.append({
                "step": "retrieve",
                "iteration": retrieval_iterations + 1,
                "sub_question": sq,
                "query_used": query,
                "error": str(e),
                "chunks_found": 0,
                "papers_found": []
            })


    # ── Cross-encoder rerank ───────────────────────────────────────────────────
    # After all sub-questions are retrieved, rerank each sub-question's
    # chunk pool using the cross-encoder for more accurate relevance scoring.
    # Only rerank if we have more chunks than the final top_k.
    # try:
    #     from reranker import get_reranker
    #     reranker = get_reranker()
    #     original_query = state.get("original_query", "")

    #     for sq in sub_questions:
    #         chunks = retrieved_chunks.get(sq, [])
    #         if len(chunks) > top_k_final:
    #             # Convert chunks to dicts for reranker
    #             chunk_dicts = [
    #                 {
    #                     "chunk_id": c.chunk_id,
    #                     "arxiv_id": c.arxiv_id,
    #                     "title": c.title,
    #                     "year": c.year,
    #                     "text": c.text or "",
    #                     "section": getattr(c, "section", ""),
    #                     "rrf_score": c.rrf_score,
    #                     "retrieval_method": c.retrieval_method,
    #                     "citation_count": getattr(c, "citation_count", 0),
    #                 }
    #                 for c in chunks
    #             ]

    #             # Rerank using the sub-question (more specific than original query)
    #             reranked_dicts = reranker.rerank(
    #                 query=sq,
    #                 chunks=chunk_dicts,
    #                 top_k=top_k_final,
    #                 text_field="text",
    #                 score_field="rerank_score"
    #             )

    #             # Convert back to RetrievedChunk objects
    #             # Keep original chunk objects but in reranked order
    #             reranked_ids = {d["chunk_id"]: d["rerank_score"] for d in reranked_dicts}
    #             reranked_chunks = [
    #                 c for c in chunks
    #                 if c.chunk_id in reranked_ids
    #             ]
    #             reranked_chunks.sort(
    #                 key=lambda c: reranked_ids.get(c.chunk_id, 0.0),
    #                 reverse=True
    #             )
    #             retrieved_chunks[sq] = reranked_chunks

    #     print(f"  Cross-encoder reranking applied")

    # except Exception as e:
    #     print(f"  Reranker unavailable — skipping ({e})")
    #     # Pipeline continues without reranking — graceful degradation

    # ── Update state ───────────────────────────────────────────────────────────
    updated_state = dict(state)
    updated_state["retrieved_chunks"] = retrieved_chunks
    updated_state["retrieval_iterations"] = retrieval_iterations + 1
    updated_state["reasoning_trace"] = state.get("reasoning_trace", []) + trace_entries

    this_iteration_chunks = sum(
        len(retrieved_chunks.get(sq, []))
        for sq in sub_questions
    )
    print(f"\n  Chunks retrieved this iteration: {this_iteration_chunks} across {len(sub_questions)} sub-questions")
    return updated_state



if __name__ == "__main__":
   
    from decomposer import run_decomposer

    test_queries = [
        "How did rotary position embeddings introduced in RoFormer influence later work on linear attention mechanisms?",
        "What are the trade-offs between LoRA and full fine-tuning for parameter efficiency?",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")

        # Start state
        state: AgentState = {
            "original_query": query,
            "max_iterations": 3,
            "current_iteration": 0,
            "reasoning_trace": []
        }

        # Step 1: Decompose
        state = run_decomposer(state)

        # Step 2: Retrieve
        state = run_retriever_node(state)

        # Show results
        print(f"\n  Results per sub-question:")
        for sq, chunks in state["retrieved_chunks"].items():
            print(f"\n  Sub-Q: {sq[:60]}")
            print(f"  Chunks: {len(chunks)}")
            seen = set()
            for c in chunks:
                if c.arxiv_id not in seen:
                    seen.add(c.arxiv_id)
                    print(f"    [{c.rrf_score:.4f}] {c.title[:50]} ({c.year})")
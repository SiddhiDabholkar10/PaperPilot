"""
Cross-encoder reranker for PaperPilot.
Reranks candidate chunks using a cross-encoder model that scores
query-document pairs with full attention (query sees document).

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
- 22M params, fast on CPU (~50ms per pair)
- Trained on MS-MARCO passage ranking
- Returns a relevance score for each (query, chunk) pair
"""

import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MAX_LENGTH = 512


class CrossEncoderReranker:
    """
    Reranks a list of chunks for a given query using a cross-encoder.
    
    Usage:
        reranker = CrossEncoderReranker()
        reranked = reranker.rerank(query, chunks, top_k=10)
    """

    def __init__(self, model_name: str = MODEL_NAME, device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"  Loading cross-encoder: {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"  Cross-encoder ready")

    def score(self, query: str, texts: list[str], batch_size: int = 32) -> list[float]:
        """
        Score a list of texts for a given query.
        Returns a list of float scores (higher = more relevant).
        """
        if not texts:
            return []

        all_scores = []

        for batch_start in range(0, len(texts), batch_size):
            batch_texts = texts[batch_start:batch_start + batch_size]

            # Build (query, text) pairs
            pairs = [[query, t] for t in batch_texts]

            # Tokenize
            features = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt"
            )
            features = {k: v.to(self.device) for k, v in features.items()}

            # Score
            with torch.no_grad():
                scores = self.model(**features).logits.squeeze(-1)
                all_scores.extend(scores.cpu().tolist())

        return all_scores

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int = 10,
        text_field: str = "text",
        score_field: str = "rerank_score"
    ) -> list[dict]:
        """
        Rerank a list of chunk dicts by relevance to the query.

        Args:
            query:       The search query
            chunks:      List of chunk dicts, each must have text_field
            top_k:       Number of top chunks to return
            text_field:  Field name containing chunk text
            score_field: Field name to write the rerank score into

        Returns:
            Top-k chunks sorted by rerank score descending,
            each with score_field added.
        """
        if not chunks:
            return []

        # Extract texts
        texts = [c.get(text_field, "") for c in chunks]

        # Score all chunks
        scores = self.score(query, texts)

        # Attach scores and sort
        scored = []
        for chunk, score in zip(chunks, scores):
            chunk_copy = dict(chunk)
            chunk_copy[score_field] = float(score)
            scored.append(chunk_copy)

        scored.sort(key=lambda x: x[score_field], reverse=True)

        return scored[:top_k]


# ── Singleton for reuse across agent nodes ────────────────────────────────────
_reranker_instance: CrossEncoderReranker | None = None


def get_reranker() -> CrossEncoderReranker:
    """Return a singleton reranker instance."""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = CrossEncoderReranker()
    return _reranker_instance
from pydantic import BaseModel, Field
from typing import Literal


class DecomposerOutput(BaseModel):
    sub_questions: list[str] = Field(
        description="1-4 targeted sub-questions that together answer the original query. Return 1 if the query is already focused.",
        min_length=1,
        max_length=4
    )
    question_type: Literal[
        "bridge", "comparison", "aggregation", "causal", "definition"
    ] = Field(
        description="Type of multi-hop reasoning required"
    )


class SubQuestionCriticResult(BaseModel):
    decision: Literal["sufficient", "insufficient"] = Field(
        description="Whether retrieved chunks sufficiently answer this sub-question"
    )
    reason: str = Field(
        description="Specific explanation of why evidence is sufficient or insufficient"
    )
    missing_evidence: list[str] = Field(
        default_factory=list,
        description="Specific concepts or claims missing from retrieved chunks"
    )


class CriticOutput(BaseModel):
    results: dict[str, SubQuestionCriticResult] = Field(
        description="Per sub-question critic results keyed by sub-question text"
    )
    overall_decision: Literal["sufficient", "insufficient"] = Field(
        description="Overall decision — insufficient if ANY sub-question is insufficient"
    )


class ReformulatorOutput(BaseModel):
    reformulated_queries: dict[str, str] = Field(
        description="Maps each insufficient sub-question to a reformulated query"
    )
    reasoning: str = Field(
        description="Brief explanation of reformulation strategy used"
    )


class SynthesizerOutput(BaseModel):
    answer: str = Field(
        description="Full answer with inline citations in [arxiv_id] format"
    )
    citations: list[str] = Field(
        description="List of arxiv_ids cited in the answer"
    )


class VerifierOutput(BaseModel):
    verification_status: Literal["passed", "needs_revision", "failed"] = Field(
        description="passed = all claims supported, needs_revision = minor issues, failed = major unsupported claims"
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Claims in the answer not supported by cited chunks"
    )
    verified_answer: str = Field(
        description="Answer with unsupported claims removed or flagged"
    )
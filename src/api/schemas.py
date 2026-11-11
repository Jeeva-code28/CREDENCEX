# src/api/schemas.py
# Pydantic v2 validation models for request and response serialization.

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class VerificationRequest(BaseModel):
    """Payload for text or URL claim verification."""
    text: Optional[str] = Field(None, description="The textual claim to verify", examples=["Bill Gates announced a new vaccine on January 15, 2024"])
    url: Optional[str] = Field(None, description="URL of a news article to verify", examples=["https://example.com/news"])
    max_evidence: Optional[int] = Field(None, description="Max evidence sources to retrieve and rank", examples=[5])
    bypass_cache: bool = Field(False, description="If True, bypasses rumor cache")
    async_mode: bool = Field(False, description="If True, returns a job_id immediately and runs in the background")


class EvidenceItem(BaseModel):
    """Single piece of retrieved evidence."""
    source: str
    title: Optional[str] = None
    text: str
    url: Optional[str] = None
    score: Optional[float] = None
    scores_breakdown: Optional[Dict[str, float]] = None
    is_official_primary: Optional[bool] = False
    authority_score: Optional[float] = None
    source_tier: Optional[str] = "general_web"


class NLIProbabilities(BaseModel):
    """NLI classification probabilities."""
    entailment: float = 0.0
    neutral: float = 0.0
    contradiction: float = 0.0


class NLIResultSchema(BaseModel):
    """Aggregated NLI result."""
    primary_verdict: str
    confidence: float
    weighted_probabilities: NLIProbabilities


class VerdictResponse(BaseModel):
    """Full verification response payload."""
    claim: str
    final_verdict: str
    final_confidence: float
    verdict: Optional[str] = None
    confidence: Optional[float] = None
    text_verdict: Optional[str] = None
    text_confidence: Optional[float] = 0.0
    image_verdict: Optional[str] = None
    image_confidence: Optional[float] = 0.0
    evidence_count: int = 0
    top_evidence: List[EvidenceItem] = []
    evidence: Optional[List[Dict[str, Any]]] = []
    probabilities: Optional[Dict[str, float]] = None
    entities: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = 0.0
    authenticity_score: Optional[float] = None
    context_score: Optional[float] = None
    extracted_ocr: Optional[Dict[str, Any]] = None
    detected_authorities: Optional[List[Dict[str, Any]]] = None
    primary_origins: Optional[List[Dict[str, Any]]] = []
    reasoning: Optional[str] = ""
    claimant: Optional[str] = None
    category: Optional[str] = None
    report: str = ""
    truth_analysis: Optional[Dict[str, Any]] = None
    forensics: Optional[Dict[str, Any]] = None
    ela_heatmap: Optional[str] = None
    cached: bool = False
    timestamp: str


class JobStatusResponse(BaseModel):
    """Response when polling background verification jobs."""
    job_id: str
    status: str = Field(..., description="'queued', 'processing', 'completed', or 'failed'")
    message: Optional[str] = None
    result: Optional[VerdictResponse] = None
    created_at: str
    completed_at: Optional[str] = None


class HealthResponse(BaseModel):
    """System health and warm model status."""
    status: str
    device: str
    models_loaded: Dict[str, bool]
    cache_entries: int
    uptime_seconds: float


class SourceCredibilityItem(BaseModel):
    """Evidence source details."""
    source: str
    domain: Optional[str] = None
    category: Optional[str] = None
    credibility_score: float
    description: str

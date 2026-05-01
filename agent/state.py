from typing import TypedDict, Optional, List, Literal
from pydantic import BaseModel, Field

class SanitizerState(TypedDict):
    raw_query: str
    extracted: Optional[dict]
    validated: bool
    errors: List[str]
    corrections: List[str]
    retry_count: int
    llm_city_guess: Optional[str] 
    confidence: float
    source: Optional[str]
    awaiting_user: bool
    pending_issue: Optional[str]
    hitl_candidates: Optional[List[str]]
    user_selection: Optional[str]
    resume_node: Optional[Literal["extractor", "validator", "corrector", "llm_correction", "llm_city_corrector_from_list"]]
    status: Optional[str]
    score: float

class HITLDecision(TypedDict):
    should_interrupt: bool
    reason: Optional[str]
    candidates: Optional[List[str]]
    resume_node: Optional[str]

class LLMCorrection(BaseModel):
    city: Optional[str]

class ExtractedInfo(BaseModel):
    city: Optional[str] = Field(default=None, description="City name")
    country: Optional[str] = Field(default=None, description="Country name")
    intent: Optional[str] = Field(default=None, description="Intent of the query")
    confidence: float = Field(default=0.0, description="Confidence score 0-1")
    source: Optional[str] = Field(default=None, description="Source of the extraction/correction")

from pydantic import BaseModel, Field
from typing import List

class CandidateScore(BaseModel):
    candidate_name: str
    overall_score: float = Field(ge=0, le=100)
    recommendation: str
    summary: str
    strengths: List[str]
    gaps: List[str]
    evidence: List[str]
    matched_requirements: List[str]
    missing_requirements: List[str]

class InterviewQuestions(BaseModel):
    candidate_name: str
    questions: List[str]
    evaluation_focus: List[str]

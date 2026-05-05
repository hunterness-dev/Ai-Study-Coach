from pydantic import BaseModel, Field, field_validator


class StudyLogRequest(BaseModel):
    subject: str = Field(..., min_length=1, max_length=128, examples=["Mathematics"])
    hours: float = Field(..., gt=0, le=24, examples=[2.5])
    score: float = Field(..., ge=0, le=100, examples=[78.5])

    @field_validator("subject")
    @classmethod
    def normalise_subject(cls, v: str) -> str:
        return v.strip().title()


class StudyLogResponse(BaseModel):
    id: int
    subject: str
    hours: float
    score: float

    model_config = {"from_attributes": True}


class HealthResponse(BaseModel):
    status: str
    service: str


class PlanResponse(BaseModel):
    allocations: dict[str, float]
    predicted_scores: dict[str, float]
    features: dict[str, dict]
    source: str
    error: str | None = None

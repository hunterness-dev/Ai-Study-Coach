from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from api.schemas import HealthResponse, PlanResponse, StudyLogRequest, StudyLogResponse
from core.config import get_settings
from services.plan_service import generate_plan
from services.tracking_service import create_log
from utils.database import get_db

router = APIRouter()
settings = get_settings()


@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check() -> HealthResponse:
    return HealthResponse(status="ok", service=settings.app_name)


@router.post(
    "/log",
    response_model=StudyLogResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Tracking"],
)
def log_study_session(
    payload: StudyLogRequest,
    db: Session = Depends(get_db),
) -> StudyLogResponse:
    try:
        entry = create_log(db, payload.subject, payload.hours, payload.score)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    return StudyLogResponse.model_validate(entry)


@router.get("/plan", response_model=PlanResponse, tags=["Planning"])
def get_study_plan(db: Session = Depends(get_db)) -> PlanResponse:
    result = generate_plan(db)
    if "error" in result and len(result) == 1:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=result["error"],
        )
    return PlanResponse(**result)

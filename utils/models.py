from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String

from utils.database import Base


class StudyLog(Base):
    __tablename__ = "study_logs"

    id = Column(Integer, primary_key=True, index=True)
    subject = Column(String(128), nullable=False, index=True)
    hours = Column(Float, nullable=False)
    score = Column(Float, nullable=False)
    logged_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self) -> str:
        return (
            f"<StudyLog id={self.id} subject={self.subject!r} "
            f"hours={self.hours} score={self.score}>"
        )

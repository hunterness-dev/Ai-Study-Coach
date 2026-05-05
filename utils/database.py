import os

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from core.config import get_settings
from core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class Base(DeclarativeBase):
    pass


def _ensure_data_dir() -> None:
    db_path = settings.database_url.replace("sqlite:///", "")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)


_ensure_data_dir()

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False},
    echo=(settings.app_env == "development"),
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    from utils import models  # noqa: F401 — ensures models are registered
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialised at %s", settings.database_url)


def get_db():
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()

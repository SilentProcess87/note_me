from __future__ import annotations

from pathlib import Path
from typing import Optional

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session as DbSession, sessionmaker

from .models import Base

_engine = None
_SessionLocal: Optional[sessionmaker] = None


def _migrate(engine) -> None:
    """Add any missing columns to existing tables (lightweight auto-migration)."""
    insp = inspect(engine)
    # TranscriptSegment.speaker was added after v1.0
    if "transcript_segments" in insp.get_table_names():
        cols = {c["name"] for c in insp.get_columns("transcript_segments")}
        if "speaker" not in cols:
            with engine.begin() as conn:
                conn.execute(text(
                    "ALTER TABLE transcript_segments ADD COLUMN speaker VARCHAR(20) DEFAULT ''"
                ))


def init_db(storage_path: Path) -> None:
    global _engine, _SessionLocal
    storage_path.mkdir(parents=True, exist_ok=True)
    db_path = storage_path / "noteme.db"
    _engine = create_engine(f"sqlite:///{db_path}", echo=False, connect_args={"check_same_thread": False})
    Base.metadata.create_all(_engine)
    _migrate(_engine)
    _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)


def get_db() -> DbSession:
    if _SessionLocal is None:
        raise RuntimeError("Database not initialized — call init_db() first.")
    return _SessionLocal()

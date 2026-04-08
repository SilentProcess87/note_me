from __future__ import annotations

import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Session(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(255), nullable=False)
    start_time = Column(DateTime, default=datetime.datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    language = Column(String(10), default="auto")
    audio_path = Column(String(512), nullable=True)
    mode = Column(String(20), default="meeting")  # meeting | speech_coach

    segments = relationship(
        "TranscriptSegment",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="TranscriptSegment.start_sec",
    )
    summaries = relationship(
        "Summary",
        back_populates="session",
        cascade="all, delete-orphan",
    )
    qa_entries = relationship(
        "QAEntry",
        back_populates="session",
        cascade="all, delete-orphan",
    )


class TranscriptSegment(Base):
    __tablename__ = "transcript_segments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    start_sec = Column(Float, nullable=False)
    end_sec = Column(Float, nullable=False)
    text = Column(Text, nullable=False)
    language = Column(String(10), nullable=True)
    confidence = Column(Float, nullable=True)

    session = relationship("Session", back_populates="segments")


class Summary(Base):
    __tablename__ = "summaries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    summary_text = Column(Text, nullable=False)
    action_items = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    ollama_model = Column(String(100), nullable=True)

    session = relationship("Session", back_populates="summaries")


class QAEntry(Base):
    __tablename__ = "qa_entries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    session = relationship("Session", back_populates="qa_entries")

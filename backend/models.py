# backend/models.py
import datetime as dt
from sqlalchemy import Column, Integer, String, Float, DateTime
from backend.db import Base, engine

class InferenceResult(Base):
    __tablename__ = "inference_results"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    media_type = Column(String, nullable=False)  # image|video
    label = Column(String, nullable=False)       # real|fake
    probability = Column(Float, nullable=False)  # prob of predicted label
    n_frames_used = Column(Integer, nullable=False, default=1)
    model_version = Column(String, nullable=False)
    created_at = Column(DateTime, default=dt.datetime.utcnow)

def init_db():
    Base.metadata.create_all(bind=engine)

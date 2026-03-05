from datetime import datetime
from .db import db

class TrainingRun(db.Model):
    __tablename__ = "training_runs"

    id = db.Column(db.Integer, primary_key=True)
    model_name = db.Column(db.String(100), nullable=False)
    dataset = db.Column(db.String(100), nullable=False)

    accuracy = db.Column(db.Float, nullable=True)
    loss = db.Column(db.Float, nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
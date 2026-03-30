# ANSVAR: Her definerer du dine databaser-tabeller som Python klasser.
# TrainingRun = opskrift på tabellen training_runs.

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


class TitanicPassenger(db.Model):
    __tablename__ = "titanic_passengers"

    id = db.Column(db.Integer, primary_key=True)
    passenger_id = db.Column(db.Integer, nullable=False)
    survived = db.Column(db.Integer, nullable=False)
    pclass = db.Column(db.Integer, nullable=False)
    sex = db.Column(db.String, nullable=False)
    age = db.Column(db.Float, nullable=True)
    sibsp = db.Column(db.Integer, nullable=True)
    parch = db.Column(db.Integer, nullable=True)
    fare = db.Column(db.Float, nullable=True)
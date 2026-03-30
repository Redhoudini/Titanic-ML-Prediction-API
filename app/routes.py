# ANSVAR: Endpoints:
# / (hjemmeside/portal)
# /health, /db-ping (tests)
# /runs (viser runs)
# /api/runs (POST: gemmer en ny run i DB)

from flask import Blueprint, jsonify, render_template, request
from .models import TrainingRun
from sqlalchemy import text
from .db import db
import joblib

model = joblib.load("ml/models/titanic_mlp_800_mean_age_features6.pkl")
scaler = joblib.load("ml/models/titanic_scaler_mean_age_features6.pkl")

bp = Blueprint("main", __name__)

@bp.get("/health")
def health():
    return jsonify(status="ok")

@bp.get("/db-ping")
def db_ping():
    db.session.execute(text("SELECT 1"))
    return jsonify(db="ok")

@bp.get("/")
def index():
    return render_template("index.html")

@bp.get("/runs")
def runs_page():
    runs = TrainingRun.query.order_by(TrainingRun.created_at.desc()).limit(50).all()
    items = "".join([
        f"<li>#{r.id} {r.model_name} {r.dataset} acc={r.accuracy} loss={r.loss}</li>"
        for r in runs
    ])
    return f"<h1>Runs</h1><ul>{items}</ul><p><a href='/'>Tilbage</a></p>"

@bp.post("/api/runs")
def create_run():
    data = request.get_json(force=True)

    run = TrainingRun(
        model_name=data["model_name"],
        dataset=data["dataset"],
        accuracy=data.get("accuracy"),
        loss=data.get("loss"),
    )
    db.session.add(run)
    db.session.commit()

    return jsonify(id=run.id), 201

@bp.post("/api/predict")
def predict():
    data = request.get_json()

    if not data or "pclass" not in data or "sex" not in data or "age" not in data:
        return jsonify({
            "error": "JSON must contain pclass, sex and age"
        }), 400

    try:
        pclass = int(data["pclass"])
        sex_raw = str(data["sex"]).strip().lower()
        age = float(data["age"])

        sibsp = int(data.get("sibsp", 0))
        parch = int(data.get("parch", 0))
        fare = float(data.get("fare", 32.0))
    except (ValueError, TypeError):
        return jsonify({
            "error": "Invalid input types"
        }), 400

    if sex_raw not in ["male", "female"]:
        return jsonify({
            "error": "sex must be 'male' or 'female'"
        }), 400

    sex = 1 if sex_raw == "male" else 0

    sample = [[pclass, sex, age, sibsp, parch, fare]]
    sample_scaled = scaler.transform(sample)

    prediction = model.predict(sample_scaled)[0]
    probability = model.predict_proba(sample_scaled)[0][1]

    return jsonify({
        "prediction": int(prediction),
        "survival_probability": float(probability)
    })
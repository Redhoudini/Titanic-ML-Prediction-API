# ANSVAR: Dine endpoints:
# / (hjemmeside/portal)
# /health, /db-ping (tests)
# /runs (viser runs)
# /api/runs (POST: gemmer en ny run i DB)

from flask import Blueprint, jsonify, render_template, request
from .models import TrainingRun
from sqlalchemy import text
from .db import db

bp = Blueprint("main", __name__)

@bp.get("/health")
def health():
    return jsonify(status="ok")

@bp.get("/db-ping")
def db_ping():
    # Tester bare at DB kan svare
    db.session.execute(text("SELECT 1"))
    return jsonify(db="ok")

@bp.get("/")
def index():
    return render_template("index.html")

@bp.get("/runs")
def runs_page():
    runs = TrainingRun.query.order_by(TrainingRun.created_at.desc()).limit(50).all()
    # super simpelt HTML-output uden ekstra template:
    items = "".join([f"<li>#{r.id} {r.model_name} {r.dataset} acc={r.accuracy} loss={r.loss}</li>" for r in runs])
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
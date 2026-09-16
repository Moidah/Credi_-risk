import os

import joblib
import pandas as pd
import psycopg2
import psycopg2.extras
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

MODEL_DIR = os.getenv("MODEL_DIR", "/opt/models")
DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "postgres"),
    "port": os.getenv("POSTGRES_PORT", "5432"),
    "dbname": os.getenv("POSTGRES_DB", "credit_risk"),
    "user": os.getenv("POSTGRES_USER", "airflow"),
    "password": os.getenv("POSTGRES_PASSWORD", "airflow"),
}


def get_db():
    return psycopg2.connect(**DB_CONFIG)


def load_model_artifacts():
    rf = joblib.load(os.path.join(MODEL_DIR, "random_forest.joblib"))
    feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.joblib"))
    return rf, feature_cols


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/metrics/latest")
def latest_metrics():
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        """
        SELECT model_name, auc_roc, n_records, default_rate, trained_at
        FROM model_runs
        ORDER BY trained_at DESC
        LIMIT 10
        """
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return jsonify(rows)


@app.route("/api/feature-importance/latest")
def latest_feature_importance():
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        """
        SELECT fi.feature_name, fi.importance
        FROM feature_importance fi
        JOIN model_runs mr ON fi.model_run_id = mr.id
        WHERE mr.id = (SELECT id FROM model_runs ORDER BY trained_at DESC LIMIT 1)
        ORDER BY fi.importance DESC
        LIMIT 15
        """
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return jsonify(rows)


@app.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json()
    rf, feature_cols = load_model_artifacts()

    df = pd.DataFrame([payload])
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_cols]

    proba = rf.predict_proba(df)[0][1]
    label = int(proba >= 0.5)

    return jsonify({"default_probability": round(float(proba), 4), "predicted_label": label})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

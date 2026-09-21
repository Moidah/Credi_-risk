"""
Pipeline de entrenamiento - Predicción de Default Crediticio
Dataset: UCI Default of Credit Card Clients (30,000 registros, ~22% default real)
Compara Random Forest vs Regresión Logística y guarda el mejor modelo.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

DATA_PATH = os.getenv("DATA_PATH", "/opt/data/credit_default.csv")
MODEL_DIR = os.getenv("MODEL_DIR", "/opt/models")
os.makedirs(MODEL_DIR, exist_ok=True)


def load_and_engineer_features(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Target binario
    df["target"] = (df["default"] == "yes").astype(int)

    # Codificar categóricas
    for col in ["gender", "education", "marital_status"]:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    timeliness_cols = [c for c in df.columns if c.startswith("timeliness_")]
    for col in timeliness_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Features de negocio derivadas
    balance_cols = [f"balance_{i}" for i in range(1, 7)]
    payment_cols = [f"payment_{i}" for i in range(1, 7)]

    df["utilization_ratio"] = df["avg_balance"] / df["credit_limit"].replace(0, np.nan)
    df["utilization_ratio"] = df["utilization_ratio"].fillna(0).clip(0, 5)
    df["payment_to_balance_ratio"] = (
        df["avg_payment"] / df["avg_balance"].replace(0, np.nan)
    ).fillna(0).clip(0, 5)
    df["balance_trend"] = df[balance_cols[0]] - df[balance_cols[-1]]
    df["late_payment_count"] = df[timeliness_cols].gt(
        df[timeliness_cols].median().median()
    ).sum(axis=1)

    feature_cols = (
        ["credit_limit", "gender", "education", "marital_status", "age"]
        + timeliness_cols
        + balance_cols
        + payment_cols
        + [
            "avg_balance",
            "avg_payment",
            "utilization_ratio",
            "payment_to_balance_ratio",
            "balance_trend",
            "late_payment_count",
        ]
    )
    return df, feature_cols


def train():
    df, feature_cols = load_and_engineer_features(DATA_PATH)
    X = df[feature_cols]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # Modelo baseline: Regresión Logística
    logreg = LogisticRegression(max_iter=1000, class_weight="balanced")
    logreg.fit(X_train_scaled, y_train)
    logreg_proba = logreg.predict_proba(X_test_scaled)[:, 1]
    logreg_auc = roc_auc_score(y_test, logreg_proba)
    results["logistic_regression"] = {
        "auc_roc": round(logreg_auc, 4),
        "report": classification_report(
            y_test, logreg.predict(X_test_scaled), output_dict=True
        ),
    }

    # Modelo principal: Random Forest
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    rf_auc = roc_auc_score(y_test, rf_proba)
    results["random_forest"] = {
        "auc_roc": round(rf_auc, 4),
        "report": classification_report(
            y_test, rf.predict(X_test), output_dict=True
        ),
    }

    # Importancia de variables (Random Forest gana -> usamos su interpretabilidad)
    importances = sorted(
        zip(feature_cols, rf.feature_importances_), key=lambda x: -x[1]
    )
    results["feature_importance"] = [
        {"feature": f, "importance": round(float(i), 4)} for f, i in importances
    ]
    results["default_rate"] = round(float(y.mean()), 4)
    results["n_records"] = int(len(df))
    results["best_model"] = "random_forest" if rf_auc > logreg_auc else "logistic_regression"

    # Guardar artefactos
    joblib.dump(rf, os.path.join(MODEL_DIR, "random_forest.joblib"))
    joblib.dump(logreg, os.path.join(MODEL_DIR, "logistic_regression.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    joblib.dump(feature_cols, os.path.join(MODEL_DIR, "feature_cols.joblib"))
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"Random Forest AUC-ROC: {rf_auc:.4f}")
    print(f"Logistic Regression AUC-ROC: {logreg_auc:.4f}")
    print(f"Mejor modelo: {results['best_model']}")
    return results


if __name__ == "__main__":
    train()

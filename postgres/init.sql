-- Esquema para el proyecto de riesgo crediticio

CREATE TABLE IF NOT EXISTS credit_applications (
    id SERIAL PRIMARY KEY,
    credit_limit NUMERIC,
    gender VARCHAR(10),
    education VARCHAR(20),
    marital_status VARCHAR(20),
    age INT,
    avg_balance NUMERIC,
    avg_payment NUMERIC,
    raw_payload JSONB,
    loaded_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    application_id INT REFERENCES credit_applications(id),
    model_name VARCHAR(50),
    default_probability NUMERIC,
    predicted_label INT,
    predicted_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS model_runs (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(50),
    auc_roc NUMERIC,
    n_records INT,
    default_rate NUMERIC,
    trained_at TIMESTAMP DEFAULT NOW(),
    metrics_json JSONB
);

CREATE TABLE IF NOT EXISTS feature_importance (
    id SERIAL PRIMARY KEY,
    model_run_id INT REFERENCES model_runs(id),
    feature_name VARCHAR(100),
    importance NUMERIC
);

CREATE INDEX IF NOT EXISTS idx_predictions_app ON predictions(application_id);
CREATE INDEX IF NOT EXISTS idx_model_runs_trained_at ON model_runs(trained_at);

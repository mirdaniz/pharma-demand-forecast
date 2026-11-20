CREATE TABLE IF NOT EXISTS sales_daily (
    datum DATE PRIMARY KEY,
    n02be DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    target_date DATE NOT NULL,
    predicted_value DOUBLE PRECISION NOT NULL,
    model_name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
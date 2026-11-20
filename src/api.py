import os
from datetime import date, timedelta
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, text

from src.features import MIN_RIWAYAT, fitur_untuk_prediksi

ROOT = Path(__file__).resolve().parent.parent
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg:///pharma_forecast")
NAMA_MODEL = "linear_regression_n02be"

engine = create_engine(DATABASE_URL)
model = joblib.load(ROOT / "models" / "model_n02be.joblib")

app = FastAPI(title="Pharma Demand Forecast API")


class PredictRequest(BaseModel):
    tanggal: date


class PredictResponse(BaseModel):
    tanggal: date
    prediksi: float
    nama_model: str


def ambil_riwayat(tanggal):
    awal = tanggal - timedelta(days=MIN_RIWAYAT)
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT datum, n02be FROM sales_daily "
                "WHERE datum >= :awal AND datum < :akhir ORDER BY datum"
            ),
            {"awal": awal, "akhir": tanggal},
        ).all()
    return pd.Series(
        [r[1] for r in rows],
        index=pd.DatetimeIndex([r[0] for r in rows]),
        dtype=float,
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    riwayat = ambil_riwayat(req.tanggal)

    try:
        X = fitur_untuk_prediksi(riwayat, req.tanggal)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    prediksi = round(max(0.0, float(model.predict(X)[0])), 2)

    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO predictions (target_date, predicted_value, model_name) "
                "VALUES (:tanggal, :prediksi, :nama_model)"
            ),
            {"tanggal": req.tanggal, "prediksi": prediksi, "nama_model": NAMA_MODEL},
        )

    return PredictResponse(tanggal=req.tanggal, prediksi=prediksi, nama_model=NAMA_MODEL)


@app.get("/predictions")
def daftar_prediksi(limit: int = 20):
    limit = min(max(limit, 1), 100)
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT id, target_date, predicted_value, model_name, created_at "
                "FROM predictions ORDER BY id DESC LIMIT :limit"
            ),
            {"limit": limit},
        ).mappings().all()
    return [dict(r) for r in rows]

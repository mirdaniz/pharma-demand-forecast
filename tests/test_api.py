from contextlib import contextmanager

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import src.api as api


class FakeConn:
    def __init__(self):
        self.inserts = []

    def execute(self, stmt, params=None):
        self.inserts.append(params)


class FakeEngine:
    def __init__(self):
        self.conn = FakeConn()

    @contextmanager
    def begin(self):
        yield self.conn


class ModelNegatif:
    def predict(self, X):
        return [-5.0]


def riwayat_lengkap(tanggal):
    idx = pd.date_range(end=pd.Timestamp(tanggal) - pd.Timedelta(days=1), periods=30, freq="D")
    return pd.Series(30.0, index=idx)


def riwayat_bolong(tanggal):
    return riwayat_lengkap(tanggal).iloc[:-5]


@pytest.fixture
def fake_engine(monkeypatch):
    fake = FakeEngine()
    monkeypatch.setattr(api, "engine", fake)
    return fake


@pytest.fixture
def client(fake_engine):
    return TestClient(api.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_sukses_dan_tersimpan(client, fake_engine, monkeypatch):
    monkeypatch.setattr(api, "ambil_riwayat", riwayat_lengkap)
    r = client.post("/predict", json={"tanggal": "2019-10-09"})
    assert r.status_code == 200
    body = r.json()
    assert body["tanggal"] == "2019-10-09"
    assert body["prediksi"] >= 0
    assert len(fake_engine.conn.inserts) == 1


def test_predict_riwayat_bolong_ditolak(client, fake_engine, monkeypatch):
    monkeypatch.setattr(api, "ambil_riwayat", riwayat_bolong)
    r = client.post("/predict", json={"tanggal": "2019-10-20"})
    assert r.status_code == 422
    assert "tidak lengkap" in r.json()["detail"]
    assert fake_engine.conn.inserts == []


def test_predict_tanggal_tidak_valid(client):
    r = client.post("/predict", json={"tanggal": "bukan-tanggal"})
    assert r.status_code == 422


def test_prediksi_negatif_dipotong_ke_nol(client, monkeypatch):
    monkeypatch.setattr(api, "ambil_riwayat", riwayat_lengkap)
    monkeypatch.setattr(api, "model", ModelNegatif())
    r = client.post("/predict", json={"tanggal": "2019-10-09"})
    assert r.status_code == 200
    assert r.json()["prediksi"] == 0.0

import numpy as np
import pandas as pd
import pytest

from src.features import FITUR, buat_fitur_historis, fitur_untuk_prediksi, is_libur_tetap


@pytest.fixture
def sales():
    rng = np.random.default_rng(42)
    idx = pd.date_range("2018-11-01", "2019-03-31", freq="D")
    return pd.Series(rng.uniform(5, 80, len(idx)), index=idx)


@pytest.mark.parametrize("tanggal", ["2019-01-02", "2019-01-07", "2019-02-14", "2019-03-31"])
def test_fitur_prediksi_sama_dengan_fitur_historis(sales, tanggal):
    t = pd.Timestamp(tanggal)
    harapan = buat_fitur_historis(sales).loc[[t], FITUR].reset_index(drop=True)
    hasil = fitur_untuk_prediksi(sales[sales.index < t], t)
    pd.testing.assert_frame_equal(hasil, harapan, check_dtype=False)


def test_riwayat_kurang_lengkap_ditolak(sales):
    riwayat = sales.drop(pd.Timestamp("2019-02-20"))
    with pytest.raises(ValueError):
        fitur_untuk_prediksi(riwayat, "2019-03-01")


def test_hanya_data_sebelum_target_yang_dipakai(sales):
    t = pd.Timestamp("2019-02-14")
    utuh = fitur_untuk_prediksi(sales, t)
    terpotong = fitur_untuk_prediksi(sales[sales.index < t], t)
    pd.testing.assert_frame_equal(utuh, terpotong)


def test_penanda_libur_tetap():
    assert is_libur_tetap("2019-01-01") == 1
    assert is_libur_tetap("2019-01-07") == 1
    assert is_libur_tetap("2019-12-19") == 1
    assert is_libur_tetap("2019-02-14") == 0

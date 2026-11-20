import pandas as pd

FITUR_NUM = ["lag_1", "lag_7", "rata_7", "rata_30", "libur_tetap", "akhir_pekan"]
FITUR_KAT = ["hari", "bulan"]
FITUR = FITUR_NUM + FITUR_KAT
MIN_RIWAYAT = 30


def is_libur_tetap(tanggal):
    t = pd.Timestamp(tanggal)
    return int((t.month == 1 and t.day in (1, 7)) or (t.month == 12 and t.day == 19))


def fitur_kalender(tanggal):
    t = pd.Timestamp(tanggal)
    hari = t.day_of_week
    return {
        "libur_tetap": is_libur_tetap(t),
        "hari": hari,
        "bulan": t.month,
        "akhir_pekan": int(hari >= 5),
    }


def buat_fitur_historis(sales):
    d = pd.DataFrame({"y": sales})
    d["lag_1"] = sales.shift(1)
    d["lag_7"] = sales.shift(7)
    d["rata_7"] = sales.shift(1).rolling(7).mean()
    d["rata_30"] = sales.shift(1).rolling(30).mean()

    idx = d.index
    d["libur_tetap"] = (
        ((idx.month == 1) & idx.day.isin([1, 7])) | ((idx.month == 12) & (idx.day == 19))
    ).astype(int)
    d["hari"] = idx.day_of_week
    d["bulan"] = idx.month
    d["akhir_pekan"] = (d["hari"] >= 5).astype(int)
    return d.dropna()


def fitur_untuk_prediksi(riwayat, tanggal_target):
    t = pd.Timestamp(tanggal_target)
    rentang = pd.date_range(end=t - pd.Timedelta(days=1), periods=MIN_RIWAYAT, freq="D")
    window = riwayat.reindex(rentang)

    if window.isna().any():
        raise ValueError(
            f"Riwayat {MIN_RIWAYAT} hari sebelum {t.date()} tidak lengkap, "
            f"{int(window.isna().sum())} tanggal kosong"
        )

    baris = {
        "lag_1": window.iloc[-1],
        "lag_7": window.iloc[-7],
        "rata_7": window.iloc[-7:].mean(),
        "rata_30": window.mean(),
        **fitur_kalender(t),
    }
    return pd.DataFrame([baris])[FITUR]
# Pharma Demand Forecast

Prediksi penjualan harian obat kategori **N02BE** (analgesik/antipiretik) satu hari ke depan, dilayani lewat API FastAPI yang mengambil riwayat dari PostgreSQL dan menyimpan setiap prediksi ke database.

Proyek ini mencakup alur lengkapnya: EDA, baseline, pemodelan, validasi deret waktu, modul fitur yang dipakai bersama oleh notebook dan API, database, API, dan test otomatis.

## Masalah

Apotek perlu memperkirakan permintaan harian supaya stok tidak kosong atau menumpuk. Di sini targetnya adalah jumlah penjualan N02BE pada hari berikutnya, dihitung dari riwayat penjualan hari-hari sebelumnya.

## Data

Dataset **Pharma Sales Data** di Kaggle: penjualan obat dari 8 kategori ATC, 2014 sampai 2019. Proyek ini memakai `salesdaily.csv`.

- 2.106 hari, dari 2 Januari 2014 sampai 8 Oktober 2019. Tidak ada tanggal yang bolong atau dobel.
- Data 2019 berhenti di Oktober, jadi tahun 2019 tidak boleh dibandingkan mentah-mentah dengan tahun lain.
- N02BE dipilih sebagai target karena hanya 1% harinya bernilai nol. N05C (68% nol) dan R03 (23% nol) dicoret.

Data tidak disertakan di repo (lihat [Lisensi dan atribusi data](#lisensi-dan-atribusi-data)). Unduh dari halaman Kaggle di atas dan taruh di folder `data/`.

## Temuan EDA

- Ada pola mingguan: Sabtu dan Minggu rata-rata sekitar 33, hari kerja sekitar 28.
- Ada musiman bulanan: penjualan tinggi di Januari, Oktober, dan Desember (sekitar 40), rendah di musim panas (sekitar 20 di Juli).
- Hari tertentu nyaris selalu nol: 1 Januari, 7 Januari, dan 19 Desember (dengan pengecualian di 2017). Pola ini dijadikan fitur `libur_tetap`.
- Ada lonjakan ekstrem, misalnya 161 pada 30 Desember 2016.

## Metode

Setiap baris memprediksi penjualan hari itu hanya dari data sebelum hari itu.

| Fitur | Keterangan |
|---|---|
| `lag_1`, `lag_7` | penjualan kemarin dan 7 hari lalu |
| `rata_7`, `rata_30` | rata-rata 7 dan 30 hari sebelumnya |
| `libur_tetap` | 1 Jan, 7 Jan, 19 Des |
| `akhir_pekan` | Sabtu atau Minggu |
| `hari`, `bulan` | kategorikal, di-one-hot |

Setelah fitur dibuat, tersisa 2.076 baris. Pembagian data:

- **Latih:** sebelum 1 Januari 2019 (1.795 hari)
- **Uji:** 1 Januari sampai 8 Oktober 2019 (281 hari)

Model yang dibandingkan: empat baseline sederhana, Linear Regression, dan Random Forest (300 pohon, tanpa tuning). Metrik utama adalah MAE.

## Hasil

### Data uji (281 hari di 2019)

| Metode | MAE | RMSE |
|---|---|---|
| Naive (kemarin) | 11,35 | 15,10 |
| Naive (minggu lalu) | 10,90 | 14,88 |
| Rata-rata 7 hari | 9,07 | 12,26 |
| Rata-rata 30 hari | 9,84 | 13,43 |
| **Linear Regression** | **8,45** | **11,01** |
| Random Forest | 8,82 | 11,50 |

### Validasi silang (TimeSeriesSplit, 5 fold, MAE)

Ini terpisah dari data uji di atas.

| Fold | Rata-rata 7 hari | Linear Regression | Random Forest |
|---|---|---|---|
| 1 | 9,08 | 9,47 | 9,66 |
| 2 | 11,13 | 10,48 | 11,16 |
| 3 | 8,41 | 8,26 | 8,83 |
| 4 | 9,26 | 8,83 | 9,22 |
| 5 | 9,51 | 8,67 | 9,06 |
| **Rata-rata** | 9,48 | **9,14** | 9,59 |

Linear Regression dipilih karena MAE-nya terendah di data uji dan di rata-rata validasi silang, dan modelnya lebih sederhana daripada Random Forest.

**Cara membacanya dengan jujur:** keunggulan atas baseline "rata-rata 7 hari" ada tetapi tipis. Di validasi silang selisihnya sekitar 0,34 MAE (3,6%), dan Linear Regression kalah di fold pertama. Di data uji selisihnya sekitar 0,62 MAE (6,8%). Menambah fitur `lag_2`, `lag_3`, dan `rata_3` tidak membantu (MAE validasi silang 9,18 dibanding 9,14).

### Di mana model salah

- Pada 277 hari berpenjualan normal di data uji, rata-rata galat absolut 8,39. Pada 4 hari penjualan nol, galatnya 12,75.
- Galat terbesar terjadi saat lonjakan: 20, 26, dan 27 Januari 2019 (aktual 93 sampai 100, prediksi 55 sampai 66). Model cenderung meremehkan puncak.
- Koefisien terbesar adalah `libur_tetap` (sekitar -41,6), jadi model sangat bergantung pada pola hari nol ini.

## API

Layanan FastAPI di `src/api.py`.

| Endpoint | Fungsi |
|---|---|
| `GET /health` | cek layanan hidup |
| `POST /predict` | prediksi N02BE untuk `{"tanggal": "YYYY-MM-DD"}`; mengambil 30 hari riwayat dari PostgreSQL, membuat fitur, memprediksi, dan menyimpan hasilnya ke tabel `predictions` |
| `GET /predictions?limit=20` | riwayat prediksi yang tersimpan (maksimal 100) |

Contoh:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"tanggal": "2019-10-09"}'
# {"tanggal":"2019-10-09","prediksi":36.03,"nama_model":"linear_regression_n02be"}
```

Perilaku penting:

- Jika riwayat 30 hari sebelum tanggal target tidak lengkap, API mengembalikan **422** dan tidak menebak.
- Prediksi dipotong minimal 0.
- Fungsi fitur yang sama (`src/features.py`) dipakai oleh notebook dan API, dan test memastikan hasilnya identik.

## Cara menjalankan

Kebutuhan: Python 3.12 dan PostgreSQL.

```bash
git clone <url-repo>
cd pharma-demand-forecast
python -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

1. Unduh dataset dari Kaggle dan taruh `salesdaily.csv` di `data/`.
2. Buat database dan isi tabelnya (jalankan dari folder utama repo; loader mengosongkan `sales_daily` lalu memuat ulang datanya dari `data/salesdaily.csv`):

   ```bash
   createdb pharma_forecast
   psql pharma_forecast -f sql/schema.sql
   python -m src.load_to_db
   ```

3. Model sudah ada di repo (`models/model_n02be.joblib`). Untuk membuatnya ulang, jalankan notebook `notebooks/eda_dan_modeling.ipynb` sampai sel terakhir.
4. Jalankan API:

   ```bash
   python -m uvicorn src.api:app --reload
   ```

   Dokumentasi interaktif ada di `http://127.0.0.1:8000/docs`. Koneksi database API bisa diatur lewat variabel `DATABASE_URL` (bawaan: `postgresql+psycopg:///pharma_forecast`). Skrip loader selalu memakai database lokal `pharma_forecast`.

Menjalankan test:

```bash
python -m pytest -v
```

Ada 12 test: 7 untuk fitur (termasuk pengecekan tidak ada kebocoran data dari masa depan) dan 5 untuk API (database dan riwayat dipalsukan, jadi test tidak menyentuh PostgreSQL asli).

## Struktur proyek

```
data/            dataset (tidak di-commit)
models/          model terlatih
notebooks/       EDA dan pemodelan
sql/schema.sql   skema database
src/             fitur, loader database, API
tests/           test fitur dan API
```

## Keterbatasan

- **Hanya satu hari ke depan.** Fitur lag memakai penjualan aktual, jadi model tidak meramal berhari-hari ke depan secara berantai.
- **Data berhenti pada 8 Oktober 2019**, jadi satu-satunya tanggal yang bisa diprediksi API adalah 9 Oktober 2019. Tanggal yang riwayat 30 harinya tidak lengkap akan ditolak.
- **Model akhir dilatih ulang pada seluruh data**, termasuk 2019. Angka data uji di atas berasal dari model yang dilatih sebelum 2019, jadi tidak berlaku persis untuk model di API.
- **Aturan `libur_tetap` dipilih setelah melihat seluruh dataset**, termasuk periode uji (1 dan 7 Januari 2019). Karena itu hasil uji bisa sedikit terlalu optimistis.
- **Satu split uji (281 hari) dan satu sumber data.** Hasil belum tentu berlaku untuk apotek atau periode lain.
- Random Forest tidak di-tuning, dan hanya dua jenis model yang dicoba.
- Lonjakan permintaan sering diremehkan, jadi hasilnya belum layak dijadikan satu-satunya dasar keputusan stok.

## Lisensi dan atribusi data

Data: **Pharma Sales Data** oleh Milan Zdravkovic (Kaggle: `milanzdravkovic`), <https://www.kaggle.com/datasets/milanzdravkovic/pharma-sales-data>, berlisensi [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Hanya untuk penggunaan nonkomersial.

Perubahan yang dilakukan pada data: konversi format tanggal, pembuatan fitur turunan (lag, rata-rata bergerak, penanda kalender), dan pemuatan ke PostgreSQL. File model di `models/` dilatih dari data tersebut.

Karena lisensi ini, dataset tidak disertakan di repo. Unduh langsung dari Kaggle.

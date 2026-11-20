from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parent.parent
engine = create_engine("postgresql+psycopg:///pharma_forecast")


def main():
    df = pd.read_csv(ROOT / "data" / "salesdaily.csv", usecols=["datum", "N02BE"])
    df["datum"] = pd.to_datetime(df["datum"], format="%m/%d/%Y").dt.date
    df = df.rename(columns={"N02BE": "n02be"}).sort_values("datum")

    with engine.begin() as conn:
        conn.execute(text("TRUNCATE sales_daily"))
        df.to_sql("sales_daily", conn, if_exists="append", index=False)

    print(f"Berhasil memuat {len(df)} baris ke sales_daily")


if __name__ == "__main__":
    main()
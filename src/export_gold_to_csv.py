"""
export_gold_to_csv.py
----------------------
Esporta il modello dimensionale GOLD (SQLite) in file CSV pronti per Power BI.

- aggiunge a fact_sales la data reale della vendita (sale_date), presa dal livello SILVER
- esclude customer_email: per le analisi non serve ed e' un dato personale (principio di minimizzazione)

Esecuzione (dalla cartella src):
    python export_gold_to_csv.py

Output: cartella powerbi/data con fact_sales.csv, dim_product.csv, dim_store.csv
"""

import sqlite3
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
GOLD_DB_PATH = BASE_DIR / "data" / "gold" / "retail.db"
SILVER_PATH = BASE_DIR / "data" / "silver" / "pos_transactions_clean.csv"
OUT_DIR = BASE_DIR / "powerbi" / "data"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(GOLD_DB_PATH)
    fact = pd.read_sql_query("SELECT * FROM fact_sales", conn)
    dim_product = pd.read_sql_query("SELECT * FROM dim_product", conn)
    dim_store = pd.read_sql_query("SELECT * FROM dim_store", conn)
    conn.close()

    silver = pd.read_csv(SILVER_PATH, usecols=["transaction_id", "transaction_date"], parse_dates=["transaction_date"])
    silver["sale_date"] = silver["transaction_date"].dt.date

    fact = fact.merge(
        silver[["transaction_id", "sale_date"]],
        left_on="sale_id", right_on="transaction_id", how="left",
    ).drop(columns=["transaction_id", "customer_email", "date_key"])

    missing_dates = int(fact["sale_date"].isna().sum())
    if missing_dates:
        raise ValueError(f"{missing_dates} vendite senza data: controlla il livello Silver")

    fact.to_csv(OUT_DIR / "fact_sales.csv", index=False)
    dim_product.to_csv(OUT_DIR / "dim_product.csv", index=False)
    dim_store.to_csv(OUT_DIR / "dim_store.csv", index=False)

    print(f"fact_sales.csv: {len(fact)} righe | date da {fact['sale_date'].min()} a {fact['sale_date'].max()}")
    print(f"dim_product.csv: {len(dim_product)} righe | dim_store.csv: {len(dim_store)} righe")
    print(f"File scritti in {OUT_DIR}")


if __name__ == "__main__":
    main()

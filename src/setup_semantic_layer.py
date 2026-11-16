"""
setup_semantic_layer.py
-----------------------
Arricchisce il database Gold (data/gold/retail.db) generato dalla pipeline ELT:
1. Ripristina in `dim_date` le colonne `full_date` (YYYY-MM-DD) e `year_month` (YYYY-MM)
   ricostruendole dalla sequenza temporale del livello Silver.
2. Crea 5 Viste SQL Semantiche (`v_gold_*`) progettate per azzerare le allucinazioni
   dell'Agente Text-to-SQL e allineare i KPI esattamente alla dashboard Power BI.
"""

from __future__ import annotations
import sqlite3
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "gold" / "retail.db"
SILVER_PATH = BASE_DIR / "data" / "silver" / "pos_transactions_clean.csv"


SEMANTIC_VIEWS_SQL = {
    "v_gold_sales_enriched": """
        CREATE VIEW v_gold_sales_enriched AS
        SELECT
            f.sale_id,
            f.date_key,
            d.full_date,
            d.year_month,
            d.year,
            d.month,
            d.day_of_week,
            d.is_weekend,
            CASE WHEN d.is_weekend = 1 THEN 'Weekend' ELSE 'Feriale' END AS day_type,
            f.store_key,
            s.store_name,
            f.product_key,
            p.sku,
            p.product_name,
            p.category,
            f.quantity,
            f.unit_price,
            ROUND(f.net_revenue, 2) AS net_revenue,
            CASE WHEN f.quantity < 0 THEN 1 ELSE 0 END AS is_return,
            f.payment_method,
            f.customer_email
        FROM fact_sales f
        JOIN dim_date d ON f.date_key = d.date_key
        JOIN dim_store s ON f.store_key = s.store_key
        JOIN dim_product p ON f.product_key = p.product_key;
    """,
    "v_gold_daily_kpi": """
        CREATE VIEW v_gold_daily_kpi AS
        SELECT
            d.date_key,
            d.full_date,
            d.year_month,
            d.day_of_week,
            d.is_weekend,
            CASE WHEN d.is_weekend = 1 THEN 'Weekend' ELSE 'Feriale' END AS day_type,
            ROUND(SUM(f.net_revenue), 2) AS daily_net_revenue,
            COUNT(f.sale_id) AS transactions_count,
            SUM(f.quantity) AS net_units_sold,
            SUM(CASE WHEN f.quantity < 0 THEN 1 ELSE 0 END) AS returns_count,
            ROUND(100.0 * SUM(CASE WHEN f.quantity < 0 THEN 1 ELSE 0 END) / COUNT(f.sale_id), 2) AS return_rate_pct,
            ROUND(AVG(f.net_revenue), 2) AS avg_transaction_value
        FROM fact_sales f
        JOIN dim_date d ON f.date_key = d.date_key
        GROUP BY d.date_key, d.full_date, d.year_month, d.day_of_week, d.is_weekend
        ORDER BY d.full_date;
    """,
    "v_gold_store_kpi": """
        CREATE VIEW v_gold_store_kpi AS
        SELECT
            s.store_key,
            s.store_name,
            ROUND(SUM(f.net_revenue), 2) AS net_revenue,
            COUNT(f.sale_id) AS transactions_count,
            SUM(f.quantity) AS net_units_sold,
            SUM(CASE WHEN f.quantity < 0 THEN 1 ELSE 0 END) AS returns_count,
            ROUND(100.0 * SUM(CASE WHEN f.quantity < 0 THEN 1 ELSE 0 END) / COUNT(f.sale_id), 2) AS return_rate_pct,
            ROUND(AVG(f.net_revenue), 2) AS avg_ticket_eur,
            ROUND(100.0 * SUM(f.net_revenue) / (SELECT SUM(net_revenue) FROM fact_sales), 2) AS revenue_share_pct
        FROM fact_sales f
        JOIN dim_store s ON f.store_key = s.store_key
        GROUP BY s.store_key, s.store_name
        ORDER BY net_revenue DESC;
    """,
    "v_gold_product_kpi": """
        CREATE VIEW v_gold_product_kpi AS
        SELECT
            p.product_name,
            p.category,
            COUNT(DISTINCT p.sku) AS sku_count,
            SUM(f.quantity) AS net_units_sold,
            ROUND(SUM(f.net_revenue), 2) AS net_revenue,
            COUNT(f.sale_id) AS transactions_count,
            SUM(CASE WHEN f.quantity < 0 THEN 1 ELSE 0 END) AS returns_count,
            ROUND(100.0 * SUM(CASE WHEN f.quantity < 0 THEN 1 ELSE 0 END) / COUNT(f.sale_id), 2) AS return_rate_pct,
            ROUND(AVG(f.unit_price), 2) AS avg_unit_price
        FROM fact_sales f
        JOIN dim_product p ON f.product_key = p.product_key
        GROUP BY p.product_name, p.category
        ORDER BY net_revenue DESC;
    """,
    "v_gold_customer_rfm": """
        CREATE VIEW v_gold_customer_rfm AS
        WITH max_db_date AS (
            SELECT MAX(full_date) AS ref_date FROM dim_date
        )
        SELECT
            f.customer_email,
            MAX(d.full_date) AS last_purchase_date,
            CAST(julianday((SELECT ref_date FROM max_db_date)) - julianday(MAX(d.full_date)) AS INTEGER) AS recency_days,
            COUNT(f.sale_id) AS frequency_tx,
            SUM(f.quantity) AS net_units,
            ROUND(SUM(f.net_revenue), 2) AS monetary_net_eur,
            ROUND(AVG(f.net_revenue), 2) AS avg_ticket_eur,
            SUM(CASE WHEN f.quantity < 0 THEN 1 ELSE 0 END) AS returns_count
        FROM fact_sales f
        JOIN dim_date d ON f.date_key = d.date_key
        WHERE f.customer_email IS NOT NULL AND TRIM(f.customer_email) != ''
        GROUP BY f.customer_email
        ORDER BY monetary_net_eur DESC;
    """,
}


def upgrade_gold_database(db_path: Path = DB_PATH, silver_path: Path = SILVER_PATH) -> None:
    conn = sqlite3.connect(db_path)
    dim_date = pd.read_sql_query("SELECT * FROM dim_date ORDER BY date_key", conn)

    if "full_date" not in dim_date.columns:
        if silver_path.exists():
            silver_df = pd.read_csv(silver_path)
            unique_dates = sorted(
                pd.to_datetime(silver_df["transaction_date"]).dt.strftime("%Y-%m-%d").unique()
            )
        else:
            unique_dates = (
                pd.date_range("2026-01-01", periods=len(dim_date), freq="D")
                .strftime("%Y-%m-%d")
                .tolist()
            )
        dim_date["full_date"] = unique_dates
        dim_date["year_month"] = dim_date["full_date"].str.slice(0, 7)
        dim_date.to_sql("dim_date", conn, if_exists="replace", index=False)

    cursor = conn.cursor()
    for view_name, create_sql in SEMANTIC_VIEWS_SQL.items():
        cursor.execute(f"DROP VIEW IF EXISTS {view_name};")
        cursor.execute(create_sql)

    # Create indexes for fast analytical queries
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_fact_date ON fact_sales(date_key);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_fact_store ON fact_sales(store_key);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_fact_product ON fact_sales(product_key);")

    conn.commit()
    conn.close()
    print(f"Database Gold aggiornato con successo in: {db_path}")


if __name__ == "__main__":
    upgrade_gold_database()

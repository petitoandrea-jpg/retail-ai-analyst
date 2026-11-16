"""
database.py
-----------
Connettore Read-Only per il database Gold (data/gold/retail.db).
Fornisce:
- Connessione SQLite sicura in sola lettura (URI mode=?mode=ro)
- Guardrail sintattico contro query distruttive (solo SELECT / WITH / PRAGMA)
- Estrazione dinamica di DDL, sample rows e valori categorici distinti per il System Prompt dell'Agente
"""

from __future__ import annotations
import re
import sqlite3
from pathlib import Path
from typing import Dict, List
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = BASE_DIR / "data" / "gold" / "retail.db"

FORBIDDEN_KEYWORDS = re.compile(
    r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|CREATE|REPLACE|ATTACH|DETACH|VACUUM|PRAGMA\s+writable_schema)\b",
    re.IGNORECASE,
)


class RetailGoldDB:
    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path).resolve()
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database non trovato in: {self.db_path}")

    def _get_ro_connection(self) -> sqlite3.Connection:
        uri = f"file:{self.db_path.as_posix()}?mode=ro"
        return sqlite3.connect(uri, uri=True, check_same_thread=False)

    @staticmethod
    def validate_sql(sql_query: str) -> str:
        clean_sql = sql_query.strip()
        clean_sql = re.sub(r"^```(?:sql)?\s*", "", clean_sql, flags=re.IGNORECASE)
        clean_sql = re.sub(r"\s*```$", "", clean_sql)
        clean_sql = clean_sql.strip().rstrip(";").strip()

        if not clean_sql:
            raise ValueError("La query SQL generata è vuota.")

        first_token = clean_sql.split()[0].upper()
        if first_token not in ("SELECT", "WITH"):
            raise ValueError(
                f"Guardrail SQL attivato: sono permesse solo query SELECT o WITH (ricevuto: {first_token})."
            )

        if FORBIDDEN_KEYWORDS.search(clean_sql):
            raise ValueError(
                "Guardrail SQL attivato: rilevata keyword di modifica dati non autorizzata."
            )

        return clean_sql + ";"

    def execute_query(self, sql_query: str) -> pd.DataFrame:
        validated_sql = self.validate_sql(sql_query)
        with self._get_ro_connection() as conn:
            return pd.read_sql_query(validated_sql, conn)

    def get_tables_and_views(self) -> List[Dict[str, str]]:
        query = """
            SELECT name, type, sql
            FROM sqlite_master
            WHERE type IN ('table', 'view')
              AND name NOT LIKE 'sqlite_%'
            ORDER BY CASE WHEN type = 'table' THEN 1 ELSE 2 END, name ASC;
        """
        with self._get_ro_connection() as conn:
            cur = conn.cursor()
            cur.execute(query)
            rows = cur.fetchall()
        return [{"name": r[0], "type": r[1], "ddl": r[2]} for r in rows]

    def build_llm_schema_context(self, sample_rows: int = 2) -> str:
        objects = self.get_tables_and_views()
        lines = ["=== SCHEMA DATABASE SQLITE GOLD (retail.db) ==="]

        with self._get_ro_connection() as conn:
            for obj in objects:
                name = obj["name"]
                obj_type = obj["type"].upper()
                ddl = obj["ddl"]
                lines.append(f"\n--- {obj_type}: {name} ---")
                lines.append(f"DDL:\n{ddl};")

                try:
                    df_sample = pd.read_sql_query(
                        f"SELECT * FROM {name} LIMIT {sample_rows};", conn
                    )
                    if not df_sample.empty:
                        lines.append(
                            f"Sample ({sample_rows} righe):\n"
                            + df_sample.to_string(index=False)
                        )
                except Exception:
                    pass

                if name.startswith("dim_"):
                    lines.append(self._extract_dimension_values(conn, name))

        return "\n".join(lines)

    def _extract_dimension_values(self, conn: sqlite3.Connection, table_name: str) -> str:
        info_df = pd.read_sql_query(f"PRAGMA table_info({table_name});", conn)
        text_cols = info_df[
            info_df["type"].str.upper().str.contains("TEXT|CHAR|VARCHAR", na=False)
        ]["name"].tolist()

        out = ["Valori distinti per filtri WHERE:"]
        for col in text_cols:
            if col == "full_date":
                min_max = pd.read_sql_query(
                    f"SELECT MIN(full_date) AS min_d, MAX(full_date) AS max_d FROM {table_name};",
                    conn,
                )
                out.append(
                    f"  - full_date: range da '{min_max['min_d'].iloc[0]}' a '{min_max['max_d'].iloc[0]}'"
                )
            else:
                vals = pd.read_sql_query(
                    f"SELECT DISTINCT {col} FROM {table_name} WHERE {col} IS NOT NULL LIMIT 20;",
                    conn,
                )[col].tolist()
                out.append(f"  - {col}: {vals}")
        return "\n".join(out)

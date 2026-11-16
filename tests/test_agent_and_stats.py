"""
test_agent_and_stats.py
-----------------------
Suite di test automatici (pytest) per verificare:
1. Guardrail di sicurezza SQL (blocco query distruttive)
2. Allineamento matematico delle Viste Semantiche con i KPI ufficiali Power BI
3. Correttezza dei test statistici in tools_stats.py
4. Risposte end-to-end dell'Agente AI
"""

import sys
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from database import RetailGoldDB
from tools_stats import RetailStatisticalAnalyst
from agent import RetailAIAnalystAgent


@pytest.fixture(scope="module")
def db():
    return RetailGoldDB(ROOT_DIR / "data" / "gold" / "retail.db")


def test_sql_guardrail_blocks_destructive_queries(db):
    with pytest.raises(ValueError):
        db.execute_query("DROP TABLE fact_sales;")
    with pytest.raises(ValueError):
        db.execute_query("DELETE FROM fact_sales WHERE quantity < 0;")
    with pytest.raises(ValueError):
        db.execute_query("SELECT * FROM dim_store; DROP TABLE dim_store;")


def test_gold_kpis_match_powerbi_exactly(db):
    df_rev = db.execute_query("SELECT ROUND(SUM(net_revenue), 2) AS tot FROM fact_sales;")
    assert df_rev["tot"].iloc[0] == 724220.44

    df_stores = db.execute_query("SELECT * FROM v_gold_store_kpi;")
    assert len(df_stores) == 4
    assert df_stores.iloc[0]["store_name"] == "Napoli Vomero"
    assert df_stores.iloc[0]["net_revenue"] == 184222.96


def test_statistical_tests_execution(db):
    analyst = RetailStatisticalAnalyst(db)
    wk_res = analyst.test_weekend_vs_weekday()
    assert wk_res["metrics"]["weekend_mean_eur"] == 2701.53
    assert wk_res["metrics"]["weekday_mean_eur"] == 2633.34
    assert wk_res["metrics"]["significant_at_005"] is False

    anova_res = analyst.test_store_anova()
    assert anova_res["metrics"]["significant_at_005"] is False


def test_agent_local_execution(db):
    agent = RetailAIAnalystAgent(db=db, provider="local")
    res = agent.ask("Nel weekend incassiamo più dei giorni feriali?")
    assert not res["dataframe"].empty
    assert res["stat_result"] is not None
    assert "Welch" in res["answer"]

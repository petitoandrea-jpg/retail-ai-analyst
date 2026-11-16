"""
app.py
------
Applicazione Web Streamlit Cloud:
"Retail AI Data Analyst & Statistical Decision Support System"
Autore: Andrea Petito
"""

from __future__ import annotations
import sys
from pathlib import Path
import streamlit as st
import pandas as pd

# Add src/ to Python path
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from database import RetailGoldDB
from setup_semantic_layer import upgrade_gold_database
from tools_stats import RetailStatisticalAnalyst
from agent import RetailAIAnalystAgent
import tools_viz as viz

st.set_page_config(
    page_title="Retail AI Data Analyst | Andrea Petito",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def init_db() -> RetailGoldDB:
    db_path = ROOT_DIR / "data" / "gold" / "retail.db"
    silver_path = ROOT_DIR / "data" / "silver" / "pos_transactions_clean.csv"
    upgrade_gold_database(db_path=db_path, silver_path=silver_path)
    return RetailGoldDB(db_path=db_path)


db = init_db()
stats_engine = RetailStatisticalAnalyst(db)

has_groq_secret = hasattr(st, "secrets") and bool(st.secrets.get("GROQ_API_KEY", ""))

# Sidebar
with st.sidebar:
    st.markdown("## 🤖 Configurazione AI Agent")
    st.caption(
        "Architettura **Medallion (Bronze → Silver → Gold)** in SQLite + **Text-to-SQL Agent** + **Inferenza Statistica (SciPy)**."
    )

    provider_choice = st.selectbox(
        "Motore di Esecuzione:",
        options=[
            "Smart Local Engine (Zero-API, SQL + SciPy Reale)",
            "Groq Cloud API (Llama-3.3-70B)",
            "OpenAI API (GPT-4o-mini)",
        ],
        index=1 if has_groq_secret else 0,
    )

    api_key = ""
    provider_code = "local"
    model_name = "local-deterministic"

    if "Groq" in provider_choice:
        provider_code = "groq"
        model_name = "llama-3.3-70b-versatile"
        default_key = st.secrets.get("GROQ_API_KEY", "") if hasattr(st, "secrets") else ""
        api_key = st.text_input("Groq API Key:", value=default_key, type="password")
    elif "OpenAI" in provider_choice:
        provider_code = "openai"
        model_name = "gpt-4o-mini"
        default_key = st.secrets.get("OPENAI_API_KEY", "") if hasattr(st, "secrets") else ""
        api_key = st.text_input("OpenAI API Key:", value=default_key, type="password")

    st.divider()
    st.markdown("### 💡 Domande Rapide per il Manager")
    sample_prompts = [
        "Confronta il fatturato dei 4 negozi e dimmi se la differenza è statisticamente significativa (ANOVA).",
        "Nel weekend incassiamo davvero più dei giorni feriali? Esegui un test t di Welch.",
        "Quali sono i migliori prodotti per fatturato e dove dobbiamo intervenire sui resi?",
        "Mostrami il trend mensile e individua i giorni di picco anomalo con lo Z-score.",
        "Analizza i clienti fidelizzati tramite RFM e individua i top spender.",
    ]
    selected_quick_prompt = None
    for idx, prompt_text in enumerate(sample_prompts):
        if st.button(prompt_text, key=f"quick_btn_{idx}", use_container_width=True):
            selected_quick_prompt = prompt_text

    st.divider()
    with st.expander("🗄️ Ispeziona Schema Database Gold"):
        st.code(db.build_llm_schema_context(sample_rows=1), language="sql")

    st.markdown(
        "**Sviluppato da:** [Andrea Petito](https://github.com/petitoandrea-jpg)  \n"
        "*Laurea in Statistica e Informatica (L-41)*"
    )

agent = RetailAIAnalystAgent(
    db=db, provider=provider_code, api_key=api_key, model_name=model_name
)

# Header & Top KPI Cards
st.title("📊 Retail AI Data Analyst & Statistical Decision System")
st.markdown(
    "Interroga in linguaggio naturale lo **Star Schema Gold (`retail.db`)** generato dalla pipeline ELT Medallion, "
    "combinando **query SQLite**, **test di ipotesi statistici in Python (`scipy.stats`)** e **grafici interattivi Plotly**."
)

kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
kpi_col1.metric("Fatturato Netto Totale", "€ 724.220,44", "+5.801 Transazioni")
kpi_col2.metric("Unità Nette Vendute", "10.456", "Scontrino Medio €124,84")
kpi_col3.metric("Tasso di Reso Globale", "5,21%", "302 Resi (χ² p=0.61)")
kpi_col4.metric("Top Punto Vendita", "Napoli Vomero", "€ 184.222,96 (25,4%)")
kpi_col5.metric("Top Prodotto", "Scarpe Pegasus", "€ 251.979,00 (34,8%)")

st.divider()

tab_chat, tab_stats, tab_bi = st.tabs(
    [
        "💬 AI Data Analyst (Chat Text-to-SQL)",
        "📐 Laboratorio Statistico Inferenziale (L-41)",
        "📈 Executive Dashboard (Replica Power BI)",
    ]
)

# TAB 1: AI Data Analyst Chat
with tab_chat:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {
                "role": "assistant",
                "content": (
                    "Ciao! Sono il tuo **AI Data Analyst**. Posso interrogare lo Star Schema `retail.db`, "
                    "eseguire test statistici di significatività (Test $t$ di Welch, ANOVA, $\\chi^2$, Outlier Z-score) "
                    "e generare grafici per il management. Clicca su una domanda nella barra laterale o scrivi qui sotto!"
                ),
            }
        ]

    for msg_idx, msg in enumerate(st.session_state.chat_history):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sql_query" in msg:
                with st.expander("🔍 Query SQL eseguita su `retail.db` & Dati Estratti", expanded=False):
                    st.code(msg["sql_query"], language="sql")
                    st.dataframe(msg["dataframe"], use_container_width=True)
            if msg.get("figure") is not None:
                st.plotly_chart(
                    msg["figure"],
                    use_container_width=True,
                    key=f"chat_hist_chart_{msg_idx}",
                )

    user_input = st.chat_input("Fai una domanda sui dati retail (es. 'Nel weekend vendiamo più dei feriali?')...")
    active_question = selected_quick_prompt or user_input

    if active_question:
        st.session_state.chat_history.append({"role": "user", "content": active_question})
        with st.chat_message("user"):
            st.markdown(active_question)

        with st.chat_message("assistant"):
            with st.spinner("Generazione SQL, interrogazione `retail.db` ed esecuzione test statistici..."):
                result = agent.ask(active_question)

            st.caption(f"⚡ Modalità: `{result['mode']}`")
            st.markdown(result["answer"])

            with st.expander("🔍 Query SQL eseguita su `retail.db` & Dati Estratti", expanded=True):
                st.code(result["sql_query"], language="sql")
                st.dataframe(result["dataframe"], use_container_width=True)

            new_msg_idx = len(st.session_state.chat_history)
            if result.get("figure") is not None:
                st.plotly_chart(
                    result["figure"],
                    use_container_width=True,
                    key=f"chat_live_chart_{new_msg_idx}",
                )

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "sql_query": result["sql_query"],
                    "dataframe": result["dataframe"],
                    "figure": result.get("figure"),
                }
            )

# TAB 2: Statistical Suite
with tab_stats:
    st.subheader("📐 Suite di Inferenza Statistica Applicata al Retail")
    st.markdown(
        "Questa sezione affianca alla Business Intelligence descrittiva i **test di verifica delle ipotesi** "
        "per distinguere il vero segnale di business dal rumore campionario."
    )

    stat_choice = st.radio(
        "Seleziona l'analisi statistica:",
        options=[
            "1. Test t di Welch & Mann-Whitney: Weekend vs Feriale",
            "2. ANOVA One-Way & Kruskal-Wallis: Confronto tra i 4 Punti Vendita",
            "3. Outlier & Peak Detection (Z-Score & IQR): Serie Storica Giornaliera",
            "4. Test Chi-Quadro (χ²) di Indipendenza: Analisi Tasso di Reso",
            "5. Segmentazione RFM Clientela & Analisi Concentrazione",
        ],
        horizontal=False,
    )

    if stat_choice.startswith("1."):
        res = stats_engine.test_weekend_vs_weekday()
        st.info(res["interpretation"])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Media Weekend", f"€ {res['metrics']['weekend_mean_eur']:,.2f}", f"σ = € {res['metrics']['weekend_std_eur']:,.2f}")
        c2.metric("Media Feriale", f"€ {res['metrics']['weekday_mean_eur']:,.2f}", f"σ = € {res['metrics']['weekday_std_eur']:,.2f}")
        c3.metric("Test t di Welch (p-value)", f"{res['metrics']['welch_p_value']:.4f}", f"t = {res['metrics']['welch_t_stat']:.4f}")
        c4.metric("Mann-Whitney U (p-value)", f"{res['metrics']['mann_whitney_p_value']:.4f}", f"d Cohen = {res['metrics']['cohens_d']:.3f}")
        st.plotly_chart(
            viz.plot_weekend_vs_weekday_box(res["dataframe"]),
            use_container_width=True,
            key="stats_tab_weekend_chart",
        )

    elif stat_choice.startswith("2."):
        res = stats_engine.test_store_anova()
        st.info(res["interpretation"])
        c1, c2 = st.columns(2)
        c1.metric("ANOVA F-Statistic", f"{res['metrics']['anova_f_stat']:.4f}", f"p-value = {res['metrics']['anova_p_value']:.4f}")
        c2.metric("Kruskal-Wallis H", f"{res['metrics']['kruskal_h_stat']:.4f}", f"p-value = {res['metrics']['kruskal_p_value']:.4f}")
        st.dataframe(res["summary_table"], use_container_width=True)

    elif stat_choice.startswith("3."):
        res = stats_engine.detect_daily_anomalies()
        st.info(res["interpretation"])
        st.plotly_chart(
            viz.plot_daily_anomalies(res["dataframe"]),
            use_container_width=True,
            key="stats_tab_anomalies_chart",
        )
        st.markdown("#### Giornate Anomale Identificate (|Z| > 2.5 o Fuori Banda IQR)")
        st.dataframe(res["anomalies_table"], use_container_width=True)

    elif stat_choice.startswith("4."):
        res = stats_engine.test_return_rate_independence()
        st.info(res["interpretation"])
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Resi per Categoria Merceologica")
            st.dataframe(res["category_table"], use_container_width=True)
        with col_b:
            st.markdown("#### Resi per Singolo Prodotto")
            st.dataframe(res["dataframe"], use_container_width=True)

    elif stat_choice.startswith("5."):
        res = stats_engine.analyze_customer_rfm()
        st.info(res["interpretation"])
        st.markdown("#### Confronto Transazioni Clienti Identificati vs Anonimi")
        st.dataframe(res["split_table"], use_container_width=True)
        st.markdown("#### Tabella RFM Clienti (Recency, Frequency, Monetary)")
        st.dataframe(res["dataframe"], use_container_width=True)

# TAB 3: Executive Dashboard (Power BI Replica)
with tab_bi:
    st.subheader("📈 Executive Dashboard (Allineata al Report Power BI)")
    df_monthly = db.execute_query(
        "SELECT year_month, ROUND(SUM(daily_net_revenue), 2) AS net_revenue FROM v_gold_daily_kpi GROUP BY year_month ORDER BY year_month;"
    )
    df_stores = db.execute_query("SELECT * FROM v_gold_store_kpi;")
    df_products = db.execute_query("SELECT * FROM v_gold_product_kpi;")

    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        st.plotly_chart(
            viz.plot_monthly_revenue(df_monthly),
            use_container_width=True,
            key="bi_tab_monthly_chart",
        )
    with row1_col2:
        st.plotly_chart(
            viz.plot_store_comparison(df_stores),
            use_container_width=True,
            key="bi_tab_store_chart",
        )

    st.plotly_chart(
        viz.plot_product_performance(df_products),
        use_container_width=True,
        key="bi_tab_product_chart",
    )

# 📊 Retail AI Data Analyst & Statistical Decision Support System

**Autore:** Andrea Petito — *Laureato in Statistica e Informatica per l'Azienda, la Finanza e le Assicurazioni (L-41)*

Applicazione **AI Data Analyst & Text-to-SQL Agent** sviluppata in **Python, SQLite, SciPy e Streamlit** che evolve il progetto di ingegneria del dato [`retail-elt-pipeline`](https://github.com/petitoandrea-jpg/retail-elt-pipeline) (Medallion Architecture: Bronze → Silver → Gold) in un sistema interattivo di supporto alle decisioni manageriali.

---

## 🎯 Valore Aggiunto del Progetto

1. **Semantic Layer SQL (`setup_semantic_layer.py`):** Arricchisce lo Star Schema Gold (`fact_sales`, `dim_product`, `dim_store`, `dim_date`) con **5 Viste SQL Semantiche** (`v_gold_store_kpi`, `v_gold_product_kpi`, `v_gold_daily_kpi`, `v_gold_customer_rfm`, `v_gold_sales_enriched`) allineate al centesimo con la dashboard Power BI (€724.220,44 di fatturato netto, 5.801 transazioni, 5,21% tasso di reso).
2. **Sicurezza Read-Only & SQL Guardrail (`database.py`):** Connessione SQLite in modalità sola lettura (`?mode=ro`) + validazione regex contro query distruttive e iniezione dinamica dello schema nel System Prompt.
3. **Inferenza Statistica in Python (`tools_stats.py`):** A differenza dei semplici bot Text-to-SQL che confrontano medie grezze, l'agente esegue test di ipotesi formali con `scipy.stats`:
   - **Weekend (€2.701,53/gg) vs Feriale (€2.633,34/gg):** Il **Test $t$ di Welch ($t = 0.7160, p = 0.4749$)** e il test **Mann-Whitney $U$ ($p = 0.3263$)** dimostrano che la differenza di +€68,19/gg **non è statisticamente significativa**, evitando al manager scelte errate di allocazione costi.
   - **Confronto 4 Negozi (Napoli Vomero, Marcianise, Caserta, Napoli Centro):** **ANOVA a una via ($F = 0.2665, p = 0.8496$)** e **Kruskal-Wallis ($p = 0.8433$)**.
   - **Outlier Detection Giornaliera ($|Z| > 2.5$ & IQR):** Isolamento automatico dei 4 giorni di picco anomalo (record il `2026-01-21` con €4.944,35).
   - **Indipendenza Tasso di Reso:** **Test $\chi^2$ di Pearson ($p = 0.6062$)** tra categorie e analisi puntuale per SKU (*T-shirt dri-fit* 6,69% vs *Calzini tecnici* 3,92%).
4. **Doppia Modalità Operativa (`agent.py`):** Funziona sia con **LLM Cloud (Groq Llama-3.3-70B / OpenAI GPT-4o-mini)** con ciclo automatico di *SQL Self-Correction*, sia in modalità **Smart Local Engine (Zero-API)** per garantire a qualsiasi recruiter di testare l'app live su Streamlit Cloud senza inserire chiavi API.

---

## 🚀 Avvio Rapido in Locale

```bash
pip install -r requirements.txt
python src/setup_semantic_layer.py
pytest tests/ -v
streamlit run app.py
```

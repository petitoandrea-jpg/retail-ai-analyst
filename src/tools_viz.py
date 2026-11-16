"""
tools_viz.py
------------
Generatore di grafici interattivi Plotly per l'Agente AI e la Dashboard Executive.
"""

from __future__ import annotations
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


PALETTE = ["#1f77b4", "#00b4d8", "#2a9d8f", "#e76f51", "#f4a261", "#264653"]


def plot_monthly_revenue(df: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        df,
        x="year_month",
        y="net_revenue",
        text_auto=".2s",
        title="Fatturato Netto Mensile (€) — Tutti i Punti Vendita",
        labels={"year_month": "Mese", "net_revenue": "Fatturato Netto (€)"},
        color_discrete_sequence=["#0077b6"],
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(template="plotly_white", height=420, margin=dict(t=50, b=40))
    return fig


def plot_daily_anomalies(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["full_date"],
            y=df["daily_net_revenue"],
            mode="lines",
            name="Fatturato Giornaliero",
            line=dict(color="#2a6f97", width=1.8),
        )
    )
    anomalies = df[df["is_anomaly"]]
    if not anomalies.empty:
        fig.add_trace(
            go.Scatter(
                x=anomalies["full_date"],
                y=anomalies["daily_net_revenue"],
                mode="markers+text",
                name="Anomalia Statistica (|Z|>2.5)",
                text=anomalies["daily_net_revenue"].apply(lambda v: f"€{v:,.0f}"),
                textposition="top center",
                marker=dict(color="#e63946", size=10, symbol="diamond"),
            )
        )
    mean_v = df["daily_net_revenue"].mean()
    fig.add_hline(
        y=mean_v,
        line_dash="dash",
        line_color="#6c757d",
        annotation_text=f"Media: €{mean_v:,.2f}",
    )
    fig.update_layout(
        title="Serie Storica Fatturato Giornaliero con Rilevamento Outlier (Z-Score & IQR)",
        xaxis_title="Data",
        yaxis_title="Fatturato Netto (€)",
        template="plotly_white",
        height=440,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_weekend_vs_weekday_box(df: pd.DataFrame) -> go.Figure:
    fig = px.box(
        df,
        x="day_type",
        y="daily_net_revenue",
        color="day_type",
        points="all",
        notched=True,
        title="Distribuzione Fatturato Giornaliero: Weekend vs Feriale (Test t di Welch p = 0.4749)",
        labels={"day_type": "Tipo Giorno", "daily_net_revenue": "Fatturato Giornaliero (€)"},
        color_discrete_map={"Weekend": "#0077b6", "Feriale": "#48cae4"},
    )
    fig.update_layout(template="plotly_white", height=430, showlegend=False)
    return fig


def plot_store_comparison(df_store: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        df_store.sort_values("net_revenue", ascending=True),
        x="net_revenue",
        y="store_name",
        orientation="h",
        text="net_revenue",
        color="store_name",
        title="Fatturato Netto per Punto Vendita (€)",
        labels={"store_name": "Punto Vendita", "net_revenue": "Fatturato Netto (€)"},
        color_discrete_sequence=PALETTE,
    )
    fig.update_traces(texttemplate="€%{text:,.2f}", textposition="outside")
    fig.update_layout(template="plotly_white", height=380, showlegend=False, xaxis=dict(range=[0, 215000]))
    return fig


def plot_product_performance(df_prod: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        df_prod.sort_values("net_revenue", ascending=True),
        x="net_revenue",
        y="product_name",
        color="category",
        orientation="h",
        text="net_revenue",
        title="Fatturato Netto per Prodotto e Categoria (€)",
        labels={
            "product_name": "Prodotto",
            "net_revenue": "Fatturato Netto (€)",
            "category": "Categoria",
        },
        color_discrete_sequence=["#0077b6", "#00b4d8", "#90e0ef"],
    )
    fig.update_traces(texttemplate="€%{text:,.0f}", textposition="outside")
    fig.update_layout(template="plotly_white", height=440, xaxis=dict(range=[0, 290000]))
    return fig


def plot_auto_chart(df: pd.DataFrame, title: str = "Visualizzazione Dati") -> go.Figure | None:
    if df is None or df.empty or len(df.columns) < 2:
        return None
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]
    if not num_cols:
        return None
    y_col = num_cols[0]
    x_col = cat_cols[0] if cat_cols else df.columns[0]
    if any(k in x_col.lower() for k in ("date", "month", "giorno", "mese")):
        fig = px.line(df, x=x_col, y=y_col, markers=True, title=title)
    else:
        fig = px.bar(df, x=x_col, y=y_col, text_auto=".2s", title=title, color_discrete_sequence=["#0077b6"])
    fig.update_layout(template="plotly_white", height=400)
    return fig

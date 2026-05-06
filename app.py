"""ReliefFlow — AI-powered humanitarian aid management (Gemma4 + Ollama)."""
import os
from pathlib import Path

from PIL import Image
import streamlit as st

# ── Inject OLLAMA_HOST from Streamlit secrets before ai.py creates its client ──
# On Streamlit Cloud: set OLLAMA_HOST in the app's Secrets panel.
# Locally: set the env-var or rely on the default (http://localhost:11434).
try:
    _secret_host = st.secrets.get("OLLAMA_HOST")
    if _secret_host:
        os.environ["OLLAMA_HOST"] = _secret_host
except Exception:
    pass

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ingest import load_kobo_data, load_all_samples
from scoring import enrich_dataframe
from needs import (
    compute_aggregate_needs, compute_food_rations, compute_medical_supplies,
    compute_financial_needs, CATEGORY_ICON, URGENCY_COLOR, RATION_BASKET,
    RATION_BASE_PEOPLE, FINANCIAL_RATES,
)
import ai
import data_entry

# ─── page config ──────────────────────────────────────────────────────────────

_LOGO = Image.open("logo.png")

st.set_page_config(
    page_title="ReliefFlow",
    page_icon=_LOGO,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── design system ────────────────────────────────────────────────────────────

TIER_COLORS  = {"CRITICAL": "#dc2626", "HIGH": "#ea580c", "MEDIUM": "#ca8a04", "LOW": "#16a34a"}
TIER_BG      = {"CRITICAL": "#fef2f2", "HIGH": "#fff7ed", "MEDIUM": "#fefce8", "LOW": "#f0fdf4"}
TIER_ORDER   = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]

SIGNAL_LABELS = {
    "is_widow":         "Widow",
    "is_orphan_family": "Orphans",
    "is_displaced":     "Displaced",
    "has_medical":      "Medical",
    "has_disability":   "Disability",
    "is_unemployed":    "No Income",
    "is_homeless":      "Homeless",
    "is_renting":       "Renting",
    "is_pregnant":      "Pregnant",
}

DISPLAY_COLS = [
    "family_num", "priority_tier", "priority_score", "family_size",
    "city", "governorate", "displacement_type", "housing_type",
    "need_type", "is_widow", "is_orphan_family", "is_displaced",
    "has_medical", "is_unemployed", "has_disability",
]

PLOTLY_BASE = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(family="Inter, system-ui, sans-serif", color="#1e293b", size=12),
    margin=dict(l=10, r=10, t=40, b=10),
    hoverlabel=dict(bgcolor="white", font_size=13),
)

st.markdown("""
<style>
/* ── app background ── */
.stApp { background: #f1f5f9; }

/* ── sidebar ── */
[data-testid="stSidebar"] {
    background: #1e3a5f !important;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stButton > button {
    background: #2563eb !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #1d4ed8 !important;
}
[data-testid="stSidebar"] hr { border-color: #334d6e !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #94a3b8 !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong { color: #e2e8f0 !important; }

/* ── tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: white;
    border-radius: 12px;
    padding: 6px;
    gap: 4px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.06);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 18px;
    font-weight: 500;
    font-size: 0.9em;
    color: #64748b;
}
.stTabs [aria-selected="true"] {
    background: #2563eb !important;
    color: white !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 20px; }

/* ── metric cards ── */
.rf-card {
    background: white;
    border-radius: 12px;
    padding: 20px 16px;
    box-shadow: 0 1px 8px rgba(0,0,0,0.06);
    text-align: center;
    border-top: 4px solid #2563eb;
}
.rf-card.critical { border-top-color: #dc2626; }
.rf-card.high     { border-top-color: #ea580c; }
.rf-card.medium   { border-top-color: #ca8a04; }
.rf-card.low      { border-top-color: #16a34a; }
.rf-card-val  { font-size: 2.2em; font-weight: 800; color: #1e293b; line-height: 1; }
.rf-card-lbl  { font-size: 0.8em; font-weight: 600; color: #64748b; margin-top: 6px; text-transform: uppercase; letter-spacing: 0.04em; }
.rf-card-sub  { font-size: 0.75em; color: #94a3b8; margin-top: 3px; }

/* ── priority badges ── */
.badge { display:inline-block; padding:3px 10px; border-radius:20px; font-weight:700; font-size:0.78em; letter-spacing:0.03em; }
.badge-CRITICAL { background:#fef2f2; color:#dc2626; border:1px solid #fca5a5; }
.badge-HIGH     { background:#fff7ed; color:#ea580c; border:1px solid #fdba74; }
.badge-MEDIUM   { background:#fefce8; color:#ca8a04; border:1px solid #fde047; }
.badge-LOW      { background:#f0fdf4; color:#16a34a; border:1px solid #86efac; }

/* ── signal pills ── */
.pill { display:inline-block; background:#f1f5f9; color:#475569; padding:2px 9px; border-radius:20px; font-size:0.78em; margin:2px; border:1px solid #e2e8f0; }

/* ── section titles ── */
.rf-section { font-size:1.1em; font-weight:700; color:#1e293b; margin:16px 0 8px; border-left:3px solid #2563eb; padding-left:10px; }

/* ── needs urgency rows ── */
.need-critical { border-left: 3px solid #dc2626; padding: 8px 12px; background:#fef2f2; border-radius:6px; margin:4px 0; }
.need-high     { border-left: 3px solid #ea580c; padding: 8px 12px; background:#fff7ed; border-radius:6px; margin:4px 0; }
.need-medium   { border-left: 3px solid #ca8a04; padding: 8px 12px; background:#fefce8; border-radius:6px; margin:4px 0; }

/* ── query chips ── */
.chip-row { display:flex; flex-wrap:wrap; gap:8px; margin: 8px 0 16px; }
.chip { background:white; border:1px solid #e2e8f0; border-radius:20px; padding:5px 14px; font-size:0.83em; color:#2563eb; cursor:pointer; font-weight:500; box-shadow:0 1px 3px rgba(0,0,0,0.04); }

/* ── info/success boxes ── */
.rf-info { background:#eff6ff; border:1px solid #bfdbfe; border-radius:10px; padding:14px 18px; color:#1e40af; font-size:0.9em; margin: 8px 0; }
.rf-success { background:#f0fdf4; border:1px solid #bbf7d0; border-radius:10px; padding:14px 18px; color:#166534; font-size:0.9em; margin: 8px 0; }

/* ── family detail card ── */
.detail-card { background:white; border-radius:12px; padding:20px; box-shadow:0 1px 8px rgba(0,0,0,0.06); }
.detail-row { display:flex; justify-content:space-between; padding:7px 0; border-bottom:1px solid #f1f5f9; font-size:0.9em; }
.detail-row:last-child { border-bottom: none; }
.detail-label { color:#64748b; font-weight:500; }
.detail-value { color:#1e293b; font-weight:600; text-align:right; }

/* ── landing page ── */
.hero { text-align:center; padding:60px 20px 40px; }
.hero h1 { font-size:2.8em; font-weight:800; color:#1e3a5f; margin-bottom:8px; }
.hero p  { font-size:1.15em; color:#64748b; max-width:560px; margin:0 auto 32px; }
.feature-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:16px; max-width:800px; margin:0 auto; }
.feature-card { background:white; border-radius:12px; padding:20px; box-shadow:0 1px 8px rgba(0,0,0,0.06); text-align:left; }
.feature-icon { font-size:1.8em; margin-bottom:8px; }
.feature-title { font-weight:700; color:#1e293b; font-size:0.95em; margin-bottom:4px; }
.feature-desc  { font-size:0.82em; color:#64748b; line-height:1.5; }
</style>
""", unsafe_allow_html=True)


# ─── helpers ──────────────────────────────────────────────────────────────────

def card(value, label, sub="", cls=""):
    sub_html = f'<div class="rf-card-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="rf-card {cls}">'
        f'<div class="rf-card-val">{value}</div>'
        f'<div class="rf-card-lbl">{label}</div>'
        f'{sub_html}'
        f'</div>'
    )


def tier_badge(tier: str) -> str:
    return f'<span class="badge badge-{tier}">{tier}</span>'


def html_table(df: pd.DataFrame, num_cols: list[str] | None = None) -> str:
    """Render a DataFrame as a styled HTML table with guaranteed readable text."""
    num_cols = set(num_cols or [])
    th = "".join(
        f'<th style="padding:8px 14px;text-align:left;font-weight:600;'
        f'color:#64748b;font-size:0.78em;text-transform:uppercase;'
        f'border-bottom:2px solid #e2e8f0;white-space:nowrap;">{c}</th>'
        for c in df.columns
    )
    rows_html = ""
    for _, row in df.iterrows():
        tds = ""
        for col, val in row.items():
            align = "right" if col in num_cols else "left"
            tds += (
                f'<td style="padding:8px 14px;color:#1e293b;font-size:0.9em;'
                f'border-bottom:1px solid #f1f5f9;text-align:{align};">{val}</td>'
            )
        rows_html += f"<tr>{tds}</tr>"
    return (
        '<div style="overflow-x:auto;">'
        '<table style="width:100%;border-collapse:collapse;background:white;'
        'border-radius:8px;overflow:hidden;">'
        f"<thead><tr>{th}</tr></thead><tbody>{rows_html}</tbody>"
        "</table></div>"
    )


def pills(row: pd.Series) -> str:
    return " ".join(
        f'<span class="pill">{lbl}</span>'
        for col, lbl in SIGNAL_LABELS.items()
        if row.get(col, False)
    )


def df_display(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in DISPLAY_COLS if c in df.columns]
    out = df[cols].copy()
    bool_cols = [c for c in SIGNAL_LABELS if c in out.columns]
    out[bool_cols] = out[bool_cols].apply(lambda s: s.map({True: "✓", False: ""}))
    return out


def plotly_fig(fig: go.Figure) -> go.Figure:
    fig.update_layout(**PLOTLY_BASE)
    return fig


# ─── sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image(_LOGO, use_container_width=True)
    st.markdown(
        '<div style="text-align:center;color:#94a3b8;font-size:0.75em;margin-top:-8px;margin-bottom:4px;">Humanitarian AI · Gemma4 · Ollama</div>',
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("**Load Data**")

    data_src = st.radio(
        "Source",
        ["Local sample", "Upload relief_data.xlsx"],
        label_visibility="collapsed",
    )
    st.caption("Upload the anonymized `relief_data.xlsx` (4 sheets: main, members, damage, needs). Generated by `anonymize.py`.")

    def _load_and_store(source=None):
        raw = load_kobo_data(source)
        if raw.empty:
            st.error("No data found in file.")
            return
        with st.spinner("Scoring families…"):
            st.session_state.update({
                "df": enrich_dataframe(raw),
                "ai_cache": {},
                "insights": None,
                "needs_df": pd.DataFrame(),
                "needs_narrative": None,
                "food_agg_df": pd.DataFrame(),
                "food_fam_df": pd.DataFrame(),
                "medical_df": pd.DataFrame(),
                "fin_fam_df": pd.DataFrame(),
                "fin_totals": {},
            })
        st.success(f"✓ {len(st.session_state['df'])} families loaded")

    if data_src == "Local sample":
        if st.button("⬆ Load & Analyse", type="primary", use_container_width=True):
            with st.spinner("Ingesting records…"):
                _load_and_store()
    else:
        uploaded = st.file_uploader(
            "Upload relief_data.xlsx",
            type=["xlsx"],
            help="Single Excel file with sheets: main, members, damage, needs",
        )
        if uploaded:
            with st.spinner("Processing…"):
                _load_and_store(uploaded)

    if "df" in st.session_state:
        _df = st.session_state["df"]
        st.divider()
        st.markdown("**Filters**")

        tier_filter = st.multiselect(
            "Priority tier",
            TIER_ORDER,
            default=TIER_ORDER,
            key="tier_filter",
        )
        city_options = sorted(_df["city"].dropna().unique().tolist())
        city_filter = st.multiselect(
            "City / Region",
            city_options,
            default=city_options,
            key="city_filter",
        )

        displacement_options = sorted(_df["displacement_type"].dropna().unique().tolist())
        displacement_filter = st.multiselect(
            "Displacement Type",
            displacement_options,
            default=displacement_options,
            key="displacement_filter",
        )

        n_filt = len(_df[
            _df["priority_tier"].isin(tier_filter) &
            _df["city"].isin(city_filter) &
            _df["displacement_type"].isin(displacement_filter)
        ])
        st.markdown(f'<div style="color:#94a3b8;font-size:0.8em;margin-top:6px;">Showing {n_filt} of {len(_df)} families</div>', unsafe_allow_html=True)

    st.divider()
    _host_display = ai.get_host().replace("http://", "").replace("https://", "")
    _is_cloud = "Google AI Studio" in _host_display
    _is_local = "localhost" in _host_display or "127.0.0.1" in _host_display
    _host_label = "☁️ cloud" if _is_cloud else ("🟢 local" if _is_local else "🌐 remote")
    st.markdown(
        f'<div style="color:#475569;font-size:0.75em;text-align:center;">'
        f'Powered by Gemma4<br>'
        f'<span style="font-family:monospace;font-size:0.9em;color:#64748b;">{_host_display}</span><br>'
        f'{_host_label}</div>',
        unsafe_allow_html=True,
    )


# ─── landing page (no data loaded) ────────────────────────────────────────────

if "df" not in st.session_state:
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.image(_LOGO, use_container_width=True)

    st.markdown("""
    <div class="hero" style="padding-top: 0;">
      <p>AI-powered humanitarian aid management — helps charities prioritize families, identify needs, and query data at scale using Gemma4 running locally.</p>
      <div class="feature-grid">
        <div class="feature-card">
          <div class="feature-icon">📊</div>
          <div class="feature-title">Smart Prioritization</div>
          <div class="feature-desc">Automatically scores every family by vulnerability signals — homelessness, medical, displacement, income loss and more.</div>
        </div>
        <div class="feature-card">
          <div class="feature-icon">📦</div>
          <div class="feature-title">Aggregate Needs</div>
          <div class="feature-desc">Generates a categorized procurement list across all families — food, medical, shelter, education, and financial support.</div>
        </div>
        <div class="feature-card">
          <div class="feature-icon">🔍</div>
          <div class="feature-title">Natural Language Search</div>
          <div class="feature-desc">Ask questions in plain English — "widows with children in Homs" — and Gemma4 filters the data for you.</div>
        </div>
      </div>
      <div style="margin-top:36px; color:#94a3b8; font-size:0.88em;">
        ← Load sample data or upload your Excel file from the sidebar to begin
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─── apply filters ────────────────────────────────────────────────────────────

df_full: pd.DataFrame = st.session_state["df"]
tier_f = st.session_state.get("tier_filter", TIER_ORDER)
city_f = st.session_state.get("city_filter", df_full["city"].unique().tolist())
disp_f = st.session_state.get("displacement_filter", df_full["displacement_type"].unique().tolist())

df: pd.DataFrame = df_full[
    df_full["priority_tier"].isin(tier_f) &
    df_full["city"].isin(city_f) &
    df_full["displacement_type"].isin(disp_f)
].copy()


# ─── tabs ─────────────────────────────────────────────────────────────────────

tab_dash, tab_log, tab_queue, tab_needs, tab_detail, tab_query, tab_insights = st.tabs([
    "🏠 Overview",
    "➕ Log Entry",
    "🚨 Priority List",
    "📦 Needs Report",
    "👤 Family Profile",
    "🔍 Smart Search",
    "💡 AI Insights",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Overview
# ═══════════════════════════════════════════════════════════════════════════════

with tab_dash:
    total        = len(df)
    total_people = int(pd.to_numeric(df["family_size"], errors="coerce").sum())
    n_critical   = int((df["priority_tier"] == "CRITICAL").sum())
    n_high       = int((df["priority_tier"] == "HIGH").sum())
    n_medium     = int((df["priority_tier"] == "MEDIUM").sum())
    n_low        = int((df["priority_tier"] == "LOW").sum())

    # ── metric row ──
    cols = st.columns(6)
    cards = [
        (total,        "Families",     "",                   ""),
        (total_people, "Individuals",  "across all families",""),
        (n_critical,   "Critical",     "immediate action",   "critical"),
        (n_high,       "High Priority","next in queue",      "high"),
        (n_medium,     "Medium",       "",                   "medium"),
        (n_low,        "Low",          "",                   "low"),
    ]
    for col, (val, lbl, sub, cls) in zip(cols, cards):
        col.markdown(card(f"{val:,}", lbl, sub, cls), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── row 1: priority donut + city bar ──
    left, right = st.columns(2)

    with left:
        st.markdown('<div class="rf-section">Priority Distribution</div>', unsafe_allow_html=True)
        tier_counts = (
            df["priority_tier"].value_counts()
            .reindex(TIER_ORDER, fill_value=0)
            .reset_index()
        )
        tier_counts.columns = ["Tier", "Families"]
        fig = px.pie(
            tier_counts, names="Tier", values="Families",
            color="Tier", color_discrete_map=TIER_COLORS,
            hole=0.5,
        )
        fig.update_traces(textposition="outside", textinfo="label+value")
        fig.update_layout(**PLOTLY_BASE, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown('<div class="rf-section">Families by City</div>', unsafe_allow_html=True)
        city_counts = df["city"].value_counts().head(10).reset_index()
        city_counts.columns = ["City", "Families"]
        fig2 = px.bar(
            city_counts, x="Families", y="City",
            orientation="h",
            color="Families",
            color_continuous_scale=["#bfdbfe", "#1d4ed8"],
        )
        fig2.update_layout(**PLOTLY_BASE, yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    # ── row 2: vulnerability signals + avg family size ──
    left2, right2 = st.columns(2)

    with left2:
        st.markdown('<div class="rf-section">Vulnerability Signals</div>', unsafe_allow_html=True)
        sig_stats = {
            lbl: int(df[col].sum())
            for col, lbl in SIGNAL_LABELS.items()
            if col in df.columns
        }
        sig_df = (
            pd.DataFrame(list(sig_stats.items()), columns=["Signal", "Families"])
            .sort_values("Families", ascending=True)
        )
        fig3 = px.bar(
            sig_df, x="Families", y="Signal",
            orientation="h",
            color="Families",
            color_continuous_scale=["#fde68a", "#dc2626"],
        )
        fig3.update_layout(**PLOTLY_BASE, coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

    with right2:
        st.markdown('<div class="rf-section">Avg Members per Priority Tier</div>', unsafe_allow_html=True)
        avg_by_tier = (
            df.groupby("priority_tier")["family_size"]
            .apply(lambda s: pd.to_numeric(s, errors="coerce").mean())
            .reindex(TIER_ORDER)
            .reset_index()
        )
        avg_by_tier.columns = ["Tier", "Avg Members"]
        fig4 = px.bar(
            avg_by_tier, x="Tier", y="Avg Members",
            color="Tier", color_discrete_map=TIER_COLORS,
            text_auto=".1f",
        )
        fig4.update_layout(**PLOTLY_BASE, showlegend=False)
        fig4.update_traces(textposition="outside")
        st.plotly_chart(fig4, use_container_width=True)

    # ── top critical families preview ──
    st.markdown('<div class="rf-section">Top 10 Critical Families</div>', unsafe_allow_html=True)
    top10 = df[df["priority_tier"] == "CRITICAL"].sort_values("priority_score", ascending=False).head(10)
    if top10.empty:
        st.markdown('<div class="rf-info">No critical families in the current filter.</div>', unsafe_allow_html=True)
    else:
        st.dataframe(
            df_display(top10),
            column_config={
                "family_num":       st.column_config.NumberColumn("#", width="small"),
                "priority_tier":    st.column_config.TextColumn("Tier", width="small"),
                "priority_score":   st.column_config.ProgressColumn("Score", min_value=0, max_value=150, width="small"),
                "family_size":      st.column_config.NumberColumn("Members", width="small"),
                "city":             st.column_config.TextColumn("City", width="small"),
                "need_type":        st.column_config.TextColumn("Need", width="large"),
                "is_widow":         st.column_config.TextColumn("Widow", width="small"),
                "is_orphan_family": st.column_config.TextColumn("Orphans", width="small"),
                "is_displaced":     st.column_config.TextColumn("Displaced", width="small"),
                "has_medical":      st.column_config.TextColumn("Medical", width="small"),
                "is_unemployed":    st.column_config.TextColumn("No Income", width="small"),
                "source_file":      st.column_config.TextColumn("Source", width="small"),
            },
            hide_index=True,
            use_container_width=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Log Entry (image + audio)
# ═══════════════════════════════════════════════════════════════════════════════

_PARSED_FIELDS = [
    ("family_size",       "Family Size (# members)", "number"),
    ("governorate",       "Governorate",             "text"),
    ("city",              "Sub-District / Village",  "text"),
    ("displacement_type", "Displacement Type",       "text"),
    ("housing_type",      "Housing Type",            "text"),
    ("dependents",        "Dependents",              "number"),
    ("breadwinners",      "Breadwinners",            "number"),
    ("need_type",         "Type of Need",            "text"),
]


def _review_form(parsed: dict, key_prefix: str) -> dict:
    """Render an editable confirmation form from parsed data. Returns edited values."""
    st.markdown('<div class="rf-section">Review & Edit before adding</div>', unsafe_allow_html=True)
    result = {}
    for field, label, ftype in _PARSED_FIELDS:
        val = parsed.get(field)
        safe = "" if val is None else str(val)
        if ftype == "number":
            try:
                num_val = int(float(safe)) if safe else 0
            except ValueError:
                num_val = 0
            result[field] = st.number_input(label, value=num_val, min_value=0, key=f"{key_prefix}_{field}")
        elif ftype == "area":
            result[field] = st.text_area(label, value=safe, key=f"{key_prefix}_{field}", height=80)
        else:
            result[field] = st.text_input(label, value=safe, key=f"{key_prefix}_{field}")
    return result


with tab_log:
    st.markdown(
        '<div class="rf-info">'
        'Field workers can log new family records without a laptop — snap a photo of a handwritten '
        'form or record a voice message. Gemma4 parses the content into structured fields for review '
        'before adding to the dataset. <strong>Everything runs locally — no internet required.</strong>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    sub_img, sub_audio = st.tabs(["📷  Scan Handwritten Form", "🎤  Voice Record"])

    # ── IMAGE ──────────────────────────────────────────────────────────────────
    with sub_img:
        st.markdown('<div class="rf-section">Upload a photo of a handwritten or printed record</div>', unsafe_allow_html=True)
        st.caption("Supports JPG, PNG, HEIC, PDF screenshots. Arabic and English handwriting supported.")

        uploaded_img = st.file_uploader(
            "Choose image",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            key="log_img_upload",
            label_visibility="collapsed",
        )

        if uploaded_img:
            img_bytes = uploaded_img.read()

            col_img, col_parse = st.columns([1, 1])
            with col_img:
                st.image(img_bytes, caption="Uploaded image", use_container_width=True)

            with col_parse:
                if st.button("🔍  Parse with Gemma4", type="primary", key="btn_parse_img"):
                    with st.spinner("Gemma4 is reading the form…"):
                        parsed = ai.parse_image_record(img_bytes)
                    if parsed:
                        st.session_state["log_img_parsed"] = parsed
                        st.markdown(
                            f'<div class="rf-success">Parsed with <strong>{parsed.get("confidence","?")}</strong> confidence. Review below.</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.error("Could not extract data from this image. Try a clearer photo.")

        parsed_img = st.session_state.get("log_img_parsed", {})
        if parsed_img:
            st.divider()
            edited_img = _review_form(parsed_img, key_prefix="img")

            if st.button("✅  Add to Dataset", type="primary", key="btn_add_img"):
                if "df" not in st.session_state:
                    st.session_state["df"] = enrich_dataframe(__import__("pandas").DataFrame())
                st.session_state["df"] = data_entry.add_record(edited_img, st.session_state["df"])
                st.session_state.pop("log_img_parsed", None)
                for _k in ("needs_df", "food_agg_df", "food_fam_df", "medical_df", "fin_fam_df", "fin_totals"):
                    st.session_state.pop(_k, None)
                new_num = int(st.session_state["df"]["family_num"].max())
                st.markdown(
                    f'<div class="rf-success">✓ Family <strong>#{new_num:03d}</strong> added to the dataset.</div>',
                    unsafe_allow_html=True,
                )
        else:
            if not uploaded_img:
                st.markdown(
                    '<div style="text-align:center;padding:50px 0;color:#94a3b8;">'
                    '<div style="font-size:2.5em;margin-bottom:10px;">📷</div>'
                    'Upload a photo of a handwritten record to get started.'
                    '</div>',
                    unsafe_allow_html=True,
                )

    # ── AUDIO ──────────────────────────────────────────────────────────────────
    with sub_audio:
        st.markdown('<div class="rf-section">Record or upload a voice message describing a family</div>', unsafe_allow_html=True)
        st.caption("The worker describes the family verbally — Whisper transcribes locally, Gemma4 extracts the fields.")

        audio_input_col, upload_col = st.columns([1, 1])

        with audio_input_col:
            st.markdown("**Record directly in browser**")
            recorded = st.audio_input("Record message", key="log_audio_recorder", label_visibility="collapsed")

        with upload_col:
            st.markdown("**Or upload an audio file**")
            uploaded_audio = st.file_uploader(
                "Audio file",
                type=["wav", "mp3", "m4a", "ogg", "webm"],
                key="log_audio_upload",
                label_visibility="collapsed",
            )

        # Determine active audio source
        audio_bytes  = None
        audio_suffix = ".wav"
        if recorded:
            audio_bytes  = recorded.read()
            audio_suffix = ".wav"
        elif uploaded_audio:
            audio_bytes  = uploaded_audio.read()
            audio_suffix = Path(uploaded_audio.name).suffix or ".wav"

        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("🎙  Transcribe with Whisper", type="primary", key="btn_transcribe"):
                with st.spinner("Transcribing audio locally (first run downloads ~145 MB Whisper model)…"):
                    try:
                        text, lang = data_entry.transcribe(audio_bytes, audio_suffix)
                        st.session_state["log_audio_transcript"] = text
                        st.session_state["log_audio_lang"]       = lang
                    except Exception as e:
                        st.error(f"Transcription failed: {e}")

        transcript = st.session_state.get("log_audio_transcript", "")
        lang       = st.session_state.get("log_audio_lang", "")

        if transcript:
            st.markdown('<div class="rf-section">Transcription</div>', unsafe_allow_html=True)
            if lang:
                st.caption(f"Detected language: **{lang}**")
            transcript = st.text_area(
                "Edit transcription if needed",
                value=transcript,
                height=120,
                key="log_audio_transcript_edit",
                label_visibility="collapsed",
            )
            st.session_state["log_audio_transcript"] = transcript

            if st.button("🤖  Parse with Gemma4", type="primary", key="btn_parse_audio"):
                with st.spinner("Gemma4 is extracting family information…"):
                    parsed_audio = ai.parse_text_record(transcript)
                if parsed_audio:
                    st.session_state["log_audio_parsed"] = parsed_audio
                    st.markdown(
                        f'<div class="rf-success">Parsed with <strong>{parsed_audio.get("confidence","?")}</strong> confidence. Review below.</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.error("Could not extract structured data. Try rephrasing the message.")

        parsed_audio = st.session_state.get("log_audio_parsed", {})
        if parsed_audio:
            st.divider()
            edited_audio = _review_form(parsed_audio, key_prefix="audio")

            if st.button("✅  Add to Dataset", type="primary", key="btn_add_audio"):
                if "df" not in st.session_state:
                    st.session_state["df"] = enrich_dataframe(__import__("pandas").DataFrame())
                st.session_state["df"] = data_entry.add_record(edited_audio, st.session_state["df"])
                st.session_state.pop("log_audio_parsed", None)
                st.session_state.pop("log_audio_transcript", None)
                for _k in ("needs_df", "food_agg_df", "food_fam_df", "medical_df", "fin_fam_df", "fin_totals"):
                    st.session_state.pop(_k, None)
                new_num = int(st.session_state["df"]["family_num"].max())
                st.markdown(
                    f'<div class="rf-success">✓ Family <strong>#{new_num:03d}</strong> added to the dataset.</div>',
                    unsafe_allow_html=True,
                )
        elif not transcript:
            st.markdown(
                '<div style="text-align:center;padding:50px 0;color:#94a3b8;">'
                '<div style="font-size:2.5em;margin-bottom:10px;">🎤</div>'
                'Record a voice message or upload an audio file to get started.'
                '</div>',
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Priority List
# ═══════════════════════════════════════════════════════════════════════════════

with tab_queue:
    sorted_df = df.sort_values("priority_score", ascending=False).reset_index(drop=True)

    top_cols = st.columns(4)
    top_cols[0].markdown(card(len(sorted_df), "Families shown", "current filters", ""), unsafe_allow_html=True)
    top_cols[1].markdown(card(int((sorted_df["priority_tier"]=="CRITICAL").sum()), "Critical", "", "critical"), unsafe_allow_html=True)
    top_cols[2].markdown(card(int((sorted_df["priority_tier"]=="HIGH").sum()), "High", "", "high"), unsafe_allow_html=True)
    top_cols[3].markdown(card(
        f"{pd.to_numeric(sorted_df['family_size'], errors='coerce').sum():.0f}",
        "Total Individuals", "in filtered set", ""), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="rf-section">Ranked by Vulnerability Score</div>', unsafe_allow_html=True)

    col_cfg = {
        "family_num":       st.column_config.NumberColumn("#", width="small"),
        "priority_tier":    st.column_config.TextColumn("Tier", width="small"),
        "priority_score":   st.column_config.ProgressColumn("Score", min_value=0, max_value=150, width="medium"),
        "family_size":      st.column_config.NumberColumn("Members", width="small"),
        "city":             st.column_config.TextColumn("City", width="small"),
        "need_type":        st.column_config.TextColumn("Need Type", width="large"),
        "is_widow":         st.column_config.TextColumn("Widow", width="small"),
        "is_orphan_family": st.column_config.TextColumn("Orphans", width="small"),
        "is_displaced":     st.column_config.TextColumn("Displaced", width="small"),
        "has_medical":      st.column_config.TextColumn("Medical", width="small"),
        "is_unemployed":    st.column_config.TextColumn("No Income", width="small"),
        "source_file":      st.column_config.TextColumn("Source", width="small"),
    }

    st.dataframe(
        df_display(sorted_df),
        column_config=col_cfg,
        hide_index=True,
        use_container_width=True,
        height=520,
    )

    csv = sorted_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇  Export priority list (CSV)",
        data=csv,
        file_name="reliefflow_priority_list.csv",
        mime="text/csv",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Needs Report
# ═══════════════════════════════════════════════════════════════════════════════

with tab_needs:
    st.markdown(
        '<div class="rf-info">All estimates are computed instantly from vulnerability signals — '
        'no AI required. Financial rates and ration quantities reflect approximate regional costs '
        'and should be adjusted to your context.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("📦  Generate Full Needs Report", type="primary"):
        with st.spinner("Computing needs across all families…"):
            st.session_state["needs_df"]      = compute_aggregate_needs(df)
            st.session_state["needs_narrative"] = None
            _food_fam, _food_agg              = compute_food_rations(df)
            st.session_state["food_fam_df"]   = _food_fam
            st.session_state["food_agg_df"]   = _food_agg
            st.session_state["medical_df"]    = compute_medical_supplies(df)
            _fin_fam, _fin_totals             = compute_financial_needs(df)
            st.session_state["fin_fam_df"]    = _fin_fam
            st.session_state["fin_totals"]    = _fin_totals

    needs_df:    pd.DataFrame = st.session_state.get("needs_df",    pd.DataFrame())
    food_agg_df: pd.DataFrame = st.session_state.get("food_agg_df", pd.DataFrame())
    food_fam_df: pd.DataFrame = st.session_state.get("food_fam_df", pd.DataFrame())
    medical_df:  pd.DataFrame = st.session_state.get("medical_df",  pd.DataFrame())
    fin_fam_df:  pd.DataFrame = st.session_state.get("fin_fam_df",  pd.DataFrame())
    fin_totals:  dict         = st.session_state.get("fin_totals",  {})

    if needs_df.empty:
        st.markdown(
            '<div style="text-align:center;padding:60px 0;color:#94a3b8;">'
            '<div style="font-size:2.5em;margin-bottom:12px;">📦</div>'
            'Click the button above to generate the full needs report for the current dataset.'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        # ── Section 1: Food Ration Planning ───────────────────────────────────
        st.markdown('<div class="rf-section">🍞 Food Ration Planning</div>', unsafe_allow_html=True)

        total_packs = int(food_fam_df["Ration Packs"].sum()) if not food_fam_df.empty else 0
        total_people_food = int(food_fam_df["Family Size"].sum()) if not food_fam_df.empty else 0

        fc1, fc2, fc3 = st.columns(3)
        fc1.markdown(card(f"{total_packs:,}", "Total ration packs / month", f"base: {RATION_BASE_PEOPLE} people per pack", ""), unsafe_allow_html=True)
        fc2.markdown(card(f"{total_people_food:,}", "Total individuals covered", "", ""), unsafe_allow_html=True)
        fc3.markdown(card(len(RATION_BASKET), "Items per ration basket", "monthly", ""), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        basket_col, total_col = st.columns(2)

        with basket_col:
            st.markdown('<div class="rf-section">Standard basket — 4 people / month</div>', unsafe_allow_html=True)
            basket_df = pd.DataFrame(RATION_BASKET)
            basket_df.columns = ["Item", "Qty", "Unit"]
            st.markdown(html_table(basket_df, num_cols=["Qty"]), unsafe_allow_html=True)

        with total_col:
            st.markdown('<div class="rf-section">Total quantities needed — all families</div>', unsafe_allow_html=True)
            if not food_agg_df.empty:
                disp = food_agg_df[["Item", "Unit", "Total needed"]].copy()
                disp["Total needed"] = disp["Total needed"].apply(lambda x: f"{x:,.1f}")
                st.markdown(html_table(disp, num_cols=["Total needed"]), unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                csv_food = food_agg_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button("⬇ Export food quantities (CSV)", data=csv_food,
                                   file_name="food_quantities.csv", mime="text/csv")

        with st.expander("Per-family ration allocation"):
            if not food_fam_df.empty:
                st.dataframe(
                    food_fam_df,
                    column_config={
                        "Family #":     st.column_config.NumberColumn("Family #", width="small"),
                        "Family Size":  st.column_config.NumberColumn("Family Size", width="small"),
                        "Ration Packs": st.column_config.ProgressColumn(
                            "Ration Packs", min_value=0,
                            max_value=int(food_fam_df["Ration Packs"].max()),
                            width="large",
                        ),
                    },
                    hide_index=True, use_container_width=True, height=300,
                )

        st.divider()

        # ── Section 2: Medical Supplies ───────────────────────────────────────
        st.markdown('<div class="rf-section">💊 Medical Supplies Estimation</div>', unsafe_allow_html=True)

        if medical_df.empty:
            st.caption("No medical signals detected in the current dataset.")
        else:
            med_families = int(medical_df["Families"].max()) if not medical_df.empty else 0
            mc1, mc2 = st.columns(2)
            mc1.markdown(card(len(medical_df), "Medical item types", "", "medical"), unsafe_allow_html=True)
            mc2.markdown(card(med_families, "Families requiring medical supplies", "", "critical"), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.dataframe(
                medical_df,
                column_config={
                    "Item":      st.column_config.TextColumn("Item", width="large"),
                    "Unit":      st.column_config.TextColumn("Unit", width="medium"),
                    "Families":  st.column_config.ProgressColumn(
                        "Families", min_value=0, max_value=int(medical_df["Families"].max()), width="medium"
                    ),
                    "Total Qty": st.column_config.NumberColumn("Total Qty", width="small"),
                },
                hide_index=True, use_container_width=True,
            )
            csv_med = medical_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button("⬇ Export medical list (CSV)", data=csv_med,
                               file_name="medical_supplies.csv", mime="text/csv")

        st.divider()

        # ── Section 3: Financial Needs ─────────────────────────────────────────
        st.markdown('<div class="rf-section">💵 Financial Needs Estimation</div>', unsafe_allow_html=True)

        with st.expander("Rate assumptions (USD / month)"):
            rate_df = pd.DataFrame([
                {"Category": k, "Rate (USD)": v, "Applied to": "per person" if "person" in k else "per qualifying family"}
                for k, v in FINANCIAL_RATES.items()
            ])
            st.dataframe(rate_df, hide_index=True, use_container_width=True)
            st.caption("Adjust these rates in needs.py → FINANCIAL_RATES to match your operational context.")

        if fin_totals:
            grand = fin_totals.get("Grand Total", 0)
            fn1, fn2, fn3, fn4 = st.columns(4)
            fn1.markdown(card(f"${grand:,}", "Total monthly budget needed", f"across {len(df)} families", "critical"), unsafe_allow_html=True)
            fn2.markdown(card(f"${fin_totals.get('Food', 0):,}",    "Food",    "", ""), unsafe_allow_html=True)
            fn3.markdown(card(f"${fin_totals.get('Medical', 0):,}", "Medical", "", "medical"), unsafe_allow_html=True)
            fn4.markdown(card(f"${fin_totals.get('Rent', 0):,}",    "Rent",    "", ""), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            fin_left, fin_right = st.columns([3, 2])

            with fin_left:
                st.markdown('<div class="rf-section">Per-family monthly estimate</div>', unsafe_allow_html=True)
                if not fin_fam_df.empty:
                    st.dataframe(
                        fin_fam_df,
                        column_config={
                            "Family #":        st.column_config.NumberColumn("Family #", width="small"),
                            "Size":            st.column_config.NumberColumn("Size", width="small"),
                            "Food $":          st.column_config.NumberColumn("Food $", format="$%d", width="small"),
                            "Utilities $":     st.column_config.NumberColumn("Utilities $", format="$%d", width="small"),
                            "Rent $":          st.column_config.NumberColumn("Rent $", format="$%d", width="small"),
                            "Medical $":       st.column_config.NumberColumn("Medical $", format="$%d", width="small"),
                            "Disability $":    st.column_config.NumberColumn("Disability $", format="$%d", width="small"),
                            "Shelter $":       st.column_config.NumberColumn("Shelter $", format="$%d", width="small"),
                            "Monthly Total $": st.column_config.ProgressColumn(
                                "Monthly Total $", min_value=0,
                                max_value=int(fin_fam_df["Monthly Total $"].max()),
                                format="$%d", width="large",
                            ),
                        },
                        hide_index=True, use_container_width=True, height=380,
                    )
                    csv_fin = fin_fam_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button("⬇ Export financial estimates (CSV)", data=csv_fin,
                                       file_name="financial_needs.csv", mime="text/csv")

            with fin_right:
                st.markdown('<div class="rf-section">Budget breakdown</div>', unsafe_allow_html=True)
                breakdown = {k: v for k, v in fin_totals.items() if k != "Grand Total" and v > 0}
                fig_fin = px.pie(
                    names=list(breakdown.keys()),
                    values=list(breakdown.values()),
                    hole=0.45,
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig_fin.update_layout(**PLOTLY_BASE)
                st.plotly_chart(fig_fin, use_container_width=True)

        st.divider()

        # ── Section 4: General Needs Overview ─────────────────────────────────
        st.markdown('<div class="rf-section">📋 General Needs Overview</div>', unsafe_allow_html=True)

        n_crit_n  = int((needs_df["urgency"] == "critical").sum())
        n_high_n  = int((needs_df["urgency"] == "high").sum())
        total_ins = int(needs_df["families_count"].sum())

        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(card(len(needs_df),    "Need types identified", "", ""),           unsafe_allow_html=True)
        m2.markdown(card(n_crit_n,         "Critical need types",  "", "critical"),    unsafe_allow_html=True)
        m3.markdown(card(n_high_n,         "High-priority types",  "", "high"),        unsafe_allow_html=True)
        m4.markdown(card(f"{total_ins:,}", "Total need instances", "", ""),            unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        tbl_col, chart_col = st.columns([3, 2])

        with tbl_col:
            dn = needs_df.copy()
            dn.insert(0, "Icon", dn["category"].map(lambda c: CATEGORY_ICON.get(c, "📦")))
            dn = dn.rename(columns={
                "need_item": "Need", "category": "Category",
                "urgency": "Urgency", "families_count": "Families", "pct_of_total": "% of Total",
            })
            st.dataframe(
                dn[["Icon", "Need", "Category", "Urgency", "Families", "% of Total"]],
                column_config={
                    "Icon":       st.column_config.TextColumn("", width="small"),
                    "Need":       st.column_config.TextColumn("Need Item", width="large"),
                    "Category":   st.column_config.TextColumn("Category", width="small"),
                    "Urgency":    st.column_config.TextColumn("Urgency", width="small"),
                    "Families":   st.column_config.ProgressColumn(
                        "# Families", min_value=0,
                        max_value=int(needs_df["families_count"].max()), width="medium",
                    ),
                    "% of Total": st.column_config.NumberColumn("% of Total", format="%.1f%%", width="small"),
                },
                hide_index=True, use_container_width=True, height=380,
            )
            csv_needs = needs_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button("⬇ Export needs list (CSV)", data=csv_needs,
                               file_name="aggregate_needs.csv", mime="text/csv")

        with chart_col:
            st.markdown('<div class="rf-section">By Category</div>', unsafe_allow_html=True)
            cat_s = (
                needs_df.groupby("category")["families_count"].sum()
                .sort_values(ascending=True).reset_index()
            )
            cat_s["label"] = cat_s["category"].map(
                lambda c: CATEGORY_ICON.get(c, "📦") + " " + c.capitalize()
            )
            fig_c = px.bar(cat_s, x="families_count", y="label", orientation="h",
                           color_discrete_sequence=["#2563eb"])
            fig_c.update_layout(**PLOTLY_BASE, showlegend=False)
            st.plotly_chart(fig_c, use_container_width=True)

            st.markdown('<div class="rf-section">By Urgency</div>', unsafe_allow_html=True)
            urg_s = (
                needs_df.groupby("urgency")["families_count"].sum()
                .reindex(["critical", "high", "medium"], fill_value=0).reset_index()
            )
            urg_s.columns = ["Urgency", "Families"]
            fig_u = px.pie(urg_s, names="Urgency", values="Families",
                           color="Urgency", color_discrete_map=URGENCY_COLOR, hole=0.45)
            fig_u.update_layout(**PLOTLY_BASE)
            st.plotly_chart(fig_u, use_container_width=True)

        st.divider()
        st.markdown('<div class="rf-section">🤖 AI Procurement Narrative (Gemma4)</div>', unsafe_allow_html=True)

        if st.button("Generate AI Action Plan", key="btn_needs_narrative"):
            with st.spinner("Gemma4 is writing the procurement action plan…"):
                narrative = ai.generate_needs_narrative(needs_df, total_families=len(df))
                st.session_state["needs_narrative"] = narrative

        if st.session_state.get("needs_narrative"):
            st.markdown(st.session_state["needs_narrative"])
        else:
            st.caption("Click above to get Gemma4's prioritized procurement plan.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Family Profile
# ═══════════════════════════════════════════════════════════════════════════════

with tab_detail:
    sorted_opts = df.sort_values("priority_score", ascending=False)
    options = sorted_opts.apply(
        lambda r: f"Family #{int(r['family_num']):03d}  ·  {r.get('city','?')}  ·  {r['priority_tier']}  (score {r['priority_score']})",
        axis=1,
    ).tolist()

    if not options:
        st.markdown('<div class="rf-info">No families match the current filters.</div>', unsafe_allow_html=True)
        st.stop()

    selected_label = st.selectbox("Select a family to view", options, label_visibility="visible")
    selected_idx   = options.index(selected_label)
    row            = sorted_opts.iloc[selected_idx]

    st.markdown("<br>", unsafe_allow_html=True)

    left_col, right_col = st.columns([1, 1])

    # ── left: record details ──
    with left_col:
        tier = row.get("priority_tier", "LOW")
        score = row.get("priority_score", 0)

        st.markdown(f"""
        <div class="detail-card">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;">
            <div style="font-weight:700;font-size:1.05em;color:#1e293b;">
              Family #{int(row['family_num']):03d}
            </div>
            <span class="badge badge-{tier}">{tier}</span>
          </div>
          <div class="detail-row">
            <span class="detail-label">Priority Score</span>
            <span class="detail-value">{score} / 150</span>
          </div>
          <div class="detail-row">
            <span class="detail-label">Family Size</span>
            <span class="detail-value">{row.get('family_size', '—')} members</span>
          </div>
          <div class="detail-row">
            <span class="detail-label">City</span>
            <span class="detail-value">{row.get('city', '—')}</span>
          </div>
          <div class="detail-row">
            <span class="detail-label">Source</span>
            <span class="detail-value">{row.get('source_file', '—')}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Vulnerability signals
        sigs_present = [lbl for col, lbl in SIGNAL_LABELS.items() if row.get(col, False)]
        if sigs_present:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                '<div class="rf-section">Vulnerability Signals</div>'
                + "".join(f'<span class="pill">{s}</span>' for s in sigs_present),
                unsafe_allow_html=True,
            )

        # Score gauge
        st.markdown("<br>", unsafe_allow_html=True)
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=int(score),
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Vulnerability Score", "font": {"size": 13}},
            gauge={
                "axis": {"range": [0, 150], "tickwidth": 1},
                "bar": {"color": TIER_COLORS.get(tier, "#64748b")},
                "steps": [
                    {"range": [0, 25],   "color": "#f0fdf4"},
                    {"range": [25, 50],  "color": "#fefce8"},
                    {"range": [50, 80],  "color": "#fff7ed"},
                    {"range": [80, 150], "color": "#fef2f2"},
                ],
                "threshold": {"line": {"color": "#dc2626", "width": 2}, "value": 80},
            },
        ))
        gauge.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=10),
                            paper_bgcolor="white", font=dict(color="#1e293b"))
        st.plotly_chart(gauge, use_container_width=True)

        # Raw record
        with st.expander("View raw record fields"):
            skip = {"family_id", "family_num", "priority_score", "priority_tier",
                    "source_file", "source_sheet", "city"}
            for k, v in row.items():
                if k in skip or k.startswith("is_") or k in ("has_medical", "has_disability"):
                    continue
                if not (hasattr(v, '__iter__') and not isinstance(v, str)) and pd.notna(v) and str(v).strip() not in ("", "nan"):
                    st.markdown(f"**{k}:** {v}")

    # ── right: AI profile + needs ──
    with right_col:
        st.markdown('<div class="rf-section">AI Profile (Gemma4)</div>', unsafe_allow_html=True)

        if st.button("🤖  Analyse Family", key="btn_profile", use_container_width=True):
            with st.spinner("Gemma4 is analysing this family…"):
                profile = ai.parse_family(row)
                st.session_state.setdefault("ai_cache", {})[str(row["family_id"])] = profile

        cached = st.session_state.get("ai_cache", {}).get(str(row["family_id"]), {})
        if cached:
            if cached.get("summary_en"):
                st.markdown(
                    f'<div class="rf-info" style="margin-bottom:12px;">{cached["summary_en"]}</div>',
                    unsafe_allow_html=True,
                )
            ai_fields = {
                "Family head role": cached.get("family_head_role", "—"),
                "Housing status":   cached.get("housing_status", "—"),
                "Location (AI)":    cached.get("city", "—"),
                "Widow":            "Yes" if cached.get("is_widow") else "No",
                "Orphan family":    "Yes" if cached.get("is_orphan_family") else "No",
                "Displaced":        "Yes" if cached.get("is_displaced") else "No",
                "No income":        "Yes" if cached.get("is_unemployed") else "No",
            }
            rows_html = "".join(
                f'<div class="detail-row"><span class="detail-label">{k}</span>'
                f'<span class="detail-value">{v}</span></div>'
                for k, v in ai_fields.items()
            )
            if cached.get("medical_conditions"):
                conditions = ", ".join(cached["medical_conditions"])
                rows_html += (
                    f'<div class="detail-row"><span class="detail-label">Medical conditions</span>'
                    f'<span class="detail-value">{conditions}</span></div>'
                )
            st.markdown(f'<div class="detail-card">{rows_html}</div>', unsafe_allow_html=True)
        else:
            st.caption("Click **Analyse Family** to generate an AI profile for this family.")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="rf-section">Needs List (Gemma4)</div>', unsafe_allow_html=True)

        if st.button("📋  Generate Needs List", key="btn_needs_fam", use_container_width=True):
            with st.spinner("Gemma4 is generating needs list…"):
                needs_list = ai.generate_needs(row)
                st.session_state.setdefault("ai_cache", {})[str(row["family_id"]) + "_needs"] = needs_list

        cached_needs = st.session_state.get("ai_cache", {}).get(str(row["family_id"]) + "_needs", [])
        if cached_needs:
            urg_icon = {"critical": "🔴", "high": "🟠", "medium": "🟡"}
            cat_icon = {
                "food": "🍞", "medical": "💊", "shelter": "🏠", "education": "📚",
                "hygiene": "🧼", "clothing": "👕", "financial": "💵",
                "psychosocial": "🧠", "other": "📦",
            }
            for need in cached_needs:
                urg = need.get("urgency", "medium")
                cat = need.get("category", "other")
                st.markdown(
                    f'<div class="need-{urg}">'
                    f'{urg_icon.get(urg,"🟡")} {cat_icon.get(cat,"📦")} '
                    f'<strong>{need.get("item","?")}</strong>'
                    f'<span style="color:#94a3b8;font-size:0.82em;margin-left:8px;">{cat}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("Click **Generate Needs List** to get a specific needs assessment.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Smart Search
# ═══════════════════════════════════════════════════════════════════════════════

with tab_query:
    st.markdown(
        '<div class="rf-info">Ask a question in plain English — Gemma4 will translate it into a '
        'data filter and return matching families.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    EXAMPLE_QUERIES = [
        "Show me all homeless families",
        "Families with medical conditions",
        "Widows with more than 4 members",
        "Displaced families with no income",
        "All critical priority families",
        "How many food baskets do we need in طرطوس?",
        "What is the total budget needed for homeless families?",
        "How many blankets needed for displaced families?",
        "Medical supplies needed for اللاذقية",
    ]

    st.markdown('<div class="rf-section">Example queries — click to use</div>', unsafe_allow_html=True)
    chip_cols = st.columns(3)
    for i, q in enumerate(EXAMPLE_QUERIES):
        if chip_cols[i % 3].button(q, key=f"chip_{i}", use_container_width=True):
            st.session_state["query_text"] = q

    question = st.text_input(
        "Your question",
        value=st.session_state.get("query_text", ""),
        placeholder="e.g. How many food baskets do we need in Tartus? or Show widows with children",
        key="query_input",
    )

    if st.button("🔍  Search / Ask", type="primary") and question:
        with st.spinner("Gemma4 is processing your query…"):
            query_result = ai.smart_query(df, question)
        st.session_state["query_result"] = query_result

    query_result = st.session_state.get("query_result", None)

    if query_result:
        qtype = query_result.get("type")
        explanation = query_result.get("explanation", "")
        result_df = query_result.get("result_df")
        answer = query_result.get("answer")

        if explanation:
            st.markdown(f'<div class="rf-info"><strong>Gemma4:</strong> {explanation}</div>', unsafe_allow_html=True)

        # Analytics answer: show big answer card
        if qtype == "analytics" and answer:
            st.markdown(
                f'<div style="background:#ecfdf5;border:1px solid #6ee7b7;border-radius:10px;padding:20px 24px;'
                f'font-size:1.25rem;font-weight:600;color:#065f46;margin:16px 0;">'
                f'📊 {answer}</div>',
                unsafe_allow_html=True,
            )

        if qtype == "error":
            st.error(explanation)

        if result_df is not None and not result_df.empty:
            st.markdown("<br>", unsafe_allow_html=True)
            # Filter result: show as family table; breakdown table: show as plain dataframe
            if qtype == "filter":
                st.markdown(
                    f'<div class="rf-success">Found <strong>{len(result_df)}</strong> matching families</div>',
                    unsafe_allow_html=True,
                )
                st.markdown("<br>", unsafe_allow_html=True)
            else:
                st.markdown('<div class="rf-section">Breakdown</div>', unsafe_allow_html=True)

        if qtype == "filter" and result_df is not None and not result_df.empty:
            st.dataframe(
            df_display(result_df.sort_values("priority_score", ascending=False)),
            column_config={
                "family_num":       st.column_config.NumberColumn("#", width="small"),
                "priority_tier":    st.column_config.TextColumn("Tier", width="small"),
                "priority_score":   st.column_config.ProgressColumn("Score", min_value=0, max_value=150, width="medium"),
                "family_size":      st.column_config.NumberColumn("Members", width="small"),
                "city":             st.column_config.TextColumn("City", width="small"),
                "need_type":        st.column_config.TextColumn("Need", width="large"),
                "is_widow":         st.column_config.TextColumn("Widow", width="small"),
                "is_orphan_family": st.column_config.TextColumn("Orphans", width="small"),
                "is_displaced":     st.column_config.TextColumn("Displaced", width="small"),
                "has_medical":      st.column_config.TextColumn("Medical", width="small"),
                "is_unemployed":    st.column_config.TextColumn("No Income", width="small"),
                "source_file":      st.column_config.TextColumn("Source", width="small"),
            },
            hide_index=True,
            use_container_width=True,
            height=400,
        )
            csv_q = result_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button("⬇  Export results (CSV)", data=csv_q,
                               file_name="search_results.csv", mime="text/csv")

        elif qtype == "analytics" and result_df is not None and not result_df.empty:
            st.dataframe(result_df, hide_index=True, use_container_width=True)
            csv_q = result_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button("⬇  Export breakdown (CSV)", data=csv_q,
                               file_name="analytics_result.csv", mime="text/csv")

        elif qtype == "filter" and (result_df is None or result_df.empty) and explanation:
            st.warning("No matching families found, or the query could not be executed.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — AI Insights
# ═══════════════════════════════════════════════════════════════════════════════

with tab_insights:
    st.markdown(
        '<div class="rf-info">Gemma4 analyses the full dataset and produces actionable '
        'recommendations for your organisation.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🧠  Generate Strategic Insights", type="primary"):
        with st.spinner("Gemma4 is analysing the dataset…"):
            insights = ai.generate_aggregate_insights(df)
            st.session_state["insights"] = insights

    if st.session_state.get("insights"):
        st.markdown(st.session_state["insights"])
    else:
        st.markdown(
            '<div style="text-align:center;padding:60px 0;color:#94a3b8;">'
            '<div style="font-size:2.5em;margin-bottom:12px;">💡</div>'
            'Click <strong>Generate Strategic Insights</strong> to get Gemma4\'s analysis.'
            '</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown('<div class="rf-section">Average Priority Score by City</div>', unsafe_allow_html=True)
    city_scores = (
        df.groupby("city")["priority_score"].mean()
        .sort_values(ascending=False).head(10).reset_index()
    )
    city_scores.columns = ["City", "Avg Score"]
    fig_cs = px.bar(
        city_scores, x="City", y="Avg Score",
        color="Avg Score", color_continuous_scale=["#fde68a", "#dc2626"],
        text_auto=".0f",
    )
    fig_cs.update_layout(**PLOTLY_BASE, coloraxis_showscale=False)
    fig_cs.update_traces(textposition="outside")
    st.plotly_chart(fig_cs, use_container_width=True)

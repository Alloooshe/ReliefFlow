"""ReliefFlow — AI-powered humanitarian aid management (Gemma4 + Ollama)."""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ingest import load_all_samples, load_excel
from scoring import enrich_dataframe
from needs import compute_aggregate_needs, CATEGORY_ICON, URGENCY_COLOR
import ai

st.set_page_config(
    page_title="ReliefFlow",
    page_icon="🤝",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.tier-CRITICAL{background:#c0392b;color:white;padding:2px 10px;border-radius:12px;font-weight:700;font-size:0.85em}
.tier-HIGH    {background:#e67e22;color:white;padding:2px 10px;border-radius:12px;font-weight:700;font-size:0.85em}
.tier-MEDIUM  {background:#f1c40f;color:#333;padding:2px 10px;border-radius:12px;font-weight:700;font-size:0.85em}
.tier-LOW     {background:#27ae60;color:white;padding:2px 10px;border-radius:12px;font-weight:700;font-size:0.85em}
.signal-badge {background:#dfe6e9;color:#2d3436;padding:2px 8px;border-radius:10px;font-size:0.8em;margin:2px}
.section-header{font-size:1.1em;font-weight:600;color:#2c3e50;margin-bottom:4px}
</style>
""", unsafe_allow_html=True)

TIER_COLORS = {"CRITICAL": "#c0392b", "HIGH": "#e67e22", "MEDIUM": "#f1c40f", "LOW": "#27ae60"}
TIER_ORDER = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]

SIGNAL_LABELS = {
    "is_widow": "Widow",
    "is_orphan_family": "Orphans",
    "is_displaced": "Displaced",
    "has_medical": "Medical",
    "has_disability": "Disability",
    "is_unemployed": "No income",
    "is_homeless": "Homeless",
    "is_renting": "Renting",
    "is_pregnant": "Pregnant",
}

DISPLAY_COLS = [
    "priority_tier", "priority_score", "contact_name", "family_size",
    "city", "need_type", "is_widow", "is_orphan_family",
    "is_displaced", "has_medical", "is_unemployed", "source_file",
]


# ─── helpers ─────────────────────────────────────────────────────────────────

def badge(tier: str) -> str:
    return f'<span class="tier-{tier}">{tier}</span>'


def signal_badges(row: pd.Series) -> str:
    return " ".join(
        f'<span class="signal-badge">{label}</span>'
        for col, label in SIGNAL_LABELS.items()
        if row.get(col, False)
    )


def df_display(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in DISPLAY_COLS if c in df.columns]
    out = df[cols].copy()
    bool_cols = [c for c in SIGNAL_LABELS if c in out.columns]
    out[bool_cols] = out[bool_cols].apply(lambda s: s.map({True: "✓", False: ""}))
    return out


# ─── sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🤝 ReliefFlow")
    st.caption("Gemma4 · Ollama · Humanitarian AI")
    st.divider()

    data_src = st.radio("Data source", ["Sample data (3 files)", "Upload Excel"])

    if data_src == "Sample data (3 files)":
        if st.button("Load & Analyse", type="primary", use_container_width=True):
            with st.spinner("Ingesting & scoring families…"):
                raw = load_all_samples()
                if raw.empty:
                    st.error("No data found in sample files.")
                else:
                    st.session_state["df"] = enrich_dataframe(raw)
                    st.session_state["ai_cache"] = {}
                    st.session_state["insights"] = None
                    st.success(f"Loaded **{len(st.session_state['df'])}** families")
    else:
        uploaded = st.file_uploader("Upload .xlsx file", type=["xlsx", "xls"])
        if uploaded:
            with st.spinner("Processing…"):
                raw = load_excel(uploaded)
                if raw.empty:
                    st.error("Could not parse this file. Check that it matches the expected format.")
                else:
                    st.session_state["df"] = enrich_dataframe(raw)
                    st.session_state["ai_cache"] = {}
                    st.session_state["insights"] = None
                    st.success(f"Loaded **{len(st.session_state['df'])}** families")

    if "df" in st.session_state:
        st.divider()
        st.markdown("**Filters**")
        tier_filter = st.multiselect(
            "Priority tier",
            TIER_ORDER,
            default=TIER_ORDER,
            key="tier_filter",
        )
        city_options = sorted(st.session_state["df"]["city"].dropna().unique().tolist())
        city_filter = st.multiselect("City", city_options, default=city_options, key="city_filter")


# ─── early exit if no data ────────────────────────────────────────────────────

if "df" not in st.session_state:
    st.title("🤝 ReliefFlow")
    st.markdown("""
**AI-powered humanitarian aid prioritization — powered by Gemma4 (Ollama)**

Upload an Excel file containing family welfare records or load the bundled sample data to get started.

**What this app does:**
- Ingests Arabic/English Excel data and normalises it to a unified schema
- Scores each family by vulnerability using keyword-based rules
- Uses **Gemma4** (locally via Ollama) to generate detailed needs lists and answer natural-language queries
- Helps charity workers prioritise families and understand aggregate needs at scale
    """)
    st.stop()

# Apply sidebar filters
df_full: pd.DataFrame = st.session_state["df"]
tier_f = st.session_state.get("tier_filter", TIER_ORDER)
city_f = st.session_state.get("city_filter", df_full["city"].unique().tolist())

df: pd.DataFrame = df_full[
    df_full["priority_tier"].isin(tier_f) & df_full["city"].isin(city_f)
].copy()


# ─── tabs ─────────────────────────────────────────────────────────────────────

tab_dash, tab_queue, tab_detail, tab_query, tab_needs, tab_insights = st.tabs(
    ["📊 Dashboard", "📋 Priority Queue", "👤 Family Detail", "🔍 AI Query", "📦 Needs Report", "💡 AI Insights"]
)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Dashboard
# ═══════════════════════════════════════════════════════════════════════════════

with tab_dash:
    st.subheader("Overview")

    total = len(df)
    total_people = int(pd.to_numeric(df["family_size"], errors="coerce").sum())
    n_critical = int((df["priority_tier"] == "CRITICAL").sum())
    n_high = int((df["priority_tier"] == "HIGH").sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Families", f"{total:,}")
    c2.metric("Total Individuals", f"{total_people:,}")
    c3.metric("🔴 Critical", n_critical)
    c4.metric("🟠 High Priority", n_high)
    c5.metric("Sources", df["source_file"].nunique())

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        tier_counts = df["priority_tier"].value_counts().reindex(TIER_ORDER, fill_value=0).reset_index()
        tier_counts.columns = ["Tier", "Count"]
        fig = px.bar(
            tier_counts, x="Tier", y="Count",
            color="Tier",
            color_discrete_map=TIER_COLORS,
            title="Families by Priority Tier",
        )
        fig.update_layout(showlegend=False, plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        city_counts = df["city"].value_counts().head(10).reset_index()
        city_counts.columns = ["City", "Count"]
        fig2 = px.bar(
            city_counts, x="Count", y="City",
            orientation="h",
            title="Top Cities",
            color_discrete_sequence=["#3498db"],
        )
        fig2.update_layout(yaxis=dict(autorange="reversed"), plot_bgcolor="white")
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    col_a, col_b, col_c = st.columns(3)

    signal_stats = {
        label: int(df.get(col, pd.Series([False] * len(df))).sum())
        for col, label in SIGNAL_LABELS.items()
        if col in df.columns
    }
    sig_df = pd.DataFrame(list(signal_stats.items()), columns=["Situation", "Families"])
    sig_df = sig_df.sort_values("Families", ascending=False)

    with col_a:
        fig3 = px.pie(
            sig_df, names="Situation", values="Families",
            title="Vulnerability Signals Distribution",
            hole=0.4,
        )
        fig3.update_traces(textposition="inside", textinfo="label+percent")
        st.plotly_chart(fig3, use_container_width=True)

    with col_b:
        avg_by_tier = df.groupby("priority_tier")["family_size"].apply(
            lambda s: pd.to_numeric(s, errors="coerce").mean()
        ).reindex(TIER_ORDER).reset_index()
        avg_by_tier.columns = ["Tier", "Avg Family Size"]
        fig4 = px.bar(
            avg_by_tier, x="Tier", y="Avg Family Size",
            color="Tier", color_discrete_map=TIER_COLORS,
            title="Avg Family Size per Tier",
        )
        fig4.update_layout(showlegend=False, plot_bgcolor="white")
        st.plotly_chart(fig4, use_container_width=True)

    with col_c:
        src_counts = df["source_file"].value_counts().reset_index()
        src_counts.columns = ["Source", "Count"]
        fig5 = px.pie(src_counts, names="Source", values="Count", title="Families by Source File")
        st.plotly_chart(fig5, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Priority Queue
# ═══════════════════════════════════════════════════════════════════════════════

with tab_queue:
    st.subheader(f"Priority Queue — {len(df)} families")

    sorted_df = df.sort_values("priority_score", ascending=False).reset_index(drop=True)
    display = df_display(sorted_df)

    col_cfg = {
        "priority_tier": st.column_config.TextColumn("Tier", width="small"),
        "priority_score": st.column_config.ProgressColumn("Score", min_value=0, max_value=150, width="small"),
        "contact_name": st.column_config.TextColumn("Name", width="medium"),
        "family_size": st.column_config.NumberColumn("Members", width="small"),
        "city": st.column_config.TextColumn("City", width="small"),
        "need_type": st.column_config.TextColumn("Need Type", width="large"),
        "is_widow": st.column_config.TextColumn("Widow", width="small"),
        "is_orphan_family": st.column_config.TextColumn("Orphans", width="small"),
        "is_displaced": st.column_config.TextColumn("Displaced", width="small"),
        "has_medical": st.column_config.TextColumn("Medical", width="small"),
        "is_unemployed": st.column_config.TextColumn("No Income", width="small"),
        "source_file": st.column_config.TextColumn("Source", width="small"),
    }

    st.dataframe(
        display,
        column_config=col_cfg,
        use_container_width=True,
        height=600,
    )

    csv = sorted_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇ Export filtered list (CSV)",
        data=csv,
        file_name="reliefflow_priority_queue.csv",
        mime="text/csv",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Family Detail (Gemma4)
# ═══════════════════════════════════════════════════════════════════════════════

with tab_detail:
    st.subheader("Family Detail — AI-generated profile")

    sorted_options = df.sort_values("priority_score", ascending=False)
    options = sorted_options.apply(
        lambda r: f"[{r['priority_tier']}] {r.get('contact_name','?')} — {r.get('city','?')} (score {r['priority_score']})",
        axis=1,
    ).tolist()

    if not options:
        st.info("No families match current filters.")
        st.stop()

    selected_label = st.selectbox("Select a family", options)
    selected_idx = options.index(selected_label)
    row = sorted_options.iloc[selected_idx]

    # Raw record
    with st.expander("Raw record", expanded=False):
        raw_fields = {
            k: v for k, v in row.items()
            if k not in ("family_id",) and not k.startswith("is_") and k not in ("has_medical", "has_disability", "priority_score", "priority_tier")
        }
        for k, v in raw_fields.items():
            if pd.notna(v) and str(v).strip() not in ("", "nan"):
                st.markdown(f"**{k}**: {v}")

    # Rule-based signals
    signals_present = [label for col, label in SIGNAL_LABELS.items() if row.get(col, False)]
    if signals_present:
        st.markdown(
            "**Detected signals:** " + " ".join(f'<span class="signal-badge">{s}</span>' for s in signals_present),
            unsafe_allow_html=True,
        )

    col_l, col_r = st.columns(2)

    with col_l:
        if st.button("🤖 Generate AI Profile", key="btn_profile"):
            with st.spinner("Gemma4 is analysing this family…"):
                profile = ai.parse_family(row)
                st.session_state.setdefault("ai_cache", {})[row["family_id"]] = profile

        cached_profile = st.session_state.get("ai_cache", {}).get(row["family_id"], {})
        if cached_profile:
            st.markdown("#### AI Profile")
            if cached_profile.get("summary_en"):
                st.info(cached_profile["summary_en"])
            fields_map = {
                "Family head role": cached_profile.get("family_head_role", "—"),
                "Clean name": cached_profile.get("clean_name", "—"),
                "Housing status": cached_profile.get("housing_status", "—"),
                "City (AI)": cached_profile.get("city", "—"),
                "Unemployed": "Yes" if cached_profile.get("is_unemployed") else "No",
                "Widow": "Yes" if cached_profile.get("is_widow") else "No",
                "Orphan family": "Yes" if cached_profile.get("is_orphan_family") else "No",
                "Displaced": "Yes" if cached_profile.get("is_displaced") else "No",
            }
            for k, v in fields_map.items():
                st.markdown(f"**{k}:** {v}")
            if cached_profile.get("medical_conditions"):
                st.markdown(f"**Medical conditions:** {', '.join(cached_profile['medical_conditions'])}")

    with col_r:
        if st.button("📋 Generate Needs List", key="btn_needs"):
            with st.spinner("Gemma4 is generating needs list…"):
                needs = ai.generate_needs(row)
                st.session_state.setdefault("ai_cache", {})[row["family_id"] + "_needs"] = needs

        cached_needs = st.session_state.get("ai_cache", {}).get(row["family_id"] + "_needs", [])
        if cached_needs:
            st.markdown("#### Needs List")
            urgency_color = {"critical": "🔴", "high": "🟠", "medium": "🟡"}
            category_icon = {
                "food": "🍞", "medical": "💊", "shelter": "🏠", "education": "📚",
                "hygiene": "🧼", "clothing": "👕", "financial": "💵",
                "psychosocial": "🧠", "other": "📦",
            }
            for need in cached_needs:
                icon = category_icon.get(need.get("category", "other"), "📦")
                urg = urgency_color.get(need.get("urgency", "medium"), "🟡")
                st.markdown(f"{urg} {icon} **{need.get('item', '?')}** _{need.get('category', '')}_")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — AI Query
# ═══════════════════════════════════════════════════════════════════════════════

with tab_query:
    st.subheader("AI Query — ask questions about the data")
    st.caption("Gemma4 will translate your question into a filter and return matching families.")

    example_queries = [
        "Show me all homeless families",
        "Families with medical conditions in Homs",
        "Widows with more than 4 family members",
        "Displaced families with no income",
        "All critical priority families",
        "Families from Masyaf",
    ]
    st.markdown("**Example queries:** " + " · ".join(f"`{q}`" for q in example_queries))

    question = st.text_input("Your question", placeholder="e.g. Show me widows with children who have no income")

    if st.button("🔍 Search", type="primary", key="btn_query") and question:
        with st.spinner("Gemma4 is processing your query…"):
            result_df, explanation = ai.answer_query(df, question)

        st.markdown(f"**Gemma4:** {explanation}")
        if result_df.empty:
            st.warning("No results found or the query could not be executed.")
        else:
            st.success(f"Found **{len(result_df)}** matching families")
            st.dataframe(
                df_display(result_df.sort_values("priority_score", ascending=False)),
                use_container_width=True,
                height=400,
            )
            csv_q = result_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button("⬇ Export results (CSV)", data=csv_q, file_name="query_results.csv", mime="text/csv")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Needs Report
# ═══════════════════════════════════════════════════════════════════════════════

with tab_needs:
    st.subheader("Aggregate Needs Report")
    st.caption(
        "Rule-based needs are computed instantly from vulnerability signals. "
        "Click **Generate AI Narrative** to have Gemma4 turn the table into a procurement action plan."
    )

    if st.button("📦 Generate Aggregate Needs List", type="primary"):
        with st.spinner("Computing needs for all families…"):
            needs_df = compute_aggregate_needs(df)
            st.session_state["needs_df"] = needs_df
            st.session_state["needs_narrative"] = None  # reset on recompute

    needs_df: pd.DataFrame = st.session_state.get("needs_df", pd.DataFrame())

    if needs_df.empty:
        st.info("Click the button above to generate the needs list for the current filtered dataset.")
    else:
        # ── summary metrics ──
        n_critical_needs = int((needs_df["urgency"] == "critical").sum())
        n_high_needs = int((needs_df["urgency"] == "high").sum())
        total_need_instances = int(needs_df["families_count"].sum())

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Unique need types", len(needs_df))
        c2.metric("🔴 Critical needs", n_critical_needs)
        c3.metric("🟠 High-priority needs", n_high_needs)
        c4.metric("Total need instances", f"{total_need_instances:,}")

        st.divider()

        # ── needs table ──
        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.markdown("#### Needs by Family Count")

            display_needs = needs_df.copy()
            display_needs.insert(
                0, "Icon",
                display_needs["category"].map(lambda c: CATEGORY_ICON.get(c, "📦"))
            )
            display_needs = display_needs.rename(columns={
                "need_item": "Need",
                "category": "Category",
                "urgency": "Urgency",
                "families_count": "Families",
                "pct_of_total": "% of Total",
            })

            st.dataframe(
                display_needs[["Icon", "Need", "Category", "Urgency", "Families", "% of Total"]],
                column_config={
                    "Icon": st.column_config.TextColumn("", width="small"),
                    "Need": st.column_config.TextColumn("Need Item", width="large"),
                    "Category": st.column_config.TextColumn("Category", width="small"),
                    "Urgency": st.column_config.TextColumn("Urgency", width="small"),
                    "Families": st.column_config.ProgressColumn(
                        "# Families", min_value=0, max_value=int(needs_df["families_count"].max()), width="medium"
                    ),
                    "% of Total": st.column_config.NumberColumn("% of Total", format="%.1f%%", width="small"),
                },
                hide_index=True,
                use_container_width=True,
                height=420,
            )

            csv_needs = needs_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "⬇ Export needs list (CSV)",
                data=csv_needs,
                file_name="aggregate_needs.csv",
                mime="text/csv",
            )

        with col_right:
            st.markdown("#### By Category")
            cat_summary = (
                needs_df.groupby("category")["families_count"].sum()
                .sort_values(ascending=True)
                .reset_index()
            )
            cat_summary["icon"] = cat_summary["category"].map(
                lambda c: CATEGORY_ICON.get(c, "📦") + " " + c.capitalize()
            )
            fig_cat = px.bar(
                cat_summary, x="families_count", y="icon",
                orientation="h",
                labels={"families_count": "Families", "icon": ""},
                color_discrete_sequence=["#3498db"],
            )
            fig_cat.update_layout(plot_bgcolor="white", showlegend=False, margin=dict(l=0))
            st.plotly_chart(fig_cat, use_container_width=True)

            st.markdown("#### By Urgency")
            urg_summary = (
                needs_df.groupby("urgency")["families_count"].sum()
                .reindex(["critical", "high", "medium"], fill_value=0)
                .reset_index()
            )
            urg_summary.columns = ["Urgency", "Families"]
            urg_colors = [URGENCY_COLOR.get(u, "#95a5a6") for u in urg_summary["Urgency"]]
            fig_urg = px.pie(
                urg_summary, names="Urgency", values="Families",
                color="Urgency",
                color_discrete_map=URGENCY_COLOR,
                title="Need instances by urgency",
                hole=0.45,
            )
            st.plotly_chart(fig_urg, use_container_width=True)

        st.divider()

        # ── AI narrative ──
        st.markdown("#### AI Procurement Narrative (Gemma4)")
        if st.button("🤖 Generate AI Narrative", key="btn_needs_narrative"):
            with st.spinner("Gemma4 is writing the procurement action plan…"):
                narrative = ai.generate_needs_narrative(needs_df, total_families=len(df))
                st.session_state["needs_narrative"] = narrative

        if st.session_state.get("needs_narrative"):
            st.markdown(st.session_state["needs_narrative"])
        else:
            st.caption("Click the button above to get Gemma4's action plan based on the needs table.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — AI Insights
# ═══════════════════════════════════════════════════════════════════════════════

with tab_insights:
    st.subheader("AI Insights — aggregate analysis by Gemma4")
    st.caption("Gemma4 analyses the full dataset and provides actionable recommendations.")

    if st.button("🧠 Generate Insights", type="primary"):
        with st.spinner("Gemma4 is analysing the full dataset…"):
            insights = ai.generate_aggregate_insights(df)
            st.session_state["insights"] = insights

    if st.session_state.get("insights"):
        st.markdown(st.session_state["insights"])
    else:
        st.info("Click **Generate Insights** to get Gemma4's analysis of the current dataset.")

    st.divider()
    st.subheader("Quick Statistics")

    sig_summary = {
        label: int(df.get(col, pd.Series([False] * len(df))).sum())
        for col, label in SIGNAL_LABELS.items()
        if col in df.columns
    }
    sig_summary_df = pd.DataFrame(
        list(sig_summary.items()), columns=["Category", "Families"]
    ).sort_values("Families", ascending=False)

    st.dataframe(sig_summary_df, use_container_width=True, hide_index=True)

    avg_score_by_city = (
        df.groupby("city")["priority_score"].mean().sort_values(ascending=False).head(10).reset_index()
    )
    avg_score_by_city.columns = ["City", "Avg Priority Score"]
    fig_city = px.bar(
        avg_score_by_city, x="City", y="Avg Priority Score",
        title="Average Priority Score by City (top 10)",
        color="Avg Priority Score",
        color_continuous_scale="Reds",
    )
    fig_city.update_layout(plot_bgcolor="white")
    st.plotly_chart(fig_city, use_container_width=True)

"""Rule-based aggregate needs computation — no AI, instant results."""
import pandas as pd

CATEGORY_ICON = {
    "food": "🍞",
    "medical": "💊",
    "shelter": "🏠",
    "education": "📚",
    "hygiene": "🧼",
    "clothing": "👕",
    "financial": "💵",
    "psychosocial": "🧠",
    "other": "📦",
}

URGENCY_RANK = {"critical": 0, "high": 1, "medium": 2}
URGENCY_COLOR = {"critical": "#c0392b", "high": "#e67e22", "medium": "#f1c40f"}

# (signal_col, need_item, category, urgency)
# None signal_col means "always"
_NEED_RULES: list[tuple[str | None, str, str, str]] = [
    (None,              "Monthly food basket",            "food",         "high"),
    ("is_widow",        "Women's hygiene kit",            "hygiene",      "high"),
    ("is_orphan_family","Women's hygiene kit",            "hygiene",      "high"),
    ("is_displaced",    "Blankets & bedding",             "shelter",      "critical"),
    ("is_homeless",     "Blankets & bedding",             "shelter",      "critical"),
    ("is_homeless",     "Emergency shelter referral",     "shelter",      "critical"),
    ("is_renting",      "Rental financial assistance",    "financial",    "high"),
    ("has_medical",     "Medication & treatment support", "medical",      "critical"),
    ("has_disability",  "Disability aid & rehab",         "medical",      "high"),
    ("is_pregnant",     "Prenatal care & supplies",       "medical",      "critical"),
    ("is_unemployed",   "Livelihood / income support",    "financial",    "high"),
    ("is_displaced",    "Household essentials kit",       "other",        "high"),
    ("is_orphan_family","School supplies for orphans",    "education",    "medium"),
    ("has_medical",     "Psychosocial support",           "psychosocial", "medium"),
]


def compute_aggregate_needs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per unique need, columns:
      need_item, category, urgency, families_count, pct_of_total
    Sorted by urgency then count descending.
    """
    total = max(len(df), 1)
    counts: dict[tuple, int] = {}

    for _, row in df.iterrows():
        seen_items: set[str] = set()
        for signal, item, cat, urg in _NEED_RULES:
            if item in seen_items:
                continue
            if signal is None or row.get(signal, False):
                key = (item, cat, urg)
                counts[key] = counts.get(key, 0) + 1
                seen_items.add(item)

    rows = [
        {
            "need_item": item,
            "category": cat,
            "urgency": urg,
            "families_count": cnt,
            "pct_of_total": round(cnt / total * 100, 1),
        }
        for (item, cat, urg), cnt in counts.items()
    ]

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    result["urgency_rank"] = result["urgency"].map(URGENCY_RANK)
    result = (
        result.sort_values(["urgency_rank", "families_count"], ascending=[True, False])
        .drop(columns="urgency_rank")
        .reset_index(drop=True)
    )
    return result

"""Rule-based needs computation — no AI, instant results."""
import math

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

URGENCY_RANK  = {"critical": 0, "high": 1, "medium": 2}
URGENCY_COLOR = {"critical": "#c0392b", "high": "#e67e22", "medium": "#f1c40f"}

# ─── 1. General needs rules ───────────────────────────────────────────────────

_NEED_RULES: list[tuple[str | None, str, str, str]] = [
    (None,               "Monthly food basket",            "food",         "high"),
    ("is_widow",         "Women's hygiene kit",            "hygiene",      "high"),
    ("is_orphan_family", "Women's hygiene kit",            "hygiene",      "high"),
    ("is_displaced",     "Blankets & bedding",             "shelter",      "critical"),
    ("is_homeless",      "Blankets & bedding",             "shelter",      "critical"),
    ("is_homeless",      "Emergency shelter referral",     "shelter",      "critical"),
    ("is_renting",       "Rental financial assistance",    "financial",    "high"),
    ("has_medical",      "Medication & treatment support", "medical",      "critical"),
    ("has_disability",   "Disability aid & rehab",         "medical",      "high"),
    ("is_pregnant",      "Prenatal care & supplies",       "medical",      "critical"),
    ("is_unemployed",    "Livelihood / income support",    "financial",    "high"),
    ("is_displaced",     "Household essentials kit",       "other",        "high"),
    ("is_orphan_family", "School supplies for orphans",    "education",    "medium"),
    ("has_medical",      "Psychosocial support",           "psychosocial", "medium"),
]


def compute_aggregate_needs(df: pd.DataFrame) -> pd.DataFrame:
    total = max(len(df), 1)
    counts: dict[tuple, int] = {}
    for _, row in df.iterrows():
        seen: set[str] = set()
        for signal, item, cat, urg in _NEED_RULES:
            if item in seen:
                continue
            if signal is None or row.get(signal, False):
                key = (item, cat, urg)
                counts[key] = counts.get(key, 0) + 1
                seen.add(item)

    rows = [
        {"need_item": item, "category": cat, "urgency": urg,
         "families_count": cnt, "pct_of_total": round(cnt / total * 100, 1)}
        for (item, cat, urg), cnt in counts.items()
    ]
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["urgency_rank"] = result["urgency"].map(URGENCY_RANK)
    return (
        result.sort_values(["urgency_rank", "families_count"], ascending=[True, False])
        .drop(columns="urgency_rank")
        .reset_index(drop=True)
    )


# ─── 2. Food ration planning ──────────────────────────────────────────────────

# Standard basket for RATION_BASE_PEOPLE people / month
RATION_BASE_PEOPLE = 4

RATION_BASKET: list[dict] = [
    {"item": "Rice",          "qty": 2.0, "unit": "kg"},
    {"item": "Bulgur",        "qty": 1.0, "unit": "kg"},
    {"item": "Pasta",         "qty": 1.0, "unit": "kg"},
    {"item": "Lentils",       "qty": 1.0, "unit": "kg"},
    {"item": "Tomato paste",  "qty": 0.5, "unit": "L"},
    {"item": "Sunflower oil", "qty": 1.0, "unit": "L"},
    {"item": "Canned tuna",   "qty": 2,   "unit": "cans"},
    {"item": "Sugar",         "qty": 1.0, "unit": "kg"},
    {"item": "Tea",           "qty": 1,   "unit": "pack"},
    {"item": "Salt",          "qty": 0.5, "unit": "kg"},
]


def compute_food_rations(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per_family_df, aggregate_items_df).

    per_family_df columns: family_num, family_size, ration_packs
    aggregate_items_df columns: item, unit, qty_per_basket (4 ppl), total_qty
    """
    sizes = (
        pd.to_numeric(df.get("family_size", pd.Series([4] * len(df))), errors="coerce")
        .fillna(4)
        .clip(lower=1)
    )

    # Per-family: discrete packs (ceiling so no family is short)
    per_fam_rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        sz = int(sizes.iloc[i])
        packs = math.ceil(sz / RATION_BASE_PEOPLE)
        per_fam_rows.append({
            "Family #":      row.get("family_num", i + 1),
            "Family Size":   sz,
            "Ration Packs":  packs,
        })
    per_fam_df = pd.DataFrame(per_fam_rows)

    # Aggregate: use continuous scale for accurate totals
    total_scale = (sizes / RATION_BASE_PEOPLE).sum()
    agg_rows = [
        {
            "Item":                   b["item"],
            "Unit":                   b["unit"],
            "Qty / basket (4 people)": b["qty"],
            "Total needed":           round(b["qty"] * total_scale, 1),
        }
        for b in RATION_BASKET
    ]
    agg_df = pd.DataFrame(agg_rows)
    return per_fam_df, agg_df


# ─── 3. Medical supplies ──────────────────────────────────────────────────────

# (signal, item, qty_per_family, unit)
_MEDICAL_RULES: list[tuple[str, str, int | float, str]] = [
    # General medical kit — for any family with a medical condition
    ("has_medical",    "Paracetamol 500mg",       2,   "strips (10 tabs each)"),
    ("has_medical",    "Ibuprofen 400mg",          1,   "strip (10 tabs)"),
    ("has_medical",    "Bandage roll",             2,   "pcs"),
    ("has_medical",    "Antiseptic solution",      1,   "250 ml bottle"),
    ("has_medical",    "Disposable gloves",        1,   "box (50 pcs)"),
    # Pregnancy kit
    ("is_pregnant",    "Prenatal vitamins",        1,   "month pack"),
    ("is_pregnant",    "Iron & folic acid tabs",   1,   "month pack"),
    ("is_pregnant",    "Sterile delivery kit",     1,   "kit"),
    # Disability support
    ("has_disability", "Wound dressing kit",       1,   "set"),
    ("has_disability", "Pressure-relief cushion",  1,   "pc"),
    ("has_disability", "Mobility aid voucher",     1,   "referral"),
]


def compute_medical_supplies(df: pd.DataFrame) -> pd.DataFrame:
    """Return aggregate medical supply table: item, unit, families_count, total_qty."""
    counts: dict[str, dict] = {}
    for _, row in df.iterrows():
        for signal, item, qty, unit in _MEDICAL_RULES:
            if row.get(signal, False):
                if item not in counts:
                    counts[item] = {"Item": item, "Unit": unit, "Families": 0, "Total Qty": 0}
                counts[item]["Families"]  += 1
                counts[item]["Total Qty"] += qty

    result = pd.DataFrame(list(counts.values()))
    if result.empty:
        return result
    # Group by signal priority: pregnancy first, then medical, then disability
    order = {r[1]: i for i, r in enumerate(_MEDICAL_RULES)}
    result["_ord"] = result["Item"].map(order)
    return result.sort_values("_ord").drop(columns="_ord").reset_index(drop=True)


# ─── 4. Financial needs estimation ───────────────────────────────────────────

# Monthly cost estimates in USD — adjust to local context
FINANCIAL_RATES: dict[str, int] = {
    "Food (per person)":    15,   # USD / person / month
    "Basic utilities":      30,   # USD / family / month
    "Rent support":        150,   # USD / month (renting families)
    "Medical support":     100,   # USD / month (has_medical)
    "Disability support":   60,   # USD / month (has_disability)
    "Emergency shelter":   200,   # USD one-time (homeless families)
}

# Signals that trigger a specific cost line
_FIN_SIGNALS: list[tuple[str, str]] = [
    ("is_renting",    "Rent support"),
    ("has_medical",   "Medical support"),
    ("has_disability","Disability support"),
    ("is_homeless",   "Emergency shelter"),
]


def compute_financial_needs(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return (per_family_df, category_totals dict).

    per_family_df columns: Family #, Family Size, Food $, Utilities $,
                           Rent $, Medical $, Disability $, Shelter $, Monthly Total $
    category_totals: {label: total_usd, ..., "Grand Total": N}
    """
    sizes = (
        pd.to_numeric(df.get("family_size", pd.Series([4] * len(df))), errors="coerce")
        .fillna(4)
        .clip(lower=1)
    )

    per_fam_rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        sz = int(sizes.iloc[i])
        food       = sz * FINANCIAL_RATES["Food (per person)"]
        utilities  = FINANCIAL_RATES["Basic utilities"]
        rent       = FINANCIAL_RATES["Rent support"]       if row.get("is_renting",    False) else 0
        medical    = FINANCIAL_RATES["Medical support"]    if row.get("has_medical",    False) else 0
        disability = FINANCIAL_RATES["Disability support"] if row.get("has_disability", False) else 0
        shelter    = FINANCIAL_RATES["Emergency shelter"]  if row.get("is_homeless",    False) else 0
        total      = food + utilities + rent + medical + disability + shelter

        per_fam_rows.append({
            "Family #":       row.get("family_num", i + 1),
            "Size":           sz,
            "Food $":         food,
            "Utilities $":    utilities,
            "Rent $":         rent,
            "Medical $":      medical,
            "Disability $":   disability,
            "Shelter $":      shelter,
            "Monthly Total $": total,
        })

    per_fam_df = pd.DataFrame(per_fam_rows)

    totals = {
        "Food":             int(per_fam_df["Food $"].sum()),
        "Utilities":        int(per_fam_df["Utilities $"].sum()),
        "Rent":             int(per_fam_df["Rent $"].sum()),
        "Medical":          int(per_fam_df["Medical $"].sum()),
        "Disability":       int(per_fam_df["Disability $"].sum()),
        "Emergency Shelter":int(per_fam_df["Shelter $"].sum()),
        "Grand Total":      int(per_fam_df["Monthly Total $"].sum()),
    }
    return per_fam_df, totals

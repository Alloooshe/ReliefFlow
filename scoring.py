import pandas as pd

# Signal weights (now applied to pre-computed boolean columns)
_WEIGHTS: dict[str, int] = {
    "is_homeless": 40,
    "is_orphan_family": 30,
    "is_widow": 25,
    "is_unemployed": 25,
    "is_displaced": 20,
    "is_pregnant": 20,
    "has_medical": 20,
    "has_disability": 15,
    "is_renting": 10,
}


def extract_signals(row: pd.Series) -> dict[str, bool]:
    """Read pre-computed boolean signal columns from the row."""
    return {
        sig: bool(row.get(sig, False))
        for sig in _WEIGHTS
    }


def compute_priority(row: pd.Series, signals: dict | None = None) -> dict:
    if signals is None:
        signals = extract_signals(row)

    score = sum(_WEIGHTS.get(sig, 0) for sig, hit in signals.items() if hit)

    # Family size bonus
    try:
        size = int(float(row.get("family_size") or 0))
        score += 15 if size >= 7 else 10 if size >= 5 else 5 if size >= 3 else 0
    except (ValueError, TypeError):
        pass

    tier = (
        "CRITICAL" if score >= 80
        else "HIGH" if score >= 50
        else "MEDIUM" if score >= 25
        else "LOW"
    )

    return {"priority_score": score, "priority_tier": tier, **signals}


def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    enriched = pd.DataFrame([compute_priority(row) for _, row in df.iterrows()])
    base = df.reset_index(drop=True)
    # Drop signal columns from base since enriched has them (avoid duplicates)
    dup_cols = [c for c in enriched.columns if c in base.columns]
    base = base.drop(columns=dup_cols)
    result = pd.concat([base, enriched], axis=1)
    result.insert(0, "family_num", range(1, len(result) + 1))
    return result

import pandas as pd

# Arabic + English keywords for each vulnerability signal
_SIGNALS: dict[str, list[str]] = {
    "is_widow": [
        "أرملة", "ارملة", "أرمل",
        "widow", "widowed",
    ],
    "is_orphan_family": [
        "أيتام", "ايتام", "يتيم", "المرحوم", "فقدان المعيل", "لايوجد معيل", "لا يوجد معيل",
        "orphan", "orphans", "no breadwinner", "deceased father", "father passed",
    ],
    "is_displaced": [
        "تهجير", "مهجر", "نازح", "النازحين",
        "displaced", "displacement", "forced relocation", "evacuated", "refugee",
    ],
    "has_medical": [
        "مريض", "مرض", "سرطان", "صرع", "قلب مفتوح", "قلب", "معاق", "إعاقة",
        "اعاقة", "شلل", "مشلول", "ضغط", "سكري", "أدوية", "ادوية", "علاج", "جريح",
        "medical", "disease", "illness", "cancer", "epilepsy", "heart", "disability",
        "paralysis", "diabetes", "medication", "treatment", "injured", "chronic",
    ],
    "has_disability": [
        "معاق", "إعاقة", "اعاقة", "شلل", "مشلول", "جريح",
        "disabled", "disability", "paralyzed", "paralysis", "wheelchair", "amputee",
    ],
    "is_unemployed": [
        "لايوجد دخل", "لا يوجد دخل", "لا دخل", "بلا دخل", "فقدان المعيل",
        "لايوجد معيل", "بدون راتب", "مفصول", "توقف مصدر الدخل", "فقر",
        "بدون إرث", "بدون ارث",
        "no income", "no job", "unemployed", "unemployment", "no work", "jobless",
        "no salary", "dismissed", "fired", "poverty", "no source of income",
    ],
    "is_homeless": [
        "بدون منزل", "بلا منزل",
        "homeless", "no home", "no house", "no shelter", "living on street",
    ],
    "is_renting": [
        "أجار", "اجار", "ايجار", "إيجار", "سكن الأجار", "سكن اجار",
        "renting", "rent", "rental", "tenant", "leasing",
    ],
    "is_pregnant": [
        "حامل",
        "pregnant", "pregnancy", "expecting",
    ],
}

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


def _combined_text(row: pd.Series) -> str:
    fields = [
        "humanitarian_situation", "need_type", "contact_name",
        "additional_needs", "financial_support_note",
    ]
    return " ".join(str(row.get(f, "") or "") for f in fields).lower()


def extract_signals(row: pd.Series) -> dict[str, bool]:
    text = _combined_text(row)
    return {sig: any(kw.lower() in text for kw in kws) for sig, kws in _SIGNALS.items()}


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

    # Reported priority bonus (1 = highest urgency in source data)
    try:
        rp = float(row.get("reported_priority") or 0)
        if 0 < rp <= 2:
            score += 20
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
    result = pd.concat([df.reset_index(drop=True), enriched], axis=1)
    result.insert(0, "family_num", range(1, len(result) + 1))
    return result

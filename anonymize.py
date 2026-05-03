#!/usr/bin/env python3
"""
Anonymize KoBoToolbox raw CSVs by dropping PII and bucketing free-text fields.
Writes 4 anonymized CSVs to data_anonymized/ directory.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re

OUTPUT_DIR = Path("data_anonymized")

# PII columns to drop from each table
MAIN_PII_COLS = {
    "رقم هاتف للعائلة",
    "مدخل البيانات",
    "رقم هاتف مدخل البيانات",
    "العنوان التفصيلي",
    "رقم دفتر العائلة",
    "ملاحظات عن العائلة",
    "ملاحظات عن العائلة لم تذكر سابقا",
    # Community center name (identifies the organization)
    "المركز المجتمعي",
    # Original address (PII + may reference org landmarks)
    "عنوان العائلة الأصلي (إذا كانت نازحة من قريبة أخرى)",
    "عنوان العائلة الأصلي",
    "_uuid",
    "_submitted_by",
    "_notes",
    "__version__",
    "_tags",
}

MEMBERS_PII_COLS = {
    # KoBoToolbox metadata containing org name
    "_parent_table_name",
    "الاسم الاول",
    "الكنية",
    "اسم الاب",
    "اسم الام",
    "الرقم الوطني",
    "رقم هاتف الفرد",
    "تفاصيل الاعتداء",
    "تفاصيل الاعتداء (لو حصل إعتداء)",
    "ملاحظات و تفاصيل خاصة بالفرد",
    "ملاحظات و تفاصيل خاصة بالفرد غير مذكورة سابقا",
    "تاريخ الميلاد",
    "العمل السابق",
    "_submission__uuid",
    "_submission__submitted_by",
    "_submission__notes",
    "_submission___version__",
    "_submission__tags",
}

DAMAGE_PII_COLS = {
    "_parent_table_name",
    "معلومات تفصيلية عن الضرر",
    "_submission__uuid",
    "_submission__submitted_by",
    "_submission__notes",
    "_submission___version__",
    "_submission__tags",
}

NEEDS_PII_COLS = {
    "_parent_table_name",
    "تفاصيل الاحتياج",
    "تحديد الاحتياج (غير مذكور سابقا)",
    "تحديد الاحتياج (في حال غير مذكور سابقا)",
    "_submission__uuid",
    "_submission__submitted_by",
    "_submission__notes",
    "_submission___version__",
    "_submission__tags",
}

# Occupation bucketing rules
OCCUPATION_BUCKETS = {
    "no_work": [
        "لايوجد", "لا يوجد", "لاتعمل", "لايعمل", "لا يعمل", "لا", "لاعمل",
        "لا تعمل", "لاشيء", "لا شيء", "بلا وظيفة", "عاطل", "عاجز",
        "غير عامل", "بلا عمل", "لا عمل", "لابعمل",
    ],
    "homemaker": ["ربة منزل", "ربه منزل", "ربة اسرة", "ربه اسرة", "ست بيت"],
    "student": ["طالب", "طالبة", "طالبه", "روضه", "روضة"],
    "agriculture": ["مزارع", "مزارعة", "زراعة", "زراعه", "بالزراعة", "الزراعة", "فلاح"],
    "employed": ["موظف", "موظفة", "معلم", "معلمة", "ممرض", "ممرضة", "طبيب", "مهندس"],
    "self_employed": ["عمل حر", "اعمال حرة", "اعمال حره", "أعمال حرة", "عامل", "عاملة"],
    "military": ["عسكري", "الجيش", "مجاهد", "أمن", "شرطي"],
    "retired": ["متقاعد", "متقاعدة", "متقاعده"],
    "deceased": ["متوفي"],
    "child": ["طفل", "طفلة", "طفله", "رضيع"],
}


def bucket_occupation(occ_str):
    """Map occupation free-text to bucket category."""
    if pd.isna(occ_str):
        return "unknown"
    occ_str = str(occ_str).strip().lower()
    if not occ_str:
        return "unknown"

    for bucket, keywords in OCCUPATION_BUCKETS.items():
        for keyword in keywords:
            if keyword.lower() in occ_str:
                return bucket
    return "other"


def extract_birth_year(dob_str):
    """Extract year from date of birth."""
    if pd.isna(dob_str):
        return np.nan
    dob_str = str(dob_str).strip()
    if not dob_str:
        return np.nan

    try:
        dt = pd.to_datetime(dob_str)
        return int(dt.year)
    except:
        match = re.search(r"\b(19\d{2}|20\d{2})\b", dob_str)
        if match:
            return int(match.group(1))
    return np.nan


def anonymize_main(csv_path):
    """Anonymize main registration CSV."""
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  Shape before: {df.shape}")
    cols_to_drop = MAIN_PII_COLS & set(df.columns)
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    print(f"  Shape after: {df.shape}")
    return df


def anonymize_members(csv_path):
    """Anonymize family members CSV."""
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  Shape before: {df.shape}")
    cols_to_drop = MEMBERS_PII_COLS & set(df.columns)
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    dob_col = next((c for c in ["تاريخ الميلاد", "date_of_birth"] if c in df.columns), None)
    if dob_col:
        df["birth_year"] = df[dob_col].apply(extract_birth_year)

    occ_col = next((c for c in ["العمل الحالي", "العمل الحالي (الرجاء التحديد)"] if c in df.columns), None)
    if occ_col:
        df["occupation_bucket"] = df[occ_col].apply(bucket_occupation)
        df = df.drop(columns=[occ_col])

    print(f"  Shape after: {df.shape}")
    return df


def anonymize_damage(csv_path):
    """Anonymize damage records CSV."""
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  Shape before: {df.shape}")
    cols_to_drop = DAMAGE_PII_COLS & set(df.columns)
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    print(f"  Shape after: {df.shape}")
    return df


def anonymize_needs(csv_path):
    """Anonymize needs records CSV."""
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  Shape before: {df.shape}")
    cols_to_drop = NEEDS_PII_COLS & set(df.columns)
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    print(f"  Shape after: {df.shape}")
    return df


def verify_anonymized(df, table_name):
    """Verify no PII columns remain."""
    all_pii = MAIN_PII_COLS | MEMBERS_PII_COLS | DAMAGE_PII_COLS | NEEDS_PII_COLS
    found_pii = all_pii & set(df.columns)
    if found_pii:
        print(f"  WARNING: {table_name} still contains PII: {found_pii}")
        return False
    print(f"  ✓ {table_name}: no PII detected")
    return True


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    data_dir = Path("data")
    if not data_dir.exists():
        print("ERROR: data/ directory not found")
        return False

    csv_files = list(data_dir.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files in data/")

    if not csv_files:
        print("ERROR: no CSV files found in data/")
        return False

    main_file = None
    members_file = None
    damage_file = None
    needs_file = None

    for f in csv_files:
        fname = f.name.lower()
        if "damag" in fname:
            damage_file = f
        elif "member" in fname or "family" in fname:
            members_file = f
        elif "need" in fname:
            needs_file = f
        else:
            main_file = f

    if not all([main_file, members_file, damage_file, needs_file]):
        sorted_files = sorted(csv_files)
        print(f"Using first 4 files: {[f.name for f in sorted_files[:4]]}")
        if len(sorted_files) >= 4:
            main_file, members_file, damage_file, needs_file = sorted_files[:4]
        else:
            print(f"ERROR: expected 4 CSVs, found {len(csv_files)}")
            return False

    print(f"\nMapping:")
    print(f"  Main: {main_file.name}")
    print(f"  Members: {members_file.name}")
    print(f"  Damage: {damage_file.name}")
    print(f"  Needs: {needs_file.name}\n")

    main_df = anonymize_main(main_file)
    members_df = anonymize_members(members_file)
    damage_df = anonymize_damage(damage_file)
    needs_df = anonymize_needs(needs_file)

    print("\nVerification:")
    verify_anonymized(main_df, "main")
    verify_anonymized(members_df, "members")
    verify_anonymized(damage_df, "damage")
    verify_anonymized(needs_df, "needs")

    print(f"\nWriting anonymized files to {OUTPUT_DIR}/...")
    main_df.to_csv(OUTPUT_DIR / "main_anon.csv", index=False)
    members_df.to_csv(OUTPUT_DIR / "members_anon.csv", index=False)
    damage_df.to_csv(OUTPUT_DIR / "damage_anon.csv", index=False)
    needs_df.to_csv(OUTPUT_DIR / "needs_anon.csv", index=False)

    print(f"  ✓ main_anon.csv ({len(main_df)} rows)")
    print(f"  ✓ members_anon.csv ({len(members_df)} rows)")
    print(f"  ✓ damage_anon.csv ({len(damage_df)} rows)")
    print(f"  ✓ needs_anon.csv ({len(needs_df)} rows)")
    print("\nAnonymization complete!")
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

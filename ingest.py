import uuid
from pathlib import Path
import pandas as pd
import numpy as np
from schema import (
    DATA_FILE, SHEET_MAIN, SHEET_MEMBERS, SHEET_DAMAGE, SHEET_NEEDS,
    MainCols, MemberCols, DamageCols, NeedsCols,
    DISPLACED_TYPES, HOUSING_RENTING, HOUSING_HOMELESS, BABY_NEEDS,
)


def load_kobo_data(source=None):
    """
    Load merged Excel file and return flat family DataFrame.
    source: file path string/Path, or a file-like object (st.UploadedFile).
    Defaults to DATA_FILE on disk.
    """
    src = source or DATA_FILE
    xl = pd.ExcelFile(src)
    main    = xl.parse(SHEET_MAIN)
    members = xl.parse(SHEET_MEMBERS)
    damage  = xl.parse(SHEET_DAMAGE)
    needs   = xl.parse(SHEET_NEEDS)
    return _build_flat_df(main, members, damage, needs)


def _build_flat_df(main, members, damage, needs):
    """Join 4 tables and aggregate to one row per family."""
    df = main.copy()
    df = df.rename(columns={MainCols.ID: "family_id"})

    # --- Members aggregations ---
    mem = members.copy()
    hoh = mem[mem[MemberCols.IS_HOH] == "نعم"]

    female_hoh_ids = hoh[
        (hoh[MemberCols.GENDER] == "أنثى") &
        (hoh[MemberCols.MARITAL_STATUS].isin(["منفصل/ة", "مطلق/ة"]))
    ][MemberCols.FAMILY_ID]
    female_flag_ids = mem[
        mem[MemberCols.V_FEMALE_HOH] == 1.0
    ][MemberCols.FAMILY_ID]
    widow_ids = set(female_hoh_ids) | set(female_flag_ids)

    orphan_ids = set(hoh[
        hoh[MemberCols.INDIVIDUAL_STATUS].isin(["متوفي", "مفقود"])
    ][MemberCols.FAMILY_ID])

    medical_ids = set(mem[
        (mem[MemberCols.V_IMMEDIATE_HEALTH] == 1.0) |
        (mem[MemberCols.V_CHRONIC_DISEASE] == 1.0)
    ][MemberCols.FAMILY_ID])

    disability_flags = [
        MemberCols.V_MENTAL_DISABILITY,
        MemberCols.V_PHYSICAL_DISABILITY,
        MemberCols.V_SPEECH_IMPAIRMENT,
        MemberCols.V_HEARING_IMPAIRMENT,
        MemberCols.V_VISION_IMPAIRMENT,
    ]
    disability_cols = [c for c in disability_flags if c in mem.columns]
    disability_mask = mem[disability_cols].fillna(0).max(axis=1) == 1.0
    disability_ids = set(mem[disability_mask][MemberCols.FAMILY_ID])

    chronic_count = (
        mem[mem[MemberCols.V_CHRONIC_DISEASE] == 1.0]
        .groupby(MemberCols.FAMILY_ID).size().rename("chronic_disease_members")
    )
    disability_count = (
        mem[disability_mask]
        .groupby(MemberCols.FAMILY_ID).size().rename("disability_members")
    )
    dropout_count = (
        mem[mem[MemberCols.V_CHILD_DROPOUT] == 1.0]
        .groupby(MemberCols.FAMILY_ID).size().rename("school_dropout_children")
    )
    member_count = (
        mem.groupby(MemberCols.FAMILY_ID).size().rename("member_count")
    )

    # --- Needs aggregations ---
    n = needs.copy()
    fin_need_ids = set(n[
        n[NeedsCols.PROGRAM] == "احتياجات مالية أو سبل العيش"
    ][NeedsCols.FAMILY_ID])
    medical_need_ids = set(n[
        n[NeedsCols.PROGRAM] == "احتياجات طبية أو صحية"
    ][NeedsCols.FAMILY_ID])
    baby_ids = set(n[
        n[NeedsCols.CLASSIFICATION].isin(BABY_NEEDS)
    ][NeedsCols.FAMILY_ID])
    service_ids = set(n[
        n[NeedsCols.SERVICE_RECEIVED] == "نعم"
    ][NeedsCols.FAMILY_ID])

    need_type_ser = (
        n.groupby(NeedsCols.FAMILY_ID)[NeedsCols.PROGRAM]
        .agg(lambda x: ", ".join(x.value_counts().head(2).index.tolist()))
        .rename("need_type")
    )
    needs_programs = (
        n.groupby(NeedsCols.FAMILY_ID)[NeedsCols.PROGRAM]
        .agg(lambda x: list(x.unique())).rename("needs_programs")
    )

    # --- Damage aggregations ---
    d = damage.copy()
    damage_ids = set(d[DamageCols.FAMILY_ID])
    damage_categories = (
        d.groupby(DamageCols.FAMILY_ID)[DamageCols.CATEGORY]
        .agg(lambda x: list(x.unique())).rename("damage_categories")
    )

    # --- Assemble flat df ---
    df["is_widow"]         = df["family_id"].isin(widow_ids)
    df["is_orphan_family"] = df["family_id"].isin(orphan_ids)
    df["is_displaced"]     = df[MainCols.DISPLACEMENT_TYPE].isin(DISPLACED_TYPES)
    df["is_homeless"]      = df[MainCols.HOUSING_TYPE] == HOUSING_HOMELESS
    df["is_renting"]       = df[MainCols.HOUSING_TYPE] == HOUSING_RENTING
    df["is_unemployed"]    = (
        (df[MainCols.BREADWINNERS] == 0) | df["family_id"].isin(fin_need_ids)
    )
    df["has_medical"]      = (
        df["family_id"].isin(medical_ids) | df["family_id"].isin(medical_need_ids)
    )
    df["has_disability"]   = df["family_id"].isin(disability_ids)
    df["is_pregnant"]      = df["family_id"].isin(baby_ids)
    df["has_damage"]       = df["family_id"].isin(damage_ids)
    df["service_received"] = df["family_id"].isin(service_ids)

    df["city"] = (
        df[MainCols.SUB_DISTRICT]
        .fillna(df[MainCols.DISTRICT])
        .fillna(df[MainCols.GOVERNORATE])
    )
    df["governorate"]       = df[MainCols.GOVERNORATE]
    df["displacement_type"] = df[MainCols.DISPLACEMENT_TYPE]
    df["housing_type"]      = df[MainCols.HOUSING_TYPE]

    df = df.rename(columns={
        MainCols.FAMILY_SIZE:  "family_size",
        MainCols.DEPENDENTS:   "dependents",
        MainCols.BREADWINNERS: "breadwinners",
        MainCols.SURVEY_DATE:  "survey_date",
    })

    for ser in [chronic_count, disability_count, dropout_count, member_count,
                need_type_ser, needs_programs, damage_categories]:
        df = df.join(ser, on="family_id", how="left")

    df["need_type"]   = df["need_type"].fillna("")
    df["source_file"] = "kobo_import"
    df["source_sheet"] = ""

    drop_cols = [
        MainCols.DISPLACEMENT_TYPE, MainCols.HOUSING_TYPE,
        MainCols.GOVERNORATE, MainCols.DISTRICT,
        MainCols.SUB_DISTRICT, MainCols.VILLAGE,
        MainCols.BREADWINNERS, MainCols.DEPENDENTS,
        MainCols.SURVEY_DATE, MainCols.FAMILY_SIZE,
        MainCols.ACCESS_TYPE,
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df.reset_index(drop=True)


def load_all_samples(pattern=None):
    """Backward-compatible entry point — loads from disk."""
    return load_kobo_data()


def load_excel(file, source_name=None):
    """Deprecated stub."""
    import warnings
    warnings.warn("load_excel is deprecated; use load_kobo_data() instead", DeprecationWarning)
    return load_kobo_data()

import uuid
import glob
from pathlib import Path

import pandas as pd

from schema import COLUMN_MAP, IDENTITY_COLS, KNOWN_CITIES

# Keywords that appear only in real header rows (not title/note rows)
_HEADER_MARKERS = ["العائلة", "الرقم التسلسلي", "هاتف", "العنوان", "الحاجة"]


def _find_header_row(df_raw: pd.DataFrame) -> int:
    for i in range(min(15, len(df_raw))):
        row_str = df_raw.iloc[i].astype(str)
        if row_str.str.contains("|".join(_HEADER_MARKERS), na=False).any():
            return i
    return 0


def _map_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        key = str(col).strip()
        if key in COLUMN_MAP:
            rename[col] = COLUMN_MAP[key]
    return df.rename(columns=rename)


def _extract_city(address: str) -> str:
    if pd.isna(address) or not str(address).strip():
        return "Unknown"
    for city in KNOWN_CITIES:
        if city in str(address):
            return city
    tokens = str(address).split()
    return tokens[0] if tokens else "Unknown"


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    present = [c for c in IDENTITY_COLS if c in df.columns]
    if present:
        df = df.dropna(subset=present, how="all")
    # Drop rows where every identity cell is a digit-only string (empty template rows)
    if "contact_name" in df.columns:
        df = df[~df["contact_name"].astype(str).str.fullmatch(r"\d+", na=False)]
    return df.reset_index(drop=True)


def _load_sheet(xl_path, sheet: str, source_name: str) -> pd.DataFrame:
    df_raw = pd.read_excel(xl_path, sheet_name=sheet, header=None)
    df_raw = df_raw.dropna(how="all").reset_index(drop=True)

    if len(df_raw) < 3:
        return pd.DataFrame()

    header_idx = _find_header_row(df_raw)
    df = df_raw.iloc[header_idx:].copy()
    df.columns = df.iloc[0].tolist()
    df = df.iloc[1:].dropna(how="all").dropna(axis=1, how="all")

    # Drop columns with NaN names (artefacts from merged cells / empty cols)
    df = df[[c for c in df.columns if not (isinstance(c, float) and pd.isna(c))]]

    df = _map_columns(df)

    # Skip sheets with fewer than 2 relevant columns
    relevant = [c for c in IDENTITY_COLS if c in df.columns]
    if len(relevant) < 2:
        return pd.DataFrame()

    df = _clean(df)
    if df.empty:
        return pd.DataFrame()

    # Coerce family_size to numeric
    if "family_size" in df.columns:
        df["family_size"] = pd.to_numeric(df["family_size"], errors="coerce")

    # Normalise phone
    if "phone" in df.columns:
        df["phone"] = df["phone"].astype(str).str.strip()

    # Extract city
    df["city"] = df.get("address", pd.Series(["Unknown"] * len(df))).apply(_extract_city)

    df["source_file"] = source_name
    df["source_sheet"] = str(sheet)
    df["family_id"] = [str(uuid.uuid4())[:8].upper() for _ in range(len(df))]

    return df


def load_excel(file, source_name: str = None) -> pd.DataFrame:
    name = source_name or (file.name if hasattr(file, "name") else Path(str(file)).name)
    xl = pd.ExcelFile(file)
    frames = [_load_sheet(file, sheet, name) for sheet in xl.sheet_names]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_all_samples(pattern: str = "data_sample*.xlsx") -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(pattern)):
        df = load_excel(path, source_name=Path(path).stem)
        if not df.empty:
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

"""Audio transcription via faster-whisper (runs fully local, no internet after first model download)."""
import tempfile
import uuid
from pathlib import Path

import pandas as pd

# Lazy-loaded — model is only downloaded/loaded on first transcription call
_whisper_model = None
WHISPER_MODEL_SIZE = "base"  # tiny(~75 MB) | base(~145 MB) | small(~460 MB)


def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device="cpu",
            compute_type="int8",
        )
    return _whisper_model


def transcribe(audio_bytes: bytes, file_suffix: str = ".wav") -> tuple[str, str]:
    """
    Transcribe audio bytes. Returns (transcription_text, detected_language).
    Supports wav, mp3, m4a, ogg, webm.
    """
    model = _get_whisper()

    with tempfile.NamedTemporaryFile(suffix=file_suffix, delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    try:
        segments, info = model.transcribe(tmp_path, beam_size=5, language=None)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return text, info.language
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def add_record(parsed: dict, df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a fully-scored row from parsed fields and append it to df.
    Returns the updated DataFrame.
    """
    from scoring import compute_priority
    from schema import DISPLACED_TYPES, HOUSING_RENTING, HOUSING_HOMELESS

    displacement_type = parsed.get("displacement_type") or ""
    housing_type = parsed.get("housing_type") or ""
    breadwinners = parsed.get("breadwinners")

    new_row: dict = {
        "family_id":         str(uuid.uuid4())[:8].upper(),
        "source_file":       "manual_entry",
        "source_sheet":      "log_entry",
        "family_size":       parsed.get("family_size"),
        "city":              parsed.get("city") or "Unknown",
        "governorate":       parsed.get("governorate") or "",
        "displacement_type": displacement_type,
        "housing_type":      housing_type,
        "dependents":        parsed.get("dependents"),
        "breadwinners":      breadwinners,
        "need_type":         parsed.get("need_type") or "",
        # Pre-computed boolean signals (manually derived from structured fields)
        "is_displaced":      displacement_type in DISPLACED_TYPES,
        "is_homeless":       housing_type == HOUSING_HOMELESS,
        "is_renting":        housing_type == HOUSING_RENTING,
        "is_unemployed":     bool(breadwinners == 0) if breadwinners is not None else False,
        "is_widow":          False,
        "is_orphan_family":  False,
        "has_medical":       False,
        "has_disability":    False,
        "is_pregnant":       False,
        "has_damage":        False,
        "service_received":  False,
    }

    row_series = pd.Series(new_row)
    priority = compute_priority(row_series)
    new_row.update(priority)

    max_num = int(df["family_num"].max()) if (not df.empty and "family_num" in df.columns) else 0
    new_row["family_num"] = max_num + 1

    new_df = pd.DataFrame([new_row])
    for col in df.columns:
        if col not in new_df.columns:
            new_df[col] = None

    return pd.concat([df, new_df[df.columns]], ignore_index=True)

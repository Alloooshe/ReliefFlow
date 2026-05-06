"""Gemma4 interface — Google AI Studio (cloud) or Ollama (local)."""
import base64
import json
import os
import re

import pandas as pd

# ── Provider selection ────────────────────────────────────────────────────────
# Cloud: set GEMINI_API_KEY (env var or st.secrets)
# Local: set OLLAMA_HOST (default: http://localhost:11434)

_GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")

# Try loading from Streamlit secrets when running inside Streamlit
try:
    import streamlit as st
    if not _GEMINI_KEY:
        _GEMINI_KEY = st.secrets.get("GEMINI_API_KEY", "")
except Exception:
    pass

GEMINI_MODEL = "gemma-4-26b-a4b-it"
OLLAMA_MODEL = "gemma4:latest"

_USE_CLOUD = bool(_GEMINI_KEY)

if _USE_CLOUD:
    import google.generativeai as genai
    genai.configure(api_key=_GEMINI_KEY)
    _genai_model = genai.GenerativeModel(GEMINI_MODEL)
    _HOST = f"Google AI Studio ({GEMINI_MODEL})"
else:
    from ollama import Client as _OllamaClient
    _HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    _ollama_client = _OllamaClient(host=_HOST)


def get_host() -> str:
    return _HOST


def _chat(prompt: str, system: str | None = None) -> str:
    if _USE_CLOUD:
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        response = _genai_model.generate_content(full_prompt)
        return response.text
    else:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = _ollama_client.chat(model=OLLAMA_MODEL, messages=messages)
        return resp["message"]["content"]


def _extract_json_object(text: str) -> dict:
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


_RECORD_FIELDS_PROMPT = """
Extract family welfare record information and return ONLY a JSON object — no explanation, no markdown.

JSON structure:
{
  "family_size": integer or null,
  "governorate": "governorate name in Arabic or null",
  "city": "sub-district / village name or null",
  "displacement_type": "نازح داخلي | نازح عائد ... | لاجئ عائد ... | مجتمع مضيف | null",
  "housing_type": "ملك | أجار | أستضافة | لا يوجد مسكن | أخرى | null",
  "dependents": integer or null,
  "breadwinners": integer or null,
  "need_type": "type of aid or help needed",
  "confidence": "high|medium|low"
}
Use null for any field that is missing or unclear.
"""


def parse_image_record(image_bytes: bytes) -> dict:
    """Parse a handwritten or printed welfare form photo using Gemma4 vision."""
    prompt = (
        "You are helping a charity digitize a humanitarian aid family welfare record.\n"
        "The image shows a handwritten or printed form, possibly in Arabic and/or English.\n"
        + _RECORD_FIELDS_PROMPT
    )
    if _USE_CLOUD:
        import google.generativeai as genai
        img_part = {"mime_type": "image/jpeg", "data": image_bytes}
        response = _genai_model.generate_content([prompt, img_part])
        return _extract_json_object(response.text)
    else:
        img_b64 = base64.b64encode(image_bytes).decode()
        resp = _ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt, "images": [img_b64]}],
        )
        return _extract_json_object(resp["message"]["content"])


def parse_text_record(text: str) -> dict:
    """Parse a free-text record (voice transcription or typed notes) using Gemma4."""
    prompt = (
        "A humanitarian aid field worker described a family in need verbally or in notes.\n"
        "Here is the text (may be in Arabic, English, or mixed):\n\n"
        f'"""\n{text}\n"""\n\n'
        + _RECORD_FIELDS_PROMPT
    )
    return _extract_json_object(_chat(prompt))


def _extract_json_array(text: str) -> list:
    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return []


def parse_family(row: pd.Series) -> dict:
    """Extract structured fields from a raw family row using Gemma4."""
    prompt = f"""You are analyzing an anonymized Syrian humanitarian aid record for a family in need.
The data is already structured — your job is to summarize and tag it. Return ONLY a JSON object — no explanation, no markdown.

Record fields:
- Family size: {row.get("family_size", "N/A")} (dependents: {row.get("dependents", "N/A")}, breadwinners: {row.get("breadwinners", "N/A")})
- Governorate: {row.get("governorate", "N/A")}
- City / Region: {row.get("city", "N/A")}
- Displacement type: {row.get("displacement_type", "N/A")}
- Housing type: {row.get("housing_type", "N/A")}
- Damage categories: {row.get("damage_categories", "none")}
- Need programs: {row.get("needs_programs", "none")}
- Pre-computed signals — widow: {row.get("is_widow", False)}, orphan: {row.get("is_orphan_family", False)},
  displaced: {row.get("is_displaced", False)}, medical: {row.get("has_medical", False)},
  disability: {row.get("has_disability", False)}, homeless: {row.get("is_homeless", False)},
  unemployed: {row.get("is_unemployed", False)}

Return exactly this JSON structure:
{{
  "family_head_role": "father|mother|widow|orphan-family|unknown",
  "is_widow": true_or_false,
  "is_orphan_family": true_or_false,
  "is_displaced": true_or_false,
  "has_medical_condition": true_or_false,
  "medical_conditions": ["list conditions if any"],
  "has_disability": true_or_false,
  "is_unemployed": true_or_false,
  "housing_status": "homeless|renting|owner|hosted|unknown",
  "city": "{row.get("city", "")}",
  "summary_en": "one sentence English summary of this family situation"
}}"""
    return _extract_json_object(_chat(prompt))


def generate_needs(row: pd.Series) -> list[dict]:
    """Generate a specific needs list for a family using Gemma4."""
    prompt = f"""You are a humanitarian aid coordinator. Based on the family record below, list their specific needs.
Return ONLY a JSON array — no explanation, no markdown.

Family size: {row.get("family_size", "N/A")} (dependents: {row.get("dependents", "N/A")}, breadwinners: {row.get("breadwinners", "N/A")})
Governorate / City: {row.get("governorate", "N/A")} / {row.get("city", "N/A")}
Displacement type: {row.get("displacement_type", "N/A")}
Housing type: {row.get("housing_type", "N/A")}
Damage categories: {row.get("damage_categories", "none")}
Stated need programs: {row.get("needs_programs", "none")}
Signals — displaced: {row.get("is_displaced", False)}, widow: {row.get("is_widow", False)},
          medical: {row.get("has_medical", False)}, disability: {row.get("has_disability", False)},
          unemployed: {row.get("is_unemployed", False)}, homeless: {row.get("is_homeless", False)}

Return a JSON array of up to 7 specific needs:
[
  {{"item": "need name", "category": "food|medical|shelter|education|hygiene|clothing|financial|psychosocial|other", "urgency": "critical|high|medium"}}
]"""
    return _extract_json_array(_chat(prompt))


def generate_aggregate_insights(df: pd.DataFrame) -> str:
    """Generate high-level insights for the loaded dataset."""
    stats = {
        "total_families": len(df),
        "total_individuals": int(pd.to_numeric(df.get("family_size"), errors="coerce").sum()),
        "critical": int((df.get("priority_tier", pd.Series()) == "CRITICAL").sum()),
        "high_priority": int((df.get("priority_tier", pd.Series()) == "HIGH").sum()),
        "widows": int(df.get("is_widow", pd.Series([False] * len(df))).sum()),
        "orphan_families": int(df.get("is_orphan_family", pd.Series([False] * len(df))).sum()),
        "displaced": int(df.get("is_displaced", pd.Series([False] * len(df))).sum()),
        "with_medical_needs": int(df.get("has_medical", pd.Series([False] * len(df))).sum()),
        "homeless": int(df.get("is_homeless", pd.Series([False] * len(df))).sum()),
        "unemployed": int(df.get("is_unemployed", pd.Series([False] * len(df))).sum()),
        "top_cities": df.get("city", pd.Series()).value_counts().head(5).to_dict(),
    }

    prompt = f"""You are a senior humanitarian aid coordinator reviewing a caseload of families in need.
Based on the statistics below, provide 4 concise, actionable insights and recommendations for the charity.
Focus on what to prioritize, resource gaps, and specific actions.

Dataset statistics:
{json.dumps(stats, ensure_ascii=False, default=str)}

Format your response as a numbered list. Be specific and practical."""
    return _chat(prompt)


def generate_needs_narrative(needs_df: pd.DataFrame, total_families: int) -> str:
    """Generate a procurement/action narrative from the aggregate needs table."""
    needs_list = needs_df.to_dict(orient="records")
    prompt = f"""You are a humanitarian aid logistics coordinator.
Below is an aggregated needs list computed from {total_families} families in the system.
Each entry shows how many families require a specific item and how urgent it is.

Aggregated needs data:
{json.dumps(needs_list, ensure_ascii=False, default=str)}

Write a clear, structured procurement and action report for the charity leadership. Include:
1. A brief executive summary (2-3 sentences)
2. Critical items to procure immediately (list with estimated quantities based on family counts)
3. High-priority items for the next distribution cycle
4. Medium-priority items to plan for
5. One specific logistical recommendation

Be concise and practical. Use the actual numbers from the data."""
    return _chat(prompt)


def answer_query(df: pd.DataFrame, question: str) -> tuple[pd.DataFrame, str]:
    """Convert a natural-language question to a pandas filter and return results + explanation."""
    col_info = "\n".join(
        f"  - {c} ({df[c].dtype}): e.g. {repr(df[c].dropna().iloc[0]) if not df[c].dropna().empty else 'N/A'}"
        for c in df.columns
    )
    prompt = f"""You are a data analyst. Convert the user's question into a pandas DataFrame filter expression.
The DataFrame is named 'df'. Return TWO things:
1. A Python filter expression (must start with 'df[')
2. A short English explanation of what you filtered

Column details:
{col_info}

User question: {question}

Reply in this exact format:
FILTER: df[<your filter here>]
EXPLANATION: <one sentence>

Rules:
- Use .str.contains(..., na=False) for text searches
- Boolean columns (is_widow, is_displaced etc.) are Python booleans
- family_size is numeric
- priority_tier values: CRITICAL, HIGH, MEDIUM, LOW
- Do not use eval(), only standard pandas operations"""

    result = _chat(prompt)

    filter_match = re.search(r"FILTER:\s*(df\[.+\])", result, re.DOTALL)
    explanation_match = re.search(r"EXPLANATION:\s*(.+)", result)

    explanation = explanation_match.group(1).strip() if explanation_match else "AI query result"
    filter_expr = filter_match.group(1).strip() if filter_match else None

    if not filter_expr:
        return pd.DataFrame(), "Could not parse a valid filter from the response."

    try:
        filtered = eval(filter_expr, {"df": df, "pd": pd, "__builtins__": {}})
        if isinstance(filtered, pd.DataFrame):
            return filtered, explanation
    except Exception as e:
        return pd.DataFrame(), f"Filter error: {e}"

    return pd.DataFrame(), explanation


def smart_query(df: pd.DataFrame, question: str) -> dict:
    """
    Route question to filter path or analytics path.
    Returns dict with keys:
      type: 'filter' | 'analytics' | 'error'
      explanation: str
      result_df: pd.DataFrame | None   (filter results or breakdown table)
      answer: str | None               (analytics answer text)
    """
    import needs as _needs

    # Step 1: classify intent
    classify_prompt = f"""Classify this question about humanitarian aid data. Reply with ONE word only: FILTER or ANALYTICS.

FILTER: questions that want to see a list of families matching some criteria.
Examples: "show homeless families", "widows with children", "families in Tartus", "critical priority families"

ANALYTICS: questions that want a calculated number or procurement quantity.
Examples: "how many food baskets", "how much rice do we need", "total families with medical needs", "budget estimate for Homs", "how many blankets", "what is the total cost"

Question: {question}

Reply with ONE word: FILTER or ANALYTICS"""

    intent = _chat(classify_prompt).strip().upper()
    intent = "ANALYTICS" if "ANALYTICS" in intent else "FILTER"

    if intent == "FILTER":
        result_df, explanation = answer_query(df, question)
        return {
            "type": "filter",
            "explanation": explanation,
            "result_df": result_df,
            "answer": None,
        }

    # Analytics path: ask Gemma4 to write a Python expression
    col_info = "\n".join(
        f"  - {c} ({df[c].dtype}): e.g. {repr(df[c].dropna().iloc[0]) if not df[c].dropna().empty else 'N/A'}"
        for c in df.columns
        if df[c].dtype != object or c in ("city", "governorate", "displacement_type", "housing_type", "need_type", "priority_tier")
    )

    analytics_prompt = f"""You are a humanitarian data analyst. Answer the user's question by writing Python code.

Available objects:
- df: pandas DataFrame with one row per family. Key columns:
{col_info}
- needs.compute_food_rations(filtered_df): returns (per_family_df, agg_items_df) — agg_items_df has columns [Item, Unit, Qty/basket, Total needed]
- needs.compute_medical_supplies(filtered_df): returns DataFrame with columns [Item, Unit, Families, Total Qty]
- needs.compute_financial_needs(filtered_df): returns (per_family_df, totals_dict) — totals_dict has category→USD and "Grand Total"
- needs.compute_aggregate_needs(filtered_df): returns DataFrame with [need_item, category, urgency, families_count, pct_of_total]

User question: {question}

Write Python code that:
1. Optionally filters df to a subset (e.g. by city/governorate)
2. Calls the appropriate needs function OR computes directly from df
3. Stores final answer as variable named `answer` (string with the key number/result)
4. Optionally stores a breakdown table as variable named `table` (pandas DataFrame or None)

Reply in this exact format:
CODE:
```python
<your code here — use df, needs, pd as available names>
```
EXPLANATION: <one sentence describing what you computed>

Rules:
- Use .str.contains(X, case=False, na=False) for location filtering
- `answer` must be a human-readable string (e.g. "1,234 food baskets needed" or "$45,200 total budget")
- `table` is optional; set table = None if no breakdown needed
- Do not use print(), return, or import statements"""

    result = _chat(analytics_prompt)

    code_match = re.search(r"CODE:\s*```python\s*([\s\S]+?)```", result)
    explanation_match = re.search(r"EXPLANATION:\s*(.+)", result)
    explanation = explanation_match.group(1).strip() if explanation_match else question

    if not code_match:
        # Fallback: try to answer directly with plain text
        fallback_prompt = f"""Answer this question about Syrian humanitarian aid data in 1-2 sentences.
The dataset has {len(df)} families, {int(df['family_size'].sum())} total members.
Key stats: {int(df['is_displaced'].sum())} displaced, {int(df['has_medical'].sum())} with medical needs,
{int(df['is_homeless'].sum())} homeless.
Cities represented: {', '.join(df['city'].value_counts().head(5).index.tolist())}

Question: {question}"""
        fallback_answer = _chat(fallback_prompt)
        return {
            "type": "analytics",
            "explanation": explanation,
            "result_df": None,
            "answer": fallback_answer.strip(),
        }

    code = code_match.group(1).strip()

    # Execute in sandboxed namespace
    namespace = {
        "df": df.copy(),
        "needs": _needs,
        "pd": pd,
        "answer": None,
        "table": None,
    }
    try:
        exec(code, namespace)  # noqa: S102
        answer = str(namespace.get("answer") or "Computation complete — see table below.")
        table = namespace.get("table")
        if not isinstance(table, pd.DataFrame):
            table = None
        return {
            "type": "analytics",
            "explanation": explanation,
            "result_df": table,
            "answer": answer,
        }
    except Exception as e:
        return {
            "type": "error",
            "explanation": f"Computation error: {e}",
            "result_df": None,
            "answer": None,
        }

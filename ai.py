"""Gemma4 (via Ollama) interface for ReliefFlow."""
import json
import re

import ollama
import pandas as pd

MODEL = "gemma4:latest"


def _chat(prompt: str, system: str | None = None) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = ollama.chat(model=MODEL, messages=messages)
    return resp["message"]["content"]


def _extract_json_object(text: str) -> dict:
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


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
    prompt = f"""You are analyzing an Arabic humanitarian aid / charity record for a family in need.
Extract the key information and return ONLY a JSON object — no explanation, no markdown.

Record fields:
- Contact name: {row.get("contact_name", "N/A")}
- Family size: {row.get("family_size", "N/A")}
- Current address: {row.get("address", "N/A")}
- Humanitarian situation / displacement reason: {row.get("humanitarian_situation", "N/A")}
- Type of need stated: {row.get("need_type", "N/A")}
- Additional needs: {row.get("additional_needs", "N/A")}

Return exactly this JSON structure:
{{
  "family_head_role": "father|mother|widow|orphan-family|unknown",
  "clean_name": "person's actual name without role prefix",
  "is_widow": true_or_false,
  "is_orphan_family": true_or_false,
  "is_displaced": true_or_false,
  "has_medical_condition": true_or_false,
  "medical_conditions": ["list conditions if any"],
  "has_disability": true_or_false,
  "is_unemployed": true_or_false,
  "housing_status": "homeless|renting|owner|unknown",
  "city": "extracted city name in Arabic",
  "summary_en": "one sentence English summary of this family situation"
}}"""
    return _extract_json_object(_chat(prompt))


def generate_needs(row: pd.Series) -> list[dict]:
    """Generate a specific needs list for a family using Gemma4."""
    prompt = f"""You are a humanitarian aid coordinator. Based on the family record below, list their specific needs.
Return ONLY a JSON array — no explanation, no markdown.

Contact: {row.get("contact_name", "N/A")}
Family size: {row.get("family_size", "N/A")} members
Address: {row.get("address", "N/A")}
Situation: {row.get("humanitarian_situation", "N/A")}
Need type stated: {row.get("need_type", "N/A")}
Additional info: {row.get("additional_needs", "N/A")}
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

    # Extract filter line
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

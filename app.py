"""
Sales Forecasting App — Streamlit
Uses a pre-trained CatBoostRegressor to predict next-day sales.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from catboost import CatBoostRegressor
from datetime import datetime, date, timedelta
import re

import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("LLM_API_KEY")
if api_key:
    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
    )
else:
    client = None

LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Sales Forecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS — product-grade dark UI
# ──────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Reset & Global ── */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="collapsedControl"] { visibility: visible !important; }


    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #09090B;
        border-right: 1px solid #1C1C22;
    }

    /* ── Segmented pill toggle ── */
    .pill-toggle-wrap {
        margin-bottom: 20px;
    }
    .pill-toggle-wrap .stRadio > div {
        flex-direction: row !important;
        background: #111114 !important;
        border: 1px solid #1C1C22 !important;
        border-radius: 14px !important;
        padding: 4px !important;
        gap: 0 !important;
    }
    .pill-toggle-wrap .stRadio > label {
        display: none !important;
    }
    .pill-toggle-wrap .stRadio > div > label {
        flex: 1 !important;
        text-align: center !important;
        padding: 12px 8px !important;
        border-radius: 11px !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px !important;
        cursor: pointer !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        color: #63636E !important;
        background: transparent !important;
        border: none !important;
        margin: 0 !important;
        line-height: 1.3 !important;
    }
    .pill-toggle-wrap .stRadio > div > label:has(input:checked) {
        background: #EDEDEF !important;
        color: #09090B !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
    }
    .pill-toggle-wrap .stRadio > div > label > div:first-child {
        display: none !important;
    }
    .pill-toggle-wrap .stRadio > div > label > div[data-testid="stMarkdownContainer"] p {
        font-size: 13px !important;
        font-weight: 600 !important;
    }

    /* ── Main area bg ── */
    .stApp {
        background: #09090B;
    }
    .block-container {
        padding-top: 2rem !important;
        max-width: 1120px;
    }

    /* ── Metric cards ── */
    div[data-testid="stMetric"] {
        background: #111114;
        border: 1px solid #1C1C22;
        border-radius: 12px;
        padding: 18px 20px;
        transition: border-color 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        border-color: #27272F;
    }
    div[data-testid="stMetric"] label {
        color: #63636E !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        letter-spacing: 0.3px !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #EDEDEF !important;
        font-weight: 600 !important;
    }

    /* ── Typography helpers ── */
    .page-title {
        font-size: 22px;
        font-weight: 700;
        color: #EDEDEF;
        letter-spacing: -0.3px;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .page-title .dot {
        width: 8px; height: 8px;
        background: #22C55E;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 8px rgba(34,197,94,0.4);
    }
    .page-subtitle {
        color: #63636E;
        font-size: 13px;
        font-weight: 400;
        margin: 4px 0 0 18px;
    }
    .label-sm {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        color: #53535E;
        margin-bottom: 6px;
    }

    /* ── Prediction hero ── */
    .pred-card {
        background: #111114;
        border: 1px solid #1C1C22;
        border-radius: 14px;
        padding: 32px;
        position: relative;
        overflow: hidden;
    }
    .pred-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #6D28D9, #8B5CF6, #A78BFA);
    }
    .pred-value {
        font-family: 'Inter', sans-serif;
        font-size: 56px;
        font-weight: 800;
        color: #EDEDEF;
        letter-spacing: -2px;
        line-height: 1;
        margin: 8px 0 0 0;
    }
    .pred-unit {
        font-size: 18px;
        font-weight: 500;
        color: #63636E;
        letter-spacing: -0.5px;
    }
    .pred-meta {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        color: #53535E;
        margin-top: 14px;
    }
    .pred-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(109, 40, 217, 0.12);
        border: 1px solid rgba(109, 40, 217, 0.2);
        color: #A78BFA;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        padding: 5px 12px;
        border-radius: 8px;
    }

    /* ── Chart container ── */
    .chart-wrap {
        background: #111114;
        border: 1px solid #1C1C22;
        border-radius: 14px;
        padding: 24px;
        margin-top: 20px;
    }
    .chart-title {
        font-size: 14px;
        font-weight: 600;
        color: #EDEDEF;
        margin-bottom: 4px;
    }
    .chart-sub {
        font-size: 12px;
        color: #53535E;
        margin-bottom: 16px;
    }

    /* ── AI Insights block ── */
    .insights-card {
        background: #111114;
        border: 1px solid #1C1C22;
        border-radius: 14px;
        padding: 24px 28px;
        margin-top: 20px;
        position: relative;
        overflow: hidden;
    }
    .insights-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, #0EA5E9, #6D28D9);
    }
    .insights-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 16px;
    }
    .insights-icon {
        width: 32px; height: 32px;
        background: rgba(14, 165, 233, 0.1);
        border: 1px solid rgba(14, 165, 233, 0.15);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 16px;
    }
    .insights-label {
        font-size: 14px;
        font-weight: 600;
        color: #EDEDEF;
    }
    .insights-sublabel {
        font-size: 11px;
        color: #53535E;
        font-weight: 400;
    }
    .insight-row {
        display: flex;
        align-items: flex-start;
        gap: 12px;
        padding: 12px 0;
        border-bottom: 1px solid #1C1C22;
    }
    .insight-row:last-child {
        border-bottom: none;
        padding-bottom: 0;
    }
    .insight-dot {
        width: 6px; height: 6px;
        border-radius: 50%;
        margin-top: 6px;
        flex-shrink: 0;
    }
    .insight-dot.high   { background: #F59E0B; box-shadow: 0 0 6px rgba(245,158,11,0.3); }
    .insight-dot.season  { background: #22C55E; box-shadow: 0 0 6px rgba(34,197,94,0.3); }
    .insight-dot.trend   { background: #8B5CF6; box-shadow: 0 0 6px rgba(139,92,246,0.3); }
    .insight-dot.neutral { background: #53535E; }
    .insight-text {
        font-size: 13px;
        color: #A1A1AA;
        line-height: 1.5;
    }
    .insight-text strong {
        color: #EDEDEF;
    }
    .insights-placeholder {
        text-align: center;
        padding: 20px 0;
        color: #53535E;
        font-size: 13px;
    }
    .insights-placeholder .placeholder-icon {
        font-size: 28px;
        margin-bottom: 8px;
        opacity: 0.5;
    }

    /* ── Sidebar styling ── */
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 4px 0 20px 0;
        border-bottom: 1px solid #1C1C22;
        margin-bottom: 20px;
    }
    .sidebar-brand-icon {
        width: 34px; height: 34px;
        background: linear-gradient(135deg, #6D28D9, #8B5CF6);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
    }
    .sidebar-brand-text {
        font-size: 15px;
        font-weight: 700;
        color: #EDEDEF;
        letter-spacing: -0.3px;
    }
    .sidebar-brand-sub {
        font-size: 11px;
        color: #53535E;
    }
    .sidebar-section {
        margin-top: 20px;
    }

    /* ── AI input area ── */
    .ai-input-wrap {
        background: #111114;
        border: 1px solid #1C1C22;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 16px;
    }
    .ai-chip {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background: rgba(14, 165, 233, 0.08);
        border: 1px solid rgba(14, 165, 233, 0.12);
        color: #0EA5E9;
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        padding: 3px 10px;
        border-radius: 6px;
        margin-bottom: 10px;
    }

    /* ── Button overrides ── */
    .stButton > button {
        background: #6D28D9 !important;
        color: #fff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        padding: 0.5rem 1.5rem !important;
        transition: all 0.15s ease !important;
    }
    .stButton > button:hover {
        background: #7C3AED !important;
        box-shadow: 0 4px 16px rgba(109, 40, 217, 0.3) !important;
    }
    .stButton > button:active {
        transform: scale(0.98) !important;
    }

    /* ── Empty state ── */
    .empty-state {
        text-align: center;
        padding: 100px 0 60px 0;
    }
    .empty-icon {
        font-size: 48px;
        margin-bottom: 16px;
        opacity: 0.4;
    }
    .empty-title {
        font-size: 16px;
        font-weight: 600;
        color: #A1A1AA;
        margin-bottom: 6px;
    }
    .empty-desc {
        font-size: 13px;
        color: #53535E;
        max-width: 360px;
        margin: 0 auto;
        line-height: 1.6;
    }

    /* ── Extracted params banner ── */
    .parsed-banner {
        background: rgba(109, 40, 217, 0.06);
        border: 1px solid rgba(109, 40, 217, 0.12);
        border-radius: 10px;
        padding: 14px 18px;
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 16px;
    }
    .parsed-banner-dot {
        width: 8px; height: 8px;
        background: #8B5CF6;
        border-radius: 50%;
        flex-shrink: 0;
    }
    .parsed-banner-text {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        color: #A78BFA;
    }

    /* ── Divider ── */
    hr { border-color: #1C1C22 !important; }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        font-size: 13px !important;
        font-weight: 500 !important;
        color: #A1A1AA !important;
    }

    /* ── Tabs ── */
    button[data-baseweb="tab"] {
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        font-size: 13px !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #27272F; border-radius: 3px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────
# Load data & model (cached)
# ──────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("train.csv", parse_dates=["date"])
    df.sort_values(["store", "item", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


@st.cache_resource
def load_model():
    model = CatBoostRegressor()
    model.load_model("sales_forcasting_single_day.cbm")
    return model


data = load_data()
model = load_model()


# ──────────────────────────────────────────────
# Forecasting logic (DO NOT MODIFY)
# ──────────────────────────────────────────────
def forecast(store_id: int, item_id: int, target_date: datetime) -> dict:
    """
    Build features and predict next-day sales.
    Returns dict with prediction value and history df.
    """
    history = data[
        (data["store"] == store_id) & (data["item"] == item_id)
    ].sort_values("date")

    if len(history) < 30:
        return {"error": "Not enough historical data (need ≥ 30 rows)."}

    month = target_date.month
    year = target_date.year
    day = target_date.day
    days_of_week = target_date.weekday()
    is_weekend = int(days_of_week >= 5)

    lag_1 = history["sales"].iloc[-1]
    lag_7 = history["sales"].iloc[-7]
    lag_30 = history["sales"].iloc[-30]
    rolling_mean_7 = history["sales"].tail(7).mean()
    rolling_mean_30 = history["sales"].tail(30).mean()

    features = pd.DataFrame(
        [
            {
                "store": str(store_id),
                "item": str(item_id),
                "month": month,
                "year": year,
                "day": day,
                "days_of_week": days_of_week,
                "is_weekend": is_weekend,
                "lag_1": lag_1,
                "lag_7": lag_7,
                "lag_30": lag_30,
                "rolling_mean_7": rolling_mean_7,
                "rolling_mean_30": rolling_mean_30,
            }
        ]
    )

    prediction = model.predict(features)[0]
    return {
        "prediction": round(prediction),
        "raw": prediction,
        "history": history,
        "features": features,
    }


# ──────────────────────────────────────────────
# LLM integration
# ──────────────────────────────────────────────
def parse_forecast_intent(
    message: str,
    available_stores: list[int],
    available_items: list[int],
    last_known_date: date,
) -> dict | None:
    """
    Use the LLM to extract forecast parameters from a natural-language message.

    Returns:
        {"store_id": int, "item_id": int, "target_date": "YYYY-MM-DD"}
        or None if extraction fails.
    """
    if client is None:
        return None
    llm_prompt = f"""
You extract sales forecast parameters from a user message.

Return ONLY valid JSON.

Valid store IDs:
{available_stores}

Valid item IDs:
{available_items}

Latest available training date:
{last_known_date.isoformat()}

If the user is NOT asking for a forecast, return:
{{"intent": null}}

If the user IS asking for a forecast, return:
{{
  "intent": "forecast",
  "store_id": 1,
  "item_id": 1,
  "target_date": "YYYY-MM-DD"
}}

Rules:
- Use only JSON.
- No markdown.
- No explanation text.
- target_date must be ISO format.
- If the date is relative like "tomorrow" or "next week", resolve it to an absolute date.
- Prefer values that exist in the provided store/item lists.
- If you cannot confidently extract all fields, return {{"intent": null}}.

User message:
{message}
"""

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": "Extract forecast intent and return JSON only."},
                {"role": "user", "content": llm_prompt},
            ],
        )

        raw = response.choices[0].message.content.strip()

        # Strip accidental code fences
        if raw.startswith("```"):
            raw = raw.replace("```json", "").replace("```", "").strip()

        data = json.loads(raw)

    except Exception:
        return None

    if data.get("intent") is None:
        return None

    try:
        return {
            "store_id": int(data["store_id"]),
            "item_id": int(data["item_id"]),
            "target_date": data["target_date"],
        }
    except Exception:
        return None


def parse_natural_language(prompt: str) -> dict | None:
    """
    Parse a natural-language forecast request.
    Tries the LLM first, falls back to regex.

    Returns:
        {"store_id": int, "item_id": int, "target_date": str} or None
    """
    # ── Try LLM extraction first ──
    try:
        last_date = data["date"].max().date()
        result = parse_forecast_intent(
            message=prompt,
            available_stores=list(range(1, 11)),
            available_items=list(range(1, 51)),
            last_known_date=last_date,
        )
        if result is not None:
            return result
    except Exception:
        pass  # Fall through to regex

    # ── Regex fallback ──
    store_match = re.search(r"[Ss]tore\s*(\d+)", prompt)
    item_match = re.search(r"[Ii]tem\s*(\d+)", prompt)
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", prompt)

    # Try to match written-out dates like "February 10 2018"
    if not date_match:
        month_names = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12,
        }
        written_date = re.search(
            r"(January|February|March|April|May|June|July|August|September|October|November|December)"
            r"\s+(\d{1,2}),?\s+(\d{4})",
            prompt,
            re.IGNORECASE,
        )
        if written_date:
            m = month_names[written_date.group(1).lower()]
            d = int(written_date.group(2))
            y = int(written_date.group(3))
            date_str = f"{y}-{m:02d}-{d:02d}"
            date_match = type("obj", (object,), {"group": lambda self, x=0: date_str})()

    if store_match and item_match and date_match:
        return {
            "store_id": int(store_match.group(1)),
            "item_id": int(item_match.group(1)),
            "target_date": date_match.group(0) if callable(getattr(date_match, "group", None)) else date_match.group(),
        }
    return None


def generate_ai_insights(store_id, item_id, target_date, prediction, features) -> list[dict]:
    """
    Generate AI-powered insights explaining the forecast using Groq LLM API.
    Falls back to rule-based insights if client is not configured or API fails.
    """
    is_weekend = int(target_date.weekday() >= 5)
    day_name = target_date.strftime("%A")
    month_name = target_date.strftime("%B")
    year = target_date.year

    lag_1 = features["lag_1"].values[0] if "lag_1" in features else 0
    lag_7 = features["lag_7"].values[0] if "lag_7" in features else 0
    lag_30 = features["lag_30"].values[0] if "lag_30" in features else 0
    rolling_7 = features["rolling_mean_7"].values[0] if "rolling_mean_7" in features else 0
    rolling_30 = features["rolling_mean_30"].values[0] if "rolling_mean_30" in features else 0

    # Define helper fallback function (our rule-based logic)
    def get_fallback_insights():
        insights = []
        # Weekend insight
        if is_weekend:
            insights.append({
                "type": "high",
                "text": f"<strong>{day_name}</strong> — weekends typically show different demand patterns. "
                        f"Consumer foot traffic tends to spike on Saturdays.",
            })
        else:
            insights.append({
                "type": "neutral",
                "text": f"<strong>{day_name}</strong> — a regular weekday. Sales usually follow "
                        f"a steady mid-week baseline for this store.",
            })

        # Trend insight
        if rolling_7 > rolling_30 * 1.1:
            insights.append({
                "type": "trend",
                "text": f"<strong>Upward momentum</strong> — the 7-day average ({rolling_7:.0f}) is "
                        f"above the 30-day average ({rolling_30:.0f}), suggesting rising demand.",
            })
        elif rolling_7 < rolling_30 * 0.9:
            insights.append({
                "type": "trend",
                "text": f"<strong>Cooling demand</strong> — recent 7-day average ({rolling_7:.0f}) has "
                        f"dipped below the 30-day trend ({rolling_30:.0f}).",
            })

        # Seasonal insight
        if target_date.month in [11, 12]:
            insights.append({
                "type": "season",
                "text": f"<strong>Holiday season ({month_name})</strong> — historically a high-demand "
                        f"period. Expect elevated sales from festive shopping.",
            })
        elif target_date.month in [1, 2]:
            insights.append({
                "type": "season",
                "text": f"<strong>Post-holiday ({month_name})</strong> — demand often normalizes after "
                        f"the festive spike. Watch for clearance-driven bumps.",
            })
        else:
            insights.append({
                "type": "season",
                "text": f"<strong>{month_name}</strong> — a standard seasonal period for this category. "
                        f"No major holiday effects expected.",
            })
        return insights

    if client is None:
        return get_fallback_insights()

    # Call LLM
    llm_prompt = f"""
Analyze the predicted sales of {prediction} units for store {store_id}, item {item_id} on {target_date.strftime('%Y-%m-%d')} ({day_name}, {month_name} {year}).

Here are the details:
- Lag 1 (sales from yesterday): {lag_1:.0f} units
- Lag 7 (sales from 7 days ago): {lag_7:.0f} units
- Lag 30 (sales from 30 days ago): {lag_30:.0f} units
- 7-day rolling average sales: {rolling_7:.1f} units
- 30-day rolling average sales: {rolling_30:.1f} units
- Is weekend: {'Yes' if is_weekend else 'No'}

Determine the main drivers for this forecast. Compare the predicted value ({prediction}) against the rolling averages and lags. Explain why the sales are at this level.
Provide 2 to 3 distinct bullet points analyzing the prediction.

Return ONLY a valid JSON array of objects. Do not include markdown code blocks, explanation text, or anything else.
Each object in the array must have exactly these keys:
- "type": Choose from: "high" (positive drivers/spikes), "season" (calendar/monthly/holiday effects), "trend" (momentum, comparison of 7-day vs 30-day), "neutral" (general notes/baseline).
- "text": A concise, HTML-safe explanation. You can and should use <strong>...</strong> tags to highlight key phrases (e.g. "<strong>weekend surge</strong>", "<strong>upward momentum</strong>", etc.). Keep explanations short, professional, and specific.

Example JSON output format:
[
  {{"type": "high", "text": "<strong>Weekend demand spike</strong> — Saturday foot traffic drives elevated sales."}},
  {{"type": "trend", "text": "<strong>Rising momentum</strong> — the 7-day average of {rolling_7:.1f} is above the 30-day average."}}
]
"""

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0.2,
            messages=[
                {"role": "system", "content": "You are a professional retail data analyst. Return JSON arrays only."},
                {"role": "user", "content": llm_prompt},
            ],
        )

        raw = response.choices[0].message.content.strip()

        # Strip accidental code fences
        if raw.startswith("```"):
            raw = raw.replace("```json", "").replace("```", "").strip()

        data = json.loads(raw)
        if isinstance(data, list):
            # Validate types of insights
            validated_insights = []
            for item in data:
                if isinstance(item, dict) and "type" in item and "text" in item:
                    # ensure valid type
                    if item["type"] not in ["high", "season", "trend", "neutral"]:
                        item["type"] = "neutral"
                    validated_insights.append(item)
            if validated_insights:
                return validated_insights
    except Exception as e:
        # If anything fails, fall back to rule-based insights
        pass

    return get_fallback_insights()


# ──────────────────────────────────────────────
# Data Analysis helper
# ──────────────────────────────────────────────
def answer_data_query(query: str, df: pd.DataFrame) -> dict:
    """
    Use LLM to generate pandas code for a data query, execute it on `df`, and summarize the results.
    """
    if client is None:
        return {"error": "LLM client not configured. Cannot perform data analysis."}
    
    # 1. Generate code
    prompt_code = f"""
You are an expert pandas programmer. Write pandas code to answer this user question about a sales dataset.

The dataset is loaded in a pandas DataFrame named `df`.
`df` has the following columns:
- `date`: datetime64[ns]
- `store`: int64 (ranges from 1 to 10)
- `item`: int64 (ranges from 1 to 50)
- `sales`: int64 (daily sales volume)

User question: "{query}"

Rules:
- Write ONLY a single expression or a small block of pandas code.
- Save the final result into a variable named `result`.
- Keep the result small (e.g., use `.head(10)`, `.sum()`, `.mean()`, or group/aggregate) so it can be converted to markdown and summarized.
- Do NOT import other libraries unless necessary (like numpy, datetime).
- Return ONLY a JSON object with two keys: "explanation" (what the query does) and "code" (the python code to execute). Do not use markdown code fences.
- If the question is completely unrelated to the sales dataset (e.g. general knowledge, greetings, math), return a JSON object with the "error" key explaining that the question is unrelated, and an empty code string.

Example JSON output:
{{
  "explanation": "Group sales by month and sum them to find the highest-selling months.",
  "code": "result = df.groupby(df['date'].dt.strftime('%B'))['sales'].sum().reset_index().sort_values('sales', ascending=False)"
}}
"""

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": "You write pandas code to query a sales dataframe and return JSON only."},
                {"role": "user", "content": prompt_code},
            ]
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.replace("```json", "").replace("```", "").strip()
        
        parsed = json.loads(raw)
        if "error" in parsed:
            return {"error": parsed["error"]}
            
        code = parsed["code"]
        explanation = parsed["explanation"]
    except Exception as e:
        return {"error": f"Failed to generate query code: {str(e)}"}

    # 2. Execute code
    try:
        # Prepare context for execution
        local_vars = {"df": df, "pd": pd, "np": np}
        exec(code, {}, local_vars)
        result = local_vars.get("result")
    except Exception as e:
        return {
            "error": f"Failed to execute generated query code:\n{code}\nError: {str(e)}",
            "code": code,
            "explanation": explanation
        }

    # 3. Format result
    # Convert result to string or markdown
    if isinstance(result, pd.DataFrame):
        result_str = result.to_markdown(index=False)
    elif isinstance(result, pd.Series):
        result_str = result.to_markdown()
    else:
        result_str = str(result)

    prompt_summary = f"""
You are a retail data analyst. A user asked this question:
"{query}"

To answer this, we executed the following pandas query:
`{code}`

The query returned the following results:
{result_str}

Write a professional, clear, and user-friendly answer summarizing these results. Use bullet points or formatted text where appropriate. Highlight key insights using **bold** text. Keep it concise.
"""
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0.2,
            messages=[
                {"role": "system", "content": "You are a professional retail data analyst summarizing query results."},
                {"role": "user", "content": prompt_summary},
            ]
        )
        summary = response.choices[0].message.content.strip()
        return {
            "success": True,
            "explanation": explanation,
            "code": code,
            "result_obj": result,
            "result_raw": result_str,
            "summary": summary
        }
    except Exception as e:
        return {
            "success": True,
            "explanation": explanation,
            "code": code,
            "result_obj": result,
            "result_raw": result_str,
            "summary": f"Query ran successfully, but could not generate summary: {str(e)}"
        }


# ──────────────────────────────────────────────
# Plotly chart helper
# ──────────────────────────────────────────────
def build_chart(history: pd.DataFrame, target_date, prediction_value):
    recent = history.tail(90).copy()

    fig = go.Figure()

    # Historical line
    fig.add_trace(
        go.Scatter(
            x=recent["date"],
            y=recent["sales"],
            mode="lines",
            name="Historical",
            line=dict(color="#8B5CF6", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(139, 92, 246, 0.04)",
            hovertemplate="<b>%{x|%b %d}</b><br>%{y} units<extra></extra>",
        )
    )

    # Forecast marker
    fig.add_trace(
        go.Scatter(
            x=[pd.Timestamp(target_date)],
            y=[prediction_value],
            mode="markers+text",
            name="Forecast",
            marker=dict(
                color="#8B5CF6",
                size=10,
                symbol="diamond",
                line=dict(color="#111114", width=2),
            ),
            text=[f" {prediction_value}"],
            textposition="top right",
            textfont=dict(color="#A78BFA", size=12, family="Inter, sans-serif"),
            hovertemplate="<b>Forecast</b><br>%{y} units<extra></extra>",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=16, t=8, b=0),
        height=320,
        showlegend=False,
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            title="",
            tickfont=dict(family="Inter", size=11, color="#53535E"),
            tickformat="%b %d",
        ),
        yaxis=dict(
            gridcolor="rgba(28,28,34,0.8)",
            griddash="dot",
            zeroline=False,
            title="",
            tickfont=dict(family="Inter", size=11, color="#53535E"),
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#1C1C22",
            bordercolor="#27272F",
            font=dict(family="Inter", size=12, color="#EDEDEF"),
        ),
    )
    return fig


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-icon">📈</div>
            <div>
                <div class="sidebar-brand-text">Forecast</div>
                <div class="sidebar-brand-sub">Sales Intelligence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="pill-toggle-wrap">', unsafe_allow_html=True)
    mode = st.radio(
        "Mode",
        ["⚙️  Manual", "✦  AI Assistant"],
        horizontal=True,
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Normalize mode value
    mode = "AI Assistant" if "AI" in mode else "Manual"

    st.markdown("---")

    if mode == "Manual":
        st.markdown('<div class="label-sm">Configuration</div>', unsafe_allow_html=True)
        store_id = st.selectbox("Store", options=list(range(1, 11)), index=0)
        item_id = st.selectbox("Item", options=list(range(1, 51)), index=0)
        target_date = st.date_input(
            "Forecast date",
            value=datetime(2018, 1, 31),
        )
        st.markdown("")
        run_forecast = st.button("Predict", use_container_width=True)

    else:
        st.markdown(
            '<span class="ai-chip">✦ natural language</span>',
            unsafe_allow_html=True,
        )
        nl_input = st.text_area(
            "Describe your forecast",
            height=100,
            placeholder="Predict sales for Store 10 Item 5 on 2018-01-31",
            label_visibility="collapsed",
        )
        st.markdown("")
        run_ai = st.button("Run forecast", use_container_width=True)

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size:11px; color:#53535E; line-height:1.7;">
        CatBoost model · Lag & rolling features<br>
        Trained on historical store-item data
        </div>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
# Main — Header
# ──────────────────────────────────────────────
st.markdown(
    """
    <div style="padding-bottom: 6px;">
        <div class="page-title"><span class="dot"></span> Sales Forecaster</div>
        <div class="page-subtitle">Next-day demand prediction for store–item pairs</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("---")


# ──────────────────────────────────────────────
# Process forecast / analysis
# ──────────────────────────────────────────────
result = None
analysis_result = None
display_store = None
display_item = None
display_date = None
is_ai_mode = mode == "AI Assistant"

if mode == "Manual" and run_forecast:
    display_store = store_id
    display_item = item_id
    display_date = target_date
    with st.spinner("Predicting…"):
        result = forecast(store_id, item_id, pd.Timestamp(target_date))

elif mode == "AI Assistant" and run_ai:
    if nl_input.strip():
        parsed = parse_natural_language(nl_input)
        if parsed:
            display_store = parsed["store_id"]
            display_item = parsed["item_id"]
            display_date = pd.Timestamp(parsed["target_date"])

            if not (1 <= display_store <= 10):
                st.error("Store ID must be between 1 and 10.")
            elif not (1 <= display_item <= 50):
                st.error("Item ID must be between 1 and 50.")
            else:
                st.markdown(
                    f"""
                    <div class="parsed-banner">
                        <div class="parsed-banner-dot"></div>
                        <div class="parsed-banner-text">
                            Extracted → store={display_store}  item={display_item}  date={display_date.strftime('%Y-%m-%d')}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                with st.spinner("Predicting…"):
                    result = forecast(display_store, display_item, display_date)
        else:
            with st.spinner("Analyzing dataset…"):
                analysis_result = answer_data_query(nl_input, data)
    else:
        st.warning("Enter a forecast request above.")


# ──────────────────────────────────────────────
# Display results
# ──────────────────────────────────────────────
if result is not None:
    if "error" in result:
        st.error(result["error"])
    else:
        pred = result["prediction"]
        raw = result["raw"]
        hist = result["history"]
        feats = result["features"]
        dt = pd.Timestamp(display_date)

        # ── Prediction card ──
        st.markdown(
            f"""
            <div class="pred-card">
                <div class="pred-badge">⚡ Forecast result</div>
                <div class="pred-value">{pred} <span class="pred-unit">units</span></div>
                <div class="pred-meta">
                    Store {display_store} · Item {display_item} · {dt.strftime('%b %d, %Y')} · raw: {raw:.2f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("")

        # ── KPI metrics ──
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Yesterday (lag-1)", f"{feats['lag_1'].values[0]:.0f}")
        with c2:
            st.metric("Last week (lag-7)", f"{feats['lag_7'].values[0]:.0f}")
        with c3:
            st.metric("7-day avg", f"{feats['rolling_mean_7'].values[0]:.1f}")
        with c4:
            st.metric("30-day avg", f"{feats['rolling_mean_30'].values[0]:.1f}")

        # ── Chart ──
        st.markdown(
            """
            <div class="chart-wrap">
                <div class="chart-title">Sales History</div>
                <div class="chart-sub">Last 90 days with forecast point</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        fig = build_chart(hist, display_date, pred)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # ── AI Insights block (only in AI Assistant mode) ──
        if is_ai_mode:
            insights = generate_ai_insights(
                display_store, display_item, dt, pred, feats
            )
            insights_html = ""
            for ins in insights:
                dot_class = ins.get("type", "neutral")
                insights_html += f"""
                <div class="insight-row">
                    <div class="insight-dot {dot_class}"></div>
                    <div class="insight-text">{ins['text']}</div>
                </div>
                """

            if not insights:
                insights_html = """
                <div class="insights-placeholder">
                    <div class="placeholder-icon">🔮</div>
                    <div>No insights generated. Connect an LLM to enable analysis.</div>
                </div>
                """

            st.markdown(
                f"""
                <div class="insights-card">
                    <div class="insights-header">
                        <div class="insights-icon">🧠</div>
                        <div>
                            <div class="insights-label">AI Insights</div>
                            <div class="insights-sublabel">Why is this prediction high or low?</div>
                        </div>
                    </div>
                    {insights_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── Feature table ──
        with st.expander("View model input features"):
            st.dataframe(feats, use_container_width=True, hide_index=True)

elif analysis_result is not None:
    if "error" in analysis_result:
        st.error(analysis_result["error"])
        st.info("Try asking about historical sales trends, top-selling items, best performing stores, or seasonal patterns (e.g., 'which months has the highest sales?').")
    else:
        explanation = analysis_result["explanation"]
        code = analysis_result["code"]
        summary = analysis_result["summary"]
        raw_obj = analysis_result.get("result_obj")
        
        st.markdown(
            f"""
            <div class="insights-card" style="margin-top: 0;">
                <div class="insights-header">
                    <div class="insights-icon" style="background: rgba(139, 92, 246, 0.1); border-color: rgba(139, 92, 246, 0.15);">📊</div>
                    <div>
                        <div class="insights-label">Data Analysis Result</div>
                        <div class="insights-sublabel">{explanation}</div>
                    </div>
                </div>
                <div style="color: #EDEDEF; font-size: 14px; line-height: 1.6; margin-top: 10px;">
                    {summary}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        if isinstance(raw_obj, (pd.DataFrame, pd.Series)):
            st.markdown('<div class="chart-wrap" style="margin-top: 20px;">', unsafe_allow_html=True)
            st.markdown('<div class="chart-title">Data Table</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.dataframe(raw_obj, use_container_width=True)
            
        with st.expander("View generated pandas query"):
            st.code(code, language="python")

else:
    # ── Empty state ──
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-icon">⌁</div>
            <div class="empty-title">Ready to forecast</div>
            <div class="empty-desc">
                Select a store and item from the sidebar, or switch to
                AI Assistant mode and describe your forecast in plain English.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
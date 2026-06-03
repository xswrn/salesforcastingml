# Prompt — Sales Forecast Dashboard + Chat Panel

Build a production Streamlit app for sales forecasting. Prioritise clean,
minimal, well-padded layout over visual complexity.

---

## Files on disk (same directory as app.py)

| File | Purpose |
|------|---------|
| `sales_forcasting_single_day.cbm` | Trained CatBoost model |
| `train.csv` | Sales data — columns: `date`, `store`, `item`, `sales` |

---

## Prediction function (do not change the logic, only wrap the UI around it)

```python
def predict_next_day(model, df, store_id, item_id, target_date):
    # Feature engineering
    target_date = pd.Timestamp(target_date)
    history = df[(df["store"]==store_id)&(df["item"]==item_id)].sort_values("date").set_index("date")["sales"]
    lookback = history[history.index < target_date]

    lag_1  = lookback.iloc[-1]  if len(lookback)>=1  else np.nan
    lag_7  = lookback.iloc[-7]  if len(lookback)>=7  else np.nan
    lag_30 = lookback.iloc[-30] if len(lookback)>=30 else np.nan
    roll_7  = lookback.tail(7).mean()  if len(lookback)>=7  else lookback.mean()
    roll_30 = lookback.tail(30).mean() if len(lookback)>=30 else lookback.mean()

    dow = target_date.dayofweek
    features = pd.DataFrame([{
        "store": str(store_id), "item": str(item_id),
        "month": target_date.month, "year": target_date.year, "day": target_date.day,
        "days_of_week": dow, "is_weekend": int(dow>=5),
        "lag_1": lag_1, "lag_7": lag_7, "lag_30": lag_30,
        "rolling_mean_7": roll_7, "rolling_mean_30": roll_30,
    }])
    return max(0.0, float(model.predict(features)[0]))
```

---

## Layout

Two columns, side by side, full width (`st.set_page_config(layout="wide")`):

```
┌──────── Sidebar ─────────┐  ┌────────── Dashboard (70%) ──────────┐  ┌── Chat (30%) ──┐
│ Store selector           │  │ Page title (Store # · Item #)        │  │ Header         │
│ Item selector            │  │                                      │  │ Messages area  │
│ Date picker              │  │ 5 KPI metric cards in a row          │  │ Input + send   │
│ Dataset info             │  │ Historical sales chart (90-day)      │  └────────────────┘
└──────────────────────────┘  │ Monthly seasonality bar chart        │
                              │ Raw data expander                    │
                              └──────────────────────────────────────┘
```

---

## Design rules — read these carefully

**Spacing**
- Block container padding: `2rem 2.5rem`
- Consistent `gap="large"` between columns
- Cards: `padding: 20px 24px`, uniform height within each row

**Colour — one palette, used consistently**
- Background: `#0a0f1a`
- Surface (cards, sidebar): `#0f1724`
- Border: `#1a2744`
- Accent: `#4f8ef7`
- Text primary: `#e2eaf6`
- Text muted: `#48607e`

**Typography — one font family only**
- Import Inter from Google Fonts
- Use three weights only: 400, 500, 700
- No mixing of font families

**Charts**
- Transparent background, no border box around them
- Single accent line colour, muted fill
- Hide the Plotly mode bar
- Height: history chart 320px, monthly bar 220px

**KPI cards**
- Flat background (no gradient), single border
- Uniform size — no "hero" card larger than the others
- Label: 0.7rem uppercase muted · Value: 1.8rem monospace accent · Subtext: 0.75rem muted

**Chat panel**
- Simple border card, no shadow theatrics
- Header: small title + green online dot
- Starter question pills: 4 buttons, shown only when chat is empty
- Text input + send button pinned to bottom of panel
- Message bubbles: minimal — just a lightly tinted background, no heavy borders

**Do not do**
- No multiple accent colours, gradients on gradients, or glow effects
- No CSS pseudo-element shimmer/shine overlays
- No animations other than the online-dot pulse
- No section dividers inside the chat panel body
- No more than 3 distinct font sizes per section
- No placeholder spinner overlays — use `st.spinner` if loading is needed

---

## Chat bot

The bot answers questions about the **current forecast context** (the store/item/date
currently selected in the sidebar). It must handle at minimum:

- Prediction value
- Comparison to 30-day average
- Lag feature values (lag_1, lag_7, lag_30)
- Best / worst month from history
- Day-of-week context

Mark the response function clearly as the LLM integration point:

```python
def process_chat_message(user_message: str, ctx: dict) -> str:
    """
    ▶▶ LLM HOOK ◀◀
    Replace this body with an LLM API call.
    `ctx` contains all forecast fields — pass it as the system prompt context.
    """
    # rule-based fallback until LLM is wired in
    ...
```

The `ctx` dict should include: `store_id`, `item_id`, `target_date`, `prediction`,
`avg_30`, `max_30`, `std_30`, `lag_1`, `lag_7`, `lag_30`, `roll_7`, `roll_30`,
`best_month`, `worst_month`, `dow_name`, `is_weekend`.

---

## Code structure

Split into named functions — no logic inside `main()` beyond orchestration:

```
load_model()          — @st.cache_resource
load_data()           — @st.cache_data
predict_next_day()    — feature engineering + model call (code above, verbatim)
compute_stats()       — 30-day avg, peak, std, total
build_history_chart() — Plotly line + rolling avg + forecast star
build_monthly_bar()   — Plotly bar, above-avg bars in accent colour
metric_card()         — renders one KPI card as HTML
build_chat_context()  — assembles the ctx dict for the chat
process_chat_message()— ▶▶ LLM HOOK ◀◀ rule-based until replaced
render_sidebar()      — store / item / date selectors
render_dashboard()    — header + KPI cards + charts + expander
render_chat_panel()   — header + messages + input
init_chat_state()     — st.session_state setup
main()                — load → sidebar → two columns → render
```

---

## Requirements

```
streamlit>=1.35.0
catboost>=1.2.0
pandas>=2.0.0
numpy>=1.26.0
plotly>=5.22.0
```
